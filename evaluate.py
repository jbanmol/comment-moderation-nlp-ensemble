"""Leakage-safe, text-only benchmark. See EVALUATION.md before running."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

SEED = 42
CANDIDATES = (0.1, 1.0, 10.0)


def validate_data(df):
    required = {'comment', 'post_id', 'label'}
    if not required.issubset(df.columns):
        raise ValueError(f'Required columns: {sorted(required)}')
    if df[list(required)].isna().any().any():
        raise ValueError('Missing comment, post_id or label; resolve explicitly before evaluation')
    if set(df.label.unique()) != {0, 1, 2, 3}:
        raise ValueError('Expected integer labels 0, 1, 2, 3')
    if df.comment.astype(str).str.strip().eq('').any():
        raise ValueError('Empty comments are not supported')


def leakage_groups(df):
    """Connected components joining same posts or normalized duplicate comments."""
    parent = list(range(len(df)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for values in (df.post_id.astype(str),
                   df.comment.astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()):
        seen = {}
        for i, value in enumerate(values):
            if value in seen:
                parent[root(i)] = root(seen[value])
            else:
                seen[value] = i
    return np.array([root(i) for i in range(len(df))])


def split_indices(groups):
    """Split without consulting labels: 60% train / 20% validation / 20% test groups."""
    indices = np.arange(len(groups))
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                            random_state=SEED).split(indices, groups=groups))
    train_local, validation_local = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                            random_state=SEED + 1).split(development, groups=groups[development]))
    return development[train_local], development[validation_local], test


def make_model(c):
    # No ID, target encoding, metadata, pretrained downloads or test-time fitting.
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000,
                                sublinear_tf=True)),
        ('classifier', LogisticRegression(C=c, class_weight='balanced',
                                         max_iter=2000, random_state=SEED)),
    ])


def evaluate(df, output, data_sha256, provenance):
    validate_data(df)
    output = Path(output)
    if output.exists():
        raise ValueError('Output path already exists; preserve previous evaluation artifacts')
    groups = leakage_groups(df)
    train, validation, test = split_indices(groups)
    for name, indices in [('train', train), ('validation', validation), ('test', test)]:
        if set(df.iloc[indices].label) != {0, 1, 2, 3}:
            raise ValueError(f'{name} lacks a class; obtain more data, do not search seeds using test labels')
    x = df.comment.astype(str)
    y = df.label
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                 .split(x.iloc[train], y.iloc[train], groups[train]))
    for a, b in folds:
        if set(y.iloc[train[a]]) != {0, 1, 2, 3} or set(y.iloc[train[b]]) != {0, 1, 2, 3}:
            raise ValueError('A CV fold lacks a class; more independent groups are needed')
    candidates = []
    models = []
    for c in CANDIDATES:
        model = make_model(c)
        cv = cross_val_score(model, x.iloc[train], y.iloc[train], cv=folds,
                             scoring='f1_macro', error_score='raise')
        model.fit(x.iloc[train], y.iloc[train])
        score = f1_score(y.iloc[validation], model.predict(x.iloc[validation]),
                         labels=[0, 1, 2, 3], average='macro', zero_division=0)
        candidates.append({'C': c, 'train_cv_macro_f1': cv.tolist(),
                           'validation_macro_f1': float(score)})
        models.append(model)
    # Stable tie-break uses the first (smallest C). Test has no role in selection.
    best = max(range(len(candidates)), key=lambda i: candidates[i]['validation_macro_f1'])
    model = models[best]
    development = np.concatenate([train, validation])
    model.fit(x.iloc[development], y.iloc[development])
    prediction = model.predict(x.iloc[test])
    result = {
        'data_sha256': data_sha256, 'provenance': provenance, 'seed': SEED,
        'split_strategy': 'connected post/normalized-duplicate groups; 60/20/20 by group',
        'rows': {name: len(idx) for name, idx in [('train', train), ('validation', validation), ('test', test)]},
        'candidates': candidates, 'selected_C': candidates[best]['C'],
        'test_macro_f1': float(f1_score(y.iloc[test], prediction, labels=[0, 1, 2, 3], average='macro')),
        'test_report': classification_report(y.iloc[test], prediction, labels=[0, 1, 2, 3], output_dict=True, zero_division=0),
        'test_confusion_matrix': confusion_matrix(y.iloc[test], prediction, labels=[0, 1, 2, 3]).tolist(),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__, 'joblib': joblib.__version__},
    }
    output.mkdir(parents=True)
    assignment = np.full(len(df), '', dtype=object)
    for name, idx in [('train', train), ('validation', validation), ('test', test)]:
        assignment[idx] = name
    pd.DataFrame({'row_position': np.arange(len(df)), 'group': groups, 'split': assignment}).to_csv(output / 'splits.csv', index=False)
    pd.DataFrame({'row_position': test, 'label': y.iloc[test].to_numpy(), 'prediction': prediction}).to_csv(output / 'test_predictions.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    joblib.dump(model, output / 'model.joblib')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Authorized labeled train.csv, NOT competition test.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provenance', required=True, help='Source and permission basis; use synthetic for fixtures')
    args = parser.parse_args()
    raw = args.data.read_bytes()
    result = evaluate(pd.read_csv(args.data), args.output, hashlib.sha256(raw).hexdigest(), args.provenance)
    print(json.dumps({'test_macro_f1': result['test_macro_f1'], 'selected_C': result['selected_C'], 'rows': result['rows']}, indent=2))
