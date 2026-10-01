"""Public CC0 benchmark; main executes only in Kaggle. Emits aggregates only."""
import hashlib
import heapq
import json
import math
import platform
import re
import time
import unicodedata
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline

DATASET = 'google/civil_comments'
REVISION = 'f2970eb3a55777454c94069077cc8d9b5866312d'
FILES = {
    'train': ['data/train-00000-of-00002.parquet', 'data/train-00001-of-00002.parquet'],
    'validation': ['data/validation-00000-of-00001.parquet'],
    'test': ['data/test-00000-of-00001.parquet'],
}
CAPS = {'train': 100000, 'validation': 20000, 'test': 20000}
CS = (0.1, 1.0)
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SEED = 42


def text_key(text):
    normalized = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()
    return hashlib.sha256(normalized.encode('utf-8')).digest()


def binary_label(value):
    if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError('Invalid toxicity fraction')
    return int(float(value) >= 0.5)


def sample_rows(rows, cap, excluded):
    """Lowest SHA256 keys, label-independent; first occurrence wins duplicates.

    Returns selected text/labels, all nonempty keys (for later exclusion), and counts.
    This function never prints or writes row data.
    """
    seen = set()
    heap = []
    counts = {'source_rows': 0, 'empty': 0, 'within_split_duplicates': 0,
              'cross_split_overlap': 0}
    for row in rows:
        counts['source_rows'] += 1
        text = row['text']
        if not isinstance(text, str) or not text.strip():
            counts['empty'] += 1
            continue
        key = text_key(text)
        if key in seen:
            counts['within_split_duplicates'] += 1
            continue
        seen.add(key)
        if key in excluded:
            counts['cross_split_overlap'] += 1
            continue
        label = binary_label(row['toxicity'])
        priority = int.from_bytes(key, 'big')
        item = (-priority, text, label, key)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)
    retained = sorted(heap, key=lambda r: r[3])
    texts = [r[1] for r in retained]
    labels = np.array([r[2] for r in retained], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Selected partition lacks a class')
    counts['retained_rows'] = len(texts)
    counts['positive_rows'] = int(labels.sum())
    counts['retained_key_sha256'] = hashlib.sha256(b''.join(r[3] for r in retained)).hexdigest()
    return texts, labels, seen, counts


def make_model(c):
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                sublinear_tf=True)),
        ('classifier', LogisticRegression(C=c, solver='liblinear', class_weight='balanced',
                                         max_iter=1000, random_state=SEED)),
    ])


def select_model(train_text, train_y, validation_text, validation_y):
    """Test is not an argument. Preprocessing sees training text only."""
    candidates = []
    models = []
    for c in CS:
        model = make_model(c)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(train_text, train_y)
        probability = model.predict_proba(validation_text)[:, 1]
        for threshold in THRESHOLDS:
            score = f1_score(validation_y, probability >= threshold, average='macro',
                             labels=[0, 1], zero_division=0)
            candidates.append({'C': c, 'threshold': threshold, 'validation_macro_f1': float(score)})
        models.append(model)
    best = max(candidates, key=lambda x: x['validation_macro_f1'])
    return models[CS.index(best['C'])], best, candidates


def metrics(y, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    return {
        'macro_f1': float(f1_score(y, predicted, average='macro', labels=[0, 1], zero_division=0)),
        'average_precision': float(average_precision_score(y, probability)),
        'roc_auc': float(roc_auc_score(y, probability)),
        'classification_report': classification_report(y, predicted, labels=[0, 1],
                                                      output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def iter_remote_rows(split, manifest):
    # These imports and downloads occur only in the Kaggle runtime.
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    for filename in FILES[split]:
        path = hf_hub_download(DATASET, filename, repo_type='dataset', revision=REVISION,
                               token=False, cache_dir='/tmp/civil-comments-cache')
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        parquet = pq.ParquetFile(path)
        manifest.append({'file': filename, 'sha256': digest.hexdigest(),
                         'source_rows': parquet.metadata.num_rows,
                         'available_columns': parquet.schema.names})
        for batch in parquet.iter_batches(batch_size=8192, columns=['text', 'toxicity']):
            values = batch.to_pydict()
            for text, toxicity in zip(values['text'], values['toxicity']):
                yield {'text': text, 'toxicity': toxicity}


def run_remote():
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Remote-only runner: no dataset access outside Kaggle')
    started = time.monotonic()
    manifest = []
    print('STAGE=train_loading', flush=True)
    tx, ty, train_keys, train_counts = sample_rows(iter_remote_rows('train', manifest), CAPS['train'], set())
    print('STAGE=validation_loading', flush=True)
    vx, vy, validation_keys, validation_counts = sample_rows(
        iter_remote_rows('validation', manifest), CAPS['validation'], train_keys)
    print('STAGE=training_and_validation_selection', flush=True)
    model, choice, candidates = select_model(tx, ty, vx, vy)
    # Freeze model/threshold before any test row or test label is accessed.
    print('STAGE=test_loading_after_selection', flush=True)
    test_x, test_y, _, test_counts = sample_rows(iter_remote_rows('test', manifest), CAPS['test'],
                                               train_keys | validation_keys)
    print('STAGE=final_test', flush=True)
    probability = model.predict_proba(test_x)[:, 1]
    result = {
        'dataset': DATASET, 'revision': REVISION, 'license': 'CC0-1.0',
        'task': 'Binary toxicity: annotator fraction >= 0.5 => 1; below => 0',
        'protocol': 'Official splits; NFKC/case/whitespace duplicate removal, earlier split wins; '
                    'lowest normalized text SHA256 keys per cap; training-only fits; '
                    'validation-only C/threshold selection; no refit or test tuning',
        'sampling_caps': CAPS, 'seed': SEED,
        'partitions': {'train': train_counts, 'validation': validation_counts, 'test': test_counts},
        'features': ['text'], 'excluded_features': ['parent_text', 'all other annotations and metadata'],
        'group_audit': 'This HF export has no author/article/parent identifiers; group independence cannot be verified',
        'selection': choice, 'candidates': candidates,
        'test': metrics(test_y, probability, choice['threshold']),
        'always_non_toxic_baseline': metrics(test_y, np.zeros(len(test_y)), 0.5),
        'files': manifest,
        'versions': {p: version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'huggingface-hub', 'pyarrow']},
        'python': platform.python_version(), 'elapsed_seconds': time.monotonic() - started,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['Resource-limited text-hash sample, not full benchmark',
                        'No article/author/temporal separation established',
                        'Near duplicates and contextual relationships can remain',
                        'Thresholded annotator agreement is not objective ground truth',
                        'No subgroup fairness, calibration or production-readiness claim'],
    }
    Path('/kaggle/working/metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print('AGGREGATE_METRICS_JSON=' + json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == '__main__':
    try:
        run_remote()
    except Exception as error:
        # Never print exception messages/tracebacks that might contain dataset text.
        print('BENCHMARK_ERROR_TYPE=' + type(error).__name__, flush=True)
        raise SystemExit(1) from None
