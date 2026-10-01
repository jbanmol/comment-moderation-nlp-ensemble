import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import evaluate, leakage_groups, make_model, split_indices, validate_data


def fixture():
    # Distinct random text; deliberately no realistic moderation signal.
    rng = np.random.default_rng(7)
    words = np.array(['apple', 'river', 'cloud', 'table', 'green', 'orange', 'paper', 'stone'])
    return pd.DataFrame([{'post_id': i // 4, 'comment': ' '.join(rng.choice(words, 12)) + f' id{i}',
                          'label': i % 4} for i in range(400)])


class EvaluationTests(unittest.TestCase):
    def test_no_post_or_duplicate_overlap(self):
        df = fixture()
        df.loc[4, 'comment'] = '  ' + df.loc[0, 'comment'].upper() + '  '
        groups = leakage_groups(df)
        self.assertEqual(groups[0], groups[4])
        splits = split_indices(groups)
        for i in range(3):
            for j in range(i):
                self.assertFalse(set(groups[splits[i]]) & set(groups[splits[j]]))
        self.assertEqual(sorted(np.concatenate(splits)), list(range(len(df))))

    def test_holdout_labels_cannot_change_training_or_splits(self):
        df = fixture()
        groups = leakage_groups(df)
        train, val, test = split_indices(groups)
        changed = df.copy()
        changed.loc[np.concatenate([val, test]), 'label'] = (changed.loc[np.concatenate([val, test]), 'label'] + 1) % 4
        for a, b in zip(split_indices(leakage_groups(changed)), (train, val, test)):
            np.testing.assert_array_equal(a, b)
        a, b = make_model(1), make_model(1)
        a.fit(df.comment.iloc[train], df.label.iloc[train])
        b.fit(changed.comment.iloc[train], changed.label.iloc[train])
        np.testing.assert_allclose(a.named_steps['classifier'].coef_, b.named_steps['classifier'].coef_)

    def test_holdout_token_not_in_vocabulary(self):
        model = make_model(1)
        model.fit(['cat dog', 'bird fish', 'cat bird', 'dog fish'], [0, 1, 2, 3])
        before = dict(model.named_steps['tfidf'].vocabulary_)
        model.predict(['secretvalidationtoken'])
        self.assertEqual(before, model.named_steps['tfidf'].vocabulary_)
        self.assertNotIn('secretvalidationtoken', before)

    def test_reproducible_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = fixture()
            a = evaluate(df, Path(tmp) / 'a', 'fixture', 'synthetic')
            b = evaluate(df, Path(tmp) / 'b', 'fixture', 'synthetic')
            self.assertEqual(a, b)
            self.assertTrue((Path(tmp) / 'a' / 'test_predictions.csv').exists())
            with self.assertRaises(ValueError):
                evaluate(df, Path(tmp) / 'a', 'fixture', 'synthetic')

    def test_test_labels_cannot_affect_selection_or_predictions(self):
        df = fixture()
        _, _, test = split_indices(leakage_groups(df))
        changed = df.copy()
        changed.loc[test, 'label'] = (changed.loc[test, 'label'] + 1) % 4
        with tempfile.TemporaryDirectory() as tmp:
            a = evaluate(df, Path(tmp) / 'a', 'fixture', 'synthetic')
            b = evaluate(changed, Path(tmp) / 'b', 'fixture', 'synthetic')
            self.assertEqual(a['selected_C'], b['selected_C'])
            self.assertEqual(a['candidates'], b['candidates'])
            pa = pd.read_csv(Path(tmp) / 'a' / 'test_predictions.csv')
            pb = pd.read_csv(Path(tmp) / 'b' / 'test_predictions.csv')
            np.testing.assert_array_equal(pa.prediction, pb.prediction)

    def test_invalid_data_fails(self):
        with self.assertRaises(ValueError):
            validate_data(pd.DataFrame({'comment': ['x']}))
        df = fixture()
        df.loc[0, 'comment'] = None
        with self.assertRaises(ValueError):
            validate_data(df)


if __name__ == '__main__':
    unittest.main()
