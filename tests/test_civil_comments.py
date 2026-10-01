import unittest
import numpy as np
from civil_comments.benchmark import binary_label, sample_rows, select_model, text_key


class CivilCommentsTests(unittest.TestCase):
    def rows(self):
        return [{'text': f'{"kind helpful" if i % 2 else "angry rude"} unique{i}',
                 'toxicity': float(i % 2)} for i in range(100)]

    def test_target_boundary_and_invalid(self):
        self.assertEqual(binary_label(.4999), 0)
        self.assertEqual(binary_label(.5), 1)
        for invalid in (float('nan'), -1, 2):
            with self.assertRaises(ValueError):
                binary_label(invalid)

    def test_normalization_and_cross_split_exclusion(self):
        rows = self.rows()
        rows += [{'text': '  '+rows[0]['text'].upper()+' ', 'toxicity': .9}]
        texts, labels, keys, counts = sample_rows(rows, 100, {text_key(rows[1]['text'])})
        self.assertEqual(counts['within_split_duplicates'], 1)
        self.assertEqual(counts['cross_split_overlap'], 1)
        self.assertNotIn(text_key(rows[1]['text']), {text_key(t) for t in texts})
        self.assertEqual(len(texts), 99)

    def test_sampling_does_not_use_labels(self):
        rows = self.rows()
        a = sample_rows(rows, 30, set())
        flipped = [dict(r, toxicity=1-r['toxicity']) for r in rows]
        b = sample_rows(flipped, 30, set())
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[3]['retained_key_sha256'], b[3]['retained_key_sha256'])
        np.testing.assert_array_equal(a[1], 1-b[1])

    def test_validation_token_excluded_from_fit(self):
        rows = self.rows()
        tx, ty, _, _ = sample_rows(rows, 100, set())
        model, choice, candidates = select_model(tx, ty, ['validationonlytoken kind', 'validationonlytoken angry'], [1, 0])
        self.assertNotIn('validationonlytoken', model.named_steps['tfidf'].vocabulary_)
        self.assertEqual(len(candidates), 14)
        self.assertIn(choice['C'], (.1, 1.0))

    def test_sampling_order_independent_for_unique_text(self):
        a = sample_rows(self.rows(), 30, set())
        b = sample_rows(list(reversed(self.rows())), 30, set())
        self.assertEqual(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


if __name__ == '__main__':
    unittest.main()
