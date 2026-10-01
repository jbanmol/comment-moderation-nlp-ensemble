# Synthetic software fixture — no real comments or competition data

Every row and prediction in this directory is generated test data, not Civil Comments or the original competition dataset. `tests/test_evaluate.py:fixture` creates 400 random-word strings with seed7 and assigns four labels by row position modulo4. `input.csv` contains that fixture; `metrics.json`, `splits.csv` and `test_predictions.csv` demonstrate reproducible software execution only. They support no moderation-accuracy claim.

These synthetic rows are retained to allow reproduction without downloading any real dataset. Civil Comments evidence is separately limited to aggregates under `evidence/civil_comments/`; no real text, individual row identifiers, predictions or model files are present there.
