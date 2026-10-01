# Verified Civil Comments baseline — private remote run

This is a **separate binary benchmark**, not a corrected score for the original four-class competition. The original project's outcome remains a verified leakage audit and synthetic pipeline reproduction.

Private [Kaggle notebook](https://www.kaggle.com/code/jbanmol9/civil-comments-private-baseline), **version 1**, completed successfully on 2026-10-01. The protocol was committed as `005bf4d` before execution. Source SHA-256 `20fd4edfc008ded7d80fe624151c925627bf5ab12cb8a1d5f7281d084b49f213` matches the local evaluator. Data and caches remained within Kaggle; only aggregate metrics, stage timestamps and notebook configuration metadata were retrieved. No dataset, row-level predictions, samples, fitted models or HTML output files were downloaded to the Mac. No competition submission, paid compute or public notebook was created.

## Task and results

The target is **toxicity annotator fraction >=0.5**. This is a thresholded annotation target, not an objective determination of harmfulness. The TF-IDF/class-balanced logistic regression baseline used 100,000 training rows, 20,000 validation rows and an untouched 20,000-row test subset, deterministically selected from the official splits after duplicate filtering. Training-only preprocessing and validation-only selection chose **C=1.0**, **prediction threshold=0.6**. The selected model was not refitted after validation. Test rows were fetched only after selection was complete; no changes followed the test result.

| Held-out test metric | Text baseline | Always predict non-toxic |
|---|---:|---:|
| Macro-F1 | **0.764925** | 0.479397 |
| Average precision | **0.593527** | 0.079150 |
| ROC AUC | **0.904736** | 0.500000 |
| Accuracy | 0.929100 | 0.920850 |

Accuracy masks minority-class failures: the constant classifier gets 92.1% accuracy while finding no positive comments. The text baseline's positive-class precision is **0.548444**, recall **0.590019**, and F1 **0.568472**. Negative-class F1 is **0.961377**. Test support is 18,417 below-cutoff comments and 1,583 above-cutoff comments. Selected validation macro-F1 is 0.765687; it is a tuning score, not the independent result.

Confusion matrix (rows=true labels, columns=predicted labels):

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| True 0 | 17,648 | 769 |
| True 1 | 649 | 934 |

The baseline therefore still misses 649 of 1,583 positives and flags 769 negatives. These error counts support further evaluation, not automated enforcement or production readiness. No test-based tuning was performed.

## Overlap audit and sampling

NFKC/case/whitespace normalization identifies exact text duplicates. Deduplication preserves first occurrences and never uses labels. Validation removes texts present anywhere in the full training source; test removes texts present anywhere in full training or validation. Sampling then keeps the lowest text SHA256 keys up to the predeclared caps.

| Partition | Source rows | Within-split duplicates | Earlier-partition overlapping unique texts | Retained sample | Positive sample rows |
|---|---:|---:|---:|---:|---:|
| Train | 1,804,874 | 28,366 | 0 | 100,000 | 8,050 |
| Validation | 97,320 | 565 | 1,320 | 20,000 | 1,589 |
| Test | 97,320 | 573 | 1,666 | 20,000 | 1,583 |

No empty texts were encountered. Full remote schemas contain text and seven annotation fields, with **no article, author or parent identifiers**. Parent context and every annotation except the target are excluded from features. This export permits an exact-text overlap audit, but cannot establish author/article independence or temporal generalization. Near duplicates and contextual relationships remain possible. The result applies to the deduplicated resource-limited subset, not the full published benchmark distribution.

## Reproducibility and checks

[Protocol and commands](PROTOCOL.md) · [Aggregate metrics](../evidence/civil_comments/metrics.json) · [Run stages and source checksum](../evidence/civil_comments/run.json) · [Uploaded privacy configuration](../evidence/civil_comments/uploaded_metadata.json).

Pinned dataset revision: `google/civil_comments@f2970eb3a55777454c94069077cc8d9b5866312d`. Every source parquet checksum and each sampled partition's combined key digest is recorded in the aggregate metrics. These identify the data/sample without exposing rows. First-party [TFDS](https://www.tensorflow.org/datasets/catalog/civil_comments) and [Google dataset-card licensing](https://huggingface.co/datasets/google/civil_comments/blob/f2970eb3a55777454c94069077cc8d9b5866312d/README.md#licensing-information) state CC0 for the dataset and underlying text.

Runtime: Python **3.12.13**, scikit-learn **1.6.1**, NumPy **2.0.2**, SciPy **1.16.3**, PyArrow **24.0.0**, huggingface-hub **1.11.0**. Evaluator elapsed time was **114.269 seconds**, including remote file fetching/checksums and training; it is not prediction latency, cost or a cross-machine speed benchmark. The local test environment uses different pinned versions and is separately documented. An independent runtime-image/privacy metadata lookup returned HTTP403, so the image identifier and server-side privacy flags could not be independently read. Uploaded metadata explicitly requested private CPU execution with no competition/data attachments; its request is preserved. No retry or access bypass was attempted. Reproduction should match the recorded packages; exact cross-platform numerical equality is not guaranteed.

All **14 local tests passed** in the isolated environment, including target boundary, label-independent sampling, normalized overlap removal, order-independent sampling for unique texts and validation-token exclusion. Script compilation and whitespace checks passed. The real remote evaluation completed on its first attempt with no convergence warning or recoverable failure. Logs contained an anonymous-HF rate-limit advisory and Kaggle's nbconvert SyntaxWarnings; neither prevented completion. No credentials were created, read or printed.

## Practical limits

This is one fixed sample and one simple model family with no multiple-seed study or uncertainty intervals. Crowd annotation disagreement, near duplicates, unknown group relationships, domain/time shift and threshold sensitivity remain. No identity-subgroup fairness, calibration, robustness, latency, annotation audit or production-readiness result is claimed. A paid evaluation pilot should first agree on the client's authorized dataset, decision costs and generalization question, then assess these gaps with an independent holdout rather than retuning against this test.
