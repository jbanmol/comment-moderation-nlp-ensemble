# Civil Comments protocol — frozen before remote evaluation

This is a separate binary benchmark. It does not reproduce or replace the original competition's four-class result.

**License and source:** the [first-party TFDS catalog](https://www.tensorflow.org/datasets/catalog/civil_comments) and [Google's Hugging Face dataset card](https://huggingface.co/datasets/google/civil_comments/blob/f2970eb3a55777454c94069077cc8d9b5866312d/README.md) identify the dataset and underlying text as CC0. The HF revision is pinned to `f2970eb3a55777454c94069077cc8d9b5866312d`; only its four parquet files are fetched, entirely inside Kaggle. No competition source/terms are attached to this notebook. No dataset, samples, predictions or row-level logs are retrieved to the Mac.

**Target:** 1 when the toxicity annotator fraction is **>=0.5**, otherwise 0. This operational definition loses information about annotator disagreement and is not objective truth or a production moderation policy.

**Splits and resource caps:** preserve the official train, validation and test partitions. Within each partition, normalize NFKC, casefold and collapse whitespace for exact-duplicate detection; keep the first occurrence. Earlier official partitions take precedence: remove any validation text hash present anywhere in train, and any test text hash present anywhere in train or validation. Report all removal counts. Select the lowest normalized-text SHA256 keys up to **100,000 train / 20,000 validation / 20,000 test**. Sampling, deduplication and exclusion use text alone, never labels. This is a deterministic resource-limited subset, not a full leaderboard benchmark. Record selected-key digests and complete source-file checksums; do not output individual keys or rows.

**Features and model:** raw text only. Parent text, all other annotations and all metadata are excluded. The TFDS catalog warns that parent context can cross split boundaries. The chosen HF export does not provide article/author/parent IDs; inspect remote parquet schemas, report the available fields, and explicitly leave group independence unverified. Exact duplicate removal does not establish chronological, article or author independence.

Fit an independent TF-IDF + class-balanced logistic regression pipeline for each predeclared **C in [0.1, 1.0]** on training rows only. TF-IDF uses word unigrams/bigrams, min_df=2, at most 50,000 features and sublinear TF. Logistic regression uses liblinear, max_iter=1000, seed42; a convergence warning fails the run rather than silently accepting an unconverged model. Select C and a decision threshold from **[0.2,0.3,0.4,0.5,0.6,0.7,0.8]** by validation macro-F1; exact ties favor earlier/smaller C and threshold. No CV is performed: preprocessing is fitted only on the designated training partition, never on validation or test. Do not refit on validation, keeping threshold selection aligned to the fitted model.

Freeze the selected model/threshold **before fetching test rows**. Evaluate it on test once. Report macro-F1, average precision, ROC AUC, confusion matrix, per-class precision/recall/F1/support, and a predefined always-non-toxic baseline. No threshold optimization, class-weight selection, tuning or retraining follows the test result. Do not report validation selection scores as independent performance. No Kaggle competition or late submission occurs.

**Execution and outputs:** private Kaggle script notebook, standard free CPU, internet enabled solely to fetch public HF files, no accelerator, no paid service or external inference. Cache files live under `/tmp` within the remote runtime. `/kaggle/working/metrics.json` and controlled aggregate/stage logs are the only outputs. No row-level CSV or model artifact is written. Only this runner's logs may be retrieved with `kaggle kernels logs`; never use `kernels output`, `competitions download`, old notebook logs or generic file downloads. Exceptions emit only the exception class, never messages or tracebacks containing dataset text.

Record notebook URL/version, evaluator checksum, elapsed time, Python/package versions, source checksums, split counts and duplicate counts with the final results. Package versions are the Kaggle runtime's installed versions; record them rather than claim the local pins are used. Cross-version/platform bitwise equivalence is not promised.

**Limits:** one fixed text-hash sample and one model family; no multiple-seed stability, confidence intervals, subgroup fairness, annotation review, near-duplicate analysis, calibration, external-site generalization, cost/latency or deployment assessment. Group metadata is absent, so this is official-split text generalization—not unseen-author or unseen-article evaluation. Test labels are untouched until final evaluation; test text hashes are used only for the predeclared overlap filter. Claims about the original competition remain limited to its leakage audit and synthetic software checks.

Local verification uses synthetic rows only:

```sh
/tmp/moderation-verification-075ad95/bin/python -m unittest discover -s tests -v
python3 -m compileall -q civil_comments tests
```

Authorized remote execution (uploads code and metadata only):

```sh
kaggle kernels push -p civil_comments -t 7200
kaggle kernels status jbanmol9/civil-comments-private-baseline
kaggle kernels logs jbanmol9/civil-comments-private-baseline
```
