# Private Kaggle preparation — execution blocked by rule scope

The authenticated Kaggle CLI 2.2.2 successfully listed the competition's file metadata and read its official pages. No dataset files, samples, predictions or row-level outputs were fetched to the Mac.

The official [competition-specific rules](https://www.kaggle.com/competitions/comment-category-prediction-challenge/rules) prohibit direct copying from other sources, explicitly including LLM tools. Their linked [MLP project guideline](https://docs.google.com/document/d/e/2PACX-1vSjp57VSo57LcJSh8XcJyYZN0NtyY2BkTM1ptC025MIfhmvIP2Oh2xYSIq-wqKDnzXEwccGVQ6A_orP/pub) requires private, individual, CPU-only work and forbids plagiarism. No explicit exception for AI-assisted, post-course portfolio evaluation was found. The generic foundational rules do not supply a broad standalone data-reuse license. Existing access does not settle that scope. No new agreements were accepted.

**Nothing has been uploaded or executed remotely.** Sponsor clarification permitting this private, non-course audit is needed before using the prepared artifact with competition data. This is a scope blocker, not a technical claim that private notebooks cannot run. Do not submit AI-assisted code as original coursework.

## Prepared code-only artifact

`build_notebook.py` embeds the current public-project evaluator into `private-baseline.py` and writes `kernel-metadata.json`. The builder reads only `evaluate.py`; it never reads personal/employer files, credentials or data. It performs no network calls or uploads. Rebuild after changing the evaluator:

```sh
python3 remote/build_notebook.py
```

Metadata explicitly sets private visibility, no accelerator, no internet, and only the original competition as a remote data source. The standalone script reads the training CSV **inside Kaggle**, uses the verified group/validation/test protocol, and produces only aggregate metrics. It emits no row-level splits, comments, predictions or model files. The notebook does not load the unlabeled competition test CSV or prepare a leaderboard submission. The evaluator source checksum and environment versions are recorded for auditability. The current container package versions would be reported rather than silently assumed identical to the local environment.

After permission is resolved, the documented CLI capability is `kaggle kernels push -p remote -t 7200` (CPU, private metadata). This command has **not** been run. The installed implementation sends only the specified code body and metadata; it does not upload the surrounding folder as dataset files. `kaggle kernels status OWNER/SLUG` reads status; `kaggle kernels logs OWNER/SLUG` reads logs. Only this controlled aggregate-only runner's logs should be retrieved. Do not fetch historical notebooks' logs, which may contain rows. Do not run `competitions download`, `kernels output`, or generic output downloads.

The CLI also supports code submissions with `competitions submit ... -k OWNER/SLUG -v VERSION -f REMOTE_FILENAME`. That is a server-side capability for **code competitions**, not proof this competition permits late/code submissions. No submission was attempted. Any eventual leaderboard score must stay separate from the untouched held-out training-data test, and must not drive repeated tuning against that holdout. If this competition instead requires local prediction upload, that path conflicts with the user's constraint and must remain blocked.

## Checks

Nine local regression tests pass, including aggregate-only output: exactly `metrics.json`, with no CSV or model artifact. The aggregate-only synthetic CLI run produced macro-F1 0.2592397275324104, C=10, and rows 240/80/80. Script compilation passed. This prepared remote script has not been validated against competition data or Kaggle execution.

## Official dataset definition

The authenticated [data description](https://www.kaggle.com/competitions/comment-category-prediction-challenge/data) identifies `post_id` as the parent discussion identifier and describes the label only as four internal handling categories. It supplies **no mapping of numeric classes to “normal”, “hate”, “hostile” or “political” meanings**. Keep class IDs neutral. Actual file metadata lists `train.csv` (73,583,455 bytes), `test.csv` (37,680,076 bytes), and `Sample.csv` (806,904 bytes); the description generically calls the latter `sample_submission.csv`. These are metadata observations, not fetched data.

Read-only commands used:

```sh
kaggle --version
kaggle competitions files -c comment-category-prediction-challenge
kaggle competitions pages comment-category-prediction-challenge --content
kaggle kernels list --mine
kaggle kernels push --help
kaggle kernels logs --help
kaggle competitions submit --help
```

The `pages` command returned official text under existing authentication. The similarly named competition search returned no results, so search absence must not be equated with inaccessible data. The first sandboxed invocation failed DNS resolution; retry with approved network access succeeded. The installed CLI emits a Requests dependency warning, but the metadata/page reads succeeded. No credential contents were read or printed.
