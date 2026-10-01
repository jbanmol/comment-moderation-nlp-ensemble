# Private Kaggle preparation — execution blocked by rule scope

The authenticated Kaggle CLI 2.2.2 successfully listed the competition's file metadata and read its official pages. No dataset files, samples, predictions or row-level outputs were fetched to the Mac.

The official [competition-specific rules](https://www.kaggle.com/competitions/comment-category-prediction-challenge/rules) prohibit direct copying from other sources, explicitly including LLM tools. Their linked [MLP project guideline](https://docs.google.com/document/d/e/2PACX-1vSjp57VSo57LcJSh8XcJyYZN0NtyY2BkTM1ptC025MIfhmvIP2Oh2xYSIq-wqKDnzXEwccGVQ6A_orP/pub) requires private, individual, CPU-only work and forbids plagiarism. No explicit exception for AI-assisted, post-course portfolio evaluation was found. The generic foundational rules do not supply a broad standalone data-reuse license. Existing access does not settle that scope. No new agreements were accepted.

**No original-competition notebook was uploaded or executed remotely.** The discarded preparation was never run; its rule scope remains unresolved. The separately authorized CC0 Civil Comments run is documented under `civil_comments/`. This is a scope blocker, not a technical claim that private notebooks cannot run. Do not submit AI-assisted code as original coursework.

## Historical preparation, excluded from publication

The original competition's executable remote-preparation files have been removed from the proposed publication to avoid exposing an unused run path while copying-rule scope is unresolved. That preparation remains recoverable in local Git history. No original-competition notebook was uploaded or run. The generic supported evaluator remains `evaluate.py`; the only active real-data benchmark is the separately authorized CC0 Civil Comments runner under `civil_comments/`.

The CLI's code-submission capability is not evidence that this competition allows late submissions or this AI-assisted workflow. No submission was attempted. No metadata endpoint restriction was bypassed.

## Checks

Nine local regression tests pass, including aggregate-only output: exactly `metrics.json`, with no CSV or model artifact. The aggregate-only synthetic CLI run produced macro-F1 0.2592397275324104, C=10, and rows 240/80/80. Script compilation passed. The discarded preparation was never validated against competition data or Kaggle execution.

## Official dataset definition

The authenticated [data description](https://www.kaggle.com/competitions/comment-category-prediction-challenge/data) identifies `post_id` as the parent discussion identifier and describes the label only as four internal handling categories. No mapping of numeric classes to “normal”, “hate”, “hostile” or “political” meanings was established from that inspected page. Keep class IDs neutral. Actual file metadata lists `train.csv` (73,583,455 bytes), `test.csv` (37,680,076 bytes), and `Sample.csv` (806,904 bytes); the description generically calls the latter `sample_submission.csv`. These are metadata observations, not fetched data.

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
