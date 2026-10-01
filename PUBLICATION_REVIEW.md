# Publication review — prepared, awaiting approval

Proposed action: push the local `fix/leakage-safe-evaluation` branch to `https://github.com/jbanmol/comment-moderation-nlp-ensemble.git` and create a **draft pull request into `main`** in that same repository. `refs/remotes/origin/HEAD` resolves to `origin/main` in this checkout. No push or PR has occurred. No merge, deployment, public Kaggle notebook, competition submission or data/model upload is part of this proposal.

Proposed PR title: **Audit evaluation leakage and add a reproducible remote Civil Comments baseline**.

Proposed description:

> V4/V5 computed post-level target encoding from all labels before CV; preprocessing and ensemble selection also reused validation information. This change archives that evaluation honestly, adds a leakage-safe grouped text baseline and tests, and removes execution outputs/widgets from current historical notebooks while preserving source and Git history.
>
> A separate predeclared, openly licensed Civil Comments benchmark ran on Kaggle with private CPU settings requested. On a deduplicated 20,000-row binary test subset it achieved macro-F1 0.76492, average precision 0.59353 and ROC AUC 0.90474; the always-non-toxic baseline's macro-F1 was 0.47940. Original-competition performance is not claimed. Only aggregate evidence was retrieved locally. Final Kaggle visibility and container image remain independently unverified after a metadata 403 and stopped browser sessions.
>
> Validation: 14 passing isolated-environment tests, Python compilation, source/sample checksum checks, independent aggregate-metric consistency checks and Git whitespace checks. Limits: resource-capped hash sample, unavailable group identifiers, no near-duplicate/fairness/calibration/confidence-interval assessment and incomplete container lock. No test tuning followed the remote result.

## Included reviewable scope

- Original audit and supported evaluator: `evaluate.py`, pinned evaluation requirements, tests, documentation, `.gitignore` and warned archival V4/V5 scripts.
- Current V4 notebook: execution outputs/counts and widget state removed; source cells unchanged. Current V5 notebook: historical warning, no outputs. No t12026 notebook change. The original Git history remains intact.
- Civil Comments runner, fixed protocol, requested private/CPU metadata, observed core package versions and exact result/limitation report.
- Civil Comments evidence: three aggregate JSON files only (`metrics.json`, `run.json`, `uploaded_metadata.json`); no text, individual hashes/IDs, row predictions, downloaded datasets or models.
- Synthetic fixture/evidence under `evidence/synthetic/`, conspicuously labeled as generated data with no real moderation-performance meaning; all input rows and checksum verified against the seed7 generator.
- Original-competition access/rule findings in `remote/README.md`. Its unused executable preparation is excluded from the current diff and retained only in local Git history.
- Concise client-facing case study and this publication review.

## Independent local review

The original baseline's pipeline is cloned inside training CV; outer validation chooses C; final refit sees development data only; held-out test predictions follow selection. Its grouping and class checks are documented, and tests perturb holdout labels and inspect vocabularies.

The separate Civil Comments runner has no test argument in selection. Both TF-IDF/class weights fit on training rows only. Validation chooses among 2 C values and 7 prediction thresholds. The binary **annotation cutoff 0.5** is distinct from the **selected prediction threshold 0.6**. There is no post-validation refit or test tuning. The source matches the pre-run protocol commit and remote source checksum.

Official partitions are preserved after the declared text-only exclusions. Deduplication/sampling do not use labels; full earlier-partition text hashes exclude overlaps in later partitions. Duplicate variants retain the first source occurrence, so annotation disagreement between duplicates is not resolved. The lower-SHA256 cap is a deterministic resource-limited sample, not a full benchmark, stratified sample or guarantee of independent authors/articles. Group metadata is absent. Near-duplicate/context/time risks and no confidence intervals are stated explicitly.

I independently recomputed macro-F1 from the confusion matrix, checked support counts and validation selection/tie behavior, and verified stage order and frozen source. No residual label/preprocessing/selection leakage was identified within this stated protocol. This conclusion does not remove the documented grouping and data-quality limitations.

Final local commands:

```sh
/tmp/moderation-verification-075ad95/bin/python -m unittest discover -s tests -v
/tmp/moderation-verification-075ad95/bin/python -m compileall -q evaluate.py tests civil_comments v4_solution.py v5_solution.py
git diff --check
```

All 14 tests passed (0.938 seconds). Compilation and whitespace checks passed. Programmatic checks confirmed: source unchanged from 005bf4d, aggregate macro-F1/support consistency, selected candidate consistency, all current notebook outputs/widgets absent, notebook source unchanged, and synthetic generator/checksum equality.

## Remaining publication considerations

Original competition copying/private-coursework rules have unresolved post-course scope; no new submission, original-data evaluation, raw dataset redistribution or employer code/data is proposed. This PR documents the existing public project's leakage and a separate CC0 evaluation. Original historical outputs remain in existing Git history; this is current-snapshot sanitization, not a history purge or a claim that the repository's full history never contained data-derived content.

First-party sources identify Civil Comments and its underlying text as CC0. No raw text or dataset files are bundled. Code/data provenance is documented; no third-party model weights or paid APIs are used. This review does not assert blanket permission for unrelated original-competition reuse.

Kaggle was submitted with private CPU settings. Execution and aggregate metrics were verified; final visibility and container-image metadata were not independently confirmed. The denied endpoint was not retried or bypassed; no browser action is proposed. These uncertainty statements must remain in the public description.

No calibrated-probability, subgroup fairness, production-readiness, paid-client, or corrected-original-score claim is supported. No model parameters, metrics, dataset access or protocol decisions were changed after reviewing test results.
