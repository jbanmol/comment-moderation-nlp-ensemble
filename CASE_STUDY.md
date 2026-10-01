# From a leaked score to a reviewable evaluation

A public four-class comment-classification experiment reported ensemble macro-F1 of 0.80701. Source inspection showed that every cross-validation row's label contributed to its own post-level target encoding. Preprocessing also ran before cross-validation, and ensemble selection reused evaluation labels. That historical score cannot support a generalization claim.

The fix established a smaller, auditable text baseline: learned preprocessing within training folds, separate validation selection, grouped held-out evaluation, dataset checksums and reproducible artifacts. Historical outputs remain available with warnings and explicit archival execution. Regression tests verify label isolation, vocabulary isolation, group separation and reproducibility. No corrected real-data result is claimed for the original competition: its copying-rule scope remains unresolved, and its data was not downloaded locally. Official definitions do not assign semantic names to its numeric classes.

For independent real-data evidence, an authorized private CPU notebook evaluated Google's openly licensed **CC0 Civil Comments** export entirely in Kaggle. This is a **separate binary task**: toxicity annotator fraction >=0.5. It is not a reproduction of the original four-class experiment.

The precommitted protocol used official splits, normalized-text deduplication and deterministic resource caps of 100,000 training / 20,000 validation / 20,000 test rows. TF-IDF and class-balanced logistic regression saw training rows only. Validation chose regularization and prediction threshold; the model was frozen before fetching test rows. The overlap audit removed 1,320 validation and 1,666 test texts present in earlier partitions.

| Held-out metric | Text baseline | Always predict non-toxic |
|---|---:|---:|
| Macro-F1 | **0.76492** | 0.47940 |
| Average precision | **0.59353** | 0.07915 |
| ROC AUC | **0.90474** | 0.50000 |

Positive-class precision/recall were **0.54844/0.59002**. Among 20,000 test rows, the model missed 649 positives and flagged 769 negatives. These concrete errors reveal limitations that 92.9% overall accuracy would obscure. They support further evaluation, not automated enforcement or production readiness.

Fourteen local tests passed and the private remote run completed on its first attempt. Only aggregate evidence came back to the Mac; no comments, samples, predictions or models were downloaded. Source/data/sample checksums and package versions are recorded. An optional container metadata lookup was denied, so a complete runtime-image lock remains unavailable.

The sampled export lacks author/article identifiers. Near duplicates, group relationships, time shift, annotation disagreement, subgroup fairness and calibration remain unassessed. A focused paid evaluation pilot could apply this process to an authorized client dataset: define the decision metric, audit leakage, freeze an independent holdout, reproduce a baseline, and deliver per-class errors with practical limitations. No paid pilot, client outreach or production deployment has occurred.

[Verified results and aggregate evidence](civil_comments/RESULTS.md) · [Frozen protocol](civil_comments/PROTOCOL.md) · [Original leakage audit](EVALUATION.md)
