# Model and ranking evaluation

This report describes the checked-in [`models/metrics.json`](../models/metrics.json), [`models/feature_schema.json`](../models/feature_schema.json), [`models/model.joblib`](../models/model.joblib), and the training/evaluation code in `scripts/`. Values below are artifact values, not a fresh retraining result. Current artifact metadata identifies model version `1.0.0`, random seed `42`, and metric timestamp `2026-10-10T07:39:00.287514+00:00`.

## Dataset and labels

The source is [ApacheJIT](https://github.com/hosseinkshvrz/apachejit), specifically `apachejit_total.csv`, whose location is defined by `APACHEJIT_URL` in `scripts/prepare_dataset.py`. The upstream README describes `buggy` as whether a commit introduced a bug. Its data construction describes linking fixed issue reports to fixing commits and tracing candidate bug-inducing changes. ImpactMap's preparation script validates the expected columns, deduplicates by `commit_id`, drops rows with missing cells, removes rows with negative size metrics, then splits by `author_date`.

The preparation code expects these source columns (additional columns are not explicitly rejected): `commit_id`, `project`, `buggy`, `fix`, `year`, `author_date`, `la`, `ld`, `nf`, `nd`, `ns`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, and `asexp`. `buggy` is converted to an integer target; the upstream description defines it as whether the commit introduced a bug. The model uses only the ten columns marked `use` below.

| Saved split | Rows |
|---|---:|
| Train | 74,671 |
| Validation | 16,001 |
| Test | 16,002 |

The test positive-label base rate saved in the metrics artifact is **0.1366**. Labels are commit-level, not file-level. Issue and fix linkage can miss bugs that were never reported, never linked to the relevant commit, or were discovered outside the observation period. The project does not have a measured estimate of that censoring or label noise.

The upstream repository README/file listing inspected for this documentation does not declare an ApacheJIT dataset license. The Apache-2.0 license in this repository is for ImpactMap code and is not evidence that ApacheJIT data can be redistributed. Review upstream terms before obtaining, using, or redistributing dataset files.

## Feature compatibility and leakage review

The model uses the ten columns marked `use` in the saved compatibility table. Definitions and decisions below follow `models/feature_schema.json`; inference order and dtypes are also enforced by `feature_extractor.py` and `predictor.py`.

| Field | Model dtype | Dataset availability | Reconstructible at inference | Leakage risk in schema | Decision | Definition / reason |
|---|---|---|---|---|---|---|
| `la` | int64 | yes | yes | none | use | Sum of lines added from diff numstat. |
| `ld` | int64 | yes | yes | none | use | Sum of lines deleted from diff numstat. |
| `nf` | int64 | yes | yes | none | use | Number of files changed in the commit diff. |
| `nd` | int64 | yes | yes | none | use | Distinct parent directories of changed files. |
| `ns` | int64 | yes | yes | none | use | Distinct top-level directories of changed files. |
| `ent` | float64 | yes | yes | none | use | Base-2 entropy of changed-file churn distribution. |
| `ndev` | float64 | yes | yes | none | use | Mean distinct prior authors of the changed files. |
| `age` | float64 | yes | yes | none | use | Mean days between the commit and previous file-touching commit. |
| `nuc` | float64 | yes | yes | none | use | Mean count of prior commits touching each changed file. |
| `aexp` | int64 | yes | yes | none | use | Prior commits by the current author in the repository. |
| `arexp` | — | yes | no | none | drop | Recent author experience decay constant is not reproducible across tools. |
| `asexp` | — | yes | no | none | drop | Subsystem-experience weighting is not reproducible across tools. |
| `fix` | — | yes | no | high | drop | Issue/fix linkage is not generally available at commit time. |
| `year` | — | yes | partial | high | drop | Excluded to avoid temporal shortcut and non-stationary artifacts. |
| `author_date` | — | yes | partial | high | split only | Used for chronological ordering, not as a model feature. |
| `commit_id` | — | yes | yes | high | drop | Arbitrary unique identifier. |
| `project` | — | yes | no | high | drop | Project identity could prevent cross-project generalization. |

The feature table is a compatibility review, not a formal proof that the retained history-derived features are free from every form of temporal leakage. Training data availability and inference reconstruction can differ in details, especially for shallow repository history.

## Split design and limitations

`scripts/prepare_dataset.py` sorts rows by `author_date` and assigns the first 70% to training, the next 15% to validation, and the remainder to test. The saved metrics describe this as a time-aware split. It preserves an overall chronological ordering, but it is **not project-grouped**: commits from one project may appear in all splits. No project-held-out or external-repository evaluation is recorded. The exact timestamp boundaries are in generated `data/processed/prep_report.json` when preparation has been run; they are not included in the checked-in metrics artifact.

## Training, calibration, and model selection

The primary estimator is `HistGradientBoostingClassifier` with `class_weight="balanced"`, `random_state=42`, `max_iter=100`, `min_samples_leaf=30`, and `l2_regularization=1.0`. It is fitted on the training split. A sigmoid `CalibratedClassifierCV` is fitted to a frozen estimator on the validation split. The decision threshold is selected by maximizing validation F1 over thresholds from `0.10` up to (but not including) `0.90` in increments of `0.01`; the saved artifact uses `0.30`.

Calibration fitting and threshold selection both reuse the validation split; there is no separate calibration split. The held-out test split is used for the saved model comparisons and reported error analysis.

The compared baselines are:

- **Dummy prior:** `DummyClassifier(strategy="prior")`, threshold `0.50`.
- **Churn heuristic:** positive when `la + ld > 50`; its probability-like score is churn divided by 200 and clipped to `[0, 1]`; reported threshold `0.50`.
- **Logistic regression:** `log1p`, standard scaling, balanced logistic regression, threshold `0.50`.

The calibrated HGB has the best saved PR-AUC and F1 among the listed rows. Its PR-AUC is `0.3861`, compared with `0.3824` for logistic regression. This is a small measured difference on this split; the artifact contains no confidence intervals or significance test. Logistic regression has higher recall (`0.8124` versus `0.7260`) at the respective reported thresholds.

## Held-out test metrics

All values below are copied from `models/metrics.json`. The first three rows use the listed decision threshold; ranking metrics use predicted scores.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated HistGradientBoosting | 0.3861 | 0.8060 | 0.2999 | 0.7260 | 0.4244 | 0.1163 | 0.30 |
| Logistic regression | 0.3824 | 0.7995 | 0.2547 | 0.8124 | 0.3878 | 0.2193 | 0.50 |
| Churn heuristic | 0.2752 | 0.7557 | 0.2135 | 0.8760 | 0.3434 | 0.3103 | 0.50 |
| Dummy prior | 0.1366 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.1424 | 0.50 |

At threshold `0.30`, the calibrated HGB confusion matrix is **TN 10,111; FP 3,705; FN 599; TP 1,587**. The test set contains 16,002 rows; the artifact's positive base rate is 0.1366.

### Calibration findings

The training script applies sigmoid calibration using the validation split. The saved test Brier score is `0.1163`, lower than the saved logistic (`0.2193`), churn (`0.3103`), and dummy-prior (`0.1424`) Brier scores. This supports a comparison of these saved probability errors on this test split only. There is no saved reliability diagram, expected calibration error, confidence interval, or uncalibrated-HGB result, so the artifacts do not establish that calibration itself improved reliability or that probabilities are calibrated for new repositories. The application also warns that calibration is not independently verified by metrics metadata.

## Error analysis

The artifact reports errors by churn bucket and project for the calibrated HGB at the saved threshold. In the training code, the fields named `fp_rate` and `fn_rate` divide FP or FN by the **total rows in that bucket**, not by the conventional negative or positive class denominator. They are reproduced below as saved and should not be read as conventional class-conditional rates.

| Churn bucket | Rows | False positives | False negatives | Saved FP/row fraction | Saved FN/row fraction |
|---|---:|---:|---:|---:|---:|
| Small (≤20) | 4,752 | 10 | 121 | 0.0021 | 0.0255 |
| Medium (21–200) | 6,528 | 1,274 | 374 | 0.1952 | 0.0573 |
| Large (>200) | 4,722 | 2,421 | 104 | 0.5127 | 0.0220 |

The large-churn bucket accounts for 2,421 false positives and the saved false-positive fraction is 0.5127. This is a prominent failure mode in the saved analysis. Project-level results also vary: for `apache/hadoop`, the artifact records 1,147 false positives among 2,983 rows and no false negatives; for `apache/hadoop-hdfs`, it records 38 false negatives among 122 rows. These subgroups have different sample counts and are not a project-held-out evaluation.

## Ranking consistency checks

The file prioritizer combines changed status, graph proximity and path support, diff magnitude, repository history when reliable, and an optional shared commit risk. It omits history on initial/truncated/shallow history and omits commit risk when ML is unavailable, then renormalizes remaining configured weights. A shared commit risk does not distinguish files and therefore cannot change their relative ordering by itself.

`tests/test_impact_ranker.py` checks deterministic changed/direct/indirect ordering, tie-breaking, weak-evidence tier limits, removed-file handling, broken-import boosts, weight renormalization, and three comparison outputs: dependency-distance, changed-status, and full combined score. The result object explicitly describes these as rule-consistency comparisons, not accuracy measurements. No file-level failure labels or measured ranking metric such as precision-at-k are available in the saved evaluation artifacts.

## Known failure cases and limits

- The target is a commit label and cannot say which changed or dependent file/test caused a later bug.
- Issue/fix linkage and observation windows may censor or mislabel bugs; the current artifacts do not quantify this.
- The temporal split does not hold out projects, so it does not measure transfer to unseen projects.
- Apache training data may not transfer to other domains, languages, or repository workflows.
- Large-churn changes show a high saved FP/row fraction, as detailed above; a higher recall baseline also produces more false positives at its reported threshold.
- Shallow/truncated Git history reduces the reliability of `ndev`, `age`, `nuc`, `aexp`, and ranking history evidence. ImpactMap marks degraded features and omits unavailable ranking evidence, but it does not reconstruct missing history.
- No file-level ground truth is available to validate heuristic ranking against actual test failures.
- Dynamic imports, reflection, generated wiring, and non-Python dependencies are outside the static import graph.
- Test metrics are one saved experiment. The artifacts contain no repeated-seed variance, confidence intervals, external validation, or calibration curve.
