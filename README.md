# ImpactMap

ImpactMap is a local Streamlit application for reviewing Git commits. It combines Git diff metrics, static Python import relationships, a commit-level machine-learning risk score, and a transparent heuristic ranking to help reviewers decide where to focus testing. It reads committed Git objects and does not execute repository code.

## Problem statement

In large software projects, a small code change can affect multiple connected components, making it difficult for developers to identify all the areas that need testing. Manually tracing these dependencies is time-consuming and can lead to important test cases being missed. ImpactMap aims to analyze code changes, map relationships between software components, and use machine learning to estimate change risk and prioritize potentially affected files and tests. This helps developers make more informed testing decisions and reduces the effort required for code review and regression testing.

## Motivation

Reviewing a commit requires two different kinds of evidence: what changed and which files depend on it. A diff alone does not show import relationships; static dependency evidence alone does not describe the size or historical context of a commit. ImpactMap places these signals together, while keeping the learned commit risk score separate from its file-level heuristic priorities. The score and ranking are decision support; neither is a claim that a defect exists or that a suggested test is complete.

## Features

- Load a local Git repository or a public HTTPS repository hosted on GitHub, GitLab, or Bitbucket.
- Select a commit and inspect its first-parent diff, including additions, modifications, deletions, renames, merge commits, and binary files.
- Build a static Python import graph from Git blobs, including relative and package imports, with external, unresolved, and syntax-error reporting.
- Trace reverse dependencies and inspect removed or renamed modules using the parent commit graph.
- Extract the shared ten-feature commit schema and, when the saved model is available, show its commit-level risk score and explanation.
- Rank changed and potentially affected files with an auditable heuristic score, evidence, component breakdown, and deterministic comparison baselines.
- Export sanitized JSON analysis reports and ranked-file CSV reports.
- Continue showing graph and heuristic results when the ML artifact or prediction is unavailable.

## Architecture overview

`app.py` provides the Streamlit interface and calls the reusable analysis orchestration in `src/impactmap/analysis.py`. The orchestrator runs diff analysis, current and (where needed) parent dependency graph analysis, impact traversal, feature extraction, optional prediction and explanation, and file prioritization. The report generator serializes selected metadata through redaction rules.

The graph convention is **A → B means A imports B**. Therefore, dependents of a changed file are found by traversing the graph in reverse. Full module flow, cache keys, and invalidation behavior are described in [docs/architecture.md](docs/architecture.md).

## Workflow

1. Load a local repository path or enter a public HTTPS repository URL.
2. Select a commit from the repository history.
3. Run the analysis. ImpactMap reads the commit and parent Git objects directly.
4. Review the changed files, dependency neighborhood, commit-level risk and explanation when available, and heuristic test priorities.
5. Export the sanitized JSON report or ranked-file CSV if useful.

## Technology stack

The workspace was verified with **Python 3.13.9**. The pinned runtime dependencies include GitPython, NetworkX, pandas, NumPy, scikit-learn, SHAP, Streamlit, Plotly, and joblib. Pytest is included in `requirements.txt` for the project test suite. See [requirements.txt](requirements.txt) for exact versions.

## Installation and startup

These commands were run successfully in a fresh Python 3.13 virtual environment. The example uses macOS or Linux shell syntax.

```bash
git clone https://github.com/Nandinigarg117/ImpactMap.git
cd ImpactMap
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens in a browser. To use a local repository, enter its absolute path in the sidebar and select **Load repository**. A remote URL must use HTTPS and an allowed public host; embedded credentials are rejected. No `.env` file is required. `.env.example` is retained as a note that the application currently reads no environment-variable overrides.

## Example workflow

For a concrete run, choose any commit in a local Git repository that contains Python files. Load the repository, select a commit with Python changes, and click **Analyze selected commit**. In **Commit Explorer**, confirm the changed paths and line counts against `git show --numstat <commit>`. In **Dependency Explorer**, follow edges toward imported modules and review reverse dependents. In **Testing Priority**, inspect the evidence and component breakdown before choosing tests. If no model is installed or the model cannot predict, the app labels ML risk as unavailable while leaving graph results and ranking available.

## Screenshots

No screenshots are included in the repository. To capture current UI screenshots, run `streamlit run app.py`, load a local Git repository with commits, select a commit, and click **Analyze selected commit**. Capture the **Overview**, **Commit Explorer**, **Dependency Explorer**, **Risk Analysis**, **Testing Priority**, and **Report Export** tabs. For an ML-unavailable screenshot, temporarily make `models/model.joblib` unavailable, rerun the analysis, capture the explicit unavailable label, and restore the artifact before committing. Avoid screenshots containing private repository paths or source content.

## Dataset and model

The checked-in model was trained against **ApacheJIT**. The upstream [ApacheJIT repository](https://github.com/hosseinkshvrz/apachejit) identifies `apachejit_total.csv` as its total dataset and describes the `buggy` field as whether a commit introduced a bug. This project downloads that CSV from the URL in `scripts/prepare_dataset.py`, removes duplicate commit IDs, drops rows with missing fields or invalid negative size metrics, and creates chronological 70/15/15 train, validation, and test splits by `author_date`.

The current model artifact records **74,671 train rows**, **16,001 validation rows**, and **16,002 test rows**. The test positive-label base rate is **0.1366**. The dataset repository page and README do not declare a dataset license. The Apache-2.0 [LICENSE](LICENSE) in this repository applies to ImpactMap code; it does not establish rights to redistribute ApacheJIT. Check the upstream dataset terms before reuse or redistribution.

### Train or evaluate

Training requires the ApacheJIT CSV. The preparation script downloads it if it is not present locally; inspect the source and dataset terms before running it. Run these commands from the repository root in the activated environment:

```bash
python scripts/prepare_dataset.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

Preparation writes ignored files under `data/raw/` and `data/processed/`. Training writes `models/model.joblib`, `models/metrics.json`, `models/risk_tiers.json`, and `models/global_importance.json`. The evaluator requires `data/processed/test.csv`. These outputs can replace tracked model metadata and artifacts; review `git status` and diffs after running training.

### Saved test-set metrics

The following values are copied from [`models/metrics.json`](models/metrics.json), model version `1.0.0`, with the threshold recorded for each row. They are held-out test-set results on the saved split, not a guarantee of performance on another repository.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier score | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated HistGradientBoosting (primary) | 0.3861 | 0.8060 | 0.2999 | 0.7260 | 0.4244 | 0.1163 | 0.30 |
| Logistic regression baseline | 0.3824 | 0.7995 | 0.2547 | 0.8124 | 0.3878 | 0.2193 | 0.50 |
| Churn heuristic baseline | 0.2752 | 0.7557 | 0.2135 | 0.8760 | 0.3434 | 0.3103 | 0.50 |
| Dummy prior baseline | 0.1366 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.1424 | 0.50 |

At the primary threshold, the saved confusion matrix is **TN 10,111; FP 3,705; FN 599; TP 1,587**. Further evaluation details, data compatibility, split caveats, and error analysis are in [docs/evaluation.md](docs/evaluation.md).

## Repository structure

```text
app.py                    Streamlit UI
src/impactmap/            Repository, diff, graph, impact, feature, model, ranking, and export modules
scripts/                  Dataset preparation, model training, and saved-model evaluation
models/                   Tracked model, feature schema, metrics, risk tiers, and importance
tests/                    Unit and integration tests using temporary Git repositories
docs/                     Architecture and evaluation documentation
requirements.txt          Pinned Python dependencies
```

## Security considerations

ImpactMap parses source as text/AST and reads Git objects; it does not import or execute analyzed repository code. Remote cloning accepts HTTPS URLs for allowlisted hosts, rejects embedded credentials, disables submodules and hooks, uses shallow clone depth, and applies a timeout. Exports redact common tokens, email addresses, URLs' credential/query portions, and absolute local paths. Raw and processed datasets, virtual environments, temporary clone caches, and `.env` files are ignored by Git.

These controls reduce risk but do not make arbitrary repositories safe. Git object parsing, diff generation, AST parsing, archive/object sizes, malformed metadata, and third-party libraries remain attack surfaces. File count and individual blob size are capped for graph indexing, but repository storage, diff processing, total runtime, and memory are not comprehensively sandboxed. Analyze untrusted repositories in an isolated environment with resource limits; do not treat URL validation or static parsing as a security boundary. Review the full residual-risk discussion in [docs/architecture.md](docs/architecture.md#security-boundary-and-residual-risk).

## Limitations and future improvements

- Dependency discovery is static, Python-only import analysis. Dynamic imports, reflection, dependency injection, generated code, and framework wiring can be missed.
- The model predicts a **commit-level** `buggy` label. It does not assign a validated failure probability to each file or test.
- ApacheJIT contains Apache projects; domain shift can affect other languages, organizations, and repository practices.
- Bug-inducing labels depend on the dataset's issue/fix linkage and observation process. Unreported, unlinked, or delayed bug findings can be censored or mislabeled.
- Shallow or truncated history degrades history features. The app exposes degradation and can omit unavailable ranking components.
- File prioritization is heuristic. Its deterministic baseline comparisons are consistency checks, not validation against file-level ground truth.
- The saved metrics do not include confidence intervals, calibration curves, or independent cross-project evaluation.

Future work can add independently labeled file/test outcomes, cross-project and cross-language evaluations, stronger calibration reporting, broader language support, and more explicit resource isolation for untrusted repositories.

## License

ImpactMap source code is provided under the [Apache License 2.0](LICENSE). The ApacheJIT dataset is separate; its license was not declared in the upstream repository documentation inspected for this README.
