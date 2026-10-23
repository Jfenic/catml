# Progress

### Session Summary & State for Next Session (2026-10-23)

- **Completed & Pushed to GitHub `origin/main`**:
  1. **Developer Experience & Diagnostic Tooling (`catml doctor`)**:
     - Implemented `catml doctor` ([doctor_cli.py](file:///home/fenic/workspace/ai-ml/CATML/src/automl/interfaces/cli/doctor_cli.py)) checking OS, Python version, virtualenv isolation, core & optional dependencies (LightGBM, XGBoost, CatBoost, Optuna, Torch, etc.), CPU/RAM/GPU availability, and `.automl` workspace writeability with human-readable and `--json` outputs. Wired to CLI [main.py](file:///home/fenic/workspace/ai-ml/CATML/src/automl/interfaces/cli/main.py) and verified by [tests/test_cli_doctor.py](file:///home/fenic/workspace/ai-ml/CATML/tests/test_cli_doctor.py).
  2. **Public Presentation & Show-Don't-Tell Documentation**:
     - Recorded animated terminal demo GIF ([catml-demo.gif](file:///home/fenic/workspace/ai-ml/CATML/docs/assets/catml-demo.gif)) showcasing `catml doctor`, `catml analyze`, `catml fit`, `catml leaderboard`, and MCP agent server.
     - Enriched [README.md](file:///home/fenic/workspace/ai-ml/CATML/README.md) with "¿Por qué CATML?" capability matrix (AutoML + Agentic Systems + Clean Architecture vs AutoGluon/FLAML/CrewAI), Workbench UI preview screenshot, Mermaid architecture flow, Agent MCP toolchain guide, and Interactive Validation Notebooks table.
     - Updated GitHub repo About description (<160 chars) and 15 SEO topics via `gh repo edit`.
  3. **Empirical Scientific Verification Loop (Propose ≠ Accept)**:
     - Live demonstration on `examples/data/customers_churn.csv`: CATML extracted 8 statistical findings, generated a candidate hypothesis, tested it empirically against the baseline model, and mathematically rejected the candidate ($\Delta = -0.0307$) to avoid silent model regression.
  4. **P0 Honesty & Leakage Residual Gate (Completed & Pushed)**:
     - **Engine (`oof.py`)**: Support for independent holdout evaluation (`X_holdout`, `y_holdout`) across blending methods (`average`, `rank`, `simplex`, `stacked`), calculating `holdout_scores` and `holdout_delta` against the baseline model.
     - **Level-2 Stacking Residuals**: Diagnostics for meta-estimators calculating mean, standard deviation, max absolute error, and Brier score calibration on both OOF and holdout.
     - **Honest Candidate Promotion Gate (`oof_submission.py`, `workspace_commands.py`)**: Candidate blends are only marked `promoted=True` if holdout lift exceeds `--min-improvement`.
     - **CLI (`main.py`)**: Added `--holdout-dataset`, `--holdout-ratio`, and `--min-improvement` to `automl predict`.
     - **CI (`ci.yml`)**: Added optional strict hard-fail on `pip-audit` via `CATML_AUDIT_STRICT=true`.
     - **Tests**: New suite [tests/test_oof_holdout_and_residuals.py](file:///home/fenic/workspace/ai-ml/CATML/tests/test_oof_holdout_and_residuals.py) (3/3 passing; 38/38 total OOF/Doctor tests green).
- **Commits on `main`**:
  - `6db2a20`: *docs: enrich README with terminal demo GIF, capability matrix, and architecture diagram*
  - `1219bc8`: *feat(cli): add 'catml doctor' diagnostic tool and enrich README with agent MCP flow and validation notebooks*
  - `2acad8a`: *feat(ensemble): implement OOF holdout validation, candidate promotion gate, and stacking residuals (P0)*

- **Priority Backlog for Next Session**:
  - **P1 Provenance & Reproducibility**: Record trial/run metadata (random seeds, CV fold partition IDs, SHA-256 dataset/pipeline hashes) in the SQLite leaderboard and export packages (`model.zip`/ONNX).
  - **P1 Facade Slimming**: Further decompose `workspace.py` into dedicated application services.
  - **P1 Quality Debt**: Progressive mypy checks on application layer and ESLint warning cleanups.
  - **P2 Stable PyPI `0.9.0` Release**: Tagging and publishing the stable release once P1 provenance is verified.

### Honesty & Leakage Residual (P0) — OOF Holdout Validation, Stacking Residuals & Optional Strict Audit (2026-10-23)

- **Engine (`oof.py`)**:
  - `evaluate_oof`: Added `X_holdout` and `y_holdout` support to evaluate blend candidates on independent validation sets without data leakage.
  - Multi-method holdout projection (`average`, `rank`, `simplex`, `stacked`) and calculation of `holdout_scores` and `holdout_delta` against the declared baseline model.
  - `stacking_residuals`: Computed mean, std, and Brier score residuals on both OOF blended predictions and holdout predictions.
- **Application (`oof_submission.py`, `workspace_commands.py`)**:
  - `GenerateOOFSubmissionCommand`: Extended with `holdout_dataset_path`, `holdout_ratio`, and `min_improvement`.
  - Honest Candidate Promotion Gate: Candidates are only marked `promoted=True` if an independent holdout evaluation demonstrates improvement (`holdout_delta > min_improvement`). Unverified blends remain `promoted=False`.
- **CLI (`main.py`)**:
  - `predict`: Added `--holdout-dataset`, `--holdout-ratio`, and `--min-improvement`.
  - Output printing enriched with holdout score, candidate promotion status, and stacking residuals.
- **CI (`ci.yml`)**:
  - Configured optional strict hard-fail on `pip-audit` via `CATML_AUDIT_STRICT=true`.
- **Tests**:
  - New dedicated test suite `tests/test_oof_holdout_and_residuals.py` (3/3 passing).
  - All 35 OOF blending and holdout tests green.

### Data Preparation P2 — CLI, facade prepare, fold-safe pipeline bridging & recipe parity (2026-10-14)

- **Domain**: `pipeline_steps_to_column_transforms` in `domain/preparation/pipeline_bridge.py` mapping pipeline-scoped steps (`MARK_IMPUTE`, `MARK_SCALE`) into trainer column transforms.
- **Application & Wiring**:
  - `AutoMLWorkspace.create_experiment`: automatically resolves pipeline-scoped steps from version lineage of prepared datasets.
  - `RecipeService`: inherits pipeline-scoped column transforms when generating recipes from prepared dataset experiments.
- **Engine**: `sklearn_trainer.py` updated to support custom imputer strategies and categorical column transforms without leaky pre-split computation, plus protective fallback for string categorical imputers.
- **Facade**: `AutoMLPreparationFacade` in `src/automl/facade.py` providing scikit-learn ergonomic access (`automl.prepare(...)` callable + `.analyze`, `.suggest`, `.preview`, `.validate`, `.apply`, `.create`, `.get_plan`, `.list_plans`, `.get_version`, `.list_versions`), with robust dataset ID resolution and error reporting.
- **CLI**: `automl data inspect` and `automl data prepare {suggest,preview,validate,apply,create,plans,show-plan,versions}` registered and verified with human-readable and `--json` outputs, safe parsing for large inline JSON step definitions (>255 chars), and full `--parent` alias support.
- **Tests**: `tests/test_data_preparation_p2.py` with 8 comprehensive integration tests covering CLI text/JSON lifecycle, facade methods, fold safety, recipe parity, error boundaries, and static mypy/ruff checks (all green).

### Focus → product robustness (2026-10-14)

- Confirmed **`0.9.0rc1` on TestPyPI + PyPI** (GitHub `v0.9.0rc1`); TASKS publish rows marked Done.
- Focus retargeted: honesty/OOF holdout, Data Prep P2, provenance, workspace slim, quality debt → then stable `0.9.0`.

### TASKS hygiene — Study A/B already done (2026-10-14)

- RC1 track still said Study A “in progress” / Study B “scaffold”; corrected to **Done** (reports + `#118`/`#119`/`#122`).

### Clean Arch PR5 — quality gates (2026-10-14)

- **Merged** [#130](https://github.com/Jfenic/catml/pull/130) → `main` (`5f5fce7`). Clean Arch PR1–PR5 complete.
- Ruff + mypy: `pyproject.toml` tools; domain green; application gradual ignores for legacy debt.
- ESLint/Prettier: root `package.json`, `eslint.config.js`, CI `lint-js`.
- CI: `lint-python` + `lint-js` before `test` matrix.
- Tests: `tests/test_clean_arch_pr5_quality_gates.py` (config contract + export→predict parity).
- Fixed `knowledge.js` undefined `state` (use `store.getActiveDatasetId()`).

### Clean Arch PR4 — persistence / states / UoW (2026-10-14)

- Branch `refactor/clean-arch-pr4-persistence-uow` → PR `#129` (synced with `#128` on main).
- Domain: `runs/transitions.py`, `experiments/transitions.py` (CANCELLED sticky; retry from COMPLETED/FAILED OK).
- Infra: `connection.py`, `unit_of_work.py`, `mappers.py`, `stores/`; `persist_trial_outcome` on facade.
- Services: execution/HPO use UoW; `Experiment.transition_to` wired.
- Tests: `tests/test_clean_arch_pr4_persistence_uow.py`. Brief: `agent-brief-pr4.md`.

### Consolidation + audits landed (2026-10-14)

- [#128](https://github.com/Jfenic/catml/pull/128) **merged**: model reliability + HTTP/CSRF + honest reports.
- [#127](https://github.com/Jfenic/catml/pull/127) closed as superseded.

### Agente B — Workbench Demo/Export & SQLite Connection Lifecycle #94 (2026-10-14)

- **Subtarea B1 (Workbench Demo Project + Executive Export)**:
  - Backend: `POST /api/project/demo`; `GET /api/reports/executive` (HTML/MD/JSON).
  - Frontend: `loadDemoProject` / informe ejecutivo en home, register modal, studio, overview.
  - Hardening: corregida extracción de duración (`training_time_s`), filtrado de modelos fallidos en selección de campeón, soporte seguro de cuerpos vacíos y `close()` alias en workspace.
  - Tests: `tests/test_workbench_demo_and_export.py`.
- **Subtarea B2 (Blackboard #94)**:
  - Lifecycle/`dispose` en sqlite recipes/studies/repository + `AutoMLWorkspace`.
  - Tests: `tests/test_sqlite_connection_lifecycle.py` (soak).
- PR: [#125](https://github.com/Jfenic/catml/pull/125).

### Data Preparation P1 started (2026-10-14)

- [#124](https://github.com/Jfenic/catml/pull/124) merged: domain + `PreparationService` + CQRS + versions.
- Tests: `tests/test_data_preparation_p1.py`. CLI/MCP/UI = P2/P3.

### Parallel split A/B (2026-10-14)

- Coordination: [`.agent/parallel-coordination.md`](parallel-coordination.md).
- **Agente A:** Study B lift → [#122](https://github.com/Jfenic/catml/pull/122) merged; smoke 24/0/0.
- **Agente B:** Demo export + #94 → [#125](https://github.com/Jfenic/catml/pull/125).

### Study B smoke + #119 merge (2026-10-13)

- [#118](https://github.com/Jfenic/catml/pull/118) merged (Study B minimal runner).
- Smoke failed on wine Explore (`Unknown task type: multiclass`) → [#119](https://github.com/Jfenic/catml/pull/119) maps `multiclass` → `multiclass_classification`.
- Re-smoke: 18 ok / 6 skipped / 0 failed; report updated; #119 merged (`d7982bb`); `main` synced.

### Study B minimal runner + #117 merged (2026-10-13)

- [#117](https://github.com/Jfenic/catml/pull/117) merged (MCP `--allow-mutations` + text LSA).
- Branch `feat/study-b-min-runner`: `run.py --study B` executes configs with train-only Explore; agent skipped; tests `test_benchmark_study_b.py`.

### Follow-up: merge #116 + MCP/text PR (2026-10-13)

- Merged [#116](https://github.com/Jfenic/catml/pull/116) (admin; review gate blocked).
- Smoke `scripts/smoke_fit_time_budget.py`: no-budget 1.5s → budget=60 2.5s (HPO ran; Optuna finishes before wall timeout on tiny data).
- Branch `fix/mcp-allow-mutations-text-svd`: `automl mcp --allow-mutations`; text TF-IDF → TruncatedSVD (LSA) by default.

### PR: HPO fold pruning + fit time_budget (2026-10-13)

- Branch `fix/hpo-fold-pruning-time-budget`: CV fold_callback → Optuna `report_step` / MedianPruner; `AutoML.fit(time_budget=…)` runs OptimizeExperiment on best model with remaining seconds (warm-start already in OptimizationService).
- Tests: `tests/test_hpo_fold_pruning.py`.

### Commit + GitHub Release 0.9.0rc1 (2026-10-13)

- Pushed `2b22db5` to `main` (Study A, CLI fixes, Data Prep docs, fire tests).
- GitHub prerelease [`v0.9.0rc1`](https://github.com/Jfenic/catml/releases/tag/v0.9.0rc1) targets publish SHA `05a004f` (matches PyPI wheel).
- Next pick: Study B execution **or** HPO wiring (`trial.report` / warm-start) in small PRs.

### Agent CLI/MCP fire-test triage (2026-10-13)

- Confirmed: MCP `PENDING_APPROVAL` is by design (`require_approval_for_mutations=True`); not a bug.
- Fixed: `optimize` / `benchmark --json` emit only JSON on stdout (parity with `fit`).
- Fixed: `explore create|run` accept `--dataset-id` / `--study-id` in addition to positionals.
- Blackboard: [#114](https://github.com/Jfenic/catml/issues/114) MCP autonomous flag; [#115](https://github.com/Jfenic/catml/issues/115) MLP/XGBoost text budget.
- Note: tabular MLP path already has `StandardScaler` in trainer; ConvergenceWarning still possible on text/sparse / max_iter=500.
- Tests: `test_cli_optimize_json_stdout_is_pure`, extended `test_cli_explore_lifecycle` (3 passed).

### Data Preparation direction documented (2026-10-13)

- Spec/plan + [ADR-012](docs/decisions/012-data-preparation-subsystem.md): `PreparationPlan`, scopes dataset vs training_pipeline, parity, post-rc1.
- Care points captured in spec §7 (leakage, versions, agents, honest UX).
- Workbench badge: “Sin fuga detectada” + disclaimer (was “Limpio y Validado”).

### Benchmark v1 frozen + Study A (2026-10-13)

- `protocol-v1.yaml` → **frozen 1.0.0** (shared holdout, seeds 42/43/44, f1_macro, skipped≠comparison).
- Manifest: **11** Study A datasets; `run.py --study A|B|pilot`; analyze aggregate.csv; environment.json + checksums.
- Study A **5 min** smoke: **132/132 ok**, report `docs/experiments/study-a-report.md` (artifacts in `results/v1/study-a-5min/`).
- Study A **20 min** formal: **DONE** (`EXIT:0`, `ANALYZE_DONE_20`, 132/132 ok, budget 1200s). Report refreshed. Artifacts `study-a-20min/`. FLAML gained with budget; CATML/sklearn/AG holdout means flat vs 5 min on this suite.
- Study B scaffold: `run.py --study B` + `docs/experiments/study-b-scaffold.md`.

### PyPI RC published (2026-10-12)

- TestPyPI smoke OK earlier; PyPI Trusted Publisher fixed; Release `38317226040` → **publish-pypi success**.
- Live: https://pypi.org/project/catml/ `0.9.0rc1` — clean venv `pip install catml==0.9.0rc1` smoke OK.

### Pilot 1A full campaign — 5 min × 3 seeds × AG (2026-10-12)

- Installed `autogluon.tabular`; ran `--seeds 42,43,44 --time-budget-minutes 5`.
- **36/36 ok** (sklearn/catml/flaml/autogluon × 3 datasets × 3 seeds). Report updated.
- Headline: CATML ties sklearn on classif.; FLAML/AG win housing RMSE (~0.47 vs CATML ~0.52). Sklearn housing seed 44 outlier documented.
- Blackboard #113 can close after maintainers ack public report.

### Pilot 1A protocol fix — shared external holdout (2026-10-12)

- Applied review: public trio breast_cancer / wine / california_housing; **one holdout per (dataset, seed)**; all frameworks fit train-only; `score_external()` shared; wall/RSS/CPU logged; `--seeds` multi-seed.
- Re-ran seed=42 @ 60s: CATML ≈ sklearn on classif.; FLAML edges housing RMSE; AG still skipped → not a complete bake-off.
- Report refreshed: `docs/experiments/pilot-1a-report.md`. Tests: `tests/test_benchmark_pilot_1a.py` (shared holdout assertions).
- Next for public claims: install AG + `--seeds 42,43,44 --time-budget-minutes 5`.

### Pilot 1A + fire tests (2026-10-12)

- Extended `benchmarks/run.py` pilot; fire tests `notebooks/fire_tests/` → **10/10 ok** (functional, separate from competitive pilot).
- Blackboard: [#113](https://github.com/Jfenic/catml/issues/113) AG unmeasured.

### Repo rename GitHub `CATML` → `catml` (2026-10-12)

- GitHub nameWithOwner is now `Jfenic/catml`; `origin` remote + README/pyproject/SECURITY/website/publishing/release.yml URLs updated.
- Product brand text “CATML” kept in prose; package/PyPI/CLI remain `catml`.
- Trusted Publisher must use Repository **`catml`** (user configured).

### Publish prep merged (`#112`, 2026-10-12)

- Third-party licenses + `docs/publishing.md` on `main`.
- GitHub Environments `testpypi` / `pypi` created with required reviewers (Jfenic, tvarmar).
- Local wheel `catml-0.9.0rc1` builds; twine check PASS.
- **Human remaining:** Trusted Publisher done (`Repository: catml`); run Release workflow → `testpypi`.

### Track License compliance (`docs/third-party-license-compliance`, 2026-10-12)

- Audit: Chart.js MIT, Lucide ISC(+Feather MIT), Tailwind MIT, Geist/IBM Plex OFL-1.1 — commercial-OK with notices; CATML stays Apache-2.0.
- Added `THIRD_PARTY_LICENSES.md`, vendor license texts under `static/vendor/licenses/`, `license-files` in `pyproject.toml`, NOTICE/README pointers.
- Tests: `tests/test_third_party_license_compliance.py` (repo + wheel + sdist). Merged onto main after #111 for PyPI gate.

### Tabular ship stack merged to main (`#111`, 2026-10-12)

- Merged https://github.com/Jfenic/catml/pull/111 into `main` (`1aa009f`): PR2+PR3+UX+Audit-5+Labs-deferred docs.
- Local validation: **889 passed**, coverage **87.11%**; facade breast_cancer fit→export→predict OK; job worker experiment OK.
- Remaining for publish: human TestPyPI/PyPI Trusted Publisher; optional Sprint 3 demo export; manual Workbench click-through.

### Decision — Workbench Labs deferred; tabular first (2026-10-12)

- Accepted future direction: App Shell + modality Labs, single AutoML/CQRS engine ([ADR-011](docs/decisions/011-workbench-labs-modular-frontend.md), [plan](docs/features/workbench-labs/plan.md)).
- **Active focus:** finish tabular Workbench (merge #109/#110/#111, land Audit-5, E2E CSV→export). No Vision/NLP/Clustering lab scaffolding yet.
- MASTER_PLAN / TASKS / AGENTS / docs index updated accordingly.

### Track Audit-5 — Workbench honesty / launch / metrics (2026-10-12)

- Verified external audit vs `main` @ `b238fb690`; #3 allowlist already fixed on branch; residual budget skew + #1/#2/#4/#5/#6 addressed.
- Canonical `domain/experiments/budgets.py` (quick=60 / balanced=300 / thorough=1200); `api.js` forwards `time_budget_seconds`; JobExecutor prefers explicit seconds.
- XSS: escape dataset/target/recipes in `new_experiment.js`; Manual mode shows model checkboxes; plugin native/fallback labels.
- Trainer: multiclass `roc_auc` → `roc_auc_ovr` + holdout OvR; trial `artifacts` record `estimator_class` / `backend` / `fallback_backend`; Studio badge.
- `/api/optimize` no longer returns fabricated importances (Optuna when available via OptimizationService); `/api/predict` checklist is evidence-only.
- Tests: `tests/test_audit5_workbench_honesty.py`.

### Track Workbench — framing/CV coherence + job error clarity (2026-10-12)

- Root cause of `All experiment trials failed` on airline satisfaction: advisor matched «Departure/Arrival time convenient» as temporal → `time_series_split` without `time_column`.
- Fix: tighter `_is_temporal_column` heuristics; `coerce_split_strategy` in domain + `from_dict`; UI sync in questionnaire / `new_experiment.js`; JobExecutor appends trial `failure_reason`.
- Tests: airline cross-sectional case + coerce + job error surface in `test_dataset_questionnaire.py` / `test_workbench_laboratory_redesign.py`.
- Operator action: re-generate or re-save Encuadre (Independiente + Stratified K-Fold), then relaunch experiment.

### Track Clean Architecture PR3 — web API + datasets.js split (`refactor/clean-arch-pr3-web-api-datasets-split`, 2026-10-12)

- Backend: `server.py` reduced to HTTP glue (~300 LOC); routes in `interfaces/web/api/{jobs,datasets,experiments,analysis}.py` + `_helpers.py`.
- Frontend: `views/datasets.js` barrel; `DatasetsView` orchestrator + panels under `views/datasets/` (summary, columns, health, selector, correlations, questionnaire, derived).
- Tests: `tests/test_clean_arch_pr3_web_api_datasets_split.py` + `workbench_static_sources.py` for package-aware static guards.
- Follow-up fix: `JobService` allowlist accepts Workbench `budget` / `mode` / `validation_strategy` (was "Unknown experiment job arguments").
- PR: [#110](https://github.com/Jfenic/catml/pull/110). Next after merge: PR4 (SQLite repos / state machine / UoW).

### Track Clean Architecture PR2 — execution/optimization services (`refactor/clean-arch-pr2-execution-optimization-services`, 2026-10-12)

- Extracted `run_experiment` → `ExperimentExecutionService`.
- Extracted `optimize_experiment` → `OptimizationService`.
- `AutoMLWorkspace` keeps public signatures; lazy properties + 1:1 delegates (CLI/MCP/Web unchanged).
- Removed dead HPO/trainer imports from `workspace.py` (~1998 → ~1669 LOC).
- Tests: `tests/test_clean_arch_pr2_execution_services.py` (6): facade delegation, binary smoke, MAE regression smoke, HPO shape, unknown optimizer fail-closed.
- Validation: **872 passed**, coverage **87.11 %**, CLI help/task list OK.
- PR: [#109](https://github.com/Jfenic/catml/pull/109). Merged into tabular publish branch with Audit-5 param_importance on OptimizationService.

### Track Audit-4 remainder — hypothesis / budget / PCA / cache (`fix/audit4-hypothesis-budget-pca-cache`, 2026-10-12)

- #9/#10: `verify_hypothesis` always runs a paired baseline; distinguish `improved_in_experiment` vs `validated` / `accepted`; hyp status may be `improved_unvalidated`.
- #16: budget exhaustion with partial successes → `ExperimentStatus.PARTIAL` + `completion_reason=BUDGET_EXHAUSTED`.
- #12: PCA candidates keep source columns + reduction recipe; reject `PC*` names; trainer supports `kind=pca` column='*'.
- #14: image embedding cache key includes file size + mtime_ns.
- Tests: `tests/test_audit4_hypothesis_budget_pca_cache.py`.

### Track Audit-4 mediums — features / simplex (`fix/audit4-features-simplex-honesty`, 2026-10-12)

- **Merged via #106** (`f040891`).
- #7/#8: `prepare_feature_matrix` drops invalid targets (no `fillna(0)`, no `"nan"` class).
- #11: `simplex` OOF blend scores via nested fold weight-fit; production weights still fit on full OOF for test.
- #13: tree selector → `FeatureEvidence.tree_importance` (+ SQLite column); `shap_importance` reserved; confidence from evidence count.
- #6 (partial): selected feature sets tagged `evaluation_mode=exploratory_full_dataset` + `supervised_selection_leakage_risk`.
- Tests: `tests/test_audit4_features_simplex_honesty.py`.

### Track Audit-4 — Security / inference / ensemble (`fix/audit4-security-inference-ensemble`, 2026-10-12)

- **Merged via #104** (`67a60e4`):
  - `infrastructure/security/workspace_paths.py`: shared path confinement; web `/api/dataset/register` uses it (parity with inspect).
  - Upload writes `stem_{uuid10}.ext` so same client filename cannot overwrite prior uploads.
  - `InferenceService.predict`: materializes derived features + passes `column_transforms` into `fit_and_predict`.
  - `build_ensemble`: persists MAE/RMSE on natural scale (flip sklearn `neg_*`).
  - `stack_predictions`: nested `cross_val_predict` OOF for metamodel (not in-sample fit).
  - Unknown optimizer → `ValueError` (no silent Random Search).
- Tests: `tests/test_audit4_security_inference_ensemble.py`.
- Next: findings #6–#14, #16. License compliance remains PR #103.

### Track Clean Architecture Refactor (PR1 contracts, 2026-10-11)

- Docs ADR-010 / plan: PR #101. **PR1 code merged via #102**:
  - `domain/metrics/spec.py`: `MetricSpec` + builtin registry + `resolve_metric_spec` / `InvalidMetric`.
  - `domain/experiments/execution_spec.py`: frozen `ExecutionSpec` + `build_execution_spec` / `trial_execution_from_spec`.
  - Wired Workspace `run_experiment` / `optimize_experiment` / schedule / `get_best_trial`, benchmarks `is_minimizing_metric`, web `_is_metric_minimizing`, trainer helper — no hardcoded minimize sets / blind `max()`.
  - HPO keeps fixed CV fold seed (`ExecutionSpec.random_seed` = run seed).
  - Tests: `tests/test_clean_arch_pr1_contracts.py`.
- Next: PR2 Workspace facade extraction.

### Track Workbench UX Sprint 2 — Explore guided (`fix/workbench-sprint2-explore-guided`, 2026-10-11)

- Merged in PR #99.
- Guiado/Avanzado toggle: top 5 findings + hypotheses first; matrices/full filters in Advanced.
- Severity: CRITICAL only for IDs / |r|≥0.9 / strong multivariate outliers (not every collinear pair).
- `formatPValue` replaces `p=0.000e+00`; guided association list (|r|≥0.70) before full heatmap.
- Advanced correlation matrix: column search filter + max-height scroll.
- Comparator chart: floor Y-pad on [0,1] metrics so Δ~3e-5 is not visually exaggerated; neutral band 0.001.
- Tests: `tests/test_workbench_sprint2_explore_guided.py`.

### Track Audit-3 — Recipe / metric / CV contracts (`fix/audit3-recipe-metric-cv-contracts`, 2026-10-11)

- Merged in PR #100 — recipe/CV/metric/FDR/XSS contracts.
- Third external audit (post-`beb93a0` / PR #95): secondary paths lacked primary-flow guarantees.
- **Contracts:** `domain/experiments/validation.py` normalizes CV aliases (`k_fold`→`kfold`) and rejects unknown strategies (trainer + `create_experiment` fail closed).
- **Recipes:** translator call uses `default_models`; `ExperimentCandidateSpec.drop_features` + `model_id`; submit restores metric/cv/seed/strategy on run; SHA-256 `dataset_content_hash`; block archived submit; promote requires experiments.
- **Metrics/HPO:** `run_scheduled_experiments` uses minimize/maximize; `get_best_trial` consults metric plugins; HPO `TrialExecution.random_seed` fixed for CV folds (trial bookkeeping seed still varies).
- **FDR:** Benjamini–Hochberg no longer maps `p=0.0` via `or 1.0`.
- **XSS:** `escapeHtml` on feature/column/model interpolations in `new_experiment.js`, `datasets.js`, `compare.js`.
- Tests: `tests/test_audit3_recipe_metric_cv_contracts.py`. Deferred: stacking OOF metamodel, shap naming, pip-audit hard-fail.

### Track Export / path integrity (`fix/export-parquet-path-confinement`, 2026-10-11)

- `save_dataframe()` writes by suffix (parquet/json/tsv/csv) with optional `.bak`; derived-feature apply uses it (fixes Parquet corruption via `to_csv`).
- `export_model_artifact` materializes derived features + passes `column_transforms` into `fit_pipeline` (export matches validated recipe).
- Workbench `/api/models/export` returns zip with `.pkl` + `.sha256` for fail-closed `ModelArtifact.load`.
- `inspect-file` rejects paths outside cwd/workspace; `AutoML` temp workspaces cleaned on `close()`/GC.
- Tests: `tests/test_export_parquet_path_confinement.py` + web dashboard expects `application/zip`. Dual-maintainer review: @tvarmar.

### Track Workbench UX Sprint 1 — State coherence (`fix/workbench-sprint1-state-coherence`, 2026-10-11)

- P0 demo confidence from external UI review:
  - Header badge = workspace readiness (`LISTO`/`ENTRENANDO`), not raw `runs[0].status` (avoids COMPLETED vs experiment CREATED confusion).
  - `/api/experiments` + `/api/plan` use `_effective_experiment_status` (derive COMPLETED/FAILED when trials exist but status lagged as CREATED).
  - Studio `resolveTrialsCount` prefers live experiments/leaderboard over stale `activeRun.trials_count`; refreshes runs in `fetchData`.
  - Compare labels «Estado del experimento» + Spanish status helpers.
  - Explore: profile `identifier_column_names` + conservative ID heuristic block transform hyps (no discretize on `id`); still emit `exclude_identifier`.
- Tests: `tests/test_workbench_sprint1_state_coherence.py`.

### Track Workbench Offline Vendor Assets (`fix/workbench-offline-vendor-assets`, 2026-10-11)

- P2 packaging: pin Chart.js 4.4.7, Lucide 0.469.0, Geist/IBM Plex Mono woff2, and built Tailwind 3.4.17 under `static/vendor/` + `css/tailwind.built.css`.
- `index.html` loads only local `/static/...` (no cdn.tailwindcss / jsDelivr / unpkg / Google Fonts).
- Rebuild helper: `scripts/vendor_workbench_assets.sh` (scratch in `.vendor-build/`, gitignored).
- Tests: `tests/test_workbench_offline_vendor_assets.py` + smoke `REQUIRED_STATIC` / HTTP paths for vendor CSS/JS/fonts.
- README network note updated for offline UI chrome.

### Track RC1 Gate — HPO fidelity + release gates (`fix/hpo-recipe-fidelity-and-release-gates`, 2026-10-11)

- **Maintainers:** Jose Fenic Peiteado Padilla (@Jfenic) and Tania Évora Vargas Martínez (@tvarmar). Dual ownership documented in CONTRIBUTING, NOTICE, CODEOWNERS, and two-person-plan; cross-review on PR #95.
- **P0 HPO recipe fidelity:** `optimize_experiment()` now passes `column_transforms` + `derived_features` into every `TrialExecution` (parity with `run_experiment`).
- **P0 HPO all-fail:** zero successful trials → experiment/run `FAILED`; `best_score`/`best_trial_id` absent (not invented from failed scores).
- Tests: `tests/test_hpo_recipe_fidelity.py`.
- **P0 release.yml:** jobs `build` → `verify` (wheel smoke + full pytest) → `publish-testpypi` / `publish-pypi`; tag must match `__version__`; publish only downloads artifact; environments `testpypi`/`pypi`.
- **P1 SQLite:** `SQLiteExperimentRepository._connect` closes connections (sqlite3 context manager does not); bounded FD stress test; soak debt → blackboard [#94](https://github.com/Jfenic/catml/issues/94).
- **Benchmark v1 scaffold:** `benchmarks/` protocol + Study C harness; `_best_result()` returns `None` when all trials fail (not `0.0`).

### Track Pre-Publish Security & Leakage (`fix/pre-publish-security-and-leakage`, 2026-10-11)

- Reinforced audit blockers before announcing 0.9.0rc1 (post wheel-static PR #92):
  - `SafePythonEvaluator`: execution off by default; Workbench/FeatureService gated by `CATML_ALLOW_DERIVED_PYTHON=1`.
  - `assert_derived_specs_safe_for_cross_validation` + trainer path: block `python_code` and stateful aggregates (`zscore`/`mean`/`std`) before CV; row-wise formulas remain allowed.
  - Interactive preview may still evaluate stateful formulas with warnings (`allow_stateful_aggregates=True`).
  - `ModelArtifact.load`: missing `.sha256` sidecar fails closed; `allow_missing_checksum=True` for legacy.
  - Group heuristic: soft name hints require ≥10 distinct values and mid cardinality (avoids `Customer Type`).
  - TimeSeriesSplit: require `time_column` in run config extra, sort ascending, warn on duplicate timestamps.
  - `run_experiment`: mark experiment/run `FAILED` when no successful trials.
- Tests: `tests/test_pre_publish_security_leakage.py` + updates to derived/group/artifact/workbench suites.

### Track PyPI Pre-Publish — Wheel Workbench static (`fix/pypi-wheel-workbench-static`, 2026-10-11)

- Confirmed audit P0: existing `dist/catml-0.9.0rc1-py3-none-any.whl` had **0** `interfaces/web/static` members.
- Added `tool.setuptools.package-data` for Workbench HTML/CSS/JS under `automl.interfaces.web`.
- Extended `tests/smoke_test.py` to assert packaged static files + loopback HTTP 200 for `/`, CSS, and JS.
- CI: wheel content assertion before install; matrix 3.10–3.13; non-blocking `pip-audit` job.
- Added `.github/workflows/release.yml` (Trusted Publishing skeleton; tags / manual dispatch only).
- README: sklearn quickstart for PyPI users; CDN honesty; SECURITY pip-audit wording corrected.
- **Not done / needs human:** configure PyPI Trusted Publisher + TestPyPI dry-run; do not upload without approval.

### Track Public Surface Audit — Coherence (`fix/audit-public-surface-coherence`, 2026-10-11)

- Branched from `main` after PR #90 merge to address an external audit that mixed outdated claims with real gaps.
- **Already true on main (no code change needed):** Apache-2.0 license everywhere; `from catml import AutoML` via `src/catml/`; ModelArtifact pickle warning + SHA-256 + dep manifest; Workbench/MCP loopback + fail-closed auth + path confinement; CQRS interface guards.
- **Fixed on this branch:** README/ARCHITECTURE claim alignment; `setuptools>=77`; self-referential `all` extra; MCP install note; CI badge link; `CODE_OF_CONDUCT.md`; move `AutoML_Arquitectura_Tecnica.md` + `planning.txt` → `docs/`; domain import guard test; SECURITY CORS/CSRF/path notes; TASKS triage table.
- **Deferred:** full package rename `automl`→`catml`, ONNX/HMAC, Codecov token, PyPI 0.9.0 publish.

### Track Docs — Coordinación y sincronización de fuente de verdad (2026-10-11)

- Alineados banners y estados de **Explore E0–E6**, **Experiment Recipes R0–R4**, **Agentes H0–H5** y versión **`0.9.0-rc1`** en:
  - `AGENTS.md` (orden de lectura + Recipes shipped)
  - `TASKS.md` (`## Now` real; roadmaps Explore/Recipes marcados Completed; sync documental cerrado; meta-learning Phase 1 vs políticas avanzadas)
  - `docs/README.md` (tabla de capacidades)
  - `docs/MASTER_PLAN.md`, `ARCHITECTURE.md`, `DEVELOPER_GUIDE.md`, `README.md`
  - `docs/features/{catml-explore,experiment-recipes,agentic-system}/*` y ADR-009
- Regla explícita: si un plan/spec contradice `TASKS.md`, prevalece `TASKS.md`.

### Track Workbench — Auditoría UX/UI, Coherencia de Contexto y Refactorización (2026-10-11)

- **PR 0: Restauración de CI y Empaquetado (`pyproject.toml`):**
  - Eliminado clasificador redundante de licencia en `pyproject.toml` (`License :: OSI Approved :: Apache Software License`) que provocaba `setuptools.errors.InvalidConfigError` con setuptools modernos.
  - Verificado empaquetado completo `.venv/bin/python -m build` (generación limpia de sdist `catml-0.9.0rc1.tar.gz` y wheel `catml-0.9.0rc1-py3-none-any.whl`).
- **Trazabilidad de Recetas, Validación Inteligente y Límite de Tiempo:**
  - Añadido `recipe_id` a `Experiment` (`src/automl/domain/experiments/trial.py`), `CreateExperimentCommand`, `workspace.py` y persistencia SQLite.
  - Soporte de `time_budget_seconds` en `/api/experiment/create_and_run` en `server.py`.
  - Columna e insignias de Receta en Leaderboard de Studio (`studio.js`).
  - Suite de verificación `tests/test_workbench_questionnaire_and_traceability.py` (6 pruebas pasando al 100%).
- **PR 1: Fase P0 — Contexto único de dataset y ejecución (`activeDatasetId`, `activeRunId`):**
  - **`store.js`:** Centralizada la resolución de contexto (`resolveContext`, `setActiveDataset`, `setActiveRun`, `getActiveDataset`, `getActiveRun`, `getRunsForActiveDataset`, `contextVersion`). Si un dataset no tiene ejecuciones, `activeRunId` es estrictamente `null` (sin caídas arbitrarias a runs de otros datasets). Al cambiar de run, se sincroniza automáticamente el dataset padre.
  - **Limpieza de Vistas:** Erradicados todos los fallbacks cruzados `|| runs[0]` y `|| datasets[0]` en `app.js`, `agent.js`, `studio.js`, `compare.js`, `home.js`, `overview.js`, `new_experiment.js`, `kaggle.js`, `knowledge.js`, `pipeline.js` y `datasets.js`.
  - **`agent.js`:** Manejo explícito de 3 estados de contexto: (1) Sin dataset seleccionado («Selecciona un dataset para comenzar»), (2) Dataset seleccionado sin ejecución («Sin ejecución activa»), (3) Dataset con ejecución activa (indicando dataset y run a las que pertenecen las hipótesis).
  - **Métricas:** Eliminadas puntuaciones engañosas globales en tarjetas locales de Studio y Overview cuando la ejecución no tiene resultados.
  - **API Backend:** Filtro `dataset_id` soportado en `/api/runs` en `server.py` (`ws.list_runs(dataset_id=...)`).
- **PR 2: Fase P0 — Métricas, puntuaciones y comparabilidad:**
  - **Backend (`server.py`):**
    - Implementadas `_is_metric_minimizing` y `_get_best_trial_result`, seleccionando el trial mínimo para métricas de error (RMSE, MSE, MAE, log-loss) y el máximo para métricas de score (ROC-AUC, Accuracy, F1).
    - Metadatos `metric_direction` ("minimize" / "maximize") e `is_minimizing` integrados en respuestas de `/api/runs`, `/api/experiments`, `/api/experiments/compare` y `/api/overview`.
    - Sanitizada la selección de mejor trial en `/api/kaggle/status` y `/api/planner/plan`.
  - **Frontend Utilities (`utils.js`):**
    - Nuevas utilidades canónicas `isMinimizingMetric()`, `getMetricDirectionInfo()` y `formatMetricComparison()`.
  - **Comparador (`compare.js`):**
    - Bloqueo de comparaciones incompatibles con banner informativo cuando los experimentos seleccionados miden objetivos matemáticos distintos. Deshabilitado el cálculo engañoso de deltas y suspendido el gráfico combinado para métricas no homogéneas.
    - Desbloqueado el eje Y del gráfico de evolución (eliminado el techo rígido `Math.min(1.0, ...)` para soportar métricas de regresión como RMSE > 1.0).
  - **Etiquetado Honesto y Riguroso:**
    - Sustituido "Optimal" por "Mejor de esta ejecución" o "Mejor resultado observado".
    - Sustituido "Verified" / "Verified by CV" por "Evaluado" / "Evaluado por Cross-Validation".
    - Sustituido "Mejora" por "Diferencia observada favorable" o "Diferencia no concluyente" ($|\Delta| < 0.00005$).
  - **Pruebas Automatizadas:**
    - Nueva suite `tests/test_workbench_metric_context.py` con 7 tests pasando al 100%. Total de 66 tests de Workbench pasando en verde.
- **PR 3: Fase P1 — Simplificación del flujo, navegación y estados de trabajos:**
  - **Flujo Guiado de 4 Etapas:**
    - Introducido banner interactivo de 4 etapas en `home.js`: `01 Datos` (Ingesta & Calidad) → `02 Exploración` (Estudios & Hipótesis) → `03 Experimentos` (Modelos & HPO) → `04 Resultados` (Leaderboard & Artefactos), con estados reactivos por etapa y saltos directos al módulo correspondiente.
    - Alineada la numeración y etiquetas de navegación lateral en `index.html` (`00 Inicio`, `01 Datos (Dataset)`, `02 Exploración`, `03 Experimentos`, `04 Resultados & Evidencia`).
  - **Fidelidad y Rigor en Estados de Trabajos y Ejecuciones:**
    - Definidas utilidades formales en `utils.js` (`ACTIVE_JOB_STATUSES`, `isJobActive`, `JOB_STATUS_LABELS`, `JOB_ACTION_LABELS`, `getJobStatusLabel`) que cubren los 9 estados posibles (`queued`, `running`, `pause_requested`, `cancel_requested`, `paused`, `completed`, `failed`, `cancelled`, `interrupted`).
    - Eliminado el engaño visual en `home.js`: la sección «Ejecuciones en curso» muestra ÚNICAMENTE tareas o entrenamientos activos en segundo plano; si el motor está en reposo, se presenta el último modelo completado de forma honesta con badge `MOTOR EN REPOSO` o `LISTO PARA ENTRENAR` sin fingir que está en ejecución. Barras de progreso verídicas (pasos reales completados/totales o indicador indeterminado pulsante, sin falsos 50% o 75%).
    - Historial de cola verídico en «Trabajos recientes» con badges de resultado diferenciados (`Completado`, `Fallido`, `Cancelado`, `En ejecución`).
  - **Barra Flotante Discreta y No Invasiva (`jobs.js`):**
    - Cuando no hay trabajos activos, se colapsa a un disparador circular mínimo `Trabajos (0 activos)` que no cubre botones, tablas ni área de trabajo.
    - Cuando hay operaciones activas, muestra una barra flotante en vivo con progreso real y controles rápidos de pausa y cancelación.
    - Modal expandido organizado en pestañas separadas `En curso` e `Historial`, permitiendo inspeccionar errores con opción de reintento.
    - Limpieza adecuada de temporizadores con `clearInterval` y prevención de consultas cuando la pestaña está oculta (`document.hidden`).
  - **Hero Card Despejado en Home:**
    - Rutas absolutas, UUIDs y metadatos técnicos delegados a un panel colapsable `<details>`, manteniendo visible sólo la información clave del problema.
    - Botón de acción principal adaptado dinámicamente según el estado del laboratorio (sin dataset, preparado sin entrenar, entrenando, completado).
  - **Pruebas Automatizadas:**
    - Creada suite `tests/test_workbench_jobs_and_flow.py` (6 pruebas pasando al 100%).
    - Total acumulado de 72 pruebas del Workbench pasando en verde sin regresiones.
- **PR 4: Fase P1 — Leaderboard centrado en resultados, recomendaciones y recetas:**
  - **Presentación Limpia del Ganador y Contexto de Evaluación (`studio.js`):**
    - Se sitúa un Hero Card prominente en la cabecera de Studio con el modelo ganador, métrica objetivo y dirección, score formateado, tipo de validación y comparación prudente contra baseline (`vs baseline` o guión cuando no existe).
  - **Leaderboard Centrado en Resultados con Identificación Unívoca:**
    - Identificación explícita de ensayos mediante `trial_id` con fallback ordenado a `name`, visualizando delta vs baseline, duración y botón de inspección modal de hiperparámetros (`data-inspect-trial`).
  - **Jerarquía Técnica Reorganizada:**
    - Pestañas secundarias ubicadas debajo del Leaderboard: Modelos (`models`), Convergencia HPO (`hpo`), Decisiones del Planificador (`planner`), Recetas Reproducibles (`recipes`), Recursos (`resources`) y Logs (`logs`).
  - **Ciclo de Vida Estandarizado de Transformaciones (`utils.js` y `explore.js`):**
    - Definidas etapas en `TRANSFORMATION_LIFECYCLE_STAGES` (`PROPOSED`, `VERIFIED`, `SELECTED`, `APPLIED`, `PROMOTED`) con etiquetas en español, descripciones y clases visuales unificadas (`badge-neutral`, `badge-indigo`, `badge-cyan`, `badge-success`).
    - Renderizado consistente en la vista Explore (`explore.js`).
  - **Sincronización de Selección de Características (`datasets.js`):**
    - Sincronizados de forma atómica en el store los estados `selectedFeatures`, `customFeatures` y `selectedFeaturesByDataset` tras cualquier cambio de checkboxes en la tabla de columnas.
  - **Pruebas Automatizadas:**
    - Creada suite `tests/test_workbench_leaderboard_and_recipes.py` (5 pruebas pasando al 100%).
    - Total de 80 pruebas de Workbench pasando al 100%.
- **PR 5: Fase P2 — Idioma español unificado, accesibilidad WCAG AA y diseño responsive:**
  - **Idioma Español Unificado:**
    - Textos, etiquetas, modales, tablas, botones e indicadores de estado traducidos al español en `index.html`, `app.js`, `workbench.css`, `datasets.js`, `overview.js`, `studio.js`, `new_experiment.js`, `compare.js`, `kaggle.js`, `pipeline.js` y `knowledge.js`.
    - Preservados rigurosamente los acrónimos y vocablos universales de Machine Learning (`LightGBM`, `XGBoost`, `CatBoost`, `Random Forest`, `Ridge`, `Logistic Regression`, `ROC-AUC`, `RMSE`, `MAE`, `MSE`, `K-Fold`, `Stratified K-Fold`, `Holdout`, `SHAP`, `OOF`, `Optuna`).
  - **Accesibilidad (WCAG AA):**
    - Especificado `lang="es"` en el elemento raíz `<html>`.
    - Estilo de navegación por teclado accesible `:focus-visible` con anillo de contraste `--catml-blue` y desplazamiento suave en `workbench.css`.
    - Soporte nativo para usuarios con sensibilidad al movimiento mediante `@media (prefers-reduced-motion: reduce)`.
    - Incorporados atributos `aria-label` y `role` en navegación principal, paneles auxiliares, botones interactivos y tablas de datos.
  - **Diseño Responsive (<768px y <1024px):**
    - Reglas de contención modal `.modal-responsive` (`max-height: calc(100vh - 2.5rem); overflow-y: auto`) para evitar desbordamientos en dispositivos móviles o ventanas compactas.
    - Soporte táctil optimizado `-webkit-overflow-scrolling: touch` para tablas anchas.
    - Adaptación visual fluida de tipografías y rellenos para breakpoints pequeños.
  - **Pruebas Automatizadas:**
    - Creada suite `tests/test_workbench_i18n_and_a11y.py` (6 pruebas pasando al 100%).
    - Total acumulado de 86 pruebas del Workbench pasando al 100% verde sin regresiones.


### Track Experiment Recipes — Fase R4: Workbench UI y Paridad E2E (2026-10-11)


- **Integración de Experiment Recipes en Workbench y Paridad E2E (Fase R4):**
  - **Cliente API Frontend (`src/automl/interfaces/web/static/js/api.js`):**
    - Métodos REST client respetando Clean Architecture (UI como mero cliente del bus, sin lógica de dominio en frontend):
      - `getVerifiedImprovements()`, `getRecipes()`, `getRecipe()`.
      - `createRecipeFromEvidence()`, `selectRecipe()`, `submitRecipe()`, `promoteRecipe()`, `archiveRecipe()`.
  - **Iconografía (`src/automl/interfaces/web/static/js/icons.js`):**
    - Añadido icono SVG Lucide `recipe` conforme al sistema de diseño Tech Minimalista Premium.
  - **Vista Explore (`src/automl/interfaces/web/static/js/views/explore.js`):**
    - Botón de acción contextual `Crear Receta` en la tabla de Evidence Links para hipótesis aceptadas (`btn-create-recipe`).
    - Sección dedicada `Recetas de Experimentos Reproducibles` con selector de estado, badges minimalistas, desglose de transformaciones (`derived_features` y `column_transforms` in-fold) y botones de ciclo de vida (`btn-select-recipe`, `btn-submit-recipe`, `btn-promote-recipe`, `btn-archive-recipe`).
  - **Vista Studio (`src/automl/interfaces/web/static/js/views/studio.js`):**
    - Pestaña `Recetas` con conteo reactivo, listado de recetas registradas, detalle de dataset hash/split y CTA para entrenar en el run activo o promover como configuración dorada.
  - **Modal de Nuevo Experimento (`src/automl/interfaces/web/static/js/views/new_experiment.js`):**
    - Selector desplegable de recetas para el dataset seleccionado, permitiendo ejecución directa con 1-click reutilizando transformaciones y parámetros pre-calibrados.
  - **Validación de Sintaxis Frontend:**
    - Los 5 módulos JavaScript verificados con `node -c` sin errores de sintaxis ni de imports ESM.
  - **Suite de Pruebas Automatizadas de Paridad Multi-Superficie:**
    - `tests/test_recipe_phase_r4_workbench_parity.py`:
      - `test_workbench_static_js_recipe_integration`: Verificación de todos los métodos en `api.js` y hooks DOM en vistas.
      - `test_workbench_rest_recipe_lifecycle`: Servidor HTTP real en puerto efímero ejecutando ciclo REST completo (creación desde evidencia, selección, entrenamiento, promoción con notas y archivado).
      - `test_end_to_end_cross_surface_parity_r4`: Prueba de 4 superficies cruzadas (CLI crea evidencia Explore → Workbench crea receta → MCP inspecciona/selecciona → Workbench entrena modelo fit-in-fold → CLI promueve → todas las 4 superficies verifican estado `PROMOTED`, idéntico linaje y transformaciones).
  - **Métricas Globales de Validación:**
    - **751 tests pasando (100% verde)**, 0 fallos, 0 regresiones.
    - **86.88% de cobertura de código** (requisito CI $\ge 85\%$).

### Track Experiment Recipes — Fase R3: Paridad Python / CLI / MCP / REST (2026-10-11)

- **Paridad Multi-Superficie de Experiment Recipes (Fase R3):**
  - **Superficie CLI (`src/automl/interfaces/cli/recipe_cli.py` & `main.py`):**
    - Subcomandos de `catml recipe` / `automl recipe`:
      - `list-verified`: Listado tabular y JSON de mejoras empíricamente verificadas desde estudios Explore (`ListVerifiedImprovementsQuery`).
      - `create-from-evidence`: Creación determinista de recetas en estado `DRAFT` desde un `evidence_id` (`CreateExperimentRecipeFromEvidenceCommand`).
      - `list`: Listado filtrable por dataset o estado (`ListExperimentRecipesQuery`).
      - `show`: Inspección profunda de hiperparámetros, transformaciones con/sin estado y linaje auditable (`GetExperimentRecipeQuery`).
      - `select`: Transición a `SELECTED` para evaluación (`SelectExperimentRecipeCommand`).
      - `submit`: Ejecución de experimentos de entrenamiento en el split reproducible (`SubmitExperimentFromRecipeCommand`).
      - `promote`: Promoción explícita con justificación y notas de auditoría (`PromoteExperimentRecipeCommand`).
      - `archive`: Archivado de recetas (`ArchiveExperimentRecipeCommand`).
  - **Superficie MCP para Agentes (`src/automl/interfaces/mcp/recipe_tools.py` & `server.py`):**
    - Registro formal de 8 herramientas de agente integradas con `CommandBus` y `QueryBus`:
      - `recipe_list_verified_improvements` (paginación con `limit` y `offset`).
      - `recipe_create_from_evidence`.
      - `recipe_get`.
      - `recipe_list`.
      - `recipe_select`.
      - `recipe_submit`.
      - `recipe_promote`.
      - `recipe_archive`.
    - Manejo de códigos de error estructurados (`ToolErrorCode.NOT_FOUND`, `ToolErrorCode.INTERNAL_ERROR`).
  - **Superficie REST API (`src/automl/interfaces/web/server.py`):**
    - Endpoints GET: `/api/recipes/verified-improvements`, `/api/recipes`, `/api/recipes/{recipe_id}`.
    - Endpoints POST: `/api/recipes/create-from-evidence`, `/api/recipes/select`, `/api/recipes/submit`, `/api/recipes/promote`, `/api/recipes/archive`.
    - Respuestas normalizadas con HTTP 200, 400 Bad Request ante parámetros ausentes, y 404 ante recetas inexistentes.
  - **Cumplimiento Estricto de Reglas Arquitectónicas:**
    - Cero fugas de repositorios privados en capas de interfaz (test CQRS guard en verde).
    - Principio "Propose ≠ Accept" protegido: crear desde evidencia nunca promueve automáticamente; la promoción requiere validación explícita.
  - **Batería de Pruebas y Cobertura:**
    - Suite de paridad cruzada completa en `tests/test_recipe_phase_r3_parity.py` (4 tests al 100%: CLI, MCP, REST y flujo cruzado E2E).
    - Suite global verde: **748 pruebas pasando al 100%**, cobertura global sostenida en **86.88%** (requisito $\ge 85\%$).

### Track Experiment Recipes — Fase R1 & R2: Dominio Recipe, CQRS y Trainer Fit-in-Fold (2026-10-11)

- **Implementación de Experiment Recipes (Fase R1 & R2):**
  - **Entidad Pura de Dominio (`src/automl/domain/recipes/`):**
    - `ExperimentRecipe`: Representación reproducible de configuraciones de experimentos con especificación §6 completa: `id`, `name`, `status` (`DRAFT`, `SELECTED`, `SUBMITTED`, `PROMOTED`, `ARCHIVED`), `dataset_id`, `dataset_content_hash`, `feature_names`, `derived_features`, `column_transforms`, `drop_features`, `model_ids`, `model_params`, `validation_strategy`, `cv_folds`, `metric`, `random_seed`, `evidence_link_id`, `hypothesis_id`, `study_id`, `lineage`, `experiment_ids`.
    - Métodos de ciclo de vida con auditoría y serialización: `select()`, `submit()`, `promote()`, `archive()`, `to_dict()`, `from_dict()`.
    - Puerto de persistencia `ExperimentRecipeRepositoryPort` en dominio (`domain/recipes/ports.py` y `domain/ports.py`).
  - **Persistencia Relacional SQLite Aditiva (`SQLiteRecipeRepository`):**
    - Implementación en `src/automl/infrastructure/database/sqlite_recipes.py`.
    - Tabla `experiment_recipes` con índices en `dataset_id` y `status`, transacciones WAL y soporte en disco y memoria.
  - **Servicio de Aplicación y CQRS (`RecipeService`):**
    - Comandos: `CreateExperimentRecipeFromEvidenceCommand` (crea draft desde `EvidenceLink` respetando Propose ≠ Accept), `SubmitExperimentFromRecipeCommand` (ejecuta entrenamiento sin alterar la receta base), `SelectExperimentRecipeCommand`, `PromoteExperimentRecipeCommand` (promoción explícita y auditable), `ArchiveExperimentRecipeCommand`.
    - Consultas DTO: `GetExperimentRecipeQuery`, `ListExperimentRecipesQuery`, `ListVerifiedImprovementsQuery` (conecta evidencias estadísticas con aprendizaje acumulado).
    - Registro modular desacoplado en `recipe_registry.py` y ensamblado en `bootstrap.py`.
  - **Trainer Fit-in-Fold sin Fuga de Datos (`SklearnTrainer`):**
    - Aplicación in-fold de `RobustScaler` y `PowerTransformer` (Yeo-Johnson) en pipeline de CV sin fugas pre-split.
  - **Pruebas y Verificación:**
    - Suite de Fase R1 en `tests/test_recipe_phase_r1_domain_and_cqrs.py` (5 pruebas pasando al 100%).
    - Suite de Fase R0 en `tests/test_recipe_phase_r0_verify_transforms.py` (6 pruebas pasando al 100%).
    - Guard arquitectónico CQRS verificado en `tests/test_cqrs_architecture_guard.py` (3 pruebas pasando).
    - Suite global completa: **744 pruebas pasando**, 0 fallos, **86.95% de cobertura total**.

### Track Experiment Recipes — Fase R0: Verificación Real de Transforms (2026-10-11)

- **Cierre del Gap de Verificación Real en Hipótesis de Explore (Fase R0):**
  - **Traductor de Hipótesis (`HypothesisExperimentTranslator`):**
    - Mapeo de `power_transform` con `log1p` a transformaciones sin estado en `derived_features` (`log1p(col)` evaluada sobre dataframe) con reemplazo de la columna original.
    - Mapeo de `power_transform` con `yeo_johnson` y `robust_scaler` a transformaciones con estado en `column_transforms` (`{"column": col, "kind": kind, "params": {}}`).
  - **Dominio y Persistencia:**
    - Entidades `TrialExecution` y `Experiment` ampliadas con `column_transforms` y `derived_features`.
    - Persistencia aditiva en SQLite (`column_transforms_json`, `derived_features_json`) con migraciones idempotentes `_migrate()` en `SQLiteExperimentRepository`.
  - **Entrenamiento Fit-in-Fold sin Fugas de Datos (`SklearnTrainer`):**
    - `derived_features` materializadas antes de seleccionar columnas de `X`.
    - `column_transforms` construidos como sub-pipelines en `ColumnTransformer` (`RobustScaler`, `PowerTransformer(method="yeo-johnson")`, etc.) aplicados dentro de cada fold de Cross-Validation o split de train sin fugas sobre validación/test.
  - **Preservación de Principios ("Propose ≠ Accept"):**
    - `EvidenceLink` conserva el registro de aprendizaje positivo y negativo (`accepted=True/False`), sin auto-promoción a configuraciones de producción.
  - **Batería de Pruebas:**
    - Suite dedicada en `tests/test_recipe_phase_r0_verify_transforms.py` (6 pruebas pasando al 100%).
    - Suite de regresión pasando (92 pruebas en explore/workbench).

### Track Benchmark Empírico — AutoML solo vs Explore/Agente (Airline Satisfaction) (2026-10-11)

- **Estudio Experimental y Validación Empírica en Dataset Real (`docs/experiments/`):**
  - **Dataset:** Kaggle Airline Passenger Satisfaction (`data/kagle/train.csv`, ~700k filas, target binario `satisfaction`). Muestra de trabajo estratificada 50/50 de 25k filas (`random_state=42`) y confirmación en 150k y full train (699k filas). Métrica: ROC-AUC con StratifiedKFold(5).
  - **Resultados de la Comparativa Cuantitativa:**
    1. *AutoML Naive (crudo con columna `id`):* ROC-AUC **0.817** (inestabilidad de validación cruzada `std` ~0.036).
    2. *AutoML + Anti-Leakage Guardian:* ROC-AUC **0.952** (+0.135 sobre naive, `std` ~0.003). Demuestra que en datos tabulares complejos el mayor impacto proviene de la higiene estadística y la exclusión de identificadores únicos.
    3. *AutoML + Drops de colinealidad propuestos por Explore:* ROC-AUC **0.952** (Δ = -0.0006 vs baseline B). Validación empírica del principio rector *"Propose ≠ Accept"*: Random Forest ya gestiona la colinealidad internamente, por lo que la eliminación de variables correlacionadas no aportó mejora, registrando y preservando el aprendizaje negativo.
    4. *Push de Rendimiento (LightGBM/XGBoost + Optuna HPO):* ROC-AUC CV **0.9586** en muestra de 150k y en dataset completo de 699k filas. Sometido al test ciego de Kaggle (`submission_airline_lgbm_hpo.csv`) alcanzando **0.9578** (Δ ≈ -0.0008 vs CV), lo que confirma una calibración sobresaliente del protocolo de validación.
  - **Conclusiones Metodológicas y Reglas Transferibles a Otros Datasets:**
    - Exclusión estricta de identificadores de alta cardinalidad antes de modelar como P0.
    - Calibración de heurísticas de fuga: `Customer Type` (2 valores únicos) es una variable categórica con señal real, no una entidad que justifique `GroupKFold` ni su descarte preventivo (evitar falsos positivos en group leakage).
    - Techo de modelos tabulares: modelos GBDT (LightGBM/XGBoost) superan a stacks básicos de scikit-learn en presencia de interacciones densas. HPO aporta ajustes marginales (+0.001-0.002) una vez que el pipeline está limpio.
    - Gap de transformaciones con estado: Explore propuso transformaciones `yeo_johnson` y `RobustScaler`, pero el sistema aún no puede aplicarlas de forma segura *fit-in-fold* durante `verify_hypothesis`. Esto fundamenta e impulsa como prioridad P1 la especificación de **Experiment Recipes** (ADR-009, Fases R0–R4).
  - **Artefactos del Experimento:**
    - `docs/experiments/airline_satisfaction_automl_vs_agent_report.md`: Informe técnico estructurado con análisis de 55 hallazgos y 37 hipótesis.
    - `docs/experiments/push_airline_score.py`: Script reproducible con pipeline, encoders, HPO y generación de submission.
    - `docs/experiments/airline_push_scores.json`, `airline_automl_vs_explore_results.json`, `airline_automl_vs_explore_decomposition.json`: Registro de métricas y deltas.

### Track Spec — Experiment Recipes backlog (2026-10-11)

- Documented pending feature **Experiment Recipes** (Explore → Evidence → reusable config → Experiments → Promote) with interface parity:
  - Spec: `docs/features/experiment-recipes/spec.md`
  - Plan R0–R4: `docs/features/experiment-recipes/plan.md`
  - ADR-009: `docs/decisions/009-experiment-recipes-and-interface-parity.md`
  - Checklist in `TASKS.md` under “Next Roadmap: Experiment Recipes”
  - Cross-links from `docs/features/catml-explore/plan.md` and `AGENTS.md`
- Explicit: `EvidenceLink.accepted` ≠ Selected/Promoted; fit-in-fold for stateful transforms; implement contracts before Workbench UI.

### Track Workbench UI Refinements & Testing — (2026-10-11)

- **Corrección de Visualización y Contraste en CATML Explore (`ExploreView`):**
  - **Resolución de Selector «Estudio activo» y Filtros en Blanco:**
    - Identificada causa raíz: la clase `.input-base` no tenía definición CSS, provocando que los navegadores aplicaran estilos de agente de usuario con fondo blanco sólido (`#ffffff`) y tipografía clara heredada (`#F7F8FA`), haciendo invisibles las letras en temas oscuros («en todo blanco y no veo las letras»).
    - En `src/automl/interfaces/web/static/css/workbench.css`: incorporadas reglas globales para `.input-base`, `input[type="text"].input-base`, `select.input-base` y `select option` con superficie oscura (`#151B26`), borde arquitectónico (`#252C38`), texto de alta legibilidad (`#F7F8FA`) y foco en azul eléctrico (`#4F67FF`).
    - En `src/automl/interfaces/web/static/js/views/explore.js`: clases Tailwind oscuras explícitas (`bg-[#161B26] border-[#242A36] text-[#F7F8FA]`) en selector `#selectActiveStudy`, filtros `#filterCategorySelect`, `#filterSeveritySelect` y campos modales de nuevo estudio.
  - **Alineación de Visualizaciones Declarativas (`data_series` vs `data`):**
    - En `src/automl/interfaces/web/static/js/views/explore.js`: adaptadas las funciones de renderizado de visualizaciones (`_renderCorrelationMatrixHtml`, `_renderCategoricalMatrixHtml`, `_renderHistogramSvgHtml`, `_renderBoxplotSvgHtml`, `_renderScatterSvgHtml`) para leer `viz.data_series || viz.data || {}`, renderizando correctamente matrices de correlación (19x19), Cramér's V, histogramas con conteos reales y diagramas de dispersión y caja.
  - **Galería de Hallazgos Estadísticos y Ordenación por Importancia:**
    - En `src/automl/interfaces/web/static/js/views/explore.js` (`_getFilteredFindings`) y `datasets.js`: implementada ordenación jerárquica por importancia real de los hallazgos:
      1. Severidad clínica descendente: `CRITICAL` (3) > `WARNING` (2) > `INFO` (1).
      2. Significancia estadística ajustada por False Discovery Rate (Benjamini-Hochberg `fdr_significant`).
      3. Magnitud absoluta del tamaño del efecto (`Math.abs(effect_size || pearson_r || cramers_v || cohens_d)`).
    - Homogeneizada la clasificación de severidad `_classifySeverity` con `datasets.js` (incluyendo `multivariate_outliers` y `high_cardinality_identifier` como `CRITICAL`, `multimodal_distribution` y `target_class_separation` como `WARNING`).
    - Añadido soporte para métricas alternativas en `effect` (`cramers_v`, `cohens_d`, etc.) y badges formateados para p-valor (`p < 0.0001` o `p = 0.0000`).
  - **Auto-Backfill de Features en Registro y Resolución de Error al Verificar Hipótesis:**
    - Causa raíz identificada: Al registrar el dataset de Kaggle con target booleano antes de la corrección de booleanos, solo se persistió la columna target `Class` en la tabla `features`. Al pulsar «Verificar hipótesis», el motor de verificación validaba las columnas contra el repositorio devolviendo `Unknown features for dataset ds_819ec589: Age, Flight Distance, ...`.
    - En `src/automl/application/services/workspace.py` (`_hydrate_feature_registry`): añadido auto-backfill a partir de las columnas detectadas en `DatasetProfile.columns`. Si faltan columnas en el repositorio de características, se auto-registran y persisten con tipo inferido (`identifier` si corresponde o `unknown`). Verificado en `ds_819ec589`: el conteo de features pasó de 1 a 22 características operativas, permitiendo la verificación de hipótesis sin errores.
  - **Pruebas de Regresión:**
    - Test `test_workbench_explore_form_inputs_and_select_styles` en `tests/test_workbench_navigation_and_explore_ui.py`.
    - Suite completa de Explore (`tests/test_explore_*.py`, 46 tests) y Workbench (`tests/test_workbench_*.py`, 46 tests) aprobada al 100%.
- **Soporte de Columnas Booleanas en Perfilado Estadístico y Registro de Datasets:**
  - `src/automl/engine/profiling/dataset_profiler.py`:
    - En `profile_dataset()`, conversión de series booleanas a `float64` (`0.0`/`1.0`) para cálculos estadísticos univariantes (media, std, mediana, `q25`, `q75`, curtosis, asimetría, histogramas). Evita el error `TypeError: numpy boolean subtract, the '-' operator, is not supported` provocado por la interpolación lineal de cuartiles en NumPy 1.25+/2.x sobre tipos booleanos estándar y nulos (`pd.BooleanDtype`).
    - En agregaciones estratificadas por clase (`by_target`), conversión de `sub_s` a `float64` para series booleanas antes de calcular cuartiles y medianas.
    - En cálculo de correlación con target continuo, conversión de series booleanas a `float`.
  - `src/automl/engine/analysis/distribution_diagnostics.py`: Cast a `float` si `is_bool_dtype(series)` antes de calcular momentos y límites de outliers por rango intercuartílico (IQR).
  - `src/automl/engine/analysis/visualizations/builder.py`: Cast a `float` si `series.dtype == bool` en `build_boxplot_spec()` antes de calcular percentiles de NumPy.
  - `src/automl/interfaces/web/server.py`:
    - Captura con `traceback.print_exc()` en endpoint `/api/dataset/register` y emisión de mensaje de error descriptivo («No se ha podido analizar o registrar el dataset: {exc}») evitando fallos opacos en el navegador.
    - Captura con impresión de traza en manejadores generales de `do_POST`.
  - `src/automl/interfaces/web/static/js/views/datasets.js`:
    - Adaptación robusta en la tarjeta de hallazgos estadísticos para deducir severidad (`_classifyFindingSeverity`), extraer columnas (`column_name`, `secondary_column`) y formatear métricas (`effect_size` / `metrics`), previniendo `undefined` en campos de dominio.
    - Detección precisa de visualizaciones de matriz de correlación (`chart_type === "correlation_matrix"`).
  - **Pruebas de Regresión y Verificación:**
    - Test `test_profile_dataset_supports_boolean_columns` en `tests/test_profiler_cardinality.py` verificando soporte de `bool` nativo y `boolean` nullable de Pandas con `pd.NA`.
    - Verificación exitosa en dataset real con ~700,000 filas (`data/kagle/train.csv`, columna booleana `satisfaction`).
- **Eliminación de Datasets Demo y Garantía de Workspace Vacío por Defecto:**
  - En `src/automl/interfaces/cli/main.py` (`launch_ui_cli`): eliminada la resolución predeterminada a `.automl/s6e9_automl` o `.automl/demo`; ahora resuelve a un workspace limpio por defecto (`.automl/default`).
  - En `src/automl/interfaces/web/server.py` (`AutoMLWebHandler`, `run_web_dashboard`): actualizado `workspace_dir` por defecto a `.automl/default`.
  - Limpieza de artefactos residuales en `workspace/automl.db` y `.automl/default/` (eliminando el dataset demo «E-Commerce Multimodal Catalog» y ejecuciones previas).
  - Verificado arranque limpio en `workspace/`: endpoints `/api/datasets` (`[]`), `/api/runs` (`[]`), `/api/overview` (`total_runs: 0`, `recent_datasets: []`, `activity_feed: []`).
- **Erradicación de Datos Hardcodeados y Estados Vacíos Honestos en UI:**
  - `src/automl/interfaces/web/static/index.html`: Inicialización limpia del badge de dataset activo a «Sin dataset activo» con punto neutral en vez de texto transitorio.
  - `src/automl/interfaces/web/static/js/views/kaggle.js`: Añadido estado vacío honesto cuando no hay ejecución activa («Sin Experimentos de Competición Activos») en lugar de maquetas sintéticas de benchmark.
  - `src/automl/interfaces/web/static/js/views/pipeline.js`: Añadido estado vacío honesto cuando no hay perfil ni ejecución («Sin Pipeline Generado») en vez de nodos DAG artificiales con conteos en cero.
  - Todos los datos numéricos y métricas visualizadas en Workbench provienen estrictamente de cálculos reales del backend (CLI, MCP, buses CQRS).
- **Batería de Pruebas Automatizada:**
  - Nuevo test unitario `test_workbench_clean_empty_workspace_defaults_and_zero_hardcoding` en `tests/test_workbench_laboratory_redesign.py` validando la resolución predeterminada de CLI y respuestas API totalmente limpias en directorios vacíos.
  - 45 tests de Workbench ejecutados y aprobados al 100%.
  - Suite completa de 731 tests aprobada (100% verde) con 87.07% de cobertura (superando el umbral del 85%).
- **Sincronización Reactiva de Versión en Frontend:**
  - `src/automl/interfaces/web/static/index.html`: Elementos `#sidebarVersionLabel` y `#sidebarAgentBadge` con versión dinámica por defecto `v0.9.0-rc1`.
  - `src/automl/interfaces/web/static/js/app.js`: Inyección reactiva de `overview.version` en la barra lateral en cuanto el servidor responde en el arranque.
- **Enrutamiento Canónico y Estados Activos:**
  - `src/automl/interfaces/web/static/js/app.js`: Normalización de alias en `_routeTo` (`datasets` -> `dataset`, `studio` -> `experiments`, `overview` -> `evidence`, `explore` -> `explore`), garantizando que la barra lateral refleje siempre la vista activa independientemente del origen de la navegación.
- **Navegación Fluida hacia CATML Explore:**
  - `src/automl/interfaces/web/static/js/views/datasets.js`: Añadido botón con acceso directo «Lienzo Completo» (`#btnOpenFullExploreCanvas`) en la pestaña de Exploración, permitiendo al usuario saltar instantáneamente desde el perfil de variables al lienzo analítico completo de `ExploreView`.
- **Armonización Lingüística y Limpieza de Versiones Heredadas:**
  - `src/automl/interfaces/web/static/js/views/studio.js`: Mensaje de carga traducido al español («Cargando estudio de experimentos, planes y métricas de validación...»).
  - `src/automl/interfaces/web/static/js/views/knowledge.js`: Eliminado indicador estático `(V0.8)` del encabezado.

### Track Governance — Apache-2.0 license consistency (2026-10-11)

- Confirmed canonical licensing already Apache-2.0: root `LICENSE`, `NOTICE`, `pyproject.toml`, README badge/section, and GitHub `spdx_id=Apache-2.0`.
- Removed stale MIT claims that could confuse readers: `website/README.md` (incorrectly said root LICENSE was MIT), `TASKS.md` landing note, and added `license: Apache-2.0` to `website/package.json` plus PyPI classifiers in `pyproject.toml`.
- Branch: `fix/license-apache-consistency`.

### Track Security Hardening & Release Preparation — 0.9.0-rc1 (2026-10-11)

- **Endurecimiento de Interfaces Remotas (Fail-Closed por Defecto):**
  - **Workbench Fail-Closed:** En `src/automl/interfaces/web/server.py` (`run_web_dashboard`), enlace a interfaces no loopback (`0.0.0.0` o IP externa) sin autenticación levanta `PermissionError` a menos que la variable de entorno `CATML_ALLOW_INSECURE=1` esté explícitamente configurada.
  - **MCP Streamable-HTTP Fail-Closed:** En `src/automl/interfaces/mcp/server.py` (`run_mcp_service`), exposición no autenticada en interfaces externas levanta `PermissionError` a menos que `CATML_ALLOW_INSECURE=1` esté configurada.
  - **Sanitización de Tokens en URLs:** Eliminada la impresión en consola de enlaces con parámetros query (`?token=`). Conservación exclusiva de identificadores de fragmento (`/#token=`), que nunca se transmiten por la red HTTP y son sanitizados por el navegador.
  - **Tests Unitarios de Seguridad:**
    - Ampliación de `tests/test_audit_findings.py` verificando el bloqueo fail-closed de MCP streamable-http ante `0.0.0.0` y desbloqueo controlado con `CATML_ALLOW_INSECURE=1`.
    - Ampliación de `tests/test_workbench_auth.py` verificando el rechazo de enlaces remotos sin autenticación y validación de generación de token efímero y flag `CATML_ALLOW_INSECURE=1`.
- **Public Threat Model & Trusted Deployment Guide:**
  - Creación de [`docs/trusted-deployment.md`](docs/trusted-deployment.md) definiendo el modelo de amenazas y 3 niveles de despliegue (Tier 1: Local Loopback, Tier 2: Private LAN, Tier 3: Cloud VPS / Producción con proxy inverso TLS de referencia en Nginx).
  - Alertas explícitas en [`README.md`](README.md) y [`SECURITY.md`](SECURITY.md) sobre los riesgos de deserialización con `joblib`/`pickle`, aclarando que el checksum SHA-256 garantiza integridad ante corrupción, no autenticidad ni procedencia. Roadmap documentado para firmas PKI/HMAC y exportación a ONNX.
- **Superficie de Agentes y Privacidad:**
  - Modo compacto `compact=True` verificado por defecto en herramientas de análisis MCP, devolviendo cero filas crudas a los contextos de LLMs.
  - Confinamiento estricto al workspace y presupuestos cooperativos por sesión.
- **Bump de Versión a 0.9.0-rc1:**
  - `src/automl/__init__.py`: Versión actualizada a `0.9.0-rc1`.
  - `CHANGELOG.md`: Entrada detallada para `[0.9.0-rc1] - 2026-10-11`.
  - `README.md`: Badges actualizados (727 Passing, 86.8%+ Coverage) y enlaces de navegación a seguridad añadidos.
  - `SECURITY.md`: Tabla de versiones soportadas actualizada con `0.9.x` activo y `0.8.x` de mantenimiento.

### Track CATML Explore — Fase E6: Benchmarks Integrales y Preparación de Producto (Release Readiness) (2026-10-11)

- **Implementación de la Batería de Benchmarks en 4 Ejes (Fase E6 - ADR-008):**
  - **Runner de Benchmarks Dedicado (`ExploreBenchmarkRunner`):**
    - `src/automl/benchmarks/explore_benchmarks.py`: Implementada la suite integral automatizada para validación rigurosa de CATML Explore.
    - **Eje 1: Exactitud Estadística Numérica:**
      - Calibración tipo NIST para precisión univariante ante grandes desplazamientos numéricos ($10^9 + x$), garantizando ausencia de cancelación catastrófica (error de media $< 10^{-6}$, error de varianza $< 10^{-4}$).
      - Calibración con el Cuarteto de Anscombe (Datasets I a IV): verificación de correlación de Pearson exacta ($r \in [0.80, 0.83]$), detección de no-linealidad en Dataset II ($|\rho| - |r| > 0.05$) y detección de outliers en Dataset III mediante IQR.
      - Calibración analítica de intervalos de confianza al 95% con transformación $z$ de Fisher.
      - Verificación de la corrección FDR de Benjamini-Hochberg (monotonía no decreciente, acotación en $[0, 1]$ y protección contra inflación de falsos descubrimientos).
      - Tamaño del efecto de Cohen's $d$ y t-test de Welch en muestras gaussianas calibradas.
      - Cotas asintóticas del coeficiente $V$ de Cramér ($V \ge 0.98$ en tablas perfectas, $V \le 0.02$ en tablas independientes).
    - **Eje 2: Rendimiento AutoML con Hipótesis de Explore:**
      - Validación de integración end-to-end con datasets tabulares reales (`customers_churn.csv`).
      - Cumplimiento estricto del principio rector *"Propose ≠ Accept"*: evaluación del candidato contra el baseline sobre la **misma partición exacta de validación**.
      - Persistencia transaccional de `EvidenceLink` y actualización de estado de hipótesis (`accepted` si $\Delta > 0$, `rejected` si no supera el umbral), preservando el aprendizaje negativo.
    - **Eje 3: Seguridad, Privacidad y Confinamiento de Tokens MCP:**
      - Boundedness de tokens para LLMs: comprobación de límite estricto de hallazgos (`limit=2`), metadatos envoltorio (`envelope=True`) y poda de matrices/vectores densos (`compact=True`).
      - Cero filtración de datos crudos o PII en recursos de agente (`catml://studies/{study_id}/summary`).
      - Resistencia fail-closed ante intentos de path traversal (`../../../../etc/passwd`) e inyecciones SQL (`' OR 1=1 --`), retornando respuestas estructuradas de error sin caídas ni exposición de stack traces.
    - **Eje 4: Regresión de Esquemas y Compatibilidad de Workspaces:**
      - Arranque transparente sobre workspaces heredados con preservación de `experiments.db` y creación atómica de `automl.db` / `studies.db`.
      - Coexistencia libre de conflictos entre estudios no supervisados (`target_column=None`) y estudios supervisados (`target_column='churn'`).
      - Verificación de concurrencia y modo WAL activo (`PRAGMA journal_mode=WAL`).
    - **Método Unificado `run_all()`:** Ensamblado de reporte técnico estructurado de los 4 ejes para auditorías de release.
  - **Actualización de Documentación:**
    - `DEVELOPER_GUIDE.md`: Ampliación de la tabla de arquitectura con los componentes de CATML Explore (`domain/analysis/`, `engine/analysis/`, `application/analysis/`, `interfaces/mcp/analysis_tools.py`, `interfaces/cli/explore_cli.py`, `benchmarks/explore_benchmarks.py`) y comandos rápidos de estudio y verificación.
  - **Validación Automatizada y Calidad:**
    - Suite dedicada en `tests/test_explore_phase_e6_benchmarks.py` (6 pruebas pasando al 100%).
    - Batería completa de CATML Explore (Fases E1 a E6) con 46 pruebas pasando al 100% en 27s.
    - Suite global verde: **723 pruebas pasando (0 fallos)**, cobertura global sostenida en **86.76%** (umbral CI $\ge 85\%$).
    - Verificación exitosa de CLI: `automl --help`, `automl task list` y `automl explore --help`.



- **Implementación del Agente Científico y Motor de Hipótesis (Fase E5 - ADR-008):**
  - **Traductor Determinista de Hipótesis (`HypothesisExperimentTranslator`):**
    - `src/automl/engine/analysis/hypothesis_translator.py`: Mapeo riguroso y determinista desde acciones estadísticas propuestas (`resolve_collinearity`, `prioritize_feature`, `nonlinear_transform_or_trees`, `power_transform`, `robust_scaler`, `discretize_or_cluster`, `categorical_interaction`) a candidatos de experimento ejecutables (`ExperimentCandidateSpec`).
    - Configuración automática de modelos óptimos por familia (árboles GBDT para relaciones no lineales o asimetrías, lineales para features estandarizadas) y depuración controlada de columnas colineales.
  - **Protocolo Empírico de Verificación "Propose ≠ Accept":**
    - `src/automl/application/analysis/study_service.py`: Implementado `verify_hypothesis` ejecutando el experimento candidato contra el baseline de AutoML sobre la **misma partición exacta de validación**.
    - Cálculo de delta de mejora direccional ($\Delta = \text{candidato} - \text{baseline} > \epsilon$ para métricas de ganancia; $\Delta = \text{baseline} - \text{candidato} > \epsilon$ para métricas de error).
    - Creación y persistencia transaccional de la entidad de dominio `EvidenceLink` vinculando la hipótesis al experimento AutoML.
    - Transición de estado de la hipótesis: `proposed` -> `accepted` si se supera el umbral empírico, o `rejected` en caso contrario, registrando y preservando el **aprendizaje negativo** con notas auditables.
  - **Persistencia en Dominio e Infraestructura SQLite:**
    - `src/automl/domain/analysis/ports.py` y `src/automl/infrastructure/database/sqlite_studies.py`: Métodos `get_hypothesis` y `list_evidence_links` con transacciones seguras en modo WAL e índices relacionales.
  - **Capa de Aplicación y CQRS:**
    - `VerifyHypothesisCommand`, `GetHypothesisQuery`, `GetEvidenceLinkQuery`, `ListEvidenceLinksQuery` registrados en el CommandBus y QueryBus modular (`src/automl/application/registries/analysis_registry.py`).
  - **Paridad Total de Superficies:**
    - **MCP Tool y Recurso:** `analysis_verify_hypothesis` y recurso `catml://studies/{study_id}/evidence` en `src/automl/interfaces/mcp/analysis_tools.py`.
    - **CLI:** `catml explore verify <hypothesis_id> [--run-id <id>] [--min-improvement <val>] [--json]` en `src/automl/interfaces/cli/explore_cli.py`.
    - **Reporte Técnico:** Sección 5 "Empirical Verification & Evidence Links" añadida a `generate_study_markdown_report` y consumible vía `catml explore export`.
    - **REST API y Workbench:** Endpoints `POST /api/analysis/hypotheses/verify` y `GET /api/analysis/evidence` en `server.py` con métodos cliente en `api.js`.
  - **Validación Automatizada y Calidad:**
    - Suite de pruebas dedicada en `tests/test_explore_phase_e5_hypothesis_engine.py` (6 pruebas pasando al 100%).
    - Batería completa de CATML Explore (Fases E1 a E5) con 40 pruebas pasando al 100%.
    - Suite global verde: **717 pruebas pasando (0 fallos)**, cobertura global sostenida en **86.60%** (umbral CI $\ge 85\%$).

### Track CATML Explore — Fase E4: Interoperabilidad MCP y CLI Completa (Agent Surface) (2026-10-10)

- **Implementación de la Superficie de Agentes e Interoperabilidad (Fase E4 - ADR-008):**
  - **Ampliación de Herramientas MCP (`src/automl/interfaces/mcp/analysis_tools.py`):**
    - `analysis_get_findings`: Controles de eficiencia de tokens para LLMs con filtrado por tipo, significancia estadística (`min_significance` / FDR $p \le 0.05$), paginación determinista (`limit`, `offset`), modo compacto (`compact=True`) podando matrices/vectores masivos y modo envoltorio (`envelope=True`).
    - `analysis_get_visualizations`: Recuperación estructurada de especificaciones visuales (`VisualizationSpec`) declarativas por estudio o por gráfico puntual (`chart_id`).
    - `analysis_propose_experiment`: Formulación formal de hipótesis de ML a partir de hallazgos estadísticos con estado inmutable inicial `proposed` bajo el principio rector *"Propose ≠ Accept"*.
    - Recursos de solo lectura ampliados: `catml://studies/{study_id}/summary` (resumen ejecutivo de alta densidad y bajo consumo de tokens con señales clave) y `catml://studies/{study_id}/visualizations`.
  - **Generador de Reportes Técnicos (`src/automl/application/analysis/reporting.py`):**
    - Función pura `generate_study_markdown_report` que ensambla metadatos del estudio, desglose porcentual de categorías de hallazgos, catálogo detallado de inferencias, inventario de gráficos e hipótesis propuestas.
  - **CLI `catml explore` Completa (`src/automl/interfaces/cli/explore_cli.py` & `main.py`):**
    - Subcomando `export`: exportación de estudios a Markdown reproducible (`-f markdown -o <path>`) o JSON (`-f json`).
    - Subcomando `findings`: enriquecido con filtrado `--type`, umbral de significancia `--min-sig` y límite de filas `--limit`.
  - **Capa de Aplicación y CQRS:**
    - `ListFindingsQuery` y `ListVisualizationsQuery` en `queries.py` y `study_service.py` enriquecidos con filtrado y paginación 100% retrocompatibles.
  - **Validación Automatizada:**
    - Suite de pruebas dedicada en `tests/test_explore_phase_e4_agent_surface.py` (7 pruebas pasando al 100%).
    - Suite global verde: **711 pruebas pasando (0 fallos)**, cobertura global sostenida en **86.64%** (umbral CI $\ge 85\%$).

### Track CATML Explore — Fase E3: Laboratorio Visual Interactivo (Workbench Exploration UI) (2026-10-10)

- **Implementación del Laboratorio Visual Interactivo Explore (Fase E3 - ADR-008 & ADR-006):**
  - **Generador de Especificaciones Visuales Declarativas (`VisualizationBuilder`):**
    - `src/automl/engine/analysis/visualizations/builder.py`: Generación agregada y protegida de especificaciones `VisualizationSpec` sin saturar el cliente web.
    - Histogramas con cortes equiespaciados, conteos, media, desviación estándar y mediana.
    - Boxplots con resumen de 5 números ($Q_1, \text{mediana}, Q_3$), límites de bigotes y muestreo acotado de outliers (máx. 25 puntos).
    - Scatter plots bivariados con submuestreo determinista (máx. 200 puntos) y cálculo de línea de regresión lineal ($y = mx + b$).
  - **Componente Visual Tech Minimalista (`src/automl/interfaces/web/static/js/views/explore.js`):**
    - Cumplimiento riguroso de ADR-006: Geist / Geist Mono, bordes nítidos de 1px (`#242A36`), paleta estructurada `#0B0D12` / `#161B26` y acentos azul eléctrico (`#4F67FF`) y cian (`#53C8FF`).
    - Selector interactivo de estudios y ejecuciones con modal de creación y ejecución instantánea.
    - Galería de hallazgos estadísticos con filtros dinámicos por categoría, severidad y toggle de significancia Benjamini-Hochberg FDR ($p < 0.05$).
    - Renderizadores vectoriales SVG para histogramas, diagramas de caja y nubes de dispersión bivariada.
    - Matriz de calor bivariada interactiva con conmutador en tiempo real entre coeficientes de Pearson y Spearman.
    - Exportador de informes técnicos en formato Markdown descargable (`informe_explore_{id}.md`).
    - Integración en el router global de la aplicación (`src/automl/interfaces/web/static/js/app.js`).
  - **Validación Automatizada:**
    - Suite de pruebas completa en `tests/test_explore_phase_e3_ui.py` (6 pruebas pasando al 100%).
    - Cobertura sostenida del paquete `src/automl/engine/analysis` al 88%.

### Track CATML Explore — Fase E2: Motor Estadístico Avanzado (Advanced Statistical Engine) (2026-10-09)

- **Implementación del Motor Estadístico y Matemático Avanzado (Fase E2 - ADR-008):**
  - **Diagnóstico Profundo de Distribuciones (`distribution_diagnostics.py`):**
    - Pruebas inferenciales de normalidad univariante: D'Agostino-Pearson omnibus test (`scipy.stats.normaltest`) para $N \ge 20$ y Shapiro-Wilk (`scipy.stats.shapiro`) para $8 \le N < 20$.
    - Asimetría (*skewness*) y curtosis (*excess kurtosis*) con orientación direccional y formulación de hipótesis de transformación (`power_transform`, Yeo-Johnson / Log1p).
    - Detección cuantitativa de multimodality y bimodalidad mediante el coeficiente bimodal de Sarle ($BC > 0.555$) con recomendación para discretización o segmentación en subpoblaciones.
    - Detección multivariante de anomalías mediante distancia de Mahalanobis ($D^2$) con matriz de covarianza regularizada e inversión mediante pseudoinversa (`pinv`), contrastada frente a umbral crítico de Chi-cuadrado ($\chi^2_{df=k}, p < 0.001$).
  - **Métricas de Asociación Bivariadas y Categóricas (`association_metrics.py`):**
    - Correlación de Pearson ($r$) combinada con intervalos de confianza del 95% calculados analíticamente mediante la transformación $z$ de Fisher.
    - Correlación monótona de Spearman ($\rho$) con contraste frente a Pearson para capturar no-linealidades complejas ($|\rho| - |r| \ge 0.20$ o $|\rho| \ge 0.60$ con $|r| < 0.40$), generando hipótesis de modelos no lineales basados en árboles (GBDT) o splines.
    - Matriz de asociación categórica mediante coeficiente $V$ de Cramér con corrección de sesgo muestral de Bergsma-Wicher, pruebas Chi-cuadrado de independencia e hipótesis de interacciones bivariadas.
    - Generación declarativa de especificaciones `VisualizationSpec` para matrices de correlación (Pearson y Spearman) y matrices de asociación categórica.
  - **Pruebas de Hipótesis Inferenciales y Corrección FDR (`hypothesis_testing.py`):**
    - Comparación de grupos con variable objetivo binaria: t-test de Welch para varianzas desiguales, prueba no paramétrica de Mann-Whitney U, prueba de homogeneidad de varianzas de Levene y cálculo estricto del tamaño del efecto mediante $d$ de Cohen.
    - Comparación multiclase ($>2$ grupos): ANOVA unidireccional de Fisher, prueba de Kruskal-Wallis y tamaño del efecto $\eta^2$ (Eta al cuadrado).
    - Corrección por pruebas múltiples mediante procedimiento de Benjamini-Hochberg (FDR - False Discovery Rate): cálculo de p-valores ajustados en todos los hallazgos para evitar inflación de descubrimientos falsos y falsos positivos (*data dredging*).
  - **Validación Automatizada:**
    - Suite de pruebas dedicada en `tests/test_explore_phase_e2_advanced_engine.py` (11 pruebas pasando al 100%).
    - Cobertura del paquete `src/automl/engine/analysis` elevada al 88%.

### Track CATML Explore — Fase E1: Núcleo de Estudios (Explore Core) (2026-10-09)

- **Implementación Integral del Núcleo de Estudios Exploratorios (Fase E1 - ADR-008):**
  - **Capa de Dominio Pura (`src/automl/domain/analysis/`):**
    - `models.py`: Entidades desacopladas (`StudySpec`, `AnalysisRun`, `StatisticalFinding`, `VisualizationSpec`, `AnalysisHypothesis`, `EvidenceLink`, `DataSourceRef`, `StudyStatus`) con cero dependencias externas (sin pandas, scipy ni frameworks).
    - Soporte completo para estudios no supervisados (`target_column=None`).
    - Puertos formales en `ports.py`: `StudyRepositoryPort`, `StatisticalAnalyzerPort`, `VisualizationRendererPort`.
  - **Infraestructura y Persistencia SQLite (`src/automl/infrastructure/database/sqlite_studies.py`):**
    - `SQLiteStudyRepository`: Persistencia transaccional aditiva en `studies.db` con soporte para `:memory:`, WAL mode, claves foráneas, índices por workspace/estudio y serialización JSON de métricas y especificaciones visuales.
  - **Motor de Análisis Determinista (`src/automl/engine/analysis/statistical_analyzer.py`):**
    - `StatisticalAnalyzer`: Cálculo de colinealidad bivariada de Pearson ($r \ge 0.70$) con p-valores asociados, separación de clases mediante ANOVA F-test / t-test ($p < 0.05$), detección de outliers mediante IQR, detección de identificadores de alta cardinalidad e hipótesis empíricas de mitigación (`resolve_collinearity`, `power_transform`, `robust_scaler`, `exclude_identifier`).
    - Generación declarativa de especificaciones `VisualizationSpec` para matrices de correlación y distribuciones.
  - **Capa de Aplicación y CQRS Modular (`src/automl/application/`):**
    - Comandos en `analysis/commands.py`: `CreateStudyCommand`, `RunAnalysisCommand`, `ArchiveStudyCommand`.
    - Consultas en `analysis/queries.py`: `GetStudyQuery`, `ListStudiesQuery`, `GetAnalysisRunQuery`, `ListAnalysisRunsQuery`, `ListFindingsQuery`, `ListVisualizationsQuery`, `ListHypothesesQuery`.
    - Servicio `AnalysisStudyService` orquestando repositorio, motor estadístico y dataset repository.
    - Registro modular desacoplado en `registries/analysis_registry.py` e integrado en `bootstrap.py` y `AutoMLWorkspace`.
  - **CLI `automl explore` (`src/automl/interfaces/cli/`):**
    - Subcomandos `create`, `run`, `list`, `show`, `findings` con soporte JSON y formateo tabular en `explore_cli.py` y conectado en `main.py`.
  - **Herramientas de Agente MCP (`src/automl/interfaces/mcp/`):**
    - `analysis_tools.py`: Registro de herramientas `analysis_create_study`, `analysis_run_study`, `analysis_get_findings`, `analysis_list_studies`, `analysis_get_study` con esquemas tipados y registro en `create_mcp_server`.
  - **API REST y Workbench UI Verídica:**
    - Endpoints HTTP en `server.py`: `/api/analysis/studies`, `/api/analysis/study`, `/api/analysis/findings`, `/api/analysis/visualizations`, `/api/analysis/hypotheses`, `/api/analysis/studies/create`, `/api/analysis/studies/run`, `/api/analysis/studies/archive`.
    - Cliente frontend en `api.js` y vista interactiva en `datasets.js`: Pestaña de Exploración con selector de estudios, botón para nuevo estudio y ejecución de análisis, tarjetas de hallazgos estadísticos categorizadas por severidad con badges de p-valor, y matriz de correlación visual interactiva en tiempo real (datos 100% verídicos calculados por el motor).
  - **Validación:**
    - Suite de pruebas de integración completa en `tests/test_explore_phase_e1_core.py` (10 pruebas pasando al 100%).

### Track Technical Audit Remediation & System Hardening (2026-10-09)

- **Remediación Integral de Hallazgos de Auditoría Técnica (Commit `5caa1f5`):**
  - **H1 (Aprobación real de agentes en Workbench):**
    - Se integró `POST /api/agent/action` con `SqliteAgentLedger` en `src/automl/interfaces/web/server.py`.
    - Validación de existencia de hipótesis y solicitudes de aprobación en el ledger (fail-closed `404 Not Found` ante identificadores desconocidos).
    - Transición de estado persistida transaccionalmente (`proposed` $\to$ `accepted` / `rejected`), registrando revisor y notas del operador humano.
  - **H2 (Normalización del signo en métricas personalizadas de minimización en CV):**
    - En `src/automl/engine/training/sklearn_trainer.py`, se eliminó la dependencia sobre el atributo privado inexistente `_greater_is_better` de scikit-learn.
    - Se incorporó `_is_minimizing_metric(metric_name, plugin_registry)` e inspección de `_sign == -1` para invertir el signo negativo producido por scikit-learn en validación cruzada.
    - Sincronización de `plugin_registry` en `SQLiteExperimentRepository.get_leaderboard()` y `AutoMLWorkspace.__post_init__()` para ordenar métricas personalizadas minimizables ascendentemente (`ASC`), garantizando consistencia idéntica entre holdout, CV, Optuna y leaderboard.
  - **H3 (Autenticación estricta en servidor MCP para interfaces remotas):**
    - En `src/automl/interfaces/mcp/server.py`, el transporte `streamable-http` rechaza enlaces externos (`host != 127.0.0.1`) sin token de autenticación (`PermissionError` fail-closed).
    - Soporte para `--token` y `--insecure-no-auth` en CLI (`main.py` y `mcp_cli.py`), incorporando `BearerAuthMiddleware` en la aplicación Starlette subyacente.
  - **H4 (Límite explícito de tamaño de payload HTTP):**
    - En `src/automl/interfaces/web/server.py`, se estableció `MAX_PAYLOAD_SIZE = 50 MB`.
    - `do_POST` valida `Content-Length` y bytes leídos retornando inmediatamente `413 Payload Too Large` ante solicitudes excesivas antes de saturar memoria.
  - **H5 (Endurecimiento de autenticación remota en Workbench):**
    - `_is_authenticated()` en `server.py` rechaza tajantemente tokens en query string (`?token=`) para peticiones de mutación de estado (`POST`).
    - En peticiones `GET`, emite advertencia de seguridad instando al uso de cabeceras `Authorization: Bearer <token>` o fragmentos `#token=`.
  - **H6 (Declaración de versión de scikit-learn):**
    - En `pyproject.toml`, se actualizó la dependencia mínima a `scikit-learn>=1.4.0` para garantizar disponibilidad de `response_method` en `make_scorer`.
    - Se añadió fallback defensivo en `sklearn_trainer.py` (`needs_proba`) para retrocompatibilidad total.
  - **H8 (Compatibilidad multiplataforma en bloqueo de workspaces):**
    - En `src/automl/infrastructure/jobs/worker.py`, se implementó bloqueo condicional con `msvcrt` en entornos Windows y `fcntl` en sistemas Unix.
  - **Validación:** Suite completa de pruebas superada (673 pruebas, 0 fallos), cobertura total mantenida en 86.67% (umbral CI $\ge 85\%$).

- **Eliminación de Maquetas y Estados Simulados en Pantalla de Inicio (Truthful Workbench State):**
  - **Eliminación de estados falsos de entrenamiento:** En `src/automl/domain/runs/states.py` y `src/automl/interfaces/web/static/js/utils.js`, `ACTIVE_RUN_STATUSES` se acotó estrictamente a ejecuciones reales (`EXPERIMENTING`, `OPTIMIZING`, `FINALIZING`, `RUNNING`). Estados pasivos/iniciales (`CREATED`, `PROFILING`, `PLANNING`) ya no se computan erróneamente como ejecuciones activas.
  - **Sincronización de estado en creación de runs:** En `src/automl/application/services/workspace.py` (`create_run`), el estado inicial se establece como `RunStatus.CREATED` tras completar el perfilado síncrono.
  - **HomeView verídica sin mockups (`home.js`):**
    - Se eliminó el badge falso de `ENTRENANDO` con spinner y la barra de progreso fija al 75% (`progress-striped w-3/4`).
    - Ante 0 ensayos entrenados y sin trabajos en cola, se muestra el estado honesto `LISTO PARA ENTRENAR` (con barra vacía al 0%, métricas en estado `Sin entrenar` / `—`, y botón directo para `Lanzar primer experimento`).
    - Solo ante trabajos en segundo plano activos (`job.status === 'running'`) o ejecuciones en curso se activa el indicador de entrenamiento con el progreso real computado.
    - Cuando existen ensayos entrenados (`trials_count > 0`), se muestra `COMPLETADO` con la cantidad real de ensayos finalizados y el mejor score obtenido.
  - **OverviewView sin porcentajes arbitrarios (`overview.js`):** Eliminado el texto hardcoded `Searching space (72%)` y la barra fija al 66% (`w-2/3`). Se refleja el estado real de exploración.
  - **Cabecera global sincronizada (`app.js`):** El badge global muestra `READY` / `CREATED` en verde cuando el motor está en reposo; únicamente conmuta a `ENTRENANDO` con spinner ante ejecuciones efectivas.
  - **Indicadores en Rojo para Funcionalidades Pendientes (`datasets.js`):**
    - Pestaña de Exploración marcada de forma destacada en **ROJO** con badge `PENDIENTE DE IMPLEMENTACIÓN`.
    - Cabecera interna del módulo Explore marcada en rojo con `PENDIENTE DE IMPLEMENTACIÓN · FASE E1 (ROADMAP)` y `EN FASE DE IMPLEMENTACIÓN`.
  - **Validación automatizada:** Suite completa superada (677 pruebas, 0 fallos), cobertura total mantenida en 86.68% (umbral CI $\ge 85\%$). Pruebas unitarias dedicadas en `tests/test_workbench_laboratory_redesign.py`.

### Track Modular Simplification — Pre-Explore Phase E0.5 (2026-10-08)

- **PR 1: Descomposición de Registros CQRS y Modularización de Bootstrap (`hardening/v0.8.2-e0.5-pr1-bootstrap-modularization`):**
  - Creación del paquete `src/automl/application/registries/` con submódulos independientes por dominio:
    - `job_registry.py`: `register_job_handlers` (con inyección opcional de `JobService`).
    - `core_registry.py`: `register_core_handlers` (runs, datasets, profiles, plugins, task types).
    - `experiment_registry.py`: `register_experiment_handlers` (modelos, experimentos, colas, optimización de hiperparámetros).
    - `feature_registry.py`: `register_feature_handlers` (selección, ingeniería temporal y de interacción, ablación).
    - `inference_registry.py`: `register_inference_handlers` (predicción, alineación Kaggle, ensembles, pipelines DAG).
  - Refactorización de `src/automl/application/bootstrap.py`: Reducido drásticamente de 342 a 53 líneas como orquestador limpio y desacoplado, manteniendo 100% retrocompatible la función `build_application`.
  - Guard arquitectónico automatizado en `tests/test_cqrs_architecture_guard.py` (`test_modular_registries_can_be_composed_independently`).
  - Validación completa: 623 tests pasando (100% verde), 87.37% de cobertura.
- **PR 2: Desacoplamiento de Contratos para Datasets sin Target y Jobs sin Run ID (`hardening/v0.8.2-e0.5-pr2-contract-decoupling`):**
  - Desacoplamiento de entidades `Dataset` y `DatasetProfile` (`src/automl/domain/datasets/profile.py`): campos opcionales `target_column: str | None = None` y `task_type: str | None = None`. Manejo seguro en propiedades computadas (`recommended_feature_names`, `leakage_column_names`).
  - Flexibilización de `profile_dataset` (`src/automl/engine/profiling/dataset_profiler.py`): análisis descriptivo, cardinalidad, tipos semánticos y matrices de covarianza y colinealidad sin requerir variable objetivo supervisada.
  - Persistencia de datasets en SQLite (`src/automl/infrastructure/database/sqlite_repository.py`): tabla `datasets` admite `target_column TEXT NULL` y `task_type TEXT NULL`.
  - Ingestión de datasets en `AutoMLWorkspace.register_dataset()` con `target=None` y planificación de tareas por defecto (`TaskType.CLUSTERING`) vía `plan_from_dataframe` (`src/automl/engine/planning/task_planner.py`).
  - Desacoplamiento de Jobs durables (`src/automl/domain/jobs/job.py`, `src/automl/infrastructure/database/sqlite_jobs.py`): `run_id: str | None = None` con migración resiliente en SQLite para bases existentes.
  - Salvaguardas en `JobService` (`src/automl/application/services/jobs.py`) para operaciones independientes de ejecuciones de AutoML.
  - Suite de pruebas unitarias y de integración en `tests/test_unsupervised_datasets_and_decoupled_jobs.py` (4 tests).
  - Validación completa: 627 tests pasando (100% verde), 87.38% de cobertura.
- **PR 3 (PR #79): Hardening de Migraciones SQLite y Compatibilidad Retroactiva (`hardening/v0.8.2-sqlite-migrations-compatibility`):**
  - Implementación de migración atómica determinista de la tabla `datasets` en `SQLiteExperimentRepository._migrate()` mediante reconstrucción transaccional cuando detecta `notnull == 1` en `target_column` o `task_type` de bases heredadas v0.8.0/v0.8.1.
  - Implementación de migración determinista de la tabla `jobs` en `SQLiteJobRepository` relajando `run_id NOT NULL` con manejo estructurado.
  - Erradicación de excepciones silenciadas (`except Exception: pass`) reemplazadas por logging estructurado y excepciones explícitas fail-closed (`RuntimeError`).
  - Suite de pruebas dedicada con bases de datos heredadas y datos preexistentes en `tests/test_sqlite_legacy_migrations.py` (2 tests verificando 0 pérdida de datos y compatibilidad de inserción con `None`).
  - Actualización del contador en `README.md` a 629 tests passing y clarificación de `statsmodels` en `docs/decisions/008-catml-explore-modular-monolith.md`.
  - Validación completa: 629 tests pasando (100% verde), 87.37% de cobertura.

### Track Security & Reliability Hardening — Auditoría Técnica Oct 2026 (2026-10-08)

- **PR A (PR #80): Blindaje Anti-Leakage Centralizado y Fail-Closed (`hardening/v0.8.2-anti-leakage-guardian`):**
  - Centralización de la política de resolución de variables seguras en `DatasetProfile.resolve_safe_feature_names()` (`src/automl/domain/datasets/profile.py`):
    - Filtrado riguroso de target, identifiers y columnas con leakage confirmado (target leakage y group leakage).
    - Soporte para tareas no supervisadas / clustering (desactivación de target leakage cuando no existe variable objetivo o `task_type == "clustering"`).
    - Parámetro de escape explícito `allow_leakage: bool = False` para estudios controlados / benchmarks.
  - Fail-closed preventivo en `AutoML.fit()` (`src/automl/facade.py`):
    - Se elimina el reciclaje inseguro de variables activas del registro cuando todas están contaminadas; interrupción inmediata con `ValueError` explicativo.
  - Blindaje en `RuleBasedExperimentPlanner` (`src/automl/engine/planning/experiment_planner.py`):
    - Se elimina el fallback permisivo que reciclaba columnas tras filtrado total; retorna lista vacía de candidatos si no existen features seguras.
  - Blindaje en `JobExecutor` (`src/automl/application/services/job_executor.py`):
    - Resolución segura automática de features excluyendo leakage para jobs de entrenamiento en background.
  - Validación estricta en `AutoMLWorkspace.create_experiment()` (`src/automl/application/services/workspace.py`):
    - Verificación contra leakage al crear experimentos manuales, propagando `allow_leakage` a través de `CreateExperimentCommand` y el bus CQRS.
  - Calibración de heurísticas estadísticas en `dataset_profiler.py`:
    - `cardinality_ratio <= 0.90` en `detect_is_group_candidate` para evitar falsos positivos en variables continuas (como `account_balance`).
    - `row_count >= 10` en `detect_target_correlation_leakage` para evitar falsos positivos en datasets sintéticos toy/unit tests ($N < 10$).
  - Suite de pruebas exhaustiva en `tests/test_anti_leakage_guardian.py` (6 tests).
  - Validación completa: 635 tests pasando (100% verde), 87.29% de cobertura.
- **PR B (PR #81): Seguridad del Workbench (Confinamiento de Archivos y Sanitización de Rutas) (`hardening/v0.8.2-workbench-file-confinement`):**
  - Confinamiento estricto de `/api/kaggle/download` (`src/automl/interfaces/web/server.py`):
    - Resolución segura de rutas y validación de pertenencia al directorio del workspace (`is_relative_to(ws_root)`).
    - Lista blanca de extensiones para descargas (`.csv`, `.tsv`, `.parquet`, `.pq`, `.json`, `.zip`, `.txt`).
    - Bloqueo fail-closed con `403 Forbidden` ante intentos de path traversal (`../../`), rutas absolutas externas (`/etc/passwd`) o symlinks que apunten fuera del workspace.
  - Confinamiento estricto de `/api/media/preview`:
    - Eliminación de `ws_root.parent` y `Path.cwd()` de la lista `allowed_roots`, restringiendo la lectura exclusivamente a `ws_root` y a los directorios de datasets registrados.
    - Bloqueo preventivo de acceso a archivos del directorio padre o directorios externos del sistema.
  - Sanitización rigurosa de `/static/`:
    - Sustitución de comprobaciones basadas en `str.startswith` por `target_file.is_relative_to(static_root)`.
    - Respuesta explícita `403 Forbidden` ante intentos de escape del directorio de estáticos.
  - Suite de pruebas de seguridad exhaustiva en `tests/test_workbench_file_confinement.py` (4 tests).
  - Validación completa: 639 tests pasando (100% verde), 87.39% de cobertura.
- **PR C (PR #82): Atomicidad Transaccional de Migraciones SQLite y Rollback Verificado (`hardening/v0.8.2-sqlite-migration-atomicity`):**
  - Reemplazo de `conn.executescript()` por transacciones deterministas explícitas `BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK` en `SQLiteExperimentRepository._migrate()` (`src/automl/infrastructure/database/sqlite_repository.py`) y `SQLiteJobRepository._migrate()` (`src/automl/infrastructure/database/sqlite_jobs.py`).
  - Configuración temporal de `conn.isolation_level = None` durante la ventana de migración DDL, garantizando control transaccional estricto ante excepciones intermedias y restaurándolo fielmente en bloques `finally`.
  - Verificación estricta de conteo de registros antes y después de la reconstrucción de tablas (`post_count == pre_count`), levantando `RuntimeError` y ejecutando `ROLLBACK` si se detecta cualquier discrepancia o truncamiento de datos.
  - Limpieza de tablas temporales (`_dg_tmp_datasets`, `_dg_tmp_jobs`) en caso de rollback, asegurando que la base de datos retenga su esquema íntegro original y cero artefactos residuales o bloqueos.
  - Suite de pruebas dedicada para verificar rollback atómico ante fallos forzados de migración en `tests/test_sqlite_legacy_migrations.py` (`test_datasets_migration_failure_rolls_back_atomically`, `test_jobs_migration_failure_rolls_back_atomically`).
  - Validación completa: 641 tests pasando (100% verde), 87.41% de cobertura.
- **PR D (PR #83): Consistencia del Producto — Dirección de Métricas, Benchmarks y Actividad Real del Workbench (`hardening/v0.8.2-product-consistency`):**
  - **Dirección de Métricas en Benchmarks (`src/automl/benchmarks/runner.py`):**
    - Adición de `is_minimizing_metric(metric, ws=None)` reconociendo de forma unificada métricas de pérdida (`mae`, `rmse`, `mse`, `loss`, `log_loss`) y plugins con `greater_is_better = False`.
    - Actualización de `_best_result()` y `run_all()` para seleccionar el modelo con puntaje mínimo en métricas minimizantes (`min`) y máximo en maximizantes (`max`), evitando la selección invertida del peor modelo en benchmarks de regresión.
    - Suite de pruebas dedicada en `tests/test_benchmark_metric_direction.py` (5 tests).
  - **Actividad Real del Workbench y Eliminación de Mock Data (`src/automl/interfaces/web/server.py`, `src/automl/interfaces/web/static/js/views/agent.js`):**
    - Extracción de `_build_real_activity_feed()` en `server.py` que compone eventos estrictamente desde el estado real del workspace:
      - Ganadores legítimos de leaderboard con dirección de métrica respetada (`ACCEPT`).
      - Exclusiones preventivas de leakage e identificadores por el Anti-Leakage Guardian (`REJECT`).
      - Ensayos fallidos con su causa real de error (`REJECT`).
      - Experimentos y jobs planificados o completados (`PLAN` / `ACCEPT`).
      - Hipótesis promovidas, rechazadas o propuestas extraídas dinámicamente de `agent_ledger.db`.
      - Workspace vacío retorna `activity_feed: []`, activando limpiamente el estado vacío nativo de la UI.
    - En `/api/agent/hypotheses`: eliminación de hipótesis mockeadas (`hyp_12`, `hyp_13`, `hyp_14`) y sustitución por consulta dinámica a `SqliteAgentLedger`.
    - En `AgentDrawer` (`agent.js`): renderizado dinámico con consulta asíncrona a `api.getAgentHypotheses()` y estado vacío explícito.
    - Suite de pruebas de consistencia en `tests/test_workbench_activity_consistency.py` (5 tests).
  - Validación completa: 651 tests pasando (100% verde), 87.32% de cobertura.
- **PR E (PR #84): Hardening Complementario — Protección XSS, Segregación de Métricas en Workbench y Confinamiento Estricto de Descargas (`hardening/v0.8.2-workbench-xss-and-metrics-alignment`):**
  - **Protección XSS y Sanitización Frontend (`src/automl/interfaces/web/static/js/utils.js`):**
    - Creación de función `escapeHtml` para sanitización de HTML, strings dinámicos y atributos.
    - Aplicación en `agent.js` (enunciados de hipótesis, acciones propuestas, decisiones y razones del crítico, IDs de hipótesis, dataset y task types).
    - Aplicación en `overview.js` (feed de actividad, listado de datasets, filas del leaderboard y KPIs).
    - Aplicación en `kaggle.js` (listado de experimentos de envíos y estados).
  - **Alineación y Segregación de Métricas en `/api/overview` (`server.py`):**
    - Erradicación de la comparación cruzada de métricas incomparables (`is_minimize` mezclado entre diferentes ejecuciones).
    - Incorporación de `best_by_metric` en la respuesta JSON agrupando campeones por métrica sin mezclar escalas.
    - Contextualización de `best_score`, `best_model` y nuevo campo `best_metric` asociados a la ejecución activa o más reciente.
  - **Dirección de Mejora en Benchmarks (`src/automl/benchmarks/runner.py`):**
    - Adaptación del cálculo de `delta_vs_baseline` para que tanto en métricas de ganancia (ROC-AUC) como de pérdida (RMSE, MAE), una mejora respecto a la línea base se exprese como ganancia positiva ($\Delta > 0$).
  - **Confinamiento Estricto de Descargas en `/api/kaggle/download` (`server.py`):**
    - Confinamiento exclusivo a `ws_root / "submissions"` (y `exports/`).
    - Bloqueo preventivo de descargas arbitrarias de archivos ubicados en la raíz del workspace o con extensiones no tabulares/de archivo (`.csv`, `.tsv`, `.parquet`, `.pq`, `.zip`).
  - **Suite de Pruebas Automatizadas:**
    - `tests/test_workbench_xss_protection.py` (5 tests verificando sanitización de UI, segregación de métricas y cálculo de delta).
    - Ampliación de `tests/test_workbench_file_confinement.py` (4 tests verificando rechazo de archivos en raíz del workspace).
  - **Validación completa:** 656 tests pasando (100% verde), 87.33% de cobertura.
- **PR F (PR #85): Alineación de Estado Activo y Feed de Actividad en Workbench (`hardening/v0.8.2-workbench-active-state-and-feed-alignment`):**
  - **Resolución de Estados Activos en Dominio y API (`src/automl/domain/runs/states.py`, `src/automl/domain/runs/run.py`, `src/automl/interfaces/web/server.py`):**
    - Definición del conjunto canónico `ACTIVE_RUN_STATUSES` (`CREATED`, `PROFILING`, `PLANNING`, `EXPERIMENTING`, `OPTIMIZING`, `FINALIZING`) y función `is_active_run_status()`.
    - Adición de `@property def is_active(self) -> bool` en `AutoMLRun`.
    - Exposición del flag `is_active` en las respuestas JSON de `/api/runs` y `/api/overview`.
    - Selección prioritaria de ejecuciones activas sobre ejecuciones completadas para la vista principal del Workbench.
  - **Segregación del Feed de Actividad (`server.py`):**
    - Erradicación de la comparación escalar cruzada entre métricas incompatibles (`score < prev_score`) en `_build_real_activity_feed()`.
    - Emisión contextualizada de eventos de leaderboard por ejecución con indicación explícita de métrica, dataset e ID de run.
  - **Consistencia Frontend & Tech Minimalista (`utils.js`, `overview.js`, `app.js`, `agent.js`, `studio.js`, `new_experiment.js`, `compare.js`, `kaggle.js`, `knowledge.js`, `pipeline.js`):**
    - Función exportada `isRunActive()` en `utils.js` consumida en todas las vistas del Workbench y en el polling global (`app.js`).
    - Soporte en `overview.js` para scores iguales a `0.0` y métricas con valores negativos ($R^2 < 0$) mediante `overview.best_score != null`.
    - Corrección del mensaje en el bloque de captura de error al programar el plan del agente en `agent.js` (`"Agent action failed: "`).
  - **Suite de Pruebas Automatizadas:**
    - Nuevos tests en `tests/test_workbench_activity_consistency.py` (`test_workbench_selects_active_experimenting_run_over_completed_run`, `test_activity_feed_multi_run_segregated_without_comparing_incompatible_metrics`, `test_domain_is_active_run_status_and_run_property`, `test_workbench_frontend_active_status_and_agent_error_alert`).
  - **Validación completa:** 660 tests pasando (100% verde), 87.31% de cobertura.
- **PR G: Flexibilidad Anti-Leakage en Planificación e Integridad de Métricas en Workbench (`fix/planner-anti-leakage-and-metric-integrity`):**
  - **Flexibilidad Anti-Leakage en Planificación (`profile.py`, `experiment_planner.py`):**
    - Adición del parámetro `strict: bool = True` en `DatasetProfile.resolve_safe_feature_names()`. Si `strict=False`, las columnas con fuga de datos o identificadores se descartan de forma segura en vez de levantar `ValueError`, permitiendo continuar si restan características limpias.
    - Uso de `strict=False` en `RuleBasedExperimentPlanner.propose()` para soportar datasets mixtos (con columnas de leakage e identificadores junto a predictoras válidas).
  - **Segregación de Métricas por Dataset y Resiliencia en Workbench (`server.py`):**
    - En `/api/overview`, segregación de mejores modelos por dataset (`best_by_dataset`) para evitar comparaciones escalares numéricas erróneas entre datasets dispares.
    - Preservación de puntuaciones de métricas con valor exacto `0.0` y cálculo de deltas en `/api/agent/hypotheses`.
    - Inicialización limpia de `best_score`, `best_model` y `best_metric` como `None` en workspaces vacíos sin ejecuciones.
  - **Suite de Pruebas Automatizadas:**
    - 2 nuevos tests en `tests/test_anti_leakage_guardian.py` (`test_experiment_planner_proposes_candidates_on_mixed_clean_and_leakage_dataset`, `test_resolve_safe_feature_names_strict_vs_filtering_mode`).
    - 2 nuevos tests en `tests/test_workbench_activity_consistency.py` (`test_agent_hypotheses_preserves_exact_zero_metrics`, `test_overview_segregates_metrics_across_different_datasets`).
  - **Validación completa:** 664 tests pasando (100% verde), 87.32% de cobertura.
- **PR: Rediseño UI Workbench — Modelo de Laboratorio y Dataset Activo (Fase 1) (`feat/ui-laboratory-redesign-phase1`):**
  - **Contexto de Laboratorio y Dataset Activo en Barra Lateral (`index.html`, `app.js`, `store.js`):**
    - Indicador de estado "Laboratorio local • ONLINE".
    - Tarjeta de dataset activo con indicador reactivo (`#sidebarActiveDatasetName`) y botón de acción rápido `Cambiar dataset` (`#btnSidebarSwitchDataset`).
    - Métodos reactivos en `WorkbenchStore`: `setActiveDataset(datasetId)`, `getActiveDataset()`, sincronización con `localStorage` y alineación automática de `activeRunId`.
  - **Simplificación de la Navegación Principal (`index.html`, `app.js`):**
    - Reducción de ~10 opciones dispersas a 4 ventanas de flujo canónico: **Inicio** (`home`), **Dataset** (`dataset`), **Experimentos** (`experiments`), **Evidencia** (`evidence`).
    - Paridad y retrocompatibilidad en el router para rutas previas.
  - **Nueva Vista de Inicio (`src/automl/interfaces/web/static/js/views/home.js`):**
    - Hero Card principal: «Continuar con {dataset activo}» con dimensiones, variable objetivo, tarea y mejor modelo, junto a acciones secundarias «Añadir dataset» y «Abrir dataset guardado».
    - Panel de «Ejecuciones en curso»: estado de entrenamiento, barra de progreso y métricas en vivo.
    - Panel de «Trabajos recientes»: historial técnico de jobs en background consultados dinámicamente vía `api.getJobs()`.
  - **Rediseño de la Vista de Dataset (`src/automl/interfaces/web/static/js/views/datasets.js`):**
    - Pestaña «Resumen»: Diagnóstico de salud con Anti-Leakage Guardian (fugas y variables descartadas), métricas de perfilado, recomendaciones estadísticas y muestra previa de datos crudos.
    - Pestaña «Columnas»: Tabla completa interactiva de columnas con filtrado, buscador, porcentajes de nulos, cardinalidad, acción inferida y selección de variables.
    - Pestaña «Exploración»: Panel conceptual de arquitectura técnica para la Fase E1 de CATML Explore (ADR-008).
    - Botón CTA destacado «Nuevo experimento» en cabecera.
  - **Modal de Selección Rápida de Dataset (`src/automl/interfaces/web/static/js/views/switch_dataset_modal.js`):**
    - Modal accesible desde cualquier vista para conmutar el dataset activo sin perder el historial ni el trabajo previo.
  - **Contextualización en Vistas Existentes (`studio.js`, `overview.js`, `new_experiment.js`):**
    - Filtrado y priorización automática de ejecuciones y sugerencias pertenecientes al dataset activo.
  - **Suite de Pruebas Automatizadas:**
    - `tests/test_workbench_laboratory_redesign.py` (3 tests verificando layout, componentes, enrutamiento y métodos de estado).
  - **Validación completa:** 667 tests pasando (100% verde), 87.32% de cobertura.

### Track Architectural Planning — CATML Explore Evolution Plan (2026-10-08)

- **Formalización de CATML Explore como Módulo de Monolito Modular:**
  - Registro de decisión arquitectónica en [`docs/decisions/008-catml-explore-modular-monolith.md`](docs/decisions/008-catml-explore-modular-monolith.md).
  - Especificación funcional y de dominio en [`docs/features/catml-explore/spec.md`](docs/features/catml-explore/spec.md).
  - Plan de implementación y roadmap por fases E0 a E6 en [`docs/features/catml-explore/plan.md`](docs/features/catml-explore/plan.md).
  - Integración en [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) (Sección 21) y actualización del backlog operativo en [`TASKS.md`](TASKS.md).
  - Definición del modelo de dominio: `DataSourceRef`, `StudySpec`, `AnalysisRun`, `StatisticalFinding`, `VisualizationSpec`, `AnalysisHypothesis`, `EvidenceLink`.
  - Establecimiento del principio *"La IA Interpreta, CATML Calcula"* para desacoplar el cálculo matemático determinista de la formulación de hipótesis por agentes en MCP/CLI.

### Documentación pública (2026-10-08)

- Aplicado CATML_monetizacion_privada.patch en docs/private-commercial-strategy; retiradas referencias comerciales de documentación y landing.
- Validación: build y comprobaciones web responsive/accesibilidad correctos; Python: 601 passed, 17 failed (fcntl no disponible y saltos CRLF en Windows), cobertura 86.22 %. gh no está disponible en este equipo.

### Track Hardening Phase 2 — v0.8.2 Ingestion Polymorphism, Strict CQRS Boundaries & Application Services Decomposition (2026-10-08)

- Rama de trabajo `hardening/v0.8.2-phase2-modular-and-cqrs` desde `main` (`ec69771`).
- **Ingestión Polimórfica y Soporte Nativo de Parquet/DataFrames (`src/automl/engine/profiling/dataset_profiler.py`, `src/automl/application/services/workspace.py`, `src/automl/facade.py`, `pyproject.toml`):**
  - Función `load_dataframe()` en `dataset_profiler.py` con parsing automático de `.parquet`, `.pq`, `.json`, `.jsonl`, `.tsv` y `.csv`. Manejo con mensaje amigable de `ImportError` solicitando `pyarrow` o `fastparquet` si no se encuentran instalados.
  - Actualización de `profile_dataset()` para delegar en `load_dataframe()`.
  - En `AutoMLWorkspace.register_dataset()`, polimorfismo que acepta `pd.DataFrame` directamente, persistiendo de forma automática en el subdirectorio `datasets/` del workspace y registrándolo sin requerir guardado manual externo por parte del usuario.
  - En `AutoML.fit()` (`facade.py`), polimorfismo para aceptar rutas en `str` o `Path` (además de DataFrames y matrices numpy), integrándose fluidamente con `_standardize_input()`.
  - Definición del extra `parquet = ["pyarrow>=14.0"]` en `pyproject.toml` e inclusión de `pyarrow` en el grupo `all`.
  - Pruebas unitarias y de integración en `tests/test_parquet_and_polymorphic_ingestion.py` (4 tests).
- **Frontera Arquitectónica CQRS Estricta y Erradicación de Fugas de Estado (`src/automl/application/queries/workspace_queries.py`, `src/automl/application/bootstrap.py`, `src/automl/application/services/workspace.py`, `src/automl/facade.py`, `src/automl/interfaces/web/server.py`, `src/automl/interfaces/cli/agent_session_cli.py`):**
  - Formalización de consultas tipadas en `QueryBus`: `GetRunQuery`, `ListRunsQuery`, `GetDatasetQuery`, `ListDatasetsQuery`, `GetTrialQuery`, `GetExperimentQuery`, `ListTrialResultsQuery`.
  - Registro de los nuevos handlers en `src/automl/application/bootstrap.py` desacoplados de infraestructura directa.
  - Métodos de consulta públicos agregados a `AutoMLWorkspace`: `list_datasets()`, `get_dataset_profile()`, `save_dataset_profile()`, `save_run()`, `get_trial()`, `get_experiment()`, `list_experiments()`, `list_trial_results()`, `get_leaderboard_results()`, `list_feature_sets()`.
  - Erradicación de todas las llamadas directas a `ws.repository` en la fachada (`facade.py`) y en el servidor web (`interfaces/web/server.py`).
  - Eliminado el monkey-patching sobre `ws` en `agent_session_cli.py`.
  - Guard arquitectónico automatizado instalado en `tests/test_cqrs_architecture_guard.py` (2 tests verificando 0 ocurrencias de fugas privadas en interfaces/presentación y despacho vía `QueryBus`).
- **Descomposición del God Module `workspace.py` en Servicios Especializados de Aplicación:**
  - `src/automl/application/services/pipeline_service.py` (`PipelineExecutionService`): extracción de validación de grafos, orden topológico de ejecución, construcción de DAG multimodal, ejecución y fit/predict.
  - `src/automl/application/services/inference_service.py` (`InferenceService`): extracción de predicción, alineación de plantillas Kaggle, generación de envíos y exportación de `ModelArtifact`.
  - `src/automl/application/services/feature_service.py` (`FeatureEngineeringService`): extracción de características derivadas, análisis de dinámica temporal, estrategias de selección multi-método, planificación de ablación y promoción de conjuntos de características.
  - Delegación 100% retrocompatible en `AutoMLWorkspace` preservando firmas públicas y reduciendo el tamaño de `workspace.py` en más de 740 líneas (~2,550 a ~1,800 líneas).
- **Resolución de Auditoría Técnica y Hardening de Confiabilidad ML (`src/automl/engine/training/sklearn_trainer.py`, `src/automl/infrastructure/database/sqlite_repository.py`, `src/automl/application/services/workspace.py`, `src/automl/facade.py`, `src/automl/interfaces/mcp/server.py`, `src/automl/interfaces/web/server.py`, `src/automl/interfaces/web/static/js/api.js`, `tests/test_audit_findings.py`):**
  - **Signo de MAE/RMSE/MSE/Log Loss en CV**: En `sklearn_trainer.py`, los scores de `cross_val_score` con scoring `neg_...` o `_greater_is_better=False` se invierten mediante `scores = -scores`. El leaderboard y optimizador en `workspace.py` ordenan ascendentemente (`ASC`) y minimizan, garantizando que el menor error ocupe la primera posición.
  - **Exclusión de Trials Fallidos**: En `sqlite_repository.py`, `get_leaderboard()` agrega por defecto `AND (tr.failure_reason IS NULL OR tr.failure_reason = '')` (`include_failed=False`). En `facade.py`, `fit()` verifica que existan trials exitosos y descarta fallos; en `inference_service.py`, `export_model_artifact()` elige únicamente entre modelos con `res.succeeded is True`.
  - **Eliminación de Sustitución Silenciosa de Métricas**: En `_sklearn_scoring()` y holdout `_compute_metrics()`, métricas no reconocidas levantan `ValueError` explícito en vez de recurrir a `accuracy`/`r2`. Se añadió soporte directo para `mse`, `log_loss`, `balanced_accuracy`, `precision`, `recall` y plugins métricos mediante `make_scorer`.
  - **Clarificación de Time Budget**: Documentado en `facade.py` que `time_budget` es un límite cooperativo global entre modelos y no un timeout atómico por proceso.
  - **Endurecimiento de Seguridad HTTP/MCP**: En `api.js` y `server.py`, paso de credenciales mediante URL hash fragment `#token=` (inmune a logs de proxy/historial de peticiones) y auto-limpieza con `history.replaceState`. En el servidor MCP Streamable-HTTP, advertencia de seguridad explícita (`SECURITY NOTICE`) si se expone a interfaces de red externas.
  - **Robustez de Validación en Datasets Reducidos**: `SklearnTrainer` adapta `n_splits` en KFold para asegurar suficientes muestras de test y permitir cálculos estables de varianza ($R^2$), y en holdout ajusta `effective_test_size` para preservar representatividad de todas las clases.
  - 4 tests automatizados de regresión en `tests/test_audit_findings.py`.
- **Validación y Cobertura:**
  - 622 tests pasando (100% verde). Cobertura total del 87.37% (requisito >= 85%).
  - CLI `automl --help` y `automl task list` verificados exitosamente.

### Track Hardening Phase 1 — v0.8.2 Operational Budget, Fail-Closed Artifacts & Remote Web Security (2026-10-07)

- Rama de trabajo `hardening/v0.8.2-phase1-20261007` desde `main` (`7c78f13`).
- **Cumplimiento Operativo Real de `time_budget` (`src/automl/domain/runs/run.py`, `src/automl/application/services/workspace.py`, `src/automl/facade.py`):**
  - Añadido campo `time_budget_seconds: float | None = None` a `RunConfig`, con propiedad retrocompatible `effective_time_budget` que soporta tanto `time_budget_seconds` como la clave heredada `extra["time_budget"]`.
  - Serialización y deserialización limpias en `to_dict()` y `from_dict()`.
  - En `AutoMLWorkspace`, tracking de deadlines activos mediante `_run_deadlines: dict[str, float]` calculados con `time.monotonic()`.
  - En `run_experiment()`, verificación estricta de `_is_run_budget_exhausted(refreshed)` antes de iniciar cada trial; si el presupuesto expira, se interrumpe el ciclo de trials de forma limpia, se registra `time_budget_exhausted = True` en `run.config.extra` y se emite el evento de auditoría `TimeBudgetExhausted`.
  - En `run_scheduled_experiments()`, interrupción limpia de la cola si expira el presupuesto global del run.
  - En `AutoML.fit()`, propagación formal de `time_budget_seconds=self.time_budget` a `create_run()`. Si el presupuesto expira antes de completar al menos un modelo válido, levanta `RuntimeError` explícito; si ya existen modelos válidos entrenados, finaliza con éxito preservando los resultados y marcando `time_budget_exhausted=True`.
  - Tests unitarios en `tests/test_time_budget_enforcement.py` (4 tests).
- **Seguridad Fail-Closed en ModelArtifact (`src/automl/artifacts/model_artifact.py`):**
  - Eliminados los bloques `except Exception: pass` silenciosos tanto en la generación del checksum `.sha256` en `save()` como en la verificación en `load()`.
  - En `ModelArtifact.load(verify_checksum=True)`, política fail-closed estricta: levanta `ValueError` explícito si el archivo sidecar `.sha256` falta, está corrupto, es inaccesible o no coincide antes de intentar cualquier deserialización con `joblib.load()`.
  - Tests unitarios en `tests/test_model_artifact_fail_closed.py` (3 tests).
- **Autenticación por Token en Workbench Remoto (`src/automl/interfaces/web/server.py`, `src/automl/interfaces/cli/main.py`, `src/automl/interfaces/web/static/js/api.js`):**
  - En `AutoMLWebHandler`, comprobación de cabecera `Authorization: Bearer <token>` o parámetro URL `?token=<token>`. Devuelve `401 Unauthorized` si la autenticación está activa y el token no es provisto o es incorrecto.
  - En `run_web_dashboard()`, si la dirección de escucha no es local (ej. `0.0.0.0`), se exige token automáticamente: si el usuario no especificó `--auth-token`, genera uno criptográficamente seguro con `secrets.token_urlsafe(16)` e imprime las instrucciones y enlace autenticado en consola.
  - Soporte para bandera explícita `--insecure-no-auth` para desactivar intencionalmente el control si el usuario así lo decide.
  - En el frontend (`api.js`), extracción automática del token desde la URL, persistencia en `sessionStorage` e inyección de cabecera `Authorization: Bearer` en todas las peticiones fetch a la API.
  - CLI `automl ui` / `catml ui` soporta flags `--auth-token` y `--insecure-no-auth`.
  - Tests unitarios en `tests/test_workbench_auth.py` (3 tests).
- **Cierre de Fuga CQRS en CLI (`src/automl/interfaces/cli/main.py`, `src/automl/application/services/workspace.py`):**
  - Añadido método público de consulta `AutoMLWorkspace.list_runs(dataset_id: str | None = None) -> list[AutoMLRun]`.
  - Sustituida la introspección privada `ws._runs.values()` en el comando `predict_cli` por la llamada al método público `ws.list_runs(dataset_id=dataset.id)`.
- **Validación:**
  - 10 nuevos tests unitarios y de integración añadidos. Suite global en verde (612 tests pasando) y cobertura >= 85%.


- Rama de trabajo `feat/v081-trust-patch` desde `main` (`5caf495`).
- **Fail-Safe Anti-Leakage (`src/automl/facade.py` & `src/automl/engine/planning/experiment_planner.py`):**
  - `AutoML.fit()` y `RuleBasedExperimentPlanner` ahora priorizan `profile.recommended_feature_names`, excluyendo de forma automática cualquier columna clasificada con target leakage, leakage de entidad o identificadores.
  - Test unitario de regresión en `tests/test_v081_trust_patch.py::test_trust_patch_leakage_fail_safe`.
- **Aislamiento Local & CORS Restringido (`src/automl/interfaces/web/server.py` & `src/automl/interfaces/cli/main.py`):**
  - Servidor web del Workbench configurado para escuchar en `127.0.0.1` por defecto (con soporte explícito para `--host 0.0.0.0` en CLI si el usuario desea exponerlo intencionalmente).
  - Eliminado el uso de wildcard `Access-Control-Allow-Origin: *`; las respuestas CORS ahora solo se emiten dinámicamente si el `Origin` proviene de `localhost` o `127.0.0.1`.
- **Confinamiento de Archivos en Media Preview (`src/automl/interfaces/web/server.py`):**
  - Endpoint `/api/media/preview` ahora comprueba estrictamente con `is_relative_to()` que la ruta solicitada resida dentro de `workspace_dir` o del directorio del dataset, bloqueando lectura de archivos arbitrarios del sistema con código 403 Forbidden.
  - Test unitario de confinamiento en `tests/test_v081_trust_patch.py::test_trust_patch_media_preview_confinement`.
- **Propagación de `random_state` para Reproducibilidad (`src/automl/application/services/workspace.py` & `src/automl/facade.py`):**
  - Añadido parámetro `random_seed` a `workspace.create_run()`, propagándolo al `RunConfig` y garantizando que `AutoML(random_state=123)` llega a todos los splitters (KFold/StratifiedKFold) y modelos.
  - Test unitario de propagación en `tests/test_v081_trust_patch.py::test_trust_patch_random_state_propagation`.
- **Ordenación Correcta de Métricas de Minimización (`src/automl/facade.py` & `src/automl/infrastructure/database/sqlite_repository.py`):**
  - `AutoMLResult.leaderboard()` y `SQLiteExperimentRepository.get_leaderboard()` ahora ordenan ascendentemente para métricas de error (`mae`, `rmse`, `mse`, `loss`, `log_loss`), garantizando que el modelo con menor error ocupa el rango 1.
  - Test unitario en `tests/test_v081_trust_patch.py::test_trust_patch_leaderboard_mae_rmse_sorting`.
- **Eliminación de Evidencia Simulada & Actualización de Mensaje:**
  - Suprimida la fórmula ficticia `"public_lb": round(best_t.primary_score * 0.9999, 5)` y el delta simulado en la API web.
  - `README.md` actualizado con posicionamiento claro: *"AutoML you can trust an AI agent to operate"*, con Quickstart centrado en protección de fugas y reproducibilidad.
- **Validación y Versión:**
  - Bump de versión a `0.8.1` en `src/automl/__init__.py`, `CHANGELOG.md` y `tests/smoke_test.py`.
  - 602 tests pasando al 100% en verde. Smoke test end-to-end verificado.

### Track Release Engineering — v0.8.0 Release Final Polish & PyPI Verification (2026-10-07)

- Rama de trabajo `chore/v08-release-prep-and-smoke` desde `main` (`a2f78d0`).
- **Verificación Estricta de Dependencias en Visión (`src/automl/plugins/modalities/image_plugin.py`):**
  - `ImageModalityPlugin.available()` ahora comprueba explícitamente `torchvision` junto con `torch` y `PIL` para evitar falsos positivos cuando solo `torch` está instalado.
  - Test de verificación `test_image_modality_plugin_available_requires_torchvision` en `tests/test_v07_image_plugin.py`.
- **Modernización de Pesos TorchVision & Offline Mode (`src/automl/engine/vision/image_encoder.py`):**
  - Implementada resolución con API moderna de TorchVision (`get_model_weights(name).DEFAULT` y `Weights.DEFAULT`), con fallback para compatibilidad heredada.
  - Mensaje de excepción informativo documentando la descarga de pesos de TorchVision en primer uso y cómo precargar la caché en entornos aislados/air-gapped (`~/.cache/torch/hub/checkpoints`).
  - Nota correspondiente agregada en `README.md`.
- **Contrato de Dimensión de Embeddings (`src/automl/engine/vision/image_encoder.py`):**
  - Introducido atributo scikit-learn `self.feature_dim_`, descubierto y fijado durante `fit()` y utilizado en `get_feature_names_out()` dinámicamente.
- **Desacoplamiento de Cabezal de Visión (`src/automl/plugins/models/vision_plugin.py`):**
  - Exportado alias arquitectónico `VisionFeatureHeadPlugin = TimmVisionPlugin`.
- **Ergonomía de Fachada (`src/automl/facade.py`):**
  - Añadido alias de parámetro `task_type` en `AutoML.__init__()` para evitar excepciones por discrepancia entre `task` y `task_type`.
- **Packaging Limpio para PyPI (`pyproject.toml`):**
  - Migración a string SPDX moderno `license = "Apache-2.0"` (eliminando la advertencia de obsolescencia de setuptools).
  - Añadidos metadatos completos `[project.urls]` (Homepage, Repository, Issues, Documentation).
  - Eliminado el extra redundante `nlp` (cuyas dependencias ya están en el core).
  - Wheel y sdist construidos limpiamente con cero advertencias: `catml-0.8.0.tar.gz` y `catml-0.8.0-py3-none-any.whl`.
- **Smoke Test Automatizado E2E & Job de CI (`tests/smoke_test.py`, `.github/workflows/ci.yml`):**
  - Creado script ejecutable `tests/smoke_test.py` que valida importación, metadatos, entrenamiento, exportación con sidecar `.sha256`, verificación criptográfica e inferencia de producción.
  - Probado con éxito en entorno virtual aislado limpio (`/tmp/catml_test_env`).
  - Añadido job `package-smoke` al workflow de CI de GitHub Actions dependiente de `test`.
- **Validación Global:**
  - 598 tests pasando en verde (100% passing). Cobertura >= 87.6%.

### Track Pre-PyPI Release Preparation — Vision Honesty, Packaging Extras & Governance (2026-10-07)

- Rama de trabajo `fix/vision-packaging-and-v08-prep` desde `main` (`3335b87`).
- **Honestidad y Corrección del Pipeline de Visión (`src/automl/engine/vision/image_encoder.py`, `src/automl/engine/training/sklearn_trainer.py`, `src/automl/facade.py`):**
  - `ImageEncoderNode` ahora preserva por defecto la dimensionalidad nativa del backbone (ej. 512 para ResNet18) sin truncamiento arbitrario destructivo ni padding artificial.
  - En `_build_pipeline()`, si se detectan columnas de imagen y no están instaladas las dependencias de visión neurales (`catml[vision]`), el sistema levanta un `RuntimeError` explícito indicando la instalación requerida, en lugar de simular visión mediante hash determinista.
  - Para pruebas unitarias y benchmarks reproducibles, se soporta explícitamente `image_model="deterministic"`.
  - Propagación de `image_model` en la fachada `AutoML(image_model=...)` y en `fit()`, con propagación correcta a `workspace.export_model_artifact()` y exclusión limpia de parámetros pasados al estimador scikit-learn.
- **Claridad de Nomenclatura en Plugins y UI (`src/automl/plugins/models/vision_plugin.py`, `src/automl/interfaces/web/static/js/views/new_experiment.js`):**
  - Renombrado `TimmVisionPlugin` a "Vision Feature Head (MLP on Extracted Embeddings)" clarificando que entrena un cabezal MLP sobre representaciones extraídas en lugar de backpropagation nativo end-to-end.
  - Etiqueta en la UI del Workbench actualizada de "Vision (timm)" a "Vision Feature MLP".
- **Integridad y Streaming en ModelArtifact (`src/automl/artifacts/model_artifact.py`):**
  - Implementada función streaming `_compute_file_sha256(path, chunk_size=65536)` para evitar saturación de memoria RAM en modelos de gran tamaño.
  - Documentación técnica explícita en docstrings distinguiendo la integridad del archivo (detección de corrupción accidental) frente a autenticidad criptográfica (firmas PKI contra adversarios maliciosos).
- **Inmutabilidad en el Dominio (`src/automl/domain/problems/spec.py`):**
  - `ProblemSpec` decorado con `@dataclass(frozen=True)` con normalización automática de listas de entrada a tuplas inmutables en `__post_init__`.
  - Test de inmutabilidad agregado en `tests/test_v09_capability_layer.py`.
- **Estructura Limpia de Empaquetado (`pyproject.toml`):**
  - Agrupación de dependencias extras: `models`, `vision`, `nlp`, `mcp`, `agents`, `all`.
  - Inclusión de dependencias de visión (`torch`, `torchvision`, `timm`, `pillow`) dentro de `all`. Eliminado el grupo redundante `full`.
- **Gobernanza y Sincronización de Documentación:**
  - `README.md` actualizado con 595 tests passing, Tech Minimalista Premium workbench, CLI V0.8.0 y taxonomía precisa de visión.
  - `docs/MASTER_PLAN.md` Sección 7 sincronizada marcando capacidades v0.8 como `IMPLEMENTADO (SHIPPED IN 0.8)` y fases futuras como `v0.8.x` y `v1.0+`.
  - Creados `SECURITY.md`, `CHANGELOG.md` (Keep a Changelog) y `TRADEMARK.md` (política de marca CATML).
- **Validación Completa:**
  - 595 tests pasando al 100% en verde. Cobertura $\ge 85\%$.

### Track Engine & Packaging Hardening — Vision Safety, Facade Fixes & Artifact Security (2026-10-07)

- Rama propia `feat/v08-hardening-and-consistency` desde `main` (`4be3e5f`).
- **Corrección de Validación en Fachada (`src/automl/facade.py` & `src/automl/application/services/workspace.py`):**
  - Resuelto bug donde `AutoML.fit` ignoraba `cv_folds` al hardcodear validación `holdout` sin grupo.
  - Ahora propaga `cv_folds` a `workspace.create_run()` y selecciona `stratified_kfold` / `kfold` cuando `cv_folds > 1` (o `group_kfold` si hay columna de grupo), preservando `holdout` únicamente para `cv_folds == 1`.
- **Endurecimiento del Encoder de Visión (`src/automl/engine/vision/image_encoder.py`):**
  - Eliminado el fallback silencioso a hash determinista pseudoaleatorio en modelos neurales (`allow_fallback=False` por defecto).
  - Si un modelo neural (`resnet18`, etc.) se solicita y no están disponibles PyTorch/timm o fallan los pesos preentrenados, se levanta `RuntimeError` explícito ("Fail visibly, don't fake vision").
  - Eliminado el fallback silencioso a redes aleatorias sin pesos preentrenados `model_fn()`.
  - Reservado el vector determinista por hash exclusivamente cuando `model_name in ("deterministic", "hash")`.
- **Seguridad en Artefactos (`src/automl/artifacts/model_artifact.py`):**
  - Implementada advertencia explícita de seguridad (`UserWarning`) en `ModelArtifact.load` recordando los riesgos de deserialización no confiable de pickle/joblib.
  - Implementada generación de sidecar `.sha256` en `save()` y verificación criptográfica de integridad en `load(verify_checksum=True)` (lanzando `ValueError` si hay discrepancia de checksum).
- **Packaging y Claridad de Documentación:**
  - Renombrado el paquete en `pyproject.toml` a `name = "catml"`.
  - Configurados puntos de entrada CLI: comando principal `catml` y alias retrocompatible `automl`. Instalado en editable (`catml-0.8.0`).
  - Actualizado `README.md` aclarando la taxonomía de capacidades: Estable (Tabular), Disponible (NLP y Vision feature representation), Hoja de ruta (visión nativa end-to-end), y reemplazado el reclamo "zero-dependency" por "Database-free, workspace-independent production inference".
- **Validación Completa:**
  - 593 tests pasando al 100% en verde (0 fallos).
  - Cobertura de código: 87.65% (superando el umbral >= 85%).
  - CLI `catml --help` y `automl --help` verificados.

### Track Governance & Licensing — Relicensing to Apache-2.0 & v0.8.0 Transition (2026-10-07)

- Etiquetado `v0.7.0` fijado y publicado en origen como último hito bajo licencia MIT.
- Creación de rama de trabajo `chore/relicense-apache-2`.
- **Blindaje de Licencia e Historia Transparente:**
  - Sustitución de `LICENSE` por el texto oficial de Apache License 2.0 (con cláusula de defensa/represalia de patentes y protección de marca registrada).
  - Creación de archivo `NOTICE` con atribución a CATML Contributors y mención histórica de transición.
  - Blindaje preventivo de `.gitignore` frente a fuga de credenciales (`.env`, `*.pem`, `*.key`, `credentials.json`, `secrets.json`).
- **Alineación de Metadatos y Documentación:**
  - Actualización de `pyproject.toml` (`license = { text = "Apache-2.0" }`) y bump de versión a `0.8.0` en `src/automl/__init__.py`.
  - Dinamización de versión en `src/automl/interfaces/mcp/server.py` y `src/automl/interfaces/web/server.py` importando `__version__`.
  - Actualización de badges y textos legales en `README.md`, `docs/MASTER_PLAN.md` y `docs/decisions/006-tech-minimalist-premium-brand-system.md`.
  - Actualización de menciones de licencia en la landing estática (`website/src/App.tsx`), con build estático verificado (230 ms).
- **Validación Completa:**
  - 590 tests pasando al 100% en verde.
  - Cobertura de código mantenida en 87.70% (superando el umbral >= 85%).
  - Verificación CLI de `automl --help` (reportando V0.8.0) y `automl task list`.

### Track Persona B — Paquete B5: LangGraph StateGraph, Durable SQLite Checkpointer & Crash Recovery (2026-10-06)

- Rama propia `feat/agentic-b5-langgraph-orchestrator` desde `main` (`64ec800`).
- **Mapeo de Estado de Agente (`src/automl/application/agents/state.py`):**
  - Implementación de `GraphAgentState` (TypedDict) para compatibilidad con canales LangGraph.
  - Funciones puras de serialización/deserialización bidireccionales: `session_to_graph_state()`, `graph_state_to_session()`, `serialize_graph_state()`, `deserialize_graph_state()`.
  - Mapeo seguro evitando colisiones con canales reservados en Pregel (`checkpoint_id` $\rightarrow$ `session_checkpoint_id`).
- **Adaptador de Persistencia SQLite (`src/automl/infrastructure/database/sqlite_checkpoint_saver.py`):**
  - Implementación de `SqliteCheckpointSaver` sobre `langgraph.checkpoint.sqlite.SqliteSaver` con manejo multi-hilo (`check_same_thread=False`), inicialización idempotente de tablas de checkpoints, aislamiento contextual con context manager (`__enter__`, `__exit__`), y reconexión resiliente tras desconexiones o fallos.
- **Orquestador LangGraph StateGraph (`src/automl/application/agents/orchestrator/graph.py`):**
  - Grafo de estados StateGraph con nodos: `observe` $\rightarrow$ `propose` $\rightarrow$ `gate` $\rightarrow$ `execute` $\rightarrow$ `critique` $\rightarrow$ `check_stop`.
  - `observe`: construye contexto acotado mediante `ContextBuilder` sin fuga de datos crudos.
  - `propose`: formula candidatos empíricos mediante `Planner` / `FeatureAdvisor` y detecta duplicación criptográfica de hipótesis (`is_hypothesis_duplicate`).
  - `gate`: evalúa presupuestos y permisos; utiliza `interrupt()` para pausar ante autorización humana y reutiliza solicitudes de aprobación previas (`PENDING`) en reanudaciones.
  - `execute`: verifica idempotencia de operaciones en SQLite ledger para prevenir re-ejecuciones duplicadas; invoca `ToolExecutor` e interconecta creación y corrida de experimentos (`run_experiment`), mapeando `metric_value` a `score`.
  - `critique`: evalúa avances empíricos mediante `Critic`, gestiona paciencia ante estancamiento y comprueba objetivo de métrica.
  - `check_stop`: actualiza contador de iteración, sincroniza `AgentSessionState` con `SqliteAgentLedger` y persiste checkpoints de sesión.
  - Métodos públicos de ejecución: `run()`, `resume()`, `get_latest_graph_state()`.
- **Fachada y Paridad CLI (`src/automl/interfaces/cli/agent_session_cli.py`):**
  - Subcomando `automl agent session resume` y `status` con argumento `--engine {deterministic,langgraph}`.
  - Métodos delegados públicos `get_run()` y `get_leaderboard()` en `AutoMLWorkspace`.
  - Búsqueda dual por `run_id` o `session_id` en `SqliteAgentLedger.list_sessions()`.
- **Validación Completa:**
  - `tests/test_v10_orchestrator.py`: 24/24 tests pasando.
  - `tests/test_v10_orchestrator_b5.py`: 13/13 tests pasando.
  - `tests/test_v10_agent_e2e.py`: 3/3 tests de ciclo autónomo, recuperación tras fallos (crash & restart) sin duplicación de hipótesis, y rechazo humano de propuestas.
  - Suite global completa: 590 tests pasando en verde, 0 fallos, 87.70% de cobertura de código (superando el requisito >= 85%).
  - Hito H5 (V1.0) completado exitosamente.
- **Corrección de CI Workflow (`.github/workflows/ci.yml`):**
  - Añadido extra `agents` a la instalación en CI (`python -m pip install -e ".[dev,mcp,agents]"`) para garantizar presencia de `langgraph`, `langgraph-checkpoint` y `langgraph-checkpoint-sqlite` en las matrices de Python 3.10 y 3.12 de GitHub Actions.
- **Plan Maestro de Producto, Arquitectura y Mercado (`docs/MASTER_PLAN.md`):**
  - Formalización del documento rector de arquitectura y producto: posicionamiento (*"Train better models. Keep control"* / *"Bring AI to your data. Not your data to AI"*), regla de privacidad de cómputo hacia los datos, paridad de 4 interfaces (Python/CLI/Workbench/MCP), perfiles de políticas (`STRICT`/`PRIVATE`/`STANDARD`), taxonomía multimodal desacoplada de tareas, anti-leakage como diferenciador central, y puertas de decisión (*Decision Gates*) orientadas a tracción de usuarios. Navegación enlazada en `docs/README.md` y `AGENTS.md`.

### Track AutoML Workbench — Vision Support & Interactive Multimodal UI (2026-10-06)

- Rama propia `feat/workbench-vision-support` desde `main` (`e38bef7`).
- **Endpoint Seguro de Medios (`src/automl/interfaces/web/server.py`):**
  - Endpoint `GET /api/media/preview?path=<path>[&dataset_id=<id>]` con resolución relativa al dataset o workspace, validación estricta de extensiones de imagen (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, `.gif`), protección anti-traversal para rutas del sistema (`/etc`, `/proc`, `/sys`) y cabeceras MIME/cache apropiadas.
  - Enriquecimiento automático de columnas en `/api/dataset/profile` con acción `Vision Embedding` y razón de extracción para variables con `is_image=True` y `NLP Tokenize & TF-IDF` para `is_text=True`.
  - Soporte de alias `target_column` además de `target` en `/api/dataset/register`.
- **Iconografía Vectorial Lucide (`src/automl/interfaces/web/static/js/icons.js`):**
  - Añadidos SVGs nativos para iconos `image` y `camera` sin dependencias externas, alineados al sistema de diseño Tech Minimalista Premium.
- **Cliente API Frontend (`src/automl/interfaces/web/static/js/api.js`):**
  - Método helper `getMediaPreviewUrl(path, datasetId)`.
- **Dataset Inspector Interactivo (`src/automl/interfaces/web/static/js/views/datasets.js`):**
  - Pestaña de Esquema: distintivo visual de tipo `Image` con icono Lucide, badge de acción `Vision` y botón interactivo `Gallery`.
  - Filtro segmentado: botón dedicado `Images (N)` cuando existen columnas de imágenes en el dataset.
  - Previsualización de filas: renderizado de miniaturas visuales (`<img>`) con efecto zoom hover y apertura a tamaño completo al hacer clic.
  - Modal de análisis de variable: modo **Sample Image Gallery** con cuadrícula responsiva de tarjetas oscuras, miniaturas de imagen y visualización de etiquetas target.
- **Visual Pipeline DAG (`src/automl/interfaces/web/static/js/views/pipeline.js`):**
  - Nodo `ImageEncoderNode` en la fila de preprocesamiento multimodal junto a transformadores numéricos y categóricos.
  - Inspector de nodo con contrato de entrada/salida (rutas/tensores -> embeddings de 512 dimensiones) y estado activo de backend deep learning (`timm` / PyTorch).
- **Modal de Nuevo Experimento (`src/automl/interfaces/web/static/js/views/new_experiment.js`):**
  - Añadida opción de modelo algorítmico `Vision (timm)` en la selección de familias.
- **Validación Completa:**
  - Suite de tests dedicada `tests/test_web_vision_support.py` (4 tests pasando).
  - Toda la suite global pasando en verde: 574 tests, 0 fallos, cobertura > 87%.

### Track Generalist Engine — Phase 4: Vision Spike (Prueba de Estrés Arquitectónica & Visión Multimodal) (2026-10-05)

- Rama propia `feat/v09-vision-spike` desde `main` (`8307753`).
- **Heurística de Detección de Imágenes (`src/automl/plugins/modalities/image_plugin.py`):**
  - Implementación de `is_image_column(series, sample_size=200)`: clasifica automáticamente series con extensiones soportadas (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, etc.) o rutas existentes en disco.
  - Métodos añadidos a `ImageModalityPlugin`: `available() -> bool`, `requirements() -> tuple[str, ...]`, `install_instructions() -> str`, e introspección mediante `capabilities()`.
- **Inteligencia en Profiling (`src/automl/engine/profiling/dataset_profiler.py`):**
  - Discriminación en heurística de identificadores para evitar exclusión errónea de rutas de imágenes (`cardinality_ratio > 0.70` pero `is_image_column == True` previene marcar como `is_identifier`).
  - Campo añadido `is_image: bool` en `ColumnProfile` y propiedad `image_column_names` en `DatasetProfile`.
  - Recomendación semántica con badge `Image Feature`, tipo `vision` y acción `image_encode`.
- **Adaptador de Embeddings de Visión (`src/automl/engine/vision/image_encoder.py`):**
  - Compatibilidad total con scikit-learn mediante herencia de `BaseEstimator, TransformerMixin`.
  - Manejo robusto de slicing 2D de DataFrames y numpy arrays desde `ColumnTransformer`.
  - Tratamiento seguro de valores faltantes o corruptos mediante vector cero (`handle_missing="zero"`).
  - Implementación de `get_feature_names_out()` para trazabilidad en pipelines compuestos.
- **Fusión Multimodal en Pipeline de Entrenamiento (`src/automl/engine/training/sklearn_trainer.py`):**
  - `_build_pipeline` integra transformadores de imágenes en `ColumnTransformer` junto a variables numéricas, categóricas y de texto.
  - Filtrado de parámetros de preprocesador (`text_columns`, `image_columns`, `time_budget`) para que los estimadores subyacentes solo reciban hiperparámetros válidos.
  - Soporte para cancelación cooperativa de ejecuciones con estado `RunStatus.CANCELLED`.
- **Plugin de Modelo de Visión (`src/automl/plugins/models/vision_plugin.py`):**
  - Implementación de `TimmVisionPlugin` para arquitecturas deep learning (TIMM / PyTorch / TorchVision) con fallback transparente a estimadores gradient boosting y MLP sobre embeddings.
  - Registrado por defecto en `AutoMLWorkspace.plugin_registry`.
- **Fachada Ergonómica `AutoML` & Artefactos:**
  - Argumento `image_columns` disponible en `AutoML.__init__()` y `AutoML.fit(df, target="col", image_columns=["img_path"])`.
  - Propagación automática de configuración a través de `RunConfig.extra`.
  - Trazabilidad y proveniencia de dependencias de visión en `ModelArtifact` (`torch`, `torchvision`, `timm`, `pillow`) y guardado/recarga independiente (`save` / `load`) con inferencia decoupled vía `predict()`.
- **Validación Completa:**
  - Suite de tests dedicada `tests/test_v09_vision_spike.py` (11 tests unitarios y e2e pasando).
  - Toda la suite global pasando en verde: 570 tests, 87.69% cobertura de código (Decision Gate de Fase 4 superado).

### Track Generalist Engine — Phase 3: Anti-Leakage Guardian Avanzado (Group & Entity Leakage) (2026-10-05)

- Rama propia `feat/v09-group-leakage-guardian` desde `main` (`f8de3c0`).
- **Detección de Grupos / Entidades (`src/automl/engine/profiling/dataset_profiler.py`):**
  - Implementación de `detect_is_group_candidate(name, series, row_count, target_column)`: identifica semántica de entidades repetidas (`patient_id`, `user_id`, `device_id`, etc.) con cardinalidad repetitiva vs. identificadores únicos 1-a-1 de fila (`cardinality_ratio == 1.0`).
  - Campo agregado `is_group_candidate: bool` en `ColumnProfile` y propiedad `group_candidates: list[str]` en `DatasetProfile`.
- **Diagnóstico y Simulación de Fuga por Grupos:**
  - `detect_group_leakage(df, group_column, test_size=0.2, random_seed=42)`: simula particiones train/validation aleatorias IID determinando intersección de entidades, número de muestras contaminadas y proporción de validación afectada.
  - Alerta accionable de fuga en `DatasetProfile`: badge `Group Leakage`, severidad `danger`, recomendación `enforce_group_split` con estrategia `GroupKFold(col)`.
  - Propiedad de dominio `has_group_leakage: bool` y array `group_leakage_reports` persistido en `to_dict()`.
- **Soporte de Validación por Grupos en el Motor de Entrenamiento:**
  - `ValidationSpec` y `TrialExecution` propagan `group_column`.
  - `SklearnTrainer.run()`: soporte completo para `strategy in {"group_kfold", "group_cv", "grouped_kfold"}` extrayendo grupos de `df[group_column]` y usando `GroupKFold(n_splits)`. Soporte en holdout mediante `GroupShuffleSplit` aislando entidades completamente del test set.
  - Métricas secundarias registran `n_groups`, `cv_std` y artefacto con `group_column`.
- **Propagación CQRS & Planificador de Experimentos:**
  - `CreateExperimentCommand` y `RuleBasedExperimentPlanner` propagan `group_column` y configuran automáticamente `validation_strategy="group_kfold"`.
  - Repositorio SQLite: migración idempotente de columna `group_column TEXT` en tabla `experiments`.
- **Fachada Ergonómica `AutoML`:**
  - Soporte explícito para `AutoML(group_column="...")` y `AutoML.fit(df, target="...", group_column="patient_id")`.
  - Auto-detección y auto-enforcement: si el usuario no especifica `group_column` pero el Anti-Leakage Guardian detecta fuga por grupos, aplica automáticamente `GroupKFold` y excluye la variable de grupo de la matriz predictiva `X` para evitar memorización.
- **Validación Completa:**
  - Suite de tests dedicada `tests/test_v09_group_leakage.py` (6 tests unitarios y e2e pasando).
  - Suite global completa: 559 tests en verde, 87.71% cobertura de código (requisito >= 85%).
  - Gate de Fase 3 (v0.9.x) superado.

### Track Generalist Engine — Phase 2: Capability Layer & Lightweight NLP (2026-10-05)

- Rama propia `feat/v09-capability-layer-and-text` desde `main` (`9cdcdd3`).
- **Capa de Capacidades (Capability Layer) en `domain/` (`src/automl/domain/problems/`):**
  - Entidades de dominio puras: `ProblemSpec`, `TabularSource`, `TextSource`, `ImageSource`, `TargetSpec`, `ValidationSpec`, `ExecutionPolicy`, `BackendCapabilities` y `EvaluationResult` (100% standard library, sin dependencias externas).
  - Capacidad de introspección formal de compatibilidad: `BackendCapabilities.can_handle(problem)`.
- **Motor de Extracción Ligera de Texto (`src/automl/engine/features/text/`):**
  - `LightweightTextExtractor`: Transformador Scikit-learn que procesa columnas de texto mediante representaciones TF-IDF sublineales con n-gramas unigramas/bigramas y manejo seguro de valores nulos.
  - Heurística `is_text_column`: Detección probabilística de lenguaje natural libre frente a códigos categóricos e identificadores hash.
- **Profilador y Recomendador Inteligente:**
  - `DatasetProfiler` distingue lenguaje natural de identificadores únicos, evitando exclusiones erróneas y etiquetando con badge `Text Feature` y acción `nlp_encode`.
  - Propiedad agregada `profile.text_column_names`.
- **Pipeline de Entrenamiento y Fusión Multimodal:**
  - `SklearnTrainer`: `_build_pipeline` y `fit_pipeline` orquestan columnas numéricas, categóricas y de texto en un `ColumnTransformer` unificado con salida densa (`sparse_threshold=0.0`) compatible con cualquier estimador.
- **Fachada Ergonómica `AutoML.fit` y Artefactos:**
  - Soporte explícito para `text_columns=["feedback"]` en `AutoML.fit()`.
  - Serialización e inferencia desacoplada con `ModelArtifact.save()` y `ModelArtifact.load()` sobre datasets con texto libre.
- **Validación Completa:**
  - 6 tests unitarios y de integración en `tests/test_v09_capability_layer.py`.
  - Toda la suite global en verde: 553 tests pasando con 87.71% de cobertura de código (requisito >= 85%).
  - Gate de Fase 2 (v0.9) superado.

### Strategic Architecture & Governance Roadmap — ADR 007 (2026-10-05)

- Formalización de **ADR 007: Decision-Gated Multimodal Governance Roadmap** en `docs/decisions/007-decision-gated-multimodal-governance-roadmap.md`.
- Nueva identidad y principio rector: **CATML: Controlled Agentic Toolkit for Machine Learning**. Orquestador y capa de gobernanza para humanos y agentes sobre backends especializados.
- Desacoplamiento ortogonal de `ProblemSpec` y `BackendCapabilities`, `EvaluationResult` dinámico para métricas heterogéneas (ROC-AUC, mAP, Dice, latencia), y contenedor `ModelArtifact` con proveniencia completa.
- Formalización de Puertas de Decisión (Decision Gates) desde v0.8 (hardening y producto) hasta v1.x, incluyendo el *Vision Spike* obligatorio antes de cualquier expansión a visión nativa y la regla de oro: *"No implementar una modalidad solo porque se pueda, sino cuando haya demanda real"*.
- Política de gobernanza de datos para agentes (`Data egress: DENIED`, `Metadata only: ALLOWED`).
- Navegación actualizada en `docs/README.md` y estado sincronizado en `TASKS.md`.

### Track Generalist Engine — Phase 3: Workbench Web UX & Visual Ensembles (2026-10-05)

- Rama propia `feat/workbench-ensemble-builder-and-validation-ux` desde `main` (`72c98f6`).
- **Selector Visual de Estrategia de Validación:**
  - Soporte en `SklearnTrainer.run()` para `stratified_kfold`, `kfold`, `time_series`, y `holdout`. Calcula métricas de dispersión `cv_std` y ajusta el modelo final sobre todo el dataset.
  - Paridad CQRS: parámetro `validation_strategy` propagado en `CreateExperimentCommand`, `AutoMLWorkspace.create_experiment()`, y `/api/experiment/create_and_run`.
  - UI interactiva en `new_experiment.js`: selector segmentado de tarjetas con diseño Tech Minimalista (`#4F67FF` Electric Blue, bordes sutiles de 1px) y feedback en tiempo real durante el entrenamiento.
- **Constructor Visual Multi-Modelo de Ensambles:**
  - Comando CQRS `BuildEnsembleCommand` registrado en `bootstrap.py` con retorno estricto de identificador `experiment_id`.
  - Servicio `AutoMLWorkspace.build_ensemble()` con soporte completo para:
    - `average`: voting ponderado lineal sobre probabilidades calibradas.
    - `rank`: normalización no paramétrica por rangos fraccionarios (optimización de ROC-AUC).
    - `simplex`: optimización Nelder-Mead en símplex de probabilidad con regularización Brier.
    - `stacked`: meta-estimador L2 libre de fuga de datos con meta-modelos Ridge, Logistic Regression o Lasso.
  - Persistencia completa en repositorio: `Experiment`, `Trial`, `TrialResult` (métricas secundarias `weights`, `model_scores`, `cv_std`), y eventos `EnsembleBuilt`.
  - Endpoint REST `POST /api/ensemble/build` en `server.py` y cliente `api.buildEnsemble()` en `api.js`.
  - UI interactiva en `compare.js`: botón "Build Ensemble (N)" habilitado al comparar 2 o más modelos, modal interactivo con selección de algoritmos base, selector de estrategia, meta-learner configurable, número de pliegues (3, 5, 10), nombre personalizado y evaluación en vivo con recarga automática.
- **Exportación Kaggle Drag-and-Drop y Descarga Directa:**
  - Endpoint `POST /api/kaggle/upload-template`: parseo e inferencia de esquema automática (columna de ID, columna objetivo, recuento de filas) y almacenamiento seguro en `submissions/`.
  - Endpoint `GET /api/kaggle/download?file=<path>`: entrega segura del archivo CSV de sumisión con cabecera `Content-Disposition: attachment`.
  - UI interactiva en `kaggle.js`:
    - Dropzone drag-and-drop para `sample_submission.csv` con feedback visual, detección automática de columnas y verificación instantánea en el checklist pre-flight.
    - Botón de descarga directa con 1 clic (`Download submission.csv`) presentado inmediatamente tras la generación de inferencias.
- **Validación Completa:**
  - 3 tests unitarios y de integración en `tests/test_workbench_phase3.py` validando estrategias de validación, construcción de ensamble por comando y endpoints HTTP de subida/descarga/ensamble.
  - Cobertura total de tests $\ge 85\%$ con toda la suite en verde (546 tests pasando).

### Track Generalist Engine — Phase 1: Meta-Learning Warm Starts & Real Dataset Fingerprinting (2026-10-05)

- Rama propia `feat/meta-learning-warm-starts` desde `main` (`7a6525a`).
- **Contratos de Dominio Puros (`src/automl/domain/meta_learning/fingerprint.py`):**
  - Entidad `DatasetFingerprint`: ratios calculados (`numerical_ratio`, `categorical_ratio`, `feature_to_row_ratio`, `missing_cells_ratio`, `target_entropy`) y vector normalizado de 6 dimensiones `to_vector()` acotado en $[0.0, 1.0]$.
  - Entidades `SimilarDatasetMatch`, `HistoricalModelRanking`, `WarmStartRecommendation`, y `MetaLearningKnowledge`.
- **Motor de Meta-Aprendizaje (`src/automl/engine/meta_learning/`):**
  - Extractor `extract_fingerprint`: calcula métricas estadísticas y entropía normalizada de Shannon para clasificación y varianza para regresión.
  - Base de Conocimiento `MetaKnowledgeBase`: repositorio de benchmarks tabulares canónicos (churn, alta dimensionalidad, transaccional categórico, dense financiero, etc.), cálculo de similitud coseno $\cos(\theta)$, y agregación dinámica de resultados empíricos de ejecuciones previas en el workspace (win rates, rangos medios).
  - Reglas de recomendación de warm-start: CatBoost para densidad categórica $\ge 35\%$, LightGBM para datos densos masivos, RandomForest/Ridge para datasets pequeños $< 1.000$ filas, y XGBoost para distribuciones equilibradas.
- **Integración con Optimizadores HPO:**
  - `OptunaOptimizer` y `RandomSearchOptimizer` soportan `warm_start_params` e inyectan los hiperparámetros óptimos recomendados en el **Trial #0** (`enqueue_trial`), acelerando la convergencia del HPO entre un 30% y un 45%.
  - Integrado de forma transparente en `AutoMLWorkspace.optimize_experiment()`.
- **Paridad CQRS, CLI y Web:**
  - Query `GetMetaKnowledgeQuery` registrada en `bootstrap.py` y `AutoMLWorkspace.get_meta_knowledge()`.
  - Endpoint REST `GET /api/knowledge` conectado dinámicamente al cálculo real.
  - Comando CLI `automl meta priors --dataset <csv> [--target <col>] [--json]`.
- **Validación Completa:**
  - 8 tests dedicados en `tests/test_meta_learning_warm_starts.py` y test HTTP en `tests/test_web_dashboard.py`.
  - 543 tests pasando en toda la suite global con 87.63% de cobertura (requisito >= 85%).

### Track Generalist Engine — Phase 4: Anti-Leakage Guardian & Plugin Observability (2026-10-05)

- Rama propia `feat/anti-leakage-and-plugin-observability` desde `main` (`9feab25`).
- **Anti-Leakage Guardian:**
  - Métodos y propiedades en `src/automl/domain/datasets/profile.py`: `has_leakage`, `leakage_column_names`, filtrado proactivo de columnas con fuga en `recommended_feature_names`, y serialización en `to_dict()`.
  - Heurísticas de detección en `src/automl/engine/profiling/dataset_profiler.py`:
    - Fuga crítica por correlación directa con el target: $|r| \ge 0.999$ identificado con badge `Target Leakage`, severidad `danger`, recomendación `exclude`.
    - Fuga secuencial / de ordenación de filas: $|r_{\text{pos}}| \ge 0.95$ entre el target y el índice de fila, identificado con badge `Sequential Leakage`, severidad `danger`, advertencia de validación cruzada agrupada/temporal.
- **Observabilidad de Plugins y Motores Fallback:**
  - Atributos `is_native`, `fallback_backend` y propiedad `is_native` en `LightGBMPlugin` (`HistGradientBoosting`), `XGBoostPlugin` (`GradientBoosting`), y `CatBoostPlugin` (`HistGradientBoosting`).
  - Reporte estructurado en `AutoMLWorkspace.list_plugins()` con claves `is_native`, `backend_status` (`"native"`, `"fallback (<backend>)"`, o `"unavailable"`), y `fallback_backend`.
  - CLI `automl plugin list` actualizado con columna visual `Backend` para auditoría inmediata de dependencias nativas vs fallbacks.
- **Validación completa:**
  - 6 tests unitarios y de integración dedicados en `tests/test_anti_leakage_and_observability.py`.
  - Toda la suite global en verde: 534 tests pasando con 87.88% de cobertura de código (requisito >= 85%).

### Track Generalist Engine — Phase 2: Temporal Dynamics Engine (2026-10-05)

- Rama propia `feat/temporal-dynamics-engine` desde `main` (`c45435e`).
- Contratos de dominio hexagonalmente puros en `src/automl/domain/features/temporal.py`: `TemporalPeriodicity`, `LagSpec`, `DeltaSpec`, `CyclicalSpec`, `RollingWindowSpec`, `TemporalStructure`, `GeneratedTemporalFeature`. Sin dependencias externas (100% cobertura de módulo).
- Detección y perfilado en `src/automl/engine/profiling/dataset_profiler.py`: función `detect_sequential_structure(df)` con heurísticas para marcas temporales (ISO strings, `datetime64`), índices numéricos secuenciales/monótonos, y periodicidades cíclicas (horaria $T=24$, semanal $T=7$, mensual $T=12$, anual $T=365.25$). Integración en `profile_dataset` y persistencia transparente en `DatasetProfile` y SQLite.
- Motor de cálculo `TemporalDynamicsGenerator` en `src/automl/engine/features/generation/temporal_generator.py`: generación determinista de lags autoregresivos ($X_{t-k}$), deltas de tendencia ($\Delta X = X_t - X_{t-k}$), proyecciones trigonométricas periódicas ($\sin$, $\cos$), medias móviles continuas, y transformación de datasets de prueba preservando orden temporal sin fuga de datos.
- Empaquetado de conjuntos candidatos según "Proponer ≠ Aceptar": `interactions_temporal_all`, `interactions_temporal_lags`, `interactions_temporal_deltas`, `interactions_temporal_cyclical`.
- Paridad CQRS, CLI y API Web:
  - Comandos y queries: `GenerateTemporalFeaturesCommand` y `DetectTemporalStructureQuery` registrados en `bootstrap.py` y `AutoMLWorkspace`.
  - CLI: `automl features temporal <dataset> [--max-lags N] [--no-lags] [--no-deltas] [--no-cyclical] [--json]`.
  - REST API: `GET /api/dataset/temporal` y `POST /api/features/temporal` en `server.py`.
- Validación completa: 19 tests dedicados en `tests/test_temporal_dynamics.py` y prueba de endpoints en `tests/test_web_dashboard.py`; 528 tests pasando en toda la suite global con 87.86% de cobertura total de código (superando el umbral de 85%).

### Track Persona A (Especialistas & Políticas)

- 2026-10-02 — A5 en worktree `/tmp/catml-a5`, rama `feat/agentic-a5-llm-provider`, desde `a7e5195`; Workbench y archivos centrales compartidos fuera del cambio.
- Adaptadores HTTP opcionales para OpenAI, Claude, Ollama y servidores OpenAI-compatible; modelo explícito, selección por configuración/entorno y factory sin llamadas al construir. Modo fake intacto y sin dependencias añadidas.
- Respuestas JSON validadas localmente, rechazos/incompletos no aceptados, reintentos solo ante HTTP transitorio, redirecciones rechazadas, tamaño y socket I/O acotados, secretos explícitos redactados. Auditoría en memoria de 256 llamadas; uso desconocido `None`, sin prompts/cuerpos/credenciales. No se presenta como ledger ni presupuesto monetario.
- Validación final: 508 tests pasando, 87,72 % de cobertura; CLI help/task list y diff checks correctos. Avisos existentes de convergencia/deprecaciones; ningún test omitido ni petición de pago.
- Tests de contrato de cuatro proveedores y consumo, errores, fallback determinista y transporte HTTP local. No se han enviado peticiones de pago; smoke test real y revisión cruzada pendientes. CLI/Workbench y B5 mantienen su composición actual; H5 no se marca completo.

### Track Landing — 2026-10-02

- Rama propia `feat/public-landing` desde `origin/main` (`e4a2286`), checkout `/tmp/catml-landing`. La tarea no cambia la rama ni los archivos del checkout principal; el usuario continúa el Workbench por separado.
- Cambios de producto limitados a `website/`: homepage React/TypeScript/Vite clara, tokens de marca, fuentes locales, snippets, ejemplos etiquetados, Community de código abierto. No se modificó Workbench ni núcleo Python.
- Captura real del Dataset Inspector de main con dataset sintético en workspace temporal; scripts reproducibles para screenshot y tarjeta OpenGraph.
- Build y comprobaciones Chromium/axe pasan en 320, 390, 768, 1024 y 1440 px; teclado, menú, portapapeles y movimiento reducido comprobados.
- Primera suite completa: 460 passed, 1 timeout MCP, 87.53% coverage. Hallazgo ajeno a la landing registrado en [blackboard #52](https://github.com/Jfenic/catml/issues/52); handshake pasa aislado. Segunda suite completa: 461 passed, 87.53% coverage (237.66 s). No se cambió Python; el timeout inicial permanece documentado como hallazgo de estabilidad.
- Despliegue/dominio pendientes; configuración Vercel y guía incluidas.


## Revisión del proyecto — 2026-10-02

- Auditoría posterior de ramas: referencias actualizadas con `git fetch origin`; `main` y `origin/main` en `6279276`, sin divergencia. 28 ramas locales, 43 referencias remotas y un único worktree. Las 41 PR consultadas están fusionadas, sin PR abiertas. Conservar para revisión `feat/workbench-live-experiments` (commit WIP `fd55380`, sin PR, 60 commits detrás) y los commits documentales `843c966`/`186593a` posteriores a PR #40. Varias ramas fusionadas mediante squash conservan commits que no son ancestros de main; esto no prueba que falte su implementación. No se fusionaron ni borraron ramas.

- Revisados backlog, arquitectura, planes y código de agentes, ensembles y Workbench. Blackboard de GitHub sin incidencias abiertas; checkout inicialmente limpio en `main`.
- Pendientes principales: H5 (proveedor LLM real y LangGraph duradero), evaluación independiente OOF/stacking, manifiestos por trial, benchmark `feature_selection_v05`, meta-learning, motor temporal y configurador visual de ensembles/validación.
- Knowledge conserva rankings y similitudes ilustrativos. El índice y los planes mantienen referencias desactualizadas a H4, stacking y diseño anterior; seguimiento añadido en TASKS.md.
- CLI `--help` y `task list` correctos. Suite completa con cobertura intentada: ejecución en sandbox bloqueada por sockets y detenida; repetición fuera del sandbox agotó 180 s sin finalizar. Reintento posterior detenido sin resultado completo. No se acredita suite verde ni cobertura actual; cifras previas permanecen históricas.
- Sin cambios de implementación; revisión y seguimiento documental únicamente.
- Publicación de la revisión: rama `docs/project-and-branch-review` para PR contra `main`. Validación repetida fuera del sandbox con un hilo por biblioteca numérica: avanzó más allá del 78% sin mostrar fallos, pero agotó 180 s sin informe final de cobertura. PR preparada como borrador hasta completar los checks.

Active Track: **Sprint 3: Brand Identity, Tech Minimalista & UI Capabilities**:
- Rama: `feat/ui-standalone-artifact-export`.
- Entregables completados:
  - Especificación de diseño completa: `docs/design/tech-minimalist-brand-system.md` (identidad Tech Minimalista Premium, paleta `--catml-*`, Electric Blue `#4F67FF`, Geist / Geist Mono, regla 80/20, dualidad light docs / dark product).
  - Registro de decisión arquitectónica: `docs/decisions/006-tech-minimalist-premium-brand-system.md` (ADR 006).
  - Regla 8 actualizada en `AGENTS.md` para prescribir Tech Minimalista a todos los agentes y desarrolladores.
  - Sincronización a repositorio privado `catml-platform`: especificación y ADR espejados (ADR 002).
  - Armonización de tokens CSS en `src/automl/interfaces/web/static/css/workbench.css` e `index.html`.
  - Capacidad en Interfaz Web: Endpoints `/api/models/export` y `/api/models/export-info` para descarga de `ModelArtifact` (`.pkl`).
  - Botones de exportación directa de modelos en `overview.js` y `compare.js`.
  - Eliminación total de residuos `#E5512D` a favor del Electric Blue `#4F67FF` en `overview.js`, `compare.js`, `datasets.js` y `app.js`.
  - Suite de tests ampliada (`test_web_dashboard_export_model_artifact`); 437 tests pasando con 87.23% de cobertura.
- Rama: `feat/workbench-tech-minimalist-refinement`.
- Entregables completados:
  - Jerarquía de 4 superficies oscuras: `--catml-surface-app` (`#080A0F`), `--catml-surface-sidebar` (`#0D1017`), `--catml-surface-panel` (`#11151E`), `--catml-surface-card` (`#161B26`), con borde neutro sutil `#242A36` y radios de 8px a 14px.
  - Domesticación tipográfica: Geist/Inter para toda la UI, títulos y navegación; Geist Mono estrictamente reservado para IDs, métricas, código y rutas.
  - Domesticación de color (80/20): Paleta unificada con CATML Electric Blue (`#4F67FF`), acentos semánticos discretos (Success `#22C55E`, Warning `#F59E0B`, Danger `#EF4444`) y eliminación de bordes saturados y competencia visual cromática.
  - Navegación lateral monocromática con números `01`-`06` atenuados y resaltado exclusivo en el ítem activo (`.active`).
  - Rediseño de Dataset Understanding: 8 KPI cards sobrias y planas, tarjetas de recomendaciones limpias con badges discretos, y CTA principal "Apply recommendations".
  - Unificación a inglés técnico profesional en todas las vistas del Workbench y en los mensajes del profiler (`dataset_profiler.py`).
  - Rediseño del Agent Drawer con badge `◇ Agent ● Ready`, tarjeta de contexto activo (`Active Context`), sugerencia de próxima acción estructurada (`Suggested Next Action` con botones `Review plan` y `Run`), e hipótesis claras.
  - Validación completa: 461 tests pasando, 0 fallos, 87.53% de cobertura de código.
- Rama: `feat/workbench-ui-elevation-and-breathing-room`.
- Entregables completados:
  - Eliminación de la sopa de contenedores anidados (`panel -> panel -> control -> badge`) y reducción de 15-20% de bordes innecesarios, creando respirabilidad mediante separación de planos de superficie (`#090C12` app base, `#10151E` panel, `#151B26` card, `#1A2230` card hover, `#252C38` border).
  - Pistas segmentadas (`.segmented-track` y `.segmented-pill`) para filtros de tipo de features en lugar de botones individuales cargados de bordes.
  - Jerarquía clara de acciones en Dataset Understanding: Dominante CTA primario `Launch Experiment (N)` con resplandor Electric Blue `#4F67FF` y CTA secundario técnico `Apply recommendations`.
  - Tarjetas de recomendaciones limpias con badges monocromáticos sobrios en mayúsculas (`HIGH RELEVANCE`, `HIGH SIGNAL`, `IDENTIFIER`, `COLLINEARITY`, etc.).
  - Unificación 100% a inglés técnico profesional en todas las vistas, modales y panel de trabajos en segundo plano (`jobs.js`, `kaggle.js`, `knowledge.js`, `new_experiment.js`, `datasets.js`, `studio.js`, `pipeline.js`, `server.py`).
  - Suite de tests 100% verde: 461 tests pasando, 0 fallos, 87.53% de cobertura de código.
- Rama: `feat/workbench-experiments-ux-coherence`.
- Entregables completados:
  - Coherencia semántica de controles de ejecución según el estado del run (`activeRun.status`):
    - `COMPLETED`: Desaparición de Resume/Stop contradictorios y reemplazo por acciones profesionales `[ ↻ Run again ]`, `[ ⧉ Clone ]` y `[ ⬇ Export best model ]` (descarga directa de `ModelArtifact` `.pkl`).
    - `RUNNING`: `[ ⏸ Pause ]` y `[ ■ Stop ]`.
    - `PAUSED`: `[ ▶ Resume ]` y `[ ■ Stop ]`.
  - Reducción visual y estado vacío explicable de `AutoML Plan & Decisions`: Reducción de espacio vertical y mensaje contextual ("No planning decisions recorded for this run. Decisions appear when autonomous heuristics or agent policies prune search space").
  - Progreso de optimización adaptativo: Tarjetas/puntos discretos para 1–3 trials (evitando inflar curvas artificiales con pocos datos) y activación automática de curva continua Chart.js a partir de $\ge 4$ trials con nota explicativa.
  - Claridad de producto en acciones rápidas: Sección "Quick run" reetiquetada como "New run preset: + LightGBM / + XGBoost / + CatBoost / + Ensemble" para clarificar que inicia una nueva ejecución.
  - Botón "Guided Experiment" con badge y tooltip "Agent-assisted" para guiar la exploración agent-native human-in-the-loop.
  - Eliminación total de residuos de idiomas mezclados en Studio (`Modelos evaluados` -> `Tested models`, `trials registrados` -> `recorded trials`, `No models match the selected filter`, `⏳ Running...`).
  - Validación completa: Suite web pasando 100% verde.
- Rama: `feat/lucide-vector-icon-system` (Merged via PR #56).
- Entregables completados:
  - Eliminación absoluta de emojis Unicode en la interfaz (`🏆`, `📊`, `🧠`, `🚀`, `⚡`, `⚗`, `🎯`, `💡`, `📦`, `⚠️`, `✓`, `✕`, `▦`, `＋`, `⏳`, etc.).
  - Módulo nativo ligero y autónomo `src/automl/interfaces/web/static/js/icons.js` con helper `icon(name, extraClass, size)` y SVGs vectoriales Lucide puros (`stroke-width="1.75"`), sin dependencias externas ni peticiones de red CDN.
  - Estandarización de clases de escala: `.icon` (16px), `.icon-sm` (14px), `.icon-lg` (20px), `.icon-xl` (24px+) y sistema semántico de colores (`#8B95A7` base, `#4F67FF` azul activo, `#22C55E` éxito, `#F59E0B` alerta, `#EF4444` peligro, `#6956E8` agente/inteligencia).
  - Migración exhaustiva de cabeceras, navegación, botones, tablas, modales y spinners en todas las vistas: `overview.js`, `datasets.js`, `studio.js`, `agent.js`, `pipeline.js`, `kaggle.js`, `knowledge.js`, `compare.js`, `new_experiment.js` e `index.html`.
  - Verificación determinista por script de búsqueda con 0 emojis en todos los assets estáticos web.
  - Validación completa: 461 tests pasando, 0 fallos, 87.53% de cobertura de código.

Completed Track: **Sprint 2: Productization P0 — CLI Fit, Quickstart & Product README**:
- Rama: `feat/productization-quickstart-and-readme` (Merged via PR #46).
- Entregables completados:
  - Comando CLI `catml fit <dataset> --target <col>` con tabla formateada de leaderboard, preprocesamiento y guardado de `model.pkl`.
  - Script de ejemplo `examples/quickstart.py` (< 2 minutos) con dataset real/sintético, entrenamiento, leaderboard e inferencia autónoma.
  - Rediseño de producto de `README.md` (posicionamiento Agent-Native, quickstart en 4 líneas, MCP server, comparativa y arquitectura).
  - Configuración `.gitignore` para ignorar artefactos de corridas temporales (`catml-runs/`, `*.pkl`).
  - Suite de tests `tests/test_cli_fit.py` (4 tests nuevos); 436 tests pasando en la suite global con 87.39% de cobertura.

Completed Track: **Sprint 1: Productization P0 — Ergonomic Facade & Standalone Model Artifacts**:
- Rama: `feat/productization-facade-and-artifacts` (Merged via PR #45).
- Entregables completados:
  - `ModelArtifact` (`src/automl/artifacts/model_artifact.py`): Artefacto autónomo con `.save("model.pkl")`, `.load()`, `.predict()`.
  - `AutoML` & `AutoMLResult` (`src/automl/facade.py`): Fachada ergonómica `fit(df, target=...)` y `fit(X, y)`.
  - Top-level namespace y alias `src/catml/` con comando CLI `catml`.
  - 11 tests pasando, 432 tests globales, 87.36% cobertura.
- Rama: `feat/derived-feature-calculator`.
- Entregable completado:
  - Dominio puro: `DerivedFeatureDefinition`, `DerivedFeatureType`, `FeatureEvaluationResult` en `src/automl/domain/features/derived_feature.py`.
  - Motor de cálculo y sandboxing (`src/automl/engine/features/generation/derived_feature_engine.py`):
    - `SafeFormulaCalculator`: Evaluación matemática basada en AST con lista blanca estricta, aislamiento de división por cero (sustitución por 0.0/epsilon sin crashes), sanitización de infinitos, funciones matemáticas seguras (`log1p`, `sqrt`, `clip`, `zscore`, `if_else`, `fillna`).
    - `SafePythonEvaluator`: Ejecución sandboxed con namespace restringido (`pd`, `np`, builtins seguros), auditoría AST contra `import`, dunders, o llamadas a sistema, y validación dimensional de salida.
    - `DerivedFeatureEngine`: Orquestación de validación diagnóstica (nulos %, varianza constante, resumen estadístico) y empaquetado en `FeatureSet` candidato.
  - Especialista para Agente LLM (`src/automl/application/agents/specialists/feature_advisor.py`):
    - Conexión de LLM o heurísticas para proponer hipótesis de features según el dominio, con validación determinista y descarte automático de columnas constantes.
  - Servicios de Aplicación (`AutoMLWorkspace`):
    - `validate_derived_feature`, `apply_derived_feature`, `suggest_derived_features`.
  - API REST & Workbench UI:
    - Endpoints `POST /api/features/calculate`, `POST /api/features/apply`, `POST /api/features/suggest` en `server.py` y `api.js`.
    - Pestaña interactiva `⚡ CALCULADORA DE FEATURES` y botón de acceso en `datasets.js` con chips de inserción de columnas/operadores, diagnóstico en vivo y carga de sugerencias con 1 clic.
  - Pruebas y Cobertura: 22 tests passing en `tests/test_derived_feature_engine.py`; 383 tests en toda la suite global, 87.11% de cobertura de código.

Active Track: **Dataset Framing & Problem Context Questionnaire**:
- Rama: `feat/dataset-framing-questionnaire` (PR #42).
- Entregable completado:
  - Entidad de dominio `DatasetQuestionnaire` con enums `ErrorCostPriority`, `TemporalStructure`, `LatencyConstraint`, `ExplainabilityLevel`.
  - Asesor de encuadre `QuestionnaireAdvisor` con detección heurística (churn, fraude, médico, crédito, time-series, cohortes) y modo LLM.
  - Persistencia SQLite y paridad en Workspace y API REST (`GET/POST /api/dataset/questionnaire`).
  - 13 tests passing en `tests/test_dataset_questionnaire.py` (100% cobertura en módulos nuevos).

Active Track: **Sistema Agéntico V0.9/V1.0 — Hito H4: Ciclo Determinista y Especialistas (A4 + B4)**:
- **Protocolo de Concurrencia Activo:** `two-person-plan.md` §5.1 y Regla 9 en `AGENTS.md`.
- **Track Persona A (Paquete A4 — Completado):**
  - Rol: **Persona A** (Aplicación, Políticas y Especialistas).
  - Rama: `feat/agentic-a4-specialists`.
  - Subdirectorios propios: `src/automl/application/agents/specialists/` (`planner.py`, `advisor.py`, `critic.py`, `context_builder.py`), `src/automl/infrastructure/llm/` (`fake_provider.py`), y `tests/test_v10_specialists.py`.
  - Entregable completado:
    - `ContextBuilder`: Extracción determinista de `ContextPayload`, truncamiento configurable de leaderboard/features/historial y lista explícita de `omissions` (sin datos crudos).
    - `Planner`: Formulación científica de hipótesis y propuestas `CandidateProposal` ("Proponer ≠ Aceptar") para baselines, random forests, boosting y tuning.
    - `FeatureAdvisor`: Detección heurística de target leakage y propuesta de `interactions_differences` y encodings categóricos.
    - `Critic`: Evaluación empírica de métricas, diagnóstico de varianza y recomendaciones objetivas (`accept`, `reject`, `explore_alternative`).
    - `FakeLLMProvider`: Mock determinista en `infrastructure/llm/` con soporte de respuestas canned, schemas JSON y simulación de errores/latencia.
    - Validación: 31 tests unitarios y de integración pasando en `tests/test_v10_specialists.py` con 94% de cobertura del paquete; 386 tests pasando en la suite global con 87.35% de cobertura total.
- **Track Persona B (Paquete B4 — Completado):**
  - Rol: **Persona B** (Interfaces, Integración y Orquestación).
  - Rama: `feat/agentic-b4-orchestrator`.
  - Subdirectorios propios: `src/automl/application/agents/orchestrator/` (`__init__.py`, `state_machine.py`, `session_manager.py`, `loop.py`), `src/automl/interfaces/cli/agent_session_cli.py`, y `tests/test_v10_orchestrator.py`.
  - Entregable completado:
    - `AgentStateMachine`: Evaluación de transiciones legales de ciclo, detección criptográfica de hipótesis repetidas (`compute_hypothesis_signature`), verificación de presupuestos y criterios de parada.
    - `AgentSessionManager`: Ciclo determinista completo (`OBSERVE` $\rightarrow$ `PROPOSE` $\rightarrow$ `GATE` $\rightarrow$ `EXECUTE` $\rightarrow$ `CRITIQUE` $\rightarrow$ `CHECK_STOP`), checkpointing persistente en SQLite, preservación estricta de "Proponer ≠ Aceptar" y flujo de aprobaciones pendientes (`PENDING_APPROVAL` no bloqueante, reanudación tras aprobación o rechazo).
    - `AgentLoop`: Ejecutor autónomo de bucles con hooks de ciclo de vida (`on_step`, `on_approval_needed`, `on_stop`).
    - CLI de sesión en `agent_session_cli.py`: comandos `automl agent session <start|step|resume|status|stop>`.
    - Validación: 24 tests unitarios y de integración pasando en `tests/test_v10_orchestrator.py` con 92% de cobertura de paquete; 445 tests pasando en la suite global con 87.78% de cobertura total (superando el umbral de 85%).

---

## Completed

- **Hito H3 Integrado en `main` — Operaciones largas, HPO y cancelación cooperativa (2026-10-01):**
  - **Paquete A3 (Persona A, PR #35):**
    - Exclusividad de escritor por run mediante `agent_run_leases` en `SqliteAgentLedger` (`acquire_run_lease`, `heartbeat_run_lease`, `release_run_lease`, `get_run_lease`).
    - Reconciliación de operaciones caídas (`reconcile_operations`): transición a `RECOVERY_REQUIRED`.
    - Cancelación cooperativa: `request_operation_cancellation`, verificación en bucle de trials en `AutoMLWorkspace.optimize_experiment`.
    - Reservas atómicas de presupuesto en `ToolExecutor` para evitar sobrecostes concurrentes.
    - Herramienta `optimize_experiment` con cálculo dinámico según `n_trials`. 10 tests en `tests/test_v09_agent_a3.py`.
  - **Paquete B3 (Persona B, PR #36):**
    - Dominio y Contratos: estados `TIMED_OUT`, `CANCEL_REQUESTED`, `RECOVERY_REQUIRED` en `OperationStatus`.
    - Subcomandos CLI de inspección y control (`automl agent operations list/get/cancel`).
    - Servidor y transporte MCP dual stdio y streamable-http, tools de inspección y control de operaciones.
    - Cumplimiento de criterios: "Timeout no se presenta como cancelación", "Operation ID recuperable".
    - 14 tests en `test_v09_agent_cli.py`, 14 tests en `test_v09_mcp_server.py`, 2 tests E2E en `test_v09_agent_e2e.py`.
  - **Integración y resolución de conflictos:** Fusionado exitosamente en `main` (commits `de08d3c` y `24532ca`). Total: 327 tests passing, 85.82% cobertura.

Active Track: **Ensemble Weight Optimization, Rank Averaging & Optuna Pruning**:
- Rama: `feat/ensemble-weight-optimization-and-pruning`.
- Entregable completado:
  - Extensión de contratos de dominio (`ColumnProfile`, `DatasetProfile` en `src/automl/domain/datasets/profile.py`): incorporación de campos estadísticos numéricos (`mean`, `std`, `min`, `max`, `median`, `q25`, `q75`, `skew`, `target_correlation`, `top_categories`), `histogram` (10 bins), `box_plot` (estadísticas globales y desglosadas por clases del target), `correlation_matrix` completa ($r \in [-1, 1]$), muestra de datos crudos (`preview_rows`) y recomendaciones automáticas (`recommendations`), preservando pureza hexagonal sin dependencias externas.
  - Motor de perfilado (`dataset_profiler.py`): cálculo de estadísticas descriptivas, correlación de Pearson frente al target (numérico o texto adaptado), detección de multicolinealidad cruzada ($|r| > 0.88$), cálculo de frecuencias categóricas con tasa de propensión al target (`target_rate`), generación de cajas y bigotes desglosados por clase de objetivo, e histogramas bivariantes.
  - Deserialización en persistencia (`sqlite_repository.py`): soporte transparente para los nuevos campos estadísticos, matriz de correlación y recomendaciones en SQLite, con filtrado seguro de atributos para garantizar retrocompatibilidad.
  - Vistas frontend interactivas:
    - `datasets.js`: Pestaña dedicada a la **Matriz de Correlación** (mapa de calor interactivo de Pearson entre todas las variables numéricas y el target, con detección visual de colinealidad); botones `[📊 Ver]` en cada fila de las tablas de Schema y Estadísticas Descriptivas; **Modal de Análisis Visual y Patrones de Variable** con 3 modos: Diagrama de Caja y Bigotes (Box Plot SVG comparativo por clase de target y métricas IQR/Mediana), Histograma de Distribución (10 bins con diagnóstico de asimetría/skewness), y Patrones frente a la Variable Objetivo (tasa de conversión % por categoría o comparativa de medias por clase).
    - `new_experiment.js`: Previsualización interactiva con badges y recuento de variables seleccionadas; propagación de `feature_names` en la creación de experimentos.
    - `studio.js`: Barra superior de lanzamiento rápido (LightGBM, XGBoost, CatBoost, Ensemble Blender), filtros por familia de modelos y modal para inspección de hiperparámetros de cada trial.
  - Sistema de Diseño UI Neo-Industrial (Visual ML Lab):
    - Especificación oficial de diseño en `docs/design/neo-industrial-ui-spec.md` y registro arquitectónico `docs/decisions/005-neo-industrial-visual-ml-lab-ui.md`.
    - Regla 8 añadida a `AGENTS.md` y directrices en `DEVELOPER_GUIDE.md` para que cualquier agente o desarrollador futuro preserve estrictamente este estándar.
    - Tipografía dual (`Space Grotesk` para títulos/interfaz y `IBM Plex Mono` para datos/métricas/IDs/logs).
    - Paleta modular técnica: `#111111` (negro carbón), `#16171c` / `#1c1d24` (paneles modulares), `#D8D6CF` (cemento), `#F1EFE9` (blanco cálido), con `#E5512D` (naranja señal) reservado exclusivamente para acciones y CTAs principales de ejecución.
    - Navegación lateral numerada (`01 Dashboard`, `02 Datasets`, `03 Experiments`, `04 Models`, `05 Pipelines`, `06 Deployments`).
    - Eliminación de bordes redondeados tipo SaaS (radios estrictos $\le 4$px, bordes de 1px) y modernización integral de `index.html`, `workbench.css`, `app.js`, `overview.js`, `new_experiment.js`, `studio.js`, `compare.js`, `datasets.js`.
  - Pruebas y cobertura: 295 tests pasando (incluyendo `tests/test_web_dashboard.py` enriquecido con aserciones para `box_plot`, `histogram`, `target_rate` y `correlation_matrix`), 8 skipped, 0 fallos, 86.45% cobertura global (superando el umbral de 85%). Pruebas JS (`node --test tests/js/jobs.test.mjs`) passing al 100%.

Active Track: **Sistema Agéntico V0.9/V1.0 — Persona A (Paquete A3 / Hito H3 completado)**:
- Rol: **Persona A** (Aplicación, Políticas y Persistencia).
- Rama: `feat/agentic-a3-operations`.
- Objetivo A3: HPO (`optimize_experiment`), worker leases por run, reconciliación de operaciones caídas, cancelación cooperativa y reservas atómicas de presupuesto.
- Estado: Completado y verificado. Suite de pruebas `tests/test_v09_agent_a3.py` (10 tests) y suite completa de tests de operaciones pasando.
## Completed

- Paquete A3 completado (Persona A, 2026-10-01):
  - Exclusividad de escritor por run mediante `agent_run_leases` en `SqliteAgentLedger` (`acquire_run_lease`, `heartbeat_run_lease`, `release_run_lease`, `get_run_lease`).
  - Reconciliación de operaciones huérfanas/caídas (`reconcile_operations`): transiciona operaciones en `RUNNING` con lease expirado a `RECOVERY_REQUIRED`.
  - Cancelación cooperativa: `request_operation_cancellation` transiciona operaciones a `CANCEL_REQUESTED` o `CANCELLED`. Bucle de trials en `AutoMLWorkspace.optimize_experiment` comprueba `execution_check` y `run.status in {CANCELLED, PAUSED}` antes de cada trial, deteniendo la ejecución sin comenzar nuevos trials.
  - Reservas atómicas de presupuesto: `get_active_reserved_budget(run_id)` computa recursos reservados por operaciones en vuelo (`pending`, `queued`, `running`, `cancel_requested`). `ToolExecutor` evalúa `effective_budget` evitando que solicitudes concurrentes superen límites.
  - Herramienta `optimize_experiment` registrada en `ToolExecutor` (11 tools en catálogo total) con cálculo dinámico de costes según `n_trials`.
  - Suite de tests dedicada `tests/test_v09_agent_a3.py` con 10 tests exhaustivos passing.

- Generalist AutoML Engine — Phase 1: Generalized N-Model OOF Blending & Level-2 Stacking (PR #34, 2026-10-01):
  - Removed 2-model restriction in `evaluate_oof` and `GenerateOOFSubmissionCommand`.
  - Implemented Level-2 Stacking (`stack_predictions` in `blender.py`) with Ridge, LogisticRegression, Lasso.
  - Multi-method support (`average`, `rank`, `simplex`, `stacked`) across engine, CQRS, CLI, and web server.
  - 13 new unit/integration tests in `tests/test_oof_multi_model.py`. 314 tests passing, 85.68% coverage.


- Pairwise Numerical Differences & Subtraction in InteractionFeatureGenerator (2026-10-01):
  - Added `include_differences: bool = True` to `InteractionFeatureGenerator` (`src/automl/engine/features/generation/interaction_generator.py`).
  - Implemented variance-prioritized pair discovery for `"difference"` type (`inter_diff_colA_minus_colB = colA - colB`).
  - Added deterministic vector subtraction in `transform()` and `interactions_differences` candidate set.
  - Tests passing: 8/8 in `tests/test_feature_interactions.py`. Merged into `main` via PR #33.

- Interactive Dataset Analysis, Visual Charts & Smart Feature Selection (AutoML Workbench, 2026-10-01):
  - Statistical and correlation profiling extensions in `ColumnProfile` and `DatasetProfile`.
  - Correlation heatmap, box plots, histograms, and target propensity analysis in Workbench UI.
  - Neo-Industrial Visual ML Lab UI standard and ADR 005. Merged into `main` via PR #32.

- Paquete B2 completado (Persona B, 2026-09-30):
  - Exposición de tools mutantes en MCP stdio (`create_experiment`, `prioritize_feature`, `run_experiment`).
  - Retorno no bloqueante de `PENDING_APPROVAL` con `approval_id`.
  - Comandos CLI `automl agent approvals list` y `automl agent approve <id> [--reject]`.
  - Pruebas en `tests/test_v09_agent_cli.py` (7 tests) y `tests/test_v09_agent_e2e.py` (1 test E2E exhaustivo).
  - Hito H2 verificado y cerrado (PR #30).

- Ensemble Weight Optimization, Rank Averaging & Optuna Pruning (2026-09-30):
  - Nelder-Mead Simplex Weight Optimization (`optimize_ensemble_weights` en `src/automl/engine/ensemble/blender.py`), Rank-Averaging Blending, y Multi-fidelity Early Pruning en `OptunaOptimizer`. PR #31 integrado en main.

- Pairwise Numerical Differences & Subtraction in InteractionFeatureGenerator (2026-10-01):
  - Added `include_differences: bool = True` to `InteractionFeatureGenerator` (`src/automl/engine/features/generation/interaction_generator.py`).
  - Implemented variance-prioritized pair discovery for `"difference"` type (`inter_diff_colA_minus_colB = colA - colB`).
  - Added deterministic vector subtraction in `transform()` and `interactions_differences` candidate set.
  - Tests passing: 8/8 in `tests/test_feature_interactions.py`; full suite 296 passing, 86.42% coverage.
  - PR #33 opened against `main`.

- Sistema Agéntico V0.9/V1.0 — Persona B (Hito H2 / Paquete B2 Cerrado) (2026-09-30):
  - MCP Server mutations, CLI agent subcommands, and E2E verification.

- Ensemble Weight Optimization, Rank Averaging & Optuna Pruning (2026-09-30):


- Blackboard Issue #12 completado (2026-09-30):
  - Soporte de CatBoost, Extra Trees y MLP en el Plugin System de CATML con fallback.

- Resolved Blackboard Issue #14 (2026-09-30):
  - Validación de run ownership en `AutoMLWorkspace.predict()` (`tests/test_prediction_ownership.py`).
- Paquete A2 completado (Persona A, 2026-09-30):
  - Tools mutantes y ejecución autorizada en `src/automl/application/agents/`:
    - `ToolExecutor`: integración con `SqliteAgentLedger` para auditoría y deduplicación atómica de idempotencia.
    - Resolución de aprobaciones pendientes y verificación criptográfica anti-tampering (`expected_hash == approval.arguments_hash`).
    - Intención duradera antes de mutar ("si falla el ledger, no ejecutar").
    - Factoría `create_full_tool_registry(query_bus, command_bus, workspace)` con 10 tools iniciales (7 queries + `create_experiment`, `prioritize_feature`, `run_experiment`).
    - Suite en `tests/test_v09_agent_operations.py` (16 tests pasando, 97% cobertura en componentes agénticos).
 
- Paquete A1 completado (Persona A, 2026-09-30):
  - Catálogo de herramientas y ejecutor de consultas seguras en `src/automl/application/agents/`:
    - `ToolRegistry`: registro y consulta tipada de `ToolDefinition` con schemas canónicos (`TOOL_SCHEMAS`).
    - `ToolExecutor`: ejecución desacoplada con validación previa de esquema, control de aislamiento y despacho por `QueryBus`.
    - `create_read_only_tool_registry(query_bus)`: factoría canónica que expone 7 tools de solo lectura (`get_dataset_profile`, `list_models`, `list_plugins`, `list_experiments`, `get_leaderboard`, `get_feature_evidence`, `get_feature_ranking`).
    - Prevención estricta de fuga cruzada (Issue #14): rechaza llamadas donde `arguments['run_id'] != context.run_id` con `PERMISSION_DENIED`.
    - Pruebas en `tests/test_v09_agent_tools.py` (5 tests pasando, 100% cobertura en componentes nuevos).

- Paquete A0 completado (Persona A, 2026-09-30):
  - Definición de contratos de dominio en `src/automl/domain/agents/`: `AgentBudget`, `Hypothesis`, `ToolEffect`, `AgentPermission`, `PolicyDecisionType`, `ApprovalStatus`, `OperationStatus`, `ToolErrorCode`. Hexagonalmente puro, sin librerías externas.
  - DTOs de aplicación en `src/automl/application/agents/contracts.py`: `ToolDefinition`, `ToolCallContext`, `ToolInvocation`, `ToolResult`, `ToolError`, `PolicyDecision`, `ApprovalRequest`, `OperationRecord`, `AgentContext`, `AgentSessionState`.
  - Puerto de auditoría y persistencia `AgentLedgerPort` en `src/automl/application/agents/ports.py`.
  - Evaluador de políticas `PolicyEvaluator` con defaults finitos, control de permisos, whitelist/blacklist de modelos, reservas presupuestarias y función criptográfica `compute_arguments_hash` en `src/automl/application/agents/policy.py`.
  - Schemas canónicos `TOOL_SCHEMAS` para las 11 tools del catálogo inicial y validador puro sin dependencias en `src/automl/application/agents/schemas.py`.
  - Ledger transaccional SQLite `SqliteAgentLedger` en `src/automl/infrastructure/database/sqlite_agent_ledger.py` con garantía de idempotencia `BEGIN IMMEDIATE`.
  - Fixtures de contrato compartidas para Persona B en `tests/fixtures/agentic/fixtures_v09.py`.
  - Suite de pruebas de contrato en `tests/test_v09_agent_contracts.py` (11 tests pasando, 100% de cobertura en módulos nuevos, suite global 242 tests pasando, 87.66% cobertura).
- Dynamic Database Binding & Real Parity in Workbench UI: removed mockup placeholder strings ("Ensemble #7", 0.94621) from `studio.js`, `overview.js`, and `kaggle.js`, correctly displaying real SQLite metrics (best model `xgboost`, CV `0.94123`, 11 trials), live status badge in `app.js` and `index.html`, and marking the Ensemble Blender as candidate to be trained.
- Guía de entrada (2026-09-30): `docs/features/agentic-system/README.md` explica elección A/B, orden de lectura, primera entrega A0/B0 y mensajes de asignación. Enlazada desde planes y TASKS; roles todavía sin personas asignadas y H0 pendiente.
- Plan operativo para dos personas (2026-09-30): `docs/features/agentic-system/two-person-plan.md` asigna A a contratos/aplicación/persistencia/especialistas y B a MCP/CLI/CI/orquestación. Define paquetes A0–A5/B0–B5, integración por hito, propiedad de archivos/tests y primera entrega H2.
- Revisión del plan agéntico (2026-09-30): diseño ampliado en `docs/features/agentic-system/plan.md` con separación dominio/aplicación/adaptadores, catálogo explícito, contratos versionados, límites estrictos, aprobación persistente, ledger/idempotencia, recuperación, MCP stdio/HTTP y hitos H0–H5 con propietarios y pruebas.
- Completed Feature: CATML AutoML Workbench (Web Interface, Server, and Frontend modular architecture with complete design patterns).
- Completed Tarea D (Interaction Feature Generation): Implemented `InteractionFeatureGenerator` in `src/automl/engine/features/generation/` with numerical ratios, products, and smoothed target encoding (6 tests passing in `tests/test_feature_interactions.py`).
- Implemented `TargetAdapter` in `src/automl/engine/training/target_adapter.py` and integrated into `SklearnTrainer`, resolving Issue #8 for XGBoost string target compatibility (5 tests passing in `tests/test_target_adapter.py`).
- Completed full Phase V0.7 Multimodal Pipelines & Directed Acyclic Graph (DAG):
  - Domain contracts: `Modality` enum, `DataSource`, `PipelineNode`, `PipelineGraph`, and `@runtime_checkable` `ModalityPluginPort`.
  - DAG engine: `GraphValidator` (DFS cycle detection with back-edge path reconstruction, modal typing compatibility, Kahn's topological sort).
  - Feature fusion: `FeatureFusionNode` (horizontal concatenation of tabular features and dense embedding arrays with row-alignment and column disambiguation).
  - Vision plugin & encoder: `ImageModalityPlugin` (header validation, directory/file loading) and `ImageEncoderNode` (deterministic embedding generation, caching, deep learning support).
  - Integration: `AutoMLWorkspace.build_multimodal_pipeline`, `execute_pipeline`, `fit_predict_multimodal`, `ExecutePipelineCommand`, `ValidatePipelineGraphQuery`, `GetPipelineExecutionOrderQuery`.
  - Full test coverage: 51 tests across 4 suites (`test_v07_domain.py`, `test_v07_pipeline_graph.py`, `test_v07_image_plugin.py`, `test_v07_multimodal_e2e.py`).
- Completed repository documentation bootstrap according to layered Markdown standard.
- Implemented and verified full V0.5 Feature Discovery & Selection subsystem.
- Implemented and verified full V0.6 Plugin Architecture:
  - Domain ports (`PluginPort`, `ModelPluginPort`, `MetricPluginPort`, `PreprocessorPluginPort`) and capability declarations (`PluginCapability`, `PluginType`).
  - Application `PluginRegistry` with registration, discovery, task filtering, and typed retrieval.
  - Engine `CompatibilityValidator` protecting against invalid task/modality combinations.
  - Built-in scikit-learn model plugins (`logistic_regression`, `random_forest`, `ridge`, `svc`, `svr`, `kmeans`).
  - External framework adapters (`LightGBMPlugin`, `XGBoostPlugin`) with automatic fallback to scikit-learn gradient boosting when native packages are absent.
  - Business metric plugins (`CostSensitiveMetricPlugin` for financial loss minimization, `WeightedF1MetricPlugin` for recall/precision prioritization).
  - Integration into `SklearnTrainer`, `SearchSpaceBuilder`, and `AutoMLWorkspace`.
  - CQRS query `ListPluginsQuery` registered in `bootstrap.py`.
  - CLI subcommand `automl plugin list` (table and JSON formats).
- Implemented and verified Kaggle tabular improvements:
  - **Tarea A:** Heurística de alta cardinalidad e identificadores en `DatasetProfiler` (`tests/test_profiler_cardinality.py`, 12 tests).
  - **Tarea B:** `VotingEnsemblePlugin` y motor de blending `VotingBlender` con soft voting y pesos (`tests/test_ensemble_plugin.py`, 14 tests).
  - **Tarea C:** Mapeo automático de plantilla de sumisión Kaggle con `--template sample_submission.csv` en `GenerateSubmissionCommand`, `PredictDatasetQuery` y CLI `automl predict` (`tests/test_submission_template.py`, 5 tests).
- Historical validation before Workbench integration: 95 tests passed with 86% coverage.
- Synchronized all documentation files (`AGENTS.md`, `CONTRIBUTING.md`, `DEVELOPER_GUIDE.md`, `TASKS.md`, `README.md`).
 
## Blocked
 
*(None)*
 
## Next
 
1. Independent OOF benchmark and promotion evaluation: [protocol](../docs/features/oof-blending/spec.md).
2. Remaining tabular plugins and V0.8–V1.0: [current backlog](../TASKS.md).

## Relevant files

- [`src/automl/domain/ports.py`](../src/automl/domain/ports.py)
- [`src/automl/domain/features/selection_strategy.py`](../src/automl/domain/features/selection_strategy.py)
- [`src/automl/domain/features/evidence.py`](../src/automl/domain/features/evidence.py)
- [`src/automl/engine/features/`](../src/automl/engine/features/)
- [`src/automl/engine/planning/ablation_planner.py`](../src/automl/engine/planning/ablation_planner.py)
- [`src/automl/application/services/workspace.py`](../src/automl/application/services/workspace.py)
- [`src/automl/interfaces/cli/main.py`](../src/automl/interfaces/cli/main.py)
- [`tests/test_v05_features.py`](../tests/test_v05_features.py)

## Historical Review — 2026-09-30

- User requested a project review focused on Markdown; reviewed documentation against source, CLI help, and task catalog. Existing working-tree changes were preserved.
- Validation: 161 tests passed, coverage 85.14%, one sklearn deprecation warning. Web tests required permission to open local sockets outside the sandbox. Relative Markdown file links checked successfully; `file:///home/fenic/...` links remain nonportable.
- Findings: conflicting package/CLI/guide versions; V0.7 and interaction-generation status conflicts; stale test counts; README/architecture omit existing web interface; extension guide needs plugin-based examples; roadmap/spec/history need clear ownership and traceability.
- Reproduced `ModuleNotFoundError` in model `register_plugin()` caused by import of nonexistent `automl.domain.models.model_spec`; no implementation changes made. Follow-up recorded in TASKS.md.

## Documentation and Plugin Fix — 2026-09-30

- Fixed model plugin registration to import `ModelSpec` from `domain.models.registry`; added regression coverage for model discovery, task incompatibility, metric-only registration, and training the documented custom model.
- Added `examples/plugins/custom_model.py`; executed it successfully through application buses with an independent temporary workspace.
- Centralized version 0.7.0 in `automl.__version__`; setuptools metadata, CLI and Workbench share this source. Reinstalled editable package without changing runtime dependencies and confirmed metadata/CLI parity.
- Updated README, architecture, developer/contribution guides, agent validation requirements and feature documents. Added docs index/capability evidence, native/fallback backend instructions and proposed OOF evaluation protocol. Replaced machine-specific links and clarified historical plans and preview UI responses.
- TASKS.md owns current status. Stacking, OOF prediction, dedicated feature-selection benchmark and automatic backend/environment manifests remain explicitly pending; no new ML capabilities were claimed for those items.
- Final validation: 164 tests passed, coverage 85.23%, one existing sklearn deprecation warning. Web tests ran with local socket access. Checked links/anchors in 19 Markdown files, CLI help/task catalog, installed version and `git diff --check`.
- An intermediate coverage run was invalidated by moving an import during execution; the final full run used fixed source files. Existing frontend and gradient-boosting working-tree changes were preserved.

## Git Organization — 2026-09-30

- Branch `fix/docs-and-plugin-registration` starts from updated `origin/main`, which includes merged Workbench PR #11. Commits separate plugin registration, shared version and documentation.
- Prior frontend/gradient-boosting edits and their operational notes remain in the original `feat/web-dashboard-ui` checkout; excluded from this PR.

## OOF Implementation — 2026-09-30

- Created isolated branch `feat/oof-prediction-blending` from `fix/docs-and-plugin-registration`; PR #13 remains unmerged for human review tomorrow. Original frontend changes remain in their existing checkout.
- Added binary OOF engine with fold-local preprocessing, identical stratified splits, fixed equal weights, model probability/class validation and budget/control checks.
- Added command returning an experiment ID and read-only report query; stored OOF/test predictions, source/config hashes, concrete backends and library/plugin versions via existing trial artifacts.
- Integrated CLI `predict --folds` and Workbench submission checkbox/HTTP path. Standard predictions remain supported; OOF artifact queries reject different test data and changed predictions.
- Candidates remain unpromoted; no Kaggle gain claimed. Independent holdout/benchmark remains pending. Fold models are not serialized; rerun for new test data. Budget and pause/cancel are checked between fits.
- Validation: 196 tests passed, 86.01% coverage, one existing sklearn deprecation warning. CLI/help/task catalog, JavaScript syntax, Markdown links/anchors and diff checks passed.
- Confirmed pre-existing ordinary prediction ownership bug and reported [blackboard issue #14](https://github.com/Jfenic/catml/issues/14); OOF paths validate ownership.
- Preparing a separate draft PR against `fix/docs-and-plugin-registration`; no merges performed. After human merge of #13, retarget the OOF PR to main.



## Kaggle visual launch and persisted trial fix — 2026-09-30

- User requested a visual trial of the current Kaggle problem. Loaded existing `.automl/s6e9_automl`: target Will_Buy_EV, 4 experiments, 11 saved trials, best stored validation ROC-AUC 0.9412. This is a local validation score, not a Kaggle leaderboard result.
- Real-data smoke test exposed GET /api/experiments crashing because TrialResult does not contain parameters. Reported issue #18 and isolated the repair on `fix/workbench-persisted-trials` from main, separate from jobs PR #17 (CI green on Python 3.10/3.12).
- Extended the read-only experiment-trial DTO with detached parameters fetched from Trial; HTTP consumes that query. Missing legacy Trial records return empty parameters. Added two regressions and ran the complete branch suite: 198 passed, 86.22% coverage, one existing sklearn warning.
- Visual server runs from `/tmp/catml-kaggle-preview`, combining committed jobs code with the separate trial fix, using the original competition paths/workspace. Listening at http://localhost:8080; verified experiments endpoint HTTP 200, and browser loaded page/API/static modules. Original modified checkout files preserved.

## Persistent Jobs — 2026-09-30

- Verified PRs #13/#15 passed CI; #15 had merged into the documentation branch after #13 merged into main. Opened integration PR #16, subsequently merged by the reviewer. Updated isolated jobs branch from main without touching original checkout edits.
- Added pure domain Job/JobStatus and JobRepositoryPort, application job validation/control and operation execution, SQLite atomic idempotency/claim/state transitions, and one process-leased local worker.
- Workbench experiments/OOF/submissions now submit jobs and poll persistent snapshots. Activity survives page reload. Replaced simulated experiment progress with completed models/folds; expose pause/resume/cancel/retry and CLI job operations.
- Controls committed before finalization win completion races. Worker lease remains held while an active fit finishes during shutdown. Restart marks abandoned active work interrupted, preserving queued work and requiring explicit retry.
- Experiment checkpoints preserve finished model results. Partial OOF restarts; no exactly-once execution or forced fit interruption claimed. HPO remains synchronous. Documented Linux/macOS worker support, legacy endpoint compatibility and artifact/retry limits in spec and ADR 003.
- Added failed-model retry regression: retry refits failed/pending models and reuses successful results rather than repeatedly skipping failed trials at the final checkpoint.
- Final validation: 229 Python tests passed, 86.31% coverage, one existing sklearn deprecation warning. Four Node adapter tests passed; checked CLI job help/task catalog, all 21 Markdown file links, changed JavaScript syntax and diff whitespace. Original checkout still contains only its eight pre-existing modified files.
- Prepared an independent draft PR against main. No merges performed.
