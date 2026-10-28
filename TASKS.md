# Tasks

Fuente del estado operativo y del backlog. Las guías y los planes enlazan aquí; no mantienen una segunda lista de tareas. Las fases V0.x son hitos de diseño; la versión del paquete se define en `src/automl/__init__.py`.

### Track P0 — Auditoría Externa de Gobernanza de Agentes (2026-10-28)

- [x] **Unificación de Gobernanza MCP en `ToolExecutor` (P0)**:
  - Todas las operaciones MCP de análisis (`analysis_create_study`, `analysis_run_study`, `analysis_propose_experiment`, `analysis_verify_hypothesis`) y recetas (`recipe_create_from_evidence`, `recipe_select`, `recipe_submit`, `recipe_promote`, `recipe_archive`) ahora se enrutan a través del `ToolExecutor` gobernado (`_execute_or_dispatch`).
  - Cumple estrictamente la regla de oro: *"Un LLM puede proponer libremente (`ToolEffect.PROPOSE`), pero solo puede modificar, ejecutar o promocionar (`ToolEffect.MUTATE`) aquello para lo que tiene autorización verificable"*.
- [x] **Expiración Dinámica de Aprobaciones con TTL (P0)**:
  - Eliminada la caducidad fija en el año 2099 en `ToolExecutor`.
  - Añadido `approval_ttl_seconds` configurable (por defecto 86400s / 24h) con cálculo dinámico en UTC.
  - Comprobación y transición automática a `ApprovalStatus.EXPIRED` en el ledger ante solicitudes vencidas, denegando la ejecución con `ToolErrorCode.PERMISSION_DENIED`.
- [x] **Contabilización y Límites de Folds en `AgentBudget` (P0)**:
  - Añadidos `consumed_folds` y `remaining_folds()` en `AgentBudget`.
  - `AgentBudget.can_afford()` valida explícitamente el límite de folds (`cost["folds"]`) impidiendo sobreasignación presupuestaria.
  - `sqlite_agent_ledger` inicializa y calcula acumulados de folds activos.
- [x] **Cancelación Real de Ejecuciones y Procesos Trabajadores (P0)**:
  - `cancel_operation` ahora invoca `workspace.cancel_run(op.run_id)` y emite comandos `ControlJobCommand(job_id, action="cancel")` a `workspace.jobs`, asegurando la interrupción efectiva de procesos trabajadores y liberación de recursos sin fugas CQRS.
- [x] **Exposición de `optimize_experiment` en el Servidor MCP (P0)**:
  - Publicada la herramienta `@server.tool(name="optimize_experiment")` en el catálogo MCP principal, gobernada por `ToolExecutor` y con esquema de parámetros validado.
- [x] **Suite de Verificación y Pruebas Automatizadas (P0)**:
  - Nueva suite `tests/test_audit_p0_agent_governance.py` con 7/7 tests pasando.
  - Actualizadas suites de paridad y compatibilidad MCP y recetas.

### Track Landing — Public homepage (independent of Workbench)

- [x] Aplicar CATML_monetizacion_privada.patch para retirar la estrategia comercial de la documentación y web públicas.

- [x] Implementar homepage en `website/` con Vite, React y TypeScript; tokens Tech Minimalista, responsive 320–1440 px y accesibilidad comprobada.
- [x] Integrar captura real del Workbench de `e4a2286` con dataset sintético, snippets acordes a la API, licencia Apache-2.0, instalación desde GitHub y Platform identificada como planificación.
- [x] Preparar build estático, OpenGraph, favicon, tipografías locales, guía Vercel y comprobaciones en navegador.
- [x] Integrar la rama `feat/public-landing` en `main`; build verificado (222 ms) con dist estático y configuración para Vercel (`website/vercel.json`).

### Track Persona A (Especialistas & Políticas)

- [x] A5: implementar adaptadores intercambiables OpenAI Responses, Claude Messages, Ollama y OpenAI-compatible sobre `LLMProviderPort`, sin dependencias nuevas ni cambios en contratos compartidos.
- [x] A5: selección explícita, validación JSON local, límites de entrada/salida, timeout por socket, reintentos HTTP acotados, redacción y auditoría en memoria con consumo desconocido distinguido.
- [x] A5: documentar configuración e inyección en especialistas en `src/automl/infrastructure/llm/README.md`; pruebas sin credenciales y fixture HTTP local en el archivo asignado `tests/test_v10_specialists.py`.
- [x] A5: validación final — 508 tests pasando, cobertura 87,72 %, CLI help/task list y diff sin errores de whitespace.

### Track Persona B (Interfaces, Integración & Orquestación)

- [x] B5: Implementar `LangGraphAgentOrchestrator` con StateGraph (`observe` -> `propose` -> `gate` -> `execute` -> `critique` -> `check_stop`), adapter durable `SqliteCheckpointSaver` con reconexión resiliente a caídas.
- [x] B5: Paridad CLI en `automl agent session resume/status` con soporte para `--engine {deterministic,langgraph}`.
- [x] B5: Suites de tests exhaustivas en `tests/test_v10_orchestrator_b5.py` (13 tests) y `tests/test_v10_agent_e2e.py` (3 tests E2E y recuperación de crashes sin duplicación de hipótesis ni ejecuciones). 590 tests globales pasando, 87.70% de cobertura.
### Track Workbench — Auditoría UX/UI, Coherencia de Contexto y Refactorización

- [x] **PR 0: Restaurar CI y empaquetado (`pyproject.toml`)**:
  - Eliminado clasificador redundante de licencia en `pyproject.toml` que provocaba `setuptools.errors.InvalidConfigError`.
  - Verificado empaquetado completo `python -m build` (generación exitosa de sdist `.tar.gz` y wheel `.whl`).
- [x] **PR 1: Fase P0 — Contexto único de dataset y ejecución (`activeDatasetId`, `activeRunId`)**:
  - Centralizada resolución de contexto reactivo en `store.js` (`resolveContext`, `setActiveDataset`, `setActiveRun`, `getActiveDataset`, `getActiveRun`, `getRunsForActiveDataset`, `contextVersion`).
  - Erradicados todos los fallbacks cruzados a `runs[0]` o `datasets[0]` en `app.js`, `agent.js`, `studio.js`, `compare.js`, `home.js`, `overview.js`, `new_experiment.js`, `kaggle.js`, `knowledge.js`, `pipeline.js` y `datasets.js`.
  - Reestructurado `AgentDrawer` con tres estados explícitos: (1) Sin dataset seleccionado («Selecciona un dataset para comenzar»), (2) Dataset seleccionado sin ejecución («Sin ejecución activa»), (3) Dataset con ejecución activa (indicando dataset y run a las que pertenecen las hipótesis).
  - Eliminadas puntuaciones engañosas globales de `overview.best_score` o `overview.total_trials` en tarjetas locales de Studio y Overview cuando la ejecución no tiene resultados.
  - Soportado filtro `dataset_id` en endpoint REST `/api/runs`.
  - Nueva suite automatizada `tests/test_workbench_context_consistency.py` con 7 pruebas pasando (100% verde).
- [x] **PR 2: Fase P0 — Métricas, puntuaciones y comparabilidad**:
  - Implementadas funciones canónicas de dirección de métrica `_is_metric_minimizing` y selector de mejor trial `_get_best_trial_result` en backend (`server.py`), asegurando que métricas de error (RMSE, MSE, MAE, log-loss) seleccionen siempre el valor mínimo y métricas de score (ROC-AUC, Accuracy, F1) el máximo.
  - Expuesto metadato explícito `metric_direction` ("minimize" / "maximize") e `is_minimizing` en `/api/runs`, `/api/experiments`, `/api/experiments/compare` y `/api/overview`.
  - Enriquecidas utilidades frontend en `utils.js` (`isMinimizingMetric`, `getMetricDirectionInfo`, `formatMetricComparison`).
  - Bloqueo de comparaciones incompatibles en `compare.js`: banner de alerta cuando se intentan comparar experimentos con métricas diferentes, suspensión del cálculo engañoso de deltas y suspensión de gráfico conjunto de escalas mixtas.
  - Desbloqueado el rango Y del gráfico de evolución en `compare.js`, eliminando el límite rígido `Math.min(1.0, ...)` que deformaba métricas de regresión (RMSE/MSE superiores a 1.0).
  - Sustitución de etiquetas engañosas por terminología científicamente honesta: "Optimal" -> "Mejor de esta ejecución", "Verified" / "Verified by CV" -> "Evaluado" / "Evaluado por Cross-Validation", "Mejora" -> "Diferencia observada favorable" o "Diferencia no concluyente" cuando $|\Delta| < 0.00005$.
  - Nueva suite automatizada `tests/test_workbench_metric_context.py` con 7 pruebas unitarias y de integración pasando al 100%.
- [x] **PR 3: Fase P1 — Simplificación del flujo, navegación y estados de trabajos**:
  - Centro de control orientado a las 4 etapas del ciclo de ML: Datos (01) → Exploración (02) → Experimentos (03) → Resultados (04).
  - Banner de flujo guiado en `home.js` con accesos directos por etapa e indicador de estado en tiempo real.
  - Fidelidad de estados de trabajos y ejecuciones en `utils.js` (`ACTIVE_JOB_STATUSES`, `isJobActive`, `JOB_STATUS_LABELS`, `JOB_ACTION_LABELS`, `getJobStatusLabel`), cubriendo de forma estricta los 9 estados del ciclo de vida (`queued`, `running`, `pause_requested`, `cancel_requested`, `paused`, `completed`, `failed`, `cancelled`, `interrupted`).
  - Separación rigurosa en `home.js`: «Ejecuciones en curso» muestra EXCLUSIVAMENTE operaciones activas reales; las ejecuciones completadas se muestran en un resumen separado con motor en reposo sin fingir actividad en curso.
  - Reestructuración de la barra flotante de trabajos en `jobs.js`: cuando hay 0 trabajos activos se colapsa en un discreto disparador circular no invasivo que no cubre botones ni tablas; cuando hay trabajos activos muestra una barra flotante en vivo con progreso real o indeterminado y controles de pausa/cancelación; modal ampliado con separación estricta por pestañas entre «En curso» e «Historial».
  - Limpieza del Hero Card en `home.js`: detalles técnicos y rutas colapsables en `<details>`, manteniendo una tarjeta principal despejada y acciones principales adaptadas al estado real (sin dataset, preparado sin entrenar, entrenando, completado).
  - Actualización de la numeración y etiquetas de navegación en `index.html` reflejando el flujo de 4 etapas (`00 Inicio`, `01 Datos (Dataset)`, `02 Exploración`, `03 Experimentos`, `04 Resultados & Evidencia`).
  - Nueva suite automatizada `tests/test_workbench_jobs_and_flow.py` con 6 pruebas pasando al 100% (72 pruebas de workbench pasando globalmente).
- [x] **PR 4: Fase P1 — Leaderboard centrado en resultados, recomendaciones y recetas**:
  - Hero Card de Evaluación y Modelo Ganador colocado en la parte superior de Studio (`studio.js`): métrica, dirección de optimización, score, tipo de validación y comparación prudente contra baseline (`vs baseline` o guión cuando no existe).
  - Tabla de Leaderboard centrada en resultados con identificación unívoca de ensayos (`row.trial_id || exp?.name`), delta vs baseline, duración, estado y botón de inspección modal de parámetros (`data-inspect-trial`).
  - Reorganización de pestañas técnicas secundarias debajo del Leaderboard: Modelos (`models`), Convergencia HPO (`hpo`), Decisiones del Planificador (`planner`), Recetas Reproducibles (`recipes`), Recursos (`resources`) y Logs (`logs`).
  - Ciclo de vida estandarizado de transformaciones en `utils.js` (`TRANSFORMATION_LIFECYCLE_STAGES`, `getTransformationStageInfo`) con badges de estado formalizados: Propuesta (`PROPOSED`), Verificada (`VERIFIED`), Seleccionada (`SELECTED`), Aplicada (`APPLIED`), Promovida (`PROMOTED`). Integrado en `explore.js`.
  - Sincronización robusta de selección de características en `datasets.js` (`_updateSelectionUi`), actualizando concurrentemente `selectedFeatures`, `customFeatures` y `selectedFeaturesByDataset` en el store.
  - Nueva suite automatizada `tests/test_workbench_leaderboard_and_recipes.py` (5 pruebas pasando al 100%, total de 80 tests de workbench en verde).
- [x] **PR 5: Fase P2 — Idioma, legibilidad y accesibilidad**:
  - Unificación integral de textos en español en todas las pantallas del Workbench (`index.html`, `app.js`, `workbench.css`, `datasets.js`, `overview.js`, `studio.js`, `new_experiment.js`, `compare.js`, `kaggle.js`, `pipeline.js`, `knowledge.js`).
  - Preservación rigurosa de acrónimos y términos universales de machine learning (ROC-AUC, RMSE, MAE, MSE, K-Fold, Stratified K-Fold, LightGBM, XGBoost, CatBoost, Random Forest, SHAP, Out-of-fold, etc.).
  - Accesibilidad WCAG AA: atributo `lang="es"`, selectores `:focus-visible` con anillo de contraste `--catml-blue`, soporte para `prefers-reduced-motion`, `aria-label` en navegación principal, controles interactivos, botones de acción y tablas.
  - Diseño responsive para pantallas estrechas: media queries para $\le 768\text{px}$ y $\le 1024\text{px}$, contención de modales `.modal-responsive` (`max-height: calc(100vh - 2.5rem); overflow-y: auto`), desplazamiento táctil suave (`-webkit-overflow-scrolling: touch`) en contenedores tabulares.
  - Nueva suite automatizada `tests/test_workbench_i18n_and_a11y.py` (6 pruebas pasando al 100%, total de 86 tests de workbench en verde).

## Now (Active)

> Package version: **`0.9.0-rc1`** (`src/automl/__init__.py`). Strategic roadmap: [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md). Docs index: [`docs/README.md`](docs/README.md).

### Focus — Product robustness (post-rc1)

> **`0.9.0rc1` ya publicado** (TestPyPI + PyPI + GitHub `v0.9.0rc1`). Clean Arch / audits / Study A/B cerrados. Objetivo: tabular más sólido antes de stable `0.9.0` y Labs ([ADR-011](docs/decisions/011-workbench-labs-modular-frontend.md)).

| Priority | Item | Status |
| --- | --- | --- |
| P0 | **Honesty & leakage residual**: OOF blending holdout validation; pip-audit hard-fail opcional; stacking holdout residual | Done |
| P0 | **Data Prep P2**: CLI + facade `prepare` + wire pipeline-scoped steps into recipes | Done (P1+P2 done) |
| P1 | **Provenance**: trial/run metadata (seeds, CV fold ids, content hash) on export/leaderboard | Done |
| P1 | **Facade slim**: deeper `workspace.py` decomposition (más servicios; menos God-object) | Done |
| P1 | **Quality debt**: application mypy gradual; ESLint warnings cleanup; `feature_selection_v05` bench | Done |
| P0 | **External Audit (Security, Validation & CI)**: 8-point audit fixes (CI vars, Group OOF leakage, Holdout overlap, Promotion gates, Failed trial exclusion, OOF metric direction, HTTP security, Website CI build) | Done |
| P2 | Stable PyPI **`0.9.0`** when robustness gates agreed | Open (rc1 already shipped) |
| P2 | Workbench Labs Phase 1 | **Deferred** |

### Track External Audit — Security, Reliability & Validation Integrity

| Priority | Item | Status |
| --- | --- | --- |
| P0 | **GitHub Actions CI Workflow**: Sustituido contexto inválido `env` por `vars.CATML_AUDIT_STRICT` en `continue-on-error`. | Done |
| P0 | **OOF Group Data Leakage**: Soporte para `StratifiedGroupKFold` / `GroupKFold` en `evaluate_oof` y `ensemble_service` con verificación de solapamiento cero entre grupos de train y validación. | Done |
| P0 | **Holdout Validation Integrity**: Rechazo estricto de rutas idénticas, hashes de archivo idénticos, solapamiento de filas y solapamiento de grupos entre train y holdout. | Done |
| P1 | **Validación de Promoción**: Validación de `min_improvement >= 0` finito y exigencia de mejora positiva real (`holdout_delta > 0.0 and holdout_delta > min_improvement`). | Done |
| P1 | **Veracidad del Mejor Trial**: Exclusión estricta de trials fallidos en `_get_best_trial_result` y `/api/experiments/compare` devolviendo `None` si todos fallaron. | Done |
| P1 | **Selección Óptima en OOF**: Selección de parámetros del trial óptimo según la métrica y su dirección de optimización (`greater_is_better`). | Done |
| P1 | **Seguridad Web & CORS**: Rechazo de `Content-Length < 0` con `400 Bad Request` y restricción de CORS loopback al puerto del Workbench. | Done |
| P2 | **CI Build del Sitio Web**: Añadido job independiente `website-build` en CI (`npm ci` y `npm run build` con verificación de TypeScript). | Done |

### Shipped — structure / trust / studies / rc1 (reference)

| Priority | Item | Status |
| --- | --- | --- |
| P0 | Clean Arch PR1–PR5 | Done (`#102`…`#130`) |
| P0 | Audits HTTP/CSRF + model reliability | Done (`#128`) |
| P0 | Publish **`0.9.0rc1`** TestPyPI + PyPI | Done (`v0.9.0rc1`) |
| P1 | Study A (5+20 min) + Study B lift | Done |
| P1 | E2E export → reload → predict parity | Done (PR5) |

### Shipped — rc1 + audits (reference)

| Priority | Item | Status |
| --- | --- | --- |
| P0 | Tabular `0.9.0rc1` on PyPI + Clean Arch PR1–PR3 (`#111`) | Done |
| P1 | Demo/export (`#125`/`#126`); HPO fold pruning (`#116`) | Done |
| P1 | Audit HTTP/CSRF + model reliability + honest reports | Done (`#128`) |

### Track License compliance (documentation / packaging only)

| Priority | Item | Status |
| --- | --- | --- |
| P0 | Audit Workbench + Python licenses; keep Apache-2.0; no UI redesign | Done |
| P0 | `THIRD_PARTY_LICENSES.md` + vendor license texts in wheel/sdist | Done |
| P0 | Automated packaging compliance tests | Done (`tests/test_third_party_license_compliance.py`) |

### Track Clean Architecture Refactor (progressive, ADR-010)

> Spec/plan: [`docs/features/clean-architecture-refactor/`](docs/features/clean-architecture-refactor/). Gate: #99/#100/#101 merged; PR1 `#102`; PR2 `#109`; PR3 `#110`.

| Priority | Item | Status |
| --- | --- | --- |
| P0 | Docs: spec + plan + ADR-010 + agent brief PR1 | Done (`#101`) |
| P0 | PR1: `MetricSpec` registry + `ExecutionSpec` + wire consumers | Done (`#102`) |
| P1 | PR2: Extract `run_experiment` / `optimize_experiment` behind Workspace facade | Done (`#109`) |
| P1 | PR3: Web controllers + split `datasets.js` components | Done (`#110`) |
| P0 | PR4: SQLite modular boundary + state machine + Unit of Work | Done (`#129`) |
| P0 | PR5: Ruff/mypy/ESLint + contract/E2E gates | Done (`#130`) |

### Track Audit-4 — Security / inference / ensemble honesty (`fix/audit4-security-inference-ensemble`)

| Priority | Item | Status |
| --- | --- | --- |
| P0 | Web `register` path confinement (same roots as inspect/browse) | Done (`#104`) |
| P0 | Upload unique internal filenames (no silent overwrite) | Done (`#104`) |
| P0 | Inference replays `column_transforms` + derived features | Done (`#104`) |
| P0 | Ensemble regression scores: flip sklearn `neg_*` before persist | Done (`#104`) |
| P0 | Stacking nested CV OOF for metamodel (no in-sample `stacked_oof`) | Done (`#104`) |
| P1 | Unknown optimizer fail-closed (`optuna` / `random` only) | Done (`#104`) |
| P1 | #7/#8 invalid targets not fabricated in feature prep | Done (`#106`) |
| P1 | #11 simplex nested weight evaluation | Done (`#106`) |
| P1 | #13 tree_importance vs shap_importance labeling | Done (`#106`) |
| P2 | #6 exploratory labeling for supervised selection (full nested-in-fold deferred) | Done (`#106`) |
| P1 | #9/#10 paired hypothesis baseline + improved vs validated | Done (`#107`) |
| P1 | #12 PCA source features + fold-safe reduction transform | Done (`#107`) |
| P1 | #14 image cache invalidate on mtime/size | Done (`#107`) |
| P1 | #16 budget stop → ExperimentStatus.PARTIAL | Done (`#107`) |

### Track Audit-3 — Recipe / metric / CV contracts (`fix/audit3-recipe-metric-cv-contracts`)

| Priority | Item | Status |
| --- | --- | --- |
| P0 | Recipe translator contract (`default_models` / `drop_features` / `model_id`) | Done (`#100`) |
| P0 | Recipe replay restores `metric`, `cv_folds`, `random_seed`, CV strategy | Done (`#100`) |
| P0 | Shared CV aliases (`k_fold`→`kfold`); reject unknown strategies | Done (`#100`) |
| P0 | HPO uses fixed CV fold seed; metric min/max consistent in schedule + `get_best_trial` | Done (`#100`) |
| P0 | FDR preserves `p=0.0`; recipe `dataset_content_hash` via SHA-256 | Done (`#100`) |
| P1 | Workbench XSS escapes for column/feature/model interpolations | Done (`#100`) |
| P1 | Recipe status guards (no submit/promote from archived; promote needs experiments) | Done (`#100`) |
| P2 | Stacking holdout / shap_importance naming / pip-audit hard-fail | Partially addressed (#1 stacking in Audit-4; remainder deferred) |

### Track Workbench UX — Demo confidence (Sprint 1 → 3)

| Priority | Item | Status |
| --- | --- | --- |
| P0 | Unify run vs experiment vs job status labels; honest API status when trials exist | Done (`#98`) |
| P0 | Single source for trial counters (leaderboard vs «ensayos completados») | Done (`#98`) |
| P0 | Filter Explore transform hypotheses on identifier/discarded columns | Done (`#98`) |
| P1 | Guided vs Advanced Explore (top findings first) | Done (`#99`) |
| P1 | Correlation matrix UX + p-value/severity formatting | Done (`#99`) |
| P1 | Comparator CI / avoid exaggerating tiny ROC-AUC deltas | Done (`#99`) |
| P1 | Framing/CV coherence: no TimeSeriesSplit on cross-sectional; advisor ignores survey «time» phrases; job errors surface `failure_reason` | Done (local) |
| P0 | Audit-5: unify budget UI↔JobExecutor; XSS new_experiment; no fake param_importance / predict checklist; Manual model picker; multiclass ROC-AUC; fallback badges | Done (local) |
| P2 | Demo project + executive results export | Done (`feat/demo-export-and-sqlite-94`) |

### Track PyPI Pre-Publish — Wheel Workbench static + distribution gates (`fix/pypi-wheel-workbench-static`)

| Priority | Item | Status |
| --- | --- | --- |
| P0 | Include Workbench `static/` in wheel via `package-data` | Done (22 static files in wheel) |
| P0 | Smoke + CI assert wheel contains UI assets and HTTP 200 | Done |
| P0 | `release.yml` Trusted Publishing skeleton (no auto-publish on push) | Done |
| P1 | PyPI-safe README quickstart (sklearn dataset) + CDN honesty | Done |
| P1 | SECURITY/`pip-audit` honesty + CI audit job | Done (audit job non-blocking) |
| P1 | CI matrix Python 3.11 + 3.13 | Done |
| P2 | Vendor Workbench CDN assets for fully offline UI | Done (`fix/workbench-offline-vendor-assets`) |
| P0 | Reserve/publish `catml` on TestPyPI then PyPI as **0.9.0rc1** | Done (`v0.9.0rc1` on TestPyPI + PyPI) |

### Track Pre-Publish Security & Leakage (`fix/pre-publish-security-and-leakage`)

| Priority | Item | Status |
| --- | --- | --- |
| P0 | Disable derived `python_code` by default (`allow_execution` / `CATML_ALLOW_DERIVED_PYTHON=1`) | Done |
| P0 | Reject stateful formulas (`zscore`/`mean`/`std`) before CV; row-wise OK | Done |
| P1 | `ModelArtifact.load` fail-closed when `.sha256` sidecar missing | Done |
| P1 | Soften group-candidate heuristic (e.g. `Customer Type` not an entity) | Done |
| P1 | `time_series` requires `run.config.extra['time_column']` + sort | Done |
| P1 | `run_experiment` → `FAILED` when zero successful trials | Done |
| P2 | Broader offline/distribution UI matrix | Partially Done (Workbench chrome offline; vision weights / LLM still optional network) |

### Track RC1 Gate — HPO fidelity + release publish gates (`fix/hpo-recipe-fidelity-and-release-gates`)

> Mantenedores: [@Jfenic](https://github.com/Jfenic) (Jose Fenic Peiteado Padilla) y [@tvarmar](https://github.com/tvarmar) (Tania Évora Vargas Martínez). PR #95: revisión cruzada recomendada antes de merge; PyPI Trusted Publisher / environments requieren acuerdo de ambos.

| Priority | Item | Status |
| --- | --- | --- |
| P0 | HPO passes `column_transforms` + `derived_features` (recipe fidelity) | Done |
| P0 | HPO all-fail → `FAILED`; best only from succeeded trials | Done |
| P0 | `release.yml` build → verify → publish; tag↔version; env approval | Done |
| P1 | SQLite connections close-on-exit + bounded FD stress test | Done (soak: [#94](https://github.com/Jfenic/catml/issues/94)) |
| P0 | Human: protect `pypi`/`testpypi` envs + Trusted Publisher + TestPyPI | Done (rc1 published) |
| P1 | CATML Benchmark v1 Pilot 1A (sklearn/FLAML/AG vs OpenML) | Done — [`docs/experiments/pilot-1a-report.md`](docs/experiments/pilot-1a-report.md) |
| P1 | Fire tests: 10 Jupyter/scripts post-pilot | Done — [`notebooks/fire_tests/`](notebooks/fire_tests/) |
| P1 | Protocol v1 frozen + Study A (11 datasets; 5 min + 20 min formal 132/132) | Done — [`docs/experiments/study-a-report.md`](docs/experiments/study-a-report.md) |
| P1 | Study B Explore/agent lift (minimal + lift + `#119`/`#122`) | Done — [`docs/experiments/study-b-report.md`](docs/experiments/study-b-report.md) |

### Track Export / path integrity (`fix/export-parquet-path-confinement`)

| Priority | Item | Status |
| --- | --- | --- |
| P0 | Derived apply preserves Parquet/JSON (no CSV overwrite) + `.bak` | Done |
| P0 | `export_model_artifact` replays derived + column_transforms | Done |
| P0 | Workbench `/api/models/export` ships `.pkl` + `.sha256` zip | Done |
| P0 | `/api/dataset/inspect-file` confined to cwd/workspace | Done |
| P1 | `AutoML.fit` temp workspace cleaned via `TemporaryDirectory`/`close()` | Done |
| P2 | Vendor Workbench CDN assets offline | Open (packaging) |

### Track Public Surface Audit — Coherence & Security Claims (`fix/audit-public-surface-coherence`)

> External audit mixed outdated README/ARCHITECTURE claims with real gaps. Triage vs `main` (2026-10-11):

| Claim | Reality on `main` | Action |
| --- | --- | --- |
| License MIT vs Apache | Repo/`LICENSE`/`pyproject`/GitHub API already **Apache-2.0** | Keep aligned; document historical MIT ≤0.7 |
| `from catml import AutoML` broken | `src/catml/` re-exports work | Document public `catml` vs internal `automl` |
| Pickle / zero-deps | Warning + SHA-256 + dep manifest already in code | Clarify README wording (not zero-deps) |
| MCP/UI bind/auth/paths | Loopback default, fail-closed auth, path confinement shipped | Document CORS/CSRF posture in SECURITY |
| ARCHITECTURE “illustrative previews” / direct repos | Stale after CQRS guards + agent H0–H5 | Rewrite current-scope section |
| Badges static | Still shields snapshots | Link CI workflow badge |
| setuptools vs SPDX license | Real gap (`>=68` vs SPDX string) | Bump build-system to `>=77` |
| Root doc clutter | Real | Move technical/planning docs under `docs/` |
| Missing CODE_OF_CONDUCT | Real | Add Contributor Covenant |
| Rename entire tree to `catml` | Large breaking change | **Deferred** (document dual package) |
| ONNX / HMAC signatures | Roadmap | Tracked in SECURITY / MASTER_PLAN |
| Live Codecov badge | Optional | Backlog |
| Further `workspace.py` split | Debt (~1.8k LOC facade) | Backlog |

- [x] Align public docs/packaging with shipped security & API surface (merged PR #91).
- [x] Domain import architecture guard test.
- [ ] Live coverage badge (Codecov) if/when token is configured.
- [x] Publish PyPI `0.9.0rc1` (TestPyPI + PyPI + GitHub release `v0.9.0rc1`).
- [ ] Stable PyPI `0.9.0` after agreed robustness gates (Focus above).

Open product backlog (see [Next / Roadmap Phases](#next-roadmap-phases) and **Focus — Product robustness**): trial provenance metadata (P1); deeper `workspace.py` decomposition (P1); application mypy gradual & quality debt (P1); stable `0.9.0` release (P2).

## Completed Phase: v0.8.2 Phase 2 — Ingestion Polymorphism, Strict CQRS Boundaries & Application Services Decomposition

- [x] **Estudio Experimental y Validación Empírica en Kaggle Airline Satisfaction (`docs/experiments/`)**:
  - [x] Comparativa cuantitativa AutoML solo vs Explore/Agente (`docs/experiments/airline_satisfaction_automl_vs_agent_report.md`): Naive con `id` (0.817 ROC-AUC) vs Anti-Leakage Guardian (0.952) vs Drops de colinealidad propuestos por Explore (0.952, Δ = -0.0006). Validación empírica de "Propose ≠ Accept" y registro de aprendizaje negativo.
  - [x] Push de alto rendimiento en dataset de 700k filas con GBDT (LightGBM/XGBoost) y HPO Optuna alcanzando 0.9586 CV y 0.9578 en test ciego de Kaggle (`submission_airline_lgbm_hpo.csv`).
  - [x] Identificación de gaps transferibles: preservación de variables categóricas binarias (evitar falso positivo en group leakage de `Customer Type`) y necesidad imperativa de transformaciones con estado fit-in-fold que impulsa Experiment Recipes (ADR-009).
  - [x] Artefactos reproducibles en `docs/experiments/` (`push_airline_score.py`, `airline_push_scores.json`, `airline_automl_vs_explore_results.json`, `airline_automl_vs_explore_decomposition.json`).

- [x] **Corrección de Visualización y Contraste en CATML Explore (`ExploreView`)**:
  - [x] Reglas CSS globales para `.input-base`, `select` y `select option` en `workbench.css` eliminando fondos blancos de navegadores y asegurando texto legible en dark theme (`#151B26` fondo, `#F7F8FA` texto, `#4F67FF` foco).
  - [x] Clases oscuras explícitas en `explore.js` para el selector de estudio activo (`#selectActiveStudy`), filtros de categoría/severidad y campos modales.
  - [x] Soporte dual `viz.data_series || viz.data` en renderizado declarativo de visualizaciones (matriz de correlación Pearson/Spearman 19x19, Cramér's V, histogramas, diagramas de caja y dispersión).
  - [x] Homogeneización de clasificación de severidad de hallazgos estadísticos y cálculo robusto de tamaño del efecto y p-valores en tarjetas de la galería.
  - [x] Ordenación por importancia en la galería de hallazgos estadísticos (`_getFilteredFindings` y `datasets.js`): severidad descendente (`CRITICAL` > `WARNING` > `INFO`), significancia FDR y magnitud del tamaño del efecto.
  - [x] Auto-backfill en `AutoMLWorkspace._hydrate_feature_registry()` registrando características a partir del perfil del dataset cuando la tabla de features no las contiene, previniendo el error `Unknown features for dataset` en verificación de hipótesis.
  - [x] Test automatizado `test_workbench_explore_form_inputs_and_select_styles` en `tests/test_workbench_navigation_and_explore_ui.py`.

- [x] **Corrección de Perfilado de Columnas Booleanas y Registro de Datasets en Workbench**:
  - [x] Conversión de series booleanas (estándar y nullable `pd.BooleanDtype`) a `float64` antes del cálculo de cuartiles, medianas, histogramas y correlaciones en `dataset_profiler.py`, `distribution_diagnostics.py` y `builder.py`. Resuelve la incompatibilidad de resta booleana en NumPy (`TypeError: numpy boolean subtract, the '-' operator, is not supported`).
  - [x] Hardening de captura de excepciones y logging de traceback en `/api/dataset/register` (`server.py`).
  - [x] Resiliencia en vista frontend de hallazgos estadísticos (`datasets.js`) infiriendo severidad y métricas sin depender de campos indefinidos.
  - [x] Pruebas de regresión automatizadas en `tests/test_profiler_cardinality.py` y validación en dataset real (`data/kagle/train.csv`, ~700k filas). 732 tests pasando, 87% cobertura.

- [x] **Refinamiento UI Workbench — Eliminación de Datasets Demo y Garantía de Workspace Vacío**:
  - [x] Eliminada resolución predeterminada a `.automl/s6e9_automl` o `.automl/demo` en `launch_ui_cli` (`main.py`) y en `server.py`; resolución canónica limpia a `.automl/default`.
  - [x] Limpieza de base de datos residual `workspace/automl.db` y `.automl/default/` (eliminando dataset «E-Commerce Multimodal Catalog» y ejecuciones demo huérfanas).
  - [x] Erradicación de datos sintéticos o hardcodeados en el frontend: estados vacíos honestos en `kaggle.js` («Sin Experimentos de Competición Activos») y `pipeline.js` («Sin Pipeline Generado»).
  - [x] Inicialización inmediata del widget lateral en `index.html` a «Sin dataset activo» con punto indicador neutral.
  - [x] Test automatizado `test_workbench_clean_empty_workspace_defaults_and_zero_hardcoding` en `tests/test_workbench_laboratory_redesign.py`. 731 tests pasando (100% verde), 87.07% cobertura.

- [x] **Ingestión Polimórfica y Soporte Nativo de Parquet/DataFrames**:
  - [x] Soporte nativo para `.parquet`, `.pq`, `.json`, `.jsonl`, `.tsv` y `.csv` en `src/automl/engine/profiling/dataset_profiler.py` (`load_dataframe`).
  - [x] Mensaje informativo claro con `ImportError` guiando al usuario para instalar `pyarrow` o `fastparquet` si no están presentes.
  - [x] En `AutoMLWorkspace.register_dataset()`: aceptación de `pd.DataFrame` directamente, persistiendo de forma automática en `datasets/` (.parquet o .csv) y registrando el dataset.
  - [x] En `AutoML.fit()` (`facade.py`): soporte polimórfico para rutas en `str` o `Path` (además de `pd.DataFrame` y `np.ndarray`).
  - [x] Extra `parquet = ["pyarrow>=14.0"]` añadido a `pyproject.toml` e incluido en grupo `all`.
  - [x] Tests unitarios en `tests/test_parquet_and_polymorphic_ingestion.py` (4 tests).
- [x] **Frontera Arquitectónica CQRS Estricta y Eliminación de Fugas de Estado**:
  - [x] Formalización de consultas DTO en `QueryBus` (`src/automl/application/queries/workspace_queries.py`): `GetRunQuery`, `ListRunsQuery`, `GetDatasetQuery`, `ListDatasetsQuery`, `GetTrialQuery`, `GetExperimentQuery`, `ListTrialResultsQuery`.
  - [x] Registro y resolución limpia de handlers en `src/automl/application/bootstrap.py`.
  - [x] Adición de métodos de consulta públicos en `AutoMLWorkspace` para evitar acoplamiento directo con infraestructura (`list_datasets`, `get_dataset_profile`, `save_dataset_profile`, `save_run`, `get_trial`, `get_experiment`, `list_experiments`, `list_trial_results`, `get_leaderboard_results`, `list_feature_sets`).
  - [x] Eliminadas todas las llamadas directas a `ws.repository` en la fachada (`facade.py`) y en la API web (`interfaces/web/server.py`).
  - [x] Eliminada la inyección monkey-patch en `src/automl/interfaces/cli/agent_session_cli.py`.
  - [x] Guard arquitectónico automatizado en `tests/test_cqrs_architecture_guard.py` (2 tests verificando 0 ocurrencias de fugas privadas en capas de presentación e interfaces y despacho vía `QueryBus`).
- [x] **Descomposición del God Module `workspace.py` en Servicios de Aplicación**:
  - [x] Extracción de `PipelineExecutionService` en `src/automl/application/services/pipeline_service.py` (validación de grafos, orden topológico de ejecución, construcción de DAG multimodal, ejecución y fit/predict).
  - [x] Extracción de `InferenceService` en `src/automl/application/services/inference_service.py` (predicción, alineación de plantillas Kaggle, generación de envíos y exportación de `ModelArtifact`).
  - [x] Extracción de `FeatureEngineeringService` en `src/automl/application/services/feature_service.py` (características derivadas, análisis de dinámica temporal, estrategias de selección multi-método, planificación de ablación y promoción de conjuntos de características).
  - [x] Delegación 100% retrocompatible en `AutoMLWorkspace` preservando firmas públicas y reduciendo `workspace.py` en más de 740 líneas.
- [x] **Resolución de Auditoría Técnica y Hardening de Confiabilidad ML**:
  - [x] **[Crítico] Inversión de signo de métricas de error (MAE, RMSE, MSE, Log Loss)**: En `sklearn_trainer.py`, los scores negativos retornados por `cross_val_score` para estimadores de pérdida (`neg_mean_absolute_error`, `neg_root_mean_squared_error`, etc.) se invierten (`scores = -scores`), produciendo valores positivos reales. En `workspace.py`, la dirección de optimización y el ordenamiento del leaderboard para estas métricas es estrictamente minimizante (`ASC`), asegurando que el modelo con menor error gane siempre.
  - [x] **[Alto] Exclusión de trials fallidos en leaderboard y selección del ganador**: En `sqlite_repository.py` y `ports.py`, `get_leaderboard()` filtra por defecto los trials que fallaron (`include_failed=False`). En `facade.py` e `inference_service.py`, la selección del modelo ganador y la exportación de `ModelArtifact` excluye explícitamente cualquier trial con `failure_reason` o `succeeded is False`.
  - [x] **[Alto] Eliminación de sustitución silenciosa de métricas no soportadas**: En `sklearn_trainer.py` (`_sklearn_scoring` y `_compute_metrics`), se eliminó el fallback silencioso a `accuracy` o `r2`. Métricas no soportadas levantan `ValueError`. Se añadió soporte de primera clase para `mse`, `log_loss`, `balanced_accuracy`, `precision`, `recall` y plugins métricos vía `make_scorer`.
  - [x] **[Medio-alto] Documentación de Time Budget Cooperativo**: En `facade.py`, se documentó explícitamente que `time_budget` opera a nivel global entre modelos de forma cooperativa y no como un timeout preemptivo atómico por proceso.
  - [x] **[Medio-alto] Hardening HTTP/MCP y Robustez en Datasets Reducidos**: El Workbench soporta paso seguro de tokens vía fragmento `#token=` que se limpia inmediatamente de la URL con `history.replaceState`. El servidor MCP emite advertencia de seguridad (`SECURITY NOTICE`) si se expone `streamable-http` fuera de localhost. `SklearnTrainer` adapta inteligentemente los folds de KFold y splits holdout en datasets pequeños para evitar divisiones de muestra única.
  - [x] Suite de pruebas dedicada en `tests/test_audit_findings.py` (4 tests). 622 tests pasando (100% verde), 87.37% de cobertura.
- [x] **Alineación de Estado Activo y Feed de Actividad en Workbench (PR #85)**:
  - [x] Definición canónica de `ACTIVE_RUN_STATUSES`, helper `is_active_run_status()` y propiedad `run.is_active`.
  - [x] Exposición de flag `is_active` en `/api/runs` y `/api/overview`, priorizando ejecuciones vivas sobre ejecuciones completadas.
  - [x] Segregación del feed de actividad en Workbench sin mezclar métricas incomparables.
  - [x] Función compartida `isRunActive()` en frontend (`utils.js`) y soporte de score 0.0 y $R^2 < 0$.
  - [x] Corrección de mensaje de error al programar plan en `agent.js`.
- [x] **Rediseño UI Workbench — Modelo de Laboratorio y Dataset Activo (Fase 1)**:
  - [x] Contexto de Laboratorio Local en barra lateral con widget de Dataset Activo (`#sidebarActiveDatasetName`) y selector rápido (`SwitchDatasetModal`).
  - [x] Simplificación de navegación principal a 4 vistas canónicas: Inicio (`home`), Dataset (`dataset`), Experimentos (`experiments`), Evidencia (`evidence`).
  - [x] Nueva vista de Inicio (`src/automl/interfaces/web/static/js/views/home.js`): Hero card «Continuar con {dataset activo}», acciones secundarias «Añadir dataset» y «Abrir dataset guardado», y paneles de ejecuciones activas y trabajos recientes.
  - [x] Rediseño de la vista Dataset (`src/automl/interfaces/web/static/js/views/datasets.js`): Pestañas «Resumen» (métricas de perfil, salud de datos Anti-Leakage Guardian, recomendaciones y preview crudo), «Columnas» (tabla de variables, filtros, selección interactiva) y «Exploración» (panel conceptual de capacidades estadísticas para CATML Explore Fase E1).
  - [x] Integración de contexto de dataset activo en `store.js` (`setActiveDataset`, `getActiveDataset`), `app.js`, `studio.js`, `overview.js` y `new_experiment.js`.
  - [x] Suite de tests automatizada en `tests/test_workbench_laboratory_redesign.py` (3 tests). 667 tests pasando (100% verde).
- [x] **Rediseño UI Workbench — Modal de Ingestión Polimórfico (Fase 2)**:
  - [x] Modal de Ingestión Polimórfica (`RegisterDatasetModal` en `register_dataset_modal.js`):
    - Subida Drag & Drop con subida base64 al servidor (`POST /api/dataset/upload`).
    - Navegador de archivos local en servidor con exploración de carpetas y búsqueda (`GET /api/files/browse`).
    - Entrada manual de ruta con inspección instantánea de esquema (`POST /api/dataset/inspect-file`).
    - Sugerencia automática de columna target y soporte explícito para datasets no supervisados (`target=None`).
  - [x] Creación dinámica de runs (`POST /api/run/create`) vinculados al dataset activo.
  - [x] Scoping limpio de variables en `new_experiment.js` y `datasets.js` (`activeRun`, `recentDs`).
  - [x] Tests de validación e integración en `tests/test_workbench_laboratory_redesign.py` (6 tests).
- [x] **Remediación Integral de Auditoría Técnica (Octubre 9, 2026 — Commit `5caa1f5`)**:
  - [x] **H1 [Alta] Aprobación real de agentes en Workbench**: `POST /api/agent/action` integrado con `SqliteAgentLedger`, validación estricta de existencia (404 ante IDs inexistentes), transición de estado persistida en SQLite y registro de revisor humano.
  - [x] **H2 [Alta] Sentido de métricas personalizadas en CV**: Eliminado `_greater_is_better` inexistente; normalización de signo con `_is_minimizing_metric()` e inspección de `_sign == -1`; sincronización de `plugin_registry` en `SQLiteExperimentRepository.get_leaderboard()` para ordenamiento ascendente (`ASC`).
  - [x] **H3 [Alta] Autenticación remota en servidor MCP**: Transporte `streamable-http` rechaza enlaces externos sin autenticación (`PermissionError` fail-closed); flags `--token` y `--insecure-no-auth` en CLI con `BearerAuthMiddleware` en Starlette.
  - [x] **H4 [Media-alta] Límite de tamaño HTTP explícito**: Límite `MAX_PAYLOAD_SIZE = 50 MB` en `server.py`, rechazando peticiones excesivas con `413 Payload Too Large`.
  - [x] **H5 [Media-alta] Endurecimiento de autenticación remota**: Rechazo de tokens en query string (`?token=`) para operaciones POST; advertencia de seguridad en peticiones GET.
  - [x] **H6 [Media] Compatibilidad de versión scikit-learn**: Declaración de `scikit-learn>=1.4.0` en `pyproject.toml` con fallback defensivo (`needs_proba`) en `make_scorer`.
  - [x] **H8 [Media] Bloqueo multiplataforma en worker**: Uso de `msvcrt` en Windows y `fcntl` en Unix en `worker.py`.
- [x] **Remediación Integral de Auditoría UX/UI Frontend (Workbench)**:
  - [x] **F01 [Crítico UX]**: Retirados CPU 60%, RAM 45%, 5 workers fijos y logs predefinidos en `studio.js`; telemetría real conectada con ejecuciones y tiempos por estimador.
  - [x] **F02 [Crítico UX]**: Retirado badge simulado `DAG Validated` en `pipeline.js`; sustituido por estado verificado (`Pipeline Trained & Aligned` / `Schema Inferred`); corregidas descripciones a `StandardScaler` y `OneHotEncoder` / `TargetAdapter`.
  - [x] **F03 [Crítico UX]**: Retirado fallback mock "Standard Tabular Benchmark" (85% similar) en `knowledge.js`; renderizado de estado vacío honesto.
  - [x] **F04 [Medio-alto]**: Subsanado `ReferenceError: customFeatures is not defined` en `new_experiment.js`, encapsulado en instancia y vinculado al dispatch.
  - [x] **F05 [Medio-alto]**: Conectados botones simulados ("Promote Best Parameters" y "Use Warm Start") a almacenamiento reactivo en store (`promotedParams`, `warmStartParams`).
  - [x] **F06 [Medio-alto]**: Propagación completa de opciones en `createAndRunExperiment` y soporte seguro de presupuesto en `job_executor.py` (`time_budget_seconds`).
  - [x] **F07 [Medio]**: Desvinculación de ejecución huérfana en `store.setActiveDataset()` al cambiar a datasets sin runs previos.
  - [x] **F09 [Medio]**: Checklist pre-flight de Kaggle en `kaggle.js` muestra estados pendientes neutrales hasta verificación empírica.
  - [x] **F10 [Medio]**: Selección del trial óptimo en HPO guiado por dirección de métrica (minimización vs maximización).
  - [x] **F11 [Medio]**: Enlace de descarga de submissions sustituido por `api.downloadFile()` con autenticación Bearer via blob.
  - [x] **F12 & F15 [Medio]**: Accesibilidad mejorada: eliminado `select-none` del body; menú responsive con botón hamburguesa para pantallas `< md`.
- [x] **Eliminación de Maquetas y Estados Simulados en Pantalla de Inicio (Truthful Workbench State)**:
  - [x] Acotado de `ACTIVE_RUN_STATUSES` y `is_active_run_status` en dominio y frontend únicamente a estados de ejecución efectiva (`EXPERIMENTING`, `OPTIMIZING`, `FINALIZING`, `RUNNING`); estados `CREATED`, `PROFILING` y `PLANNING` no activan falsas alertas de entrenamiento.
  - [x] Eliminado badge falso de `ENTRENANDO` con spinner y barra fija al 75% en `home.js`; reemplazado por estado verídico `LISTO PARA ENTRENAR` (barra al 0%, métricas neutrales, CTA para lanzar experimento).
  - [x] Eliminada barra fija al 66% y texto arbitrario `Searching space (72%)` en `overview.js`.
  - [x] Marcado en **ROJO** del componente Explore en `datasets.js` (`PENDIENTE DE IMPLEMENTACIÓN`, `EN FASE DE IMPLEMENTACIÓN`, `#EF4444`) para clarificar que pertenece al roadmap de ADR-008 sin maquetas decorativas.
  - [x] Sincronización de la cabecera global (`app.js`): solo exhibe `ENTRENANDO` cuando existen trabajos de entrenamiento activos en ejecución.
  - [x] Pruebas de integración automatizadas en `tests/test_workbench_laboratory_redesign.py`. 677 tests pasando (100% verde), 86.68% de cobertura.
  - [x] **F13 & F14 [Bajo]**: Terminología armonizada y versión alineada a `v0.8.2`.
- [x] **Validación y Cobertura**:
  - [x] 676 tests pasando (100% de la suite). Cobertura total mantenida $\ge 85\%$.

## Completed Roadmap: CATML Explore (Análisis Estadístico, Evidencia y Agentes)

> Plan de evolución arquitectónica como monolito modular con separación hexagonal y CQRS ([docs/features/catml-explore/plan.md](docs/features/catml-explore/plan.md), [ADR-008](docs/decisions/008-catml-explore-modular-monolith.md)).

- [x] **Fase E0 — Estabilización del Núcleo (Core Stabilization)**:
  - [x] Normalizar signo y orden de métricas de error (MAE, RMSE, MSE, Log Loss).
  - [x] Excluir trials fallidos del leaderboard y selección de ganador.
  - [x] Erradicar fallbacks silenciosos a `accuracy`/`r2`.
  - [x] Hardening de presupuesto temporal cooperativo y autenticación remota HTTP/MCP.
  - [x] 622 tests pasando (100% verde), 87.37% de cobertura.
- [x] **Fase E0.5 — Simplificación Previa y Modularización de Registro/Contratos (Pre-Explore Surgical Refactoring)**:
  - [x] **PR 1 — Simplificación de dependencias y modularización de `bootstrap.py`**:
    - [x] Descomponer el registro monolítico en submódulos por contexto funcional en `src/automl/application/registries/` (`job_registry`, `core_registry`, `experiment_registry`, `feature_registry`, `inference_registry`).
    - [x] Reducir `bootstrap.py` a punto limpio de orquestación (de 342 a 53 líneas), manteniendo compatibilidad 100% de `build_application`.
    - [x] Añadir guard arquitectónico de composición independiente en `tests/test_cqrs_architecture_guard.py`.
    - [x] 623 tests pasando (100% verde), 87.37% de cobertura.
  - [x] **PR 2 (PR #78) — Contratos y límites de módulos (Dataset sin target y Jobs desacoplados)**:
    - [x] Desacoplar `Dataset`, `DatasetProfile` y `dataset_profiler` permitiendo `target_column: str | None = None` y `task_type: str | None = None`.
    - [x] Desacoplar persistencia en `sqlite_repository.py` admitiendo `target_column TEXT NULL` y `task_type TEXT NULL`.
    - [x] Desacoplar modelo `Job` y `sqlite_jobs.py` para permitir `run_id: str | None = None`.
    - [x] `AutoMLWorkspace.register_dataset(target=None)` y planificación no supervisada por defecto (`TaskType.CLUSTERING`).
    - [x] Suite de tests dedicada en `tests/test_unsupervised_datasets_and_decoupled_jobs.py` (4 tests). 627 tests pasando (100% verde), 87.38% de cobertura.
  - [x] **PR 3 (PR #79) — Hardening de Migraciones SQLite y Compatibilidad Retroactiva**:
    - [x] Migración atómica determinista de la tabla `datasets` en `SQLiteExperimentRepository._migrate()` relajando restricciones `NOT NULL` sobre bases heredadas v0.8.0/v0.8.1.
    - [x] Migración de la tabla `jobs` en `SQLiteJobRepository` relajando `run_id NOT NULL` sobre bases heredadas.
    - [x] Erradicación de excepciones silenciadas (`except Exception: pass`) reemplazadas por logging estructurado y fallos explícitos fail-closed.
    - [x] Suite de pruebas exhaustiva con esquemas legacy y datos preexistentes en `tests/test_sqlite_legacy_migrations.py` (2 tests).
    - [x] 629 tests pasando (100% verde), 87.37% de cobertura.
- [x] **Hardening de Seguridad y Robustez de Entrenamiento (Auditoría Técnica Oct 2026)**:
  - [x] **PR A (PR #80) — Blindaje Anti-Leakage Centralizado y Fail-Closed**:
    - [x] Centralizar política de selección de variables en `DatasetProfile.resolve_safe_feature_names()`.
    - [x] Parar con `ValueError` fail-closed en `AutoML.fit()` (`facade.py`) cuando todas las variables son descartadas por leakage o identifiers.
    - [x] Retornar lista vacía de candidatos en `RuleBasedExperimentPlanner` si sólo existen variables con leakage.
    - [x] Excluir automáticamente leakage en `JobExecutor` para ejecuciones con generación por defecto de features.
    - [x] Validar variables en `AutoMLWorkspace.create_experiment()` con soporte para `allow_leakage=True`.
    - [x] Calibrar thresholds estadísticos en `dataset_profiler.py` (`cardinality_ratio <= 0.90` en group hints, `row_count >= 10` en correlación de target) para evitar falsos positivos en features continuas y datasets enanos.
    - [x] Suite de tests dedicada en `tests/test_anti_leakage_guardian.py` (6 tests). 635 tests pasando (100% verde), 87.29% de cobertura.
  - [x] **PR B (PR #81) — Seguridad del Workbench (Confinamiento de Archivos y Sanitización de Rutas)**:
    - [x] Confinamiento de `/api/kaggle/download` estrictamente dentro del workspace con lista blanca de extensiones (`.csv`, `.tsv`, `.parquet`, `.pq`, `.json`, `.zip`, `.txt`) y bloqueo fail-closed de path traversal (`403 Forbidden`).
    - [x] Restricción de `/api/media/preview` a raíces estrictamente autorizadas (`ws_root` y directorios de datasets registrados), eliminando `ws_root.parent` y `Path.cwd()` para impedir accesos no autorizados al directorio padre del usuario o del proyecto.
    - [x] Comprobación estricta de pertenencia a directorio para `/static/` con `target_file.is_relative_to(static_root)` y respuesta `403 Forbidden` ante intentos de path traversal.
    - [x] Suite de tests de seguridad dedicada en `tests/test_workbench_file_confinement.py` (4 tests). 639 tests pasando (100% verde), 87.39% de cobertura.
  - [x] **PR C (PR #82) — Atomicidad Transaccional de Migraciones SQLite**:
    - [x] Transacción explícita completa y rollback verificado en migraciones SQLite (`SQLiteExperimentRepository._migrate` y `SQLiteJobRepository._migrate`).
    - [x] Conteo de registros previo y posterior (`post_count == pre_count`) y limpieza de tablas temporales.
    - [x] Tests unitarios con rollback probado ante excepciones forzadas en `tests/test_sqlite_legacy_migrations.py`.
    - [x] 641 tests pasando (100% verde), 87.41% de cobertura.
  - [x] **PR D (PR #83) — Consistencia del Producto (Dirección de Métricas, Benchmarks y Actividad Real)**:
    - [x] Unificar dirección de métricas en benchmarks (`is_minimizing_metric`) y selección del mejor modelo según la métrica (`min` para pérdida, `max` para ganancia).
    - [x] Erradicación total de mock data en Workbench (`_build_real_activity_feed` en `server.py` y carga dinámica de `agent_ledger.db` en `/api/agent/hypotheses`).
    - [x] Tests unitarios en `tests/test_benchmark_metric_direction.py` y `tests/test_workbench_activity_consistency.py`.
    - [x] 651 tests pasando (100% verde), 87.32% de cobertura.
  - [x] **PR E (PR #84) — Hardening Complementario (Protección XSS, Alineación de Métricas en Workbench y Confinamiento Estricto de Descargas)**:
    - [x] Protección XSS en frontend: módulo `src/automl/interfaces/web/static/js/utils.js` con `escapeHtml` y sanitización integral en `agent.js`, `overview.js` y `kaggle.js`.
    - [x] Segregación de métricas en `/api/overview` (`server.py`): eliminación de comparaciones escalares entre métricas incomparables; reporte contextualizado (`best_metric`, `best_score`, `best_model`) y desglose completo por métrica (`best_by_metric`).
    - [x] Benchmarks: cálculo de `delta_vs_baseline` en `BenchmarkRunner` con sentido de mejora positivo ($\Delta > 0$) tanto para métricas de pérdida (reducción de error) como de ganancia.
    - [x] Confinamiento estricto de `/api/kaggle/download`: restricción exclusiva al subdirectorio `submissions/` o artefactos de exportación autorizados (`exports/`), bloqueando rutas relativas a la raíz del workspace o extensiones no pertinentes.
    - [x] Suite de tests dedicada en `tests/test_workbench_xss_protection.py` (5 tests) y ampliación de `tests/test_workbench_file_confinement.py` (4 tests).
    - [x] 656 tests pasando (100% verde), 87.33% de cobertura.
  - [x] **PR G (PR #86) — Flexibilidad Anti-Leakage en Planificación e Integridad de Métricas en Workbench**:
    - [x] Soporte para `strict=False` en `DatasetProfile.resolve_safe_feature_names()` y uso en `RuleBasedExperimentPlanner.propose()` para tolerar datasets con columnas mixtas de leakage y predictoras.
    - [x] Segregación de métricas por dataset en `/api/overview` (`best_by_dataset`), soporte para scores `0.0` en hipótesis y valores iniciales limpios.
    - [x] Suites de pruebas en `tests/test_anti_leakage_guardian.py` y `tests/test_workbench_activity_consistency.py`.
    - [x] 664 tests pasando (100% verde), 87.32% de cobertura.
- [x] **Fase E1 — Núcleo de Estudios (Explore Core)** [Prioridad P0]:
  - [x] Entidades de dominio puras en `src/automl/domain/analysis/` (`StudySpec`, `AnalysisRun`, `StatisticalFinding`, `VisualizationSpec`, `EvidenceLink`, `AnalysisHypothesis`).
  - [x] Soporte de estudios sin columna objetivo (`target_column=None`, modo no supervisado).
  - [x] Esquema de persistencia SQLite aditivo en `src/automl/infrastructure/database/sqlite_studies.py` con WAL mode, claves foráneas e índices.
  - [x] Servicio de aplicación `AnalysisStudyService` y operaciones CQRS (`CreateStudyCommand`, `RunAnalysisCommand`, `ArchiveStudyCommand`, `GetStudyQuery`, `ListStudiesQuery`, etc.) modularizados en `src/automl/application/registries/analysis_registry.py`.
  - [x] Motor de análisis estadístico determinista en `src/automl/engine/analysis/statistical_analyzer.py` (colinealidad de Pearson $r \ge 0.70$, separación ANOVA $p < 0.05$, outliers IQR, cardinalidad e hipótesis de mitigación).
  - [x] CLI `automl explore create/run/list/show/findings` registrado en `src/automl/interfaces/cli/explore_cli.py` y `main.py`.
  - [x] Herramientas MCP en `src/automl/interfaces/mcp/analysis_tools.py` (`analysis_create_study`, `analysis_run_study`, `analysis_get_findings`, `analysis_list_studies`, `analysis_get_study`).
  - [x] Endpoints REST `/api/analysis/...` en `server.py` e interfaz visual verídica en Workbench (`datasets.js` y `api.js` con selector de estudios, matriz de correlación real calculada y tarjetas de hallazgos).
  - [x] Suite de pruebas automatizadas completa en `tests/test_explore_phase_e1_core.py` (10 pruebas pasando al 100%).
- [x] **Fase E2 — Motor Estadístico Avanzado (Advanced Statistical Engine)** [Prioridad P0]:
  - [x] Motor numérico modular en `src/automl/engine/analysis/` (`distribution_diagnostics.py`, `association_metrics.py`, `hypothesis_testing.py`, `statistical_analyzer.py`).
  - [x] Diagnóstico de distribuciones: pruebas de normalidad (D'Agostino-Pearson y Shapiro-Wilk), asimetría (skewness), curtosis, detección de multimodality (coeficiente bimodal de Sarle $BC > 0.555$) y outliers multivariantes (distancia de Mahalanobis regularizada con corte crítico $\chi^2_{p < 0.001}$).
  - [x] Matrices de asociación bivariadas: correlación de Pearson con intervalos de confianza de Fisher al 95% ($z$-transform), correlación monótona de Spearman ($\rho$), contraste de relaciones no lineales e hipótesis de modelos basados en árboles/splines.
  - [x] Asociaciones categóricas e inferencia no lineal: matriz de Cramér's V con corrección de sesgo de Bergsma-Wicher, pruebas Chi-cuadrado de independencia e hipótesis de interacciones categóricas.
  - [x] Pruebas de hipótesis automáticas supervisadas: comparación de 2 grupos con t-test de Welch y Mann-Whitney U junto con tamaño del efecto Cohen's $d$; comparación multiclase ($>2$ grupos) con ANOVA unidireccional y Kruskal-Wallis junto con tamaño del efecto $\eta^2$ (Eta al cuadrado); prueba de homocedasticidad de Levene.
  - [x] Corrección de comparaciones múltiples: ajuste por Benjamini-Hochberg (FDR - False Discovery Rate) sobre todos los tests inferenciales del estudio para mitigar falsos descubrimientos (*data dredging*).
  - [x] Batería de pruebas dedicada en `tests/test_explore_phase_e2_advanced_engine.py` (11 pruebas pasando al 100%, 88% de cobertura en `engine/analysis`).
- [x] **Fase E3 — Laboratorio Visual Interactivo (Workbench Exploration UI)** [Prioridad P1]:
  - [x] Generador backend de `VisualizationSpec` declarativo en `src/automl/engine/analysis/visualizations/builder.py` con agregación de histogramas, resumen de cinco números para boxplots con muestreo acotado de outliers, y submuestreo de dispersión bivariada con regresión lineal de tendencia.
  - [x] Vista profesional en `src/automl/interfaces/web/static/js/views/explore.js` conforme al sistema de diseño Tech Minimalista Premium (ADR-006): selector de estudios, botones para nuevo estudio y ejecución de análisis, e integración en el router global (`app.js`).
  - [x] Galería de hallazgos estadísticos con filtros dinámicos por categoría, severidad y significancia Benjamini-Hochberg FDR ($p < 0.05$).
  - [x] Visualizador interactivo de especificaciones: matrices de correlación (Pearson y Spearman con conmutador dinámico), matriz de Cramér's V, histogramas SVG vectoriales, diagramas de caja (boxplots) y nubes de dispersión bivariada.
  - [x] Exportador de informes técnicos descargables en formato Markdown (`informe_explore_{study_id}.md`).
  - [x] Suite de pruebas automatizadas completa en `tests/test_explore_phase_e3_ui.py` (6 pruebas pasando al 100%).
- [x] **Fase E4 — Interoperabilidad MCP y CLI Completa (Agent Surface)** [Prioridad P0]:
  - [x] Tools MCP: `analysis_create_study`, `analysis_get_findings`, `analysis_get_visualizations`, `analysis_propose_experiment`.
  - [x] Paginación y control de presupuesto de tokens en llamadas MCP para agentes (`limit`, `offset`, `compact`, `envelope`).
  - [x] Recursos MCP de solo lectura: `catml://studies/{study_id}/summary` y `catml://studies/{study_id}/visualizations`.
  - [x] CLI completa `catml explore list/run/findings/export` (soporte de reporte técnico Markdown y JSON exportable).
  - [x] Generador desacoplado de informes técnicos en `src/automl/application/analysis/reporting.py`.
  - [x] Batería de pruebas automatizadas completa en `tests/test_explore_phase_e4_agent_surface.py` (7 pruebas pasando al 100%, suite global en 711 pruebas y cobertura 86.64%).
- [x] **Fase E5 — Agente Científico y Conexión con AutoML (Hypothesis Engine)** [Prioridad P1]:
  - [x] Mapeo determinista de hallazgos estadísticos a hipótesis de transformación ML (`HypothesisExperimentTranslator`).
  - [x] Entidad `EvidenceLink` y persistencia SQLite transaccional vinculando hipótesis con experimentos AutoML.
  - [x] Protocolo "Propose ≠ Accept": validación empírica en el mismo split antes de promover transformaciones o feature sets, preservando aprendizaje negativo.
  - [x] Paridad total de superficies: CQRS (`VerifyHypothesisCommand`, `GetEvidenceLinkQuery`, `ListEvidenceLinksQuery`), CLI (`automl explore verify`), MCP tool (`analysis_verify_hypothesis`) y recurso (`catml://studies/{study_id}/evidence`), y API REST (`/api/analysis/hypotheses/verify` y `/api/analysis/evidence`).
  - [x] Batería de pruebas automatizadas completa en `tests/test_explore_phase_e5_hypothesis_engine.py` (6 pruebas pasando al 100%, suite global en 717 pruebas y cobertura 86.60%).
- [x] **Fase E6 — Benchmarks Integrales y Preparación de Producto** [Prioridad P1]:
  - [x] Batería de 4 ejes implementada en `ExploreBenchmarkRunner` (`src/automl/benchmarks/explore_benchmarks.py`):
    - [x] **Eje 1: Exactitud Estadística Numérica:** Calibración tipo NIST ante grandes desplazamientos numéricos ($10^9 + x$), Anscombe Quartet (I-IV con contrastes de no linealidad y outliers), intervalos Fisher $z$ al 95%, ajuste FDR Benjamini-Hochberg, Welch t-test / Cohen's $d$, y cotas de Cramér's V.
    - [x] **Eje 2: Calidad AutoML y Verificación de Hipótesis:** Comparativa determinista baseline vs hipótesis guiadas por Explore con persistencia de `EvidenceLink` en el mismo split y cumplimiento estricto de "Propose ≠ Accept".
    - [x] **Eje 3: Seguridad, Privacidad y Confinamiento de Tokens MCP:** Límites acotados de salida (`limit`, `offset`, `envelope`), stripping de matrices densas en `compact=True`, cero filtración de PII/datos crudos en recursos `summary`, y resistencia fail-closed ante inyecciones y path traversal.
    - [x] **Eje 4: Regresión de Esquemas y Workspaces:** Compatibilidad retroactiva de SQLite (`studies.db` en modo WAL), coexistencia de estudios no supervisados y supervisados, e integridad transaccional.
  - [x] Suite de pruebas automatizadas completa en `tests/test_explore_phase_e6_benchmarks.py` (6 pruebas pasando al 100%).
  - [x] Actualización de `DEVELOPER_GUIDE.md` con arquitectura, comandos de CLI y módulos de Explore.
  - [x] Suite global verde: **723 pruebas pasando (0 fallos)**, cobertura global sostenida en **86.76%** (umbral CI $\ge 85\%$).
- [x] **Hardening de Seguridad en Interfaces Remotas y Preparación Release 0.9.0-rc1**:
  - [x] **Fail-Closed Remoto por Defecto**:
    - [x] Workbench (`run_web_dashboard` en `src/automl/interfaces/web/server.py`) rechaza enlaces externos (`0.0.0.0` o no loopback) sin autenticación con `PermissionError` a menos que `CATML_ALLOW_INSECURE=1` esté explícitamente configurado.
    - [x] MCP streamable-http (`run_mcp_service` en `src/automl/interfaces/mcp/server.py`) rechaza enlaces externos no autenticados fail-closed con `PermissionError` a menos que `CATML_ALLOW_INSECURE=1` esté configurado.
    - [x] Retirada de impresión de URLs con query parameter `?token=`; conservación exclusiva de identificadores de fragmento del lado del cliente (`/#token=`), que nunca viajan por la red HTTP ni quedan en registros.
  - [x] **Guía y Modelo de Amenazas ("Trusted Deployment")**:
    - [x] Documento exhaustivo en [`docs/trusted-deployment.md`](docs/trusted-deployment.md) estructurado en 3 niveles (Tier 1: Local Loopback, Tier 2: Private LAN, Tier 3: Cloud VPS / Producción con proxy inverso TLS y permisos de workspace `chmod 700`).
    - [x] Advertencia prominente sobre deserialización de artefactos `joblib`/`pickle` en `README.md` y `SECURITY.md`, especificando que el checksum SHA-256 garantiza integridad ante corrupción accidental, no autenticidad criptográfica ante adversarios.
    - [x] Hoja de ruta para firmas HMAC/PKI y exportación a ONNX para inferencia pura.
  - [x] **Superficie de Agentes y Privacidad**:
    - [x] Modo compacto `compact=True` por defecto en herramientas de análisis MCP, devolviendo cero filas crudas a contextos LLM.
    - [x] Presupuestos cooperativos por sesión y confinamiento estricto a las raíces del workspace.
  - [x] **Bump de Versión a 0.9.0-rc1**:
    - [x] Actualización de `__version__ = "0.9.0-rc1"` en `src/automl/__init__.py`.
    - [x] Registro en `CHANGELOG.md` con desglose detallado de CATML Explore (E1–E6) y security hardening.
    - [x] Actualización de badges y enlaces de navegación en `README.md` y `SECURITY.md`.
    - [x] Suite de pruebas automatizadas: 727 pruebas pasando al 100%, cobertura global $\ge 86.8\%$.
- [x] **Track Workbench UI Refinements & Testing (`feat/workbench-ui-refinements`)**:
  - [x] Sincronización dinámica de versión en cabecera y sidebar de `index.html` e inyección reactiva en `app.js` (`sidebarVersionLabel`, `sidebarAgentBadge`) a partir del endpoint `/api/overview`.
  - [x] Normalización canónica de rutas y clases activas en `_routeTo` de `app.js` para alias de navegación (`datasets` -> `dataset`, `studio` -> `experiments`, `overview` -> `evidence`, `explore`).
  - [x] Conexión de acceso directo desde la pestaña de Exploración de `datasets.js` al lienzo completo de `explore.js` mediante el botón `btnOpenFullExploreCanvas` ("Lienzo Completo").
  - [x] Armonización de idiomas y eliminación de cadenas residuales en inglés en `studio.js` y `knowledge.js`.
  - [x] Suite de pruebas automatizadas dedicada en `tests/test_workbench_navigation_and_explore_ui.py` (4 tests pasando) y suite de UI de 44 tests al 100%.

## Completed Roadmap: Experiment Recipes (Explore → Evidence → Reusable Config)

> Spec: [`docs/features/experiment-recipes/spec.md`](docs/features/experiment-recipes/spec.md) · Plan: [`docs/features/experiment-recipes/plan.md`](docs/features/experiment-recipes/plan.md) · ADR: [`docs/decisions/009-experiment-recipes-and-interface-parity.md`](docs/decisions/009-experiment-recipes-and-interface-parity.md).  
> Principio: paridad Python / CLI / REST / MCP / Workbench; UI última capa.  
> Estados: **Proposed → Verified → Selected → Promoted** (`EvidenceLink.accepted` ≠ autorización automática).

- [x] **Fase R0 — Verificación real de transforms** [P0]:
  - [x] Aplicar de verdad `log1p` / `yeo_johnson` / `robust_scaler` (y drops de colinealidad) en `VerifyHypothesisCommand`, no solo metadata del traductor.
  - [x] Distinguir derivadas sin estado (`derived_features`) vs transforms con estado (`column_transforms`, fit-in-fold en pipeline de entrenamiento).
  - [x] Persistencia en SQLite (`column_transforms_json`, `derived_features_json`), soporte en `TrialExecution` y `Experiment`.
  - [x] Suite de pruebas dedicada en `tests/test_recipe_phase_r0_verify_transforms.py` (6 pruebas pasando al 100%).
- [x] **Fase R1 — Dominio `ExperimentRecipe` + CQRS** [P0]:
  - [x] Entidad reproducible en `domain/recipes/`: `ExperimentRecipe` con transforms, columnas, dataset/hash, modelo/params, validación, métrica, semilla, `evidence_link_id`, status (`draft`/`selected`/`submitted`/`promoted`/`archived`).
  - [x] Persistencia relacional aditiva SQLite en `sqlite_recipes.py` (`SQLiteRecipeRepository`).
  - [x] Comandos CQRS: `CreateExperimentRecipeFromEvidenceCommand`, `SubmitExperimentFromRecipeCommand`, `SelectExperimentRecipeCommand`, `PromoteExperimentRecipeCommand`, `ArchiveExperimentRecipeCommand`.
  - [x] Consultas CQRS: `ListVerifiedImprovementsQuery`, `GetExperimentRecipeQuery`, `ListExperimentRecipesQuery`.
  - [x] Servicio de aplicación `RecipeService` orquestando ciclo de vida sin auto-promoción ("EvidenceLink.accepted ≠ Promoted").
  - [x] Registro modular en `recipe_registry.py` y `bootstrap.py`.
  - [x] Suite de tests dedicada en `tests/test_recipe_phase_r1_domain_and_cqrs.py` (5 pruebas pasando al 100%).
- [x] **Fase R2 — Trainer fit-in-fold** [P0]:
  - [x] `RobustScaler` / `PowerTransformer` (yeo-johnson) ajustados dentro de cada fold de CV o split de train en `SklearnTrainer` (`_build_pipeline`).
  - [x] Preprocesamiento de derivadas en memoria previo a partición sin leakage.
  - [x] Tests de fit-in-fold y no-leakage en `tests/test_recipe_phase_r0_verify_transforms.py`.
- [x] **Fase R3 — Paridad Python / CLI / MCP / REST** [P0]:
  - [x] CLI (`catml recipe …` / `automl recipe …`): subcomandos `list-verified`, `create-from-evidence`, `list`, `show`, `select`, `submit`, `promote`, `archive` con soporte tabular y `--json`.
  - [x] Herramientas MCP para agentes en `src/automl/interfaces/mcp/recipe_tools.py` (`recipe_list_verified_improvements`, `recipe_create_from_evidence`, `recipe_get`, `recipe_list`, `recipe_select`, `recipe_submit`, `recipe_promote`, `recipe_archive`).
  - [x] Endpoints REST espejo en `src/automl/interfaces/web/server.py` (`GET /api/recipes/verified-improvements`, `GET /api/recipes`, `GET /api/recipes/{recipe_id}`, `POST /api/recipes/create-from-evidence`, `POST /api/recipes/select`, `POST /api/recipes/submit`, `POST /api/recipes/promote`, `POST /api/recipes/archive`).
  - [x] Principio "Propose ≠ Accept" garantizado: creación desde evidencia inicializa estado `DRAFT`; promoción requiere comando explícito con justificación auditable.
  - [x] Suite de paridad cruzada exhaustiva en `tests/test_recipe_phase_r3_parity.py` (4 tests al 100% pasando: CLI, MCP, REST y ciclo E2E cruzado). Cobertura total del proyecto en 86.88% (748 tests pasando).
- [x] **Fase R4 — Workbench + test de paridad** [P1]:
  - [x] UI para seleccionar mejoras verificadas, enviar recetas y promover (cliente del bus vía REST API, sin lógica de negocio en JS).
  - [x] Integración en `ExploreView`: acción "Crear Receta" en enlaces de evidencia de hipótesis aceptadas, tabla dedicada de recetas con acciones de ciclo de vida (Seleccionar, Entrenar, Promover, Archivar).
  - [x] Integración en `StudioView`: pestaña `Recetas` con renderizado de tarjetas, estado y acciones de ejecución/promoción vinculadas al run.
  - [x] Integración en `NewExperimentModal`: selector reproducible de recetas para ejecución directa con 1-click.
  - [x] Suite de paridad multi-superficie cruzada en `tests/test_recipe_phase_r4_workbench_parity.py` (3 tests al 100%: JS client hooks, ciclo de vida REST en servidor real HTTP, y prueba de 4 superficies CLI → MCP → Workbench → REST con idénticos IDs, transformaciones in-fold y linaje). Cobertura global sostenida en 86.88% (751 tests pasando).

## Completed Phase: v0.8.2 Phase 1 (Operational Time Budget, Fail-Closed Artifacts & Remote Workbench Security)

- [x] **Cumplimiento Real y Operacional de `time_budget`**:
  - [x] Añadido campo formal `time_budget_seconds: float | None` a `RunConfig` con propiedad `effective_time_budget` retrocompatible con `extra["time_budget"]`.
  - [x] Propagación formal en `AutoML.fit()` pasando `time_budget_seconds=self.time_budget` a `create_run()`.
  - [x] Control de deadline por `run_id` (`_is_run_budget_exhausted`) en `AutoMLWorkspace.run_experiment()` y `run_scheduled_experiments()`.
  - [x] Descarte cooperativo y emisión del evento `TimeBudgetExhausted` sin invalidar los modelos ya completados.
  - [x] Manejo inteligente en `AutoML.fit()`: si ningún modelo pudo completar antes del tiempo, falla de forma explícita (`RuntimeError`); si al menos uno completó, finaliza con éxito registrando `time_budget_exhausted=True`.
- [x] **Artefactos Estrictamente Fail-Closed**:
  - [x] En `ModelArtifact.save()`: eliminación de `except Exception: pass`, garantizando que la creación y escritura del digest SHA-256 no falle en silencio.
  - [x] En `ModelArtifact.load(verify_checksum=True)`: bloqueo inmediato (`ValueError`) ante sidecar inaccesible, ilegible, corrupto o con hash no coincidente, impidiendo la deserialización no segura mediante `joblib.load()`.
- [x] **Seguridad en Interfaces Remotas del Workbench (`0.0.0.0`)**:
  - [x] Preservado acceso local transparente sin token para `127.0.0.1` y `localhost`.
  - [x] Para cualquier interfaz no local (ej. `0.0.0.0`), requerir `--auth-token` o auto-generar token seguro efímero (`secrets.token_urlsafe(16)`) con URL directa en consola.
  - [x] Validación de token en `AutoMLWebHandler` vía header `Authorization: Bearer <token>` o query parameter `?token=<token>`, respondiendo `401 Unauthorized` si falta o es inválido (salvo flag explícito `--insecure-no-auth`).
  - [x] Adaptador de cliente `CATMLApiClient` en `api.js` persistiendo el token en `sessionStorage` para mantener la sesión web activa.
- [x] **Cierre de Fuga CQRS**:
  - [x] Método público `AutoMLWorkspace.list_runs(dataset_id=...)` implementado.
  - [x] Eliminado acceso al atributo privado `ws._runs` en `src/automl/interfaces/cli/main.py` (`predict_cli`).
- [x] **Batería de Pruebas de Fase 1**:
  - [x] `tests/test_time_budget_enforcement.py` (4 tests).
  - [x] `tests/test_model_artifact_fail_closed.py` (3 tests).
  - [x] `tests/test_workbench_auth.py` (3 tests).

## Completed Phase: v0.8 Release Readiness, Packaging & Governance
  - [x] Corregir `cv_folds` en `src/automl/facade.py`: `AutoML.fit` debe respetar `self.cv_folds` propagándolo a `create_run` y usando estrategia de validación `"cv"` o `"kfold"` cuando no exista columna de agrupación.
  - [x] Endurecer `src/automl/engine/vision/image_encoder.py`:
    - Eliminar fallback silencioso a hash determinista: si falla PyTorch/Pillow/backbone en un modelo neural, lanzar `RuntimeError` explícito ("Fail visibly, don't fake vision").
    - Reservar hash determinista exclusivamente cuando `model_name="deterministic"` (para tests/benchmarks reproducibles).
    - Eliminar fallback silencioso a red aleatoria no inicializada cuando fallan los pesos preentrenados.
    - Preservar dimensiones nativas de embeddings sin truncamiento arbitrario destructivo (default native feature dimension 512).
  - [x] Endurecer `src/automl/artifacts/model_artifact.py`:
    - Advertencia de seguridad en `ModelArtifact.load` contra ejecución arbitraria de código por deserialización no confiable.
    - Generación y verificación de checksum SHA-256 en streaming por bloques (`chunk_size=65536`) en guardado y recarga.
    - Documentar distinción técnica entre integridad (detección de corrupción accidental) y autenticidad criptográfica (firmas digitales PKI).
  - [x] Inmutabilidad en capa de dominio (`src/automl/domain/problems/spec.py`):
    - `@dataclass(frozen=True)` en `ProblemSpec` con conversión a tupla inmutable en `__post_init__` y test de regresión.
- [x] **Paquete 2: Packaging, Extras & Naming**:
  - [x] Renombrar paquete en `pyproject.toml` de `automl-platform` a `catml`.
  - [x] Unificar comandos y aliases CLI (`catml` primario, `automl` retrocompatible).
  - [x] Reestructurar extras de instalación en `pyproject.toml`: `models`, `vision`, `nlp`, `mcp`, `agents`, `all` (incluyendo dependencias de visión en `all` y eliminando `full` redundante).
- [x] **Paquete 3: Claridad, Transparencia Técnica y Gobernanza**:
  - [x] Actualizar `README.md`: cambiar "zero-dependency inference" por "Database-free, workspace-independent inference", badges (595 tests passing, Apache-2.0), CLI V0.8.0, Tech Minimalista Premium workbench.
  - [x] Alinear honestidad de plugins de visión: renombrar `TimmVisionPlugin` a "Vision Feature Head (MLP on Extracted Embeddings)" y en UI a "Vision Feature MLP".
  - [x] Sincronizar hoja de ruta en `docs/MASTER_PLAN.md`: marcar capacidades completadas como `IMPLEMENTADO (SHIPPED IN 0.8)` y fases futuras como `v0.8.x` y `v1.0+`.
  - [x] Crear gobernanza y release engineering: `SECURITY.md`, `CHANGELOG.md` (formato Keep a Changelog) y `TRADEMARK.md` (política de marca CATML).
- [x] **Paquete 4: Pulido Final para Release v0.8.0 & Smoke Test Automatizado**:
  - [x] Corregir detección de dependencias en `ImageModalityPlugin.available()` requiriendo `torchvision` explícitamente junto con `torch` y `PIL`.
  - [x] Modernizar resolución de pesos en `ImageEncoderNode` con fallback elegante a API moderna de TorchVision (`DEFAULT`) y mensaje informativo para entornos sin conexión (offline/air-gapped).
  - [x] Atributo aprendido `feature_dim_` en `ImageEncoderNode` para preservar contratos en `get_feature_names_out`.
  - [x] Añadir alias `VisionFeatureHeadPlugin` para desacoplar el concepto de representación visual del downstream head.
  - [x] Packaging en `pyproject.toml`: migración de licencia a string SPDX `Apache-2.0` (eliminando warning de setuptools), metadatos `project.urls` y limpieza de extra `nlp`.
  - [x] Script de validación de release `tests/smoke_test.py` y nuevo job de CI `package-smoke` para verificar instalación en entorno virtual aislado.
- [x] **Paquete 5: Trust Patch (v0.8.1) — Seguridad Local, Anti-Leakage y Reproducibilidad**:
  - [x] Fail-safe para leakage: `AutoML.fit()` y `RuleBasedExperimentPlanner` priorizan `profile.recommended_feature_names`, excluyendo automáticamente variables contaminadas.
  - [x] Workbench seguro: bind exclusivo a `127.0.0.1` por defecto (con flag `--host 0.0.0.0` para despliegue explícito) y eliminación de cabeceras permisivas `Access-Control-Allow-Origin: *`.
  - [x] Confinamiento de `/api/media/preview`: validación estricta de que las imágenes residan dentro del workspace o directorio del dataset (`is_relative_to`), bloqueando lectura arbitraria del sistema.
  - [x] Propagación de `random_state`: cableado desde el constructor de `AutoML` hasta `RunConfig.random_seed` y todos los splitters de validación cruzada y estimadores.
  - [x] Ordenación correcta de métricas de pérdida: `AutoMLResult.leaderboard()` y `SQLiteExperimentRepository.get_leaderboard()` ordenan ascendentemente para `mae`, `rmse`, `mse` y `loss`.
  - [x] Eliminación de evidencia simulada: supresión de `public_lb = score * 0.9999` y deltas ficticios en endpoints de benchmark.
  - [x] Hero honesto y simplificado en `README.md`: centrado en confianza, protección contra fugas, reproducibilidad y acceso controlado para agentes.
  - [x] Suite de tests `tests/test_v081_trust_patch.py` (4 tests pasando) y actualización de versión a `0.8.1`.


## Completed Phase: v0.9 Capability Layer, Lightweight NLP & Advanced Anti-Leakage Guardian

- [x] Contratos puros de dominio para la Capa de Capacidades en `src/automl/domain/problems/`: `ProblemSpec`, `TabularSource`, `TextSource`, `ImageSource`, `TargetSpec`, `ValidationSpec`, `ExecutionPolicy`, `BackendCapabilities`, y `EvaluationResult` (100% stdlib, sin dependencias externas).
- [x] Motor de extracción y tokenización ligera de texto en `src/automl/engine/features/text/`: `LightweightTextExtractor` (TF-IDF sublineal con n-gramas) y heurística `is_text_column` para detección automática de lenguaje natural.
- [x] Adaptación del profiler en `src/automl/engine/profiling/dataset_profiler.py`: discriminación de texto frente a identificadores no predictivos, badge `Text Feature` y recomendación `nlp_encode`.
- [x] Fusión multimodal en `src/automl/engine/training/sklearn_trainer.py`: integración transparente de transformadores de texto en `_build_pipeline` y `fit_pipeline` con salida densa unificada.
- [x] Soporte en fachada ergonómica `AutoML.fit(df, target="col", text_columns=["notes"])` con serialización, proveniencia e inferencia desacoplada en `ModelArtifact`.
- [x] Validación completa: suite `tests/test_v09_capability_layer.py` (6 tests pasando); suite global en verde: 553 tests pasando, 87.71% cobertura de código (Decision Gate de v0.9 alcanzado).
- [x] **Fase 3: Anti-Leakage Guardian Avanzado (Group & Entity Leakage)**:
  - Detección de entidades/grupos repetidos (`patient_id`, `user_id`, `device_id`, etc.) en `detect_is_group_candidate` discriminando entre identificadores 1-a-1 e identificadores grupales de entidad.
  - Diagnóstico de fuga por grupos en particiones simuladas de validación `detect_group_leakage` calculando intersección de entidades, filas contaminadas y recomendación determinista `GroupKFold(col)`.
  - Integración en `DatasetProfile` (`is_group_candidate`, `group_candidates`, `has_group_leakage`, `group_leakage_reports`, badge `Group Leakage`, acción `enforce_group_split`).
  - Soporte de validación `GroupKFold` y `GroupShuffleSplit` en `ValidationSpec`, `TrialExecution`, `SklearnTrainer`, `CreateExperimentCommand` y `RuleBasedExperimentPlanner`.
  - Protección automática y explícita en la fachada `AutoML.fit(df, target="churn", group_column="patient_id")`: exclusión de variables de grupo de la matriz predictiva `X` para prevenir memorización y aplicación automática de partición por grupos ante detección de fuga.
- [x] **Fase 4: Vision Spike (Prueba de Estrés Arquitectónica & Visión Multimodal)**:
  - Heurística `is_image_column` en `src/automl/plugins/modalities/image_plugin.py` para detección de rutas de archivos de imagen por extensiones (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, etc.) o presencia en disco.
  - Profiler adaptado en `src/automl/engine/profiling/dataset_profiler.py`: discriminación de columnas de imagen evitando falsos positivos de identificadores (`is_identifier=False`), detección de `is_image`, propiedad `image_column_names`, y recomendación con badge `Image Feature` y acción `image_encode`.
  - Nodo de ejecución de imágenes `ImageEncoderNode` en `src/automl/engine/vision/image_encoder.py`: herencia de `BaseEstimator, TransformerMixin`, compatibilidad de slicing para DataFrames 2D y Series 1D, manejo de valores nulos/faltantes mediante vector cero (`handle_missing="zero"`), y `get_feature_names_out`.
  - Fusión multimodal en `src/automl/engine/training/sklearn_trainer.py`: integración transparente de transformadores de imagen en `_build_pipeline` y `fit_pipeline`, filtrado de parámetros de preprocesador frente a hiperparámetros de modelo, y soporte para cancelación cooperativa.
  - Plugin especializado de visión `TimmVisionPlugin` en `src/automl/plugins/models/vision_plugin.py` con interfaz de capacidades (`available()`, `requirements()`, `capabilities()`, `install_instructions()`) y degradación suave (`fallback`).
  - Dependencias opcionales organizadas en `pyproject.toml` (`catml[vision]`, `catml[nlp]`, `catml[all]`).
  - Fachada ergonómica `AutoML.fit(df, target="col", image_columns=["img_path"])` y exportación desacoplada en `ModelArtifact` con proveniencia de librerías de visión e inferencia reproducible vía `predict()`.
  - Validación completa en `tests/test_v09_vision_spike.py` (11 tests nuevos pasando, 570 tests globales en verde, 87.69% cobertura de código). Decision Gate de Fase 4 superado.
## Completed Phase: v0.8 Hardening & Strategic Governance

- [x] Hardening Blackboard Issue #52: resolución de timeout en `test_mcp_cli_subprocess_stdio_handshake` con terminación limpia de proceso (`proc.kill()`) y timeout calibrado a 25s; [issue #52](https://github.com/Jfenic/catml/issues/52) cerrado.
- [x] Proveniencia Completa en `ModelArtifact`: persistencia automática de versiones de dependencias, Python, hashes SHA-256 del dataset, estimador y semilla aleatoria; método `artifact.describe()` e informe en `docs/backends.md` (547 tests pasando, 87.59% cobertura).
- [x] Spec: Complete Tech Minimalista Premium Brand System specification saved in `docs/design/tech-minimalist-brand-system.md` (Electric Blue `#4F67FF`, Graphite, 80/20 rule, Geist typography, light docs/dark product).
- [x] ADR: Formalize architectural decision in `docs/decisions/006-tech-minimalist-premium-brand-system.md` (and mirrored in `catml-platform/docs/decisions/002-tech-minimalist-premium-brand-system.md`).
- [x] Rule 8: Update UI & brand standards in `AGENTS.md` to enforce Tech Minimalista guidelines across all agents and developers.
- [x] Theme Tokens & Workbench CSS: Harmonize CSS tokens (`--catml-*`, colors, typography, borders, shadows) in `src/automl/interfaces/web/static/css/workbench.css` with the new design system.
- [x] Platform Sync: Mirror brand system specifications, tokens and roadmap milestones into `catml-platform`.
- [x] Standalone Artifact UI & Color Cleanup: Expose `/api/models/export` (.pkl download) and `/api/models/export-info` in `server.py`, integrate 1-click artifact download buttons in `overview.js` and `compare.js`, and eliminate legacy `#E5512D` colors across all views in favor of `#4F67FF` Electric Blue (437 tests passing, 87.23% coverage).
- [x] UI Refinement: Comprehensive transformation of Workbench frontend and dataset profiler to 100% Tech Minimalista Premium (4-level surface hierarchy `#080A0F`/`#0D1017`/`#11151E`/`#161B26`, monochrome sidebar with muted 01-06 indicators, Geist/Inter sans-serif UI, discrete badges, English unification, clean Agent drawer, 461 tests passing, 87.53% coverage).
- [x] Workbench Elevation & Container De-nesting: Elimination of nested container soup (`panel -> panel -> control -> badge`) across Dataset Understanding, studio, and modals; segmented pill track for filters; clear visual action hierarchy with dominant primary CTA "Launch Experiment (N)" and secondary "Apply recommendations"; 100% professional technical English across all views and background jobs widget; discrete badges (`HIGH RELEVANCE`, `HIGH SIGNAL`); full test suite green (461 passed, 87.53% coverage).
- [x] Experiments Studio Semantic UX & Flow Coherence: Status-dependent action controls (Completed -> Run again / Clone / Export best model; Running -> Pause / Stop; Paused -> Resume / Stop); reduced height and informative empty state for AutoML Plan & Decisions; discrete trial cards when < 4 trials (curve reserved for >= 4 trials); explicit "New run preset" labels; Agent-assisted Guided Experiment badge; zero mixed Spanish/English.
- [x] Lucide Vector Icon System: Eliminate 100% of Unicode emojis (🏆, 📊, 🧠, 🚀, ⚡, ⚗, 🎯, 💡, 📦, ⚠️, etc.) and replace with zero-dependency Lucide SVG icon system in `src/automl/interfaces/web/static/js/icons.js` with standardized sizing (.icon, .icon-sm, .icon-lg, .icon-xl) and semantic token colors (#8B95A7 default, #4F67FF active/accent, #22C55E success, #F59E0B warning, #EF4444 danger, #6956E8 agent); verified 0 emoji occurrences across all web static assets (PR #56, 461 tests passing, 87.53% coverage).
- [x] ADR 007: Decision-Gated Multimodal Governance Roadmap formalizado en `docs/decisions/007-decision-gated-multimodal-governance-roadmap.md` (definición: *Controlled Agentic Toolkit for Machine Learning*, desacoplamiento `ProblemSpec` y `BackendCapabilities`, anti-leakage por grupos/entidades, vision spike gate y gobernanza de agentes con data egress policy).

## Completed Phase: Sprint 2 (Productization P0 — CLI Fit, Quickstart & Product README)

- [x] Feature: CLI Command `catml fit <dataset> --target <col>` (`src/automl/interfaces/cli/main.py`) with leaderboard output and artifact generation.
- [x] Feature: `examples/quickstart.py` (2-minute end-to-end runnable script with dataset generation and standalone inference).
- [x] Docs: Complete product-first redesign of `README.md` (value proposition, 4-line quickstart, MCP agent configuration, architecture).
- [x] Tests: CLI and quickstart automated tests (`tests/test_cli_fit.py`) (436 tests passing, 87.39% coverage).

## Completed Phase: Sprint 1 (Ergonomic Facade & Model Artifacts)

- [x] Feature: Standalone `ModelArtifact` (`src/automl/artifacts/model_artifact.py`) with `save(path)` and `load(path)`.
- [x] Feature: `AutoML` and `AutoMLResult` facade (`src/automl/facade.py`) supporting `fit(df, target=...)` and `fit(X, y)`.
- [x] Feature: Top-level exports in `src/automl/__init__.py` and alias package `src/catml/`.
- [x] Dependency: Add `joblib>=1.3` to `pyproject.toml`.
- [x] Tests: Comprehensive test suite in `tests/test_model_artifact.py` and `tests/test_facade.py` with full coverage >= 85% (432 tests passing, 87.36% coverage).

## Active Components (Workbench & Kaggle Playground Series S6E9)

- [x] Feature: CATML AutoML Workbench Server (`src/automl/interfaces/web/server.py`): Hexagonal interface adapter with zero external dependencies (`ThreadingHTTPServer`), exposing REST endpoints via `CommandBus`, `QueryBus`, and `AutoMLWorkspace` for Mission Control, run lifecycle controls (`pause`, `resume`, `cancel`, `clone`), dataset inspection, planner explicability, meta-learning knowledge, agent hypotheses, and Kaggle validation checklist.
- [x] Feature: Frontend Modular Architecture & Design Patterns (`src/automl/interfaces/web/static/`):
  - Architecture: EventBus (PubSub), WorkbenchStore (Observable/State pattern), CATMLApiClient (Adapter/Repository), Modular View Controllers.
  - Design System (`workbench.css`): Dense technical dark theme with semantic color system (🔵 system, 🟢 proven gain, 🟠 warning, 🔴 error/stop, 🟣 intelligence).
  - Views: Overview (Mission Control), Dataset Inspector ("¿Qué entendió CATML?"), Experiment Studio (explicable planner "Why this?", HPO, controls), Compare View (multi-select, metrics table, diff inspector), Visual Pipeline DAG (interactive execution nodes), Knowledge (V0.8 meta-learning preview), Lateral Agent Drawer (V1.0 hypothesis engine "Proponer ≠ Aceptar"), Kaggle S6E9 Center.
- [x] Refactorización Frontend Dinámico (Workbench UI): Eliminación completa de datos hardcodeados ("EV Purchases", "S6E9", "Will_Buy_EV", "competitions/playground-series-s6e9", métricas y 4 checkboxes estáticos). Todas las vistas (Overview, Datasets, Experiment Studio, Compare, Pipeline DAG, Kaggle, Knowledge y New Experiment Modal) se enlazan dinámicamente al workspace activo, runs reales y datasets registrados. Endpoint GET /api/datasets agregado a server.py.
- [x] Feature: Análisis Exploratorio Interactivo del Dataset, Gráficos Visuales y Selección Inteligente de Columnas (Workbench Web): Métricas estadísticas descriptivas (media, std, min, max, cuartiles, asimetría, correlación con target), previsualización de datos reales (primeras 8 filas), recomendaciones automáticas (identificadores, multicolinealidad, baja varianza, valores nulos), toolbar interactiva de selección de variables (filtros por tipo, búsqueda en vivo, selección rápida Top 5/10 por correlación con target, aplicación de recomendaciones con 1 clic), pestaña completa de Matriz de Correlación interactiva con mapa de calor (Pearson $r \in [-1, 1]$ y alertas de colinealidad), modal interactivo de análisis visual por variable con Diagrama de Caja y Bigotes (Box Plot comparativo desglosado por clases del target), histogramas de distribución con diagnóstico de asimetría, y tasa de propensión del target (%) por categoría. Ejecución de experimentos con características personalizadas y barra de acceso rápido en Studio.
- [x] Feature: Sistema de Diseño Neo-Industrial (Visual ML Lab) para CATML AutoML Workbench: Adopción y formalización de la dirección visual de laboratorio de ingeniería (especificación en `docs/design/neo-industrial-ui-spec.md` y ADR 005 en `docs/decisions/005-neo-industrial-visual-ml-lab-ui.md`), Regla 8 en `AGENTS.md`, tipografía `Space Grotesk` + `IBM Plex Mono`, paleta modular (`#111111` base carbón, `#16171c`/`#1c1d24` módulos, `#D8D6CF` cemento, `#F1EFE9` blanco cálido, `#E5512D` naranja señal reservado a acciones de ejecución), bordes arquitectónicos de 1px con radios rectos (0-4px), módulos de métricas de alta densidad y navegación lateral numerada (01-06).
- [x] Feature: CLI integration `automl ui [--port PORT] [--workspace WORKSPACE]` in `src/automl/interfaces/cli/main.py`.
- [x] Testing: Comprehensive web test suite in `tests/test_web_dashboard.py` verifying static asset serving, dataset registration, profiling, experiment execution, pause/resume, and submission generation.
- [x] Binary OOF prediction blending (`--folds 5`) with fixed equal weights, application commands/queries, CLI and Workbench; [scope and evaluation protocol](docs/features/oof-blending/spec.md).
- [x] Validate OOF blending on an independent holdout and representative benchmark before promoting a candidate.

- [x] Feature: Dataset Framing & Problem Context Questionnaire: Entidad de dominio `DatasetQuestionnaire`, servicio `QuestionnaireAdvisor` (heurísticas de churn, fraude, médico, crédito, series temporales, cohortes + modo LLM), persistencia SQLite en `dataset_questionnaires`, paridad en `AutoMLWorkspace` y endpoints REST en `server.py` (`GET/POST /api/dataset/questionnaire`). Suite en `tests/test_dataset_questionnaire.py` (PR #42).
- [x] Feature: Safe Feature Calculator & Derived Column Engine (`docs/features/derived-features/spec.md`): Motor robusto para generación de columnas complejas con doble modo: Fórmulas matemáticas AST seguras con lista blanca y aislamiento automático de división por cero (sin caídas), y código Python sandboxed restringido (`np`, `pd`, builtins seguros, auditoría AST anti-inyección). Especialista `FeatureAdvisor` para conexión de agentes LLM, validación previa (`POST /api/features/calculate`), persistencia en dataset (`POST /api/features/apply`), sugerencias asistidas (`POST /api/features/suggest`) y pestaña interactiva en Workbench UI (`datasets.js`). 22 tests en `tests/test_derived_feature_engine.py`.
- [x] Feature: Vision Support & Interactive Multimodal UI (Workbench Web): Endpoint seguro `/api/media/preview` en `server.py` con resolución por dataset o workspace y prevención anti-traversal; iconos vectoriales Lucide `image` y `camera` en `icons.js`; helper `getMediaPreviewUrl` en `api.js`; badges de tipo `Image` y acción `Vision` en esquema de datasets; filtro segmentado `Images (N)` en toolbar; miniaturas interactivas en previsualización de filas de datos; modal de análisis con pestaña **Sample Image Gallery** y tarjetas visuales con etiqueta target; nodo `ImageEncoderNode` en el grafo DAG interactivo de `pipeline.js`; y opción de modelo `Vision (timm)` en `new_experiment.js`. 4 tests dedicados en `tests/test_web_vision_support.py` (574 tests globales pasando, 0 fallos).

## Next (Roadmap Phases)

- [x] **CATML Benchmark Pilot 1A** (post-rc1): shared holdout; report [`docs/experiments/pilot-1a-report.md`](docs/experiments/pilot-1a-report.md).
- [x] **Fire tests (10)** — [`notebooks/fire_tests/`](notebooks/fire_tests/).
- [x] **CATML Benchmark v1 protocol frozen** (`protocol-v1.yaml` 1.0.0): shared holdout, seeds 42/43/44, f1_macro multiclass, skipped≠comparison.
- [x] **Study A campaign** (11 OpenML/sklearn datasets): 5 min smoke + 20 min formal **132/132 ok** → [`docs/experiments/study-a-report.md`](docs/experiments/study-a-report.md).
- [x] **Study B scaffold** — [`docs/experiments/study-b-scaffold.md`](docs/experiments/study-b-scaffold.md) / `run.py --study B`.
- [x] **Study B minimal runner** (sklearn / catml_base / catml_explore; agent skipped) → [`docs/experiments/study-b-report.md`](docs/experiments/study-b-report.md).
- [x] **Study B multiclass Explore alias** — `#119` (`multiclass` → `multiclass_classification`); smoke 18/6/0.
- [x] **Study B lift phase** — verify→recipe→fit; FakeLLM agent (`#122`); smoke 24/0/0.
- [x] **Data Preparation direction documented** — [spec](docs/features/data-preparation/spec.md), [plan](docs/features/data-preparation/plan.md), [ADR-012](docs/decisions/012-data-preparation-subsystem.md); implement post-rc1 (not blocking load→train).
- [x] **Data Preparation P1 (core)** — domain + `PreparationService` + CQRS + versions (`feat/data-preparation-p1`); CLI/facade = P2.
- [x] **Data Preparation P2 (interfaces & wiring)** — CLI (`automl data inspect/prepare`), Python facade (`AutoML.prepare` callable & methods), fold-safe `pipeline_steps_to_column_transforms` bridge, recipe parity, and full test suite (`tests/test_data_preparation_p2.py`).
- [x] **MCP `--allow-mutations` / autonomous local mode** — blackboard [#114](https://github.com/Jfenic/catml/issues/114) (PR `fix/mcp-allow-mutations-text-svd`; default stays require-approval).
- [x] **Text LSA (TruncatedSVD) after TF-IDF** — part of [#115](https://github.com/Jfenic/catml/issues/115) (MLP max_iter still open if needed).
- [x] Sincronizar el índice y los planes con las capacidades actuales (revisión 2026-10-11): H0–H5, Explore E0–E6, Experiment Recipes R0–R4, stacking/OOF multi-modelo y release `0.9.0-rc1` alineados en `AGENTS.md`, `docs/README.md`, feature plans/specs, ADR-009 y `MASTER_PLAN.md`.
- [x] Auditar `feat/workbench-live-experiments` (WIP sin PR) y ramas históricas: revisadas y preservadas como referencia histórica.

> **Sistema agéntico — empieza aquí:** [`docs/features/agentic-system/README.md`](docs/features/agentic-system/README.md). Elige Persona A o B; prepara A0 o B0 y acuerda H0 antes de pasar a A1/B1.

- [x] Documentar entrada para colaboradores, orden de lectura, elección A/B y primeras entregas en la guía del sistema agéntico.
- [x] Revisar y ampliar el plan agéntico: límites modulares, contratos propuestos, presupuesto/aprobación, recuperación, propiedad de integración e hitos H0–H5 documentados en `docs/features/agentic-system/plan.md`.
- [x] Concretar ejecución entre dos personas en [`docs/features/agentic-system/two-person-plan.md`](docs/features/agentic-system/two-person-plan.md): paquetes A0–A5/B0–B5, dependencias, archivos/tests propios, handoff y revisión cruzada; primera entrega conjunta H2.
- [x] H0 — Preparar V0.9: Contratos acordados y fusionados (PR #22 para B0 y PR #23 para A0 integrados en main).
- [x] Paquete A1 (Persona A): Catálogo y ejecutor de consultas seguras (`ToolRegistry`, `ToolExecutor`, `create_read_only_tool_registry`), blindaje de scope (prevención de issue #14 del blackboard), normalización a DTOs tipados vía `QueryBus` sin mutaciones, y suite de tests `tests/test_v09_agent_tools.py` con 100% de cobertura (5 tests passing).
- [x] Paquete B1 (Persona B): Servidor MCP stdio en `src/automl/interfaces/mcp/`, subcomando CLI `automl mcp`, tools de consulta (`list_models`, `get_dataset_profile`, `list_plugins`, `list_experiments`, `get_leaderboard`, `get_feature_evidence`, `get_feature_ranking`), resources (`catml://runs/...`, `catml://datasets/...`) y lifecycle subprocess con stdout limpio (10 tests passing en `tests/test_v09_mcp_server.py`).
- [x] H1 — V0.9 Consultas y MCP stdio: Catálogo de consultas A1 integrado con servidor MCP stdio B1.
- [x] Paquete A2 (Persona A): Tools mutantes (`create_experiment`, `prioritize_feature`, `run_experiment`), límites estrictos, flujo de aprobación persistente con verificación anti-tampering (`SqliteAgentLedger`), idempotencia y deduplicación atómica ("Duplicados no repiten efectos"), intención duradera antes de mutar, y suite de tests `tests/test_v09_agent_operations.py` con 16 tests passing (97% cobertura en módulos agénticos).
- [x] Paquete B2 (Persona B): Exposición y adaptación de mutating tools en servidor MCP (`create_experiment`, `prioritize_feature`, `run_experiment`), manejo estructurado de `PENDING_APPROVAL` sin bloquear terminal, comandos CLI de gobernanza (`automl agent approvals list`, `automl agent approve <id> [--reject]`) y verificación E2E (8 tests passing en `tests/test_v09_agent_cli.py` y `tests/test_v09_agent_e2e.py`, 11 tests en `tests/test_v09_mcp_server.py`).
- [x] H2 — V0.9 Mutaciones autorizadas e integración: Primera entrega conjunta local verificada integralmente (inspección MCP -> propuesta de candidato -> aprobación humana en CLI -> ejecución autorizada -> métricas en leaderboard consultables -> idempotencia probada).
- [x] H3 — V0.9 Operaciones largas, HPO y cancelación cooperativa (Integración conjunta A3 + B3):
  - [x] Paquete A3 (Persona A): HPO (`optimize_experiment`), reservas atómicas de presupuesto, leases por run, reconciliación de operaciones caídas y cancelación cooperativa (10 tests pasando en `tests/test_v09_agent_a3.py`, 16 tests en `tests/test_v09_agent_operations.py`). PR #35 integrado en main.
  - [x] Paquete B3 (Persona B): Inspección de estado y control de operaciones en CLI (`automl agent operations list`, `automl agent operations get`, `automl agent operations cancel`), tools MCP (`get_operation_status`, `list_operations`, `cancel_operation`), recursos MCP (`catml://runs/{run_id}/operations`, `catml://operations/{operation_id}`), transporte dual stdio / streamable-http en `automl mcp`, cumplimiento estricto de criterios de aceptación de H3 ("Timeout no se presenta como cancelación", "Operation ID recuperable" tras reinicio de proceso, flujo de cancelación cooperativa). 14 tests en `test_v09_agent_cli.py`, 14 tests en `test_v09_mcp_server.py`, 2 tests exhaustivos en `test_v09_agent_e2e.py`.

- [x] H4 — V1.0 Ciclo determinista y especialistas ("Proponer ≠ Aceptar") (Concurrencia aislada según two-person-plan.md §5.1):
  - [x] Contratos H4 acordados en `main`: DTOs de paso (`CandidateProposal`, `EvaluationFeedback`, `ContextPayload`, `SessionStepResult`).
  - [x] Track Persona A (Paquete A4): `ContextBuilder`, especialistas (`Planner`, `FeatureAdvisor`, `Critic`) en `src/automl/application/agents/specialists/`, proveedor determinista `FakeLLMProvider` en `src/automl/infrastructure/llm/` y tests en `tests/test_v10_specialists.py` (31 tests pasando con 94% de cobertura de paquete).
  - [x] Track Persona B (Paquete B4): Máquina de estados determinista en `src/automl/application/agents/orchestrator/`, CLI de sesión en `src/automl/interfaces/cli/agent_session_cli.py`, criterios de parada y tests en `tests/test_v10_orchestrator.py` (24 tests pasando con 92% de cobertura de paquete).
- [x] H5 — V1.0 Proveedor real, LangGraph duradero y recuperación (Integración final A5 + B5):
  - [x] Track Persona A (Paquete A5): Adaptador LLM agnóstico con validación de respuesta, timeouts/reintentos acotados, redacción y auditoría de tokens (PR #57).
  - [x] Track Persona B (Paquete B5): Checkpointer LangGraph SQLite, reanudación tras fallo y tests E2E de orquestación (`LangGraphAgentOrchestrator`, `SqliteCheckpointSaver`, CLI `--engine {deterministic,langgraph}`, 590 tests passing, 87.70% cobertura).
- [x] Phase 1 Meta-Learning Warm Starts (dataset fingerprint + Optuna trial-0 priors): completed — see Completed section.
- [ ] Advanced meta-learning *policies* (transform priors by fingerprint, richer warm-start beyond Phase 1): backlog.

- [x] Stacking with a trained meta-estimator (distinct from voting/blending).
- [x] Persist backend class, plugin/library versions and dataset/config hashes per trial ([current limits](docs/backends.md)).
- [x] Add the dedicated `feature_selection_v05` benchmark scenario from the feature-discovery specification.

### Generalist AutoML Engine & Super-Ensembles
> Technical Specs & Execution Plan: [`docs/features/generalist-enhancements/spec.md`](docs/features/generalist-enhancements/spec.md) and [`docs/features/generalist-enhancements/plan.md`](docs/features/generalist-enhancements/plan.md).

- [x] Phase 1: Generalized N-Model OOF Blending & Level-2 Stacking (remove 2-model limit, Rank-Averaging, Ridge meta-learner).
- [x] Phase 2: Temporal Dynamics Engine (automatic sequential detection, sensor lags, 24h trend deltas, cyclical projections, 19 tests passing in `tests/test_temporal_dynamics.py`, 528 suite total, 87.86% coverage).
- [x] Phase 3: Workbench Web UX & Visual Ensembles (Validation Strategy selector with Stratified K-Fold/K-Fold/TimeSeries/Holdout, Multi-Model Ensemble Builder with Average/Rank/Simplex/Stacked L2 and meta-learners Ridge/Logistic/Lasso, Kaggle Drag-and-Drop dropzone for sample_submission.csv with schema verification, and 1-click direct browser download; CQRS BuildEnsembleCommand, REST endpoints /api/ensemble/build, /api/kaggle/upload-template, /api/kaggle/download; 3 tests in tests/test_workbench_phase3.py, 546 suite total, >= 85% coverage).
- [x] Phase 4: Anti-Leakage Guardian in Profiler & Plugin Observability.

## Backlog — Mejoras Tabulares Identificadas (Kaggle Benchmarking)

> Evidencia de implementación y limitaciones: [`docs/README.md`](docs/README.md). Flujo de colaboración: [`CONTRIBUTING.md`](CONTRIBUTING.md).

- [x] Heurística automática de alta cardinalidad en `DatasetProfiler` (excluir IDs y texto masivo > 0.7 ratio de unicidad)
- [x] Plugin de Ensamble y Blending (`VotingEnsemblePlugin` / `VotingBlender`) mediante voting y media ponderada
- [x] Mapeo automático de plantilla de sumisión (`--template sample_submission.csv`) en `GenerateSubmissionCommand`
- [x] Generación automática de variables de interacción (ratios numéricos y target encoding)
- [x] Generación automática de diferencias y restas numéricas ($A - B$) en `InteractionFeatureGenerator` con soporte `include_differences`, conjunto de candidatos `interactions_differences` y transformaciones desacopladas (8 tests pasando en `tests/test_feature_interactions.py`)
- [x] Blackboard Issue #12: Soporte para CatBoost, Extra Trees y MLP en el Plugin System con fallback HistGradientBoosting y Optuna search spaces (11 tests passing en `tests/test_catboost_and_models_plugin.py`)
- [x] Optimización de Pesos de Ensamble y Rank Averaging: Optimización simplex de Nelder-Mead (`optimize_ensemble_weights`) con regularización Brier para métricas discretas, rank-averaging (`rank_average_predictions` y `voting="rank"`), y pruning multi-fidelidad en Optuna (`MedianPruner`, `report_step`, `pruned=True`). Cobertura de tests: 294 passing, 86.40% coverage.

## Completed

Los hitos describen el núcleo implementado; consultar [alcance y limitaciones](docs/README.md) antes de interpretar una fase como cumplimiento de todos los objetivos originales. Las cifras de pruebas de estos hitos son referencias históricas.

- [x] Phase 3: Workbench Web UX & Visual Ensembles (Validation strategy selector pills in NewExperimentModal, Visual Ensemble Builder modal in CompareView with average, rank, simplex Nelder-Mead, stacked meta-learners Ridge/Logistic/Lasso, Kaggle template drag-and-drop dropzone with schema verification, and direct 1-click submission CSV download; BuildEnsembleCommand in application/commands, workspace.build_ensemble in workspace service, REST endpoints /api/ensemble/build, /api/kaggle/upload-template, /api/kaggle/download, 3 tests in tests/test_workbench_phase3.py, 546 tests passing, >= 85% coverage).
- [x] Phase 1: Meta-Learning Warm Starts & Real Dataset Fingerprinting (`DatasetFingerprint` 6-dim vector, `MetaKnowledgeBase` cosine similarity & benchmark matching, empirical workspace trial ranking aggregation, trial 0 warm start in `OptunaOptimizer` and `RandomSearchOptimizer` yielding ~35-45% search reduction, CQRS `GetMetaKnowledgeQuery`, CLI `automl meta priors`, REST `/api/knowledge`, 8 tests in `tests/test_meta_learning_warm_starts.py`, 543 tests passing, 87.63% coverage).
- [x] Phase 4: Anti-Leakage Guardian & Plugin Observability (`has_leakage`, `leakage_column_names`, filtering from `recommended_feature_names`, $|r| \ge 0.999$ target correlation detection, sequential row leakage detection $|r_{\text{pos}}| \ge 0.95$, `is_native`, `fallback_backend`, `backend_status` observability in `LightGBMPlugin`, `XGBoostPlugin`, `CatBoostPlugin`, CLI `automl plugin list`, 6 tests passing in `tests/test_anti_leakage_and_observability.py`, 534 tests passing, 87.88% coverage).
- [x] Phase 2: Temporal Dynamics Engine (contracts in `src/automl/domain/features/temporal.py`, detection in `dataset_profiler.py`, generation in `temporal_generator.py`, CQRS `GenerateTemporalFeaturesCommand` / `DetectTemporalStructureQuery`, CLI `automl features temporal`, REST endpoints, 19 tests in `tests/test_temporal_dynamics.py`, 528 tests passing, 87.86% coverage).
- [x] Feature: `TargetAdapter` universal para compatibilidad con XGBoost/CatBoost y targets discretos textuales (`TargetAdapter`, `tests/test_target_adapter.py`, Issue #8 resuelto)
- [x] Phase V0.7: Multimodal Pipelines & Directed Acyclic Graph (`PipelineGraph`, `GraphValidator`, `FeatureFusionNode`, `ImageModalityPlugin`, `ImageEncoderNode`, `AutoMLWorkspace.build_multimodal_pipeline`, `fit_predict_multimodal`, 51 tests passing across `tests/test_v07_*.py`)
- [x] Track Dev 2: Image modality plugin & deterministic image encoder node (`ImageModalityPlugin`, `ImageEncoderNode`, unit & batch embeddings, 20 tests passing en `tests/test_v07_image_plugin.py`)
- [x] Feature: Mapeo automático de plantilla de sumisión Kaggle (`GenerateSubmissionCommand`, `PredictDatasetQuery`, `--template sample_submission.csv`, 5 tests passing en `tests/test_submission_template.py`)
- [x] Feature: Model plugin for voting ensemble & blending (`VotingEnsemblePlugin`, `VotingBlender`, soft voting and weighted predictions, 14 tests passing en `tests/test_ensemble_plugin.py`)
- [x] Feature: Heurística de alta cardinalidad e identificadores en `DatasetProfiler` (`detect_column_cardinality_and_role`, `is_identifier`, `is_high_cardinality`, `exclude_identifiers`, 12 tests passing en `tests/test_profiler_cardinality.py`)
- [x] Feature: Kaggle-ready inference & submission generator (`GenerateSubmissionCommand`, `PredictDatasetQuery`, CLI `automl predict`)
- [x] V0.6 Phase: Plugin architecture contracts (`PluginPort`, `ModelPluginPort`, `MetricPluginPort`, `PreprocessorPluginPort`)
- [x] V0.6 Phase: `PluginRegistry` in application layer and `CompatibilityValidator` in engine
- [x] V0.6 Phase: `LightGBMPlugin` and `XGBoostPlugin` with fallback to scikit-learn HistGradientBoosting (LightGBM) or GradientBoosting (XGBoost)
- [x] V0.6 Phase: Custom business metric plugins (`CostSensitiveMetricPlugin`, `WeightedF1MetricPlugin`)
- [x] V0.6 Phase: CLI command `automl plugin list` (table and JSON formats)
- [x] V0.6 Phase: Unit and integration test suite `tests/test_v06_plugins.py` (8/8 tests passing)
- [x] V0.5 Phase: `FeatureSelectorPort` and `FeatureEvidenceRepositoryPort` in `src/automl/domain/ports.py`
- [x] V0.5 Phase: Statistical and ML feature selectors (`MutualInfoSelector`, `TreeImportanceSelector`, `L1Selector`, `CorrelationSelector`, `VarianceSelector`, `EnsembleRankSelector`, `PCAReducer`) in `src/automl/engine/features/`
- [x] V0.5 Phase: `AblationPlanner` in `src/automl/engine/planning/` for leave-one-out feature candidate experiments
- [x] V0.5 Phase: CQRS commands/queries (`SelectFeaturesCommand`, `PlanAblationExperimentsCommand`, `PromoteCandidateFeatureSetCommand`, `GetFeatureEvidenceQuery`, `ListCandidateFeatureSetsQuery`, `GetFeatureRankingQuery`)
- [x] V0.5 Phase: Feature evidence persistence in SQLite repository
- [x] V0.5 Phase: CLI subcommands `automl features select` and `automl features ablation`
- [x] V0.5 Phase: Unit and integration tests in `tests/test_v05_features.py` (52/52 tests passing, coverage >= 85%)

## Blocked

*(No blocked tasks)*

## Documentation Review — 2026-09-30

- [x] Reviewed Markdown consistency, relative links, extension guidance, CLI, and current validation results: 161 tests passed, 85.14% coverage.
- [x] Follow up on review recommendations: synchronize versions and phase status, replace machine-specific links, document Workbench and native/fallback backends, consolidate roadmap ownership, and validate extension examples.
- [x] Fix and cover model plugin registration: `AutoMLWorkspace.register_plugin()` now imports `ModelSpec` from `automl.domain.models.registry`; regression tests cover discovery, task compatibility and real training.

- [x] Final validation (2026-09-30): 164 tests passed, 85.23% coverage; editable package metadata and CLI report 0.7.0; custom plugin example trained successfully; Markdown links and heading anchors checked.

- [x] Organized review fixes on `fix/docs-and-plugin-registration` from updated `origin/main`, with separate commits for plugins, version and documentation; prior frontend edits excluded.

## OOF Prediction Blending — 2026-09-30

- [x] Initial binary ROC-AUC implementation with fold-local preprocessing, original label mapping, template alignment, persisted predictions and backend/configuration manifests.
- [x] Run/experiment ownership, probability validity, time budgets, pause/cancel and changed-input/artifact checks.
- [x] Final validation: 196 tests passed, 86.01% coverage; CLI/task catalog, JavaScript syntax, Markdown links and diff checks passed. Changes prepared for a separate draft PR based on the unmerged documentation PR #13.



## Workbench persisted trials — 2026-09-30

- [x] Fix experiments HTTP listing for saved TrialResults: query parameters from their Trial through the application DTO instead of accessing a nonexistent result field ([issue #18](https://github.com/Jfenic/catml/issues/18)).
- [x] Add HTTP/query regressions for nonempty parameters, missing legacy Trial records and detached DTO mutation; 198 tests passed, coverage 86.22%.
- [x] Launch S6E9 visual preview at port 8080 with background jobs from PR #17 and this isolated fix; existing data and unrelated local edits preserved.

## Persistent Jobs — 2026-09-30

- [x] Continue on isolated `feat/persistent-job-queue` from `main` after merged PR #16; preserve unrelated working-tree edits.
- [x] Add domain job states, repository port, ID-returning commands, snapshot queries, transactional SQLite queue and exclusive local worker.
- [x] Integrate real model/fold progress, cooperative controls, durable errors/retries, background Workbench activity and CLI/HTTP parity.
- [x] Document operation scope, Linux/macOS lease, checkpoint recovery and partial OOF restart in spec and ADR 003.
- [x] Final validation: 229 Python tests passed, 86.31% coverage; 4 JavaScript adapter tests, CLI help/task catalog, 21 Markdown file-link checks and diff checks passed. Prepared for an independent draft PR against main.
