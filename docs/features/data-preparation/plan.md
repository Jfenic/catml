# Plan: Data Preparation Subsystem

> **Spec:** [spec.md](spec.md) · **ADR:** [012](../../decisions/012-data-preparation-subsystem.md)  
> **Status:** Documented direction · **Implement after** tabular `0.9.0rc1` stabilization  
> **Do not** block load→train on a mandatory wizard.

## Phases

### P0 — Documentation & honesty (this PR / docs cut)

- [x] Spec + plan + ADR-012  
- [x] Soften Workbench quality badge (“Sin fuga detectada” + disclaimer)  
- [x] Index in `docs/README.md` / `TASKS.md` / `AGENTS.md`

### P1 — Domain + application core

- [x] `PreparationPlan` / `PreparationStep` / scopes in `domain/` (pure)  
- [x] `PreparationService`: analyze, suggest, preview, validate, apply  
- [x] Dataset version entity + storage under workspace (hash + lineage)  
- [x] Register commands/queries in `bootstrap.py`  
- [x] Unit tests: scope guard, preview purity, apply immutability (`tests/test_data_preparation_p1.py`)

### P2 — CLI + Python facade

- [x] `catml data inspect|prepare …`  
- [x] Public `catml` / `AutoML` prepare surface (thin)  
- [x] Wire pipeline-scoped steps into experiment/recipe config (no full-frame fit)

### P3 — MCP + Workbench adapters

- [ ] MCP tools (analyze/suggest/preview; apply gated)  
- [ ] Workbench “Preparación” tab (controlled actions + preview; brand system)  
- [ ] Link Explore findings → suggested steps (`EvidenceLink` / finding ids)

### P4 — Hardening

- [ ] Recipe replay of preparation lineage on export/predict  
- [ ] Retention policy for old versions  
- [ ] Optional ydata-profiling / Great Expectations adapters (extras, not required)

## Non-goals (explicit)

- Competing with OpenRefine/Alteryx as general ETL  
- Implementing Workbench Labs ADR-011 as a prerequisite  
- Auto outlier drop, target-based relabel, SMOTE outside folds in v0

## Validation

```bash
.venv/bin/pytest tests/test_data_preparation*.py -q   # when added
.venv/bin/pytest --cov=src/automl --cov-fail-under=85
```
