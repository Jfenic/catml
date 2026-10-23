# CATML

> **AutoML you can trust an AI agent to operate.**  
> Local-first AutoML with automatic leakage protection, reproducible experiments, and controlled AI agent access.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-0.9.0--rc1-blue.svg)](src/automl/__init__.py)
[![CI](https://github.com/Jfenic/catml/actions/workflows/ci.yml/badge.svg)](https://github.com/Jfenic/catml/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-788%20Collected-brightgreen.svg)](https://github.com/Jfenic/catml/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/Coverage-%E2%89%A585%25%20(CI%20gate)-brightgreen.svg)](https://github.com/Jfenic/catml/actions/workflows/ci.yml)
[![Architecture](https://img.shields.io/badge/Architecture-Hexagonal%20%2B%20CQRS-orange.svg)](ARCHITECTURE.md)
[![MCP](https://img.shields.io/badge/MCP-Ready-purple.svg)](src/automl/interfaces/mcp/)

<p align="center">
  <img src="docs/assets/catml-demo.gif" alt="CATML Terminal CLI in Action" width="100%" />
</p>

---

## Why CATML?

Most AutoML platforms (AutoGluon, H2O, FLAML, PyCaret) are monolithic libraries or proprietary cloud silos designed before the rise of autonomous AI coding agents. 

CATML is engineered from the ground up with **five distinct advantages**:

1. **Fail-Safe Anti-Leakage Guardian:** Automatically detects target leakage, group leakage, and pseudo-identifiers before training. Contaminated columns are automatically excluded by default.
2. **Agent-Native by Design (MCP Server):** Native Model Context Protocol (MCP) dual transport (stdio/HTTP) enables external AI agents (Claude, Cursor, OpenAI) to discover features, run trials, and explore hypotheses within budget-aware leases and cooperative cancellation controls.
3. **3-Line Ergonomics & Reproducibility:** Simple, scikit-learn compatible facade (`AutoML(random_state=42).fit(df, target="churn")`) over the same CQRS/hexagonal stages used by CLI and agents—not a separate monolithic engine.
4. **Portable Artifacts (Database-Free Inference Runtime):** Winning models serialize into `model.pkl` + `.sha256` sidecars for workspace-independent inference. Runtime still needs compatible scientific Python stacks (see security note below)—not a zero-dependency binary.
5. **Local-First by Default:** Workbench and MCP bind to `127.0.0.1` by default, confine filesystem access to the workspace, and keep an SQLite ledger locally. CATML does not phone home or upload dataset rows. Optional extras (vision weight downloads, remote LLM endpoints) may use the network when you enable them. Workbench UI chrome (Tailwind, Chart.js, Lucide, fonts) is vendored offline in the wheel.

### Feature Comparison

| Capability | Traditional AutoML (AutoGluon, FLAML, PyCaret) | Ad-hoc ML Scripts | CATML |
|---|---|---|---|
| **Data Leakage Guardian** | Manual / Absent (unnoticed test contamination) | High risk of subtle target/group leakage | **Automatic Anti-Leakage Guardian** halts on target/group leakage & pseudo-IDs |
| **AI Agent Governance** | None / Monolithic black boxes | Fragile prompts without cost controls | **Native MCP dual transport** (stdio/HTTP) with token audits & budget leases |
| **Clean Architecture** | Monolithic framework-bound pipelines | Hard-to-maintain spaghetti scripts | **Hexagonal + CQRS Bus** with pure Python domain core |
| **Inference Artifacts** | Heavy pickled environments tied to DB/cloud | Ad-hoc `.pkl` without integrity checksums | **Portable artifacts** (`model.pkl` + `.sha256`) independent of training workspace |
| **Privacy & Telemetry** | Cloud lock-in / Telemetry phone-home | Variable | **100% Local-First** (`127.0.0.1`, local SQLite, 0 dataset egress) |

---

## Quickstart (Python API)

Works after `pip install catml` with no repository checkout (uses a built-in scikit-learn dataset):

```python
from sklearn.datasets import load_breast_cancer
from catml import AutoML, ModelArtifact

# 1. Any tabular DataFrame with a target column
df = load_breast_cancer(as_frame=True).frame

# 2. Fit candidate models with automatic leakage protection & reproducible seed
automl = AutoML(metric="roc_auc", random_state=42)
result = automl.fit(df, target="target")

# 3. View the experiment leaderboard
print(result.leaderboard())

# 4. Save portable winning model for production inference
result.save_model("model.pkl")

# 5. Load and predict (workspace-independent; requires matching scientific deps)
model = ModelArtifact.load("model.pkl")
predictions = model.predict(df.head(5))
probabilities = model.predict_proba(df.head(5))
```

CSV examples under `examples/data/` ship with the **Git repository**, not the PyPI wheel.

> **Public API note:** Install/import the distribution as `catml`. The implementation package remains `automl` for CQRS modules (`from automl.application...`). See [`ARCHITECTURE.md`](ARCHITECTURE.md).

> [!WARNING]
> **Model Artifact Deserialization Security:** Never load `.pkl` / `joblib` artifacts received from untrusted third parties. Deserialization can execute arbitrary code. Companion `.sha256` checksums detect accidental corruption only—not authenticity. Artifacts embed dependency version manifests; keep scikit-learn/numpy/pandas aligned with training. HMAC/PKI signatures and ONNX export are on the hardening roadmap ([`SECURITY.md`](SECURITY.md), [`docs/trusted-deployment.md`](docs/trusted-deployment.md)).

---

## Quickstart (Command Line)

```bash
# 1. Verify your environment, hardware acceleration, and optional backends
catml doctor

# 2. Fit candidate AutoML models on a dataset (examples/ is in the Git repo)
catml fit examples/data/customers_churn.csv --target churn

# 3. Or point at any local CSV with JSON-formatted machine output
catml fit data.csv --target churn --models logistic_regression,random_forest --json
```

Example output (illustrative scores; `lightgbm` requires the optional `catml[models]` extra):
```text
=================================================================
  CATML Automated Machine Learning (Platform 0.9.0-rc1)
=================================================================
  Dataset:     examples/data/customers_churn.csv
  Target:      churn
  Candidates:  logistic_regression, random_forest, lightgbm

  Evaluating candidate models and tuning...

  Leaderboard:
  ------------------------------------------------------------
  Rank  Model                   Metric      Score     Time (s)  
  ------------------------------------------------------------
  1     lightgbm                roc_auc     0.8924    0.28      
  2     random_forest           roc_auc     0.8750    0.16      
  3     logistic_regression     roc_auc     0.8512    0.12      
  ------------------------------------------------------------

  ✓ Best Model:   lightgbm (roc_auc: 0.8924)
  ✓ Model Saved:  /home/user/project/catml-runs/model.pkl
  ✓ Run in Prod:  ModelArtifact.load("catml-runs/model.pkl")
=================================================================
```

---

## Interactive Web Workbench (Visual ML Lab)

<p align="center">
  <img src="website/public/workbench.png" alt="CATML Interactive Web Workbench & Dataset Inspector" width="100%" />
</p>

CATML includes a built-in interactive Tech Minimalista Premium visual dashboard for monitoring runs, inspecting datasets, and managing experiments:

```bash
catml ui --port 8080 --workspace .automl/default
```

Defaults to **`127.0.0.1`**. Binding a non-loopback host requires an auth token (fail-closed). Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** to access:
- **Mission Control Overview:** Live experiment status and resource telemetry.
- **Dataset Inspector:** Deep statistical profiling, cardinality detection, and feature recommendations.
- **Visual Experiment Studio:** Interactive comparison diffs, HPO trial history, and model explainability.
- **Persistent Job Queue:** Resilient SQLite worker with pause, resume, and cooperative cancellation.

> **Network note:** Training and dataset rows stay local. Workbench UI chrome is served from packaged `/static/vendor/*` and `/static/css/tailwind.built.css` (no browser CDN). Rebuild vendored assets with `bash scripts/vendor_workbench_assets.sh` when bumping Chart.js/Lucide/Tailwind/fonts.

---

## Agentic Protocol: Model Context Protocol (MCP)

CATML exposes AutoML capabilities to LLMs via MCP. Install the extra first:

```bash
pip install -e ".[mcp]"   # or: pip install "catml[mcp]"
```

### Connect to Claude Desktop or Cursor

Add CATML to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "catml": {
      "command": "catml",
      "args": ["mcp", "--transport", "stdio"]
    }
  }
}
```

Or start the streamable HTTP transport (loopback by default; remote bind requires a token):

```bash
catml mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

Available tools exposed to AI agents:
- `get_dataset_profile`: Deep statistical profiling and column cardinality.
- `list_models` / `list_plugins`: Inspect available algorithms and metric plugins.
- `create_experiment`: Propose targeted hypothesis-driven experiment runs.
- `get_leaderboard`: Inspect ranked metric scores and cross-validation variance.
- `cancel_operation`: Cooperative cancellation under budget constraints.

### How AI Agents Use CATML via MCP (Real-World Flow)

When an AI agent connects via MCP, CATML enforces strict CQRS separation and human-in-the-loop safety governance:

1. **Discovery & Dataset Profiling (Zero Side-Effects):**
   The agent calls `get_dataset_profile` to inspect column types, cardinality, and missingness without mutating state:
   ```json
   // Agent calls: get_dataset_profile(dataset_id="ds_churn")
   // CATML returns:
   {
     "row_count": 10000,
     "column_count": 14,
     "target_column": "churn",
     "task_type": "binary_classification"
   }
   ```

2. **Hypothesis Proposal with Approval Gate:**
   When the agent proposes a modeling hypothesis via `create_experiment`, CATML intercepts the mutating action according to workspace policy:
   ```json
   // Agent calls: create_experiment(run_id="run_01", model_name="random_forest")
   // CATML returns governance lease:
   {
     "status": "PENDING_APPROVAL",
     "approval_id": "appr-876d21bd24740517",
     "message": "Action 'create_experiment' requires human approval. Run: automl agent approve appr-876d21bd24740517"
   }
   ```

3. **Leaderboard Inspection:**
   Once verified, the agent retrieves the cross-validated rankings via `get_leaderboard(run_id="run_01", top_k=3)` to determine next steps.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Jfenic/catml.git
cd CATML

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode (tests + parquet helpers)
pip install -e ".[dev]"

# MCP / agent extras are optional — install when you need those surfaces:
pip install -e ".[mcp,agents]"
```

### Optional Modality & Agentic Extensions

```bash
# Gradient boosting backends (LightGBM, XGBoost, CatBoost)
pip install "catml[models]"

# Deep learning vision representations (PyTorch, torchvision, timm, Pillow)
pip install "catml[vision]"

# MCP server SDK and LangGraph agent orchestration
pip install "catml[mcp]"
pip install "catml[agents]"

# Full stack with all extensions
pip install "catml[all]"
```

> [!NOTE]
> **Local-First & Offline Environments:** CATML executes completely local-first without transmitting dataset rows to external cloud services. However, when using deep neural vision backbones (`resnet18`, etc.), TorchVision downloads pretrained weights upon first execution if not already cached. For strictly offline or air-gapped environments, pre-populate the PyTorch cache directory (`~/.cache/torch/hub/checkpoints`).

---

## Advanced Architecture (CQRS Application Bus)

CATML enforces a strict Hexagonal (Ports & Adapters) architecture with CQRS separation between mutation commands and inspection queries:

```mermaid
flowchart TD
    subgraph S1["01 · Dataset & Inspection"]
        D["Tabular CSV / Parquet"]
        LG["Anti-Leakage Guardian<br/>(Target, Group & Pseudo-ID Checks)"]
        FD["Feature Discovery<br/>(Categorical, Temporal, TF-IDF Text)"]
    end

    subgraph S2["02 · CQRS Application Core"]
        BUS["CommandBus & QueryBus<br/>(Pure Hexagonal Orchestration)"]
        HPO["HPO & Model Ensembles<br/>(LightGBM, XGBoost, CatBoost, Stacking)"]
    end

    subgraph S3["03 · Control & Interfaces"]
        FAC["Python API (3 Lines)<br/>AutoML.fit()"]
        CLI["Modular CLI<br/>catml fit / data / recipe"]
        MCP["Native MCP Server<br/>(Claude Desktop, Cursor, Agents)"]
        WB["Local Web Workbench<br/>(Visual Lab at 127.0.0.1)"]
    end

    subgraph S4["04 · Verified Artifacts"]
        ART["Portable Model Artifact<br/>model.pkl + .sha256"]
        REC["Reproducible Experiment Recipe<br/>YAML Specification"]
    end

    D --> LG
    LG --> FD
    FD --> BUS
    BUS --> HPO
    FAC -.-> BUS
    CLI -.-> BUS
    MCP -.-> BUS
    WB -.-> BUS
    HPO --> ART
    HPO --> REC
```

For power users, MLOps pipelines, and custom orchestrators, CATML provides strict CQRS separation via `CommandBus` and `QueryBus`:

```python
from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    PlanExperimentsCommand,
    RunScheduledExperimentsCommand,
)
from automl.application.queries.workspace_queries import GetLeaderboardQuery

# Build application workspace
ws, cmd, qry = build_application(root_dir=".automl/demo")

# Register dataset and create run
dataset = ws.register_dataset(
    name="churn",
    path="examples/data/customers_churn.csv",
    target="churn",
)
run = ws.create_run(dataset)

# Dispatch CQRS commands
cmd.dispatch(PlanExperimentsCommand(run_id=run.id))
cmd.dispatch(RunScheduledExperimentsCommand(run_id=run.id, max_experiments=3))

# Dispatch zero-side-effect queries
leaderboard = qry.dispatch(GetLeaderboardQuery(run.id))
for entry in leaderboard:
    print(f"{entry['model_id']}: {entry['metric']} = {entry['score']:.4f}")
```

---

## How to Test

```bash
# Run all unit and integration tests
pytest

# Run tests with strict coverage validation (>= 85%)
pytest --cov=src/automl --cov-fail-under=85
```

---

## Interactive Validation Notebooks

The repository ships with focused, self-contained fire-test notebooks under [`notebooks/fire_tests/`](notebooks/fire_tests/):

| Notebook | Topic / Workflow | Description |
|---|---|---|
| [`01_titanic_like.ipynb`](notebooks/fire_tests/01_titanic_like.ipynb) | Binary Classification Baseline | End-to-end baseline modeling, feature imputation, and scoring. |
| [`03_credit_risk.ipynb`](notebooks/fire_tests/03_credit_risk.ipynb) | Anti-Leakage Guardian | Real-world simulation of target contamination and automatic feature isolation. |
| [`04_multiclass.ipynb`](notebooks/fire_tests/04_multiclass.ipynb) | Multiclass Classification | Multi-target probability calibration and log-loss optimization. |
| [`08_export_predict.ipynb`](notebooks/fire_tests/08_export_predict.ipynb) | Production Artifacts | Exporting `model.pkl` + `.sha256` sidecar and running offline inference. |
| [`10_ensemble_compare.ipynb`](notebooks/fire_tests/10_ensemble_compare.ipynb) | Stacking & Blending | Multi-model Out-of-Fold (OOF) blending vs level-2 super learner stacking. |

---

## Capabilities & Modality Status

| Modality / Workflow | Status | Details |
|---|---|---|
| **Tabular Classification & Regression** | **Stable** | LightGBM, XGBoost, CatBoost, Scikit-learn, Voting Ensembles, N-Model OOF Blending, Level-2 Stacking, Temporal Dynamics, and Group Anti-Leakage. |
| **Text Representations (NLP)** | **Available** | Sublinear TF-IDF n-gram extractor with automated `is_text_column` discovery and multimodal column fusion. |
| **Image Representations (Vision)** | **Available** | Representation-level vision with PyTorch/torchvision backbone (`catml[vision]`, feature extractor + downstream estimators), deterministic hash encoder (testing/benchmarks fallback), image path discovery, and gallery preview in Workbench. |
| **Agent Governance (MCP & LangGraph)** | **Available** | Native Model Context Protocol (stdio/HTTP), LangGraph StateGraph orchestrator, persistent SQLite ledger, human approval flow, and token auditing. |
| **Native Vision (Detection / Segmentation)** | *Roadmap* | Planned for future v1.x with specialized deep learning backends. |
| **Biomedical Extensions (`catml[medical]`)** | *Roadmap* | Planned extension for DICOM/NIfTI with strict zero data egress. |

---

## Documentation Navigation

- **System Architecture & Hexagonal Rules:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Technical Specification & Roadmap (historical):** [`docs/AutoML_Arquitectura_Tecnica.md`](docs/AutoML_Arquitectura_Tecnica.md)
- **Active Tasks & Operational Backlog:** [`TASKS.md`](TASKS.md)
- **Agent Rules & Development Guidelines:** [`AGENTS.md`](AGENTS.md)
- **Developer Extension Guide:** [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)
- **Feature Specifications & Plans:** [`docs/features/`](docs/features/)
- **Architectural Decision Records (ADRs):** [`docs/decisions/`](docs/decisions/)
- **Trusted Deployment & Threat Model:** [`docs/trusted-deployment.md`](docs/trusted-deployment.md)
- **Security Policy & Reporting:** [`SECURITY.md`](SECURITY.md)
- **Code of Conduct:** [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md)

---

## License
 
Starting with version 0.8.0, CATML is licensed under the Apache License, Version 2.0 - see the [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) files for details. Earlier releases (<= 0.7.0) remain available under the MIT License.

Third-party Workbench libraries and fonts (Chart.js, Lucide, Tailwind CSS output, Geist, IBM Plex Mono) and Python dependencies are documented in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). Full texts for vendored UI assets ship in the package under `automl/interfaces/web/static/vendor/licenses/`.
