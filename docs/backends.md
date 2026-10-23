# Backends opcionales y reproducibilidad

La instalación base utiliza scikit-learn. LightGBM y XGBoost son dependencias opcionales:

```bash
.venv/bin/python -m pip install lightgbm xgboost
```

| Plugin | Backend si está instalado | Alternativa si falta |
| --- | --- | --- |
| `lightgbm` | `LGBMClassifier` / `LGBMRegressor` | `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor` |
| `xgboost` | `XGBClassifier` / `XGBRegressor` | `GradientBoostingClassifier` / `GradientBoostingRegressor` |

La alternativa conserva el flujo de experimentación, pero cambia el algoritmo y solo traduce parte de los hiperparámetros. Comparar resultados requiere identificar el backend concreto.

## Comprobar el estimador

Desde la raíz del repositorio:

```bash
.venv/bin/python - <<'PY'
from automl.plugins.models.gradient_boosting import LightGBMPlugin, XGBoostPlugin

for plugin in (LightGBMPlugin(), XGBoostPlugin()):
    estimator = plugin.build_estimator(task_type="binary_classification")
    print(plugin.plugin_id, "native_available=", plugin.is_available,
          type(estimator).__module__, type(estimator).__name__)
PY
```

Para exigir el backend nativo, registrar `LightGBMPlugin(use_fallback_if_missing=False)` o `XGBoostPlugin(use_fallback_if_missing=False)` durante la composición del workspace. Sin el paquete correspondiente, construir el estimador lanza `ImportError`; el trainer registra el fallo del trial. Esta opción se configura en Python, no mediante un flag de CLI.

## Alcance de reproducibilidad

Los trials registran parámetros, semilla aleatoria (`seed`), estrategia de validación, partición de cross-validation (número de folds, identificadores de fold `cv_fold_ids`, y hash SHA-256 de asignación de folds `cv_fold_assignment_hash`), hash SHA-256 del dataset (`dataset_content_hash`), hash SHA-256 de configuración (`config_hash`), clase y módulo concreto del estimador, backend (`native` o `fallback`), y versiones de librerías en SQLite. Toda esta información está disponible directamente en el leaderboard de `AutoMLWorkspace`, en el DataFrame `AutoMLResult.leaderboard()`, en las respuestas JSON de CLI y MCP, y en el bloque `provenance` de cada `ModelArtifact` exportado.

Asimismo, la exportación de modelos mediante `ModelArtifact` (`workspace.export_model_artifact()`) persiste de forma automática un bloque `provenance` completo que documenta:
- Versión de CATML y timestamp de creación UTC.
- Versión de Python y versiones de dependencias del entorno (`numpy`, `pandas`, `scikit-learn`, `joblib`, `lightgbm`, `xgboost`, `catboost`, `optuna`).
- Clase y módulo exacto del estimador final (`estimator_class`, `estimator_module`).
- Hash SHA-256 del dataset de entrenamiento (`dataset_hash`, `content_hash`) y hash de configuración (`config_hash`).
- Semilla aleatoria (`seed`, `trial_seed`, `run_seed`).
- Esquema de validación (`validation_strategy`, `cv_folds`, `cv_fold_ids`, `cv_fold_assignment_hash`).
- Esquema de características y configuración de hiperparámetros consultables con `artifact.describe()`.

Para una comparación reproducible, conservar el workspace, una copia o hash del dataset, configuración de validación, clase concreta del estimador y versiones del entorno. Los encoders de imagen también tienen modos distintos: el modo predeterminado es determinista y no acredita calidad equivalente a un encoder preentrenado.
