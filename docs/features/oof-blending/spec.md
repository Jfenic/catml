# Blending de predicciones out-of-fold — primera versión

Estado: Implementado y completado para clasificación binaria con ROC-AUC, evaluación en holdout independiente, validación honesta de promoción de candidatos (`promoted`) y diagnóstico de residuos de stacking. Consultar [TASKS.md](../../../TASKS.md).

## Objetivo

Comparar blending de modelos entrenados con cinco folds estratificados frente a un baseline tabular, manteniendo separación entre evaluación y ajuste de pesos. Una puntuación Kaggle superior a 0.945 es una aspiración, no un criterio garantizado ni una métrica ya medida.

## Protocolo de evaluación

1. Registrar hash y esquema del dataset, objetivo, backend concreto y versiones. Para clasificación binaria, usar ROC-AUC.
2. Fijar seed 42 y cinco folds estratificados idénticos para los modelos. Ajustar imputación, encoding y selección de features exclusivamente con las filas de entrenamiento de cada fold.
3. Definir antes de ejecutar el baseline: mejor modelo individual bajo los mismos splits y presupuesto, guardando su configuración y predicciones OOF. El baseline no puede usar métricas ilustrativas de la UI.
4. Generar una predicción OOF por fila con un modelo que no la haya visto. Promediar las predicciones del test entre los cinco modelos de folds y alinear con la plantilla de submission.
5. Evaluar inicialmente pesos iguales. Si se ajustan pesos usando OOF, medir el resultado final en un holdout independiente o validación anidada; no usar el mismo OOF para ajustar y declarar mejora imparcial.
6. Registrar ROC-AUC, dispersión entre folds, delta contra baseline, tiempo y memoria. Una puntuación pública de Kaggle debe quedar identificada por separado.

## Presupuesto y aceptación

- Primera comparación: dos modelos individuales y su combinación, cinco folds por modelo, un trial por modelo/fold; sin HPO adicional. Registrar tiempo real y detenerse con el límite de recursos configurado antes de comenzar.
- Pruebas de integridad: cobertura OOF exactamente una vez por fila, ausencia de fuga entre folds, orden de IDs, probabilidades válidas, reproducibilidad y persistencia de resultados.
- Promoción: evidencia de mejora en evaluación independiente bajo el mismo presupuesto. Si no mejora, conservar el candidato y registrar el resultado sin promoverlo.
- Integración: `GenerateOOFSubmissionCommand` retorna el ID del experimento; `GetOOFResultQuery` lee su informe sin entrenar ni escribir. CLI y Workbench despachan estos mismos buses. La predicción anterior sigue disponible sin `--folds`.

## Uso y alcance actual

```bash
automl predict --workspace .automl/demo --run-id RUN_ID \
  --experiment-id SOURCE_EXPERIMENT_ID --test-dataset data/test.csv \
  --template data/sample_submission.csv --output submission_oof.csv \
  --folds 5 --models logistic_regression,random_forest --proba --json \
  --holdout-dataset data/holdout.csv --min-improvement 0.001
```

`--experiment-id` identifica el experimento fuente y su lista fija de columnas, no el nuevo experimento OOF. Sin esa opción se usa el experimento no OOF de un resultado exitoso del leaderboard. Sin `--models`, se toman hasta dos modelos individuales de la fuente; una fuente compuesta usa logistic regression y random forest. El primer modelo seleccionado es el baseline declarado antes de evaluar; se reutilizan parámetros de trials exitosos de la fuente cuando existen.

El flujo soporta métodos de ensembling `average`, `rank`, `simplex` y `stacked` (con meta-estimadores Ridge, LogisticRegression y Lasso). La validación contra un holdout independiente (`--holdout-dataset` o `--holdout-ratio`) garantiza que el candidato solo se promueve (`promoted=True`) si `holdout_delta > min_improvement`, reportando diagnósticos de residuos de stacking (media, std, error absoluto máximo y Brier score).

El preprocesado de cada fold ajusta imputación, escalado y one-hot encoding exclusivamente con sus filas de entrenamiento. Las columnas son una hipótesis fija heredada: este flujo no recalcula selección supervisada ni target encoding. Deben proporcionarse columnas sin transformaciones previas que hayan usado el objetivo completo.

El presupuesto predeterminado es 300 segundos de evaluación, configurable con `--oof-timeout`. Se comprueba en los límites de cada fit; no interrumpe un estimador en mitad de su entrenamiento. Pause/cancel abortan en esos límites, conservan el estado del run y registran el fallo; tras resume se inicia un nuevo experimento, sin reusar fits parciales.

En el Workbench, marcar **Promediar 5 folds (OOF)** al generar la submission. El endpoint `POST /api/predict` acepta `folds`, `model_ids`, `experiment_id` y `max_seconds`; retorna el informe dentro de `oof`.

## Artefactos y evidencia

Cada experimento guarda `oof/<experiment_id>/oof.csv`, `test_predictions.csv` y `report.json`. El informe registra ROC-AUC individual y del blend, scores por fold, dispersión, delta frente al baseline, scores en holdout independiente, bandera de promoción honesta, residuos de stacking, clases, parámetros, seeds, backend y versiones, hashes de datos/configuración/artefactos, tiempo y máximo de memoria del proceso cuando el sistema lo permite. Esta memoria es el máximo acumulado del proceso, no una medida aislada de cada fit.

El resultado se referencia desde un `TrialResult` de `oof_blend` en SQLite. Las consultas reutilizan las probabilidades guardadas para el mismo test y rechazan cambios en los datos o artefactos. Para un test diferente hay que volver a ejecutar `--folds`; los modelos de cada fold no se serializan en esta versión.

| Capacidad | Implementación / pruebas |
| --- | --- |
| Folds, probabilidades y blending | [motor OOF](../../../src/automl/engine/ensemble/oof.py), [tests](../../../tests/test_oof_blending.py) |
| Validación holdout y residuos de stacking | [motor OOF](../../../src/automl/engine/ensemble/oof.py), [tests](../../../tests/test_oof_holdout_and_residuals.py) |
| Persistencia, controles y consultas | [servicio de aplicación](../../../src/automl/application/services/oof_submission.py), [tests](../../../tests/test_oof_blending.py) |
| Paridad CLI / HTTP | [CLI](../../../src/automl/interfaces/cli/main.py), [servidor](../../../src/automl/interfaces/web/server.py), [tests](../../../tests/test_oof_blending.py) |
