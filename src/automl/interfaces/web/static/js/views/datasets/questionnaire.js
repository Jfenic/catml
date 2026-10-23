/** Extracted from DatasetsView — questionnaire.js (Clean Architecture PR3). */
import { store } from "../../store.js";
import { bus } from "../../bus.js";
import { api } from "../../api.js";
import { icon } from "../../icons.js";
import { escapeHtml } from "../../utils.js";

export function _renderOptionCard(view, field, val, currentVal, title, desc, tag = "") {
  const isSelected = currentVal === val;
  return `
    <div class="q-option-card p-3 rounded-xl border cursor-pointer transition-all ${
      isSelected
        ? 'border-[#4F67FF] bg-[#161B26] ring-1 ring-[#4F67FF]/30'
        : 'border-[#242A36] bg-[#0F131C] hover:border-[#384357] hover:bg-[#151B26]'
    }" data-q-field="${field}" data-q-value="${val}">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-2">
          <span class="w-3.5 h-3.5 rounded-full border flex items-center justify-center ${
            isSelected ? 'border-[#4F67FF] bg-[#4F67FF]' : 'border-[#4B5563]'
          }">
            ${isSelected ? '<span class="w-1.5 h-1.5 rounded-full bg-white"></span>' : ''}
          </span>
          <span class="font-medium text-xs ${isSelected ? 'text-[#F7F8FA]' : 'text-[#8B95A7]'} font-sans">${title}</span>
        </div>
        ${tag ? `<span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#1C2230] text-[#8B95A7] border border-[#242A36]">${tag}</span>` : ''}
      </div>
      <p class="text-[11px] text-[#8B95A7] mt-1.5 pl-5 font-sans leading-relaxed">${desc}</p>
    </div>
  `;
}

export function _getCompatibleToolsSummary(view, q, p) {
  if (!q) return [];
  const hints = [];
  const taskType = p?.task_type || "classification";
  const isRegression = taskType.includes("regression");

  if (q.latency_constraint === "ultra_low_realtime") {
    hints.push({
      type: "recommended",
      badge: "Recomendado",
      badgeClass: "badge-gain",
      title: "Modelos Rápidos (< 10 ms)",
      desc: isRegression ? "Ridge Regression, Lasso, DecisionTreeRegressor." : "Logistic Regression, Shallow Decision Trees.",
    });
    hints.push({
      type: "warning",
      badge: "Desaconsejado",
      badgeClass: "badge-err",
      title: "Modelos Pesados en Inferencia",
      desc: "Ensembles apilados (Voting/Blender) y Redes Neuronales exceden el SLA < 10ms.",
    });
  } else if (q.latency_constraint === "standard_interactive") {
    hints.push({
      type: "recommended",
      badge: "Recomendado",
      badgeClass: "badge-gain",
      title: "GBDTs (< 200 ms)",
      desc: "LightGBM, XGBoost, CatBoost ofrecen el balance óptimo de precisión y latencia interactiva.",
    });
  } else if (q.latency_constraint === "batch_offline") {
    hints.push({
      type: "recommended",
      badge: "Recomendado",
      badgeClass: "badge-gain",
      title: "Ensembles y Modelos Complejos",
      desc: "Voting Blenders, Stacking y MLPs habilitados para maximizar métrica sin límite de tiempo.",
    });
  }

  if (q.explainability === "highly_regulated") {
    hints.push({
      type: "recommended",
      badge: "Auditoría Estricta",
      badgeClass: "badge-intel",
      title: "Modelos Transparentes e Interpretabilidad",
      desc: "Modelos lineales y árboles simples con TreeSHAP determinista para cumplimiento regulatorio.",
    });
    hints.push({
      type: "warning",
      badge: "Restringido",
      badgeClass: "badge-warn",
      title: "Modelos de Caja Negra",
      desc: "Redes Neuronales profundas y Ensembles complejos no recomendados en entorno auditado.",
    });
  }

  if (q.error_cost === "avoid_false_negatives") {
    hints.push({
      type: "info",
      badge: "Prioridad Recall",
      badgeClass: "badge-sys",
      title: "Optimización de Umbral y PR-AUC",
      desc: "Métrica primaria PR-AUC / Recall; optimizador de corte de decisión bayesiano activado.",
    });
  } else if (q.error_cost === "avoid_false_positives") {
    hints.push({
      type: "info",
      badge: "Prioridad Precision",
      badgeClass: "badge-sys",
      title: "Filtrado Estricto de Falsas Alarmas",
      desc: "Métrica primaria Precision; umbral conservador para evitar falsos positivos.",
    });
  }

  if (q.temporal_structure === "sequential_time_series") {
    hints.push({
      type: "warning",
      badge: "Validación Obligatoria",
      badgeClass: "badge-warn",
      title: "TimeSeriesSplit / Purged CV",
      desc: "Partición temporal hacia el futuro requerida. K-Fold aleatorio descartado para prevenir fuga temporal.",
    });
  } else if (q.temporal_structure === "grouped_cohorts") {
    hints.push({
      type: "info",
      badge: "Validación por Grupos",
      badgeClass: "badge-sys",
      title: "GroupKFold Requerido",
      desc: "Partición asegurando que las observaciones de una misma entidad no se repartan entre train y test.",
    });
  }

  return hints;
}

export function _renderQuestionnaireTab(view, p) {
  const q = view.questionnaire || {
    dataset_id: p.dataset_id,
    error_cost: "balanced",
    temporal_structure: "cross_sectional",
    latency_constraint: "standard_interactive",
    explainability: "moderate",
    imbalance_strategy: "auto",
    primary_metric_override: null,
    split_strategy_recommendation: (p.task_type || "").includes("regression") ? "kfold" : "stratified_kfold",
    user_confirmed: false,
    engine_source: "heuristic",
    confidence: 0.9,
    reasoning: "Sin encuadre guardado todavía. Puedes ajustar las opciones interactivas o pulsar 'Generar sugerencias' para inferir el encuadre óptimo.",
  };

  const isConfirmed = Boolean(q.user_confirmed);
  const engineSource = q.engine_source || "heuristic";
  const engineLabel = engineSource === "jev"
    ? "Generado mediante JEV"
    : "Generado mediante reglas heurísticas";
  const engineBadgeClass = engineSource === "jev"
    ? "bg-[#53C8FF]/10 text-[#53C8FF] border-[#53C8FF]/25"
    : "bg-[#151B26] text-[#8B95A7] border-[#242A36]";

  const compatibleTools = view._getCompatibleToolsSummary(q, p);
  const isRegression = (p.task_type || "").includes("regression");

  return `
    <div class="space-y-6">
      <!-- Top Toolbar & Status Header -->
      <div class="workbench-card p-5 space-y-4">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-[#242A36]">
          <div>
            <div class="flex items-center space-x-2.5">
              <h3 class="text-base font-bold text-[#F7F8FA] font-sans">Encuadre del Problema & Cuestionario Operativo</h3>
              <span class="text-xs px-2.5 py-0.5 rounded font-mono border ${engineBadgeClass}">
                ${engineLabel}
              </span>
              ${isConfirmed ? `
                <span class="badge-gain text-xs px-2.5 py-0.5 rounded font-mono font-medium flex items-center space-x-1.5">
                  <span class="w-1.5 h-1.5 rounded-full bg-[#22C55E]"></span>
                  <span>Confirmado por el usuario</span>
                </span>
              ` : `
                <span class="text-xs px-2.5 py-0.5 rounded font-mono font-medium bg-[#F59E0B]/10 border border-[#F59E0B]/25 text-[#F59E0B] flex items-center space-x-1.5" title="Sugerencias generadas pendientes de validación por el operador humano">
                  <span class="w-1.5 h-1.5 rounded-full bg-[#F59E0B]"></span>
                  <span>Sugerencias pendientes de confirmación</span>
                </span>
              `}
            </div>
            <p class="text-xs text-[#8B95A7] mt-1 font-sans">
              Define el contexto de negocio, los costes asimétricos de error y las restricciones de inferencia. La IA propone; el operador humano valida y decide.
            </p>
          </div>

          <div class="flex items-center space-x-3">
            <button id="btnGenerateQuestionnaire" ${view.questionnaireGenerating ? "disabled" : ""} class="btn-technical text-xs flex items-center space-x-1.5 ${view.questionnaireGenerating ? 'opacity-60 cursor-not-allowed' : ''}" title="Inferir heurísticamente el contexto óptimo del dataset">
              ${view.questionnaireGenerating
                ? `<svg class="animate-spin h-3.5 w-3.5 text-[#4F67FF]" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> <span>Analizando metadatos...</span>`
                : `${icon("sparkles", "icon-sm")} <span>Generar sugerencias</span>`
              }
            </button>

            <button id="btnSaveQuestionnaire" ${view.questionnaireSaving ? "disabled" : ""} class="btn-signal text-xs flex items-center space-x-1.5 shadow-md shadow-[#4F67FF]/20 ${view.questionnaireSaving ? 'opacity-60 cursor-not-allowed' : ''}">
              ${view.questionnaireSaving
                ? `<span>Guardando...</span>`
                : `${icon("check", "icon-sm")} <span>Guardar y Confirmar Encuadre</span>`
              }
            </button>
          </div>
        </div>

        <!-- Section 1: Error Cost Priority -->
        <div class="space-y-2 pt-1">
          <div class="flex items-center justify-between">
            <label class="text-xs font-semibold text-[#F7F8FA] uppercase tracking-wider font-sans flex items-center space-x-1.5">
              <span class="text-[#4F67FF]">${icon("shield-alert", "icon-sm")}</span>
              <span>1. Prioridad de Costo de Error de Negocio</span>
            </label>
            <span class="text-[11px] text-[#8B95A7] font-mono">Función de Coste & Métrica Primaria</span>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
            ${view._renderOptionCard("error_cost", "balanced", q.error_cost, "Equilibrado (Simétrico)", "Pérdida simétrica entre falsos positivos y negativos. Métrica estándar (F1 / Accuracy / RMSE).", "F1 / RMSE")}
            ${view._renderOptionCard("error_cost", "avoid_false_negatives", q.error_cost, "Minimizar Falsos Negativos", "Crítico en Fraude, Churn o Diagnóstico Médico. El coste de omitir un positivo es crítico. Optimiza Recall y umbral.", "Recall / PR-AUC")}
            ${view._renderOptionCard("error_cost", "avoid_false_positives", q.error_cost, "Minimizar Falsos Positivos", "Crítico en Spam o Bloqueos automáticos de usuarios. Evita falsas alarmas y costes reputacionales.", "Precision")}
            ${view._renderOptionCard("error_cost", "custom_cost_matrix", q.error_cost, "Matriz de Costo Personalizada", "Ponderación asimétrica específica configurada en la función objetivo del negocio.", "Cost-Weighted")}
          </div>
        </div>

        <!-- Section 2: Data regime — drives which CV options appear in the experiment launcher -->
        <div class="space-y-2 pt-2 border-t border-[#242A36]">
          <div class="flex flex-col gap-1">
            <label class="text-xs font-semibold text-[#F7F8FA] uppercase tracking-wider font-sans flex items-center space-x-1.5">
              <span class="text-[#6956E8]">${icon("clock", "icon-sm")}</span>
              <span>2. Tipo de datos (elige uno)</span>
            </label>
            <p class="text-[11px] text-[#8B95A7] font-sans">
              Decide la validación del experimento: independiente → K-Fold; serie temporal → TimeSeriesSplit; grupos → GroupKFold.
              En el lanzador, Time-Series solo aparece si eliges serie temporal aquí.
            </p>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
            ${view._renderOptionCard("temporal_structure", "cross_sectional", q.temporal_structure, "Independiente (corte transversal)", "Cada fila es una observación i.i.d. Validación K-Fold / Stratified K-Fold. Caso tabular más habitual.", "K-Fold")}
            ${view._renderOptionCard("temporal_structure", "sequential_time_series", q.temporal_structure, "Serie temporal", "El orden en el tiempo importa. No mezclar pasado/futuro: TimeSeriesSplit. Requiere columna temporal en el run.", "TimeSeriesSplit")}
            ${view._renderOptionCard("temporal_structure", "grouped_cohorts", q.temporal_structure, "Grupos / cohortes", "Varias filas por entidad (cliente, máquina…). GroupKFold evita fugas entre train y test.", "GroupKFold")}
          </div>
        </div>

        <!-- Section 3: Latency Constraint -->
        <div class="space-y-2 pt-2 border-t border-[#242A36]">
          <div class="flex items-center justify-between">
            <label class="text-xs font-semibold text-[#F7F8FA] uppercase tracking-wider font-sans flex items-center space-x-1.5">
              <span class="text-[#53C8FF]">${icon("zap", "icon-sm")}</span>
              <span>3. Restricción de Latencia Operativa en Inferencia</span>
            </label>
            <span class="text-[11px] text-[#8B95A7] font-mono">Filtro de Catálogo Algorítmico</span>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
            ${view._renderOptionCard("latency_constraint", "ultra_low_realtime", q.latency_constraint, "Ultra Baja (< 10 ms)", "Inferencia en streaming o SLA de alta concurrencia. Prioriza Modelos Lineales o Árboles poco profundos.", "< 10ms")}
            ${view._renderOptionCard("latency_constraint", "standard_interactive", q.latency_constraint, "Interactiva Estándar (< 200 ms)", "Microservicios web y dashboards analíticos. Óptimo para GBDTs (LightGBM, XGBoost, CatBoost).", "< 200ms")}
            ${view._renderOptionCard("latency_constraint", "batch_offline", q.latency_constraint, "Batch / Offline (Sin límite estricto)", "Scoring por lotes o pipelines nocturnos. Permite Ensembles, Stacking y redes profundas.", "Batch")}
          </div>
        </div>

        <!-- Section 4: Explainability Level -->
        <div class="space-y-2 pt-2 border-t border-[#242A36]">
          <div class="flex items-center justify-between">
            <label class="text-xs font-semibold text-[#F7F8FA] uppercase tracking-wider font-sans flex items-center space-x-1.5">
              <span class="text-[#22C55E]">${icon("file-text", "icon-sm")}</span>
              <span>4. Nivel de Explicabilidad & Regulación</span>
            </label>
            <span class="text-[11px] text-[#8B95A7] font-mono">Auditoría & Interpretabilidad</span>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
            ${view._renderOptionCard("explainability", "highly_regulated", q.explainability, "Altamente Regulado (Auditoría Estricta)", "Banca, crédito o salud. Requiere modelos transparentes (Lineales, Árboles) o SHAP exacto.", "Interpretable")}
            ${view._renderOptionCard("explainability", "moderate", q.explainability, "Moderado", "Auditoría estándar de explicabilidad mediante TreeSHAP e importancia de características.", "TreeSHAP")}
            ${view._renderOptionCard("explainability", "performance_first", q.explainability, "Máximo Rendimiento (Caja Negra)", "Competiciones Kaggle y entornos donde prima la máxima capacidad predictiva.", "Black-box OK")}
          </div>
        </div>

        <!-- Section 5: Imbalance Strategy -->
        <div class="space-y-2 pt-2 border-t border-[#242A36]">
          <div class="flex items-center justify-between">
            <label class="text-xs font-semibold text-[#F7F8FA] uppercase tracking-wider font-sans flex items-center space-x-1.5">
              <span class="text-[#F59E0B]">${icon("pie-chart", "icon-sm")}</span>
              <span>5. Estrategia para Desbalanceo de Clases</span>
            </label>
            <span class="text-[11px] text-[#8B95A7] font-mono">Tratamiento de Clases Minoritarias</span>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
            ${view._renderOptionCard("imbalance_strategy", "auto", q.imbalance_strategy, "Automático", "Determinado automáticamente según el ratio de clases del target.", "Auto")}
            ${view._renderOptionCard("imbalance_strategy", "balanced_weights", q.imbalance_strategy, "Ponderación de Clases", "Pondera la función de pérdida con class_weight='balanced'.", "Balanced Weights")}
            ${view._renderOptionCard("imbalance_strategy", "resampling", q.imbalance_strategy, "Re-muestreo (SMOTE)", "Genera muestras sintéticas in-fold de la clase minoritaria.", "SMOTE")}
            ${view._renderOptionCard("imbalance_strategy", "none", q.imbalance_strategy, "Distribución Natural", "Entrena con la distribución original sin ponderación.", "None")}
          </div>
        </div>

        <!-- Section 6: Technical Overrides -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 pt-3 border-t border-[#242A36] text-xs font-sans">
          <div class="p-3.5 rounded-xl bg-[#11151E] border border-[#242A36] space-y-1.5">
            <label class="block font-medium text-[#F7F8FA]">Métrica Primaria de Optimización:</label>
            <select id="qMetricSelect" class="bg-[#161B26] border border-[#242A36] text-[#F7F8FA] text-xs rounded-lg px-3 py-2 w-full font-mono outline-none focus:border-[#4F67FF]">
              <option value="" ${!q.primary_metric_override ? "selected" : ""}>Automática según problema (${isRegression ? "RMSE" : "ROC-AUC"})</option>
              ${isRegression ? `
                <option value="rmse" ${q.primary_metric_override === "rmse" ? "selected" : ""}>RMSE (Root Mean Squared Error)</option>
                <option value="mae" ${q.primary_metric_override === "mae" ? "selected" : ""}>MAE (Mean Absolute Error)</option>
                <option value="r2" ${q.primary_metric_override === "r2" ? "selected" : ""}>R² (Coeficiente de Determinación)</option>
              ` : `
                <option value="roc_auc" ${q.primary_metric_override === "roc_auc" ? "selected" : ""}>ROC-AUC (Área bajo la curva ROC)</option>
                <option value="pr_auc" ${q.primary_metric_override === "pr_auc" ? "selected" : ""}>PR-AUC (Precision-Recall AUC)</option>
                <option value="f1" ${q.primary_metric_override === "f1" ? "selected" : ""}>F1-Score (Media armónica Precision/Recall)</option>
                <option value="accuracy" ${q.primary_metric_override === "accuracy" ? "selected" : ""}>Accuracy (Exactitud global)</option>
              `}
            </select>
          </div>

          <div class="p-3.5 rounded-xl bg-[#11151E] border border-[#242A36] space-y-1.5">
            <label class="block font-medium text-[#F7F8FA]">Estrategia de Partición (Cross-Validation):</label>
            <select id="qSplitSelect" class="bg-[#161B26] border border-[#242A36] text-[#F7F8FA] text-xs rounded-lg px-3 py-2 w-full font-mono outline-none focus:border-[#4F67FF]">
              ${!isRegression ? `
                <option value="stratified_kfold" ${q.split_strategy_recommendation === "stratified_kfold" ? "selected" : ""}>Stratified K-Fold (Estratificado por clases)</option>
              ` : ""}
              <option value="kfold" ${q.split_strategy_recommendation === "kfold" ? "selected" : ""}>Standard K-Fold (Aleatorio)</option>
              <option value="time_series" ${q.split_strategy_recommendation === "time_series" || q.split_strategy_recommendation === "time_series_split" ? "selected" : ""}>Time-Series Split (Progresión Temporal)</option>
              <option value="group_kfold" ${q.split_strategy_recommendation === "group_kfold" ? "selected" : ""}>Group K-Fold (Agrupamiento por Entidad)</option>
              <option value="holdout" ${q.split_strategy_recommendation === "holdout" ? "selected" : ""}>Holdout Simple (80/20)</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Advisor Intelligence & Compatible Tools Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Reasoning Card (2 cols) -->
        <div class="lg:col-span-2 workbench-card p-5 space-y-3">
          <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
            <div class="flex items-center space-x-2">
              <span class="text-[#53C8FF]">${icon("sparkles", "icon-sm")}</span>
              <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">Razonamiento & Justificación del Asesor</h4>
            </div>
            <span class="badge-intel text-xs px-2 py-0.5 rounded font-mono font-bold">
              Confianza: ${(q.confidence * 100).toFixed(0)}%
            </span>
          </div>
          <div class="p-3.5 rounded-xl bg-[#11151E] border border-[#242A36] text-xs font-sans text-[#F7F8FA] leading-relaxed">
            ${escapeHtml(q.reasoning || "Configuración inicial. Pulsa 'Generar sugerencias' para obtener la justificación analítica basada en los metadatos.")}
          </div>

          <!-- Launch CTA button -->
          <div class="pt-2 flex items-center justify-between">
            <div class="text-[11px] text-[#8B95A7] font-sans">
              El encuadre se inyecta en el planificador y restringe el catálogo de modelos candidatos.
            </div>
            <button id="btnLaunchWithFraming" class="btn-signal text-xs flex items-center space-x-1.5 shadow-lg shadow-[#4F67FF]/25">
              <span>Lanzar experimento con este encuadre →</span>
            </button>
          </div>
        </div>

        <!-- Compatible Tools Panel (1 col) -->
        <div class="workbench-card p-5 space-y-3">
          <div class="flex items-center space-x-2 border-b border-[#242A36] pb-3">
            <span class="text-[#4F67FF]">${icon("cpu", "icon-sm")}</span>
            <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">Herramientas Compatibles</h4>
          </div>

          <div class="space-y-2.5">
            ${compatibleTools.length > 0 ? compatibleTools.map(t => `
              <div class="p-2.5 rounded-lg bg-[#11151E] border border-[#242A36] space-y-1">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-semibold text-[#F7F8FA] font-sans">${t.title}</span>
                  <span class="${t.badgeClass} text-[10px] px-1.5 py-0.2 rounded font-mono font-medium">${t.badge}</span>
                </div>
                <p class="text-[11px] text-[#8B95A7] font-sans">${t.desc}</p>
              </div>
            `).join("") : `
              <div class="text-xs text-[#8B95A7] p-4 text-center">Configura el encuadre para ver las herramientas y modelos compatibles.</div>
            `}
          </div>
        </div>
      </div>
    </div>
  `;
}

export function _bindQuestionnaireEvents(view) {
  // Option Cards click
  view.container.querySelectorAll(".q-option-card").forEach(card => {
    card.addEventListener("click", () => {
      const field = card.getAttribute("data-q-field");
      const val = card.getAttribute("data-q-value");
      if (!view.questionnaire) {
        view.questionnaire = {
          dataset_id: view.activeDatasetId,
          error_cost: "balanced",
          temporal_structure: "cross_sectional",
          latency_constraint: "standard_interactive",
          explainability: "moderate",
          imbalance_strategy: "auto",
          primary_metric_override: null,
          split_strategy_recommendation: (view.profile?.task_type || "").includes("regression") ? "kfold" : "stratified_kfold",
          user_confirmed: false,
          engine_source: "heuristic",
          confidence: 0.9,
          reasoning: "Ajuste manual del operador humano.",
        };
      }
      view.questionnaire[field] = val;
      view.questionnaire.user_confirmed = false;
      // Keep split recommendation aligned with temporal framing so New Experiment
      // does not inherit a leftover TimeSeriesSplit on cross-sectional data.
      if (field === "temporal_structure") {
        const isReg = (view.profile?.task_type || "").includes("regression");
        if (val === "sequential_time_series") {
          view.questionnaire.split_strategy_recommendation = "time_series";
        } else if (val === "grouped_cohorts") {
          view.questionnaire.split_strategy_recommendation = "group_kfold";
        } else {
          view.questionnaire.split_strategy_recommendation = isReg ? "kfold" : "stratified_kfold";
        }
      }
      view.render();
    });
  });

  // Metric override select
  view.container.querySelector("#qMetricSelect")?.addEventListener("change", e => {
    if (!view.questionnaire) {
      view.questionnaire = { dataset_id: view.activeDatasetId, user_confirmed: false };
    }
    view.questionnaire.primary_metric_override = e.target.value || null;
    view.questionnaire.user_confirmed = false;
    view.render();
  });

  // Split strategy select
  view.container.querySelector("#qSplitSelect")?.addEventListener("change", e => {
    if (!view.questionnaire) {
      view.questionnaire = { dataset_id: view.activeDatasetId, user_confirmed: false };
    }
    let split = e.target.value;
    const temporal = view.questionnaire.temporal_structure || "cross_sectional";
    // Guard: selecting Time-Series while framing is cross-sectional upgrades framing.
    if ((split === "time_series" || split === "time_series_split") && temporal !== "sequential_time_series") {
      view.questionnaire.temporal_structure = "sequential_time_series";
    } else if ((split === "group_kfold") && temporal !== "grouped_cohorts") {
      view.questionnaire.temporal_structure = "grouped_cohorts";
    } else if (
      (split === "stratified_kfold" || split === "kfold" || split === "holdout")
      && (temporal === "sequential_time_series" || temporal === "grouped_cohorts")
    ) {
      view.questionnaire.temporal_structure = "cross_sectional";
    }
    view.questionnaire.split_strategy_recommendation = split;
    view.questionnaire.user_confirmed = false;
    view.render();
  });

  // Generate Suggestions button
  view.container.querySelector("#btnGenerateQuestionnaire")?.addEventListener("click", async () => {
    if (view.questionnaireGenerating || !view.activeDatasetId) return;
    view.questionnaireGenerating = true;
    view.render();
    try {
      const res = await api.generateDatasetQuestionnaire(view.activeDatasetId);
      if (res && res.questionnaire) {
        view.questionnaire = res.questionnaire;
      }
    } catch (err) {
      alert("Error generando sugerencias del cuestionario: " + err.message);
    } finally {
      view.questionnaireGenerating = false;
      view.render();
    }
  });

  // Save and Confirm button
  view.container.querySelector("#btnSaveQuestionnaire")?.addEventListener("click", async () => {
    if (view.questionnaireSaving || !view.activeDatasetId) return;
    if (!view.questionnaire) {
      view.questionnaire = {
        dataset_id: view.activeDatasetId,
        error_cost: "balanced",
        temporal_structure: "cross_sectional",
        latency_constraint: "standard_interactive",
        explainability: "moderate",
        imbalance_strategy: "auto",
        primary_metric_override: null,
        split_strategy_recommendation: (view.profile?.task_type || "").includes("regression") ? "kfold" : "stratified_kfold",
        user_confirmed: true,
        engine_source: "heuristic",
        confidence: 1.0,
        reasoning: "Encuadre confirmado por el usuario.",
      };
    }
    view.questionnaire.user_confirmed = true;
    view.questionnaireSaving = true;
    view.render();
    try {
      const res = await api.saveDatasetQuestionnaire(view.questionnaire);
      if (res && res.questionnaire) {
        view.questionnaire = res.questionnaire;
      }
    } catch (err) {
      alert("Error guardando el encuadre: " + err.message);
    } finally {
      view.questionnaireSaving = false;
      view.render();
    }
  });

  // Launch experiment with framing
  view.container.querySelector("#btnLaunchWithFraming")?.addEventListener("click", async () => {
    if (view.questionnaire) {
      view.questionnaire.user_confirmed = true;
      await api.saveDatasetQuestionnaire(view.questionnaire).catch(() => null);
    }
    const feats = Array.from(view.selectedFeatures);
    const currentMap = store.getState().selectedFeaturesByDataset || {};
    store.setState({
      customFeatures: feats,
      selectedFeatures: feats,
      selectedFeaturesByDataset: { ...currentMap, [view.activeDatasetId]: feats },
    });
    bus.emit("modal:new-experiment");
  });
}

