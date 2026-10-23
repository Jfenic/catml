/**
 * NewExperimentModal — Clean Experiment Creator
 * Provides Auto, Guided, and Manual operating strategies with Compute Budget selection.
 */
import { store } from "../store.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { escapeHtml, isRunActive } from "../utils.js";

export class NewExperimentModal {
  constructor() {
    this.container = null;
    this.mode = "auto"; // 'auto', 'guided', 'manual'
    this.budget = "balanced"; // 'quick', 'balanced', 'thorough'
    this.validationStrategy = "stratified_kfold";
    this.customFeatures = [];
    this.recipes = [];
    this.selectedRecipeId = "";
    this.preselectedModel = null;
    this.questionnaire = null;
    this.datasetProfile = null;
    this.modelPlugins = {};
  }

  _modelLabel(id, displayName) {
    const p = this.modelPlugins[id];
    const label = displayName || id;
    if (!p) return escapeHtml(label);
    if (p.is_native) return `${escapeHtml(label)} <span class="text-[9px] text-[#22C55E] font-mono">nativo</span>`;
    if (p.fallback_backend) {
      return `${escapeHtml(label)} <span class="text-[9px] text-[#F59E0B] font-mono">fallback: ${escapeHtml(p.fallback_backend)}</span>`;
    }
    return `${escapeHtml(label)} <span class="text-[9px] text-[#EF4444] font-mono">no disponible</span>`;
  }

  async mount(container, options = {}) {
    this.container = container;
    const state = store.getState();
    const targetDatasetId = state.activeDatasetId;

    if (options.preselectedModel || state.preselectedModel) {
      this.mode = "guided";
      this.preselectedModel = options.preselectedModel || state.preselectedModel;
      store.setState({ preselectedModel: null });
    } else {
      this.preselectedModel = null;
    }

    if (options.preselectedRecipeId) {
      this.selectedRecipeId = options.preselectedRecipeId;
    }

    const pluginsPromise = api.getPlugins().catch(() => []);
    if (targetDatasetId) {
      const [recipes, q, profile, plugins] = await Promise.all([
        api.getRecipes(targetDatasetId).catch(() => []),
        api.getDatasetQuestionnaire(targetDatasetId).catch(() => null),
        api.getDatasetProfile(targetDatasetId).catch(() => null),
        pluginsPromise,
      ]);
      this.recipes = recipes || [];
      this.questionnaire = q;
      this.datasetProfile = profile;
      this.modelPlugins = Object.fromEntries(
        (plugins || [])
          .filter(p => (p.plugin_type || "") === "model" || (p.plugin_type || "").endsWith("MODEL"))
          .map(p => [p.plugin_id, p])
      );
    } else {
      const plugins = await pluginsPromise;
      this.modelPlugins = Object.fromEntries(
        (plugins || [])
          .filter(p => (p.plugin_type || "") === "model" || (p.plugin_type || "").endsWith("MODEL"))
          .map(p => [p.plugin_id, p])
      );
    }

    const activeDs = state.activeDataset
      || (state.datasets || []).find(d => d.id === targetDatasetId);
    const taskType = (this.datasetProfile?.task_type || activeDs?.task_type || "").toLowerCase();
    const isRegression = taskType.includes("regression");

    if (this.questionnaire?.split_strategy_recommendation) {
      this.validationStrategy = this.questionnaire.split_strategy_recommendation;
    } else if (isRegression) {
      this.validationStrategy = "kfold";
    } else {
      this.validationStrategy = "stratified_kfold";
    }

    // Coerce CV to framing: never launch TimeSeriesSplit without sequential framing
    // (and never leave a hidden time_series_split selected when the card is hidden).
    const temporal = this.questionnaire?.temporal_structure;
    const isTsSplit = this.validationStrategy === "time_series"
      || this.validationStrategy === "time_series_split";
    const isGroupSplit = this.validationStrategy === "group_kfold"
      || this.validationStrategy === "group_cv"
      || this.validationStrategy === "grouped_kfold";
    if (temporal === "sequential_time_series") {
      this.validationStrategy = "time_series";
    } else if (temporal === "grouped_cohorts") {
      this.validationStrategy = "group_kfold";
    } else if (isTsSplit || isGroupSplit) {
      this.validationStrategy = isRegression ? "kfold" : "stratified_kfold";
    }

    this.render();
  }

  render() {
    const state = store.getState();
    this.customFeatures = (state.selectedFeaturesByDataset && state.selectedFeaturesByDataset[state.activeDatasetId])
      || state.customFeatures
      || state.selectedFeatures
      || [];
    const customFeatures = this.customFeatures;
    const activeDs = store.getActiveDataset();
    const dsRuns = store.getRunsForActiveDataset();
    const activeRun = store.getActiveRun()
      || dsRuns.find(r => isRunActive(r))
      || dsRuns[0]
      || null;

    const taskTypeRaw = (this.datasetProfile?.task_type || activeDs?.task_type || "binary_classification").toLowerCase();
    const isRegression = taskTypeRaw.includes("regression");
    const hasImages = Boolean(this.datasetProfile?.columns?.some(c => c.is_image));
    // Only surface Time-Series CV when the framing questionnaire says so (avoids mixing i.i.d. vs temporal).
    const isTimeSeries = this.questionnaire?.temporal_structure === "sequential_time_series";
    const isGrouped = this.questionnaire?.temporal_structure === "grouped_cohorts";
    const hasDatetimeHint = Boolean(
      this.datasetProfile?.columns?.some(c =>
        c.dtype?.includes("datetime")
        || c.name.toLowerCase().includes("date")
        || c.name.toLowerCase().includes("timestamp")
      )
    );

    const datasetName = escapeHtml(activeDs ? (activeDs.name || activeDs.id) : "No Dataset Registered");
    const targetCol = escapeHtml(activeDs ? (activeDs.target_column || activeDs.target || "Target") : "None");
    const metricName = escapeHtml(activeRun ? (activeRun.metric || (isRegression ? "RMSE" : "ROC-AUC")) : (isRegression ? "RMSE" : "ROC-AUC"));
    const taskType = escapeHtml((activeDs?.task_type || "binary_classification").replace("_", " "));

    const hasTarget = Boolean(activeDs || activeRun);
    const showModelPicker = this.mode === "guided" || this.mode === "manual";

    this.container.innerHTML = `
      <div class="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="workbench-card modal-responsive max-w-xl w-full p-6 space-y-5 border-slate-700 shadow-2xl">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 class="text-base font-bold text-[#F7F8FA] font-sans">Crear Nuevo Experimento</h3>
              <p class="text-xs text-[#8B95A7] font-sans">Configuración rápida de experimento AutoML guiada por evidencia</p>
            </div>
            <button id="btnCloseModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded hover:bg-[#161B26] transition-colors" aria-label="Cerrar modal">${icon("x", "icon-sm")}</button>
          </div>

          <!-- Dataset Card -->
          <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-1">
            <div class="text-[10px] uppercase font-semibold text-[#8B95A7] tracking-wider font-sans">Dataset Objetivo</div>
            <div class="text-sm font-bold text-[#F7F8FA] font-mono">${datasetName}</div>
            <div class="text-xs text-[#8B95A7] font-sans">Tarea: <span class="capitalize text-[#F7F8FA]">${taskType}</span> • Variable Objetivo: <span class="text-[#4F67FF] font-mono font-bold">${targetCol}</span> • Métrica: <span class="text-[#22C55E] font-mono font-bold">${metricName}</span></div>
          </div>

          ${!hasTarget ? `
            <div class="p-3 rounded-lg bg-amber-950/30 border border-amber-800/60 text-xs text-amber-300 flex items-center justify-between font-sans">
              <span>No hay ningún dataset registrado en este espacio de trabajo.</span>
              <button id="btnGoToDatasets" class="font-bold underline text-amber-200 hover:text-white">Registrar Dataset →</button>
            </div>
          ` : ""}

          <!-- How should CATML operate? -->
          <div class="space-y-2">
            <label class="text-xs font-semibold text-[#8B95A7] uppercase tracking-wider font-sans">¿Cómo debe operar CATML?</label>
            <div class="grid grid-cols-3 gap-3">
              <div class="mode-card p-3 rounded-xl border cursor-pointer text-xs ${this.mode === 'auto' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-mode="auto">
                <div class="flex items-center gap-1.5 font-bold text-[#F7F8FA] font-sans">${icon("zap", "icon-sm text-[#4F67FF]")} <span>Automático</span></div>
                <div class="text-[11px] text-[#8B95A7] mt-1 font-sans">Selección autónoma de modelos, ingeniería de variables y HPO bayesiano.</div>
              </div>

              <div class="mode-card p-3 rounded-xl border cursor-pointer text-xs ${this.mode === 'guided' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-mode="guided">
                <div class="flex items-center gap-1.5 font-bold text-[#F7F8FA] font-sans">${icon("compass", "icon-sm text-[#6956E8]")} <span>Guiado</span></div>
                <div class="text-[11px] text-[#8B95A7] mt-1 font-sans">CATML propone candidatos; tú configuras restricciones y modelos.</div>
              </div>

              <div class="mode-card p-3 rounded-xl border cursor-pointer text-xs ${this.mode === 'manual' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-mode="manual">
                <div class="flex items-center gap-1.5 font-bold text-[#F7F8FA] font-sans">${icon("sliders-horizontal", "icon-sm text-[#8B95A7]")} <span>Manual</span></div>
                <div class="text-[11px] text-[#8B95A7] mt-1 font-sans">Control manual paso a paso del pipeline y selección directa.</div>
              </div>
            </div>
          </div>

          <!-- Model picker (Guided + Manual — Manual previously hid checkboxes and silently defaulted to lightgbm) -->
          ${
            showModelPicker
              ? `
            <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-3 text-xs font-sans">
              <div class="font-semibold text-[#F7F8FA]">Familias de Modelos Algorítmicos:</div>
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[#8B95A7]">
                <label class="flex items-center space-x-1.5"><input type="checkbox" ${(!this.preselectedModel || this.preselectedModel === 'lightgbm') ? 'checked' : ''} value="lightgbm" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">${this._modelLabel("lightgbm", "LightGBM")}</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" ${(!this.preselectedModel || this.preselectedModel === 'xgboost') ? 'checked' : ''} value="xgboost" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">${this._modelLabel("xgboost", "XGBoost")}</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" ${(!this.preselectedModel || this.preselectedModel === 'catboost') ? 'checked' : ''} value="catboost" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">${this._modelLabel("catboost", "CatBoost")}</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" ${this.preselectedModel === 'extra_trees' ? 'checked' : ''} value="extra_trees" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">Extra Trees</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" ${this.preselectedModel === 'mlp' ? 'checked' : ''} value="mlp" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">MLP Neural</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" ${this.preselectedModel === 'random_forest' ? 'checked' : ''} value="random_forest" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">Random Forest</span></label>
                ${isRegression ? `
                  <label class="flex items-center space-x-1.5"><input type="checkbox" ${this.preselectedModel === 'ridge' ? 'checked' : ''} value="ridge" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">Ridge Regression</span></label>
                ` : `
                  <label class="flex items-center space-x-1.5"><input type="checkbox" ${this.preselectedModel === 'logistic_regression' ? 'checked' : ''} value="logistic_regression" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">Logistic Regression</span></label>
                `}
                <label class="flex items-center space-x-1.5"><input type="checkbox" ${this.preselectedModel === 'voting_ensemble' ? 'checked' : ''} value="voting_ensemble" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">Ensamble / Mezclador</span></label>
                ${hasImages ? `
                  <label class="flex items-center space-x-1.5" title="Extracción de Embeddings de Visión + Red MLP"><input type="checkbox" ${this.preselectedModel === 'vision' ? 'checked' : ''} value="vision" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#53C8FF]">Visión MLP</span></label>
                ` : `
                  <label class="flex items-center space-x-1.5 opacity-40 cursor-not-allowed" title="Requiere columnas de tipo imagen en el dataset"><input type="checkbox" disabled value="vision" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#8B95A7]">Visión MLP (Sin imágenes)</span></label>
                `}
              </div>
              <div class="grid grid-cols-2 gap-2 pt-2 border-t border-[#252C38] text-[11px] text-[#8B95A7]">
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span>Ingeniería de variables</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span>Optuna Bayesian HPO</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span>Cross-Validation</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span>Principio Proponer ≠ Aceptar</span></label>
              </div>
            </div>
          `
              : ""
          }

          <!-- Feature Set Selection -->
          <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-2 text-xs font-sans">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-[#4F67FF]">${icon("sparkles", "icon-sm")}</span>
                <span class="font-semibold text-[#F7F8FA]">Selección de Características:</span>
              </div>
              <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono font-bold">${customFeatures && customFeatures.length > 0 ? `${customFeatures.length} Características seleccionadas` : 'Todas las recomendadas'}</span>
            </div>
            ${customFeatures && customFeatures.length > 0 ? `
              <div class="flex flex-wrap gap-1 max-h-20 overflow-y-auto p-1.5 bg-[#090C12] rounded-lg border border-[#252C38]">
                ${customFeatures.map(f => `<span class="bg-[#151B26] text-[#4F67FF] border border-[#252C38] px-1.5 py-0.5 rounded text-[10px] font-mono">${escapeHtml(f)}</span>`).join("")}
              </div>
            ` : `
              <div class="text-[11px] text-[#8B95A7]">Utilizando todas las características seleccionadas y recomendadas por el perfilado estadístico.</div>
            `}
          </div>

          <!-- Recipe Selection (Optional) -->
          ${this.recipes && this.recipes.length > 0 ? `
            <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-2 text-xs font-sans">
              <div class="flex items-center justify-between">
                <div class="flex items-center space-x-2">
                  <span class="text-[#4F67FF]">${icon("recipe", "icon-sm")}</span>
                  <span class="font-semibold text-[#F7F8FA]">Receta Reproducible (Opcional):</span>
                </div>
                <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono font-bold">${this.recipes.length} disponibles</span>
              </div>
              <select id="selectExperimentRecipe" class="input-base w-full bg-[#090C12] border border-[#252C38] text-[#F7F8FA] text-xs font-mono py-1.5 px-3 rounded-lg focus:border-[#4F67FF] outline-none">
                <option value="">Ninguna — Configuración estándar</option>
                ${this.recipes.map(r => `
                  <option value="${escapeHtml(r.id)}" ${this.selectedRecipeId === r.id ? 'selected' : ''}>${escapeHtml(r.name || r.id)} [${escapeHtml(String(r.status || "").toUpperCase())}]</option>
                `).join("")}
              </select>
            </div>
          ` : ""}

          <!-- Compute Budget -->
          <div class="space-y-2">
            <label class="text-xs font-semibold text-slate-300 uppercase tracking-wider">Presupuesto de Cómputo</label>
            <div class="grid grid-cols-3 gap-3">
              <div class="budget-card p-2.5 rounded-lg border text-center cursor-pointer text-xs ${this.budget === 'quick' ? 'border-indigo-500 bg-indigo-950/30' : 'border-slate-800 bg-slate-900/50'}" data-budget="quick">
                <div class="font-bold text-slate-200">Rápido</div>
                <div class="text-[10px] text-slate-400">~60s • Baselines</div>
              </div>

              <div class="budget-card p-2.5 rounded-lg border text-center cursor-pointer text-xs ${this.budget === 'balanced' ? 'border-indigo-500 bg-indigo-950/30' : 'border-slate-800 bg-slate-900/50'}" data-budget="balanced">
                <div class="font-bold text-slate-200">Equilibrado</div>
                <div class="text-[10px] text-slate-400">~5m • LightGBM + HPO</div>
              </div>

              <div class="budget-card p-2.5 rounded-lg border text-center cursor-pointer text-xs ${this.budget === 'thorough' ? 'border-indigo-500 bg-indigo-950/30' : 'border-slate-800 bg-slate-900/50'}" data-budget="thorough">
                <div class="font-bold text-slate-200">Exhaustivo</div>
                <div class="text-[10px] text-slate-400">~20m • Ensamble Completo</div>
              </div>
            </div>
          </div>

          <!-- Validation Strategy -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <label class="text-xs font-semibold text-[#8B95A7] uppercase tracking-wider font-sans">Estrategia de Validación</label>
              <span class="text-[10px] text-[#4F67FF] font-mono">CV Splitting</span>
            </div>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
              ${!isRegression ? `
                <div class="val-strategy-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${this.validationStrategy === 'stratified_kfold' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-strategy="stratified_kfold">
                  <div class="font-bold text-[#F7F8FA] font-sans">Stratified K-Fold</div>
                  <div class="text-[10px] text-[#8B95A7] font-sans">Clases equilibradas</div>
                </div>
              ` : `
                <div class="val-strategy-card p-2.5 rounded-xl border text-center text-xs opacity-40 cursor-not-allowed border-[#252C38] bg-[#090C12]" data-strategy="stratified_kfold" data-disabled="true" title="No aplicable a regresión">
                  <div class="font-bold text-[#8B95A7] font-sans">Stratified K-Fold</div>
                  <div class="text-[10px] text-[#8B95A7] font-sans">Solo Clasificación</div>
                </div>
              `}

              <div class="val-strategy-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${this.validationStrategy === 'kfold' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-strategy="kfold">
                <div class="font-bold text-[#F7F8FA] font-sans">K-Fold Estándar</div>
                <div class="text-[10px] text-[#8B95A7] font-sans">5 pliegues aleatorios</div>
              </div>

              ${isTimeSeries ? `
                <div class="val-strategy-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${this.validationStrategy === 'time_series' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-strategy="time_series">
                  <div class="font-bold text-[#F7F8FA] font-sans">Time-Series</div>
                  <div class="text-[10px] text-[#22C55E] font-sans">Según encuadre temporal</div>
                </div>
              ` : ""}

              ${isGrouped ? `
                <div class="val-strategy-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${this.validationStrategy === 'group_kfold' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-strategy="group_kfold">
                  <div class="font-bold text-[#F7F8FA] font-sans">Group K-Fold</div>
                  <div class="text-[10px] text-[#22C55E] font-sans">Según encuadre por grupos</div>
                </div>
              ` : ""}

              <div class="val-strategy-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${this.validationStrategy === 'holdout' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-strategy="holdout">
                <div class="font-bold text-[#F7F8FA] font-sans">Holdout (80/20)</div>
                <div class="text-[10px] text-[#8B95A7] font-sans">División simple rápida</div>
              </div>
            </div>
            ${!isTimeSeries && hasDatetimeHint ? `
              <p class="text-[11px] text-[#F59E0B] font-sans">
                Hay columnas con aspecto de fecha, pero el encuadre es independiente. Si el orden temporal importa, abre <strong class="text-[#F7F8FA]">01 Datos → Encuadre</strong> y elige «Serie temporal».
              </p>
            ` : ""}
          </div>

          <!-- Footer Actions -->
          <div class="pt-4 border-t border-[#27272e] flex items-center justify-end space-x-3">
            <button id="btnCancelModal" class="btn-ghost">
              Cancelar
            </button>
            <button id="btnStartAutoML" ${!hasTarget ? "disabled" : ""} class="btn-signal ${!hasTarget ? 'opacity-50 cursor-not-allowed' : ''}">
              <span class="inline-flex items-center gap-1.5">${icon("play", "icon-sm")} <span>Iniciar Entrenamiento</span></span>
            </button>
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
  }

  _bindEvents() {
    this.container.querySelector("#btnCloseModal")?.addEventListener("click", () => this.destroy());
    this.container.querySelector("#btnCancelModal")?.addEventListener("click", () => this.destroy());
    this.container.querySelector("#btnGoToDatasets")?.addEventListener("click", () => {
      this.destroy();
      store.setNav("datasets");
    });

    this.container.querySelectorAll(".mode-card").forEach(el => {
      el.addEventListener("click", () => {
        this.mode = el.getAttribute("data-mode");
        this.render();
      });
    });

    this.container.querySelectorAll(".budget-card").forEach(el => {
      el.addEventListener("click", () => {
        this.budget = el.getAttribute("data-budget");
        this.render();
      });
    });

    this.container.querySelectorAll(".val-strategy-card").forEach(el => {
      el.addEventListener("click", () => {
        if (el.getAttribute("data-disabled") === "true") return;
        this.validationStrategy = el.getAttribute("data-strategy");
        this.render();
      });
    });

    this.container.querySelector("#selectExperimentRecipe")?.addEventListener("change", (e) => {
      this.selectedRecipeId = e.target.value;
    });

    this.container.querySelector("#btnStartAutoML")?.addEventListener("click", async () => {
      const btn = this.container.querySelector("#btnStartAutoML");
      const modalBox = this.container.querySelector(".workbench-card");
      if (!btn || !modalBox) return;

      try {
        const state = store.getState();
        let activeRun = (state.runs || []).find(r => r.dataset_id === state.activeDatasetId) || null;
        if (!activeRun) {
          const freshRuns = await api.getRuns().catch(() => []);
          store.setState({ runs: freshRuns });
          activeRun = freshRuns.find(r => r.dataset_id === state.activeDatasetId) || null;
        }
        if (!activeRun && state.activeDatasetId) {
          const createRes = await api.createRun(state.activeDatasetId).catch(() => null);
          if (createRes && createRes.run_id) {
            const freshRuns = await api.getRuns().catch(() => []);
            store.setState({ runs: freshRuns, activeRunId: createRes.run_id });
            activeRun = freshRuns.find(r => r.id === createRes.run_id) || { id: createRes.run_id, dataset_id: state.activeDatasetId };
          }
        }
        if (!activeRun) {
          alert("No se encontró ninguna ejecución activa. Por favor selecciona o registra un dataset primero.");
          return;
        }

        let selectedModels = ["lightgbm"];
        if (this.mode === "guided" || this.mode === "manual") {
          const checked = Array.from(this.container.querySelectorAll(".model-check:checked")).map(c => c.value);
          if (checked.length > 0) {
            selectedModels = checked;
          }
        } else {
          selectedModels = ["lightgbm", "xgboost", "catboost"];
        }

        // Render Live Animated Training Overlay with Spinner and Progress Bar
        modalBox.innerHTML = `
          <div class="p-8 text-center space-y-6">
            <div class="relative w-20 h-20 mx-auto">
              <!-- Outer glowing rotating ring -->
              <svg class="animate-spin w-20 h-20 text-indigo-500" viewBox="0 0 24 24" fill="none">
                <circle class="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3"></circle>
                <path class="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <!-- Center pulse icon -->
              <div class="absolute inset-0 flex items-center justify-center text-[#4F67FF] animate-pulse">${icon("zap", "icon-lg", 24)}</div>
            </div>

            <div class="space-y-2">
              <h3 id="trainingStatusTitle" class="text-base font-bold text-slate-100">Training AutoML Pipeline...</h3>
              <p id="trainingStatusDesc" class="text-xs text-indigo-300 font-mono">Waiting for worker</p>
            </div>

            <!-- Animated Progress Bar -->
            <div class="space-y-1.5 max-w-md mx-auto">
              <div class="flex justify-between text-[11px] font-mono text-slate-400">
                <span id="trainingPercentLabel">0%</span>
                <span class="text-slate-500">Verified progress</span>
              </div>
              <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden p-0.5 border border-slate-700">
                <div id="trainingProgressBar" class="bg-indigo-500 h-2 rounded-full progress-striped transition-all duration-500 ease-out" style="width: 0%"></div>
              </div>
            </div>

            <div class="text-[11px] text-slate-400 font-mono bg-slate-950 p-2.5 rounded border border-slate-800/80">
              Models: <span class="text-slate-200 font-bold">${escapeHtml(selectedModels.join(", "))}</span> • Budget: <span class="text-indigo-400 font-bold">${escapeHtml(this.budget)}</span> • Strategy: <span class="text-[#4F67FF] font-bold">${escapeHtml(this.validationStrategy)}</span>
            </div>
          </div>
        `;

        const progressBar = modalBox.querySelector("#trainingProgressBar");
        const percentLabel = modalBox.querySelector("#trainingPercentLabel");
        const statusDesc = modalBox.querySelector("#trainingStatusDesc");

        if (this.selectedRecipeId) {
          if (statusDesc) statusDesc.textContent = `Submitting reproducible recipe ${this.selectedRecipeId}...`;
          const res = await api.submitRecipe(this.selectedRecipeId, activeRun.id);
          if (progressBar) progressBar.style.width = "100%";
          if (percentLabel) percentLabel.textContent = "100%";
          if (statusDesc) {
            statusDesc.className = "text-xs text-emerald-400 font-mono font-bold";
            statusDesc.textContent = `Recipe submitted! Executed: ${(res.experiment_ids || []).join(", ")}`;
          }
          setTimeout(() => {
            this.destroy();
            store.setNav("studio");
          }, 800);
          return;
        }

        // Keep in sync with automl.domain.experiments.budgets.EXPERIMENT_BUDGET_SECONDS
        const budgetSecondsMap = {
          quick: 60,
          balanced: 300,
          thorough: 1200,
        };
        const timeBudgetSeconds = budgetSecondsMap[this.budget] || 300;

        const res = await api.createAndRunExperiment({
          run_id: activeRun.id,
          mode: this.mode,
          budget: this.budget,
          time_budget_seconds: timeBudgetSeconds,
          models: selectedModels,
          feature_names: this.customFeatures && this.customFeatures.length > 0 ? this.customFeatures : undefined,
          validation_strategy: this.validationStrategy,
          recipe_id: this.selectedRecipeId || undefined,
        }, job => {
          const percent = job.total ? Math.round(job.completed * 100 / job.total) : 0;
          if (progressBar) progressBar.style.width = `${percent}%`;
          if (percentLabel) percentLabel.textContent = `${percent}%`;
          if (statusDesc) statusDesc.textContent = `${job.status}: ${job.message}`;
        });

        if (progressBar) progressBar.style.width = "100%";
        if (percentLabel) percentLabel.textContent = "100%";
        if (statusDesc) {
          statusDesc.className = "text-xs text-emerald-400 font-mono font-bold";
          statusDesc.textContent = `Completed! ${res.primary_metric}: ${res.primary_score.toFixed(5)}`;
        }

        setTimeout(() => {
          this.destroy();
          store.setNav("studio");
        }, 800);

      } catch (err) {
        alert("Error launching experiment: " + err.message);
        this.destroy();
      }
    });
  }

  destroy() {
    if (this.container) {
      this.container.innerHTML = "";
    }
  }
}
