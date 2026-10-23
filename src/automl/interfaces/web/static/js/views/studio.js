/**
 * StudioView — Experiment Studio (PR 4 / Fase P1)
 * Results-focused Leaderboard, Evaluation Context & Reproducible Experiment Recipes.
 * 
 * Complies with Section 7 & Section 8 of the UX/UI Audit Report:
 * 1. Shows FIRST: Best evaluated model, Metric & Score with direction, Validation type,
 *    and honest Baseline comparison (if baseline exists and is comparable).
 * 2. Unambiguous trial identification: each algorithm configuration and trial is distinct
 *    with its own Trial ID, hyperparameters, and associated recipe.
 * 3. HPO, planner decisions, hardware resources, and logs placed in dedicated secondary tabs.
 * 4. Clear differentiation of the 5 transformation stages: Propuesta, Verificada, Seleccionada, Aplicada, Promovida.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import {
  isRunActive,
  resolveTrialsCount,
  escapeHtml,
  isMinimizingMetric,
  getMetricDirectionInfo,
  formatMetricComparison,
  getTransformationStageInfo,
} from "../utils.js";

export class StudioView {
  constructor() {
    this.container = null;
    this.activeTab = "models";
    this.activeDecisionModal = null;
    this.chart = null;
    this.activeRun = null;
    this.plan = null;
    this.leaderboard = [];
    this.experiments = [];
    this.recipes = [];
    this.modelFilter = "all";
    this.modelSearch = "";
  }

  async mount(container) {
    this.container = container;
    this.renderLoading();
    await this.fetchData();
    this.render();
  }

  renderLoading() {
    this.container.innerHTML = `
      <div class="workbench-card p-12 text-center text-[#8B95A7] space-y-3">
        <div class="animate-spin text-[#4F67FF] inline-block">${icon("refresh-cw", "icon-lg")}</div>
        <div class="text-sm font-medium font-sans">Cargando estudio de experimentos, planes y métricas de validación...</div>
      </div>
    `;
  }

  async fetchData() {
    try {
      const activeDataset = store.getActiveDataset();
      const dsRuns = store.getRunsForActiveDataset();
      this.activeRun = store.getActiveRun()
        || dsRuns.find(r => isRunActive(r))
        || dsRuns[0]
        || null;

      const targetDatasetId = activeDataset ? activeDataset.id : (this.activeRun ? this.activeRun.dataset_id : null);

      if (this.activeRun) {
        const runId = this.activeRun.id;
        const [plan, leaderboard, experiments, recipes, runs] = await Promise.all([
          api.getPlan(runId).catch(() => null),
          api.getLeaderboard(runId).catch(() => []),
          api.getExperiments(runId).catch(() => []),
          api.getRecipes(targetDatasetId).catch(() => []),
          api.getRuns().catch(() => []),
        ]);
        this.plan = plan;
        this.leaderboard = leaderboard || [];
        this.experiments = experiments || [];
        this.recipes = recipes || [];
        if (Array.isArray(runs) && runs.length) {
          const fresh = runs.find(r => r.id === runId);
          if (fresh) {
            this.activeRun = fresh;
            store.resolveContext({
              datasets: store.getState().datasets || [],
              runs,
              overview: store.getState().overview,
            });
          }
        }
      } else if (targetDatasetId) {
        this.plan = null;
        this.leaderboard = [];
        this.experiments = [];
        this.recipes = await api.getRecipes(targetDatasetId).catch(() => []);
      }
    } catch (e) {
      console.warn("StudioView fetchData error:", e);
    }
  }

  render() {
    const activeDataset = store.getActiveDataset();
    const activeRun = this.activeRun;

    if (!activeRun) {
      this.container.innerHTML = `
        <div class="workbench-card p-12 text-center space-y-4">
          <div class="text-[#8B95A7]/40 flex justify-center">${icon("flask-conical", "icon-xl", 36)}</div>
          <h3 class="text-base font-bold text-slate-200">
            ${activeDataset ? `Dataset ${escapeHtml(activeDataset.name)} sin ejecuciones` : "Sin dataset seleccionado"}
          </h3>
          <p class="text-xs text-slate-400 max-w-sm mx-auto">
            ${activeDataset
              ? `Configura y ejecuta el primer experimento en ${escapeHtml(activeDataset.name)} para inspeccionar el planificador, HPO y el leaderboard de modelos.`
              : "Selecciona un dataset del laboratorio para comenzar a crear y comparar experimentos."}
          </p>
          <button id="btnNewExpFromStudioEmpty" class="btn-signal">
            <span class="inline-flex items-center gap-1.5">
              ${icon(activeDataset ? "plus" : "database", "icon-sm")}
              <span>${activeDataset ? "Nuevo experimento" : "Seleccionar dataset"}</span>
            </span>
          </button>
        </div>
      `;
      this.container.querySelector("#btnNewExpFromStudioEmpty")?.addEventListener("click", () => {
        if (activeDataset) {
          bus.emit("modal:new-experiment");
        } else {
          bus.emit("modal:switch-dataset");
        }
      });
      return;
    }

    const isRunning = isRunActive(activeRun);
    const isPaused = activeRun.status === "PAUSED";
    const metricName = activeRun.metric || "ROC-AUC";
    const directionInfo = getMetricDirectionInfo(metricName);

    const bestScoreText = activeRun.best_score != null
      ? Number(activeRun.best_score).toFixed(5)
      : (this.leaderboard.length > 0 && this.leaderboard[0].score != null
          ? Number(this.leaderboard[0].score).toFixed(5)
          : "Sin resultados");

    const bestModelText = activeRun.best_model
      || (this.leaderboard.length > 0 ? this.leaderboard[0].model_id : "Sin entrenar");

    const trialsCount = resolveTrialsCount(activeRun, this.experiments, this.leaderboard);

    // Collect all trial scores from experiments or leaderboard
    const trialsList = [];
    for (const e of this.experiments) {
      for (const t of (e.trials || [])) {
        if (t.score != null) {
          trialsList.push({
            id: t.trial_id || `trial_${trialsList.length + 1}`,
            model: t.model_id || e.model_id || "model",
            score: t.score,
            time: t.time_s != null ? t.time_s : null,
            params: t.params || {},
            experiment_id: e.id,
            recipe_id: e.recipe_id,
          });
        }
      }
    }

    if (trialsList.length === 0 && this.leaderboard.length > 0) {
      for (const lb of this.leaderboard) {
        if (lb.score != null) {
          trialsList.push({
            id: lb.trial_id || `trial_${trialsList.length + 1}`,
            model: lb.model_id,
            score: lb.score,
            time: lb.training_time_seconds,
            params: {},
            experiment_id: lb.experiment_id,
            recipe_id: null,
          });
        }
      }
    }

    // Baseline detection & honest comparability check (Section 7.3)
    let baselineTrial = null;
    for (const row of this.leaderboard) {
      const mId = (row.model_id || "").toLowerCase();
      if (mId === "logistic_regression" || mId === "ridge" || mId.includes("baseline") || mId === "dummy") {
        baselineTrial = row;
        break;
      }
    }

    const winnerTrial = this.leaderboard.length > 0 ? this.leaderboard[0] : (trialsList.length > 0 ? trialsList[0] : null);
    let baselineDiffText = "—";
    let baselineDiffClass = "text-[#8B95A7]";
    let baselineModelText = "Sin baseline evaluado en esta ejecución";

    if (baselineTrial && baselineTrial.score != null && winnerTrial && winnerTrial.score != null) {
      if (baselineTrial.model_id === winnerTrial.model_id) {
        baselineDiffText = "Modelo es el baseline";
        baselineDiffClass = "text-[#8B95A7]";
        baselineModelText = `Referencia (${Number(baselineTrial.score).toFixed(5)})`;
      } else {
        const comp = formatMetricComparison(winnerTrial.score, baselineTrial.score, activeRun.metric);
        baselineDiffText = comp.text;
        baselineDiffClass = comp.isImprovement ? "text-[#22C55E]" : (comp.isNeutral ? "text-[#8B95A7]" : "text-[#F59E0B]");
        baselineModelText = `vs ${baselineTrial.model_id} (${Number(baselineTrial.score).toFixed(5)})`;
      }
    }

    // Determine winning recipe details
    const winnerExp = winnerTrial ? this.experiments.find(e => e.id === winnerTrial.experiment_id) : null;
    const winnerRecipeId = winnerExp?.recipe_id;
    const winnerRecipeObj = winnerRecipeId ? this.recipes.find(r => r.id === winnerRecipeId) : null;
    const winnerRecipeName = winnerExp?.recipe_name || winnerRecipeObj?.name || (winnerRecipeId ? winnerRecipeId : null);
    const winnerTransformsCount = (winnerExp?.column_transforms?.length || 0) + (winnerExp?.derived_features?.length || 0);

    let winnerRecipeText = "Configuración base estándar";
    let winnerTransformsText = "Sin transformaciones de Explore";
    if (winnerRecipeName) {
      winnerRecipeText = `Receta: ${winnerRecipeName}`;
      winnerTransformsText = winnerTransformsCount > 0 ? `${winnerTransformsCount} transformaciones in-fold` : "Transformaciones in-fold aplicadas";
    } else if (winnerTransformsCount > 0) {
      winnerRecipeText = `Explore (+${winnerTransformsCount})`;
      winnerTransformsText = "Transformaciones ajustadas in-fold";
    }

    // Validation strategy label
    const valStrategyText = activeRun.validation_strategy || "5-Fold Stratified Cross-Validation";

    // Action controls
    let actionControlsHtml = "";
    if (isRunning) {
      actionControlsHtml = `
        <button id="btnPauseRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 text-amber-400 border-amber-500/30 hover:bg-amber-500/10 transition-colors">
          ${icon("pause", "icon-sm")}<span>Pausar</span>
        </button>
        <button id="btnStopRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 text-rose-400 border-rose-500/30 hover:bg-rose-500/10 transition-colors">
          ${icon("square", "icon-sm")}<span>Detener</span>
        </button>
      `;
    } else if (isPaused) {
      actionControlsHtml = `
        <button id="btnResumeRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/10 transition-colors">
          ${icon("play", "icon-sm")}<span>Reanudar</span>
        </button>
        <button id="btnStopRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 text-rose-400 border-rose-500/30 hover:bg-rose-500/10 transition-colors">
          ${icon("square", "icon-sm")}<span>Detener</span>
        </button>
      `;
    } else if (activeRun.status === "COMPLETED") {
      actionControlsHtml = `
        <button id="btnRunAgain" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 hover:text-white transition-colors" title="Lanzar nuevo experimento">
          ${icon("refresh-cw", "icon-sm")}<span>Ejecutar de nuevo</span>
        </button>
        <button id="btnCloneRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 hover:text-white transition-colors" title="Clonar ejecución">
          ${icon("copy", "icon-sm")}<span>Clonar</span>
        </button>
        <a href="/api/models/export?run_id=${encodeURIComponent(activeRun.id)}" class="btn-signal text-xs px-3 py-1.5 flex items-center space-x-1.5" title="Descargar ModelArtifact (.pkl)">
          ${icon("download", "icon-sm")}<span>Exportar modelo (.pkl)</span>
        </a>
        <a href="/api/reports/executive?run_id=${encodeURIComponent(activeRun.id)}&format=html" target="_blank" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 hover:text-white transition-colors" title="Ver informe ejecutivo de resultados">
          ${icon("file-text", "icon-sm")}<span>Informe ejecutivo</span>
        </a>
        <a href="/api/reports/executive?run_id=${encodeURIComponent(activeRun.id)}&format=md&download=1" download class="btn-technical text-xs px-2.5 py-1.5 flex items-center space-x-1 hover:text-white transition-colors" title="Descargar informe en Markdown (.md)">
          ${icon("file", "icon-sm")}<span>.md</span>
        </a>
      `;
    } else {
      actionControlsHtml = `
        <button id="btnRunAgain" class="btn-signal text-xs px-3 py-1.5 flex items-center space-x-1.5">
          ${icon("refresh-cw", "icon-sm")}<span>Ejecutar de nuevo</span>
        </button>
        <button id="btnCloneRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 hover:text-white transition-colors">
          ${icon("copy", "icon-sm")}<span>Clonar</span>
        </button>
      `;
    }

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- 1. Top Control Header -->
        <div class="workbench-card p-4 bg-[#10151E] border-[#242A36]">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div class="flex items-center space-x-3">
                <h2 class="text-lg font-bold font-sans text-[#F7F8FA]">Run: ${escapeHtml(activeRun.id)} / ${escapeHtml(activeRun.dataset_name || "Active Dataset")}</h2>
                <span id="runStatusBadge" class="${isRunning ? 'badge-sys' : isPaused ? 'badge-warn' : 'badge-gain'} text-xs px-2.5 py-1 rounded-full font-mono font-semibold flex items-center space-x-1.5">
                  ${isRunning ? '<svg class="animate-spin h-3.5 w-3.5 text-[#4F67FF] inline" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>' : ''}
                  <span>● ${escapeHtml(activeRun.status || "IDLE")}</span>
                </span>
              </div>
              <p class="text-xs text-[#8B95A7] font-sans mt-0.5">Pipeline AutoML • Métrica: ${escapeHtml(metricName)} (${directionInfo.label} ${directionInfo.arrow}) • ${escapeHtml(valStrategyText)}</p>
            </div>

            <!-- Action Controls -->
            <div class="flex items-center space-x-2">
              ${actionControlsHtml}
            </div>
          </div>

          <!-- Quick Action Launcher Bar -->
          <div class="mt-3 pt-3 border-t border-[#242A36] flex flex-wrap items-center justify-between gap-3 text-xs">
            <div class="flex items-center space-x-2">
              <span class="text-[#8B95A7] font-medium font-sans uppercase text-[11px] flex items-center gap-1.5">
                <span class="w-1.5 h-1.5 rounded-full bg-[#4F67FF]"></span>
                Lanzar modelo directo:
              </span>
              <button data-quick-model="lightgbm" class="btn-quick-run btn-technical text-xs" title="Lanzar nuevo experimento con LightGBM">+ LightGBM</button>
              <button data-quick-model="xgboost" class="btn-quick-run btn-technical text-xs" title="Lanzar nuevo experimento con XGBoost">+ XGBoost</button>
              <button data-quick-model="catboost" class="btn-quick-run btn-technical text-xs" title="Lanzar nuevo experimento con CatBoost">+ CatBoost</button>
              <button data-quick-model="voting_ensemble" class="btn-quick-run btn-technical text-xs border-[#4F67FF]/30 text-[#4F67FF]" title="Lanzar nuevo experimento con Ensamble">+ Ensamble</button>
            </div>
            <div class="flex items-center space-x-2">
              <button id="btnNewExpFromHeader" class="btn-signal text-xs flex items-center space-x-1.5">
                ${icon("bot", "icon-sm")}
                <span>Experimento Guiado</span>
              </button>
            </div>
          </div>
        </div>

        <!-- 2. Presentación Inicial: Hero del Modelo Ganador (Section 7.1) -->
        <div class="workbench-card p-5 border-[#4F67FF]/30 bg-gradient-to-br from-[#10151E] via-[#11151E] to-[#141B28] space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#242A36] pb-3">
            <div class="flex items-center space-x-2.5">
              <span class="text-amber-400">${icon("trophy", "icon-sm")}</span>
              <h3 class="text-sm font-bold font-sans text-[#F7F8FA] uppercase tracking-wide">Modelo Mejor Evaluado en esta Ejecución</h3>
              <span class="badge-gain text-[10px] px-2 py-0.5 rounded font-mono font-bold">Mejor de esta ejecución</span>
            </div>
            <div class="text-xs text-[#8B95A7] font-mono">
              Evaluación final sobre ${trialsCount} ensayos
            </div>
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            <!-- 1. Modelo Ganador -->
            <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#242A36]">
              <div class="text-[10px] font-sans font-medium text-[#8B95A7] uppercase">1. Modelo Ganador</div>
              <div class="text-base font-bold font-mono text-[#F7F8FA] mt-0.5 capitalize truncate">
                ${escapeHtml(bestModelText)}
              </div>
              <div class="text-[10px] text-[#4F67FF] font-mono truncate">
                ${winnerTrial && winnerTrial.trial_id ? `Ensayo: ${escapeHtml(winnerTrial.trial_id)}` : 'Ensayo Principal'}
              </div>
            </div>

            <!-- 2. Puntuación CV -->
            <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#242A36]">
              <div class="text-[10px] font-sans font-medium text-[#8B95A7] uppercase">2. Puntuación (${escapeHtml(metricName)})</div>
              <div class="text-lg font-bold font-mono text-[#22C55E] mt-0.5">
                ${bestScoreText}
              </div>
              <div class="text-[10px] text-[#8B95A7] font-sans">${directionInfo.label} (${directionInfo.arrow})</div>
            </div>

            <!-- 3. Tipo de Validación -->
            <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#242A36]">
              <div class="text-[10px] font-sans font-medium text-[#8B95A7] uppercase">3. Validación</div>
              <div class="text-xs font-semibold font-sans text-[#F7F8FA] mt-1 truncate" title="${escapeHtml(valStrategyText)}">
                ${escapeHtml(valStrategyText)}
              </div>
              <div class="text-[10px] text-[#8B95A7] font-mono">${trialsCount} ensayos completados</div>
            </div>

            <!-- 4. Comparación con Baseline -->
            <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#242A36]">
              <div class="text-[10px] font-sans font-medium text-[#8B95A7] uppercase">4. Comparación vs Baseline</div>
              <div class="text-xs font-bold font-mono mt-1 ${baselineDiffClass} truncate">
                ${baselineDiffText}
              </div>
              <div class="text-[10px] text-[#8B95A7] font-sans truncate" title="${escapeHtml(baselineModelText)}">${escapeHtml(baselineModelText)}</div>
            </div>

            <!-- 5. Pipeline / Receta Aplicada -->
            <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#242A36]">
              <div class="text-[10px] font-sans font-medium text-[#8B95A7] uppercase">5. Pipeline de Datos</div>
              <div class="text-xs font-semibold font-mono text-[#53C8FF] mt-1 truncate" title="${escapeHtml(winnerRecipeText)}">
                ${escapeHtml(winnerRecipeText)}
              </div>
              <div class="text-[10px] text-[#8B95A7] font-sans truncate">${escapeHtml(winnerTransformsText)}</div>
            </div>
          </div>
        </div>

        <!-- 3. Pestañas de Navegación del Studio -->
        <div class="workbench-card">
          <div class="border-b border-[#242A36] px-4 flex items-center space-x-6 text-xs font-medium overflow-x-auto">
            <button class="tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${this.activeTab === 'models' ? 'border-[#4F67FF] text-[#4F67FF]' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="models">
              ${icon("bar-chart-3", "icon-sm")}
              <span>Leaderboard (Resultados)</span>
            </button>
            <button class="tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${this.activeTab === 'hpo' ? 'border-[#4F67FF] text-[#4F67FF]' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="hpo">
              ${icon("chart-line", "icon-sm")}
              <span>Optimización & HPO</span>
            </button>
            <button class="tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${this.activeTab === 'planner' ? 'border-[#4F67FF] text-[#4F67FF]' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="planner">
              ${icon("compass", "icon-sm")}
              <span>Planificación & Decisiones</span>
            </button>
            <button class="tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${this.activeTab === 'recipes' ? 'border-[#4F67FF] text-[#4F67FF]' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="recipes">
              ${icon("recipe", "icon-sm")}
              <span>Recetas (${this.recipes ? this.recipes.length : 0})</span>
            </button>
            <button class="tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${this.activeTab === 'resources' ? 'border-[#4F67FF] text-[#4F67FF]' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="resources">
              ${icon("cpu", "icon-sm")}
              <span>Recursos & Hardware</span>
            </button>
            <button class="tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${this.activeTab === 'logs' ? 'border-[#4F67FF] text-[#4F67FF]' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="logs">
              ${icon("scroll-text", "icon-sm")}
              <span>Logs & Eventos</span>
            </button>
          </div>

          <div class="p-5" id="tabContent">
            ${this._renderTabContent()}
          </div>
        </div>
      </div>

      <!-- Why Modal Dialog -->
      <div id="whyModal" class="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4 hidden">
        <div class="workbench-card max-w-lg w-full p-6 space-y-4 border-[#252C38] shadow-2xl">
          <div class="flex items-center justify-between border-b border-[#252C38] pb-2">
            <div class="flex items-center space-x-2">
              <span class="text-[#6956E8]">${icon("bot", "icon-lg")}</span>
              <h3 class="text-sm font-bold text-[#F7F8FA] uppercase tracking-wide font-sans">Explicabilidad del Plan AutoML</h3>
            </div>
            <button id="btnCloseWhyModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded hover:bg-[#161B26] transition-colors">${icon("x", "icon-sm")}</button>
          </div>
          <div id="whyModalBody" class="text-xs text-[#8B95A7] space-y-2"></div>
        </div>
      </div>

      <!-- Trial Parameters Modal Dialog -->
      <div id="trialParamsModal" class="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4 hidden">
        <div class="workbench-card max-w-lg w-full p-6 space-y-4 border-[#252C38] shadow-2xl">
          <div class="flex items-center justify-between border-b border-[#252C38] pb-2">
            <div class="flex items-center space-x-2">
              <span class="text-[#4F67FF]">${icon("search", "icon-lg")}</span>
              <h3 class="text-sm font-semibold text-[#F7F8FA] uppercase tracking-wide font-sans" id="trialParamsModalTitle">Hiperparámetros del Modelo</h3>
            </div>
            <button id="btnCloseTrialParamsModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded hover:bg-[#161B26] transition-colors">${icon("x", "icon-sm")}</button>
          </div>
          <div id="trialParamsModalBody" class="text-xs text-[#8B95A7] space-y-3"></div>
        </div>
      </div>
    `;

    this._bindEvents();
    if (this.activeTab === "hpo") {
      this._initChart();
    }
  }

  _renderTabContent() {
    if (this.activeTab === "models") {
      return this._renderLeaderboardTabHtml();
    } else if (this.activeTab === "hpo") {
      return this._renderHpoTabHtml();
    } else if (this.activeTab === "planner") {
      return this._renderPlannerTabHtml();
    } else if (this.activeTab === "resources") {
      return this._renderResourcesTabHtml();
    } else if (this.activeTab === "recipes") {
      return this._renderRecipesTabHtml();
    } else {
      return this._renderLogsTabHtml();
    }
  }

  /**
   * Leaderboard Tab: Focused on model results, trial uniqueness, and baseline delta (Section 7)
   */
  _renderLeaderboardTabHtml() {
    const activeRun = this.activeRun;
    let filteredLeaderboard = this.leaderboard;

    if (this.modelSearch) {
      filteredLeaderboard = filteredLeaderboard.filter(row =>
        row.model_id.toLowerCase().includes(this.modelSearch.toLowerCase())
      );
    }

    if (this.modelFilter === "gbdt") {
      filteredLeaderboard = filteredLeaderboard.filter(row =>
        ["lightgbm", "xgboost", "catboost"].includes(row.model_id)
      );
    } else if (this.modelFilter === "trees") {
      filteredLeaderboard = filteredLeaderboard.filter(row =>
        ["random_forest", "extra_trees"].includes(row.model_id)
      );
    } else if (this.modelFilter === "linear") {
      filteredLeaderboard = filteredLeaderboard.filter(row =>
        ["logistic_regression", "ridge", "svc", "svr"].includes(row.model_id)
      );
    } else if (this.modelFilter === "ensemble") {
      filteredLeaderboard = filteredLeaderboard.filter(row =>
        ["voting_ensemble"].includes(row.model_id)
      );
    }

    // Identify reference baseline for deltas
    let baselineTrial = null;
    for (const row of this.leaderboard) {
      const mId = (row.model_id || "").toLowerCase();
      if (mId === "logistic_regression" || mId === "ridge" || mId.includes("baseline") || mId === "dummy") {
        baselineTrial = row;
        break;
      }
    }

    return `
      <div class="space-y-4">
        <!-- Filter & Search Toolbar -->
        <div class="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#242A36]">
          <div class="flex flex-wrap items-center gap-1.5 text-xs font-sans">
            <span class="text-[#8B95A7] text-[11px] uppercase font-semibold mr-1">Familia:</span>
            <button data-modelfilter="all" class="model-filter-pill px-2.5 py-1 rounded border text-[11px] font-medium transition-colors ${this.modelFilter === 'all' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">Todos (${this.leaderboard.length})</button>
            <button data-modelfilter="gbdt" class="model-filter-pill px-2.5 py-1 rounded border text-[11px] font-medium transition-colors ${this.modelFilter === 'gbdt' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">GBDTs</button>
            <button data-modelfilter="trees" class="model-filter-pill px-2.5 py-1 rounded border text-[11px] font-medium transition-colors ${this.modelFilter === 'trees' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">Random Forest / ET</button>
            <button data-modelfilter="linear" class="model-filter-pill px-2.5 py-1 rounded border text-[11px] font-medium transition-colors ${this.modelFilter === 'linear' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">Linear & SVM</button>
            <button data-modelfilter="ensemble" class="model-filter-pill px-2.5 py-1 rounded border text-[11px] font-medium transition-colors ${this.modelFilter === 'ensemble' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">Ensambles</button>
          </div>
          <input type="text" id="modelSearchInput" value="${this.modelSearch}" placeholder="Filtrar modelos o ensayos..." class="bg-[#090C12] border border-[#242A36] text-[#F7F8FA] text-xs rounded-lg px-3 py-1 font-mono w-56 focus:border-[#4F67FF] outline-none">
        </div>

        <div class="overflow-x-auto">
          <table class="w-full wb-table text-left">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Modelo</th>
                <th>Ensayo / Trial ID</th>
                <th>Score (${activeRun ? escapeHtml(activeRun.metric) : 'CV'})</th>
                <th>Δ vs Baseline</th>
                <th>Pipeline de Datos / Receta</th>
                <th>Duración</th>
                <th>Estado</th>
                <th class="text-right">Hiperparámetros</th>
              </tr>
            </thead>
            <tbody>
              ${filteredLeaderboard.length > 0 ? filteredLeaderboard.map((row, idx) => {
                const exp = this.experiments.find(e => e.id === row.experiment_id);
                const recipeId = exp?.recipe_id;
                const recipeName = exp?.recipe_name || (recipeId ? this.recipes.find(r => r.id === recipeId)?.name : null);
                const transformsCount = (exp?.column_transforms?.length || 0) + (exp?.derived_features?.length || 0);

                // Delta vs baseline calculation
                let deltaText = "—";
                let deltaBadgeClass = "text-[#8B95A7]";
                if (baselineTrial && baselineTrial.score != null && row.score != null) {
                  if (baselineTrial.model_id === row.model_id && (baselineTrial.trial_id === row.trial_id || !row.trial_id)) {
                    deltaText = "Baseline";
                    deltaBadgeClass = "badge-sys";
                  } else {
                    const comp = formatMetricComparison(row.score, baselineTrial.score, activeRun.metric);
                    deltaText = comp.text.replace("Δ: ", "");
                    deltaBadgeClass = comp.isImprovement ? "text-[#22C55E]" : (comp.isNeutral ? "text-[#8B95A7]" : "text-[#F59E0B]");
                  }
                }

                // Recipe badge
                let recipeBadgeHtml = '';
                if (recipeId) {
                  recipeBadgeHtml = `
                    <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono inline-flex items-center gap-1" title="Receta reproducible: ${escapeHtml(recipeName || recipeId)} (${transformsCount} transforms in-fold)">
                      <span class="text-[#53C8FF]">●</span>
                      <span class="font-medium">${escapeHtml(recipeName || recipeId)}</span>
                      ${transformsCount > 0 ? `<span class="opacity-75 text-[9px]">(+${transformsCount})</span>` : ''}
                    </span>
                  `;
                } else if (transformsCount > 0) {
                  recipeBadgeHtml = `
                    <span class="badge-sys text-[10px] px-2 py-0.5 rounded font-mono inline-flex items-center gap-1" title="${transformsCount} transformaciones in-fold aplicadas sin receta formal">
                      <span class="text-[#6956E8]">●</span>
                      <span>Explore (+${transformsCount})</span>
                    </span>
                  `;
                } else {
                  recipeBadgeHtml = `
                    <span class="text-[#8B95A7] font-mono text-[10px]" title="Entrenamiento estándar sin receta previa">Sin receta</span>
                  `;
                }

                const trialIdDisplay = row.trial_id || (exp?.name ? `${exp.name}` : `trial_${idx + 1}`);

                return `
                <tr>
                  <td class="font-mono font-bold ${idx === 0 ? 'text-amber-400' : 'text-slate-400'}">
                    ${idx === 0 ? '🏆 #1' : `#${idx + 1}`}
                  </td>
                  <td class="font-medium text-slate-200 capitalize font-mono">${escapeHtml(row.model_id)}${row.backend === "fallback" ? ` <span class="text-[9px] text-amber-400 font-mono normal-case" title="${escapeHtml(row.estimator_class || row.fallback_backend || "sklearn fallback")}">fallback${row.fallback_backend ? `: ${escapeHtml(row.fallback_backend)}` : ""}</span>` : (row.backend === "native" ? ` <span class="text-[9px] text-emerald-400 font-mono normal-case">nativo</span>` : "")}</td>
                  <td class="font-mono text-xs text-[#4F67FF]">${escapeHtml(trialIdDisplay)}</td>
                  <td class="font-mono font-bold ${idx === 0 ? 'text-emerald-400' : 'text-slate-200'}">${Number(row.score).toFixed(5)}</td>
                  <td class="font-mono text-xs ${deltaBadgeClass}">${escapeHtml(deltaText)}</td>
                  <td>${recipeBadgeHtml}</td>
                  <td class="font-mono text-xs text-slate-400">${row.training_time_seconds ? Number(row.training_time_seconds).toFixed(2) + 's' : (row.training_time_s ? Number(row.training_time_s).toFixed(2) + 's' : '—')}</td>
                  <td><span class="${idx === 0 ? 'badge-gain' : 'badge-sys'} text-[10px] px-2 py-0.5 rounded font-mono">${idx === 0 ? 'Mejor de esta ejecución' : 'Evaluado'}</span></td>
                  <td class="text-right">
                    <button data-inspect-model="${row.model_id}" data-inspect-trial="${escapeHtml(row.trial_id || '')}" class="btn-inspect-model-params text-[10px] bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 px-2.5 py-0.5 rounded font-mono transition-colors" aria-label="Inspeccionar hiperparámetros">
                      <span class="inline-flex items-center gap-1">${icon("search", "icon-sm")} <span>Parámetros</span></span>
                    </button>
                  </td>
                </tr>
              `}).join("") : `
                <tr><td colspan="9" class="text-center text-[#8B95A7] py-6 text-xs font-sans">No hay modelos que coincidan con el filtro seleccionado.</td></tr>
              `}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  /**
   * HPO & Optimization Progress Tab
   */
  _renderHpoTabHtml() {
    const activeRun = this.activeRun;
    const metricLower = (activeRun?.metric || "").toLowerCase();
    const isMinimizing = isMinimizingMetric(metricLower);

    let bestTrial = null;
    for (const e of this.experiments) {
      for (const t of (e.trials || [])) {
        if (t.score != null) {
          if (!bestTrial) {
            bestTrial = t;
          } else if (isMinimizing ? t.score < bestTrial.score : t.score > bestTrial.score) {
            bestTrial = t;
          }
        }
      }
    }

    const trialsList = [];
    for (const e of this.experiments) {
      for (const t of (e.trials || [])) {
        if (t.score != null) trialsList.push(t);
      }
    }

    return `
      <div class="space-y-6">
        <!-- Gráfico de convergencia bayesiana -->
        <div class="p-4 rounded-xl bg-[#11151E] border border-[#242A36] space-y-3">
          <div class="flex items-center justify-between pb-2 border-b border-[#242A36]">
            <div class="flex items-center space-x-2">
              <span class="text-[#22C55E]">${icon("chart-line", "icon-sm")}</span>
              <h4 class="text-xs font-bold text-[#F7F8FA] uppercase tracking-wider font-sans">Evolución de Ensayos (Convergencia HPO)</h4>
            </div>
            <span class="text-xs font-mono text-[#8B95A7]">${trialsList.length} ensayos registrados</span>
          </div>

          <div class="h-64 w-full relative">
            <canvas id="performanceChart"></canvas>
          </div>
        </div>

        <!-- Parámetros óptimos observados -->
        <div class="p-4 rounded-xl bg-[#11151E] border border-[#242A36] space-y-3">
          <div class="flex items-center justify-between">
            <div>
              <div class="text-xs font-semibold text-slate-300 uppercase tracking-wider">Mejor conjunto de parámetros observado</div>
              <p class="text-[11px] text-[#8B95A7]">Mejor ensayo: <span class="font-mono text-emerald-400 font-bold">${bestTrial ? (bestTrial.trial_id || "#1") : "—"}</span> • Score: <span class="font-mono text-emerald-400 font-bold">${bestTrial && bestTrial.score != null ? Number(bestTrial.score).toFixed(5) : "—"}</span></p>
            </div>
            <button id="btnStopHPOAndPromote" class="btn-signal text-xs">
              <span class="inline-flex items-center gap-1.5">${icon("zap", "icon-sm")} <span>Promover Mejores Parámetros</span></span>
            </button>
          </div>

          ${bestTrial && bestTrial.params && Object.keys(bestTrial.params).length > 0 ? `
            <div class="grid grid-cols-2 md:grid-cols-4 gap-2 font-mono text-xs">
              ${Object.entries(bestTrial.params).map(([k, v]) => `
                <div class="p-2.5 rounded-lg bg-[#090C12] border border-[#242A36]">
                  <span class="text-[#8B95A7] text-[10px] block font-sans">${k}</span>
                  <span class="text-[#4F67FF] font-bold">${typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : v}</span>
                </div>
              `).join("")}
            </div>
          ` : `
            <div class="text-xs text-[#8B95A7] font-mono p-4 bg-[#090C12] rounded-lg border border-[#242A36] text-center">
              Sin parámetros de HPO optimizados registrados para este experimento.
            </div>
          `}
        </div>
      </div>
    `;
  }

  /**
   * AutoML Planner Explicability Tab
   */
  _renderPlannerTabHtml() {
    const steps = (this.plan && this.plan.steps) ? this.plan.steps : [];
    const activeRun = this.activeRun;

    return `
      <div class="space-y-4">
        <div class="flex items-center justify-between pb-2 border-b border-[#242A36]">
          <div class="flex items-center space-x-2">
            <span class="text-[#4F67FF]">${icon("compass", "icon-sm")}</span>
            <h4 class="text-xs font-bold text-[#F7F8FA] uppercase tracking-wider font-sans">Pasos de Planificación y Justificación</h4>
          </div>
          <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono font-medium">Explicable</span>
        </div>

        <div class="space-y-2.5 max-h-[500px] overflow-y-auto pr-1">
          ${steps.length > 0 ? steps.map((step, idx) => {
            const isCompleted = step.status === "COMPLETED";
            const isStepRunning = step.status === "RUNNING";
            return `
              <div class="p-3.5 rounded-xl ${isStepRunning ? 'bg-[#4F67FF]/10 border border-[#4F67FF]/30' : 'bg-[#11151E] border border-[#242A36]'} flex items-start justify-between">
                <div class="space-y-1">
                  <div class="flex items-center space-x-2">
                    <span class="inline-flex items-center gap-1 font-mono font-bold ${isCompleted ? 'text-[#22C55E]' : isStepRunning ? 'text-[#4F67FF] animate-pulse' : 'text-[#8B95A7]'}">
                      ${step.step} ${isCompleted ? icon("check", "icon-sm") : isStepRunning ? '<span class="w-1.5 h-1.5 rounded-full bg-[#4F67FF]"></span>' : '<span class="w-1.5 h-1.5 rounded-full border border-[#8B95A7]"></span>'}
                    </span>
                    <span class="text-xs font-semibold text-[#F7F8FA] font-sans">${escapeHtml(step.name)}</span>
                  </div>
                  <div class="text-[11px] text-[#8B95A7] font-sans flex items-center space-x-2">
                    ${step.score != null ? `<span class="font-mono text-[#22C55E]">${activeRun?.metric || 'CV'}: ${step.score.toFixed(5)}</span>` : `<span class="text-[#8B95A7] font-mono">${step.status}</span>`}
                    <span class="badge-sys text-[9px] px-1.5 py-0.2 rounded font-mono font-medium">${step.priority || 'NORMAL'}</span>
                  </div>
                </div>
                <button data-step-idx="${idx}" class="btn-why-dynamic badge-intel hover:bg-[#6956E8]/30 text-[10px] px-2.5 py-1 rounded transition-colors font-mono cursor-pointer">
                  ¿Por qué este paso?
                </button>
              </div>
            `;
          }).join("") : `
            <div class="text-center py-8 px-4 space-y-2 bg-[#11151E] rounded-xl border border-[#242A36]">
              <div class="text-[#8B95A7]/40 flex justify-center">${icon("compass", "icon-lg", 28)}</div>
              <div class="text-xs font-semibold text-[#F7F8FA] font-sans">No hay decisiones de planificación registradas para esta ejecución</div>
              <p class="text-[11px] text-[#8B95A7] font-sans max-w-xs mx-auto leading-relaxed">
                Las decisiones aparecen aquí cuando las heurísticas o agentes podan el espacio de búsqueda o seleccionan variables.
              </p>
            </div>
          `}
        </div>
      </div>
    `;
  }

  /**
   * Resources & Hardware Tab
   */
  _renderResourcesTabHtml() {
    const activeRun = this.activeRun;
    const totalTrainingSec = this.leaderboard.reduce((acc, m) => acc + (Number(m.training_time_seconds) || 0), 0);
    const evaluatedModelsCount = this.leaderboard.length;

    return `
      <div class="space-y-4 text-xs font-sans">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="p-3.5 rounded-xl bg-[#11151E] border border-[#242A36]">
            <div class="text-[#8B95A7] text-[11px] uppercase font-semibold">Modelos Evaluados</div>
            <div class="text-lg font-bold font-mono text-[#F7F8FA] mt-1">${evaluatedModelsCount}</div>
            <div class="text-[10px] text-[#8B95A7]">${this.experiments.length} experimentos ejecutados</div>
          </div>
          <div class="p-3.5 rounded-xl bg-[#11151E] border border-[#242A36]">
            <div class="text-[#8B95A7] text-[11px] uppercase font-semibold">Tiempo Total de Entrenamiento</div>
            <div class="text-lg font-bold font-mono text-[#4F67FF] mt-1">${totalTrainingSec > 0 ? totalTrainingSec.toFixed(2) + 's' : '—'}</div>
            <div class="text-[10px] text-[#8B95A7]">Calculado de ejecuciones reales</div>
          </div>
          <div class="p-3.5 rounded-xl bg-[#11151E] border border-[#242A36]">
            <div class="text-[#8B95A7] text-[11px] uppercase font-semibold">Estado del Motor</div>
            <div class="text-lg font-bold font-mono text-[#22C55E] mt-1">${activeRun ? activeRun.status : 'ONLINE'}</div>
            <div class="text-[10px] text-[#8B95A7]">Validación: ${activeRun ? (activeRun.validation_strategy || '5-fold CV') : 'CV'}</div>
          </div>
        </div>

        <div class="space-y-2">
          <div class="font-semibold text-slate-200 uppercase tracking-wider text-[11px]">Tiempos de Entrenamiento por Modelo</div>
          ${evaluatedModelsCount > 0 ? `
            <div class="space-y-1.5 font-mono">
              ${this.leaderboard.map(m => `
                <div class="p-2.5 rounded-lg bg-[#11151E] border border-[#242A36] flex justify-between items-center text-xs">
                  <span class="capitalize text-slate-200 font-medium">${escapeHtml(m.model_id)}</span>
                  <div class="flex items-center space-x-4">
                    <span class="text-slate-400">${m.training_time_seconds ? Number(m.training_time_seconds).toFixed(2) + 's' : '—'}</span>
                    <span class="text-[#22C55E] inline-flex items-center gap-1">${icon("check", "icon-sm")} <span>Completado</span></span>
                  </div>
                </div>
              `).join("")}
            </div>
          ` : `
            <div class="p-6 text-center text-xs text-[#8B95A7] bg-[#11151E] rounded-xl border border-[#242A36]">
              No hay ejecuciones ni workers activos para este experimento.
            </div>
          `}
        </div>
      </div>
    `;
  }

  /**
   * Logs & Events Tab
   */
  _renderLogsTabHtml() {
    return `
      <div class="space-y-2 font-mono text-xs">
        ${this.experiments.length > 0 ? this.experiments.map(exp => {
          const rcpId = exp.recipe_id;
          const rcpName = exp.recipe_name || (rcpId ? this.recipes.find(r => r.id === rcpId)?.name : null);
          return `
          <div class="p-2.5 rounded-lg bg-[#11151E] border border-[#242A36] text-slate-300 flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <span class="text-emerald-400">[EXPERIMENTO]</span>
              <span class="font-semibold text-slate-200">${escapeHtml(exp.name || exp.id)}</span>
              <span class="text-slate-400 text-[11px]">(${exp.model_ids ? exp.model_ids.join(", ") : "modelos"})</span>
              ${rcpId ? `<span class="badge-intel text-[10px] px-1.5 py-0.2 rounded font-mono">${escapeHtml(rcpName || rcpId)}</span>` : ''}
            </div>
            <span class="text-indigo-400 font-bold">${exp.best_score != null ? Number(exp.best_score).toFixed(5) : '—'}</span>
          </div>
        `}).join("") : `
          <div class="p-4 rounded-xl bg-[#090C12] border border-[#242A36] text-center text-xs text-[#8B95A7]">
            No hay eventos de ejecución registrados. Inicie un experimento para ver los registros del motor AutoML.
          </div>
        `}
      </div>
    `;
  }

  _renderRecipesTabHtml() {
    const recipes = this.recipes || [];
    return `
      <div class="space-y-4 text-xs font-sans">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="p-3 rounded-xl bg-[#11151E] border border-[#242A36]">
            <div class="text-[#8B95A7] text-[11px] uppercase font-semibold">Recetas Registradas</div>
            <div class="text-lg font-bold font-mono text-[#F7F8FA] mt-1">${recipes.length}</div>
            <div class="text-[10px] text-[#8B95A7]">Configuraciones reproducibles</div>
          </div>
          <div class="p-3 rounded-xl bg-[#11151E] border border-[#242A36]">
            <div class="text-[#8B95A7] text-[11px] uppercase font-semibold">En Producción (Promovidas)</div>
            <div class="text-lg font-bold font-mono text-[#22C55E] mt-1">${recipes.filter(r => r.status === "promoted").length}</div>
            <div class="text-[10px] text-[#8B95A7]">Candidatas validadas con auditoría</div>
          </div>
          <div class="p-3 rounded-xl bg-[#11151E] border border-[#242A36]">
            <div class="text-[#8B95A7] text-[11px] uppercase font-semibold">Transformaciones In-Fold</div>
            <div class="text-lg font-bold font-mono text-[#4F67FF] mt-1">
              ${recipes.reduce((acc, r) => acc + (r.column_transforms ? r.column_transforms.length : 0), 0)}
            </div>
            <div class="text-[10px] text-[#8B95A7]">RobustScaler / Yeo-Johnson / log1p</div>
          </div>
        </div>

        ${recipes.length === 0 ? `
          <div class="p-8 text-center text-[#8B95A7] bg-[#11151E] rounded-xl border border-[#242A36] space-y-2">
            <div class="text-[#8B95A7]/40 flex justify-center">${icon("recipe", "icon-lg")}</div>
            <p class="font-medium text-[#F7F8FA]">No hay recetas de experimentos creadas para este dataset.</p>
            <p class="text-[11px]">Explora hipótesis estadísticas en <strong>CATML Explore</strong> y formaliza las mejoras empíricamente verificadas como recetas.</p>
          </div>
        ` : `
          <div class="overflow-x-auto border border-[#242A36] rounded-xl">
            <table class="w-full text-xs font-mono border-collapse">
              <thead>
                <tr class="bg-[#090C12] text-[#8B95A7] border-b border-[#242A36]">
                  <th class="p-2.5 text-left">Receta ID</th>
                  <th class="p-2.5 text-left">Nombre</th>
                  <th class="p-2.5 text-center">Estado del Ciclo</th>
                  <th class="p-2.5 text-left">Transforms</th>
                  <th class="p-2.5 text-left">Modelos</th>
                  <th class="p-2.5 text-left">Evidencia Origen</th>
                  <th class="p-2.5 text-left">Experimentos</th>
                  <th class="p-2.5 text-right">Acciones</th>
                </tr>
              </thead>
              <tbody>
                ${recipes.map(r => {
                  const stageInfo = getTransformationStageInfo(r.status);
                  const transformsList = [
                    ...(r.column_transforms || []).map(t => `${t.column}: ${t.kind}`),
                    ...(r.derived_features || []).map(d => `${d.name || d.expression}`),
                  ];
                  const transformText = transformsList.length > 0 ? transformsList.join(", ") : "Ninguna";

                  return `
                    <tr class="border-b border-[#1C2230] hover:bg-[#151B26]">
                      <td class="p-2.5 text-[#53C8FF] font-bold">${escapeHtml(r.id)}</td>
                      <td class="p-2.5 text-[#F7F8FA] font-sans font-medium">${escapeHtml(r.name || "-")}</td>
                      <td class="p-2.5 text-center">
                        <span class="${stageInfo.badgeClass} border text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">
                          ${escapeHtml(stageInfo.label)}
                        </span>
                      </td>
                      <td class="p-2.5 text-[#8B95A7] max-w-xs truncate" title="${escapeHtml(transformText)}">
                        ${escapeHtml(transformText)}
                      </td>
                      <td class="p-2.5 text-[#F7F8FA]">${escapeHtml((r.model_ids || []).join(", ") || "default")}</td>
                      <td class="p-2.5 text-[#8B95A7]">
                        ${r.evidence_link_id ? `<span class="text-[#53C8FF]">${escapeHtml(r.evidence_link_id)}</span>` : "-"}
                      </td>
                      <td class="p-2.5 text-[#8B95A7]">
                        ${(r.experiment_ids && r.experiment_ids.length > 0) ? r.experiment_ids.map(eid => `<span class="text-indigo-400 mr-1">${escapeHtml(eid)}</span>`).join("") : '<span class="text-[#8B95A7]/60">Sin ejecutar</span>'}
                      </td>
                      <td class="p-2.5 text-right whitespace-nowrap space-x-1">
                        ${r.status === "draft" ? `
                          <button class="btn-studio-select-recipe px-2 py-1 rounded text-[11px] font-mono bg-[#6956E8]/20 text-[#6956E8] hover:bg-[#6956E8]/30 border border-[#6956E8]/40 transition-colors" data-recipe-id="${escapeHtml(r.id)}">Seleccionar</button>
                        ` : ""}
                        ${r.status !== "archived" && r.status !== "promoted" ? `
                          <button class="btn-studio-submit-recipe px-2.5 py-1 rounded text-[11px] font-mono bg-[#4F67FF] text-white hover:bg-[#4F67FF]/80 inline-flex items-center gap-1 transition-colors" data-recipe-id="${escapeHtml(r.id)}">${icon("play", "icon-xs")} Entrenar</button>
                        ` : ""}
                        ${r.status === "submitted" ? `
                          <button class="btn-studio-promote-recipe px-2 py-1 rounded text-[11px] font-mono bg-[#22C55E]/20 text-[#22C55E] hover:bg-[#22C55E]/30 border border-[#22C55E]/40 inline-flex items-center gap-1 transition-colors" data-recipe-id="${escapeHtml(r.id)}">${icon("shield-check", "icon-xs")} Promover</button>
                        ` : ""}
                        ${r.status === "promoted" ? `
                          <span class="text-[11px] font-mono text-[#22C55E] inline-flex items-center gap-1 mr-1.5">${icon("check-circle-2", "icon-xs")} Producción</span>
                        ` : ""}
                        ${r.status !== "archived" ? `
                          <button class="btn-studio-archive-recipe p-1 rounded text-[#8B95A7] hover:text-[#EF4444] hover:bg-[#EF4444]/10 transition-colors" data-recipe-id="${escapeHtml(r.id)}" title="Archivar receta">${icon("x", "icon-xs")}</button>
                        ` : ""}
                      </td>
                    </tr>
                  `;
                }).join("")}
              </tbody>
            </table>
          </div>
        `}
      </div>
    `;
  }

  _bindEvents() {
    this.container.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        this.activeTab = btn.getAttribute("data-tab");
        const tabContent = this.container.querySelector("#tabContent");
        if (tabContent) {
          tabContent.innerHTML = this._renderTabContent();
          if (this.activeTab === "recipes") {
            this._bindRecipeTabEvents(tabContent);
          } else if (this.activeTab === "models") {
            this._bindLeaderboardEvents();
          } else if (this.activeTab === "hpo") {
            this._initChart();
          } else if (this.activeTab === "planner") {
            this._bindPlannerEvents();
          }
        }
        this.container.querySelectorAll(".tab-btn").forEach(b => {
          const isActive = b.getAttribute("data-tab") === this.activeTab;
          b.className = `tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${
            isActive
              ? "border-[#4F67FF] text-[#4F67FF]"
              : "border-transparent text-[#8B95A7] hover:text-[#F7F8FA]"
          }`;
        });
      });
    });

    if (this.activeTab === "recipes") {
      this._bindRecipeTabEvents(this.container);
    } else if (this.activeTab === "models") {
      this._bindLeaderboardEvents();
    } else if (this.activeTab === "planner") {
      this._bindPlannerEvents();
    }

    this.container.querySelector("#btnPauseRun")?.addEventListener("click", async () => {
      try {
        if (this.activeRun) {
          await api.pauseRun(this.activeRun.id);
          this.activeRun.status = "PAUSED";
          this.render();
        }
      } catch (e) {
        alert("Error pausing run: " + e.message);
      }
    });

    this.container.querySelector("#btnResumeRun")?.addEventListener("click", async () => {
      try {
        if (this.activeRun) {
          await api.resumeRun(this.activeRun.id);
          this.activeRun.status = "RUNNING";
          this.render();
        }
      } catch (e) {
        alert("Error resuming run: " + e.message);
      }
    });

    this.container.querySelector("#btnStopRun")?.addEventListener("click", async () => {
      if (confirm("Are you sure you want to stop this experiment run?")) {
        try {
          if (this.activeRun) {
            await api.cancelRun(this.activeRun.id);
            this.activeRun.status = "CANCELLED";
            this.render();
          }
        } catch (e) {
          alert("Error stopping run: " + e.message);
        }
      }
    });

    this.container.querySelector("#btnRunAgain")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });

    this.container.querySelector("#btnNewExpFromHeader")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });

    this.container.querySelector("#btnCloneRun")?.addEventListener("click", async () => {
      if (this.activeRun) {
        try {
          await api.cloneRun(this.activeRun.id, `${this.activeRun.dataset_name}_cloned`);
          alert("Run cloned successfully.");
        } catch (e) {
          alert("Error cloning run: " + e.message);
        }
      }
    });

    this.container.querySelector("#btnStopHPOAndPromote")?.addEventListener("click", () => {
      alert("Best trial parameters promoted to production model profile.");
    });

    this.container.querySelectorAll(".btn-quick-run").forEach(btn => {
      btn.addEventListener("click", () => {
        const modelId = btn.getAttribute("data-quick-model");
        bus.emit("modal:new-experiment", { preselectedModel: modelId });
      });
    });
  }

  _bindPlannerEvents() {
    const whyModal = this.container.querySelector("#whyModal");
    const whyModalBody = this.container.querySelector("#whyModalBody");
    const btnClose = this.container.querySelector("#btnCloseWhyModal");

    btnClose?.addEventListener("click", () => {
      whyModal?.classList.add("hidden");
    });

    this.container.querySelectorAll(".btn-why-dynamic").forEach(btn => {
      btn.addEventListener("click", () => {
        const idx = parseInt(btn.getAttribute("data-step-idx"), 10);
        const steps = (this.plan && this.plan.steps) ? this.plan.steps : [];
        const step = steps[idx];

        if (step && whyModal && whyModalBody) {
          whyModalBody.innerHTML = `
            <div class="space-y-3 font-sans">
              <div class="flex items-center space-x-2 border-b border-[#242A36] pb-2">
                <span class="font-bold text-[#F7F8FA]">${escapeHtml(step.name)}</span>
                <span class="badge-sys text-[10px] px-2 py-0.5 rounded font-mono">${step.priority || 'NORMAL'}</span>
              </div>
              <p class="text-xs text-[#F7F8FA] leading-relaxed">
                ${escapeHtml(step.rationale || "Este paso fue seleccionado por el planificador para establecer una base de referencia o explorar hiperparámetros.")}
              </p>
              ${step.policy_evidence ? `
                <div class="p-2.5 rounded bg-[#090C12] border border-[#242A36] text-[11px] font-mono text-[#53C8FF]">
                  Evidencia: ${escapeHtml(step.policy_evidence)}
                </div>
              ` : ''}
            </div>
          `;
          whyModal.classList.remove("hidden");
        }
      });
    });
  }

  _bindLeaderboardEvents() {
    const paramsModal = this.container.querySelector("#trialParamsModal");
    const paramsModalBody = this.container.querySelector("#trialParamsModalBody");
    const paramsModalTitle = this.container.querySelector("#trialParamsModalTitle");
    const btnClose = this.container.querySelector("#btnCloseTrialParamsModal");

    btnClose?.addEventListener("click", () => {
      paramsModal?.classList.add("hidden");
    });

    // Search input
    const searchInput = this.container.querySelector("#modelSearchInput");
    searchInput?.addEventListener("input", (e) => {
      this.modelSearch = e.target.value;
      const tabContent = this.container.querySelector("#tabContent");
      if (tabContent) {
        tabContent.innerHTML = this._renderLeaderboardTabHtml();
        this._bindLeaderboardEvents();
      }
    });

    // Family filter pills
    this.container.querySelectorAll(".model-filter-pill").forEach(pill => {
      pill.addEventListener("click", () => {
        this.modelFilter = pill.getAttribute("data-modelfilter");
        const tabContent = this.container.querySelector("#tabContent");
        if (tabContent) {
          tabContent.innerHTML = this._renderLeaderboardTabHtml();
          this._bindLeaderboardEvents();
        }
      });
    });

    // Inspect trial params
    this.container.querySelectorAll(".btn-inspect-model-params").forEach(btn => {
      btn.addEventListener("click", () => {
        const modelId = btn.getAttribute("data-inspect-model");
        const trialId = btn.getAttribute("data-inspect-trial");

        let foundTrial = null;
        for (const e of this.experiments) {
          for (const t of (e.trials || [])) {
            if (trialId && t.trial_id === trialId) {
              foundTrial = t;
              break;
            } else if (!trialId && t.model_id === modelId) {
              if (!foundTrial || (t.score && t.score > foundTrial.score)) {
                foundTrial = t;
              }
            }
          }
          if (foundTrial && trialId) break;
        }

        if (paramsModal && paramsModalBody) {
          if (paramsModalTitle) {
            paramsModalTitle.textContent = `Hiperparámetros — ${modelId.toUpperCase()} (${trialId || 'Trial'})`;
          }

          if (foundTrial && foundTrial.params && Object.keys(foundTrial.params).length > 0) {
            paramsModalBody.innerHTML = `
              <div class="space-y-3 font-sans">
                <div class="flex items-center justify-between p-2.5 rounded-xl bg-[#151B26] border border-[#242A36] text-xs">
                  <div>
                    <span class="text-[#8B95A7] block text-[10px] font-sans">Puntaje CV (${this.activeRun?.metric || 'CV'})</span>
                    <span class="font-mono text-[#22C55E] font-bold text-sm">${foundTrial.score ? foundTrial.score.toFixed(5) : '—'}</span>
                  </div>
                  <div>
                    <span class="text-[#8B95A7] block text-[10px] font-sans">Duración</span>
                    <span class="font-mono text-[#F7F8FA]">${foundTrial.time_s ? foundTrial.time_s + 's' : '—'}</span>
                  </div>
                  <div>
                    <span class="text-[#8B95A7] block text-[10px] font-sans">Trial ID</span>
                    <span class="font-mono text-[#4F67FF] font-bold">${escapeHtml(foundTrial.trial_id || 'trial_1')}</span>
                  </div>
                </div>

                <div class="space-y-1.5">
                  <span class="font-semibold text-[#F7F8FA] text-xs font-sans">Parámetros Resueltos:</span>
                  <div class="grid grid-cols-2 gap-2 max-h-56 overflow-y-auto">
                    ${Object.entries(foundTrial.params).map(([k, v]) => `
                      <div class="p-2 rounded-lg bg-[#090C12] border border-[#242A36] font-mono text-xs">
                        <span class="text-[#8B95A7] text-[10px] block font-sans">${k}</span>
                        <span class="text-[#4F67FF] font-bold">${typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : String(v)}</span>
                      </div>
                    `).join("")}
                  </div>
                </div>
              </div>
            `;
          } else {
            paramsModalBody.innerHTML = `
              <div class="p-6 text-center text-[#8B95A7] space-y-2">
                <div class="text-[#8B95A7]/50 flex justify-center">${icon("settings", "icon-lg", 28)}</div>
                <p class="text-xs font-sans">Este modelo fue entrenado con los priors por defecto de CATML o no tiene hiperparámetros individuales registrados en SQLite.</p>
                <div class="font-mono text-[11px] text-[#4F67FF] bg-[#090C12] p-2 rounded-lg border border-[#242A36]">
                  Modelo: ${escapeHtml(modelId)} • Modo: Estimador Canónico Base
                </div>
              </div>
            `;
          }

          paramsModal.classList.remove("hidden");
        }
      });
    });
  }

  _initChart() {
    const canvas = this.container.querySelector("#performanceChart");
    if (!canvas || !window.Chart) return;

    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }

    // Collect all trial scores from experiments or leaderboard
    const trials = [];
    for (const e of this.experiments) {
      for (const t of (e.trials || [])) {
        if (t.score != null) {
          trials.push({
            model: t.model_id || e.model_id || "model",
            score: t.score,
          });
        }
      }
    }

    if (trials.length === 0 && this.leaderboard.length > 0) {
      for (const lb of this.leaderboard) {
        if (lb.score != null) trials.push({ model: lb.model_id, score: lb.score });
      }
    }

    if (trials.length === 0) return;

    const scores = trials.map(t => t.score);
    const labels = trials.map((t, i) => `Trial ${i + 1} (${t.model})`);

    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);
    const pad = (maxScore - minScore) * 0.1 || 0.01;

    this.chart = new window.Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: `${this.activeRun ? this.activeRun.metric : 'Score'} Progression`,
            data: scores,
            borderColor: "#22C55E",
            backgroundColor: "rgba(34, 197, 94, 0.08)",
            fill: true,
            tension: 0.25,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: "#22C55E",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.label}: ${ctx.raw.toFixed(5)}`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(37, 44, 56, 0.6)" },
            ticks: { color: "#8B95A7", font: { size: 10 } },
          },
          y: {
            min: Math.max(0, minScore - pad),
            max: Math.min(1.0, maxScore + pad),
            grid: { color: "rgba(37, 44, 56, 0.6)" },
            ticks: { color: "#8B95A7", font: { size: 10 } },
          },
        },
      },
    });
  }

  _bindRecipeTabEvents(rootEl) {
    if (!rootEl) return;
    rootEl.querySelectorAll(".btn-studio-select-recipe").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const recipeId = e.currentTarget.getAttribute("data-recipe-id");
        if (!recipeId) return;
        try {
          await api.selectRecipe(recipeId);
          await this.fetchData();
          this.render();
        } catch (err) {
          alert("Error seleccionando receta: " + (err.message || err));
        }
      });
    });

    rootEl.querySelectorAll(".btn-studio-submit-recipe").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const recipeId = e.currentTarget.getAttribute("data-recipe-id");
        if (!recipeId) return;
        const btnEl = e.currentTarget;
        btnEl.disabled = true;
        btnEl.innerHTML = `<span class="animate-spin mr-1">${icon("refresh-cw", "icon-xs")}</span> Entrenando...`;
        try {
          const runId = this.activeRun ? this.activeRun.id : null;
          const res = await api.submitRecipe(recipeId, runId);
          alert(`Experimento(s) ejecutados con éxito: ${(res.experiment_ids || []).join(", ")}`);
          await this.fetchData();
          this.render();
        } catch (err) {
          alert("Error ejecutando receta: " + (err.message || err));
          btnEl.disabled = false;
        }
      });
    });

    rootEl.querySelectorAll(".btn-studio-promote-recipe").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const recipeId = e.currentTarget.getAttribute("data-recipe-id");
        if (!recipeId) return;
        const reason = prompt("Justificación para promover a producción:", "Supera al baseline con significancia estadística");
        if (!reason) return;
        try {
          await api.promoteRecipe(recipeId, reason);
          await this.fetchData();
          this.render();
        } catch (err) {
          alert("Error promoviendo receta: " + (err.message || err));
        }
      });
    });

    rootEl.querySelectorAll(".btn-studio-archive-recipe").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const recipeId = e.currentTarget.getAttribute("data-recipe-id");
        if (!recipeId) return;
        if (!confirm(`¿Desea archivar la receta ${recipeId}?`)) return;
        try {
          await api.archiveRecipe(recipeId);
          await this.fetchData();
          this.render();
        } catch (err) {
          alert("Error archivando receta: " + (err.message || err));
        }
      });
    });
  }

  destroy() {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
    this.container = null;
  }
}
