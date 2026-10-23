/**
 * KnowledgeView — V0.8 Meta-Learning & Memory Layer
 * Dataset fingerprinting, meta-learning similarity & warm start recommendations.
 */
import { store } from "../store.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { isRunActive } from "../utils.js";

export class KnowledgeView {
  constructor() {
    this.container = null;
    this.knowledge = null;
    this.activeRun = null;
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
        <div class="animate-spin text-[#6956E8] inline-block">${icon("refresh-cw", "icon-lg")}</div>
        <div class="text-sm font-medium font-sans">Obteniendo huella meta-estadística y base de conocimiento de CATML...</div>
      </div>
    `;
  }

  async fetchData() {
    try {
      const dsRuns = store.getRunsForActiveDataset();
      this.activeRun = store.getActiveRun()
        || dsRuns.find(r => isRunActive(r))
        || dsRuns[0]
        || null;

      const datasetId = this.activeRun
        ? this.activeRun.dataset_id
        : (store.getActiveDatasetId() || "");
      this.knowledge = await api.getKnowledge(datasetId).catch(() => null);
    } catch (e) {
      console.warn("KnowledgeView fetchData error:", e);
    }
  }

  render() {
    const k = this.knowledge || {};
    const fp = k.current_fingerprint || {
      dataset_name: this.activeRun ? this.activeRun.dataset_name : "Dataset Activo",
      rows: null,
      features: null,
      categorical_ratio: null,
      numerical_ratio: null,
      missing_ratio: null,
      target_entropy: null,
    };
    const similar = k.similar_datasets || [];
    const rankings = k.historical_rankings || [];
    const warmStart = k.warm_start || null;

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- V0.8 Meta-Learning Header -->
        <div class="workbench-card p-4 bg-purple-950/20 border-purple-900/60 flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <span class="text-[#6956E8]">${icon("brain", "icon-xl", 24)}</span>
            <div>
              <div class="flex items-center space-x-2">
                <h2 class="text-base font-bold text-slate-100">Base de Conocimiento y Meta-Aprendizaje CATML</h2>
                <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono">Capa de Memoria</span>
              </div>
              <p class="text-xs text-slate-400">Meta-aprendizaje autónomo a partir de experimentos históricos y meta-características tabulares</p>
            </div>
          </div>
          <span class="badge-sys text-xs px-3 py-1 rounded-full font-mono font-semibold">${similar.length} Datasets en Memoria</span>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <!-- Left: Dataset Fingerprint & Similarity -->
          <div class="lg:col-span-5 space-y-6">
            <!-- Fingerprint Card -->
            <div class="workbench-card p-5 space-y-3">
              <div class="flex items-center justify-between border-b border-slate-800 pb-2">
                <span class="text-xs uppercase font-bold text-slate-300 tracking-wider">Huella Digital del Dataset Actual</span>
                <span class="font-mono text-xs text-indigo-400 font-semibold">${fp.dataset_name || "Activo"}</span>
              </div>

              <div class="grid grid-cols-2 gap-3 text-xs pt-1 font-mono">
                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Filas</span>
                  <div class="text-sm font-bold text-slate-100">${fp.rows != null ? Number(fp.rows).toLocaleString() : "—"}</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Características</span>
                  <div class="text-sm font-bold text-slate-100">${fp.features != null ? fp.features : "—"}</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Proporción Categórica</span>
                  <div class="text-sm font-bold text-purple-400">${fp.categorical_ratio != null ? (fp.categorical_ratio * 100).toFixed(0) + '%' : "—"}</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Proporción Numérica</span>
                  <div class="text-sm font-bold text-indigo-400">${fp.numerical_ratio != null ? (fp.numerical_ratio * 100).toFixed(0) + '%' : "—"}</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Proporción de Nulos</span>
                  <div class="text-sm font-bold text-amber-400">${fp.missing_ratio != null ? (fp.missing_ratio * 100).toFixed(1) + '%' : "0.0%"}</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Entropía del Target</span>
                  <div class="text-sm font-bold text-emerald-400">${fp.target_entropy != null ? Number(fp.target_entropy).toFixed(3) : "—"}</div>
                </div>
              </div>
            </div>

            <!-- Similar Datasets in CATML Memory -->
            <div class="workbench-card p-5 space-y-3">
              <span class="text-xs uppercase font-bold text-slate-300 tracking-wider">Datasets Similares en Memoria</span>
              <div class="space-y-2 pt-1 text-xs">
                ${similar.length > 0 ? similar.map(ds => `
                  <div class="p-3 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 cursor-pointer flex items-center justify-between" tabindex="0" role="button" aria-label="Dataset similar ${ds.name}">
                    <div>
                      <div class="font-bold text-slate-200">${ds.name}</div>
                      <div class="text-[11px] text-slate-400">${(ds.reasons || []).join(" • ")}</div>
                    </div>
                    <span class="badge-gain text-xs px-2.5 py-1 rounded font-mono font-bold">${Math.round(ds.similarity * 100)}% de similitud</span>
                  </div>
                `).join("") : `
                  <div class="p-4 rounded-xl bg-[#090C12] border border-[#252C38] text-center text-[#8B95A7] font-sans">
                    Sin datasets similares registrados en el almacén de meta-aprendizaje.
                  </div>
                `}
              </div>
            </div>
          </div>

          <!-- Right: Meta-learning Evidence & Warm Start -->
          <div class="lg:col-span-7 space-y-6">
            <!-- Historical Empirical Evidence on Similar Problems -->
            <div class="workbench-card p-5 space-y-3">
              <span class="text-xs uppercase font-bold text-slate-300 tracking-wider">Rendimiento Histórico de Algoritmos en Datasets Similares</span>
              <div class="overflow-x-auto pt-1">
                <table class="w-full wb-table text-left" aria-label="Rendimiento histórico de algoritmos">
                  <thead>
                    <tr>
                      <th>Algoritmo</th>
                      <th>Experimentos Evaluados</th>
                      <th>Rango Medio (Mean Rank)</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${rankings.length > 0 ? rankings.map(r => `
                      <tr>
                        <td class="font-bold text-slate-200">${r.model}</td>
                        <td class="font-mono text-slate-400">${r.experiments} experimentos</td>
                        <td class="font-mono font-bold text-emerald-400">${r.mean_rank}</td>
                      </tr>
                    `).join("") : `
                      <tr>
                        <td colspan="3" class="text-center text-slate-500 py-6 text-xs font-sans">
                          Sin evaluaciones empíricas previas para esta distribución.
                        </td>
                      </tr>
                    `}
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Recommended Warm Start Box -->
            ${warmStart ? `
              <div class="workbench-card p-5 space-y-4 border-indigo-700/60 bg-indigo-950/20">
                <div class="flex items-center justify-between">
                  <div>
                    <div class="text-xs uppercase font-bold text-indigo-300 tracking-wider">Warm Start Recomendado para HPO</div>
                    <div class="text-base font-bold text-slate-100 mt-0.5">Prior de ${warmStart.recommended_model || "LightGBM"}</div>
                  </div>
                  <span class="badge-gain text-xs px-3 py-1 rounded font-mono font-bold">${warmStart.expected_search_reduction || "~35%"} de Reducción en Búsqueda</span>
                </div>

                <div class="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs space-y-1 text-slate-300">
                  ${warmStart.params ? Object.entries(warmStart.params).map(([k, v]) => `
                    <div>${k}: <span class="text-indigo-400 font-bold">${v}</span></div>
                  `).join("") : `
                    <div>learning_rate: <span class="text-indigo-400 font-bold">0.05</span></div>
                  `}
                </div>

                <div class="flex justify-end">
                  <button id="btnUseWarmStart" class="btn-signal" aria-label="Usar Warm Start en próximo HPO">
                    <span class="inline-flex items-center gap-1.5">${icon("zap", "icon-sm")} <span>Usar Warm Start en Próximo HPO</span></span>
                  </button>
                </div>
              </div>
            ` : `
              <div class="workbench-card p-5 space-y-3 border-[#242A36] bg-[#090C12]">
                <div class="text-xs uppercase font-bold text-[#8B95A7] tracking-wider font-sans">Priors de Warm Start</div>
                <p class="text-xs text-[#8B95A7] font-sans">No hay priors de warm-start disponibles todavía para este dataset. A medida que ejecute optimizaciones bayesianas en diferentes datasets, el motor meta-heurístico aprenderá priors de inicialización.</p>
              </div>
            `}
          </div>
        </div>
      </div>
    `;

    this.container.querySelector("#btnUseWarmStart")?.addEventListener("click", () => {
      if (warmStart && warmStart.params) {
        store.setState({ warmStartParams: warmStart.params });
        alert(`Priors de warm start cargados con éxito (${warmStart.recommended_model || "modelo"}). Los próximos experimentos de HPO se inicializarán con esta configuración.`);
      } else {
        alert("No hay priors de warm-start disponibles para aplicar.");
      }
    });
  }

  destroy() {
    this.container = null;
  }
}
