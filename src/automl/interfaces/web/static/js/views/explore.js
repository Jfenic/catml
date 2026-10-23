/**
 * ExploreView — CATML Explore Visual Interactive Laboratory (Phase E3)
 * Professional analytical workbench view complying strictly with ADR-006 (Tech Minimalista Premium).
 * 
 * Features:
 * - Study and Execution Selector with New Study Modal & Instant Run CTA.
 * - Dynamic Findings Gallery with Category, Severity & Benjamini-Hochberg FDR filters.
 * - Declarative SVG/Table Visualizations:
 *   * Bivariate Correlation Heatmap (Pearson & Spearman toggle).
 *   * Categorical Association Matrix (Bias-corrected Cramér's V).
 *   * Interactive Vector Distribution Histograms.
 *   * Quantile Boxplots with Outlier markers.
 *   * Bivariate Scatter Plots with Linear Trend Lines.
 * - Technical Markdown / JSON Report Export.
 */
import { store } from "../store.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { escapeHtml, getTransformationStageInfo, formatPValue } from "../utils.js";

const GUIDED_FINDINGS_LIMIT = 5;
const GUIDED_HYPOTHESES_LIMIT = 5;

export class ExploreView {
  constructor() {
    this.container = null;
    this.activeDatasetId = null;
    this.studies = [];
    this.selectedStudyId = null;
    this.studyDetail = null;
    this.findings = [];
    this.visualizations = [];
    this.hypotheses = [];
    this.evidence = [];
    this.recipes = [];
    this.verifyingHypothesisId = null;
    this.loading = true;
    this.runningStudy = false;
    this.showCreateModal = false;

    // Guiado = top findings first; Avanzado = full gallery + matrices
    this.showAdvancedExplore = false;

    // Filter states
    this.filterCategory = "ALL";
    this.filterSeverity = "ALL";
    this.filterFdrOnly = false;
    this.activeVizType = "correlation_matrix";
    this.matrixMethod = "pearson"; // "pearson" | "spearman"
    this.matrixColumnFilter = "";
    this.guidedCorrExpanded = false;
  }

  async mount(container) {
    this.container = container;
    const state = store.getState();
    this.activeDatasetId = state.activeDatasetId;
    this.renderLoading();
    await this.fetchData();
    this.render();
  }

  destroy() {
    this.container = null;
  }

  renderLoading() {
    if (!this.container) return;
    this.container.innerHTML = `
      <div class="workbench-card p-12 text-center text-[#8B95A7] space-y-3">
        <div class="animate-spin text-[#4F67FF] inline-block">${icon("refresh-cw", "icon-lg")}</div>
        <div class="text-sm font-sans font-medium">Cargando estudios y motor analítico...</div>
      </div>
    `;
  }

  async fetchData() {
    this.loading = true;
    try {
      if (!this.activeDatasetId) {
        this.studies = [];
        this.studyDetail = null;
        return;
      }

      this.studies = await api.getAnalysisStudies(this.activeDatasetId).catch(() => []);

      if (this.studies.length > 0) {
        if (!this.selectedStudyId || !this.studies.some(s => s.id === this.selectedStudyId)) {
          this.selectedStudyId = this.studies[0].id;
        }

        const detail = await api.getAnalysisStudy(this.selectedStudyId).catch(() => null);
        if (detail) {
          this.studyDetail = detail;
          this.findings = detail.findings || [];
          this.visualizations = detail.visualizations || [];
          this.hypotheses = detail.hypotheses || [];
        }
        const evList = await api.getAnalysisEvidence(this.selectedStudyId).catch(() => []);
        this.evidence = evList || [];
      } else {
        this.selectedStudyId = null;
        this.studyDetail = null;
        this.findings = [];
        this.visualizations = [];
        this.hypotheses = [];
        this.evidence = [];
      }

      this.recipes = await api.getRecipes(this.activeDatasetId).catch(() => []);
    } catch (err) {
      console.error("ExploreView fetch error:", err);
    } finally {
      this.loading = false;
    }
  }

  _getFilteredFindings() {
    const severityWeight = {
      CRITICAL: 3,
      WARNING: 2,
      INFO: 1,
    };

    const list = this.findings.filter(f => {
      // Category filter
      if (this.filterCategory === "COLLINEARITY" && f.finding_type !== "high_collinearity") return false;
      if (this.filterCategory === "DISTRIBUTION" && !["skewness", "normality_violation", "multimodal_distribution", "outlier_density", "multivariate_outliers"].includes(f.finding_type)) return false;
      if (this.filterCategory === "CATEGORICAL" && f.finding_type !== "categorical_association") return false;
      if (this.filterCategory === "TARGET" && !["target_class_separation", "target_correlation"].includes(f.finding_type)) return false;

      // Severity filter
      const sev = this._classifySeverity(f);
      if (this.filterSeverity !== "ALL" && sev !== this.filterSeverity) return false;

      // FDR significance toggle
      if (this.filterFdrOnly) {
        const isFdrSig = f.metrics && f.metrics.fdr_significant === true;
        if (!isFdrSig) return false;
      }

      return true;
    });

    // Ordenar por importancia: Severidad (CRITICAL > WARNING > INFO), significancia FDR y magnitud del efecto
    return list.sort((a, b) => {
      const sevA = severityWeight[this._classifySeverity(a)] || 0;
      const sevB = severityWeight[this._classifySeverity(b)] || 0;
      if (sevA !== sevB) return sevB - sevA;

      const fdrA = a.metrics?.fdr_significant ? 1 : 0;
      const fdrB = b.metrics?.fdr_significant ? 1 : 0;
      if (fdrA !== fdrB) return fdrB - fdrA;

      const effA = Math.abs(Number(a.effect_size != null ? a.effect_size : (a.metrics?.pearson_r ?? a.metrics?.cramers_v ?? a.metrics?.cohens_d ?? 0)));
      const effB = Math.abs(Number(b.effect_size != null ? b.effect_size : (b.metrics?.pearson_r ?? b.metrics?.cramers_v ?? b.metrics?.cohens_d ?? 0)));
      return effB - effA;
    });
  }

  _classifySeverity(f) {
    const type = f?.finding_type || "";
    const effect = Math.abs(Number(
      f?.effect_size != null
        ? f.effect_size
        : (f?.metrics?.pearson_r ?? f?.metrics?.cramers_v ?? f?.metrics?.cohens_d ?? 0)
    ));
    const fdrSig = f?.metrics?.fdr_significant === true;

    // Reserve CRITICAL for strong, actionable risks — not every collinear pair.
    if (type === "high_cardinality_identifier") return "CRITICAL";
    if (type === "multivariate_outliers" && (fdrSig || effect >= 0.5)) return "CRITICAL";
    if (type === "high_collinearity" && effect >= 0.90) return "CRITICAL";
    if (type === "high_collinearity") return "WARNING";
    if (type === "multivariate_outliers") return "WARNING";
    if (["skewness", "outlier_density", "categorical_association", "multimodal_distribution", "target_class_separation", "target_correlation"].includes(type)) {
      return "WARNING";
    }
    return "INFO";
  }

  _visibleFindings(filtered) {
    if (this.showAdvancedExplore) return filtered;
    return filtered.slice(0, GUIDED_FINDINGS_LIMIT);
  }

  _visibleHypotheses() {
    if (this.showAdvancedExplore) return this.hypotheses;
    return (this.hypotheses || []).slice(0, GUIDED_HYPOTHESES_LIMIT);
  }

  render() {
    if (!this.container) return;

    if (!this.activeDatasetId) {
      this.container.innerHTML = `
        <div class="workbench-card p-12 text-center space-y-4">
          <div class="text-[#8B95A7] inline-block">${icon("database", "icon-lg")}</div>
          <h3 class="text-base font-bold font-sans text-[#F7F8FA]">Sin dataset activo</h3>
          <p class="text-xs text-[#8B95A7]">Selecciona o carga un dataset para realizar estudios exploratorios.</p>
        </div>
      `;
      return;
    }

    const filteredFindings = this._getFilteredFindings();
    const visibleFindings = this._visibleFindings(filteredFindings);
    const activeViz = this.visualizations.find(v => v.chart_type === this.activeVizType) || this.visualizations[0];

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Header & Study Selector Controls -->
        <div class="workbench-card p-5 space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div class="flex items-center space-x-3">
              <span class="text-[#4F67FF]">${icon("compass", "icon-md")}</span>
              <div>
                <h3 class="text-sm font-bold font-sans text-[#F7F8FA]">Laboratorio Exploratorio · CATML Explore</h3>
                <p class="text-[11px] text-[#8B95A7] font-mono">Inferencia matemática determinista sin alucinaciones</p>
              </div>
            </div>

            <!-- Action buttons -->
            <div class="flex items-center space-x-2 flex-wrap justify-end">
              <div class="flex items-center bg-[#090C12] p-0.5 rounded-lg border border-[#242A36]" role="group" aria-label="Modo de exploración">
                <button type="button" id="btnExploreModeGuided" class="px-2.5 py-1 rounded text-[11px] font-mono ${!this.showAdvancedExplore ? 'bg-[#4F67FF] text-white font-semibold' : 'text-[#8B95A7]'}">Guiado</button>
                <button type="button" id="btnExploreModeAdvanced" class="px-2.5 py-1 rounded text-[11px] font-mono ${this.showAdvancedExplore ? 'bg-[#4F67FF] text-white font-semibold' : 'text-[#8B95A7]'}">Avanzado</button>
              </div>
              <button id="btnOpenNewStudyModal" class="btn-technical text-xs flex items-center space-x-1.5">
                ${icon("plus", "icon-sm")}
                <span>Nuevo Estudio</span>
              </button>

              ${this.studyDetail ? `
                <button id="btnRunStudyExecution" ${this.runningStudy ? "disabled" : ""} class="btn-signal text-xs flex items-center space-x-1.5 shadow-md shadow-[#4F67FF]/20 ${this.runningStudy ? 'opacity-60 cursor-not-allowed' : ''}">
                  ${this.runningStudy ? `<span class="animate-spin">${icon("refresh-cw", "icon-sm")}</span>` : icon("play", "icon-sm")}
                  <span>${this.runningStudy ? "Calculando..." : "Ejecutar Análisis"}</span>
                </button>
                <button id="btnExportStudyReport" class="btn-technical text-xs flex items-center space-x-1.5">
                  ${icon("download", "icon-sm")}
                  <span>Exportar Informe</span>
                </button>
              ` : ""}
            </div>
          </div>

          <!-- Study dropdown & quick status -->
          ${this.studies.length > 0 ? `
            <div class="border-t border-[#242A36] pt-3 flex flex-wrap items-center justify-between gap-3">
              <div class="flex items-center space-x-2">
                <span class="text-xs font-sans text-[#8B95A7]">Estudio activo:</span>
                <select id="selectActiveStudy" class="input-base bg-[#161B26] border border-[#242A36] text-[#F7F8FA] text-xs font-mono py-1.5 px-3 rounded-lg focus:border-[#4F67FF] outline-none max-w-sm cursor-pointer shadow-sm">
                  ${this.studies.map(s => `
                    <option value="${s.id}" class="bg-[#161B26] text-[#F7F8FA]" ${s.id === this.selectedStudyId ? "selected" : ""}>
                      ${escapeHtml(s.name)} [${s.status}]
                    </option>
                  `).join("")}
                </select>
              </div>

              ${this.studyDetail ? `
                <div class="flex items-center space-x-3 text-xs font-mono">
                  <span class="text-[#8B95A7]">Target: <strong class="text-[#F7F8FA]">${escapeHtml(this.studyDetail.target_column || "Ninguno (Descriptivo)")}</strong></span>
                  <span class="text-[#8B95A7]">Hallazgos: <strong class="text-[#53C8FF]">${this.findings.length}</strong></span>
                  <span class="text-[#8B95A7]">Hipótesis: <strong class="text-[#22C55E]">${this.hypotheses.length}</strong></span>
                </div>
              ` : ""}
            </div>
          ` : `
            <div class="border-t border-[#242A36] pt-4 text-center py-6 space-y-2">
              <p class="text-xs text-[#8B95A7] font-sans">No hay estudios registrados para este dataset.</p>
              <button id="btnCreateInitialStudy" class="btn-signal text-xs inline-flex items-center space-x-1.5">
                ${icon("plus", "icon-sm")}
                <span>Crear Primer Estudio Exploratorio</span>
              </button>
            </div>
          `}
        </div>

        ${this.studyDetail ? `
          <!-- Findings first (conclusions before dense matrices) -->
          <div class="workbench-card p-5 space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#242A36] pb-3 gap-3">
              <div class="flex items-center space-x-2 flex-wrap">
                <span class="text-[#F59E0B]">${icon("alert-circle", "icon-sm")}</span>
                <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">
                  ${this.showAdvancedExplore ? "Galería de Hallazgos Estadísticos" : "Hallazgos prioritarios"}
                </h4>
                <span class="badge-sys text-[10px] font-mono">
                  ${visibleFindings.length} de ${filteredFindings.length}
                  ${!this.showAdvancedExplore && filteredFindings.length > GUIDED_FINDINGS_LIMIT ? ` · top ${GUIDED_FINDINGS_LIMIT}` : ""}
                </span>
              </div>

              ${this.showAdvancedExplore ? `
              <div class="flex flex-wrap items-center gap-2 text-xs font-sans">
                <select id="filterCategorySelect" class="input-base bg-[#161B26] border border-[#242A36] text-[#F7F8FA] text-xs font-mono py-1.5 px-3 rounded-lg focus:border-[#4F67FF] outline-none cursor-pointer shadow-sm">
                  <option value="ALL" class="bg-[#161B26] text-[#F7F8FA]" ${this.filterCategory === "ALL" ? "selected" : ""}>Todas las categorías</option>
                  <option value="COLLINEARITY" class="bg-[#161B26] text-[#F7F8FA]" ${this.filterCategory === "COLLINEARITY" ? "selected" : ""}>Colinealidad</option>
                  <option value="DISTRIBUTION" class="bg-[#161B26] text-[#F7F8FA]" ${this.filterCategory === "DISTRIBUTION" ? "selected" : ""}>Distribución / Outliers</option>
                  <option value="CATEGORICAL" class="bg-[#161B26] text-[#F7F8FA]" ${this.filterCategory === "CATEGORICAL" ? "selected" : ""}>Asociación Categórica</option>
                  <option value="TARGET" class="bg-[#161B26] text-[#F7F8FA]" ${this.filterCategory === "TARGET" ? "selected" : ""}>Separación Objetivo</option>
                </select>
                <select id="filterSeveritySelect" class="input-base bg-[#161B26] border border-[#242A36] text-[#F7F8FA] text-xs font-mono py-1.5 px-3 rounded-lg focus:border-[#4F67FF] outline-none cursor-pointer shadow-sm">
                  <option value="ALL" class="bg-[#161B26] text-[#F7F8FA]" ${this.filterSeverity === "ALL" ? "selected" : ""}>Toda severidad</option>
                  <option value="CRITICAL" class="bg-[#161B26] text-[#F7F8FA]" ${this.filterSeverity === "CRITICAL" ? "selected" : ""}>Críticos</option>
                  <option value="WARNING" class="bg-[#161B26] text-[#F7F8FA]" ${this.filterSeverity === "WARNING" ? "selected" : ""}>Advertencias</option>
                  <option value="INFO" class="bg-[#161B26] text-[#F7F8FA]" ${this.filterSeverity === "INFO" ? "selected" : ""}>Informativos</option>
                </select>
                <label class="flex items-center space-x-1.5 cursor-pointer text-[#8B95A7] hover:text-[#F7F8FA] select-none text-[11px] font-mono">
                  <input type="checkbox" id="checkFdrOnly" ${this.filterFdrOnly ? "checked" : ""} class="rounded border-[#242A36] bg-[#090C12] text-[#4F67FF] focus:ring-0">
                  <span>Solo FDR p &lt; 0.05</span>
                </label>
              </div>
              ` : `
              <button type="button" id="btnShowAllFindings" class="btn-technical text-[11px] font-mono">
                Ver todos (${filteredFindings.length})
              </button>
              `}
            </div>

            ${!this.showAdvancedExplore ? `
              <p class="text-xs text-[#8B95A7] font-sans">
                Resumen ejecutivo: los ${Math.min(GUIDED_FINDINGS_LIMIT, filteredFindings.length)} hallazgos más relevantes por severidad y efecto. Activa <strong class="text-[#F7F8FA]">Avanzado</strong> para filtros, matrices completas e hipótesis restantes.
              </p>
            ` : ""}

            ${filteredFindings.length === 0 ? `
              <div class="p-8 text-center text-[#8B95A7] font-sans text-xs space-y-1">
                <p>No se encontraron hallazgos que coincidan con los filtros aplicados.</p>
              </div>
            ` : `
              <div class="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                ${visibleFindings.map(f => this._renderFindingCardHtml(f)).join("")}
              </div>
            `}
          </div>

          ${this._renderHypothesesAndEvidenceHtml()}

          <!-- Visualizations: guided = strong pairs; advanced = full matrices -->
          ${this.visualizations.length > 0 ? `
            <div class="workbench-card p-5 space-y-4">
              <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#242A36] pb-3 gap-2">
                <div class="flex items-center space-x-2">
                  <span class="text-[#53C8FF]">${icon("bar-chart-2", "icon-sm")}</span>
                  <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">
                    ${this.showAdvancedExplore ? "Visualizaciones Declarativas" : "Asociaciones destacadas"}
                  </h4>
                </div>
                ${this.showAdvancedExplore ? `
                <div class="flex flex-wrap items-center gap-1.5 text-xs font-sans">
                  ${this._renderVizTypeButtons(activeViz)}
                </div>
                ` : ""}
              </div>
              <div id="activeVisualizationContainer">
                ${this.showAdvancedExplore
                  ? this._renderVisualizationContent(activeViz)
                  : this._renderGuidedAssociationsHtml(activeViz)}
              </div>
            </div>
          ` : ""}

          ${this._renderRecipesSectionHtml()}
        ` : ""}

        <!-- Create Study Modal -->
        ${this.showCreateModal ? this._renderCreateStudyModalHtml() : ""}
      </div>
    `;

    this._bindEvents();
  }

  _renderVizTypeButtons(activeViz) {
    const typesPresent = [...new Set(this.visualizations.map(v => v.chart_type))];
    const labels = {
      correlation_matrix: "Correlación (Pearson/Spearman)",
      categorical_association_matrix: "Asociación Categórica (Cramér's V)",
      histogram: "Histogramas",
      boxplot: "Diagramas de Caja",
      scatter: "Dispersión",
    };

    return typesPresent.map(t => {
      const isSelected = activeViz && activeViz.chart_type === t;
      return `
        <button class="btn-viz-tab px-2.5 py-1 rounded text-xs font-mono transition-colors ${isSelected ? 'bg-[#4F67FF] text-white font-bold' : 'bg-[#161B26] text-[#8B95A7] hover:text-[#F7F8FA]'}" data-viz-type="${t}">
          ${labels[t] || t}
        </button>
      `;
    }).join("");
  }

  _renderVisualizationContent(viz) {
    if (!viz) return `<div class="text-xs text-[#8B95A7] p-4 text-center">Sin visualizaciones disponibles.</div>`;

    if (viz.chart_type === "correlation_matrix") {
      return this._renderCorrelationMatrixHtml(viz);
    } else if (viz.chart_type === "categorical_association_matrix") {
      return this._renderCategoricalMatrixHtml(viz);
    } else if (viz.chart_type === "histogram") {
      return this._renderHistogramSvgHtml(viz);
    } else if (viz.chart_type === "boxplot") {
      return this._renderBoxplotSvgHtml(viz);
    } else if (viz.chart_type === "scatter") {
      return this._renderScatterSvgHtml(viz);
    }

    return `<pre class="text-xs font-mono p-4 bg-[#090C12] rounded border border-[#242A36] overflow-x-auto">${escapeHtml(JSON.stringify(viz, null, 2))}</pre>`;
  }

  _renderGuidedAssociationsHtml(viz) {
    const series = (viz && (viz.data_series || viz.data)) || {};
    const cols = series.columns || [];
    const matrix = series.matrix || [];
    const pairs = [];
    for (let i = 0; i < cols.length; i++) {
      for (let j = i + 1; j < cols.length; j++) {
        const val = matrix[i] && matrix[i][j] != null ? Number(matrix[i][j]) : NaN;
        if (!Number.isFinite(val)) continue;
        const abs = Math.abs(val);
        if (abs >= 0.70) {
          pairs.push({ a: cols[i], b: cols[j], r: val, abs });
        }
      }
    }
    pairs.sort((x, y) => y.abs - x.abs);
    const top = pairs.slice(0, 5);

    const corrViz = this.visualizations.find(v => v.chart_type === "correlation_matrix") || viz;
    const catViz = this.visualizations.find(v => v.chart_type === "categorical_association_matrix");

    const pairsBlock = top.length === 0
      ? `
        <div class="text-xs text-[#8B95A7] font-sans p-3 bg-[#090C12] rounded-lg border border-[#242A36]">
          No hay pares con |r| ≥ 0.70. Amplía abajo para ver la matriz completa (Pearson / Spearman) y, si aplica, Cramér's V.
        </div>`
      : `
        <div class="overflow-x-auto border border-[#242A36] rounded-lg">
          <table class="w-full text-xs font-mono border-collapse">
            <thead>
              <tr class="bg-[#090C12] text-[#8B95A7] border-b border-[#242A36]">
                <th class="p-2.5 text-left">Variable A</th>
                <th class="p-2.5 text-left">Variable B</th>
                <th class="p-2.5 text-right">r (Pearson)</th>
              </tr>
            </thead>
            <tbody>
              ${top.map(p => `
                <tr class="border-b border-[#1C2230]">
                  <td class="p-2.5 text-[#F7F8FA]">${escapeHtml(p.a)}</td>
                  <td class="p-2.5 text-[#F7F8FA]">${escapeHtml(p.b)}</td>
                  <td class="p-2.5 text-right font-semibold ${p.abs >= 0.9 ? 'text-[#EF4444]' : 'text-[#53C8FF]'}">${p.r.toFixed(3)}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>`;

    return `
      <div class="space-y-3">
        <p class="text-xs text-[#8B95A7] font-sans">Pares más correlacionados (|r| ≥ 0.70). Amplía para matrices Pearson, Spearman y Cramér's V.</p>
        ${pairsBlock}
        <details id="guidedCorrExpand" class="rounded-xl border border-[#242A36] bg-[#090C12] open:border-[#4F67FF]/35" ${this.guidedCorrExpanded ? "open" : ""}>
          <summary class="cursor-pointer select-none px-4 py-3 text-xs font-sans font-medium text-[#F7F8FA] flex items-center justify-between gap-2 list-none">
            <span class="inline-flex items-center gap-2">
              ${icon("layers", "icon-sm")}
              <span>Ampliar matrices de correlación</span>
            </span>
            <span class="text-[10px] font-mono text-[#8B95A7]">Pearson · Spearman · Cramér's V</span>
          </summary>
          <div class="px-4 pb-4 space-y-5 border-t border-[#242A36] pt-4">
            <div class="space-y-2">
              <h5 class="text-xs font-semibold font-sans text-[#F7F8FA]">Correlación numérica (Pearson / Spearman)</h5>
              ${corrViz ? this._renderCorrelationMatrixHtml(corrViz) : `<p class="text-xs text-[#8B95A7]">Sin matriz de correlación numérica en este estudio.</p>`}
            </div>
            ${catViz ? `
              <div class="space-y-2 pt-2 border-t border-[#242A36]">
                <h5 class="text-xs font-semibold font-sans text-[#F7F8FA]">Asociación categórica (Cramér's V)</h5>
                ${this._renderCategoricalMatrixHtml(catViz)}
              </div>
            ` : `
              <p class="text-[11px] text-[#8B95A7] font-sans pt-1 border-t border-[#242A36]">
                Cramér's V aparece cuando el estudio incluye variables categóricas con matriz de asociación.
              </p>
            `}
          </div>
        </details>
      </div>`;
  }

  _renderCorrelationMatrixHtml(viz) {
    const series = viz.data_series || viz.data || {};
    const allCols = series.columns || [];
    const isSpearman = this.matrixMethod === "spearman";
    const matrix = isSpearman ? (series.spearman_matrix || series.matrix || []) : (series.matrix || []);
    const q = (this.matrixColumnFilter || "").trim().toLowerCase();
    const idxs = allCols
      .map((c, i) => ({ c, i }))
      .filter(({ c }) => !q || String(c).toLowerCase().includes(q))
      .map(({ i }) => i);
    const cols = idxs.map(i => allCols[i]);

    return `
      <div class="space-y-3">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs font-mono">
          <span class="text-[#8B95A7]">${viz.title || "Matriz de Correlación"}</span>
          <div class="flex flex-wrap items-center gap-2">
            <input id="inputMatrixColumnFilter" type="search" value="${escapeHtml(this.matrixColumnFilter)}"
              placeholder="Filtrar variables…"
              class="input-base bg-[#161B26] border border-[#242A36] text-[#F7F8FA] text-xs font-mono py-1 px-2 rounded-lg focus:border-[#4F67FF] outline-none w-44" />
            <div class="flex items-center space-x-1 bg-[#090C12] p-0.5 rounded border border-[#242A36]">
              <button class="btn-matrix-toggle px-2 py-0.5 rounded ${!isSpearman ? 'bg-[#4F67FF] text-white font-semibold' : 'text-[#8B95A7]'}" data-method="pearson">Pearson</button>
              <button class="btn-matrix-toggle px-2 py-0.5 rounded ${isSpearman ? 'bg-[#4F67FF] text-white font-semibold' : 'text-[#8B95A7]'}" data-method="spearman">Spearman</button>
            </div>
          </div>
        </div>

        ${cols.length === 0 ? `
          <div class="text-xs text-[#8B95A7] p-4 text-center">Ninguna variable coincide con el filtro.</div>
        ` : `
        <div class="overflow-x-auto border border-[#242A36] rounded-lg max-h-[70vh]">
          <table class="w-full text-xs font-mono border-collapse">
            <thead>
              <tr>
                <th class="p-2 text-left text-[#8B95A7] bg-[#090C12] border-b border-r border-[#242A36] sticky left-0 z-10">Variable</th>
                ${cols.map(c => `
                  <th class="p-2 text-center text-[#8B95A7] bg-[#090C12] border-b border-r border-[#242A36] truncate max-w-[100px]" title="${escapeHtml(c)}">
                    ${escapeHtml(c)}
                  </th>
                `).join("")}
              </tr>
            </thead>
            <tbody>
              ${idxs.map((rIdx) => {
                const rowName = allCols[rIdx];
                return `
                <tr>
                  <td class="p-2 font-medium text-[#F7F8FA] bg-[#090C12] border-b border-r border-[#242A36] truncate max-w-[140px] sticky left-0" title="${escapeHtml(rowName)}">
                    ${escapeHtml(rowName)}
                  </td>
                  ${idxs.map((cIdx) => {
                    const colName = allCols[cIdx];
                    const val = (matrix[rIdx] && matrix[rIdx][cIdx] != null) ? matrix[rIdx][cIdx] : 0.0;
                    const absVal = Math.abs(val);
                    const isDiag = rIdx === cIdx;
                    let bgColor = "rgba(22, 27, 38, 0.4)";
                    let textColor = "#8B95A7";

                    if (!isDiag) {
                      if (val > 0) {
                        bgColor = `rgba(79, 103, 255, ${Math.min(0.85, absVal * 0.9)})`;
                        textColor = absVal > 0.4 ? "#FFFFFF" : "#F7F8FA";
                      } else {
                        bgColor = `rgba(239, 68, 68, ${Math.min(0.85, absVal * 0.9)})`;
                        textColor = absVal > 0.4 ? "#FFFFFF" : "#F7F8FA";
                      }
                    }

                    return `
                      <td style="background-color: ${bgColor}; color: ${textColor};" class="p-2 text-center border-b border-r border-[#242A36] font-mono ${absVal >= 0.70 && !isDiag ? 'font-bold' : ''}" title="${escapeHtml(rowName)} vs ${escapeHtml(colName)}: ${Number(val).toFixed(3)}">
                        ${Number(val).toFixed(2)}
                      </td>
                    `;
                  }).join("")}
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

  _renderCategoricalMatrixHtml(viz) {
    const series = viz.data_series || viz.data || {};
    const cols = series.columns || [];
    const matrix = series.matrix || [];

    return `
      <div class="space-y-3">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="text-[#8B95A7]">Cramér's V corregido por sesgo muestral [0.0 - 1.0]</span>
        </div>

        <div class="overflow-x-auto border border-[#242A36] rounded-lg">
          <table class="w-full text-xs font-mono border-collapse">
            <thead>
              <tr>
                <th class="p-2 text-left text-[#8B95A7] bg-[#090C12] border-b border-r border-[#242A36]">Variable</th>
                ${cols.map(c => `
                  <th class="p-2 text-center text-[#8B95A7] bg-[#090C12] border-b border-r border-[#242A36] truncate max-w-[100px]" title="${escapeHtml(c)}">
                    ${escapeHtml(c)}
                  </th>
                `).join("")}
              </tr>
            </thead>
            <tbody>
              ${cols.map((rowName, rIdx) => `
                <tr>
                  <td class="p-2 font-medium text-[#F7F8FA] bg-[#090C12] border-b border-r border-[#242A36] truncate max-w-[140px]" title="${escapeHtml(rowName)}">
                    ${escapeHtml(rowName)}
                  </td>
                  ${cols.map((colName, cIdx) => {
                    const val = (matrix[rIdx] && matrix[rIdx][cIdx] != null) ? matrix[rIdx][cIdx] : 0.0;
                    const isDiag = rIdx === cIdx;
                    let bgColor = isDiag ? "rgba(22, 27, 38, 0.4)" : `rgba(83, 200, 255, ${Math.min(0.85, val * 0.9)})`;
                    let textColor = val > 0.4 && !isDiag ? "#FFFFFF" : "#F7F8FA";

                    return `
                      <td style="background-color: ${bgColor}; color: ${textColor};" class="p-2 text-center border-b border-r border-[#242A36] font-mono ${val >= 0.40 && !isDiag ? 'font-bold' : ''}" title="${escapeHtml(rowName)} vs ${escapeHtml(colName)}: ${Number(val).toFixed(3)}">
                        ${Number(val).toFixed(2)}
                      </td>
                    `;
                  }).join("")}
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  _renderHistogramSvgHtml(viz) {
    const series = viz.data_series || viz.data || {};
    const counts = series.counts || [];
    const edges = series.bin_edges || [];
    const maxCount = counts.length > 0 ? Math.max(...counts, 1) : 1;
    const svgWidth = 600;
    const svgHeight = 180;
    const padding = 30;
    const chartWidth = svgWidth - padding * 2;
    const chartHeight = svgHeight - padding * 2;
    const barWidth = counts.length > 0 ? chartWidth / counts.length : 10;
    const mean = series.mean != null ? Number(series.mean).toFixed(2) : "-";
    const std = series.std != null ? Number(series.std).toFixed(2) : "-";

    return `
      <div class="space-y-2 p-3 bg-[#090C12] rounded-lg border border-[#242A36]">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="text-[#F7F8FA] font-medium">${viz.title || "Histograma"}</span>
          <span class="text-[#8B95A7]">Media: ${mean} · Desv: ${std}</span>
        </div>

        <div class="w-full overflow-x-auto flex justify-center">
          <svg viewBox="0 0 ${svgWidth} ${svgHeight}" class="w-full max-w-2xl h-44">
            <!-- Background grid lines -->
            <line x1="${padding}" y1="${padding}" x2="${svgWidth - padding}" y2="${padding}" stroke="#242A36" stroke-dasharray="2" />
            <line x1="${padding}" y1="${padding + chartHeight / 2}" x2="${svgWidth - padding}" y2="${padding + chartHeight / 2}" stroke="#242A36" stroke-dasharray="2" />
            <line x1="${padding}" y1="${padding + chartHeight}" x2="${svgWidth - padding}" y2="${padding + chartHeight}" stroke="#384355" />

            <!-- Bars -->
            ${counts.map((c, i) => {
              const barHeight = (c / maxCount) * chartHeight;
              const x = padding + i * barWidth;
              const y = padding + chartHeight - barHeight;
              const edgeLow = edges[i] != null ? Number(edges[i]).toFixed(1) : "";
              const edgeHigh = edges[i+1] != null ? Number(edges[i+1]).toFixed(1) : "";
              return `
                <rect x="${x + 2}" y="${y}" width="${Math.max(1, barWidth - 4)}" height="${barHeight}" fill="#4F67FF" rx="2" opacity="0.85">
                  <title>Bin [${edgeLow} - ${edgeHigh}]: ${c} observaciones</title>
                </rect>
              `;
            }).join("")}
          </svg>
        </div>
      </div>
    `;
  }

  _renderBoxplotSvgHtml(viz) {
    const data = viz.data_series || viz.data || {};
    const q25 = data.q25 != null ? Number(data.q25).toFixed(2) : 0;
    const med = data.median != null ? Number(data.median).toFixed(2) : 0;
    const q75 = data.q75 != null ? Number(data.q75).toFixed(2) : 0;
    const wLow = data.whisker_low != null ? Number(data.whisker_low).toFixed(2) : 0;
    const wHigh = data.whisker_high != null ? Number(data.whisker_high).toFixed(2) : 0;

    return `
      <div class="space-y-2 p-4 bg-[#090C12] rounded-lg border border-[#242A36]">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="text-[#F7F8FA] font-medium">${viz.title || "Diagrama de Caja"}</span>
          <span class="text-[#F59E0B] font-semibold">${data.outliers_count || 0} outliers detectados</span>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-5 gap-2 pt-2 text-xs font-mono text-[#8B95A7]">
          <div class="bg-[#11151E] p-2 rounded border border-[#242A36]">Bigote Inf: <strong class="text-[#F7F8FA]">${wLow}</strong></div>
          <div class="bg-[#11151E] p-2 rounded border border-[#242A36]">Q1 (25%): <strong class="text-[#F7F8FA]">${q25}</strong></div>
          <div class="bg-[#11151E] p-2 rounded border border-[#242A36]">Mediana: <strong class="text-[#53C8FF]">${med}</strong></div>
          <div class="bg-[#11151E] p-2 rounded border border-[#242A36]">Q3 (75%): <strong class="text-[#F7F8FA]">${q75}</strong></div>
          <div class="bg-[#11151E] p-2 rounded border border-[#242A36]">Bigote Sup: <strong class="text-[#F7F8FA]">${wHigh}</strong></div>
        </div>
      </div>
    `;
  }

  _renderScatterSvgHtml(viz) {
    const series = viz.data_series || viz.data || {};
    const xPoints = series.x_points || [];
    const yPoints = series.y_points || [];
    const n = Math.min(xPoints.length, yPoints.length);
    if (n === 0) return `<div class="text-xs text-[#8B95A7] p-4">Sin datos de dispersión.</div>`;

    const minX = Math.min(...xPoints);
    const maxX = Math.max(...xPoints);
    const minY = Math.min(...yPoints);
    const maxY = Math.max(...yPoints);
    const rangeX = Math.max(1e-6, maxX - minX);
    const rangeY = Math.max(1e-6, maxY - minY);

    const svgWidth = 600;
    const svgHeight = 200;
    const padding = 30;
    const chartWidth = svgWidth - padding * 2;
    const chartHeight = svgHeight - padding * 2;
    const pearsonR = series.pearson_r != null ? Number(series.pearson_r).toFixed(3) : "-";

    return `
      <div class="space-y-2 p-3 bg-[#090C12] rounded-lg border border-[#242A36]">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="text-[#F7F8FA] font-medium">${viz.title || "Dispersión"}</span>
          <span class="text-[#53C8FF]">Pearson r = ${pearsonR}</span>
        </div>

        <div class="w-full overflow-x-auto flex justify-center">
          <svg viewBox="0 0 ${svgWidth} ${svgHeight}" class="w-full max-w-2xl h-48">
            <line x1="${padding}" y1="${padding + chartHeight}" x2="${svgWidth - padding}" y2="${padding + chartHeight}" stroke="#384355" />
            <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${padding + chartHeight}" stroke="#384355" />

            <!-- Points -->
            ${xPoints.map((xp, i) => {
              const cx = padding + ((xp - minX) / rangeX) * chartWidth;
              const cy = padding + chartHeight - ((yPoints[i] - minY) / rangeY) * chartHeight;
              return `<circle cx="${cx}" cy="${cy}" r="3" fill="#53C8FF" opacity="0.75" />`;
            }).join("")}
          </svg>
        </div>
      </div>
    `;
  }

  _renderFindingCardHtml(f) {
    const sev = this._classifySeverity(f);
    const sevBadge = sev === "CRITICAL" ? "badge-err" : sev === "WARNING" ? "badge-warn" : "badge-sys";
    const fdrSig = f.metrics && f.metrics.fdr_significant === true;
    const fdrVal = (f.metrics && f.metrics.p_value_fdr != null) ? Number(f.metrics.p_value_fdr) : null;
    const effect = f.effect_size != null
      ? Number(f.effect_size)
      : (f.metrics && (f.metrics.pearson_r ?? f.metrics.cramers_v ?? f.metrics.cohens_d ?? f.metrics.bimodal_coefficient));

    return `
      <div class="p-4 rounded-xl bg-[#090C12] border border-[#242A36] space-y-3 hover:border-[#384355] transition-colors flex flex-col justify-between">
        <div class="space-y-2">
          <div class="flex items-start justify-between gap-2">
            <span class="text-xs font-mono font-bold text-[#F7F8FA]">${escapeHtml(f.finding_type || "HALLAZGO")}</span>
            <div class="flex items-center space-x-1.5">
              ${fdrVal != null && Number.isFinite(fdrVal) ? `
                <span class="${fdrSig ? 'text-[#22C55E] bg-[#22C55E]/10 border-[#22C55E]/30' : 'text-amber-400 bg-amber-400/10 border-amber-400/30'} border text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold">
                  ${fdrSig ? 'FDR p < 0.05' : `FDR p=${formatPValue(fdrVal)}`}
                </span>
              ` : ""}
              <span class="${sevBadge} text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">
                ${escapeHtml(sev)}
              </span>
            </div>
          </div>

          <p class="text-xs text-[#F7F8FA] font-sans leading-relaxed">
            ${escapeHtml(f.summary)}
          </p>

          ${f.limitations && f.limitations.length > 0 ? `
            <div class="text-[11px] font-sans text-amber-400/90 bg-amber-400/5 p-2 rounded border border-amber-400/20">
              <span class="font-semibold">Nota metodológica:</span> ${escapeHtml(f.limitations[0])}
            </div>
          ` : ""}
        </div>

        <div class="pt-2 border-t border-[#1C2230] flex flex-wrap items-center justify-between gap-2">
          <div class="flex items-center gap-1.5">
            ${f.column_name ? `<span class="text-[11px] font-mono text-[#8B95A7] bg-[#161B26] px-2 py-0.5 rounded border border-[#242A36]">${escapeHtml(f.column_name)}</span>` : ""}
            ${f.secondary_column ? `<span class="text-[11px] font-mono text-[#8B95A7] bg-[#161B26] px-2 py-0.5 rounded border border-[#242A36]">${escapeHtml(f.secondary_column)}</span>` : ""}
          </div>

          <div class="flex items-center gap-1.5 ml-auto">
            ${effect != null && Number.isFinite(Number(effect)) ? `
              <span class="text-[11px] font-mono font-semibold text-[#53C8FF] bg-[#53C8FF]/10 px-2 py-0.5 rounded border border-[#53C8FF]/20">
                Efecto = ${Number(effect).toFixed(3)}
              </span>
            ` : ""}
            ${f.p_value != null && Number.isFinite(Number(f.p_value)) ? `
              <span class="text-[11px] font-mono text-[#22C55E] bg-[#22C55E]/10 px-2 py-0.5 rounded border border-[#22C55E]/20">
                p = ${formatPValue(f.p_value)}
              </span>
            ` : ""}
          </div>
        </div>
      </div>
    `;
  }

  _renderHypothesesAndEvidenceHtml() {
    const visibleHyps = this._visibleHypotheses();
    return `
      <div class="workbench-card p-5 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#242A36] pb-3 gap-3">
          <div class="flex items-center space-x-2">
            <span class="text-[#22C55E]">${icon("zap", "icon-sm")}</span>
            <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">Hipótesis de ML y Evidencia Empírica</h4>
            <span class="badge-sys text-[10px] font-mono">${visibleHyps.length} de ${this.hypotheses.length} · ${this.evidence.length} verificadas</span>
          </div>
          <span class="text-[11px] font-mono text-[#8B95A7]">Protocolo Propose ≠ Accept</span>
        </div>

        <p class="text-xs text-[#8B95A7] font-sans">
          Las hipótesis estadísticas son candidatas a experimentos de AutoML. Bajo el principio <em>"Propose ≠ Accept"</em>, una hipótesis solo se acepta si supera empíricamente al baseline en la misma partición exacta de datos.
        </p>

        <!-- Hypotheses Catalog -->
        <div class="space-y-3">
          <div class="flex items-center justify-between">
            <h5 class="text-xs font-mono font-semibold text-[#8B95A7] uppercase tracking-wider">Catálogo de Hipótesis</h5>
            <span class="text-[11px] font-mono text-[#8B95A7]">Mostrando: ${visibleHyps.length} / ${this.hypotheses.length}</span>
          </div>
          ${this.hypotheses.length === 0 ? `
            <div class="p-4 text-center text-[#8B95A7] font-sans text-xs bg-[#090C12] rounded-lg border border-[#242A36]">
              No hay hipótesis formuladas para este estudio.
            </div>
          ` : `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
              ${visibleHyps.map(h => this._renderHypothesisCardHtml(h)).join("")}
            </div>
          `}
        </div>

        <!-- Evidence Links Table -->
        ${this.evidence.length > 0 ? `
          <div class="space-y-3 pt-3 border-t border-[#242A36]">
            <div class="flex items-center justify-between">
              <h5 class="text-xs font-mono font-semibold text-[#8B95A7] uppercase tracking-wider">Enlaces de Evidencia Empírica (EvidenceLink)</h5>
              <span class="text-[11px] font-mono text-[#22C55E] font-medium">${this.evidence.filter(e => e.accepted).length} aceptadas • ${this.evidence.filter(e => !e.accepted).length} rechazadas</span>
            </div>
            <div class="overflow-x-auto border border-[#242A36] rounded-lg">
              <table class="w-full text-xs font-mono border-collapse">
                <thead>
                  <tr class="bg-[#090C12] text-[#8B95A7] border-b border-[#242A36]">
                    <th class="p-2.5 text-left">Link ID</th>
                    <th class="p-2.5 text-left">Hipótesis</th>
                    <th class="p-2.5 text-left">Experimento</th>
                    <th class="p-2.5 text-right">Baseline</th>
                    <th class="p-2.5 text-right">Candidato</th>
                    <th class="p-2.5 text-center">Métrica</th>
                    <th class="p-2.5 text-center">Veredicto</th>
                    <th class="p-2.5 text-left">Detalle / Notas</th>
                    <th class="p-2.5 text-center">Acción</th>
                  </tr>
                </thead>
                <tbody>
                  ${this.evidence.map(ev => `
                    <tr class="border-b border-[#1C2230] hover:bg-[#11151E]">
                      <td class="p-2.5 text-[#53C8FF]">${escapeHtml(ev.id)}</td>
                      <td class="p-2.5 text-[#F7F8FA]">${escapeHtml(ev.hypothesis_id)}</td>
                      <td class="p-2.5 text-[#8B95A7]">${escapeHtml(ev.experiment_id || "-")}</td>
                      <td class="p-2.5 text-right text-[#F7F8FA]">${ev.baseline_score != null ? Number(ev.baseline_score).toFixed(4) : "-"}</td>
                      <td class="p-2.5 text-right text-[#F7F8FA]">${ev.candidate_score != null ? Number(ev.candidate_score).toFixed(4) : "-"}</td>
                      <td class="p-2.5 text-center text-[#8B95A7]">${escapeHtml(ev.metric || "-")}</td>
                      <td class="p-2.5 text-center">
                        <span class="${ev.accepted ? 'text-[#22C55E] bg-[#22C55E]/10 border-[#22C55E]/30' : 'text-[#EF4444] bg-[#EF4444]/10 border-[#EF4444]/30'} border text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">
                          ${ev.accepted ? "Aceptada" : "Rechazada"}
                        </span>
                      </td>
                      <td class="p-2.5 text-left text-[11px] font-sans text-[#8B95A7] max-w-xs truncate" title="${escapeHtml(ev.notes || "")}">
                        ${escapeHtml(ev.notes || "-")}
                      </td>
                      <td class="p-2.5 text-center whitespace-nowrap">
                        ${ev.accepted ? `
                          <button class="btn-create-recipe px-2 py-1 rounded text-[10px] font-mono font-medium bg-[#4F67FF]/20 text-[#53C8FF] hover:bg-[#4F67FF]/30 border border-[#4F67FF]/40 inline-flex items-center gap-1 transition-colors" data-ev-id="${escapeHtml(ev.id)}">
                            ${icon("plus", "icon-xs")} Crear Receta
                          </button>
                        ` : `
                          <span class="text-[#8B95A7]/40 text-[10px]">—</span>
                        `}
                      </td>
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            </div>
          </div>
        ` : ""}
      </div>
    `;
  }

  _renderRecipesSectionHtml() {
    return `
      <div class="workbench-card p-5 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#242A36] pb-3 gap-3">
          <div class="flex items-center space-x-2">
            <span class="text-[#4F67FF]">${icon("recipe", "icon-sm")}</span>
            <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">Recetas de Experimentos Reproducibles</h4>
            <span class="badge-sys text-[10px] font-mono">${this.recipes.length} registradas • ${this.recipes.filter(r => r.status === 'promoted').length} promovidas</span>
          </div>
          <span class="text-[11px] font-mono text-[#8B95A7]">Transforms Fit-in-Fold sin Fuga</span>
        </div>

        <p class="text-xs text-[#8B95A7] font-sans">
          Las <strong>Experiment Recipes</strong> formalizan hipótesis validadas empíricamente en pipelines de producción reproducibles. Incorporan transformaciones fit-in-fold (<code class="text-[#53C8FF]">RobustScaler</code>, <code class="text-[#53C8FF]">PowerTransformer (Yeo-Johnson)</code>, <code class="text-[#53C8FF]">log1p</code>) con trazabilidad estricta a sus evidencias.
        </p>

        ${this.recipes.length === 0 ? `
          <div class="p-6 text-center text-[#8B95A7] font-sans text-xs bg-[#090C12] rounded-lg border border-[#242A36] space-y-2">
            <div class="text-[#8B95A7]/40 flex justify-center">${icon("recipe", "icon-lg")}</div>
            <p>No hay recetas creadas aún para este dataset.</p>
            <p class="text-[11px] text-[#8B95A7]/80">Verifica una hipótesis empírica arriba y pulsa <strong>«Crear Receta»</strong> en los enlaces de evidencia aceptados para formalizar un pipeline reproducible.</p>
          </div>
        ` : `
          <div class="overflow-x-auto border border-[#242A36] rounded-lg">
            <table class="w-full text-xs font-mono border-collapse">
              <thead>
                <tr class="bg-[#090C12] text-[#8B95A7] border-b border-[#242A36]">
                  <th class="p-2.5 text-left">Receta ID</th>
                  <th class="p-2.5 text-left">Nombre</th>
                  <th class="p-2.5 text-center">Estado</th>
                  <th class="p-2.5 text-left">Transformaciones</th>
                  <th class="p-2.5 text-left">Modelos</th>
                  <th class="p-2.5 text-left">Evidencia Origen</th>
                  <th class="p-2.5 text-left">Experimentos</th>
                  <th class="p-2.5 text-right">Acciones</th>
                </tr>
              </thead>
              <tbody>
                ${this.recipes.map(r => {
                  const stageInfo = getTransformationStageInfo(r.status);
                  const transformsList = [
                    ...(r.column_transforms || []).map(t => `${t.column}: ${t.kind}`),
                    ...(r.derived_features || []).map(d => `${d.name || d.expression}`),
                  ];
                  const transformText = transformsList.length > 0 ? transformsList.join(", ") : "Ninguna (Base)";

                  return `
                    <tr class="border-b border-[#1C2230] hover:bg-[#11151E]">
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
                          <button class="btn-select-recipe px-2 py-1 rounded text-[11px] font-mono bg-[#6956E8]/20 text-[#6956E8] hover:bg-[#6956E8]/30 border border-[#6956E8]/40 transition-colors" data-recipe-id="${escapeHtml(r.id)}">Seleccionar</button>
                        ` : ""}
                        ${r.status !== "archived" && r.status !== "promoted" ? `
                          <button class="btn-submit-recipe px-2.5 py-1 rounded text-[11px] font-mono bg-[#4F67FF] text-white hover:bg-[#4F67FF]/80 inline-flex items-center gap-1 transition-colors" data-recipe-id="${escapeHtml(r.id)}">${icon("play", "icon-xs")} Entrenar</button>
                        ` : ""}
                        ${r.status === "submitted" ? `
                          <button class="btn-promote-recipe px-2 py-1 rounded text-[11px] font-mono bg-[#22C55E]/20 text-[#22C55E] hover:bg-[#22C55E]/30 border border-[#22C55E]/40 inline-flex items-center gap-1 transition-colors" data-recipe-id="${escapeHtml(r.id)}">${icon("shield-check", "icon-xs")} Promover</button>
                        ` : ""}
                        ${r.status === "promoted" ? `
                          <span class="text-[11px] font-mono text-[#22C55E] inline-flex items-center gap-1 mr-1.5">${icon("check-circle-2", "icon-xs")} Producción</span>
                        ` : ""}
                        ${r.status !== "archived" ? `
                          <button class="btn-archive-recipe p-1 rounded text-[#8B95A7] hover:text-[#EF4444] hover:bg-[#EF4444]/10 transition-colors" data-recipe-id="${escapeHtml(r.id)}" title="Archivar receta">${icon("x", "icon-xs")}</button>
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

  _renderHypothesisCardHtml(h) {
    const isVerifying = this.verifyingHypothesisId === h.id;
    const statusBadge = h.status === "accepted"
      ? `<span class="badge-gain text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">ACEPTADA</span>`
      : h.status === "rejected"
      ? `<span class="badge-err text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">RECHAZADA</span>`
      : `<span class="badge-warn text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">PROPUESTA</span>`;

    return `
      <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#242A36] space-y-2.5 flex flex-col justify-between">
        <div class="space-y-1.5">
          <div class="flex items-center justify-between gap-2">
            <span class="text-xs font-mono font-bold text-[#53C8FF]">${escapeHtml(h.proposed_action || "acción")}</span>
            ${statusBadge}
          </div>
          <p class="text-xs text-[#F7F8FA] font-sans leading-relaxed">
            ${escapeHtml(h.description)}
          </p>
        </div>

        <div class="pt-2 border-t border-[#1C2230] flex items-center justify-between gap-2">
          <span class="text-[10px] font-mono text-[#8B95A7]">${escapeHtml(h.id)}</span>
          ${h.status === "proposed" ? `
            <button class="btn-verify-hyp btn-signal text-[11px] py-1 px-2.5 font-mono inline-flex items-center space-x-1" data-hyp-id="${escapeHtml(h.id)}" ${isVerifying ? "disabled" : ""}>
              ${isVerifying ? `<span class="animate-spin mr-1">${icon("refresh-cw", "icon-xs")}</span>` : `${icon("play", "icon-xs")}`}
              <span>${isVerifying ? "Verificando..." : "Verificar"}</span>
            </button>
          ` : `
            <span class="text-[11px] font-mono text-[#8B95A7] flex items-center space-x-1">
              ${icon("check-circle-2", "icon-xs")}
              <span>Evaluada</span>
            </span>
          `}
        </div>
      </div>
    `;
  }

  _renderCreateStudyModalHtml() {
    return `
      <div class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="workbench-card max-w-md w-full p-6 space-y-5 border border-[#4F67FF]/30 shadow-2xl">
          <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
            <div class="flex items-center space-x-2">
              <span class="text-[#4F67FF]">${icon("compass", "icon-sm")}</span>
              <h3 class="text-sm font-bold font-sans text-[#F7F8FA]">Nuevo Estudio Exploratorio</h3>
            </div>
            <button id="btnCloseStudyModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1">${icon("x", "icon-sm")}</button>
          </div>

          <div class="space-y-4 text-xs font-sans">
            <div class="space-y-1.5">
              <label class="font-medium text-[#F7F8FA]">Nombre del Estudio</label>
              <input id="inputStudyName" type="text" class="input-base bg-[#161B26] border border-[#242A36] text-[#F7F8FA] w-full text-xs rounded-lg px-3 py-2 outline-none focus:border-[#4F67FF]" value="Estudio de Diagnóstico Completo" />
            </div>

            <div class="space-y-1.5">
              <label class="font-medium text-[#F7F8FA]">Columna Objetivo (Opcional)</label>
              <input id="inputTargetCol" type="text" class="input-base bg-[#161B26] border border-[#242A36] text-[#F7F8FA] w-full text-xs font-mono rounded-lg px-3 py-2 outline-none focus:border-[#4F67FF]" placeholder="Dejar vacío para modo no supervisado" />
              <p class="text-[11px] text-[#8B95A7]">Si se especifica, se ejecutarán pruebas inferenciales contra el target (ANOVA / Welch t-test).</p>
            </div>
          </div>

          <div class="flex items-center justify-end space-x-2 border-t border-[#242A36] pt-4">
            <button id="btnCancelStudyModal" class="btn-technical text-xs">Cancelar</button>
            <button id="btnConfirmCreateStudy" class="btn-signal text-xs flex items-center space-x-1.5 shadow-md shadow-[#4F67FF]/20">
              <span>Crear y Configurar</span>
            </button>
          </div>
        </div>
      </div>
    `;
  }

  _bindEvents() {
    document.getElementById("btnExploreModeGuided")?.addEventListener("click", () => {
      this.showAdvancedExplore = false;
      this.render();
    });
    document.getElementById("btnExploreModeAdvanced")?.addEventListener("click", () => {
      this.showAdvancedExplore = true;
      this.render();
    });
    document.getElementById("btnShowAllFindings")?.addEventListener("click", () => {
      this.showAdvancedExplore = true;
      this.render();
    });
    document.getElementById("inputMatrixColumnFilter")?.addEventListener("input", (e) => {
      this.matrixColumnFilter = e.target.value || "";
      // Debounced re-render would be nicer; immediate is OK for filter
      clearTimeout(this._matrixFilterTimer);
      this._matrixFilterTimer = setTimeout(() => this.render(), 200);
    });

    // Select active study
    document.getElementById("selectActiveStudy")?.addEventListener("change", async (e) => {
      this.selectedStudyId = e.target.value;
      this.renderLoading();
      await this.fetchData();
      this.render();
    });

    // Run study execution
    document.getElementById("btnRunStudyExecution")?.addEventListener("click", async () => {
      if (!this.selectedStudyId || this.runningStudy) return;
      this.runningStudy = true;
      this.render();
      try {
        await api.runAnalysisStudy(this.selectedStudyId);
        await this.fetchData();
      } catch (err) {
        console.error("Run study error:", err);
      } finally {
        this.runningStudy = false;
        this.render();
      }
    });

    // Modal open / close
    document.getElementById("btnOpenNewStudyModal")?.addEventListener("click", () => {
      this.showCreateModal = true;
      this.render();
    });
    document.getElementById("btnCreateInitialStudy")?.addEventListener("click", () => {
      this.showCreateModal = true;
      this.render();
    });
    document.getElementById("btnCloseStudyModal")?.addEventListener("click", () => {
      this.showCreateModal = false;
      this.render();
    });
    document.getElementById("btnCancelStudyModal")?.addEventListener("click", () => {
      this.showCreateModal = false;
      this.render();
    });

    // Confirm create study
    document.getElementById("btnConfirmCreateStudy")?.addEventListener("click", async () => {
      const name = document.getElementById("inputStudyName")?.value || "Estudio Exploratorio";
      const target = document.getElementById("inputTargetCol")?.value.trim() || null;
      try {
        const res = await api.createAnalysisStudy({
          dataset_id: this.activeDatasetId,
          name,
          target_column: target,
        });
        this.showCreateModal = false;
        if (res && res.study_id) {
          this.selectedStudyId = res.study_id;
        }
        this.renderLoading();
        await this.fetchData();
        this.render();
      } catch (err) {
        console.error("Create study error:", err);
      }
    });

    // Filters
    document.getElementById("filterCategorySelect")?.addEventListener("change", (e) => {
      this.filterCategory = e.target.value;
      this.render();
    });
    document.getElementById("filterSeveritySelect")?.addEventListener("change", (e) => {
      this.filterSeverity = e.target.value;
      this.render();
    });
    document.getElementById("checkFdrOnly")?.addEventListener("change", (e) => {
      this.filterFdrOnly = e.target.checked;
      this.render();
    });

    // Viz tab switches
    document.querySelectorAll(".btn-viz-tab").forEach(btn => {
      btn.addEventListener("click", (e) => {
        this.activeVizType = e.currentTarget.getAttribute("data-viz-type");
        this.render();
      });
    });

    // Matrix method toggle
    document.querySelectorAll(".btn-matrix-toggle").forEach(btn => {
      btn.addEventListener("click", (e) => {
        this.matrixMethod = e.currentTarget.getAttribute("data-method");
        const expand = document.getElementById("guidedCorrExpand");
        this.guidedCorrExpanded = Boolean(expand?.open);
        this.render();
      });
    });
    document.getElementById("guidedCorrExpand")?.addEventListener("toggle", (e) => {
      this.guidedCorrExpanded = Boolean(e.currentTarget.open);
    });

    // Export report
    document.getElementById("btnExportStudyReport")?.addEventListener("click", () => {
      this._downloadMarkdownReport();
    });

    // Verify hypothesis action
    document.querySelectorAll(".btn-verify-hyp").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const hypId = e.currentTarget.getAttribute("data-hyp-id");
        if (!hypId || this.verifyingHypothesisId) return;
        this.verifyingHypothesisId = hypId;
        this.render();
        try {
          await api.verifyHypothesis(hypId, null, 0.0);
          await this.fetchData();
        } catch (err) {
          console.error("Error verifying hypothesis:", err);
          alert("Error verificando hipótesis: " + (err.message || err));
        } finally {
          this.verifyingHypothesisId = null;
          this.render();
        }
      });
    });

    // Recipe: Create from Evidence
    document.querySelectorAll(".btn-create-recipe").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const evId = e.currentTarget.getAttribute("data-ev-id");
        if (!evId) return;
        const name = prompt("Nombre descriptivo para la receta:", `Receta desde ${evId}`);
        if (!name) return;
        try {
          await api.createRecipeFromEvidence(evId, name);
          await this.fetchData();
          this.render();
        } catch (err) {
          console.error("Error creating recipe:", err);
          alert("Error creando receta: " + (err.message || err));
        }
      });
    });

    // Recipe: Select
    document.querySelectorAll(".btn-select-recipe").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const recipeId = e.currentTarget.getAttribute("data-recipe-id");
        if (!recipeId) return;
        try {
          await api.selectRecipe(recipeId);
          await this.fetchData();
          this.render();
        } catch (err) {
          console.error("Error selecting recipe:", err);
          alert("Error seleccionando receta: " + (err.message || err));
        }
      });
    });

    // Recipe: Submit (Train)
    document.querySelectorAll(".btn-submit-recipe").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const recipeId = e.currentTarget.getAttribute("data-recipe-id");
        if (!recipeId) return;
        const btnEl = e.currentTarget;
        btnEl.disabled = true;
        btnEl.innerHTML = `<span class="animate-spin mr-1">${icon("refresh-cw", "icon-xs")}</span> Entrenando...`;
        try {
          const res = await api.submitRecipe(recipeId);
          alert(`Experimento(s) ejecutados con éxito: ${(res.experiment_ids || []).join(", ")}`);
          await this.fetchData();
          this.render();
        } catch (err) {
          console.error("Error submitting recipe:", err);
          alert("Error ejecutando receta: " + (err.message || err));
          btnEl.disabled = false;
        }
      });
    });

    // Recipe: Promote
    document.querySelectorAll(".btn-promote-recipe").forEach(btn => {
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
          console.error("Error promoting recipe:", err);
          alert("Error promoviendo receta: " + (err.message || err));
        }
      });
    });

    // Recipe: Archive
    document.querySelectorAll(".btn-archive-recipe").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const recipeId = e.currentTarget.getAttribute("data-recipe-id");
        if (!recipeId) return;
        if (!confirm(`¿Desea archivar la receta ${recipeId}?`)) return;
        try {
          await api.archiveRecipe(recipeId);
          await this.fetchData();
          this.render();
        } catch (err) {
          console.error("Error archiving recipe:", err);
          alert("Error archivando receta: " + (err.message || err));
        }
      });
    });
  }

  _downloadMarkdownReport() {
    if (!this.studyDetail) return;
    const s = this.studyDetail;
    let md = `# Informe de Exploración Estadística · CATML Explore\n\n`;
    md += `**Estudio:** ${s.name} (\`${s.id}\`)\n`;
    md += `**Dataset:** ${s.data_source?.dataset_id || s.dataset_id}\n`;
    md += `**Variable Objetivo:** ${s.target_column || "Ninguna (No supervisado)"}\n`;
    md += `**Estado:** ${s.status}\n\n`;
    md += `## Hallazgos Estadísticos (${this.findings.length} detectados)\n\n`;

    this.findings.forEach((f, i) => {
      const fdrStr = f.metrics?.p_value_fdr != null ? ` (FDR p=${f.metrics.p_value_fdr})` : "";
      md += `### ${i + 1}. [${f.finding_type.toUpperCase()}] ${f.summary}\n`;
      md += `- **Variables:** \`${f.column_name || "-"}\` ${f.secondary_column ? " / `" + f.secondary_column + "`" : ""}\n`;
      md += `- **Método:** ${f.method_name}\n`;
      if (f.p_value != null) md += `- **P-valor crudo:** ${f.p_value}${fdrStr}\n`;
      if (f.effect_size != null) md += `- **Tamaño del efecto:** ${f.effect_size}\n`;
      md += `\n`;
    });

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `informe_explore_${s.id}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}
