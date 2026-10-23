/**
 * DatasetsView — Dataset Understanding & Interactive Feature Selection
 * Orchestrator; tab/panel renderers live under ./ (Clean Architecture PR3).
 */
import { store } from "../../store.js";
import { bus } from "../../bus.js";
import { api } from "../../api.js";
import { icon } from "../../icons.js";
import { escapeHtml } from "../../utils.js";

import { _renderHealthCard } from "./data_quality_panel.js";
import { _renderResumenTab } from "./dataset_summary.js";
import {
  _renderColumnasTab,
  _renderActiveTabContent,
  _renderSchemaTab,
  _renderStatsTab,
  _renderPreviewTab,
  _renderCategoriesTab,
  _renderVariableModal,
  _renderVarModalContent,
  _renderImageGalleryContent,
  _renderBoxPlotContent,
  _renderHistogramContent,
  _renderTargetPatternContent,
} from "./dataset_columns.js";
import {
  _renderOptionCard,
  _getCompatibleToolsSummary,
  _renderQuestionnaireTab,
  _bindQuestionnaireEvents,
} from "./questionnaire.js";
import { _renderCorrelationTab } from "./correlations_view.js";
import { _renderCalculatorTab, _bindCalculatorEvents } from "./derived_features.js";
import {
  _getFilteredColumns,
  _updateSelectionUi,
  _initSelectedFeatures,
} from "./feature_selector.js";

export class DatasetsView {
  constructor() {
    this.container = null;
    this.profile = null;
    this.datasets = [];
    this.activeDatasetId = null;
    this.showRegisterForm = false;
    this.selectedFeatures = new Set();
    this.activeMainTab = "resumen"; // 'resumen' | 'columnas' | 'cuestionario'
    this.activeTab = "schema"; // 'schema' | 'stats' | 'preview' | 'categories' | 'correlation'
    this.filterType = "all"; // 'all' | 'numeric' | 'categorical' | 'selected' | 'excluded'
    this.searchQuery = "";
    this.activeModalVar = null;
    this.activeModalTab = "boxplot"; // 'boxplot' | 'histogram' | 'pattern'
    this.calcMode = "formula"; // 'formula' | 'python_code'
    this.calcFeatureName = "";
    this.calcExpression = "";
    this.calcEvaluationResult = null;
    this.calcSuggestions = [];
    this.calcEvaluating = false;
    this.calcApplying = false;

    // Framing Questionnaire state (Phase 2)
    this.questionnaire = null;
    this.questionnaireGenerating = false;
    this.questionnaireSaving = false;
  }

  async mount(container) {
    this.container = container;
    this._unregisterBus = [
      bus.on("dataset:show-register", () => {
        bus.emit("modal:register-dataset");
      }),
      bus.on("dataset:switched", async (id) => {
        this.activeDatasetId = id;
        this.renderLoading();
        await this.fetchData();
        this._initSelectedFeatures();
        this.render();
      }),
    ];
    this.renderLoading();
    await this.fetchData();
    this._initSelectedFeatures();
    this.render();
  }

  destroy() {
    if (this._unregisterBus) {
      this._unregisterBus.forEach(fn => fn && fn());
      this._unregisterBus = [];
    }
  }

  renderLoading() {
    this.container.innerHTML = `
      <div class="workbench-card p-12 text-center text-slate-400 space-y-3">
        <div class="animate-spin text-[#4F67FF] inline-block">${icon("refresh-cw", "icon-lg")}</div>
        <div class="text-sm font-medium">Analyzing dataset profile and CATML inferences...</div>
      </div>
    `;
  }

  async fetchData() {
    try {
      this.datasets = await api.getDatasets().catch(() => []);

      this.activeDatasetId = store.getActiveDatasetId()
        || (this.datasets.length ? this.datasets[0].id : null);
      if (this.activeDatasetId && !store.getActiveDatasetId()) {
        store.setActiveDataset(this.activeDatasetId);
      }

      if (this.activeDatasetId) {
        this.profile = await api.getDatasetProfile(this.activeDatasetId).catch(() => null);
        this.questionnaire = await api.getDatasetQuestionnaire(this.activeDatasetId).catch(() => null);

        // Guided path: auto-suggest framing and open questionnaire when still unconfirmed.
        if (this.profile && (!this.questionnaire || !this.questionnaire.user_confirmed)) {
          if (!this.questionnaire) {
            this.questionnaireGenerating = true;
            try {
              const res = await api.generateDatasetQuestionnaire(this.activeDatasetId);
              if (res && res.questionnaire) {
                this.questionnaire = res.questionnaire;
              }
            } catch (genErr) {
              console.warn("Could not auto-generate questionnaire:", genErr);
            } finally {
              this.questionnaireGenerating = false;
            }
          }
          this._questionnaireGuidedSeen = this._questionnaireGuidedSeen || {};
          if (!this._questionnaireGuidedSeen[this.activeDatasetId]) {
            this.activeMainTab = "cuestionario";
            this._questionnaireGuidedSeen[this.activeDatasetId] = true;
          }
        }
      }
    } catch (e) {
      console.warn("Could not load dataset profile:", e);
    }
  }

  render() {
    if (!this.profile && this.datasets.length === 0) {
      this._renderEmptyOrRegisterState();
      return;
    }

    const p = this.profile;
    if (!p) {
      this._renderEmptyOrRegisterState();
      return;
    }

    const columns = p.columns || [];
    const featureCols = columns.filter(c => c.name !== p.target_column);
    const numCols = columns.filter(c => c.dtype && (c.dtype.includes("int") || c.dtype.includes("float")) && c.name !== p.target_column).length;
    const catCols = columns.filter(c => c.dtype && (c.dtype === "object" || c.dtype === "category" || c.dtype === "string") && !c.is_image && c.name !== p.target_column).length;
    const excludedCount = columns.filter(c => c.is_identifier || c.catml_action === "Exclude").length;
    const currentDs = this.datasets.find(d => d.id === this.activeDatasetId);
    const datasetName = currentDs ? currentDs.name : (p.name || p.dataset_id);

    const recommendations = p.recommendations || [];

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Header del Dataset Activo con Selector y Pestañas Clave -->
        <div class="workbench-card p-5 space-y-4">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div class="space-y-1">
              <div class="flex items-center space-x-2">
                <span class="text-[10px] font-mono text-[#4F67FF] bg-[#4F67FF]/10 px-2 py-0.5 rounded border border-[#4F67FF]/25 uppercase font-semibold">Dataset Activo</span>
                <span class="text-xs font-mono text-[#8B95A7]">${escapeHtml(currentDs?.path || p.dataset_id)}</span>
              </div>
              <h2 class="text-xl font-bold font-sans text-[#F7F8FA]">${escapeHtml(datasetName)}</h2>
            </div>

            <div class="flex items-center space-x-3">
              ${this.datasets.length > 1 ? `
                <select id="datasetSelect" class="bg-[#161B26] border border-[#242A36] text-[#F7F8FA] text-xs rounded-lg px-3 py-1.5 focus:border-[#4F67FF] font-mono outline-none">
                  ${this.datasets.map(d => `
                    <option value="${d.id}" ${d.id === this.activeDatasetId ? "selected" : ""}>${escapeHtml(d.name)}</option>
                  `).join("")}
                </select>
              ` : `
                <button id="btnSwitchDatasetFromDS" class="bg-[#161B26] hover:bg-[#1E2536] text-[#8B95A7] hover:text-[#F7F8FA] border border-[#242A36] text-xs px-3 py-1.5 rounded-lg font-sans flex items-center space-x-1.5 transition-colors">
                  ${icon("layers", "icon-sm")}
                  <span>Cambiar</span>
                </button>
              `}

              <button id="btnToggleRegisterForm" class="btn-technical text-xs">
                ${this.showRegisterForm ? "Cerrar Registro" : "+ Registrar nuevo"}
              </button>

              <button id="btnOpenFullExploreCanvas" class="bg-[#161B26] hover:bg-[#1E2536] text-[#53C8FF] border border-[#242A36] hover:border-[#53C8FF]/40 text-xs px-3 py-1.5 rounded-lg font-sans font-medium flex items-center space-x-1.5 transition-colors" title="Abrir análisis estadístico y visualizaciones en 03 Explore">
                ${icon("compass", "icon-sm")}
                <span>Lienzo Completo (Explore)</span>
              </button>

              <button id="btnNewExperimentFromDS" class="btn-signal text-xs flex items-center space-x-1.5 shadow-md shadow-[#4F67FF]/20">
                ${icon("play", "icon-sm")}
                <span>Nuevo experimento</span>
              </button>
            </div>
          </div>

          <!-- Pestañas Principales: Resumen | Columnas | Encuadre & Cuestionario -->
          <div class="border-t border-[#242A36] pt-3 flex items-center space-x-2 text-xs font-sans">
            <button class="main-tab-btn px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2 ${this.activeMainTab === 'resumen' ? 'bg-[#161B26] text-[#F7F8FA] border border-[#242A36]' : 'text-[#8B95A7] hover:text-[#F7F8FA]'}" data-main-tab="resumen">
              ${icon("activity", "icon-sm")}
              <span>Resumen</span>
            </button>
            <button class="main-tab-btn px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2 ${this.activeMainTab === 'columnas' ? 'bg-[#161B26] text-[#F7F8FA] border border-[#242A36]' : 'text-[#8B95A7] hover:text-[#F7F8FA]'}" data-main-tab="columnas">
              ${icon("table", "icon-sm")}
              <span>Columnas</span>
              <span class="text-[10px] font-mono text-[#8B95A7] bg-[#090C12] px-1.5 py-0.2 rounded">${columns.length}</span>
            </button>
            <button class="main-tab-btn px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2 ${this.activeMainTab === 'cuestionario' ? 'bg-[#161B26] text-[#F7F8FA] border border-[#242A36]' : 'text-[#8B95A7] hover:text-[#F7F8FA]'}" data-main-tab="cuestionario">
              ${icon("help-circle", "icon-sm")}
              <span>Encuadre & Cuestionario</span>
              ${this.questionnaire?.user_confirmed ? `
                <span class="text-[10px] font-mono text-[#22C55E] bg-[#22C55E]/10 px-1.5 py-0.5 rounded border border-[#22C55E]/20 font-semibold">Confirmado</span>
              ` : (this.questionnaire ? `
                <span class="text-[10px] font-mono text-[#F59E0B] bg-[#F59E0B]/10 px-1.5 py-0.5 rounded border border-[#F59E0B]/20 font-semibold">Sugerencias</span>
              ` : `
                <span class="text-[10px] font-mono text-[#8B95A7] bg-[#090C12] px-1.5 py-0.5 rounded">Pendiente</span>
              `)}
            </button>
          </div>
        </div>

        ${this.showRegisterForm ? this._getRegisterFormHtml() : ""}

        <!-- Contenido según la pestaña activa -->
        ${this.activeMainTab === "resumen" ? this._renderResumenTab(p, featureCols, numCols, catCols, excludedCount, recommendations) : ""}
        ${this.activeMainTab === "columnas" ? this._renderColumnasTab(p, featureCols, numCols, catCols) : ""}
        ${this.activeMainTab === "cuestionario" ? this._renderQuestionnaireTab(p) : ""}

        <!-- Variable Visual Analytics Modal Container -->
        <div id="variableModalContainer"></div>
      </div>
    `;

    this._bindEvents();
  }

  _renderAndMountModal() {
    const modalContainer = this.container.querySelector("#variableModalContainer");
    if (!modalContainer) return;
    modalContainer.innerHTML = this._renderVariableModal();

    modalContainer.querySelector("#btnCloseVarModal")?.addEventListener("click", () => {
      this.activeModalVar = null;
      modalContainer.innerHTML = "";
    });

    modalContainer.querySelectorAll(".var-tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        this.activeModalTab = btn.getAttribute("data-modal-tab");
        this._renderAndMountModal();
      });
    });
  }

  _renderEmptyOrRegisterState() {
    this.container.innerHTML = `
      <div class="space-y-6">
        <div class="workbench-card p-8 text-center space-y-4">
          <div class="flex justify-center text-slate-600">${icon("database", "icon-xl", 36)}</div>
          <div>
            <h3 class="text-base font-bold text-slate-200">No Profiled Datasets Found</h3>
            <p class="text-xs text-slate-400 mt-1">Register a tabular CSV or Parquet dataset to generate autonomous statistical profiles and candidate pipelines.</p>
          </div>
        </div>
        ${this._getRegisterFormHtml()}
      </div>
    `;
    this._bindRegisterFormEvents();
  }

  _getRegisterFormHtml() {
    return `
      <div class="workbench-card p-6 space-y-4 border-indigo-900/60 bg-indigo-950/20" id="registerFormCard">
        <div class="flex items-center justify-between border-b border-slate-800 pb-3">
          <div class="flex items-center space-x-2">
            <span class="text-indigo-400">${icon("plus", "icon-sm")}</span>
            <span class="text-sm font-semibold text-slate-200">Register New Dataset</span>
          </div>
          ${this.profile ? `<button id="btnCloseRegisterForm" class="text-slate-400 hover:text-slate-200 text-xs inline-flex items-center space-x-1">${icon("x", "icon-sm")}<span>Close</span></button>` : ""}
        </div>

        <form id="formRegisterDataset" class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div>
            <label class="block text-slate-300 font-medium mb-1">Dataset Name *</label>
            <input type="text" id="regName" required placeholder="e.g. My Tabular Dataset" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-100 focus:border-indigo-500">
          </div>

          <div>
            <label class="block text-slate-300 font-medium mb-1">File Path (CSV or Parquet) *</label>
            <input type="text" id="regPath" required placeholder="e.g. data/train.csv" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-100 focus:border-indigo-500 font-mono">
          </div>

          <div>
            <label class="block text-slate-300 font-medium mb-1">Target Column *</label>
            <input type="text" id="regTarget" required placeholder="e.g. target" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-100 focus:border-indigo-500 font-mono">
          </div>

          <div>
            <label class="block text-slate-300 font-medium mb-1">Task Type</label>
            <select id="regTaskType" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-100 focus:border-indigo-500">
              <option value="binary_classification">Binary Classification</option>
              <option value="multiclass_classification">Multiclass Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>

          <div class="md:col-span-2 flex justify-end pt-2">
            <button type="submit" id="btnSubmitRegister" class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-4 py-2 rounded-lg transition-colors shadow-lg shadow-indigo-600/20">
              Register & Profile Dataset
            </button>
          </div>
        </form>
      </div>
    `;
  }

  _bindEvents() {
    this.container.querySelectorAll(".main-tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const tab = btn.getAttribute("data-main-tab");
        if (tab && tab !== this.activeMainTab) {
          this.activeMainTab = tab;
          this.render();
        }
      });
    });

    this.container.querySelector("#btnSwitchDatasetFromDS")?.addEventListener("click", () => {
      bus.emit("modal:switch-dataset");
    });

    this.container.querySelector("#btnNewExperimentFromDS")?.addEventListener("click", () => {
      const feats = Array.from(this.selectedFeatures);
      const currentMap = store.getState().selectedFeaturesByDataset || {};
      store.setState({
        customFeatures: feats,
        selectedFeatures: feats,
        selectedFeaturesByDataset: { ...currentMap, [this.activeDatasetId]: feats },
      });
      bus.emit("modal:new-experiment");
    });

    this.container.querySelector("#btnLaunchWithSelection")?.addEventListener("click", () => {
      const feats = Array.from(this.selectedFeatures);
      const currentMap = store.getState().selectedFeaturesByDataset || {};
      store.setState({
        customFeatures: feats,
        selectedFeatures: feats,
        selectedFeaturesByDataset: { ...currentMap, [this.activeDatasetId]: feats },
      });
      bus.emit("modal:new-experiment");
    });

    this.container.querySelector("#btnToggleRegisterForm")?.addEventListener("click", () => {
      bus.emit("modal:register-dataset");
    });

    this.container.querySelector("#datasetSelect")?.addEventListener("change", async e => {
      const newId = e.target.value;
      if (newId && newId !== this.activeDatasetId) {
        this.activeDatasetId = newId;
        store.setActiveDataset(newId);
        this.renderLoading();
        this.profile = await api.getDatasetProfile(newId).catch(() => null);
        this.questionnaire = await api.getDatasetQuestionnaire(newId).catch(() => null);
        this._initSelectedFeatures();
        this.render();
      }
    });

    // Recommendations Application
    this.container.querySelector("#btnApplyRecommendations")?.addEventListener("click", () => {
      if (!this.profile) return;
      const target = this.profile.target_column;
      const excludedCols = new Set(
        (this.profile.recommendations || [])
          .filter(r => r.action === "exclude")
          .map(r => r.column)
      );

      this.selectedFeatures.clear();
      (this.profile.columns || []).forEach(c => {
        if (c.name !== target && !c.is_identifier && !excludedCols.has(c.name)) {
          this.selectedFeatures.add(c.name);
        }
      });
      this._updateSelectionUi();
    });

    // Batch Selection Buttons
    this.container.querySelector("#btnSelectAll")?.addEventListener("click", () => {
      if (!this.profile) return;
      const target = this.profile.target_column;
      (this.profile.columns || []).forEach(c => {
        if (c.name !== target) this.selectedFeatures.add(c.name);
      });
      this._updateSelectionUi();
    });

    this.container.querySelector("#btnDeselectAll")?.addEventListener("click", () => {
      this.selectedFeatures.clear();
      this._updateSelectionUi();
    });

    this.container.querySelector("#btnSelectTop5")?.addEventListener("click", () => {
      if (!this.profile) return;
      const target = this.profile.target_column;
      const sorted = (this.profile.columns || [])
        .filter(c => c.name !== target && c.target_correlation != null)
        .sort((a, b) => Math.abs(b.target_correlation) - Math.abs(a.target_correlation));

      this.selectedFeatures.clear();
      sorted.slice(0, 5).forEach(c => this.selectedFeatures.add(c.name));
      this._updateSelectionUi();
    });

    this.container.querySelector("#btnSelectTop10")?.addEventListener("click", () => {
      if (!this.profile) return;
      const target = this.profile.target_column;
      const sorted = (this.profile.columns || [])
        .filter(c => c.name !== target && c.target_correlation != null)
        .sort((a, b) => Math.abs(b.target_correlation) - Math.abs(a.target_correlation));

      this.selectedFeatures.clear();
      sorted.slice(0, 10).forEach(c => this.selectedFeatures.add(c.name));
      this._updateSelectionUi();
    });

    // Checkbox toggles in table
    this.container.querySelectorAll(".col-toggle-checkbox").forEach(cb => {
      cb.addEventListener("change", e => {
        const col = e.target.getAttribute("data-col");
        if (e.target.checked) {
          this.selectedFeatures.add(col);
        } else {
          this.selectedFeatures.delete(col);
        }
        this._updateSelectionUi();
      });
    });

    // Master row checkbox
    this.container.querySelector("#checkSelectAllRows")?.addEventListener("change", e => {
      const checked = e.target.checked;
      const filtered = this._getFilteredColumns(this.profile?.columns || []);
      filtered.forEach(c => {
        if (c.name !== this.profile?.target_column) {
          if (checked) {
            this.selectedFeatures.add(c.name);
          } else {
            this.selectedFeatures.delete(c.name);
          }
        }
      });
      this._updateSelectionUi();
    });

    // Search input
    this.container.querySelector("#featureSearchInput")?.addEventListener("input", e => {
      this.searchQuery = e.target.value.trim();
      const tabContent = this.container.querySelector("#tabContentContainer");
      if (tabContent && this.profile) {
        tabContent.innerHTML = this._renderActiveTabContent(this.profile);
        this._bindTabSpecificEvents();
      }
    });

    // Filter pills
    this.container.querySelectorAll(".segmented-pill, .filter-pill").forEach(btn => {
      btn.addEventListener("click", () => {
        this.filterType = btn.getAttribute("data-filter");
        this.render();
      });
    });

    // View tabs
    this.container.querySelectorAll(".view-tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        this.activeTab = btn.getAttribute("data-tab");
        this.render();
      });
    });

    this._bindRegisterFormEvents();
    this._bindTabSpecificEvents();
    this._bindCalculatorEvents();

    const openExploreHandler = () => {
      if (this.activeDatasetId) {
        store.setState({ activeDatasetId: this.activeDatasetId });
      }
      store.setNav("explore");
    };
    this.container.querySelector("#btnOpenFullExploreCanvas")?.addEventListener("click", openExploreHandler);
    this.container.querySelector("#btnOpenExploreFromDS")?.addEventListener("click", openExploreHandler);
  }

  _bindTabSpecificEvents() {
    this.container.querySelectorAll(".col-toggle-checkbox").forEach(cb => {
      cb.addEventListener("change", e => {
        const col = e.target.getAttribute("data-col");
        if (e.target.checked) {
          this.selectedFeatures.add(col);
        } else {
          this.selectedFeatures.delete(col);
        }
        this._updateSelectionUi();
      });
    });

    this.container.querySelectorAll(".btn-visualize-var").forEach(btn => {
      btn.addEventListener("click", () => {
        const varName = btn.getAttribute("data-var");
        const col = this.profile?.columns.find(c => c.name === varName);
        this.activeModalVar = varName;
        this.activeModalTab = col && col.is_image ? "gallery" : "boxplot";
        this._renderAndMountModal();
      });
    });

    this._bindQuestionnaireEvents();
  }

  _bindRegisterFormEvents() {
    this.container.querySelector("#btnCloseRegisterForm")?.addEventListener("click", () => {
      this.showRegisterForm = false;
      this.render();
    });

    this.container.querySelector("#formRegisterDataset")?.addEventListener("submit", async e => {
      e.preventDefault();
      const btn = this.container.querySelector("#btnSubmitRegister");
      if (btn) {
        btn.disabled = true;
        btn.textContent = "Profiling...";
      }

      try {
        const payload = {
          name: this.container.querySelector("#regName").value.trim(),
          path: this.container.querySelector("#regPath").value.trim(),
          target_column: this.container.querySelector("#regTarget").value.trim(),
          task_type: this.container.querySelector("#regTaskType").value,
        };

        const res = await api.registerDataset(payload);
        this.showRegisterForm = false;
        store.setState({ activeDatasetId: res.dataset_id });
        this.renderLoading();
        await this.fetchData();
        this._initSelectedFeatures();
        this.render();
      } catch (err) {
        alert("Failed to register dataset: " + err.message);
        if (btn) {
          btn.disabled = false;
          btn.textContent = "Register & Profile Dataset";
        }
      }
    });
  }

  _renderHealthCard(p) {
    return _renderHealthCard(this, p);
  }

  _renderResumenTab(p, featureCols, numCols, catCols, excludedCount, recommendations) {
    return _renderResumenTab(this, p, featureCols, numCols, catCols, excludedCount, recommendations);
  }

  _renderColumnasTab(p, featureCols, numCols, catCols) {
    return _renderColumnasTab(this, p, featureCols, numCols, catCols);
  }

  _renderActiveTabContent(p) {
    return _renderActiveTabContent(this, p);
  }

  _renderSchemaTab(p) {
    return _renderSchemaTab(this, p);
  }

  _renderStatsTab(p) {
    return _renderStatsTab(this, p);
  }

  _renderPreviewTab(p) {
    return _renderPreviewTab(this, p);
  }

  _renderCategoriesTab(p) {
    return _renderCategoriesTab(this, p);
  }

  _renderVariableModal() {
    return _renderVariableModal(this);
  }

  _renderVarModalContent(col) {
    return _renderVarModalContent(this, col);
  }

  _renderImageGalleryContent(col) {
    return _renderImageGalleryContent(this, col);
  }

  _renderBoxPlotContent(col) {
    return _renderBoxPlotContent(this, col);
  }

  _renderHistogramContent(col) {
    return _renderHistogramContent(this, col);
  }

  _renderTargetPatternContent(col) {
    return _renderTargetPatternContent(this, col);
  }

  _renderOptionCard(field, val, currentVal, title, desc, tag = "") {
    return _renderOptionCard(this, field, val, currentVal, title, desc, tag);
  }

  _getCompatibleToolsSummary(q, p) {
    return _getCompatibleToolsSummary(this, q, p);
  }

  _renderQuestionnaireTab(p) {
    return _renderQuestionnaireTab(this, p);
  }

  _bindQuestionnaireEvents() {
    return _bindQuestionnaireEvents(this);
  }

  _renderCorrelationTab(p) {
    return _renderCorrelationTab(this, p);
  }

  _renderCalculatorTab(p) {
    return _renderCalculatorTab(this, p);
  }

  _bindCalculatorEvents() {
    return _bindCalculatorEvents(this);
  }

  _getFilteredColumns(columns) {
    return _getFilteredColumns(this, columns);
  }

  _updateSelectionUi() {
    return _updateSelectionUi(this);
  }

  _initSelectedFeatures() {
    return _initSelectedFeatures(this);
  }

}
