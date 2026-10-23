/**
 * SwitchDatasetModal — Quick Active Dataset Switcher
 * Allows switching the active workspace dataset across views.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { escapeHtml } from "../utils.js";

export class SwitchDatasetModal {
  constructor() {
    this.container = null;
  }

  async mount(container) {
    this.container = container;
    this.renderLoading();
    try {
      const datasets = await api.getDatasets().catch(() => []);
      store.setState({ datasets });
      this.render(datasets);
    } catch (_) {
      this.render(store.getState().datasets || []);
    }
  }

  renderLoading() {
    this.container.innerHTML = `
      <div class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="workbench-card max-w-lg w-full p-6 text-center text-[#8B95A7] space-y-3">
          <div class="animate-spin text-[#4F67FF] inline-block">${icon("refresh-cw", "icon-lg")}</div>
          <div class="text-sm font-sans">Cargando datasets del laboratorio...</div>
        </div>
      </div>
    `;
  }

  render(datasets = []) {
    const state = store.getState();
    const activeId = state.activeDatasetId;

    this.container.innerHTML = `
      <div class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="workbench-card max-w-xl w-full p-6 space-y-5 border-[#242A36] shadow-2xl">
          <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
            <div>
              <h3 class="text-base font-bold text-[#F7F8FA] font-sans">Cambiar dataset activo</h3>
              <p class="text-xs text-[#8B95A7] font-sans">Selecciona el dataset sobre el que trabajarán Experimentos y Evidencia</p>
            </div>
            <button id="btnCloseSwitchModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded-md hover:bg-[#161B26] transition-colors">
              ${icon("x", "icon-sm")}
            </button>
          </div>

          <div class="space-y-2.5 max-h-80 overflow-y-auto pr-1">
            ${datasets.length === 0 ? `
              <div class="p-8 text-center text-[#8B95A7] font-sans text-xs bg-[#11151E] rounded-xl border border-[#242A36]">
                No hay datasets registrados en este workspace.
              </div>
            ` : datasets.map(d => {
              const isActive = d.id === activeId;
              const rowText = d.row_count != null ? Number(d.row_count).toLocaleString() + " filas" : "Sin perfilar";
              const colText = d.column_count != null ? d.column_count + " cols" : "";
              const taskText = (d.task_type || "No supervisado").replace("_", " ");
              const targetText = d.target_column ? `Target: <span class="font-mono text-[#F7F8FA]">${escapeHtml(d.target_column)}</span>` : "Sin target";

              return `
                <div class="ds-item-card p-3.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between gap-4 ${
                  isActive
                    ? 'bg-[#161B26] border-[#4F67FF] shadow-[0_0_0_1px_#4F67FF]'
                    : 'bg-[#11151E] border-[#242A36] hover:border-[#384355] hover:bg-[#161B26]'
                }" data-dataset-id="${escapeHtml(d.id)}">
                  <div class="space-y-1 min-w-0">
                    <div class="flex items-center space-x-2">
                      <span class="font-sans font-semibold text-sm text-[#F7F8FA] truncate">${escapeHtml(d.name || d.id)}</span>
                      ${isActive ? `
                        <span class="badge-gain px-2 py-0.5 rounded text-[10px] font-mono font-medium flex items-center space-x-1">
                          <span class="w-1.5 h-1.5 rounded-full bg-[#22C55E]"></span>
                          <span>ACTIVO</span>
                        </span>
                      ` : ''}
                    </div>
                    <div class="text-[11px] text-[#8B95A7] font-sans flex items-center space-x-3">
                      <span>${rowText} ${colText ? '• ' + colText : ''}</span>
                      <span>•</span>
                      <span class="capitalize">${taskText}</span>
                      <span>•</span>
                      <span>${targetText}</span>
                    </div>
                  </div>

                  <button class="btn-select-ds px-3 py-1.5 rounded-lg text-xs font-sans font-medium transition-colors ${
                    isActive
                      ? 'bg-[#4F67FF] text-white pointer-events-none'
                      : 'bg-[#161B26] text-[#8B95A7] hover:text-[#F7F8FA] border border-[#242A36] hover:border-[#4F67FF]/50'
                  }" data-dataset-id="${escapeHtml(d.id)}">
                    ${isActive ? 'Seleccionado' : 'Seleccionar'}
                  </button>
                </div>
              `;
            }).join("")}
          </div>

          <div class="flex items-center justify-between pt-3 border-t border-[#242A36] text-xs">
            <button id="btnSwitchModalAdd" class="bg-[#161B26] hover:bg-[#1E2536] text-[#F7F8FA] border border-[#242A36] px-3 py-1.5 rounded-lg font-sans font-medium flex items-center space-x-1.5 transition-colors">
              ${icon("plus", "icon-sm")}
              <span>Añadir nuevo dataset</span>
            </button>
            <button id="btnSwitchModalCancel" class="text-[#8B95A7] hover:text-[#F7F8FA] px-3 py-1.5 font-sans transition-colors">
              Cerrar
            </button>
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
  }

  _bindEvents() {
    const close = () => this.close();

    this.container.querySelector("#btnCloseSwitchModal")?.addEventListener("click", close);
    this.container.querySelector("#btnSwitchModalCancel")?.addEventListener("click", close);

    this.container.querySelectorAll(".ds-item-card, .btn-select-ds").forEach(el => {
      el.addEventListener("click", () => {
        const id = el.getAttribute("data-dataset-id");
        if (id) {
          store.setActiveDataset(id);
          this.close();
          bus.emit("dataset:switched", id);
        }
      });
    });

    this.container.querySelector("#btnSwitchModalAdd")?.addEventListener("click", () => {
      this.close();
      bus.emit("modal:register-dataset");
    });
  }

  close() {
    if (this.container) {
      this.container.innerHTML = "";
    }
  }
}
