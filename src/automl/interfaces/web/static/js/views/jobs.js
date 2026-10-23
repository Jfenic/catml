import { api } from "../api.js";
import { icon } from "../icons.js";
import {
  escapeHtml,
  isJobActive,
  getJobStatusLabel,
  JOB_ACTION_LABELS,
} from "../utils.js";

const actions = {
  queued: ["pause", "cancel"],
  running: ["pause", "cancel"],
  pause_requested: ["cancel"],
  paused: ["resume", "cancel"],
  failed: ["retry", "cancel"],
  interrupted: ["retry", "cancel"],
};

/**
 * JobsPanel — Discrete Floating Background Jobs Controller (Fase P1)
 * Complies with ADR-006 (Tech Minimalista Premium).
 * 
 * Behavior:
 * - 0 active jobs: Collapsed into a discreet non-intrusive floating trigger chip. Never covers bottom buttons/tables.
 * - Active jobs: Displays an unobtrusive live floating banner with real progress and pause/cancel controls.
 * - Full Drawer / Dialog: Separates 'Activos' (running/queued) from 'Historial' (completed/failed/cancelled).
 * - Full fidelity for all lifecycle states: queued, running, pause_requested, cancel_requested, paused, completed, failed, cancelled, interrupted.
 */
export class JobsPanel {
  constructor() {
    this.panel = null;
    this.jobs = [];
    this.activeJobs = [];
    this.recentJobs = [];
    this.isExpanded = false;
    this.isMinimized = false;
    this.activeTab = "active"; // "active" | "history"
    this.refreshing = false;
    this.timer = null;
  }

  mount(container = null) {
    this.panel = document.createElement("div");
    this.panel.id = "jobsPanelContainer";
    this.panel.className = "fixed bottom-4 left-4 z-40 font-sans text-xs select-none";

    const target = container || document.body;
    target.append(this.panel);

    this.refresh();
    this.timer = setInterval(() => {
      if (!document.hidden) this.refresh();
    }, 2000);
  }

  async refresh() {
    if (this.refreshing) return;
    this.refreshing = true;
    try {
      const jobs = await api.getJobs();
      this.jobs = jobs || [];
      this.activeJobs = this.jobs.filter(job => isJobActive(job));
      this.recentJobs = this.jobs.filter(job => !isJobActive(job)).slice(0, 25);
      this._render();
    } catch (error) {
      console.warn("[JobsPanel] Error fetching jobs:", error);
    } finally {
      this.refreshing = false;
    }
  }

  _render() {
    if (!this.panel) return;

    if (this.isExpanded) {
      this._renderExpandedModal();
      return;
    }

    if (this.activeJobs.length === 0 || this.isMinimized) {
      this._renderDiscreetTrigger();
      return;
    }

    this._renderActiveBanner();
  }

  /**
   * Mode A: Discreet Trigger Chip (when 0 active jobs, or user minimized)
   * Takes minimal screen space and never obscures interactive content.
   */
  _renderDiscreetTrigger() {
    const hasActive = this.activeJobs.length > 0;
    this.panel.innerHTML = `
      <button id="btnToggleJobsDrawer" class="bg-[#10151E]/95 hover:bg-[#161B26] border ${hasActive ? 'border-[#4F67FF]/60 text-[#F7F8FA]' : 'border-[#242A36] text-[#8B95A7] hover:text-[#F7F8FA]'} rounded-full px-3 py-1.5 shadow-xl backdrop-blur flex items-center space-x-2 transition-all cursor-pointer" title="Ver historial de trabajos">
        ${hasActive ? `
          <span class="relative flex h-2 w-2">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#4F67FF] opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2 w-2 bg-[#4F67FF]"></span>
          </span>
          <span class="font-medium text-[11px]">${this.activeJobs.length} ${this.activeJobs.length === 1 ? 'trabajo activo' : 'trabajos activos'}</span>
        ` : `
          <span class="w-2 h-2 rounded-full bg-[#8B95A7]/70"></span>
          <span class="font-medium text-[11px]">Trabajos (0 activos)</span>
        `}
      </button>
    `;

    this.panel.querySelector("#btnToggleJobsDrawer")?.addEventListener("click", () => {
      if (hasActive && this.isMinimized) {
        this.isMinimized = false;
      } else {
        this.isExpanded = true;
      }
      this._render();
    });
  }

  /**
   * Mode B: Active Floating Banner (when jobs are in progress)
   * Surfaces live progress, cancel/pause controls, and quick expansion.
   */
  _renderActiveBanner() {
    const primaryJob = this.activeJobs[0];
    const otherCount = this.activeJobs.length - 1;
    const isIndeterminate = !primaryJob.total || primaryJob.total <= 0;
    const percent = isIndeterminate ? 0 : Math.round(((primaryJob.completed || 0) * 100) / primaryJob.total);

    this.panel.innerHTML = `
      <div class="bg-[#10151E] border border-[#4F67FF]/40 rounded-xl p-3.5 text-slate-200 shadow-2xl space-y-2.5 max-w-sm w-[calc(100vw-32px)] sm:w-96 backdrop-blur">
        <!-- Header con indicador en vivo y controles -->
        <div class="flex items-center justify-between pb-1.5 border-b border-[#242A36]">
          <div class="flex items-center space-x-2">
            <span class="relative flex h-2 w-2">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#4F67FF] opacity-75"></span>
              <span class="relative inline-flex rounded-full h-2 w-2 bg-[#4F67FF]"></span>
            </span>
            <span class="font-semibold text-xs text-[#F7F8FA]">
              ${this.activeJobs.length === 1 ? 'Trabajo en curso' : `${this.activeJobs.length} trabajos en curso`}
            </span>
          </div>
          <div class="flex items-center space-x-1">
            <button id="btnMinimizeJobsBar" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded hover:bg-[#161B26] transition-colors" title="Minimizar barra">
              ${icon("arrow-down", "icon-xs", 14)}
            </button>
            <button id="btnExpandJobsModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded hover:bg-[#161B26] transition-colors" title="Ver lista completa">
              ${icon("layers", "icon-xs", 14)}
            </button>
          </div>
        </div>

        <!-- Detalles del trabajo primario -->
        <div class="space-y-1.5">
          <div class="flex items-center justify-between text-[11px]">
            <span class="font-medium text-[#F7F8FA] capitalize truncate">${escapeHtml(primaryJob.operation || 'Entrenamiento')}</span>
            <span class="text-[10px] font-mono text-[#4F67FF] uppercase">${escapeHtml(getJobStatusLabel(primaryJob.status))}</span>
          </div>

          <!-- Barra de progreso real -->
          <div class="w-full bg-[#080A0F] h-1.5 rounded-full overflow-hidden border border-[#242A36]">
            ${isIndeterminate ? `
              <div class="bg-[#4F67FF] h-full w-full animate-pulse"></div>
            ` : `
              <div class="bg-[#4F67FF] h-full transition-all duration-300" style="width: ${Math.max(5, percent)}%"></div>
            `}
          </div>

          <div class="flex justify-between items-center text-[10px] text-[#8B95A7] font-mono pt-0.5">
            <span>${isIndeterminate ? 'En progreso (pasos indeterminados)' : `${primaryJob.completed}/${primaryJob.total} pasos (${percent}%)`}</span>
            ${otherCount > 0 ? `<span>+${otherCount} en cola</span>` : ''}
          </div>
        </div>

        <!-- Acciones rápidas -->
        <div class="flex items-center justify-between pt-1">
          <div class="flex items-center space-x-1.5">
            ${(actions[primaryJob.status] || []).map(action => `
              <button data-job-id="${primaryJob.id}" data-action="${action}" class="btn-job-action bg-[#161B26] hover:bg-[#202738] border border-[#242A36] text-[#F7F8FA] text-[11px] px-2.5 py-1 rounded transition-colors font-medium">
                ${JOB_ACTION_LABELS[action] || action}
              </button>
            `).join("")}
          </div>
          <button id="btnViewAllJobsFromBar" class="text-[11px] text-[#4F67FF] hover:text-[#53C8FF] font-medium transition-colors">
            Ver todos
          </button>
        </div>
      </div>
    `;

    this.panel.querySelector("#btnMinimizeJobsBar")?.addEventListener("click", () => {
      this.isMinimized = true;
      this._render();
    });

    this.panel.querySelector("#btnExpandJobsModal")?.addEventListener("click", () => {
      this.isExpanded = true;
      this._render();
    });

    this.panel.querySelector("#btnViewAllJobsFromBar")?.addEventListener("click", () => {
      this.isExpanded = true;
      this._render();
    });

    this._bindActionButtons();
  }

  /**
   * Mode C: Full Expanded Modal / Drawer
   * Clearly separates Active jobs from Completed / Failed history.
   */
  _renderExpandedModal() {
    this.panel.innerHTML = `
      <div class="bg-[#10151E] border border-[#242A36] rounded-xl p-4 text-slate-200 shadow-2xl max-w-md w-[calc(100vw-32px)] sm:w-[420px] max-h-[70vh] flex flex-col backdrop-blur space-y-3">
        <!-- Modal Header -->
        <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
          <div class="flex items-center space-x-2">
            <span class="text-[#4F67FF]">${icon("cpu", "icon-sm")}</span>
            <h3 class="font-bold text-sm text-[#F7F8FA]">Gestión de Trabajos</h3>
          </div>
          <button id="btnCloseExpandedJobs" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded hover:bg-[#161B26] transition-colors" title="Cerrar panel">
            ${icon("x", "icon-sm")}
          </button>
        </div>

        <!-- Segmented Tabs: Activos | Historial -->
        <div class="flex items-center space-x-1.5 bg-[#0A0D14] p-1 rounded-lg border border-[#242A36] text-xs font-medium">
          <button id="tabJobsActive" class="flex-1 py-1 px-2.5 rounded-md text-center transition-all ${this.activeTab === 'active' ? 'bg-[#161B26] text-[#F7F8FA] shadow' : 'text-[#8B95A7] hover:text-[#F7F8FA]'}">
            <span>En curso</span>
            <span class="ml-1.5 px-1.5 py-0.2 rounded font-mono text-[10px] ${this.activeJobs.length > 0 ? 'bg-[#4F67FF]/20 text-[#4F67FF]' : 'bg-[#1E2536] text-[#8B95A7]'}">${this.activeJobs.length}</span>
          </button>
          <button id="tabJobsHistory" class="flex-1 py-1 px-2.5 rounded-md text-center transition-all ${this.activeTab === 'history' ? 'bg-[#161B26] text-[#F7F8FA] shadow' : 'text-[#8B95A7] hover:text-[#F7F8FA]'}">
            <span>Historial</span>
            <span class="ml-1.5 px-1.5 py-0.2 rounded font-mono text-[10px] bg-[#1E2536] text-[#8B95A7]">${this.recentJobs.length}</span>
          </button>
        </div>

        <!-- Content Area -->
        <div class="flex-1 overflow-y-auto space-y-2 pr-1 max-h-[48vh]">
          ${this.activeTab === 'active' ? this._renderActiveList() : this._renderHistoryList()}
        </div>
      </div>
    `;

    this.panel.querySelector("#btnCloseExpandedJobs")?.addEventListener("click", () => {
      this.isExpanded = false;
      this._render();
    });

    this.panel.querySelector("#tabJobsActive")?.addEventListener("click", () => {
      this.activeTab = "active";
      this._render();
    });

    this.panel.querySelector("#tabJobsHistory")?.addEventListener("click", () => {
      this.activeTab = "history";
      this._render();
    });

    this._bindActionButtons();
  }

  _renderActiveList() {
    if (this.activeJobs.length === 0) {
      return `
        <div class="p-8 text-center text-[#8B95A7] space-y-2">
          <div class="inline-block text-[#8B95A7]/60">${icon("check-circle-2", "icon-lg")}</div>
          <div class="text-xs font-medium text-[#F7F8FA]">Sin operaciones activas</div>
          <p class="text-[11px]">No hay trabajos en cola ni entrenamientos en ejecución en este momento.</p>
        </div>
      `;
    }

    return this.activeJobs.map(job => this._renderJobCard(job, true)).join("");
  }

  _renderHistoryList() {
    if (this.recentJobs.length === 0) {
      return `
        <div class="p-8 text-center text-[#8B95A7] space-y-2">
          <div class="text-xs font-medium">Sin historial</div>
          <p class="text-[11px]">Los trabajos completados, fallidos o cancelados aparecerán aquí.</p>
        </div>
      `;
    }

    return this.recentJobs.map(job => this._renderJobCard(job, false)).join("");
  }

  _renderJobCard(job, _isActive) {
    const isCompleted = job.status === "completed";
    const isFailed = job.status === "failed";
    const isCancelled = job.status === "cancelled" || job.status === "interrupted";
    const badgeClass = isCompleted ? "badge-gain" : isFailed ? "badge-err" : isCancelled ? "text-[#8B95A7] bg-[#161B26]" : "badge-warn";
    const availableActions = actions[job.status] || [];

    return `
      <div class="p-3 rounded-lg bg-[#0E121B] border border-[#242A36] space-y-2 text-xs">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-2 min-w-0 pr-2">
            <span class="font-semibold text-[#F7F8FA] capitalize truncate">${escapeHtml(job.operation || 'Tarea')}</span>
            <span class="${badgeClass} text-[9px] px-1.5 py-0.2 rounded font-mono uppercase">${escapeHtml(getJobStatusLabel(job.status))}</span>
          </div>
          <span class="text-[10px] font-mono text-[#8B95A7] whitespace-nowrap">Intento ${job.attempt || 1}</span>
        </div>

        <div class="text-[10px] text-[#8B95A7] font-mono break-all">
          ID: ${escapeHtml(job.id)}
        </div>

        ${job.total > 0 ? `
          <div class="space-y-1">
            <div class="w-full bg-[#080A0F] h-1.5 rounded-full overflow-hidden border border-[#242A36]">
              <div class="bg-[#4F67FF] h-full" style="width: ${Math.round(((job.completed || 0) * 100) / job.total)}%"></div>
            </div>
            <div class="text-[10px] font-mono text-[#8B95A7] text-right">
              ${job.completed}/${job.total} pasos
            </div>
          </div>
        ` : ''}

        ${job.error ? `
          <div class="p-2 rounded bg-[#EF4444]/10 border border-[#EF4444]/20 text-[#EF4444] text-[11px] font-mono break-all">
            ${escapeHtml(job.error)}
          </div>
        ` : (job.message ? `
          <div class="text-[11px] text-[#8B95A7]">
            ${escapeHtml(job.message)}
          </div>
        ` : '')}

        ${isCompleted && job.result ? `
          <div class="text-[11px] font-mono text-[#22C55E] pt-1 border-t border-[#242A36]/60">
            ${job.result.experiment_id ? `Experimento: ${escapeHtml(job.result.experiment_id)}` : (job.result.output_path ? `Salida: ${escapeHtml(job.result.output_path)}` : 'Completado con éxito')}
          </div>
        ` : ''}

        ${availableActions.length > 0 ? `
          <div class="flex items-center space-x-1.5 pt-1 border-t border-[#242A36]/40">
            ${availableActions.map(action => `
              <button data-job-id="${job.id}" data-action="${action}" class="btn-job-action bg-[#161B26] hover:bg-[#222B3D] border border-[#242A36] text-[#F7F8FA] text-[11px] px-2.5 py-1 rounded transition-colors font-medium">
                ${JOB_ACTION_LABELS[action] || action}
              </button>
            `).join("")}
          </div>
        ` : ''}
      </div>
    `;
  }

  _bindActionButtons() {
    this.panel.querySelectorAll(".btn-job-action").forEach(button => {
      button.addEventListener("click", async () => {
        const jobId = button.getAttribute("data-job-id");
        const action = button.getAttribute("data-action");
        if (!jobId || !action) return;

        button.disabled = true;
        button.textContent = "...";
        try {
          await api.controlJob(jobId, action);
          await this.refresh();
        } catch (error) {
          console.error(`[JobsPanel] Error controlling job ${jobId}:`, error);
          button.disabled = false;
          button.textContent = JOB_ACTION_LABELS[action] || action;
        }
      });
    });
  }

  destroy() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.panel) {
      this.panel.remove();
      this.panel = null;
    }
  }
}
