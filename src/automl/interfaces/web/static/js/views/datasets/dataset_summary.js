import { icon } from "../../icons.js";
import { escapeHtml } from "../../utils.js";

export function _renderResumenTab(view, p, featureCols, numCols, catCols, excludedCount, recommendations) {
  return `
    <div class="space-y-6">
      <!-- Diagnostic Metrics Grid -->
      <div class="workbench-card p-6 space-y-5">
        <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
          <div>
            <h3 class="text-sm font-semibold text-[#F7F8FA] font-sans">Métricas del Perfil</h3>
            <p class="text-xs text-[#8B95A7] mt-0.5 font-sans">Dimensiones, tipo de tarea y roles de variables inferidos.</p>
          </div>
          <span class="badge-gain text-xs px-2.5 py-0.5 rounded-md font-mono font-medium">Perfilado</span>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 text-left">
          <div class="p-3.5 rounded-xl bg-[#151B26]">
            <div class="text-[11px] uppercase text-[#8B95A7] font-medium font-sans">Tarea</div>
            <div class="text-sm font-semibold text-[#F7F8FA] mt-1 truncate capitalize">${(p.task_type || "No supervisado").replace("_", " ")}</div>
          </div>

          <div class="p-3.5 rounded-xl bg-[#151B26]">
            <div class="text-[11px] uppercase text-[#8B95A7] font-medium font-sans">Target</div>
            <div class="text-sm font-semibold text-[#4F67FF] font-mono mt-1 truncate">${p.target_column || "—"}</div>
          </div>

          <div class="p-3.5 rounded-xl bg-[#151B26]">
            <div class="text-[11px] uppercase text-[#8B95A7] font-medium font-sans">Métrica</div>
            <div class="text-sm font-semibold text-[#22C55E] font-mono mt-1">${(p.task_type || "").includes("regression") ? "RMSE" : "ROC-AUC"}</div>
          </div>

          <div class="p-3.5 rounded-xl bg-[#151B26]">
            <div class="text-[11px] uppercase text-[#8B95A7] font-medium font-sans">Filas</div>
            <div class="text-sm font-semibold text-[#F7F8FA] font-mono mt-1">${p.row_count != null ? Number(p.row_count).toLocaleString() : "—"}</div>
          </div>

          <div class="p-3.5 rounded-xl bg-[#151B26]">
            <div class="text-[11px] uppercase text-[#8B95A7] font-medium font-sans">Predictoras</div>
            <div class="text-sm font-semibold text-[#F7F8FA] font-mono mt-1">${featureCols.length}</div>
          </div>

          <div class="p-3.5 rounded-xl bg-[#151B26]">
            <div class="text-[11px] uppercase text-[#8B95A7] font-medium font-sans">Numéricas</div>
            <div class="text-sm font-semibold text-[#F7F8FA] font-mono mt-1">${numCols}</div>
          </div>

          <div class="p-3.5 rounded-xl bg-[#151B26]">
            <div class="text-[11px] uppercase text-[#8B95A7] font-medium font-sans">Categóricas</div>
            <div class="text-sm font-semibold text-[#F7F8FA] font-mono mt-1">${catCols}</div>
          </div>

          <div class="p-3.5 rounded-xl bg-[#151B26]">
            <div class="text-[11px] uppercase text-[#8B95A7] font-medium font-sans">Excluidas</div>
            <div class="text-sm font-semibold text-[#EF4444] font-mono mt-1">${excludedCount}</div>
          </div>
        </div>
      </div>

      <!-- Tarjeta de Salud y Anti-Leakage Guardian -->
      ${view._renderHealthCard(p)}

      <!-- Smart Recommendations Panel -->
      ${recommendations.length > 0 ? `
        <div class="workbench-card p-5 space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#242A36] pb-3">
            <div class="flex items-center space-x-2.5">
              <span class="text-[#4F67FF]">${icon("lightbulb", "icon-sm")}</span>
              <span class="text-sm font-semibold text-[#F7F8FA] font-sans">Recomendaciones del motor</span>
              <span class="text-xs font-mono text-[#8B95A7] bg-[#151B26] px-2 py-0.5 rounded-md">${recommendations.length} detectadas</span>
            </div>
            <button id="btnApplyRecommendations" class="btn-technical text-xs font-sans flex items-center space-x-1.5">
              ${icon("check", "icon-sm")}
              <span>Aplicar recomendaciones</span>
            </button>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5 pt-1">
            ${recommendations.slice(0, 6).map(r => {
              const isDanger = r.severity === 'danger' || r.type === 'exclude';
              const isSuccess = r.severity === 'success' || r.type === 'recommend';
              const isWarning = r.severity === 'warning' || r.type === 'warn';
              const badgeClass = isDanger ? 'badge-err' : isSuccess ? 'badge-gain' : isWarning ? 'badge-warn' : 'badge-intel';
              return `
                <div class="p-4 rounded-xl bg-[#151B26] hover:bg-[#1A2230] transition-colors space-y-2">
                  <div class="flex items-center justify-between">
                    <span class="${badgeClass} text-[10px] px-2 py-0.5 rounded font-mono font-medium">${(r.badge || 'INFO').toUpperCase()}</span>
                    ${r.column ? `<code class="text-[11px] font-mono text-[#8B95A7]">${escapeHtml(r.column)}</code>` : ""}
                  </div>
                  <h4 class="font-sans font-semibold text-xs text-[#F7F8FA]">${escapeHtml(r.title)}</h4>
                  <p class="text-xs text-[#8B95A7] font-sans leading-relaxed">${escapeHtml(r.description)}</p>
                </div>
              `;
            }).join("")}
          </div>
        </div>
      ` : ""}

      <!-- Muestra de Datos Crudos (Preview) -->
      <div class="workbench-card p-5 space-y-4">
        <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
          <div class="flex items-center space-x-2">
            <span class="text-[#4F67FF]">${icon("scroll-text", "icon-sm")}</span>
            <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">Muestra previa de datos</h4>
          </div>
          <span class="text-xs text-[#8B95A7] font-mono">Primeras filas del dataset</span>
        </div>
        ${view._renderPreviewTab(p)}
      </div>
    </div>
  `;
}

