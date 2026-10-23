import { icon } from "../../icons.js";
import { escapeHtml } from "../../utils.js";

export function _renderHealthCard(view, p) {
  const hasLeakage = Boolean(p.has_leakage || (p.leakage_columns && p.leakage_columns.length > 0));
  const hasGroupLeakage = Boolean(p.has_group_leakage || (p.group_leakage_reports && p.group_leakage_reports.length > 0));
  const leakageCols = p.leakage_columns || [];
  const idCols = (p.columns || []).filter(c => c.is_identifier).map(c => c.name);

  return `
    <div class="workbench-card p-5 space-y-3">
      <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
        <div class="flex items-center space-x-2">
          <span class="${hasLeakage ? 'text-[#EF4444]' : 'text-[#22C55E]'}">${icon("shield-check", "icon-sm")}</span>
          <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">Diagnóstico de Salud y Anti-Leakage Guardian</h4>
        </div>
        ${hasLeakage || hasGroupLeakage ? `
          <span class="badge-err text-[11px] px-2.5 py-0.5 rounded font-mono font-medium">Fuga Detectada</span>
        ` : `
          <span class="badge-gain text-[11px] px-2.5 py-0.5 rounded font-mono font-medium flex items-center space-x-1">
            <span class="w-1.5 h-1.5 rounded-full bg-[#22C55E]"></span>
            <span>Sin fuga detectada</span>
          </span>
        `}
      </div>
      ${!(hasLeakage || hasGroupLeakage) ? `
      <p class="text-[11px] text-[#8B95A7] font-sans">
        Ausencia de fuga no implica dataset validado al completo (nulos, tipos, duplicados, etc.).
        La preparación reproducible de datos es un módulo futuro del núcleo CATML.
      </p>` : ""}

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-sans pt-1">
        <div class="p-3.5 rounded-xl bg-[#11151E] border border-[#242A36] space-y-1.5">
          <div class="flex items-center justify-between">
            <span class="font-medium text-[#F7F8FA]">Fuga de Datos (Data Leakage)</span>
            <span class="font-mono ${hasLeakage ? 'text-[#EF4444]' : 'text-[#22C55E]'} font-semibold">
              ${hasLeakage ? `${leakageCols.length} columna(s)` : '0 detectadas'}
            </span>
          </div>
          <p class="text-[11px] text-[#8B95A7]">
            ${hasLeakage
              ? `Columnas con correlación perfecta o fuga identificadas: <code class="text-[#EF4444] font-mono">${leakageCols.map(c => escapeHtml(c)).join(", ")}</code>. Descartadas automáticamente en el entrenamiento.`
              : 'Ninguna variable muestra correlación determinista ni contaminación con el target.'
            }
          </p>
        </div>

        <div class="p-3.5 rounded-xl bg-[#11151E] border border-[#242A36] space-y-1.5">
          <div class="flex items-center justify-between">
            <span class="font-medium text-[#F7F8FA]">Identificadores Descartados</span>
            <span class="font-mono ${idCols.length > 0 ? 'text-[#F59E0B]' : 'text-[#8B95A7]'} font-semibold">
              ${idCols.length} columna(s)
            </span>
          </div>
          <p class="text-[11px] text-[#8B95A7]">
            ${idCols.length > 0
              ? `Identificadores únicos no predictivos detectados: <code class="text-[#F7F8FA] font-mono">${idCols.map(c => escapeHtml(c)).join(", ")}</code>.`
              : 'No se detectaron columnas con IDs secuenciales o cardinalidad espuria.'
            }
          </p>
        </div>
      </div>
    </div>
  `;
}

