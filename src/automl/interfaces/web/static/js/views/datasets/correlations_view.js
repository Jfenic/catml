import { icon } from "../../icons.js";
import { escapeHtml } from "../../utils.js";

export function _renderCorrelationTab(view, p) {
  const cm = p.correlation_matrix;
  if (!cm || !cm.columns || !cm.columns.length) {
    return `
      <div class="p-8 text-center text-[#8B95A7] space-y-2">
        <p class="text-sm font-semibold">Numerical correlation matrix not computed.</p>
        <p class="text-xs text-[#8B95A7]/70">Requires at least 2 numerical features in dataset.</p>
      </div>
    `;
  }

  const cols = cm.columns;
  const matrix = cm.matrix || [];

  return `
    <div class="p-4 space-y-4">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
        <div>
          <span class="font-semibold text-[#F7F8FA] font-sans">Pearson Correlation Matrix ($r \\in [-1, 1]$)</span>
          <p class="text-[11px] text-[#8B95A7]">Bivariate linear dependency, target covariance, and collinear feature detection.</p>
        </div>
        <div class="flex flex-wrap items-center gap-3 text-[11px] font-mono">
          <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-[#22C55E] inline-block"></span><span class="text-[#8B95A7]">Strong Positive (&gt; 0.5)</span></span>
          <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-[#161B26] border border-[#242A36] inline-block"></span><span class="text-[#8B95A7]">Neutral (-0.2 to 0.2)</span></span>
          <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-[#EF4444] inline-block"></span><span class="text-[#8B95A7]">Negative (&lt; -0.2)</span></span>
          <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-sm ring-1 ring-[#F59E0B] bg-[#F59E0B]/20 inline-block"></span><span class="text-[#8B95A7]">Collinear (&gt; 0.88)</span></span>
        </div>
      </div>

      <div class="overflow-x-auto max-h-[580px] border border-[#242A36] rounded-xl">
        <table class="w-full text-center text-xs border-collapse">
          <thead class="sticky top-0 bg-[#0D1017] z-10 border-b border-[#242A36]">
            <tr>
              <th class="p-2.5 text-left font-mono text-[11px] text-[#8B95A7] bg-[#0D1017] sticky left-0 z-20 border-r border-[#242A36] min-w-[130px]">Feature</th>
              ${cols.map(c => `
                <th class="p-2 font-mono text-[11px] text-[#8B95A7] min-w-[70px] max-w-[110px] truncate" title="${escapeHtml(c)}">
                  ${c === p.target_column ? `<span class="text-[#4F67FF] inline-flex items-center space-x-1">${icon("target", "icon-sm")}<span>${escapeHtml(c)}</span></span>` : escapeHtml(c)}
                </th>
              `).join("")}
              <th class="p-2 font-mono text-[11px] text-[#8B95A7]">Visuals</th>
            </tr>
          </thead>
          <tbody>
            ${cols.map((rowName, rIdx) => `
              <tr class="border-b border-[#242A36]/60 hover:bg-[#161B26]/50">
                <td class="p-2 text-left font-mono font-medium text-[#F7F8FA] bg-[#0D1017]/95 sticky left-0 z-10 border-r border-[#242A36] truncate max-w-[150px]" title="${escapeHtml(rowName)}">
                  ${rowName === p.target_column ? `<span class="text-[#4F67FF] inline-flex items-center space-x-1">${icon("target", "icon-sm")}<span>${escapeHtml(rowName)}</span></span>` : escapeHtml(rowName)}
                </td>
                ${cols.map((colName, cIdx) => {
                  const val = (matrix[rIdx] && matrix[rIdx][cIdx] != null) ? matrix[rIdx][cIdx] : 0.0;
                  const isDiag = (rIdx === cIdx);
                  let cellBg = "bg-[#080A0F] text-[#8B95A7]";
                  let ringStyle = "";
                  if (!isDiag) {
                    if (val >= 0.70) cellBg = "bg-emerald-950 text-emerald-300 font-bold";
                    else if (val >= 0.40) cellBg = "bg-emerald-950/60 text-emerald-400 font-semibold";
                    else if (val >= 0.15) cellBg = "bg-emerald-950/20 text-emerald-400";
                    else if (val <= -0.50) cellBg = "bg-rose-950 text-rose-300 font-bold";
                    else if (val <= -0.20) cellBg = "bg-rose-950/60 text-rose-400 font-semibold";
                    else if (val <= -0.10) cellBg = "bg-amber-950/20 text-amber-400";
                    if (Math.abs(val) >= 0.88) ringStyle = "ring-1 ring-[#F59E0B] ring-inset";
                  } else {
                    cellBg = "bg-[#11151E] text-[#8B95A7] font-semibold";
                  }

                  return `
                    <td class="p-2 font-mono text-[11px] ${cellBg} ${ringStyle} transition-colors" title="${rowName} ↔ ${colName}: r = ${val}">
                      ${isDiag ? '1.00' : (val > 0 ? '+' : '') + val.toFixed(2)}
                    </td>
                  `;
                }).join("")}
                <td class="p-1.5 bg-[#0D1017]/60 text-center">
                  <button class="btn-visualize-var btn-technical text-[10px] py-0.5 px-2" data-var="${rowName}">
                    View
                  </button>
                </td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

