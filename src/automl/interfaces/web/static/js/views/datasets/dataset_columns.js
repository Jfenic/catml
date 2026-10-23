import { icon } from "../../icons.js";

export function _renderColumnasTab(view, p, featureCols, numCols, catCols) {
  const imgCols = (p.columns || []).filter(c => c.is_image && c.name !== p.target_column).length;
  return `
    <div class="space-y-6">
      <!-- Interactive Selection & Exploration Toolbar -->
      <div class="workbench-card p-4 space-y-3">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="flex flex-wrap items-center gap-3">
            <input type="text" id="featureSearchInput" value="${view.searchQuery}" placeholder="Filtrar columnas..." class="bg-[#090C12] border border-[#252C38] text-[#F7F8FA] text-xs rounded-lg px-3 py-1.5 focus:border-[#4F67FF] w-48 font-mono outline-none">

            <div class="segmented-track">
              <button data-filter="all" class="segmented-pill ${view.filterType === 'all' ? 'active-signal' : ''}">Todas (${featureCols.length})</button>
              <button data-filter="numeric" class="segmented-pill ${view.filterType === 'numeric' ? 'active-signal' : ''}">Numéricas (${numCols})</button>
              <button data-filter="categorical" class="segmented-pill ${view.filterType === 'categorical' ? 'active-signal' : ''}">Categóricas (${catCols})</button>
              ${imgCols > 0 ? `<button data-filter="image" class="segmented-pill ${view.filterType === 'image' ? 'active-signal' : ''}">Imágenes (${imgCols})</button>` : ''}
              <button data-filter="selected" class="segmented-pill ${view.filterType === 'selected' ? 'active-signal' : ''}">Seleccionadas (<span id="pillSelectedCount">${view.selectedFeatures.size}</span>)</button>
            </div>
          </div>

          <!-- Batch Selection Controls -->
          <div class="flex items-center gap-1 text-xs">
            <span class="text-[11px] text-[#8B95A7] uppercase font-sans font-medium mr-1.5">Selección:</span>
            <button id="btnSelectAll" class="btn-ghost text-xs py-1 px-2">Todas</button>
            <button id="btnDeselectAll" class="btn-ghost text-xs py-1 px-2">Ninguna</button>
            <span class="text-[#252C38] px-1">·</span>
            <button id="btnSelectTop5" class="btn-ghost text-xs py-1 px-2 text-[#4F67FF] hover:text-white">Top 5 Señal</button>
            <button id="btnSelectTop10" class="btn-ghost text-xs py-1 px-2 text-[#4F67FF] hover:text-white">Top 10 Señal</button>
          </div>
        </div>

        <!-- Bottom Status Counter & Dominant Launch CTA -->
        <div class="pt-3 border-t border-[#252C38]/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div class="flex items-center space-x-2 text-xs font-sans text-[#8B95A7]">
            <span>Estado de selección:</span>
            <span id="selectionLiveCount" class="font-mono text-xs text-[#F7F8FA]">
              <strong class="text-[#4F67FF]">${view.selectedFeatures.size}</strong> de ${featureCols.length} características seleccionadas
            </span>
          </div>

          <button id="btnLaunchWithSelection" class="btn-signal font-sans font-semibold text-xs px-5 py-2 shadow-md shadow-[#4F67FF]/20 flex items-center space-x-1.5">
            ${icon("rocket", "icon-sm")}
            <span>Crear experimento con selección (<span id="ctaSelectedCount">${view.selectedFeatures.size}</span>)</span>
          </button>
        </div>
      </div>

      <!-- Multi-Tab Equipment Module -->
      <div class="workbench-card overflow-hidden">
        <div class="border-b border-[#242A36] px-4 flex items-center space-x-6 text-xs font-sans font-medium bg-[#0D1017] overflow-x-auto">
          <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap transition-colors flex items-center space-x-1.5 ${view.activeTab === 'schema' ? 'border-[#4F67FF] text-[#F7F8FA] font-semibold' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="schema">
            ${icon("database", "icon-sm")}
            <span>Esquema y Selección</span>
          </button>
          <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap transition-colors flex items-center space-x-1.5 ${view.activeTab === 'stats' ? 'border-[#4F67FF] text-[#F7F8FA] font-semibold' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="stats">
            ${icon("bar-chart-3", "icon-sm")}
            <span>Estadísticas Descriptivas</span>
          </button>
          <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap transition-colors flex items-center space-x-1.5 ${view.activeTab === 'categories' ? 'border-[#4F67FF] text-[#F7F8FA] font-semibold' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="categories">
            ${icon("tags", "icon-sm")}
            <span>Distribución Categórica</span>
          </button>
          <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap transition-colors flex items-center space-x-1.5 ${view.activeTab === 'correlation' ? 'border-[#4F67FF] text-[#F7F8FA] font-semibold' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="correlation">
            ${icon("chart-line", "icon-sm")}
            <span>Matriz de Correlación</span>
          </button>
        </div>

        <div class="p-0" id="tabContentContainer">
          ${view._renderActiveTabContent(p)}
        </div>
      </div>
    </div>
  `;
}

export function _renderActiveTabContent(view, p) {
  if (view.activeTab === "schema") {
    return view._renderSchemaTab(p);
  } else if (view.activeTab === "stats") {
    return view._renderStatsTab(p);
  } else if (view.activeTab === "preview") {
    return view._renderPreviewTab(p);
  } else if (view.activeTab === "categories") {
    return view._renderCategoriesTab(p);
  } else if (view.activeTab === "correlation") {
    return view._renderCorrelationTab(p);
  } else if (view.activeTab === "calculator") {
    return view._renderCalculatorTab(p);
  }
  return "";
}

export function _renderSchemaTab(view, p) {
  const columns = view._getFilteredColumns(p.columns || []);

  return `
    <div class="overflow-x-auto">
      <table class="w-full wb-table text-left">
        <thead>
          <tr>
            <th class="w-10 text-center">
              <input type="checkbox" id="checkSelectAllRows" class="rounded text-[#4F67FF] bg-[#161B26] border border-[#242A36]" ${view.selectedFeatures.size > 0 ? "checked" : ""}>
            </th>
            <th>Característica</th>
            <th>Tipo</th>
            <th>Nulos %</th>
            <th>Valores Únicos</th>
            <th>Correlación con Target (r)</th>
            <th>Acción del Motor</th>
            <th>Justificación</th>
            <th class="text-center">Visualización</th>
          </tr>
        </thead>
        <tbody>
          ${columns.map(c => {
            const isTarget = c.name === p.target_column;
            const isChecked = view.selectedFeatures.has(c.name);
            const corr = c.target_correlation;

            return `
              <tr class="${isChecked ? 'bg-[#161B26]/40' : 'opacity-70'}">
                <td class="text-center">
                  ${isTarget ? `<span class="text-[#4F67FF] flex justify-center" title="Variable Objetivo (Target)">${icon("target", "icon-sm")}</span>` : `
                    <input type="checkbox" data-col="${c.name}" class="col-toggle-checkbox rounded text-[#4F67FF] bg-[#161B26] border border-[#242A36] cursor-pointer" ${isChecked ? "checked" : ""}>
                  `}
                </td>
                <td class="font-mono font-medium text-[#F7F8FA] flex items-center space-x-2">
                  <span>${c.name}</span>
                  ${isTarget ? `<span class="badge-sys text-[9px] px-1.5 py-0.2 rounded font-mono font-medium">TARGET</span>` : ""}
                </td>
                <td>
                  <span class="text-xs ${c.is_image ? 'text-[#53C8FF] inline-flex items-center gap-1' : c.is_text ? 'text-[#6956E8] inline-flex items-center gap-1' : c.dtype && (c.dtype.includes('float') || c.dtype.includes('int')) ? 'text-[#4F67FF]' : 'text-[#6956E8]'} font-mono">
                    ${c.is_image ? `${icon("image", "icon-sm")} Imagen` : c.is_text ? `${icon("message-square-text", "icon-sm")} Texto` : c.is_identifier ? 'Identificador' : (c.dtype && (c.dtype.includes('float') || c.dtype.includes('int')) ? 'Numérica' : 'Categórica')}
                  </span>
                </td>
                <td class="font-mono text-xs ${c.null_count > 0 ? 'text-[#F59E0B] font-semibold' : 'text-[#8B95A7]'}">
                  ${p.row_count ? ((c.null_count / p.row_count) * 100).toFixed(1) + '%' : '0.0%'}
                </td>
                <td class="font-mono text-xs text-[#F7F8FA]">
                  ${c.unique_count != null ? Number(c.unique_count).toLocaleString() : '—'}
                </td>
                <td class="font-mono text-xs">
                  ${corr != null ? `
                    <span class="${Math.abs(corr) >= 0.25 ? 'text-[#22C55E] font-semibold' : 'text-[#8B95A7]'}">
                      ${corr > 0 ? '+' : ''}${corr.toFixed(3)}
                    </span>
                  ` : '<span class="text-[#8B95A7]/50">—</span>'}
                </td>
                <td>
                  ${
                    isTarget
                      ? `<span class="badge-sys text-[11px] px-2 py-0.5 rounded font-mono font-medium">Target</span>`
                      : c.catml_action === "Exclude" || c.is_identifier
                      ? `<span class="badge-err text-[11px] px-2 py-0.5 rounded font-mono font-medium">Excluir</span>`
                      : c.is_image || c.catml_action === "Vision Embedding"
                      ? `<span class="badge-sys text-[11px] px-2 py-0.5 rounded font-mono font-medium border border-[#53C8FF]/30 text-[#53C8FF]">Visión</span>`
                      : c.catml_action === "Encode"
                      ? `<span class="badge-intel text-[11px] px-2 py-0.5 rounded font-mono font-medium">Codificar</span>`
                      : `<span class="badge-gain text-[11px] px-2 py-0.5 rounded font-mono font-medium">Conservar</span>`
                  }
                </td>
                <td class="text-xs text-[#8B95A7] font-sans">${c.action_reason || (isTarget ? "Objetivo de predicción" : "Variable predictora")}</td>
                <td class="text-center">
                  <button class="btn-visualize-var btn-technical text-[11px] py-0.5 px-2 ${c.is_image ? 'border-[#53C8FF]/40 text-[#53C8FF] hover:bg-[#53C8FF]/10' : ''}" data-var="${c.name}">
                    ${c.is_image ? `${icon("image", "icon-sm mr-1")} Galería` : 'Ver'}
                  </button>
                </td>
              </tr>
            `;
          }).join("")}
        </tbody>
      </table>
    </div>
  `;
}

export function _renderStatsTab(view, p) {
  const columns = view._getFilteredColumns(p.columns || []);

  return `
    <div class="overflow-x-auto">
      <table class="w-full wb-table text-left">
        <thead>
          <tr>
            <th>Característica</th>
            <th>Tipo</th>
            <th>Recuento</th>
            <th>Nulos %</th>
            <th>Media</th>
            <th>Std</th>
            <th>Mín</th>
            <th>25%</th>
            <th>Mediana (50%)</th>
            <th>75%</th>
            <th>Máx</th>
            <th>Asimetría (Skew)</th>
            <th>Correlación r</th>
            <th class="text-center">Visualización</th>
          </tr>
        </thead>
        <tbody>
          ${columns.map(c => {
            const isTarget = c.name === p.target_column;
            const hasStats = c.mean != null;
            return `
              <tr>
                <td class="font-mono font-medium text-[#F7F8FA] flex items-center space-x-1.5">
                  <span>${c.name}</span>
                  ${isTarget ? `<span class="text-[#4F67FF]" title="Target Column">${icon("target", "icon-sm")}</span>` : ''}
                </td>
                <td class="font-mono text-xs text-[#4F67FF]">${c.dtype}</td>
                <td class="font-mono text-xs text-[#F7F8FA]">${p.row_count ? (p.row_count - c.null_count).toLocaleString() : '—'}</td>
                <td class="font-mono text-xs ${c.null_count > 0 ? 'text-[#F59E0B]' : 'text-[#8B95A7]'}">${p.row_count ? ((c.null_count / p.row_count) * 100).toFixed(1) + '%' : '0%'}</td>
                <td class="font-mono text-xs text-[#F7F8FA]">${hasStats ? c.mean.toFixed(2) : '—'}</td>
                <td class="font-mono text-xs text-[#8B95A7]">${hasStats ? c.std.toFixed(2) : '—'}</td>
                <td class="font-mono text-xs text-[#F7F8FA]">${hasStats ? c.min.toFixed(2) : '—'}</td>
                <td class="font-mono text-xs text-[#8B95A7]">${hasStats ? c.q25.toFixed(2) : '—'}</td>
                <td class="font-mono text-xs text-[#4F67FF] font-semibold">${hasStats ? c.median.toFixed(2) : '—'}</td>
                <td class="font-mono text-xs text-[#8B95A7]">${hasStats ? c.q75.toFixed(2) : '—'}</td>
                <td class="font-mono text-xs text-[#F7F8FA]">${hasStats ? c.max.toFixed(2) : '—'}</td>
                <td class="font-mono text-xs ${c.skew != null && Math.abs(c.skew) > 1.5 ? 'text-[#F59E0B]' : 'text-[#8B95A7]'}">${c.skew != null ? c.skew.toFixed(2) : '—'}</td>
                <td class="font-mono text-xs ${c.target_correlation != null && Math.abs(c.target_correlation) >= 0.25 ? 'text-[#22C55E] font-semibold' : 'text-[#8B95A7]'}">
                  ${c.target_correlation != null ? (c.target_correlation > 0 ? '+' : '') + c.target_correlation.toFixed(3) : '—'}
                </td>
                <td class="text-center">
                  <button class="btn-visualize-var btn-technical text-[11px] py-0.5 px-2" data-var="${c.name}">
                    View
                  </button>
                </td>
              </tr>
            `;
          }).join("")}
        </tbody>
      </table>
    </div>
  `;
}

export function _renderPreviewTab(view, p) {
  const rows = p.preview_rows || [];
  const columns = p.columns || [];

  if (rows.length === 0) {
    return `
      <div class="p-8 text-center text-[#8B95A7] text-xs">
        No raw preview rows available for this dataset.
      </div>
    `;
  }

  return `
    <div class="p-4 space-y-2">
      <div class="text-xs text-[#8B95A7]">First 8 sampled rows from source dataset:</div>
      <div class="overflow-x-auto border border-[#242A36] rounded-xl">
        <table class="w-full wb-table text-left">
          <thead>
            <tr class="bg-[#0D1017]">
              <th class="w-12 text-[#8B95A7] text-center font-mono">#</th>
              ${columns.map(c => `
                <th class="font-mono text-xs ${c.name === p.target_column ? 'text-[#4F67FF]' : 'text-[#8B95A7]'}">${c.name}</th>
              `).join("")}
            </tr>
          </thead>
          <tbody>
            ${rows.map((row, idx) => `
              <tr>
                <td class="text-center font-mono text-xs text-[#8B95A7]">${idx + 1}</td>
                ${columns.map(c => {
                  const val = row[c.name];
                  const isTarget = c.name === p.target_column;
                  if (c.is_image && val) {
                    const imgUrl = `/api/media/preview?path=${encodeURIComponent(val)}&dataset_id=${encodeURIComponent(p.dataset_id)}`;
                    return `
                      <td class="font-mono text-xs text-[#F7F8FA]">
                        <div class="flex items-center gap-2">
                          <img src="${imgUrl}"
                               alt="${c.name}"
                               class="w-7 h-7 object-cover rounded border border-[#242A36] bg-[#090C12] cursor-pointer hover:scale-150 transition-transform shadow-sm"
                               loading="lazy"
                               title="Click to view full image"
                               onclick="window.open('${imgUrl}', '_blank')"
                               onerror="view.style.display='none';" />
                          <span class="truncate max-w-[130px] font-mono text-[11px] text-[#8B95A7]" title="${val}">${val}</span>
                        </div>
                      </td>
                    `;
                  }
                  return `
                    <td class="font-mono text-xs ${isTarget ? 'text-[#4F67FF] font-semibold bg-[#4F67FF]/10' : val === null ? 'text-[#8B95A7]/40 italic' : 'text-[#F7F8FA]'}">
                      ${val !== null && val !== undefined ? String(val) : 'null'}
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

export function _renderCategoriesTab(view, p) {
  const catCols = (p.columns || []).filter(c => c.top_categories && c.top_categories.length > 0);

  if (catCols.length === 0) {
    return `
      <div class="p-8 text-center text-[#8B95A7] text-xs">
        No categorical features detected with discrete frequency distributions.
      </div>
    `;
  }

  return `
    <div class="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      ${catCols.map(c => `
        <div class="workbench-card p-4 space-y-3 bg-[#161B26] border-[#242A36]">
          <div class="flex items-center justify-between border-b border-[#242A36] pb-2">
            <span class="font-mono font-medium text-[#F7F8FA] text-xs">${c.name}</span>
            <span class="text-[10px] font-mono text-[#6956E8] bg-[#6956E8]/10 border border-[#6956E8]/20 px-2 py-0.5 rounded">${c.unique_count} categories</span>
          </div>

          <div class="space-y-2">
            ${c.top_categories.map(cat => `
              <div class="space-y-1">
                <div class="flex justify-between text-[11px] font-mono">
                  <span class="text-[#F7F8FA] truncate max-w-[150px]">${cat.value}</span>
                  <span class="text-[#8B95A7]">${cat.pct}% (${cat.count})</span>
                </div>
                <div class="w-full bg-[#11151E] rounded-full h-1.5 overflow-hidden">
                  <div class="bg-[#4F67FF] h-1.5 rounded-full" style="width: ${Math.min(cat.pct, 100)}%"></div>
                </div>
              </div>
            `).join("")}
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

export function _renderVariableModal(view) {
  if (!view.activeModalVar || !view.profile) return "";
  const col = view.profile.columns.find(c => c.name === view.activeModalVar);
  if (!col) return "";

  const isImage = Boolean(col.is_image);
  const isNumeric = col.dtype && (col.dtype.includes("int") || col.dtype.includes("float"));
  const isTarget = col.name === view.profile.target_column;

  return `
    <div id="variableVisualModal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div class="workbench-card max-w-3xl w-full p-6 space-y-4 border-slate-700 shadow-2xl max-h-[90vh] overflow-y-auto">
        <!-- Header -->
        <div class="flex items-start justify-between border-b border-slate-800 pb-3">
          <div>
            <div class="flex items-center space-x-2">
              <span class="text-xs font-mono px-2 py-0.5 rounded font-bold ${isImage ? 'bg-[#53C8FF]/10 text-[#53C8FF] border border-[#53C8FF]/20 flex items-center gap-1' : isNumeric ? 'bg-[#4F67FF]/10 text-[#4F67FF] border border-[#4F67FF]/20' : 'bg-[#6956E8]/10 text-[#6956E8] border border-[#6956E8]/20'}">
                ${isImage ? `${icon("image", "icon-sm")} IMAGE` : isNumeric ? 'NUMERIC' : 'CATEGORICAL'}
              </span>
              <h3 class="text-base font-bold text-[#F7F8FA] font-mono">${col.name}</h3>
              ${isTarget ? `<span class="badge-sys text-[10px] px-2 py-0.5 rounded font-bold">TARGET</span>` : ""}
            </div>
            <p class="text-xs text-[#8B95A7] mt-1 font-sans">
              Type: <span class="font-mono text-[#F7F8FA]">${col.dtype}</span> •
              Nulls: <span class="font-mono ${col.null_count > 0 ? 'text-[#F59E0B]' : 'text-[#F7F8FA]'}">${col.null_count} (${((col.null_count / (view.profile.row_count || 1)) * 100).toFixed(1)}%)</span> •
              Unique: <span class="font-mono text-[#F7F8FA]">${col.unique_count != null ? col.unique_count.toLocaleString() : '—'}</span>
              ${col.target_correlation != null ? ` • Target correlation: <span class="font-mono font-bold ${Math.abs(col.target_correlation) >= 0.25 ? 'text-[#22C55E]' : 'text-[#F7F8FA]'}">${col.target_correlation > 0 ? '+' : ''}${col.target_correlation.toFixed(3)}</span>` : ""}
            </p>
          </div>
          <button id="btnCloseVarModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1.5 rounded hover:bg-[#161B26] transition-colors" title="Close">${icon("x", "icon-sm")}</button>
        </div>

        <!-- Modal Tabs -->
        <div class="flex items-center space-x-2 border-b border-[#252C38] pb-2 text-xs font-sans">
          ${isImage ? `
            <button class="var-tab-btn px-3 py-1.5 rounded-lg border font-medium transition-colors ${view.activeModalTab === 'gallery' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}" data-modal-tab="gallery">
              Sample Image Gallery
            </button>
          ` : `
            <button class="var-tab-btn px-3 py-1.5 rounded-lg border font-medium transition-colors ${view.activeModalTab === 'boxplot' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}" data-modal-tab="boxplot">
              Box Plot (IQR Separation)
            </button>
            <button class="var-tab-btn px-3 py-1.5 rounded-lg border font-medium transition-colors ${view.activeModalTab === 'histogram' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}" data-modal-tab="histogram">
              Histogram & Distribution
            </button>
            <button class="var-tab-btn px-3 py-1.5 rounded-lg border font-medium transition-colors ${view.activeModalTab === 'pattern' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}" data-modal-tab="pattern">
              Target Association
            </button>
          `}
        </div>

        <!-- Modal Tab Content -->
        <div class="pt-2">
          ${view._renderVarModalContent(col)}
        </div>
      </div>
    </div>
  `;
}

export function _renderVarModalContent(view, col) {
  if (col.is_image || view.activeModalTab === "gallery") {
    return view._renderImageGalleryContent(col);
  }
  if (view.activeModalTab === "boxplot") {
    return view._renderBoxPlotContent(col);
  } else if (view.activeModalTab === "histogram") {
    return view._renderHistogramContent(col);
  } else if (view.activeModalTab === "pattern") {
    return view._renderTargetPatternContent(col);
  }
  return "";
}

export function _renderImageGalleryContent(view, col) {
  const p = view.profile;
  const rows = p.preview_rows || [];
  const samples = [];
  rows.forEach(r => {
    const val = r[col.name];
    if (val) {
      samples.push({
        path: String(val),
        target: r[p.target_column],
      });
    }
  });

  if (samples.length === 0 && Array.isArray(col.sample_values)) {
    col.sample_values.forEach(v => {
      if (v) samples.push({ path: String(v), target: null });
    });
  }

  if (samples.length === 0) {
    return `
      <div class="p-8 text-center text-[#8B95A7] space-y-2">
        <div class="flex justify-center text-[#8B95A7]/50">${icon("image", "icon-xl", 32)}</div>
        <p class="text-sm font-medium">No sample images found for this feature.</p>
      </div>
    `;
  }

  return `
    <div class="space-y-4">
      <div class="flex items-center justify-between text-xs text-[#8B95A7]">
        <span>Displaying sampled images (${samples.length} items) with target labels</span>
        <span class="font-mono text-[#53C8FF] text-[11px] font-semibold">Modality: Image Feature (timm)</span>
      </div>
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 max-h-[500px] overflow-y-auto p-1">
        ${samples.map((s, idx) => {
          const imgUrl = `/api/media/preview?path=${encodeURIComponent(s.path)}&dataset_id=${encodeURIComponent(p.dataset_id)}`;
          return `
            <div class="p-2.5 rounded-xl bg-[#0D1017] border border-[#242A36] space-y-2 hover:border-[#4F67FF]/50 transition-colors">
              <div class="relative w-full aspect-square bg-[#080A0F] rounded-lg overflow-hidden border border-[#161B26] flex items-center justify-center">
                <img src="${imgUrl}"
                     alt="Sample ${idx + 1}"
                     class="w-full h-full object-cover cursor-pointer hover:scale-105 transition-transform"
                     loading="lazy"
                     onclick="window.open('${imgUrl}', '_blank')"
                     onerror="view.parentElement.innerHTML='<div class=\\'text-center p-2 text-[10px] text-slate-500 font-mono\\'>Image not accessible</div>';"
                />
              </div>
              <div class="space-y-1">
                <div class="text-[11px] font-mono text-[#F7F8FA] truncate" title="${s.path}">${s.path.split('/').pop()}</div>
                ${s.target !== null && s.target !== undefined ? `
                  <div class="flex items-center justify-between text-[10px] font-mono">
                    <span class="text-[#8B95A7]">Target:</span>
                    <span class="text-[#4F67FF] font-semibold">${s.target}</span>
                  </div>
                ` : ''}
              </div>
            </div>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

export function _renderBoxPlotContent(view, col) {
  const bp = col.box_plot;
  if (!bp || bp.min == null || bp.max == null) {
    return `
      <div class="p-8 text-center text-[#8B95A7] space-y-2">
        <div class="flex justify-center text-[#8B95A7]/50">${icon("boxes", "icon-xl", 32)}</div>
        <p class="text-sm font-medium">Box plots are designed for quantitative numerical features.</p>
        <p class="text-xs text-[#8B95A7]/70">For categorical features, inspect the Distribution or Target Association tabs.</p>
      </div>
    `;
  }

  const byTarget = bp.by_target || [];
  const hasTargetSplit = byTarget.length >= 2;

  const minVal = bp.min;
  const maxVal = bp.max;
  const range = (maxVal - minVal) || 1;

  const svgWidth = 640;
  const svgHeight = hasTargetSplit ? (60 + byTarget.length * 60) : 120;
  const padL = 90;
  const padR = 40;
  const plotW = svgWidth - padL - padR;

  const scaleX = val => padL + Math.max(0, Math.min(plotW, ((val - minVal) / range) * plotW));

  let svgRows = "";
  if (hasTargetSplit) {
    byTarget.forEach((t, idx) => {
      const yCenter = 45 + idx * 55;
      const color = idx === 0 ? "#4F67FF" : (idx === 1 ? "#22C55E" : "#F59E0B");
      const xMin = scaleX(t.min);
      const xQ1 = scaleX(t.q25);
      const xMed = scaleX(t.median);
      const xQ3 = scaleX(t.q75);
      const xMax = scaleX(t.max);
      const xMean = t.mean != null ? scaleX(t.mean) : null;

      svgRows += `
        <text x="${padL - 10}" y="${yCenter + 4}" fill="${color}" font-size="11" font-weight="bold" font-family="monospace" text-anchor="end">
          Class ${t.class_label}
        </text>
        <line x1="${xMin}" y1="${yCenter}" x2="${xMax}" y2="${yCenter}" stroke="${color}" stroke-width="2" stroke-dasharray="3,2" />
        <line x1="${xMin}" y1="${yCenter - 9}" x2="${xMin}" y2="${yCenter + 9}" stroke="${color}" stroke-width="2" />
        <line x1="${xMax}" y1="${yCenter - 9}" x2="${xMax}" y2="${yCenter + 9}" stroke="${color}" stroke-width="2" />
        <rect x="${xQ1}" y="${yCenter - 13}" width="${Math.max(2, xQ3 - xQ1)}" height="26" fill="${color}" fill-opacity="0.25" stroke="${color}" stroke-width="2" rx="3" />
        <line x1="${xMed}" y1="${yCenter - 13}" x2="${xMed}" y2="${yCenter + 13}" stroke="#ffffff" stroke-width="3" />
        ${xMean != null ? `
          <polygon points="${xMean},${yCenter - 6} ${xMean + 5},${yCenter} ${xMean},${yCenter + 6} ${xMean - 5},${yCenter}" fill="#fbbf24" stroke="#ffffff" stroke-width="1" />
        ` : ""}
      `;
    });
  } else {
    const yCenter = 50;
    const color = "#4F67FF";
    const xMin = scaleX(bp.min);
    const xQ1 = scaleX(bp.q25);
    const xMed = scaleX(bp.median);
    const xQ3 = scaleX(bp.q75);
    const xMax = scaleX(bp.max);
    const xMean = bp.mean != null ? scaleX(bp.mean) : null;

    svgRows = `
      <text x="${padL - 10}" y="${yCenter + 4}" fill="${color}" font-size="11" font-weight="bold" font-family="monospace" text-anchor="end">
        Distribution
      </text>
      <line x1="${xMin}" y1="${yCenter}" x2="${xMax}" y2="${yCenter}" stroke="${color}" stroke-width="2" stroke-dasharray="3,2" />
      <line x1="${xMin}" y1="${yCenter - 12}" x2="${xMin}" y2="${yCenter + 12}" stroke="${color}" stroke-width="2" />
      <line x1="${xMax}" y1="${yCenter - 12}" x2="${xMax}" y2="${yCenter + 12}" stroke="${color}" stroke-width="2" />
      <rect x="${xQ1}" y="${yCenter - 16}" width="${Math.max(2, xQ3 - xQ1)}" height="32" fill="${color}" fill-opacity="0.25" stroke="${color}" stroke-width="2" rx="3" />
      <line x1="${xMed}" y1="${yCenter - 16}" x2="${xMed}" y2="${yCenter + 16}" stroke="#ffffff" stroke-width="3" />
      ${xMean != null ? `
        <polygon points="${xMean},${yCenter - 6} ${xMean + 5},${yCenter} ${xMean},${yCenter + 6} ${xMean - 5},${yCenter}" fill="#fbbf24" stroke="#ffffff" stroke-width="1" />
      ` : ""}
    `;
  }

  const yAxis = svgHeight - 20;
  const ticks = [0, 0.25, 0.5, 0.75, 1.0].map(frac => {
    const val = minVal + frac * range;
    const x = padL + frac * plotW;
    return `
      <line x1="${x}" y1="${yAxis - 4}" x2="${x}" y2="${yAxis}" stroke="#384355" stroke-width="1" />
      <text x="${x}" y="${yAxis + 12}" fill="#8B95A7" font-size="9" font-family="monospace" text-anchor="middle">${val.toFixed(1)}</text>
    `;
  }).join("");

  return `
    <div class="space-y-4">
      <div class="p-3 bg-[#090C12] rounded-xl border border-[#252C38] flex flex-col items-center">
        <div class="w-full flex items-center justify-between text-[11px] text-[#8B95A7] font-sans mb-2">
          <span>Tukey Box Plot Dispersion Analysis</span>
          <div class="flex items-center space-x-3 text-xs">
            <span class="flex items-center space-x-1"><span class="w-2.5 h-0.5 bg-white inline-block"></span><span>Median</span></span>
            <span class="flex items-center space-x-1"><span class="w-2 h-2 rotate-45 bg-amber-400 inline-block"></span><span>Mean</span></span>
            <span class="flex items-center space-x-1"><span class="w-3 h-2 bg-[#4F67FF]/30 border border-[#4F67FF] inline-block"></span><span>IQR Range</span></span>
          </div>
        </div>

        <svg viewBox="0 0 ${svgWidth} ${svgHeight}" class="w-full max-w-2xl overflow-visible">
          <line x1="${padL}" y1="${yAxis}" x2="${padL + plotW}" y2="${yAxis}" stroke="#384355" stroke-width="1" />
          ${ticks}
          ${svgRows}
        </svg>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full wb-table text-left text-xs">
          <thead>
            <tr>
              <th>Segment / Class</th>
              <th>Min</th>
              <th>Q1 (25%)</th>
              <th>Median (50%)</th>
              <th>Q3 (75%)</th>
              <th>Max</th>
              <th>IQR</th>
              <th>Mean</th>
            </tr>
          </thead>
          <tbody>
            ${hasTargetSplit ? byTarget.map(t => `
              <tr>
                <td class="font-mono font-bold text-[#F7F8FA]">${view.profile.target_column} = ${t.class_label}</td>
                <td class="font-mono text-[#8B95A7]">${t.min != null ? t.min.toFixed(2) : '—'}</td>
                <td class="font-mono text-[#8B95A7]">${t.q25 != null ? t.q25.toFixed(2) : '—'}</td>
                <td class="font-mono font-bold text-[#4F67FF]">${t.median != null ? t.median.toFixed(2) : '—'}</td>
                <td class="font-mono text-[#8B95A7]">${t.q75 != null ? t.q75.toFixed(2) : '—'}</td>
                <td class="font-mono text-[#8B95A7]">${t.max != null ? t.max.toFixed(2) : '—'}</td>
                <td class="font-mono text-[#8B95A7]">${t.iqr != null ? t.iqr.toFixed(2) : '—'}</td>
                <td class="font-mono text-amber-300">${t.mean != null ? t.mean.toFixed(2) : '—'}</td>
              </tr>
            `).join("") : `
              <tr>
                <td class="font-mono font-bold text-[#F7F8FA]">Global (${col.name})</td>
                <td class="font-mono text-[#8B95A7]">${bp.min != null ? bp.min.toFixed(2) : '—'}</td>
                <td class="font-mono text-[#8B95A7]">${bp.q25 != null ? bp.q25.toFixed(2) : '—'}</td>
                <td class="font-mono font-bold text-[#4F67FF]">${bp.median != null ? bp.median.toFixed(2) : '—'}</td>
                <td class="font-mono text-[#8B95A7]">${bp.q75 != null ? bp.q75.toFixed(2) : '—'}</td>
                <td class="font-mono text-[#8B95A7]">${bp.max != null ? bp.max.toFixed(2) : '—'}</td>
                <td class="font-mono text-[#8B95A7]">${bp.iqr != null ? bp.iqr.toFixed(2) : '—'}</td>
                <td class="font-mono text-amber-300">${bp.mean != null ? bp.mean.toFixed(2) : '—'}</td>
              </tr>
            `}
          </tbody>
        </table>
      </div>

      ${hasTargetSplit && byTarget.length === 2 ? `
        <div class="p-3 bg-[#151B26] border border-[#252C38] rounded-xl text-xs space-y-1">
          <span class="font-bold text-[#4F67FF] font-sans">Bivariate Target Interpretation:</span>
          <p class="text-[#8B95A7] font-sans">
            Inter-class median delta is <span class="font-mono font-bold text-[#22C55E]">${Math.abs(byTarget[1].median - byTarget[0].median).toFixed(2)}</span>
            ${Math.abs(byTarget[1].median - byTarget[0].median) > (bp.iqr * 0.25) ?
              '(strong distribution shift, high discriminative power for tree ensembles).' :
              '(moderate overlap between distributions; combining with interaction features recommended).'
            }
          </p>
        </div>
      ` : ""}
    </div>
  `;
}

export function _renderHistogramContent(view, col) {
  const isNumeric = col.dtype && (col.dtype.includes("int") || col.dtype.includes("float"));
  if (isNumeric && col.histogram && col.histogram.bins && col.histogram.bins.length > 0) {
    const h = col.histogram;
    const maxCount = Math.max(...h.counts, 1);

    return `
      <div class="space-y-4">
        <div class="flex items-center justify-between text-xs">
          <span class="text-[#F7F8FA] font-medium font-sans">Frequency Histogram (10 Equidistant Bins)</span>
          ${col.skew != null ? `
            <span class="font-mono text-xs px-2 py-0.5 rounded font-bold ${Math.abs(col.skew) > 1.0 ? 'badge-warn' : 'badge-neutral'}">
              Skewness: ${col.skew.toFixed(2)}
            </span>
          ` : ""}
        </div>

        <div class="p-4 bg-[#090C12] rounded-xl border border-[#252C38] space-y-2">
          <div class="h-44 flex items-end justify-between gap-1.5 pt-6 pb-2 border-b border-[#252C38]">
            ${h.bins.map((binLabel, i) => {
              const count = h.counts[i];
              const pct = h.percentages ? h.percentages[i] : ((count / (col.null_count + count)) * 100).toFixed(1);
              const barHeight = Math.max(4, Math.round((count / maxCount) * 100));

              return `
                <div class="flex-1 flex flex-col items-center h-full justify-end group relative">
                  <span class="opacity-0 group-hover:opacity-100 transition-opacity absolute -top-5 text-[10px] font-mono text-[#4F67FF] bg-[#151B26] px-1 rounded border border-[#252C38] whitespace-nowrap z-10">
                    ${count.toLocaleString()} (${pct}%)
                  </span>
                  <div style="height: ${barHeight}%;" class="w-full bg-[#4F67FF]/70 hover:bg-[#4F67FF] rounded-t transition-all"></div>
                </div>
              `;
            }).join("")}
          </div>

          <div class="flex items-center justify-between text-[9px] text-[#8B95A7] font-mono">
            <span class="truncate max-w-[80px]">${h.bins[0]}</span>
            <span class="truncate max-w-[80px]">${h.bins[Math.floor(h.bins.length / 2)]}</span>
            <span class="truncate max-w-[80px]">${h.bins[h.bins.length - 1]}</span>
          </div>
        </div>

        <div class="p-3 bg-[#151B26] border border-[#252C38] rounded-xl text-xs space-y-1">
          <span class="font-bold text-[#F7F8FA] font-sans">Distribution & Skewness Diagnostics:</span>
          <p class="text-[#8B95A7] font-sans">
            ${col.skew == null ? 'Quantitative distribution without computed skewness.' :
              Math.abs(col.skew) <= 0.5 ? 'Approximately symmetric and bell-shaped. Optimal for linear models and tree ensembles without transformation.' :
              col.skew > 0.5 ? `Positive skewness (+${col.skew.toFixed(2)}). Values concentrate on the left with a long right tail. CATML suggests log1p(x) transformation to stabilize variance.` :
              `Negative skewness (${col.skew.toFixed(2)}). Long left tail. Power transformation or robust scaling recommended.`
            }
          </p>
        </div>
      </div>
    `;
  } else if (col.top_categories && col.top_categories.length > 0) {
    return `
      <div class="space-y-4">
        <div class="text-xs text-[#F7F8FA] font-medium font-sans">Categorical Frequency Distribution</div>
        <div class="space-y-2.5 p-4 bg-[#090C12] rounded-xl border border-[#252C38]">
          ${col.top_categories.map(c => `
            <div class="space-y-1">
              <div class="flex items-center justify-between text-xs">
                <span class="font-mono font-medium text-[#F7F8FA]">${c.value}</span>
                <span class="font-mono text-[#8B95A7]">${c.count.toLocaleString()} (${c.pct}%)</span>
              </div>
              <div class="h-1.5 w-full bg-[#151B26] rounded-full overflow-hidden">
                <div class="h-full bg-[#6956E8] rounded-full" style="width: ${c.pct}%;"></div>
              </div>
            </div>
          `).join("")}
        </div>
      </div>
    `;
  }

  return `
    <div class="p-6 text-center text-[#8B95A7] text-xs font-sans">
      No distribution or frequency data available for this feature.
    </div>
  `;
}

export function _renderTargetPatternContent(view, col) {
  const isTarget = col.name === view.profile.target_column;
  if (isTarget) {
    return `
      <div class="p-6 text-center text-[#8B95A7] text-xs font-sans">
        This column is the primary target variable.
      </div>
    `;
  }

  const isNumeric = col.dtype && (col.dtype.includes("int") || col.dtype.includes("float"));

  if (!isNumeric && col.top_categories && col.top_categories.length > 0) {
    const hasTargetRate = col.top_categories.some(c => c.target_rate != null);
    if (hasTargetRate) {
      return `
        <div class="space-y-4">
          <div class="flex items-center justify-between text-xs">
            <span class="text-[#F7F8FA] font-medium font-sans">Target Incidence Rate (%) by Category</span>
            <span class="text-[11px] font-mono text-[#4F67FF]">Propensity pattern</span>
          </div>

          <div class="space-y-3 p-4 bg-[#090C12] rounded-xl border border-[#252C38]">
            ${col.top_categories.map(c => {
              const rate = c.target_rate != null ? c.target_rate : 0.0;
              let colorClass = "bg-[#4F67FF]";
              if (rate >= 60) colorClass = "bg-[#22C55E]";
              else if (rate <= 40) colorClass = "bg-[#EF4444]";

              return `
                <div class="space-y-1">
                  <div class="flex items-center justify-between text-xs">
                    <span class="font-mono font-medium text-[#F7F8FA]">${c.value}</span>
                    <span class="font-mono font-bold ${rate >= 50 ? 'text-[#22C55E]' : 'text-[#8B95A7]'}">
                      ${rate.toFixed(1)}% positive rate (${c.count.toLocaleString()} samples)
                    </span>
                  </div>
                  <div class="h-2 w-full bg-[#151B26] rounded-full overflow-hidden flex">
                    <div class="h-full ${colorClass} rounded-full transition-all" style="width: ${rate}%;"></div>
                  </div>
                </div>
              `;
            }).join("")}
          </div>

          <div class="p-3 bg-[#151B26] border border-[#252C38] rounded-xl text-xs space-y-1">
            <span class="font-bold text-[#F7F8FA] font-sans">Modeling Utility:</span>
            <p class="text-[#8B95A7] font-sans">
              Variation in positive rate across categories demonstrates discriminative power. CATML will apply out-of-fold Target Encoding to leverage propensity differences without data leakage.
            </p>
          </div>
        </div>
      `;
    }
  } else if (isNumeric) {
    const corr = col.target_correlation;
    const bp = col.box_plot;
    const byTarget = bp?.by_target || [];

    return `
      <div class="space-y-4">
        <div class="grid grid-cols-2 gap-3 text-xs">
          <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-1">
            <span class="text-[#8B95A7] text-[10px] block font-sans uppercase">Pearson Correlation with Target</span>
            <span class="font-mono text-base font-bold ${corr != null && Math.abs(corr) >= 0.25 ? 'text-[#22C55E]' : 'text-[#F7F8FA]'}">
              ${corr != null ? (corr > 0 ? '+' : '') + corr.toFixed(4) : 'Not computed'}
            </span>
            <p class="text-[10px] text-[#8B95A7] font-sans">
              ${corr != null && Math.abs(corr) >= 0.4 ? 'Strong correlation: prime predictive candidate.' :
                corr != null && Math.abs(corr) >= 0.2 ? 'Moderate correlation: useful predictor.' :
                'Low linear correlation: tree models will extract non-linear patterns.'
              }
            </p>
          </div>

          <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-1">
            <span class="text-[#8B95A7] text-[10px] block font-sans uppercase">Interquartile Separation Across Classes</span>
            <span class="font-mono text-base font-bold text-[#4F67FF]">
              ${byTarget.length === 2 ? Math.abs(byTarget[1].median - byTarget[0].median).toFixed(2) : (bp?.iqr != null ? bp.iqr.toFixed(2) : '—')}
            </span>
            <p class="text-[10px] text-[#8B95A7] font-sans">
              ${byTarget.length === 2 ? 'Absolute delta between class 0 and class 1 medians.' : 'Interquartile range (IQR = Q75 - Q25).'}
            </p>
          </div>
        </div>

        ${byTarget.length >= 2 ? `
          <div class="p-3.5 bg-[#090C12] rounded-xl border border-[#252C38] space-y-2">
            <span class="text-xs font-semibold text-[#F7F8FA] font-sans">Target Class Distribution Comparison</span>
            <div class="space-y-2">
              ${byTarget.map(t => `
                <div class="flex items-center justify-between text-xs font-mono">
                  <span class="text-[#8B95A7]">${view.profile.target_column} = ${t.class_label}:</span>
                  <span class="font-bold text-[#F7F8FA]">Mean = ${t.mean.toFixed(2)} • Median = ${t.median.toFixed(2)} (${t.count.toLocaleString()} rows)</span>
                </div>
              `).join("")}
            </div>
          </div>
        ` : ""}
      </div>
    `;
  }

  return `
    <div class="p-6 text-center text-[#8B95A7] text-xs font-sans">
      Insufficient bivariate data to correlate with target.
    </div>
  `;
}

