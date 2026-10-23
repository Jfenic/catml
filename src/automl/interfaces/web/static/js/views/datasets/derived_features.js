import { api } from "../../api.js";
import { icon } from "../../icons.js";

export function _renderCalculatorTab(view, p) {
  const columns = p.columns || [];
  const numCols = columns.filter(c => c.dtype && (c.dtype.includes("int") || c.dtype.includes("float")));
  const catCols = columns.filter(c => !numCols.includes(c));
  const evalRes = view.calcEvaluationResult;

  return `
    <div class="p-6 space-y-6">
      <!-- Header Banner -->
      <div class="p-4 rounded-xl border border-[#252C38] bg-[#151B26] flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div class="space-y-1">
          <div class="flex items-center space-x-2">
            <span class="text-[#4F67FF]">${icon("sparkles", "icon-sm")}</span>
            <h3 class="text-sm font-semibold text-[#F7F8FA] font-sans">Feature Calculator & Derived Feature Engine</h3>
          </div>
          <p class="text-xs text-[#8B95A7] font-sans">
            Engineers complex tabular features via mathematical formulas or sandboxed Python expressions with division-by-zero isolation and type validation.
          </p>
        </div>

        <div class="flex items-center space-x-2">
          <button id="btnCalcModeFormula" class="px-3 py-1.5 text-xs font-sans rounded-md border transition-colors ${view.calcMode === 'formula' ? 'bg-[#4F67FF] text-white border-[#4F67FF] font-semibold' : 'bg-[#10151E] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">
            Formula Mode
          </button>
          <button id="btnCalcModePython" class="px-3 py-1.5 text-xs font-sans rounded-md border transition-colors ${view.calcMode === 'python_code' ? 'bg-[#4F67FF] text-white border-[#4F67FF] font-semibold' : 'bg-[#10151E] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">
            Python Mode
          </button>
          <button id="btnCalcSuggest" class="btn-technical text-xs text-[#6956E8] border-[#6956E8]/30 hover:border-[#6956E8] flex items-center space-x-1.5">
            <span>${icon("sparkles", "icon-sm")}</span>
            <span>Sugerir con IA</span>
          </button>
        </div>
      </div>

      <!-- Inputs Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Left Column: Name & Expression Editor -->
        <div class="lg:col-span-2 space-y-4">
          <div class="space-y-1.5">
            <label class="text-xs font-sans text-[#8B95A7] uppercase font-semibold tracking-wider">Nombre de Característica Derivada:</label>
            <input type="text" id="calcFeatureName" value="${view.calcFeatureName || ''}" placeholder="ej: debt_to_income_ratio" class="w-full bg-[#090C12] border border-[#252C38] text-[#F7F8FA] text-xs font-mono px-3 py-2 rounded-lg focus:border-[#4F67FF] outline-none">
          </div>

          <div class="space-y-1.5">
            <div class="flex items-center justify-between">
              <label class="text-xs font-sans text-[#8B95A7] uppercase font-semibold tracking-wider">
                ${view.calcMode === 'formula' ? 'Fórmula Matemática:' : 'Función Python Aislada (df):'}
              </label>
              <span class="text-[10px] text-[#8B95A7] font-mono">
                ${view.calcMode === 'formula' ? 'Protección contra división por cero activa' : 'Restringido a numpy y pandas'}
              </span>
            </div>
            <textarea id="calcExpression" rows="${view.calcMode === 'python_code' ? 7 : 4}" placeholder="${view.calcMode === 'python_code' ? 'def compute_feature(df):\n    ratio = df[\'col_a\'] / (df[\'col_b\'] + 1e-5)\n    return np.log1p(ratio)' : 'ej: debt / (income + 1e-5)'}" class="w-full bg-[#090C12] border border-[#252C38] text-[#F7F8FA] text-xs font-mono p-3 rounded-lg focus:border-[#4F67FF] outline-none">${view.calcExpression || ''}</textarea>
          </div>

          <!-- Quick Operators Toolbar (Formula Mode) -->
          ${view.calcMode === 'formula' ? `
            <div class="space-y-2">
              <span class="text-[11px] font-sans text-[#8B95A7] font-semibold uppercase">Operadores Rápidos y Transformaciones:</span>
              <div class="flex flex-wrap gap-1.5 text-xs font-mono">
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op=" + ">+</button>
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op=" - ">-</button>
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op=" * ">*</button>
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op=" / ">/</button>
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op=" ** 2">**2</button>
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="log1p()">log1p()</button>
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="sqrt()">sqrt()</button>
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="clip(col, 0, 100)">clip()</button>
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="zscore()">zscore()</button>
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="if_else(cond, x, y)">if_else()</button>
                <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="fillna(col, 0)">fillna()</button>
              </div>
            </div>
          ` : ""}

          <!-- Action CTAs -->
          <div class="flex items-center space-x-3 pt-2">
            <button id="btnCalcValidate" class="btn-technical text-xs font-semibold">
              <span>Probar y Previsualizar</span>
            </button>
            <button id="btnCalcApply" class="btn-signal text-xs ${evalRes && evalRes.is_valid ? '' : 'opacity-50 cursor-not-allowed'}" ${evalRes && evalRes.is_valid ? '' : 'disabled'}>
              <span>Aplicar al Dataset y Guardar</span>
            </button>
          </div>
        </div>

        <!-- Right Column: Available Column Chips -->
        <div class="space-y-3 p-4 rounded-xl border border-[#252C38] bg-[#090C12]">
          <div class="flex items-center justify-between border-b border-[#252C38] pb-2">
            <span class="text-xs font-sans text-[#F7F8FA] font-semibold uppercase tracking-wider">Características Disponibles</span>
            <span class="text-[10px] font-mono text-[#8B95A7]">Clic para insertar</span>
          </div>
          <div class="space-y-2 max-h-64 overflow-y-auto pr-1">
            <div class="text-[10px] font-sans text-[#8B95A7] font-semibold uppercase">Numéricas:</div>
            <div class="flex flex-wrap gap-1.5">
              ${numCols.map(c => `
                <button class="calc-col-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#4F67FF] font-mono text-xs hover:border-[#4F67FF] hover:text-white transition-colors" data-col="${c.name}">
                  ${c.name}
                </button>
              `).join("")}
            </div>

            ${catCols.length > 0 ? `
              <div class="text-[10px] font-sans text-[#8B95A7] font-semibold uppercase pt-2">Categóricas / Otras:</div>
              <div class="flex flex-wrap gap-1.5">
                ${catCols.map(c => `
                  <button class="calc-col-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#6956E8] font-mono text-xs hover:border-[#4F67FF] hover:text-white transition-colors" data-col="${c.name}">
                    ${c.name}
                  </button>
                `).join("")}
              </div>
            ` : ""}
          </div>
        </div>
      </div>

      <!-- Evaluation Results & Diagnostics Card -->
      ${evalRes ? `
        <div class="p-5 rounded-xl border ${evalRes.is_valid ? 'border-emerald-800/60 bg-emerald-950/15' : 'border-rose-800/60 bg-rose-950/15'} space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#252C38] pb-3">
            <div class="flex items-center space-x-3">
              <span class="text-xs px-2.5 py-1 rounded-md font-mono font-bold uppercase inline-flex items-center space-x-1.5 ${evalRes.is_valid ? 'badge-gain' : 'badge-err'}">
                ${evalRes.is_valid ? `${icon("check", "icon-sm")}<span>FÓRMULA VÁLIDA</span>` : `${icon("x", "icon-sm")}<span>ERROR DE VALIDACIÓN</span>`}
              </span>
              <span class="text-sm font-bold font-mono text-[#F7F8FA]">${evalRes.feature_name}</span>
              <span class="text-xs font-mono text-[#8B95A7]">(${evalRes.dtype || 'unknown'})</span>
            </div>

            <div class="flex items-center space-x-2 text-xs font-mono">
              <span class="px-2 py-0.5 rounded-md bg-[#090C12] border border-[#252C38] text-[#8B95A7]">
                Filas: <strong class="text-[#F7F8FA]">${evalRes.row_count}</strong>
              </span>
              <span class="px-2 py-0.5 rounded-md bg-[#090C12] border border-[#252C38] ${evalRes.null_percentage > 0.2 ? 'text-amber-400' : 'text-[#8B95A7]'}">
                Nulos: <strong>${evalRes.null_count} (${(evalRes.null_percentage * 100).toFixed(1)}%)</strong>
              </span>
              <span class="px-2 py-0.5 rounded-md bg-[#090C12] border border-[#252C38] ${evalRes.zero_division_occurred ? 'text-[#22C55E] font-bold' : 'text-[#8B95A7]'}">
                División por cero: ${evalRes.zero_division_occurred ? 'Protegida (0.0/eps)' : 'Ninguna'}
              </span>
            </div>
          </div>

          ${evalRes.error_message ? `
            <div class="p-3 rounded-lg bg-rose-950/30 border border-rose-800/60 text-rose-300 font-mono text-xs space-y-1">
              <div class="font-bold">Detalles del Fallo de Validación:</div>
              <div>${evalRes.error_message}</div>
            </div>
          ` : ""}

          ${evalRes.warnings && evalRes.warnings.length > 0 ? `
            <div class="space-y-1">
              ${evalRes.warnings.map(w => `
                <div class="text-xs font-mono text-amber-300 flex items-center space-x-1.5">
                  <span class="text-[#F59E0B]">${icon("triangle-alert", "icon-sm")}</span>
                  <span>${w}</span>
                </div>
              `).join("")}
            </div>
          ` : ""}

          ${evalRes.is_valid ? `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono pt-1">
              <!-- Summary Stats -->
              <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#252C38] space-y-2">
                <div class="text-[10px] text-[#8B95A7] uppercase font-sans font-semibold tracking-wider">Estadísticas Descriptivas:</div>
                <div class="grid grid-cols-4 gap-2 text-center">
                  <div class="p-1.5 rounded-lg bg-[#151B26]">
                    <div class="text-[10px] text-[#8B95A7]">Mín</div>
                    <div class="font-bold text-[#F7F8FA]">${evalRes.summary_stats?.min != null ? evalRes.summary_stats.min.toFixed(3) : '-'}</div>
                  </div>
                  <div class="p-1.5 rounded-lg bg-[#151B26]">
                    <div class="text-[10px] text-[#8B95A7]">Media</div>
                    <div class="font-bold text-[#F7F8FA]">${evalRes.summary_stats?.mean != null ? evalRes.summary_stats.mean.toFixed(3) : '-'}</div>
                  </div>
                  <div class="p-1.5 rounded-lg bg-[#151B26]">
                    <div class="text-[10px] text-[#8B95A7]">Std</div>
                    <div class="font-bold text-[#F7F8FA]">${evalRes.summary_stats?.std != null ? evalRes.summary_stats.std.toFixed(3) : '-'}</div>
                  </div>
                  <div class="p-1.5 rounded-lg bg-[#151B26]">
                    <div class="text-[10px] text-[#8B95A7]">Máx</div>
                    <div class="font-bold text-[#F7F8FA]">${evalRes.summary_stats?.max != null ? evalRes.summary_stats.max.toFixed(3) : '-'}</div>
                  </div>
                </div>
              </div>

              <!-- Sample Preview -->
              <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#252C38] space-y-2">
                <div class="text-[10px] text-[#8B95A7] uppercase font-sans font-semibold tracking-wider">Valores Calculados de Muestra (Primeras 5 Filas):</div>
                <div class="flex items-center space-x-2">
                  ${(evalRes.sample_values || []).map((val, idx) => `
                    <span class="px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#F7F8FA] font-bold">
                      #${idx + 1}: ${val != null ? val : 'NaN'}
                    </span>
                  `).join("")}
                </div>
              </div>
            </div>
          ` : ""}
        </div>
      ` : ""}

      <!-- AI Feature Suggestions Card -->
      ${view.calcSuggestions && view.calcSuggestions.length > 0 ? `
        <div class="p-4 rounded-xl border border-indigo-900/40 bg-indigo-950/15 space-y-3">
          <div class="flex items-center justify-between border-b border-indigo-900/30 pb-2">
            <span class="text-xs font-sans font-semibold text-indigo-300 uppercase tracking-wider">Hipótesis de Características Propuestas por IA</span>
            <span class="text-[10px] font-mono text-[#8B95A7]">${view.calcSuggestions.length} propuestas validadas</span>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            ${view.calcSuggestions.map(sugg => `
              <div class="p-3 rounded-xl bg-[#090C12] border border-[#252C38] space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-mono font-bold text-indigo-300">${sugg.definition.name}</span>
                  <button class="btn-load-suggestion px-2.5 py-0.5 rounded-md bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-200 border border-indigo-500/40 text-[11px] font-sans font-medium inline-flex items-center space-x-1" data-name="${sugg.definition.name}" data-expr="${encodeURIComponent(sugg.definition.expression)}" data-type="${sugg.definition.expression_type}">
                    <span>Cargar en Calculadora</span>
                    <span>${icon("arrow-right", "icon-sm")}</span>
                  </button>
                </div>
                <div class="text-[11px] font-mono text-[#F7F8FA] bg-[#151B26] px-2 py-1 rounded-md border border-[#252C38] overflow-x-auto">
                  ${sugg.definition.expression}
                </div>
                <p class="text-[10px] text-[#8B95A7] font-sans">${sugg.definition.description || ''}</p>
              </div>
            `).join("")}
          </div>
        </div>
      ` : ""}
    </div>
  `;
}

export function _bindCalculatorEvents(view) {
  // Mode toggles
  view.container.querySelector("#btnCalcModeFormula")?.addEventListener("click", () => {
    view.calcMode = "formula";
    view.calcEvaluationResult = null;
    view.render();
  });

  view.container.querySelector("#btnCalcModePython")?.addEventListener("click", () => {
    view.calcMode = "python_code";
    view.calcEvaluationResult = null;
    view.render();
  });

  // Column chips insertion
  view.container.querySelectorAll(".calc-col-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      const col = btn.getAttribute("data-col");
      const exprArea = view.container.querySelector("#calcExpression");
      if (exprArea) {
        exprArea.value += (exprArea.value.length && !exprArea.value.endsWith(" ") ? " " : "") + col;
        view.calcExpression = exprArea.value;
      }
    });
  });

  // Operator chips insertion
  view.container.querySelectorAll(".calc-op-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      const op = btn.getAttribute("data-op");
      const exprArea = view.container.querySelector("#calcExpression");
      if (exprArea) {
        exprArea.value += op;
        view.calcExpression = exprArea.value;
      }
    });
  });

  // Validate & Preview
  view.container.querySelector("#btnCalcValidate")?.addEventListener("click", async () => {
    const name = view.container.querySelector("#calcFeatureName")?.value.trim();
    const expr = view.container.querySelector("#calcExpression")?.value.trim();
    if (!name || !expr) {
      alert("Please enter both the feature name and expression.");
      return;
    }
    view.calcFeatureName = name;
    view.calcExpression = expr;
    try {
      const res = await api.calculateDerivedFeature({
        dataset_id: view.activeDatasetId,
        name: name,
        expression: expr,
        expression_type: view.calcMode,
      });
      view.calcEvaluationResult = res.result;
      view.render();
    } catch (err) {
      alert(`Validation error: ${err.message}`);
    }
  });

  // Apply & Save to Dataset
  view.container.querySelector("#btnCalcApply")?.addEventListener("click", async () => {
    const name = view.calcFeatureName || view.container.querySelector("#calcFeatureName")?.value.trim();
    const expr = view.calcExpression || view.container.querySelector("#calcExpression")?.value.trim();
    if (!name || !expr) return;

    const btn = view.container.querySelector("#btnCalcApply");
    if (btn) btn.textContent = "Applying to Dataset...";

    try {
      await api.applyDerivedFeature({
        dataset_id: view.activeDatasetId,
        name: name,
        expression: expr,
        expression_type: view.calcMode,
      });

      // Refetch profile so the new feature is included
      await view.fetchData();
      view.selectedFeatures.add(name);
      view.activeTab = "schema";
      view.calcEvaluationResult = null;
      view.render();
    } catch (err) {
      alert(`Failed to apply derived feature: ${err.message}`);
      view.render();
    }
  });

  // AI Suggestions
  view.container.querySelector("#btnCalcSuggest")?.addEventListener("click", async () => {
    const btn = view.container.querySelector("#btnCalcSuggest");
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `<span class="animate-spin inline-block mr-1">${icon("refresh-cw", "icon-sm")}</span><span>Analyzing...</span>`;
    }
    try {
      const res = await api.suggestDerivedFeatures(view.activeDatasetId);
      view.calcSuggestions = res.suggestions || [];
      view.render();
    } catch (err) {
      alert(`Failed to generate suggestions: ${err.message}`);
      view.render();
    }
  });

  // Load suggestion into calculator
  view.container.querySelectorAll(".btn-load-suggestion").forEach(btn => {
    btn.addEventListener("click", () => {
      view.calcFeatureName = btn.getAttribute("data-name");
      view.calcExpression = decodeURIComponent(btn.getAttribute("data-expr"));
      view.calcMode = btn.getAttribute("data-type") || "formula";
      view.calcEvaluationResult = null;
      view.render();
    });
  });

  // Quick toolbar button jump to calculator
  view.container.querySelector("#btnOpenCalculator")?.addEventListener("click", () => {
    view.activeTab = "calculator";
    view.render();
  });
}

