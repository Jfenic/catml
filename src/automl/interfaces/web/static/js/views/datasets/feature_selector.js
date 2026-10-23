import { store } from "../../store.js";

export function _getFilteredColumns(view, columns) {
  return columns.filter(c => {
    // Search filter
    if (view.searchQuery && !c.name.toLowerCase().includes(view.searchQuery.toLowerCase())) {
      return false;
    }
    // Type / Status filter
    if (view.filterType === "numeric") {
      return c.dtype && (c.dtype.includes("int") || c.dtype.includes("float"));
    } else if (view.filterType === "categorical") {
      return c.dtype && (c.dtype === "object" || c.dtype === "category" || c.dtype === "string") && !c.is_image;
    } else if (view.filterType === "image") {
      return Boolean(c.is_image);
    } else if (view.filterType === "selected") {
      return view.selectedFeatures.has(c.name);
    } else if (view.filterType === "excluded") {
      return !view.selectedFeatures.has(c.name) && c.name !== view.profile?.target_column;
    }
    return true;
  });
}

export function _updateSelectionUi(view) {
  const p = view.profile;
  if (!p) return;
  const featureCols = (p.columns || []).filter(c => c.name !== p.target_column);

  // Save to store so NewExperimentModal immediately knows active selection
  const feats = Array.from(view.selectedFeatures);
  const currentMap = store.getState().selectedFeaturesByDataset || {};
  store.setState({
    selectedFeatures: feats,
    customFeatures: feats,
    selectedFeaturesByDataset: { ...currentMap, [view.activeDatasetId]: feats },
  });

  const liveCount = view.container.querySelector("#selectionLiveCount");
  if (liveCount) {
    liveCount.innerHTML = `<strong class="text-[#4F67FF]">${view.selectedFeatures.size}</strong> of ${featureCols.length} features selected (${featureCols.length - view.selectedFeatures.size} excluded)`;
  }

  const pillCount = view.container.querySelector("#pillSelectedCount");
  if (pillCount) {
    pillCount.textContent = view.selectedFeatures.size;
  }

  const ctaCount = view.container.querySelector("#ctaSelectedCount");
  if (ctaCount) {
    ctaCount.textContent = view.selectedFeatures.size;
  }

  // Update checkboxes in table
  view.container.querySelectorAll(".col-toggle-checkbox").forEach(cb => {
    const colName = cb.getAttribute("data-col");
    cb.checked = view.selectedFeatures.has(colName);
    const row = cb.closest("tr");
    if (row) {
      if (cb.checked) {
        row.classList.add("bg-indigo-950/10");
        row.classList.remove("opacity-75");
      } else {
        row.classList.remove("bg-indigo-950/10");
        row.classList.add("opacity-75");
      }
    }
  });
}

export function _initSelectedFeatures(view) {
  if (!view.profile) return;
  const target = view.profile.target_column;
  const columns = view.profile.columns || [];

  // Initialize with recommended features (non-identifier, non-target)
  view.selectedFeatures.clear();
  columns.forEach(col => {
    if (col.name !== target && !col.is_identifier && col.catml_action !== "Exclude") {
      view.selectedFeatures.add(col.name);
    }
  });

  // Save to store so NewExperimentModal immediately knows active selection
  const feats = Array.from(view.selectedFeatures);
  const dsFeats = store.getState().selectedFeaturesByDataset || {};
  if (view.activeDatasetId) {
    dsFeats[view.activeDatasetId] = feats;
  }
  store.setState({
    selectedFeatures: feats,
    customFeatures: feats,
    selectedFeaturesByDataset: dsFeats,
  });
}

