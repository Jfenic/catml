from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    RunExperimentCommand,
)
from automl.application.queries.workspace_queries import (
    GetLeaderboardQuery,
    ListPluginsQuery,
)
from automl.domain.plugins.plugin import PluginType
from automl.domain.tasks.task_type import TaskType, models_for_task
from automl.engine.optimization.search_space_builder import SearchSpaceBuilder
from automl.plugins.models.catboost_plugin import CatBoostPlugin
from automl.plugins.models.gradient_boosting import (
    CatBoostPlugin as ReExportedCatBoostPlugin,
    LightGBMPlugin,
    XGBoostPlugin,
)
from automl.plugins.models.sklearn_models import (
    ExtraTreesPlugin,
    MLPPlugin,
    build_sklearn_model,
    default_model_specs,
)


@pytest.fixture
def sample_classification_csv(tmp_path: Path) -> str:
    csv_path = tmp_path / "classification_sample.csv"
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(10, 100, size=n)
    x2 = np.random.uniform(1, 50, size=n)
    target = ((x1 * 0.6 + x2 * 0.4) > 40).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": target})
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_regression_csv(tmp_path: Path) -> str:
    csv_path = tmp_path / "regression_sample.csv"
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(10, 100, size=n)
    x2 = np.random.uniform(1, 50, size=n)
    target = x1 * 2.5 + x2 * 1.5 + np.random.normal(0, 1, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": target})
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_catboost_plugin_properties_and_reexport() -> None:
    cb = CatBoostPlugin()
    assert cb.plugin_id == "catboost"
    assert cb.name == "CatBoost Gradient Boosting"
    assert cb.version == "1.0.0"
    assert cb.plugin_type == PluginType.MODEL
    assert cb.capabilities.supported_tasks == [
        "binary_classification",
        "multiclass_classification",
        "regression",
    ]
    assert cb.capabilities.supported_modalities == ["tabular"]
    assert cb.capabilities.supports_proba is True
    assert isinstance(cb.is_available, bool)
    assert cb is not None
    assert ReExportedCatBoostPlugin is CatBoostPlugin


def test_catboost_fallback_estimator_classification_and_regression() -> None:
    with patch.object(CatBoostPlugin, "is_available", False):
        cb = CatBoostPlugin(use_fallback_if_missing=True)

        # Binary classification fallback
        clf = cb.build_estimator(
            parameters={"iterations": 80, "learning_rate": 0.05, "depth": 4},
            task_type="binary_classification",
        )
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        assert isinstance(clf, HistGradientBoostingClassifier)
        assert clf.max_iter == 80
        assert clf.learning_rate == 0.05
        assert clf.max_depth == 4

        X = np.random.normal(size=(40, 3))
        y_bin = (X[:, 0] > 0).astype(int)
        clf.fit(X, y_bin)
        preds = clf.predict(X)
        assert len(preds) == 40
        probas = clf.predict_proba(X)
        assert probas.shape == (40, 2)

        # Multiclass classification fallback
        y_multi = np.random.choice([0, 1, 2], size=40)
        clf_multi = cb.build_estimator(task_type="multiclass_classification")
        assert isinstance(clf_multi, HistGradientBoostingClassifier)
        clf_multi.fit(X, y_multi)
        probas_multi = clf_multi.predict_proba(X)
        assert probas_multi.shape == (40, 3)

        # Regression fallback
        y_reg = X[:, 0] * 3.0 + X[:, 1]
        reg = cb.build_estimator(
            parameters={"n_estimators": 60, "learning_rate": 0.08},
            task_type="regression",
        )
        assert isinstance(reg, HistGradientBoostingRegressor)
        assert reg.max_iter == 60
        assert reg.learning_rate == 0.08
        reg.fit(X, y_reg)
        reg_preds = reg.predict(X)
        assert len(reg_preds) == 40


def test_catboost_missing_without_fallback_raises_importerror() -> None:
    cb = CatBoostPlugin(use_fallback_if_missing=False)
    with patch.object(CatBoostPlugin, "is_available", False):
        with pytest.raises(ImportError, match="CatBoost is not installed"):
            cb.build_estimator(task_type="binary_classification")


def test_catboost_estimator_with_mocked_native_library() -> None:
    mock_cb_module = MagicMock()
    mock_classifier_cls = MagicMock()
    mock_regressor_cls = MagicMock()
    mock_cb_module.CatBoostClassifier = mock_classifier_cls
    mock_cb_module.CatBoostRegressor = mock_regressor_cls

    with patch.dict("sys.modules", {"catboost": mock_cb_module}):
        with patch.object(CatBoostPlugin, "is_available", True):
            cb = CatBoostPlugin()

            # Classification
            cb.build_estimator(
                parameters={"n_estimators": 150, "learning_rate": 0.03},
                task_type="binary_classification",
            )
            mock_classifier_cls.assert_called_once_with(
                random_seed=42, verbose=0, iterations=150, learning_rate=0.03
            )

            # Regression
            cb.build_estimator(
                parameters={"iterations": 200, "depth": 5},
                task_type="regression",
            )
            mock_regressor_cls.assert_called_once_with(
                random_seed=42, verbose=0, iterations=200, depth=5
            )


def test_extra_trees_plugin() -> None:
    et = ExtraTreesPlugin()
    assert et.plugin_id == "extra_trees"
    assert et.name == "Extra Trees (Extremely Randomized Trees)"
    assert et.capabilities.supported_tasks == [
        "binary_classification",
        "multiclass_classification",
        "regression",
    ]
    assert et.capabilities.supports_proba is True

    # Build classification
    clf = et.build_estimator(
        parameters={"n_estimators": 25, "max_depth": 5},
        task_type="binary_classification",
    )
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    assert isinstance(clf, ExtraTreesClassifier)
    assert clf.n_estimators == 25
    assert clf.max_depth == 5

    X = np.random.normal(size=(30, 3))
    y = (X[:, 0] > 0).astype(int)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 30
    probas = clf.predict_proba(X)
    assert probas.shape == (30, 2)

    # Build regression
    reg = et.build_estimator(parameters={"n_estimators": 20}, task_type="regression")
    assert isinstance(reg, ExtraTreesRegressor)
    reg.fit(X, y.astype(float))
    assert len(reg.predict(X)) == 30

    # Search space
    space = et.get_search_space("binary_classification")
    assert "n_estimators" in space
    assert "max_depth" in space
    assert "min_samples_split" in space


def test_mlp_plugin() -> None:
    mlp = MLPPlugin()
    assert mlp.plugin_id == "mlp"
    assert mlp.name == "Multi-Layer Perceptron (MLP)"
    assert mlp.capabilities.supported_tasks == [
        "binary_classification",
        "multiclass_classification",
        "regression",
    ]
    assert mlp.capabilities.supports_proba is True

    # Build classification
    clf = mlp.build_estimator(
        parameters={"max_iter": 50, "alpha": 0.01},
        task_type="binary_classification",
    )
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    assert isinstance(clf, MLPClassifier)
    assert clf.max_iter == 50
    assert clf.alpha == 0.01

    X = np.random.normal(size=(30, 3))
    y = (X[:, 0] > 0).astype(int)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 30
    probas = clf.predict_proba(X)
    assert probas.shape == (30, 2)

    # Build regression
    reg = mlp.build_estimator(parameters={"max_iter": 50}, task_type="regression")
    assert isinstance(reg, MLPRegressor)
    reg.fit(X, y.astype(float))
    assert len(reg.predict(X)) == 30

    # Search space
    space = mlp.get_search_space("binary_classification")
    assert "alpha" in space
    assert "learning_rate_init" in space
    assert "max_iter" in space


def test_search_space_builder_for_new_models() -> None:
    # CatBoost space
    cb_space = SearchSpaceBuilder.build("catboost", "binary_classification")
    assert "iterations" in cb_space
    assert "learning_rate" in cb_space
    assert "depth" in cb_space
    sampled_cb = cb_space.sample(seed=42)
    assert 50 <= sampled_cb["iterations"] <= 300
    assert 0.01 <= sampled_cb["learning_rate"] <= 0.3
    assert 3 <= sampled_cb["depth"] <= 10

    # ExtraTrees space
    et_space = SearchSpaceBuilder.build("extra_trees", "binary_classification")
    assert "n_estimators" in et_space
    assert "max_depth" in et_space
    assert "min_samples_split" in et_space
    sampled_et = et_space.sample(seed=42)
    assert 20 <= sampled_et["n_estimators"] <= 200

    # MLP space
    mlp_space = SearchSpaceBuilder.build("mlp", "binary_classification")
    assert "alpha" in mlp_space
    assert "learning_rate_init" in mlp_space
    assert "max_iter" in mlp_space
    sampled_mlp = mlp_space.sample(seed=42)
    assert 1e-5 <= sampled_mlp["alpha"] <= 1e-1

    # Alias build_for_model
    alias_space = SearchSpaceBuilder.build_for_model("catboost")
    assert "iterations" in alias_space


def test_task_catalog_and_default_model_specs() -> None:
    # TASK_CATALOG verification
    binary_models = models_for_task(TaskType.BINARY_CLASSIFICATION)
    assert "catboost" in binary_models
    assert "extra_trees" in binary_models
    assert "mlp" in binary_models

    multi_models = models_for_task(TaskType.MULTICLASS_CLASSIFICATION)
    assert "catboost" in multi_models
    assert "extra_trees" in multi_models
    assert "mlp" in multi_models

    reg_models = models_for_task(TaskType.REGRESSION)
    assert "catboost" in reg_models
    assert "extra_trees" in reg_models
    assert "mlp" in reg_models

    clustering_models = models_for_task(TaskType.CLUSTERING)
    assert "catboost" not in clustering_models
    assert "extra_trees" not in clustering_models
    assert "mlp" not in clustering_models

    # default_model_specs verification
    specs = default_model_specs()
    spec_map = {s.id: s for s in specs}
    assert "catboost" in spec_map
    assert spec_map["catboost"].name == "CatBoost"
    assert "binary_classification" in spec_map["catboost"].task_types
    assert "regression" in spec_map["catboost"].task_types

    assert "extra_trees" in spec_map
    assert spec_map["extra_trees"].name == "Extra Trees"

    assert "mlp" in spec_map
    assert spec_map["mlp"].name == "Multi-Layer Perceptron (MLP)"


def test_build_sklearn_model_direct_dispatch() -> None:
    # Direct invocation of build_sklearn_model
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    m_cb = build_sklearn_model("catboost", "binary_classification")
    assert isinstance(m_cb, HistGradientBoostingClassifier) or m_cb.__class__.__name__ == "CatBoostClassifier"

    m_et_c = build_sklearn_model("extra_trees", "binary_classification")
    assert isinstance(m_et_c, ExtraTreesClassifier)

    m_et_r = build_sklearn_model("extra_trees", "regression")
    assert isinstance(m_et_r, ExtraTreesRegressor)

    m_mlp_c = build_sklearn_model("mlp", "binary_classification")
    assert isinstance(m_mlp_c, MLPClassifier)

    m_mlp_r = build_sklearn_model("mlp", "regression")
    assert isinstance(m_mlp_r, MLPRegressor)


def test_e2e_workspace_training_with_catboost_extra_trees_mlp(
    sample_classification_csv: str, tmp_path: Path
) -> None:
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "e2e_classification"))

    # Verify query bus lists the new plugins
    all_plugins = qry.dispatch(ListPluginsQuery(plugin_type="model"))
    plugin_ids = {p["plugin_id"] for p in all_plugins}
    assert "catboost" in plugin_ids
    assert "extra_trees" in plugin_ids
    assert "mlp" in plugin_ids

    dataset = ws.register_dataset(
        name="churn_data",
        path=sample_classification_csv,
        target="target",
    )
    run = ws.create_run(dataset, metric="roc_auc")

    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="new_models_experiment",
            feature_names=["x1", "x2"],
            model_ids=["catboost", "extra_trees", "mlp"],
            hypothesis="Evaluate newly added models",
        )
    )

    results = cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))
    assert len(results) == 3
    assert all(r.succeeded for r in results)

    lb = qry.dispatch(GetLeaderboardQuery(run.id))
    assert len(lb) == 3
    lb_models = {entry["model_id"] for entry in lb}
    assert lb_models == {"catboost", "extra_trees", "mlp"}
    for entry in lb:
        assert entry["score"] is not None
        assert entry["score"] > 0.0


def test_e2e_workspace_regression_with_catboost_extra_trees_mlp(
    sample_regression_csv: str, tmp_path: Path
) -> None:
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "e2e_regression"))

    dataset = ws.register_dataset(
        name="housing_data",
        path=sample_regression_csv,
        target="target",
        task_type="regression",
    )
    run = ws.create_run(dataset, metric="r2")

    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="regression_experiment",
            feature_names=["x1", "x2"],
            model_ids=["catboost", "extra_trees", "mlp"],
            hypothesis="Evaluate regression capability of new models",
        )
    )

    results = cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))
    assert len(results) == 3
    assert all(r.succeeded for r in results)

    lb = qry.dispatch(GetLeaderboardQuery(run.id))
    assert len(lb) == 3
    lb_models = {entry["model_id"] for entry in lb}
    assert lb_models == {"catboost", "extra_trees", "mlp"}
    for entry in lb:
        assert entry["score"] is not None
