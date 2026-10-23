from __future__ import annotations

from typing import Any
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from automl.domain.optimization.search_space import SearchSpace
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import ModelPluginPort


def build_sklearn_model(model_id: str, task_type: str, parameters: dict | None = None):
    params = parameters or {}
    is_regression = task_type == "regression"
    is_clustering = task_type == "clustering"

    if is_clustering:
        if model_id == "kmeans":
            return KMeans(
                n_clusters=int(params.get("n_clusters", 3)),
                random_state=int(params.get("random_state", 42)),
                n_init=int(params.get("n_init", 10)),
            )
        elif model_id == "agglomerative":
            return AgglomerativeClustering(
                n_clusters=int(params.get("n_clusters", 3)),
            )
        elif model_id == "dbscan":
            return DBSCAN(
                eps=float(params.get("eps", 0.5)),
                min_samples=int(params.get("min_samples", 5)),
            )
        else:
            raise ValueError(f"Unknown clustering model: {model_id}")

    if model_id == "logistic_regression":
        lr_params = {"max_iter": 1000, "random_state": 42}
        lr_params.update(params)
        return LogisticRegression(**lr_params)
    elif model_id == "random_forest":
        rf_params = {"n_estimators": 100, "random_state": 42}
        rf_params.update(params)
        return (
            RandomForestRegressor(**rf_params)
            if is_regression
            else RandomForestClassifier(**rf_params)
        )
    elif model_id == "extra_trees":
        et_params = {"n_estimators": 100, "random_state": 42}
        et_params.update(params)
        return (
            ExtraTreesRegressor(**et_params)
            if is_regression
            else ExtraTreesClassifier(**et_params)
        )
    elif model_id == "mlp":
        mlp_params = {"max_iter": 500, "random_state": 42, "early_stopping": True}
        mlp_params.update(params)
        return (
            MLPRegressor(**mlp_params)
            if is_regression
            else MLPClassifier(**mlp_params)
        )
    elif model_id == "catboost":
        from automl.plugins.models.catboost_plugin import CatBoostPlugin

        return CatBoostPlugin().build_estimator(parameters=params, task_type=task_type)
    elif model_id in {"svc", "svr"}:
        if is_regression or model_id == "svr":
            svr_params = {}
            svr_params.update(params)
            return SVR(**svr_params)
        else:
            svc_params = {"random_state": 42}
            svc_params.update(params)
            return CalibratedClassifierCV(SVC(**svc_params), method="sigmoid", cv=3)
    elif model_id == "ridge":
        ridge_params = {"alpha": 1.0}
        ridge_params.update(params)
        return Ridge(**ridge_params)
    elif model_id == "voting_ensemble":
        from automl.engine.ensemble.voting import VotingEnsembleEstimator
        factory = lambda tt: default_ensemble_factory(tt)
        return VotingEnsembleEstimator(
            task_type=task_type,
            default_estimator_factory=factory,
            **params,
        )
    else:
        raise ValueError(f"Unknown supervised model: {model_id}")


class ExtraTreesPlugin(ModelPluginPort):
    def __init__(self) -> None:
        self.plugin_id = "extra_trees"
        self.name = "Extra Trees (Extremely Randomized Trees)"
        self.version = "1.0.0"
        self.plugin_type = PluginType.MODEL
        self.capabilities = PluginCapability(
            supported_tasks=["binary_classification", "multiclass_classification", "regression"],
            supported_modalities=["tabular"],
            requires_gpu=False,
            supports_proba=True,
        )

    def build_estimator(
        self,
        parameters: dict[str, Any] | None = None,
        task_type: str = "binary_classification",
        **kwargs: Any,
    ) -> Any:
        params = dict(parameters or {})
        params.update(kwargs)
        return build_sklearn_model("extra_trees", task_type=task_type, parameters=params)

    def get_search_space(self, task_type: str) -> SearchSpace:
        from automl.engine.optimization.search_space_builder import SearchSpaceBuilder

        return SearchSpaceBuilder.build("extra_trees", task_type=task_type)


class MLPPlugin(ModelPluginPort):
    def __init__(self) -> None:
        self.plugin_id = "mlp"
        self.name = "Multi-Layer Perceptron (MLP)"
        self.version = "1.0.0"
        self.plugin_type = PluginType.MODEL
        self.capabilities = PluginCapability(
            supported_tasks=["binary_classification", "multiclass_classification", "regression"],
            supported_modalities=["tabular"],
            requires_gpu=False,
            supports_proba=True,
        )

    def build_estimator(
        self,
        parameters: dict[str, Any] | None = None,
        task_type: str = "binary_classification",
        **kwargs: Any,
    ) -> Any:
        params = dict(parameters or {})
        params.update(kwargs)
        return build_sklearn_model("mlp", task_type=task_type, parameters=params)

    def get_search_space(self, task_type: str) -> SearchSpace:
        from automl.engine.optimization.search_space_builder import SearchSpaceBuilder

        return SearchSpaceBuilder.build("mlp", task_type=task_type)


def default_ensemble_factory(task_type: str) -> list[tuple[str, Any]]:
    """Central factory for default base models of an ensemble."""
    is_regression = task_type == "regression"
    if is_regression:
        return [
            ("ridge", build_sklearn_model("ridge", task_type)),
            ("random_forest", build_sklearn_model("random_forest", task_type)),
        ]
    return [
        ("logistic_regression", build_sklearn_model("logistic_regression", task_type)),
        ("random_forest", build_sklearn_model("random_forest", task_type)),
    ]


def default_model_specs():
    from automl.domain.models.registry import ModelSpec
    from automl.domain.tasks.task_type import TaskType, models_for_task

    specs: list[ModelSpec] = []
    seen: dict[str, set[str]] = {}

    for task in TaskType:
        for model_id in models_for_task(task):
            seen.setdefault(model_id, set()).add(task.value)

    labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "mlp": "Multi-Layer Perceptron (MLP)",
        "svc": "Support Vector Classifier",
        "ridge": "Ridge Regression",
        "svr": "Support Vector Regressor",
        "kmeans": "K-Means",
        "agglomerative": "Agglomerative Clustering",
        "dbscan": "DBSCAN",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "voting_ensemble": "Voting Ensemble & Blending",
    }

    for model_id, task_types in sorted(seen.items()):
        specs.append(
            ModelSpec(
                id=model_id,
                name=labels.get(model_id, model_id),
                task_types=sorted(task_types),
                description=f"Compatible with: {', '.join(sorted(task_types))}",
            )
        )
    return specs
