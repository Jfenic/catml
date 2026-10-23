from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd


def _compute_file_sha256(path: Path, chunk_size: int = 65536) -> str:
    """
    Compute the SHA-256 digest of a file in streaming chunks.

    Notice on Integrity vs. Authenticity:
        This checksum guarantees file integrity (detection of accidental disk
        corruption or truncated transfers). It does NOT guarantee cryptographic
        authenticity or non-repudiation against an active adversary modifying both
        the artifact and the sidecar checksum file without digital signatures / PKI.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_environment_dependencies() -> dict[str, str]:
    """Collect installed package versions for ML framework reproducibility."""
    from importlib.metadata import PackageNotFoundError, version

    dep_versions: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "lightgbm",
        "xgboost",
        "catboost",
        "optuna",
        "torch",
        "torchvision",
        "timm",
        "pillow",
    ):
        try:
            dep_versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return dep_versions


@dataclass
class ModelArtifact:
    """
    Self-contained, serializable machine learning model artifact.

    Encapsulates the fully fitted scikit-learn Pipeline (preprocessing, encoders,
    scalers, imputers, and estimator), feature metadata, target adapter, and metrics.
    Can be serialized/deserialized independently without database or workspace dependencies.
    """

    pipeline: Any
    model_id: str
    task_type: str
    feature_names: list[str]
    target_name: str | None = None
    target_adapter: Any = None
    metric: str = "roc_auc"
    score: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provenance and "provenance" in self.metadata:
            self.provenance = dict(self.metadata["provenance"])
        elif not self.provenance:
            self.provenance = self._collect_default_provenance()

    def _collect_default_provenance(self) -> dict[str, Any]:
        import sys
        from datetime import datetime, timezone

        dep_versions = collect_environment_dependencies()
        estimator_step = self.pipeline[-1] if hasattr(self.pipeline, "__getitem__") and hasattr(self.pipeline, "steps") else self.pipeline

        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "estimator_class": type(estimator_step).__name__,
            "estimator_module": type(estimator_step).__module__,
            "dependencies": dep_versions,
        }

    def describe(self) -> dict[str, Any]:
        """
        Return a structured summary of the artifact's metadata and provenance.
        """
        return {
            "model_id": self.model_id,
            "task_type": self.task_type,
            "target_name": self.target_name,
            "metric": self.metric,
            "score": self.score,
            "feature_count": len(self.feature_names),
            "features": list(self.feature_names),
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Run inference using the self-contained pipeline.

        Accepts either a pandas DataFrame (with required column names) or a numpy
        ndarray (with matching number of feature columns). Automatically inverts
        target label encoding if a TargetAdapter was fitted.
        """
        X_proc = self._prepare_features(X)
        raw_preds = self.pipeline.predict(X_proc)

        if self.target_adapter is not None and getattr(self.target_adapter, "is_encoded", False):
            try:
                preds = self.target_adapter.inverse_transform(raw_preds)
                return np.asarray(preds)
            except Exception:
                pass
        return np.asarray(raw_preds)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute predicted class probabilities if supported by the underlying model.
        """
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError(
                f"Underlying model '{self.model_id}' does not support predict_proba."
            )
        X_proc = self._prepare_features(X)
        probs = self.pipeline.predict_proba(X_proc)
        return np.asarray(probs)

    def _apply_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Materialize recipe-derived columns from original inputs when missing."""
        derived_specs = list(self.metadata.get("derived_features") or [])
        if not derived_specs:
            return df

        from automl.domain.features.derived_feature import DerivedFeatureDefinition
        from automl.engine.features.generation.derived_feature_engine import (
            DerivedFeatureEngine,
            assert_derived_specs_safe_for_cross_validation,
        )

        assert_derived_specs_safe_for_cross_validation(derived_specs)
        engine = DerivedFeatureEngine()
        out = df
        for spec in derived_specs:
            defn = DerivedFeatureDefinition.from_dict(spec) if isinstance(spec, dict) else spec
            if defn.name in out.columns:
                continue
            out, eval_res = engine.apply(
                out,
                defn,
                allow_python_code=False,
                allow_stateful_aggregates=False,
            )
            if not eval_res.is_valid:
                raise ValueError(
                    f"Failed to materialize derived feature '{defn.name}' during inference: "
                    f"{eval_res.error_message}"
                )
        return out

    def _prepare_features(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            prepared = self._apply_derived_features(X)
            missing = [f for f in self.feature_names if f not in prepared.columns]
            if missing:
                raise ValueError(f"Input DataFrame is missing required features: {missing}")
            return prepared[self.feature_names]
        elif isinstance(X, np.ndarray):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Feature count mismatch: expected {len(self.feature_names)} features, got {arr.shape[1]}."
                )
            return pd.DataFrame(arr, columns=self.feature_names)
        else:
            try:
                df = pd.DataFrame(X)
                if df.shape[1] == len(self.feature_names):
                    df.columns = self.feature_names
                    return df
            except Exception:
                pass
            raise TypeError(f"Unsupported input type for inference: {type(X)}")

    def save(self, path: str | Path, generate_checksum: bool = True) -> Path:
        """
        Serialize this artifact to disk as a standalone portable file.

        Args:
            path: Destination file path (e.g. 'model.pkl').
            generate_checksum: If True, writes a sidecar '.sha256' file with the artifact hash.
                Note: The sidecar checksum guarantees integrity against accidental corruption,
                not authenticity against untrusted adversaries.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        if generate_checksum:
            digest = _compute_file_sha256(target)
            sha_file = target.with_suffix(target.suffix + ".sha256")
            sha_file.write_text(digest, encoding="utf-8")
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        verify_checksum: bool = True,
        *,
        allow_missing_checksum: bool = False,
    ) -> ModelArtifact:
        """
        Load a standalone ModelArtifact from disk.

        Security Notice:
            Deserializing pickle/joblib files can execute arbitrary code. Only load
            CATML model artifacts from trusted sources and verified environments.
            While SHA-256 sidecar verification protects against accidental file corruption
            (integrity), it does NOT provide cryptographic signature verification (authenticity).

        Args:
            path: Path to the serialized artifact.
            verify_checksum: If True, require and verify the sidecar '.sha256' digest.
            allow_missing_checksum: Legacy escape hatch when ``verify_checksum`` is True but
                the sidecar is absent. Default False (fail closed).
        """
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Model artifact not found at '{target}'")

        if verify_checksum:
            sha_file = target.with_suffix(target.suffix + ".sha256")
            if not sha_file.is_file():
                if not allow_missing_checksum:
                    raise ValueError(
                        f"Artifact checksum sidecar missing for '{target}'. "
                        "Refusing to load without integrity verification. "
                        "Pass allow_missing_checksum=True only for trusted legacy artifacts."
                    )
            else:
                try:
                    expected_sha = sha_file.read_text(encoding="utf-8").strip()
                    actual_sha = _compute_file_sha256(target)
                except Exception as exc:
                    raise ValueError(
                        f"Artifact checksum verification failed for '{target}': "
                        f"unable to read or compute hash integrity ({exc})."
                    ) from exc

                if actual_sha != expected_sha:
                    raise ValueError(
                        f"Artifact checksum verification failed for '{target}'. "
                        f"Expected SHA-256 '{expected_sha}', got '{actual_sha}'. "
                        "The artifact may have been modified or corrupted."
                    )

        warnings.warn(
            "Security Notice: Only load CATML artifacts from trusted sources. "
            "Deserializing pickle/joblib files from untrusted environments can execute arbitrary code.",
            UserWarning,
            stacklevel=2,
        )

        loaded = joblib.load(target)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Loaded object is not a ModelArtifact instance (got {type(loaded)})"
            )
        return loaded
