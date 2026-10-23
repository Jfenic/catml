"""Univariate and multivariate distribution diagnostics for CATML Explore."""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats as stats

from automl.domain.analysis.models import AnalysisHypothesis, StatisticalFinding

logger = logging.getLogger(__name__)


def diagnose_univariate_distributions(
    df: pd.DataFrame,
    num_cols: list[str],
    study_id: str,
    run_id: str,
) -> tuple[list[StatisticalFinding], list[AnalysisHypothesis]]:
    """Compute normality tests, skewness, kurtosis, multimodality, and IQR outliers."""
    findings: list[StatisticalFinding] = []
    hypotheses: list[AnalysisHypothesis] = []

    for col in num_cols:
        series = df[col].dropna()
        if pd.api.types.is_bool_dtype(series):
            series = series.astype(float)
        n = len(series)
        if n < 5:
            continue

        # 1. Moments: Skewness & Kurtosis
        try:
            skew_val = float(stats.skew(series, nan_policy="omit"))
            kurt_val = float(stats.kurtosis(series, nan_policy="omit"))  # excess kurtosis

            if abs(skew_val) >= 1.2:
                direction = "positiva (cola derecha)" if skew_val > 0 else "negativa (cola izquierda)"
                f = StatisticalFinding.create(
                    study_id=study_id,
                    analysis_run_id=run_id,
                    finding_type="skewness",
                    summary=f"Asimetría severa {direction} detectada en '{col}' (skewness={skew_val:.2f}, curtosis={kurt_val:.2f}).",
                    column_name=col,
                    method_name="scipy.stats.skew",
                    metrics={"skewness": round(skew_val, 3), "kurtosis": round(kurt_val, 3)},
                    effect_size=abs(round(skew_val, 3)),
                )
                findings.append(f)

                transform_type = "yeo_johnson" if (series <= 0).any() else "log1p"
                hyp = AnalysisHypothesis.create(
                    study_id=study_id,
                    finding_id=f.id,
                    description=f"Aplicar transformación {transform_type.title()} en '{col}' para estabilizar varianza y normalizar colas.",
                    proposed_action="power_transform",
                    experiment_delta={"transform": transform_type, "column": col},
                )
                hypotheses.append(hyp)

            # 2. Multimodality (Bimodal Coefficient BC)
            # BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)*(n-3)))
            if n >= 30 and kurt_val > -3.0:
                denom_adj = 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
                denom = kurt_val + denom_adj
                if denom > 0:
                    bc = (skew_val ** 2 + 1.0) / denom
                    # Standard Sarle threshold for bimodal/multimodal distributions is 5/9 ≈ 0.555
                    if bc > 0.555:
                        f_multi = StatisticalFinding.create(
                            study_id=study_id,
                            analysis_run_id=run_id,
                            finding_type="multimodal_distribution",
                            summary=f"Distribución potencialmente multimodal o bimodal en '{col}' (Coeficiente Bimodal={bc:.3f} > 0.555).",
                            column_name=col,
                            method_name="sarle_bimodal_coefficient",
                            metrics={"bimodal_coefficient": round(bc, 3), "sample_size": n},
                            limitations=["El coeficiente bimodal es heurístico; se recomienda validar con estimación de densidad KDE."],
                        )
                        findings.append(f_multi)
                        hyp_multi = AnalysisHypothesis.create(
                            study_id=study_id,
                            finding_id=f_multi.id,
                            description=f"Evaluar discretización por cuantiles o agrupamiento de subpoblaciones para '{col}'.",
                            proposed_action="discretize_or_cluster",
                            experiment_delta={"action": "quantile_discretization", "column": col},
                        )
                        hypotheses.append(hyp_multi)

            # 3. Normality Inferences (D'Agostino or Shapiro-Wilk)
            norm_stat: float | None = None
            norm_p: float | None = None
            test_method = ""

            if n >= 20:
                test_method = "scipy.stats.normaltest"
                s, p = stats.normaltest(series)
                norm_stat, norm_p = float(s), float(p)
            elif n >= 8:
                test_method = "scipy.stats.shapiro"
                s, p = stats.shapiro(series)
                norm_stat, norm_p = float(s), float(p)

            if norm_p is not None and norm_p < 0.01:
                f_norm = StatisticalFinding.create(
                    study_id=study_id,
                    analysis_run_id=run_id,
                    finding_type="normality_violation",
                    summary=f"Rechazo de normalidad para '{col}' ({test_method.split('.')[-1]}, stat={norm_stat:.3f}, p={norm_p:.4e}).",
                    column_name=col,
                    method_name=test_method,
                    metrics={"test_statistic": round(norm_stat, 4) if norm_stat is not None else 0.0, "sample_size": n},
                    p_value=norm_p,
                    limitations=["En muestras grandes, pruebas de normalidad rechazan H0 incluso ante desviaciones menores."],
                )
                findings.append(f_norm)

        except Exception as exc:
            logger.debug("Distribution error for column %s: %s", col, exc)

        # 4. Outliers using IQR
        try:
            q25 = float(np.percentile(series, 25))
            q75 = float(np.percentile(series, 75))
            iqr = q75 - q25
            if iqr > 1e-9:
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                outliers_mask = (series < lower) | (series > upper)
                outlier_count = int(outliers_mask.sum())
                outlier_rate = float(outlier_count / n)
                if outlier_rate >= 0.03:
                    f_out = StatisticalFinding.create(
                        study_id=study_id,
                        analysis_run_id=run_id,
                        finding_type="outlier_density",
                        summary=f"Densidad de outliers en '{col}': {outlier_count} observaciones fuera de IQR ({outlier_rate * 100:.1f}%).",
                        column_name=col,
                        method_name="interquartile_range_1.5",
                        metrics={
                            "outlier_count": outlier_count,
                            "outlier_rate": round(outlier_rate, 4),
                            "q25": round(q25, 3),
                            "q75": round(q75, 3),
                            "iqr": round(iqr, 3),
                        },
                        effect_size=round(outlier_rate, 3),
                    )
                    findings.append(f_out)
                    hyp_out = AnalysisHypothesis.create(
                        study_id=study_id,
                        finding_id=f_out.id,
                        description=f"Utilizar escalado robusto (RobustScaler) o recorte de cuantiles (winsorizing) en '{col}'.",
                        proposed_action="robust_scaler",
                        experiment_delta={"scaler": "robust", "column": col},
                    )
                    hypotheses.append(hyp_out)
        except Exception as exc:
            logger.debug("Outlier calculation error for column %s: %s", col, exc)

    return findings, hypotheses


def diagnose_multivariate_outliers(
    df: pd.DataFrame,
    num_cols: list[str],
    study_id: str,
    run_id: str,
) -> tuple[list[StatisticalFinding], list[AnalysisHypothesis]]:
    """Detect multivariate anomalies using regularized Mahalanobis distance or Isolation Forest."""
    findings: list[StatisticalFinding] = []
    hypotheses: list[AnalysisHypothesis] = []

    k = len(num_cols)
    if k < 2:
        return findings, hypotheses

    # Limit to at most 10 numeric columns with highest variance to prevent singularity
    selected_cols = num_cols
    if k > 10:
        variances = df[num_cols].var().fillna(0)
        selected_cols = list(variances.nlargest(10).index)
        k = len(selected_cols)

    data = df[selected_cols].dropna()
    n = len(data)
    if n < max(20, k + 5):
        return findings, hypotheses

    try:
        X = data.values.astype(float)
        mean_vec = np.mean(X, axis=0)
        X_centered = X - mean_vec

        cov_matrix = np.cov(X_centered, rowvar=False)
        # Regularize covariance to avoid near-singularity
        diag_reg = 1e-4 * np.eye(k)
        inv_cov = np.linalg.pinv(cov_matrix + diag_reg)

        # Mahalanobis D^2 = sum_j (X_c * inv_cov)_ij * X_c_ij
        left = np.dot(X_centered, inv_cov)
        d2 = np.sum(left * X_centered, axis=1)

        # Chi-square critical value at alpha = 0.001
        critical_val = float(stats.chi2.ppf(0.999, df=k))
        outlier_indices = np.where(d2 > critical_val)[0]
        outlier_count = int(len(outlier_indices))
        outlier_ratio = float(outlier_count / n)

        if outlier_ratio >= 0.01:
            f = StatisticalFinding.create(
                study_id=study_id,
                analysis_run_id=run_id,
                finding_type="multivariate_outliers",
                summary=(
                    f"Detección multivariante de anomalías (Mahalanobis D²): {outlier_count} registros atípicos "
                    f"conjuntos ({outlier_ratio * 100:.1f}%) superan umbral crítico χ²(df={k}, p<0.001)={critical_val:.1f}."
                ),
                method_name="mahalanobis_distance_chi2",
                metrics={
                    "outlier_count": outlier_count,
                    "outlier_ratio": round(outlier_ratio, 4),
                    "dimension_k": k,
                    "critical_threshold": round(critical_val, 2),
                    "columns_evaluated": selected_cols,
                },
                effect_size=round(outlier_ratio, 3),
            )
            findings.append(f)
            hyp = AnalysisHypothesis.create(
                study_id=study_id,
                finding_id=f.id,
                description="Aplicar filtro multivariante o regularización robusta para mitigar distorsión en estimadores lineales.",
                proposed_action="multivariate_filtering",
                experiment_delta={"filter": "mahalanobis", "max_contamination": 0.05},
            )
            hypotheses.append(hyp)

    except Exception as exc:
        logger.debug("Multivariate outlier computation failed: %s", exc)

    return findings, hypotheses
