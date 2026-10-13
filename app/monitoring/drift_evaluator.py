from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.monitoring.drift_metrics import (
    calculate_feature_js,
    calculate_feature_ks,
    calculate_feature_psi,
)
from app.monitoring.psi_baseline import validate_psi_baseline


def evaluate_drift(
    current_features: pd.DataFrame,
    baseline: dict[str, Any],
    *,
    model_version: int | str | None = None,
    threshold: float = 0.20,
) -> dict[str, Any]:
    validate_psi_baseline(baseline)
    if current_features.empty:
        raise ValueError("current_features must not be empty")
    expected_model_version = baseline.get("model_version")
    if model_version is not None and str(model_version) != str(expected_model_version):
        raise ValueError(
            f"Model/baseline version mismatch: model {model_version}, "
            f"baseline {expected_model_version}"
        )
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative number")

    missing = [name for name in baseline["feature_names"] if name not in current_features]
    if missing:
        raise ValueError(f"Current features are missing baseline features: {missing}")
    results: list[dict[str, Any]] = []
    for feature_name in baseline["feature_names"]:
        config = baseline["features"][feature_name]
        current_values = current_features[feature_name].to_numpy()
        psi = calculate_feature_psi(current_values, config, baseline["epsilon"])
        ks, ks_pvalue = calculate_feature_ks(current_values, config)
        js = calculate_feature_js(current_values, config, baseline["epsilon"])
        is_vocabulary = feature_name.startswith("tfidf_term:")
        results.append({
            "feature": feature_name,
            "distribution_type": config["distribution_type"],
            "scope": "vocabulary" if is_vocabulary else "primary",
            "psi": psi,
            "ks": ks,
            "ks_pvalue": ks_pvalue,
            "js": js,
            "breach": bool(psi > threshold),
        })

    primary_results = [row for row in results if row["scope"] == "primary"]
    vocabulary_results = [row for row in results if row["scope"] == "vocabulary"]
    if not primary_results:
        raise ValueError("Baseline has no primary NLP monitoring features")
    primary_results.sort(key=lambda row: row["psi"], reverse=True)
    breached = sum(bool(row["psi"] > threshold) for row in primary_results)
    return {
        "feature_results": results,
        "primary_results": primary_results,
        "vocabulary_results": vocabulary_results,
        "most_drifted_feature": primary_results[0]["feature"],
        "psi": primary_results[0]["psi"],
        "ks": primary_results[0]["ks"],
        "ks_pvalue": primary_results[0]["ks_pvalue"],
        "js": primary_results[0]["js"],
        "maximum_primary_psi": primary_results[0]["psi"],
        "monitored_features": len(primary_results),
        "breached_features": breached,
        "breach_ratio": breached / len(primary_results),
        "breach": breached > 0,
    }
