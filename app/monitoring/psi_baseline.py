from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BASELINE_SCHEMA_VERSION = 1
DEFAULT_EPSILON = 1e-6


def _finite_numeric(values: pd.Series, feature_name: str) -> np.ndarray:
    try:
        array = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature {feature_name!r} must contain numerical values") from exc
    if array.size == 0:
        raise ValueError(f"Feature {feature_name!r} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Feature {feature_name!r} must contain only finite values")
    return array


def _count_categories(values: list[str], categories: list[str]) -> list[int]:
    counts = {category: 0 for category in categories}
    for value in values:
        if value not in counts:
            raise ValueError(f"Categorical value {value!r} is not in the fixed class set")
        counts[value] += 1
    return [counts[category] for category in categories]


def build_psi_baseline(
    reference_features: pd.DataFrame,
    bins: int = 10,
    sparsity_threshold: float = 0.80,
    *,
    model_version: int | str | None = None,
    dataset_version: str | None = None,
    categorical_classes: dict[str, list[Any]] | None = None,
    selected_vocabulary_terms: list[dict[str, Any]] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    if not isinstance(reference_features, pd.DataFrame) or reference_features.empty:
        raise ValueError("reference_features must be a non-empty pandas DataFrame")
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if not 0 <= sparsity_threshold <= 1:
        raise ValueError("sparsity_threshold must be between 0 and 1")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive number")
    if reference_features.columns.has_duplicates:
        raise ValueError("reference_features cannot contain duplicate feature names")

    categorical_classes = categorical_classes or {}
    features: dict[str, dict[str, Any]] = {}
    for feature_name in reference_features.columns:
        name = str(feature_name)
        series = reference_features[feature_name]
        if name in categorical_classes or name == "predicted_sentiment":
            if series.isna().any():
                raise ValueError(f"Categorical feature {name!r} cannot contain missing values")
            values = [str(value) for value in series.tolist()]
            categories = [str(value) for value in categorical_classes.get(name, [])]
            if not categories:
                categories = sorted(set(values))
            if not categories:
                raise ValueError(f"Categorical feature {name!r} has no classes")
            features[name] = {
                "distribution_type": "categorical",
                "categories": categories,
                "reference_counts": _count_categories(values, categories),
            }
            continue

        values = _finite_numeric(series, name)
        zero_ratio = float(np.count_nonzero(values == 0) / values.size)
        unique_values = np.unique(values)
        if unique_values.size == 1 and not name.startswith("tfidf_term:"):
            constant = float(unique_values[0])
            features[name] = {
                "distribution_type": "constant",
                "constant_value": constant,
                "categories": ["below", "equal", "above"],
                "reference_counts": [0, int(values.size), 0],
                "reference_samples": values.tolist(),
            }
            continue

        if name.startswith("tfidf_term:") or zero_ratio >= sparsity_threshold:
            categories = ["zero", "nonzero"]
            counts = [int(np.count_nonzero(values == 0)), int(np.count_nonzero(values != 0))]
            features[name] = {
                "distribution_type": "sparse_binary",
                "categories": categories,
                "reference_counts": counts,
                "reference_samples": values.tolist(),
            }
            continue

        quantiles = np.quantile(values, np.linspace(0, 1, bins + 1))
        interior = np.unique(quantiles[1:-1])
        boundaries = [-np.inf, *[float(value) for value in interior], np.inf]
        counts, _ = np.histogram(values, bins=np.asarray(boundaries, dtype=float))
        features[name] = {
            "distribution_type": "continuous",
            "boundaries": boundaries,
            "reference_counts": counts.astype(int).tolist(),
            "reference_samples": values.tolist(),
        }

    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "created_at": datetime.now(UTC).isoformat(),
        "reference_sample_count": int(len(reference_features)),
        "feature_names": list(features),
        "features": features,
        "bins": int(bins),
        "sparsity_threshold": float(sparsity_threshold),
        "epsilon": float(epsilon),
        "selected_vocabulary_terms": selected_vocabulary_terms or [],
        "compatibility": {
            "model_version": model_version,
            "dataset_version": dataset_version,
            "feature_names": list(features),
        },
    }


def validate_psi_baseline(baseline: dict[str, Any]) -> None:
    required = {
        "schema_version", "model_version", "dataset_version", "created_at",
        "reference_sample_count", "feature_names", "features", "bins",
        "sparsity_threshold", "epsilon", "compatibility",
    }
    if not isinstance(baseline, dict) or not required.issubset(baseline):
        raise ValueError("PSI baseline is missing required schema fields")
    if baseline["schema_version"] != BASELINE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported PSI baseline schema version: {baseline['schema_version']}")
    if (
        not isinstance(baseline["features"], dict)
        or not baseline["features"]
        or not isinstance(baseline["feature_names"], list)
    ):
        raise ValueError("PSI baseline must define at least one feature")
    if baseline["feature_names"] != list(baseline["features"]):
        raise ValueError("PSI baseline feature_names do not match feature configurations")
    if baseline["reference_sample_count"] <= 0 or baseline["bins"] < 1:
        raise ValueError("PSI baseline sample count and bin count must be positive")
    if (
        not np.isfinite(baseline["epsilon"])
        or baseline["epsilon"] <= 0
        or not np.isfinite(baseline["sparsity_threshold"])
        or not 0 <= baseline["sparsity_threshold"] <= 1
    ):
        raise ValueError("PSI baseline numeric configuration is invalid")
    if baseline["compatibility"].get("model_version") != baseline["model_version"]:
        raise ValueError("PSI baseline model compatibility metadata is inconsistent")
    if baseline["compatibility"].get("dataset_version") != baseline["dataset_version"]:
        raise ValueError("PSI baseline dataset compatibility metadata is inconsistent")
    if baseline["compatibility"].get("feature_names") != baseline["feature_names"]:
        raise ValueError("PSI baseline feature compatibility metadata is inconsistent")
    for feature_name, config in baseline["features"].items():
        if not isinstance(config, dict) or "distribution_type" not in config:
            raise ValueError(f"Invalid PSI baseline configuration for {feature_name!r}")
        distribution_type = config["distribution_type"]
        if distribution_type not in {"categorical", "sparse_binary", "constant", "continuous"}:
            raise ValueError(f"Unsupported PSI distribution for {feature_name!r}")
        reference_counts = np.asarray(config.get("reference_counts", []), dtype=float)
        if (
            reference_counts.size == 0
            or np.any(reference_counts < 0)
            or not np.all(np.isfinite(reference_counts))
        ):
            raise ValueError(f"Invalid reference counts for PSI feature {feature_name!r}")
        if distribution_type in {"categorical", "sparse_binary", "constant"}:
            expected_count = len(config.get("categories", []))
        else:
            boundaries = np.asarray(config.get("boundaries", []), dtype=float)
            if boundaries.size < 2 or np.any(np.isnan(boundaries)) or np.any(np.diff(boundaries) <= 0):
                raise ValueError(f"Invalid histogram boundaries for PSI feature {feature_name!r}")
            expected_count = boundaries.size - 1
        if reference_counts.size != expected_count:
            raise ValueError(f"Invalid reference counts for PSI feature {feature_name!r}")
        if distribution_type == "constant" and not np.isfinite(config.get("constant_value", np.nan)):
            raise ValueError(f"Invalid constant reference value for PSI feature {feature_name!r}")
        if "reference_samples" in config:
            samples = np.asarray(config["reference_samples"], dtype=float)
            if samples.size == 0 or not np.all(np.isfinite(samples)):
                raise ValueError(f"Invalid reference samples for PSI feature {feature_name!r}")


def save_psi_baseline(baseline: dict[str, Any], path: Path) -> None:
    validate_psi_baseline(baseline)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing PSI baseline: {destination}")
    fd, temporary_path = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        joblib.dump(baseline, temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        Path(temporary_path).unlink(missing_ok=True)
        raise


def load_psi_baseline(path: Path) -> dict[str, Any]:
    baseline = joblib.load(Path(path))
    validate_psi_baseline(baseline)
    return baseline
