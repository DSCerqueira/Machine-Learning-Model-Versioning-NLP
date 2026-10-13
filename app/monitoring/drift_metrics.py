from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


def _distribution_counts(values: np.ndarray, baseline_config: dict[str, Any]) -> np.ndarray:
    data = np.asarray(values)
    if data.size == 0:
        raise ValueError("Current feature values cannot be empty")
    kind = baseline_config["distribution_type"]

    if kind == "categorical":
        categories = baseline_config["categories"]
        indexed = {str(category): index for index, category in enumerate(categories)}
        counts = np.zeros(len(categories), dtype=float)
        for value in data.tolist():
            category = str(value)
            if category not in indexed:
                raise ValueError(f"Current category {category!r} is not in the baseline class set")
            counts[indexed[category]] += 1
        return counts

    try:
        numeric = data.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Current feature values must be numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Current feature values must be finite")

    if kind == "sparse_binary":
        return np.asarray([np.count_nonzero(numeric == 0), np.count_nonzero(numeric != 0)], dtype=float)
    if kind == "constant":
        constant = float(baseline_config["constant_value"])
        return np.asarray([
            np.count_nonzero(numeric < constant),
            np.count_nonzero(numeric == constant),
            np.count_nonzero(numeric > constant),
        ], dtype=float)
    if kind == "continuous":
        boundaries = np.asarray(baseline_config["boundaries"], dtype=float)
        if np.any(np.isnan(boundaries)) or np.any(np.diff(boundaries) <= 0):
            raise ValueError("Baseline histogram boundaries are invalid")
        counts, _ = np.histogram(numeric, bins=boundaries)
        return counts.astype(float)
    raise ValueError(f"Unsupported PSI distribution type: {kind!r}")


def _smoothed_probabilities(counts: np.ndarray, epsilon: float) -> np.ndarray:
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive number")
    smoothed = np.asarray(counts, dtype=float) + epsilon
    total = smoothed.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Distribution counts must have a positive finite total")
    return smoothed / total


def _validated_current_counts(current_values: np.ndarray, baseline_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    current = _distribution_counts(current_values, baseline_config)
    reference = np.asarray(baseline_config["reference_counts"], dtype=float)
    if reference.shape != current.shape or np.any(reference < 0) or not np.all(np.isfinite(reference)):
        raise ValueError("Baseline reference counts are invalid")
    return reference, current


def calculate_feature_psi(
    current_values: np.ndarray,
    baseline_config: dict[str, Any],
    epsilon: float = 1e-6,
) -> float:
    reference_counts, current_counts = _validated_current_counts(current_values, baseline_config)
    expected = _smoothed_probabilities(reference_counts, epsilon)
    actual = _smoothed_probabilities(current_counts, epsilon)
    result = float(np.sum((actual - expected) * np.log(actual / expected)))
    if not np.isfinite(result):
        raise ValueError("PSI calculation produced a non-finite result")
    return max(0.0, result)


def calculate_feature_js(
    current_values: np.ndarray,
    baseline_config: dict[str, Any],
    epsilon: float = 1e-6,
) -> float:
    reference_counts, current_counts = _validated_current_counts(current_values, baseline_config)
    expected = _smoothed_probabilities(reference_counts, epsilon)
    actual = _smoothed_probabilities(current_counts, epsilon)
    result = float(jensenshannon(expected, actual, base=2) ** 2)
    if not np.isfinite(result):
        raise ValueError("Jensen-Shannon calculation produced a non-finite result")
    return result


def calculate_feature_ks(
    current_values: np.ndarray,
    baseline_config: dict[str, Any],
) -> tuple[float | None, float | None]:
    if baseline_config["distribution_type"] == "categorical":
        return None, None
    reference = np.asarray(baseline_config.get("reference_samples", []), dtype=float)
    current = np.asarray(current_values, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("KS comparison requires non-empty reference and current samples")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(current)):
        raise ValueError("KS samples must contain only finite values")
    statistic, pvalue = ks_2samp(reference, current)
    return float(statistic), float(pvalue)
