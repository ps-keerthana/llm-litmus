"""
Phase 7: Drift Detection Module.

Implements:
    - Population Stability Index (PSI) between baseline and current distributions
    - Wasserstein Distance (Earth Mover's Distance) for 1D continuous scores
"""

import math
from typing import Any, Dict, List, Sequence, Tuple


def compute_psi(
    expected: Sequence[float],
    actual: Sequence[float],
    num_bins: int = 10,
    eps: float = 1e-4,
) -> Dict[str, Any]:
    """
    Computes Population Stability Index (PSI) between expected (baseline) and actual (current)
    distributions of continuous scores [0.0, 1.0].

    Formula:
        PSI = sum_{i=1}^B (Actual_i - Expected_i) * ln(Actual_i / Expected_i)

    Rule of thumb:
        - PSI < 0.10: No significant distribution change (STABLE)
        - 0.10 <= PSI <= 0.25: Moderate shift, requires monitoring (MODERATE_SHIFT)
        - PSI > 0.25: Significant distribution shift, alert/fail gate (SIGNIFICANT_DRIFT)
    """
    if len(expected) == 0 or len(actual) == 0:
        return {
            "psi": 0.0,
            "status": "INSUFFICIENT_DATA",
            "is_drifted": False,
            "details": "At least one distribution is empty",
        }

    # Quantile/adaptive binning with Laplace smoothing
    # Using 5 bins (quintiles) standard for model drift monitoring
    num_bins = min(num_bins, 5)
    sorted_exp = sorted(expected)
    n_exp = len(sorted_exp)
    n_act = len(actual)

    # Attempt quintile split from expected distribution
    raw_edges = [sorted_exp[int((i / num_bins) * (n_exp - 1))] for i in range(num_bins + 1)]
    is_increasing = all(raw_edges[i] < raw_edges[i + 1] for i in range(len(raw_edges) - 1))

    if is_increasing:
        bin_edges = [0.0] + raw_edges[1:-1] + [1.0001]
    else:
        # Uniform bins if quantiles collapse
        bin_edges = [i * (1.0 / num_bins) for i in range(num_bins)] + [1.0001]

    k_bins = len(bin_edges) - 1
    exp_counts = [0] * k_bins
    act_counts = [0] * k_bins

    def place_in_bin(val: float) -> int:
        v = max(0.0, min(1.0, val))
        for b in range(k_bins):
            if bin_edges[b] <= v < bin_edges[b + 1]:
                return b
        return k_bins - 1

    for v in expected:
        exp_counts[place_in_bin(v)] += 1
    for v in actual:
        act_counts[place_in_bin(v)] += 1

    psi_total = 0.0
    bin_details = []

    for i in range(k_bins):
        # Laplace smoothing ensures non-zero probability mass in every bin
        p_exp = (exp_counts[i] + 1) / (n_exp + k_bins)
        p_act = (act_counts[i] + 1) / (n_act + k_bins)

        psi_i = (p_act - p_exp) * math.log(p_act / p_exp)
        psi_total += psi_i

        bin_details.append({
            "bin": f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})",
            "expected_count": exp_counts[i],
            "actual_count": act_counts[i],
            "expected_pct": round(p_exp, 4),
            "actual_pct": round(p_act, 4),
            "psi_contribution": round(psi_i, 5),
        })

    psi_val = round(max(0.0, psi_total), 4)

    if psi_val < 0.10:
        status = "STABLE"
        is_drifted = False
    elif psi_val <= 0.25:
        status = "MODERATE_SHIFT"
        is_drifted = False
    else:
        status = "SIGNIFICANT_DRIFT"
        is_drifted = True

    return {
        "psi": psi_val,
        "status": status,
        "is_drifted": is_drifted,
        "threshold": 0.25,
        "bin_details": bin_details,
    }


def compute_wasserstein_distance_1d(
    u_values: Sequence[float],
    v_values: Sequence[float],
) -> float:
    """
    Computes 1D Wasserstein-1 distance (Earth Mover's Distance) between two continuous samples.
    W_1(u, v) = integral_0^1 |U^{-1}(t) - V^{-1}(t)| dt
    Computed via the sorted quantile averages.
    """
    if len(u_values) == 0 or len(v_values) == 0:
        return 0.0

    u_sorted = sorted(u_values)
    v_sorted = sorted(v_values)

    # Standard step evaluation across combined points
    all_values = sorted(set(u_sorted + v_sorted))
    if len(all_values) <= 1:
        return 0.0

    n_u = len(u_sorted)
    n_v = len(v_sorted)

    # Compute empirical CDFs
    import bisect
    dist = 0.0
    for i in range(len(all_values) - 1):
        x0 = all_values[i]
        x1 = all_values[i + 1]
        dx = x1 - x0
        # CDF at midpoint
        mid = (x0 + x1) / 2.0
        cdf_u = bisect.bisect_right(u_sorted, mid) / n_u
        cdf_v = bisect.bisect_right(v_sorted, mid) / n_v
        dist += abs(cdf_u - cdf_v) * dx

    return round(dist, 4)
