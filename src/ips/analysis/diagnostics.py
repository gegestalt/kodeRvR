"""Deeper, evidence-labelled diagnostics for adaptive IPS experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


PARAMETERS = ("hidden_dim", "batch_size", "gamma", "learning_rate", "target_update_steps")
OUTCOMES = ("safety_quality", "runtime_s", "checkpoint_mb", "throughput_steps_s")


def parameter_importance(frame: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Return Spearman, permutation, and effect-size evidence per outcome.

    Permutation values are in-sample diagnostics, not causal estimates. Small
    candidate grids are labelled exploratory so rankings cannot masquerade as
    statistically established importance.
    """
    parameters = [column for column in PARAMETERS if frame[column].nunique() > 1]
    if len(frame) < 4 or not parameters:
        raise ValueError("parameter importance requires >=4 varied configurations")
    X = frame[parameters].astype(float)
    rows = []
    evidence = "exploratory: n<30" if len(frame) < 30 else "moderate: n>=30"
    for outcome in OUTCOMES:
        y = frame[outcome].astype(float)
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed)
        model.fit(X, y)
        permutation = permutation_importance(
            model, X, y, n_repeats=30, random_state=seed, scoring="neg_mean_squared_error"
        )
        y_scale = max(float(y.std(ddof=0)), 1e-12)
        for index, parameter in enumerate(parameters):
            groups = frame.groupby(parameter)[outcome].mean()
            standardized_range = float((groups.max() - groups.min()) / y_scale)
            rho = float(frame[[parameter, outcome]].corr(method="spearman").iloc[0, 1])
            rows.append(
                {
                    "parameter": parameter,
                    "outcome": outcome,
                    "spearman_rho": rho,
                    "standardized_group_range": standardized_range,
                    "permutation_importance": float(permutation.importances_mean[index]),
                    "permutation_std": float(permutation.importances_std[index]),
                    "evidence_strength": evidence,
                    "causal_claim": False,
                }
            )
    result = pd.DataFrame(rows)
    result["rank_within_outcome"] = result.groupby("outcome")["permutation_importance"].rank(
        method="dense", ascending=False
    )
    return result.sort_values(["outcome", "rank_within_outcome"])


def matched_outlier_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare every flagged run with its nearest non-outlier parameter control."""
    if "is_outlier" not in frame:
        raise ValueError("frame must include is_outlier")
    outliers, controls = frame[frame.is_outlier], frame[~frame.is_outlier]
    if outliers.empty:
        return pd.DataFrame()
    if controls.empty:
        raise ValueError("matched diagnostics require at least one normal control")
    parameters = [column for column in PARAMETERS if frame[column].nunique() > 1]
    scaler = StandardScaler().fit(frame[parameters].astype(float))
    all_scaled = pd.DataFrame(scaler.transform(frame[parameters]), index=frame.index, columns=parameters)
    rows = []
    for index, run in outliers.iterrows():
        distances = ((all_scaled.loc[controls.index] - all_scaled.loc[index]) ** 2).sum(axis=1) ** 0.5
        control_index = distances.idxmin()
        control = controls.loc[control_index]
        parameter_delta = (all_scaled.loc[index] - all_scaled.loc[control_index]).abs()
        unusual = str(parameter_delta.idxmax())
        def percent(column: str) -> float:
            denominator = max(abs(float(control[column])), 1e-12)
            return 100 * (float(run[column]) - float(control[column])) / denominator
        rows.append(
            {
                "run": run["name"],
                "matched_control": control["name"],
                "parameter_distance": float(distances.loc[control_index]),
                "most_unusual_parameter": unusual,
                "parameter_delta_standardized": float(parameter_delta[unusual]),
                "runtime_pct_vs_control": percent("runtime_s"),
                "memory_pct_vs_control": percent("peak_python_memory_mb"),
                "quality_pct_vs_control": percent("safety_quality"),
                "explanation": (
                    f"Compared with nearest tested control {control['name']}, {unusual} has the "
                    f"largest standardized parameter difference ({parameter_delta[unusual]:.2f} SD). "
                    "This is an association within the tested grid, not causal attribution."
                ),
            }
        )
    return pd.DataFrame(rows)


def expected_vs_observed(frame: pd.DataFrame) -> pd.DataFrame:
    """Test directional engineering expectations against measured associations."""
    checks = (
        ("Larger hidden width increases checkpoint memory", "hidden_dim", "checkpoint_mb", 1),
        ("Larger hidden width increases training runtime", "hidden_dim", "runtime_s", 1),
        ("Larger batches improve step throughput", "batch_size", "throughput_steps_s", 1),
        ("Larger batches increase Python peak memory", "batch_size", "peak_python_memory_mb", 1),
    )
    rows = []
    for expectation, parameter, outcome, expected_sign in checks:
        rho = float(frame[[parameter, outcome]].corr(method="spearman").iloc[0, 1])
        if not np.isfinite(rho) or abs(rho) < 0.20:
            status = "inconclusive"
        elif np.sign(rho) == expected_sign:
            status = "supported"
        else:
            status = "contradicted"
        rows.append(
            {
                "expectation": expectation,
                "parameter": parameter,
                "outcome": outcome,
                "observed": f"Spearman rho={rho:.3f}",
                "status": status,
                "next_action": (
                    "repeat with more seeds/configurations"
                    if status == "inconclusive"
                    else "investigate confounding and measurement noise"
                    if status == "contradicted"
                    else "retain as measured engineering expectation"
                ),
            }
        )
    return pd.DataFrame(rows)


def repeated_run_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize seed-level policy measurements with uncertainty and stability."""
    required = {"policy", "seed", "mean_return", "containment_rate"}
    if missing := required - set(runs):
        raise ValueError(f"repeated runs missing columns: {sorted(missing)}")
    rows = []
    for policy, group in runs.groupby("policy"):
        values = group.mean_return.astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        cv = std / max(abs(mean), 1e-12)
        rows.append(
            {
                "policy": policy,
                "seed_count": len(group),
                "return_mean": mean,
                "return_median": float(values.median()),
                "return_std": std,
                "return_min": float(values.min()),
                "return_max": float(values.max()),
                "return_cv": cv,
                "return_ci95_halfwidth": 1.96 * std / np.sqrt(len(group)),
                "containment_mean": float(group.containment_rate.mean()),
                "stability": "high" if cv <= 0.10 else "moderate" if cv <= 0.25 else "low",
            }
        )
    return pd.DataFrame(rows).sort_values("return_mean", ascending=False)


def pairwise_bootstrap_decisions(
    runs: pd.DataFrame, *, metric: str = "mean_return", repeats: int = 10_000, seed: int = 42
) -> pd.DataFrame:
    """Bootstrap pairwise seed means and report winner, tie, or inconclusive."""
    rng = np.random.default_rng(seed)
    policies = sorted(runs.policy.unique())
    rows = []
    for left_index, left in enumerate(policies):
        for right in policies[left_index + 1 :]:
            a = runs.loc[runs.policy == left, metric].to_numpy(float)
            b = runs.loc[runs.policy == right, metric].to_numpy(float)
            differences = np.empty(repeats)
            for index in range(repeats):
                differences[index] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
            low, high = np.percentile(differences, [2.5, 97.5])
            if low > 0:
                decision = f"{left} wins"
            elif high < 0:
                decision = f"{right} wins"
            elif abs(differences.mean()) <= 0.05 * max(abs(np.r_[a, b]).mean(), 1e-12):
                decision = "practical tie"
            else:
                decision = "inconclusive"
            rows.append(
                {"policy_a": left, "policy_b": right, "mean_difference": differences.mean(),
                 "ci95_low": low, "ci95_high": high, "decision": decision}
            )
    return pd.DataFrame(rows)
