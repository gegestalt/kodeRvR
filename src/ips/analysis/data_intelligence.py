"""Data-health diagnostics for CICAPT-IIoT2024.

These routines explain dataset and representation quality.  They deliberately
exclude the locked final holdout from fitted statistics and model diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score


def feature_health(frame: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Return auditable quality statistics for numeric detector features."""
    rows: list[dict[str, object]] = []
    for feature in features:
        raw = pd.to_numeric(frame[feature], errors="coerce")
        finite = raw.replace([np.inf, -np.inf], np.nan)
        valid = finite.dropna()
        counts = valid.value_counts(dropna=False)
        dominant = float(counts.iloc[0] / len(valid)) if len(valid) else 1.0
        quantiles = valid.quantile([.01, .25, .5, .75, .99]) if len(valid) else pd.Series(dtype=float)
        missing = int(raw.isna().sum())
        inf = int(np.isinf(raw.to_numpy(dtype=float, na_value=np.nan)).sum())
        unique = int(valid.nunique())
        skew = float(valid.skew()) if len(valid) > 2 else np.nan
        constant = unique <= 1
        near_constant = not constant and dominant > .999
        status = "FAIL" if missing or inf or constant else "WARN" if near_constant or abs(skew) > 20 or unique > .9 * len(valid) else "PASS"
        rows.append({
            "feature": feature, "dtype": str(frame[feature].dtype), "rows_audited": len(frame),
            "missing_count": missing, "missing_pct": missing / max(len(frame), 1), "inf_count": inf,
            "zero_pct": float(valid.eq(0).mean()) if len(valid) else np.nan, "unique_count": unique,
            "dominant_pct": dominant, "constant": constant, "near_constant": near_constant,
            "min": valid.min() if len(valid) else np.nan, "p01": quantiles.get(.01, np.nan),
            "p25": quantiles.get(.25, np.nan), "median": quantiles.get(.5, np.nan),
            "p75": quantiles.get(.75, np.nan), "p99": quantiles.get(.99, np.nan),
            "max": valid.max() if len(valid) else np.nan, "mean": valid.mean() if len(valid) else np.nan,
            "std": valid.std() if len(valid) else np.nan, "skew": skew, "status": status,
        })
    return pd.DataFrame(rows)


def _distribution_distances(reference: pd.Series, candidate: pd.Series, bins: int = 20) -> dict[str, float]:
    reference = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    candidate = pd.to_numeric(candidate, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if not len(reference) or not len(candidate):
        return {name: np.nan for name in ("psi", "ks", "jsd", "wasserstein")}
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return {"psi": 0.0, "ks": float(ks_2samp(reference, candidate).statistic), "jsd": 0.0,
                "wasserstein": float(wasserstein_distance(reference, candidate))}
    ref_hist = np.histogram(reference, bins=edges)[0].astype(float) + 1e-8
    cand_hist = np.histogram(candidate, bins=edges)[0].astype(float) + 1e-8
    ref_hist /= ref_hist.sum(); cand_hist /= cand_hist.sum()
    psi = float(np.sum((cand_hist - ref_hist) * np.log(cand_hist / ref_hist)))
    return {"psi": psi, "ks": float(ks_2samp(reference, candidate).statistic),
            "jsd": float(jensenshannon(ref_hist, cand_hist) ** 2),
            "wasserstein": float(wasserstein_distance(reference, candidate))}


def temporal_feature_drift(events: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Compare train-day features with validation/development; never final."""
    rows = []
    reference = events.loc[events.split_role.eq("train")]
    for role in ("validation", "development_test"):
        candidate = events.loc[events.split_role.eq(role)]
        for feature in features:
            rows.append({"feature": feature, "reference_role": "train", "candidate_role": role,
                         **_distribution_distances(reference[feature], candidate[feature])})
    return pd.DataFrame(rows)


def benign_sampling_fidelity(reference: pd.DataFrame, sampled: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Compare an independently streamed benign reference with kept benign rows."""
    rows = []
    for feature in features:
        rows.append({"feature": feature, **_distribution_distances(reference[feature], sampled[feature])})
    result = pd.DataFrame(rows)
    result["psi_band"] = pd.cut(result.psi, [-np.inf, .1, .25, np.inf], labels=["stable", "moderate", "unstable"])
    return result


def feature_separation(events: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Rank univariate benign/attack separation without fitting on final data."""
    visible = events.loc[~events.split_role.eq("locked_final_holdout")].copy()
    y = visible.attack_present.astype(int).to_numpy()
    rows = []
    for feature in features:
        values = pd.to_numeric(visible[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median()).to_numpy()
        attack, benign = values[y == 1], values[y == 0]
        std = np.sqrt((np.var(attack) + np.var(benign)) / 2) if len(attack) and len(benign) else 0
        score = pd.Series(values).rank(pct=True).to_numpy()
        pr = average_precision_score(y, score)
        inverse_pr = average_precision_score(y, 1 - score)
        rows.append({"feature": feature, "univariate_pr_auc": max(pr, inverse_pr),
                     "direction": "high" if pr >= inverse_pr else "low",
                     "ks": float(ks_2samp(benign, attack).statistic),
                     "standardized_effect": float((np.mean(attack) - np.mean(benign)) / std) if std else 0.0})
    return pd.DataFrame(rows).sort_values("univariate_pr_auc", ascending=False)


def correlation_redundancy(events: pd.DataFrame, features: Iterable[str], *, threshold: float = .98) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return clustered-plot input and near-duplicate feature pairs."""
    correlation = events.loc[~events.split_role.eq("locked_final_holdout"), list(features)].corr(method="spearman")
    pairs = []
    for left_index, left in enumerate(correlation.columns):
        for right in correlation.columns[left_index + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"feature_a": left, "feature_b": right, "spearman_rho": float(value)})
    return correlation, pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman_rho"])


def provenance_window_statistics(provenance: pd.DataFrame, *, window_seconds: int = 60) -> pd.DataFrame:
    """Build graph-structure features per window from native node/edge rows."""
    frame = provenance.copy()
    timestamp = frame["time"].combine_first(frame["seen time"]).combine_first(frame["start time"])
    frame = frame.loc[timestamp.notna()].copy()
    frame["window"] = (pd.to_numeric(timestamp[timestamp.notna()]) // window_seconds).astype("int64")
    kinds = frame.type.astype(str).str.casefold()
    frame["is_process"] = kinds.eq("process")
    frame["is_artifact"] = kinds.eq("artifact")
    frame["is_edge"] = kinds.str.contains("used|wasgeneratedby|wastriggeredby|wasderivedfrom|edge", regex=True)
    grouped = frame.groupby("window")
    result = grouped.agg(rows=("id", "size"), new_processes=("is_process", "sum"),
                         new_artifacts=("is_artifact", "sum"), new_edges=("is_edge", "sum"),
                         unique_processes=("pid", "nunique"), attack_present=("label", "max"),
                         tactic=("subLabel", lambda values: next((str(v) for v in values if str(v) not in {"0", "nan"}), "normal")))
    sockets = frame.get("remote address", pd.Series(index=frame.index, dtype=object)).notna()
    frame["is_socket"] = sockets
    result["new_sockets"] = frame.groupby("window").is_socket.sum()
    result["socket_fan_out"] = frame.groupby("window")["remote address"].nunique() if "remote address" in frame else 0
    result["graph_density_proxy"] = result.new_edges / np.maximum((result.new_processes + result.new_artifacts) ** 2, 1)
    result["edges_per_minute"] = result.new_edges * 60 / window_seconds
    return result.reset_index()


def tactic_split_coverage(events: pd.DataFrame) -> pd.DataFrame:
    """Seen/unseen matrix with final-holdout labels intentionally concealed."""
    visible = events.loc[~events.split_role.eq("locked_final_holdout") & events.attack_present]
    counts = visible.groupby(["attack_tactic", "split_role"]).size().unstack(fill_value=0)
    for role in ("train", "validation", "development_test"):
        if role not in counts: counts[role] = 0
    counts["locked_final_holdout"] = "LOCKED"
    counts["seen_in_train"] = counts["train"].gt(0)
    return counts.reset_index()


def technique_day_distribution(events: pd.DataFrame) -> pd.DataFrame:
    """Technique counts for development roles, hiding locked-final contents."""
    visible = events.loc[~events.split_role.eq("locked_final_holdout") & events.attack_present]
    table = visible.groupby(["attack_tactic", "attack_technique", "split_role"]).size().unstack(fill_value=0)
    for role in ("train", "validation", "development_test"):
        if role not in table: table[role] = 0
    table["locked_final_holdout"] = "LOCKED"
    table["seen_in_train"] = table["train"].gt(0)
    return table.reset_index()


def uncertainty_diagnostics(beliefs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Risk-coverage and confidence diagnostics for known/OOD tactic beliefs."""
    probability_columns = [column for column in beliefs if column.startswith("p_")]
    frame = beliefs.copy()
    frame["correct"] = frame.predicted_tactic.eq(frame.true_tactic)
    frame["confidence"] = frame[["normal_probability", *probability_columns]].max(axis=1)
    ordered = frame.sort_values("uncertainty", kind="stable")
    rows = []
    for coverage in np.linspace(.1, 1, 10):
        kept = ordered.iloc[:max(1, int(np.ceil(len(ordered) * coverage)))]
        rows.append({"coverage": coverage, "selective_error": float((~kept.correct).mean()),
                     "attack_error": float((~kept.loc[kept.true_tactic.ne("normal"), "correct"]).mean()) if kept.true_tactic.ne("normal").any() else np.nan})
    known = set(frame.attrs.get("known_tactics", []))
    ood = frame.true_tactic.ne("normal") & ~frame.true_tactic.isin(known)
    ood_score = frame.uncertainty.to_numpy()
    metrics = {"mean_uncertainty_correct": float(frame.loc[frame.correct, "uncertainty"].mean()),
               "mean_uncertainty_wrong": float(frame.loc[~frame.correct, "uncertainty"].mean()),
               "false_ood_rate_at_0_5": float((ood_score[~ood.to_numpy()] >= .5).mean())}
    if ood.nunique() == 2:
        metrics.update({"ood_auroc": float(roc_auc_score(ood, ood_score)),
                        "ood_aupr": float(average_precision_score(ood, ood_score))})
    else:
        metrics.update({"ood_auroc": np.nan, "ood_aupr": np.nan})
    return frame[["timestamp", "true_tactic", "predicted_tactic", "correct", "confidence", "uncertainty"]], pd.DataFrame(rows), metrics


def label_alignment(timeline: pd.DataFrame, network_attack_times: pd.Series, provenance_attack_times: pd.Series) -> pd.DataFrame:
    """Nearest-clock deltas for every visible campaign step."""
    steps = pd.to_datetime(timeline.attack_time, utc=True).map(lambda value: value.timestamp()).to_numpy(dtype=float)
    network = np.sort(pd.to_numeric(network_attack_times, errors="coerce").dropna().to_numpy())
    provenance = np.sort(pd.to_numeric(provenance_attack_times, errors="coerce").dropna().to_numpy())
    def nearest(values: np.ndarray, point: float) -> float:
        if not len(values): return np.nan
        index = np.searchsorted(values, point)
        choices = values[max(0, index - 1):min(len(values), index + 1)]
        return float(np.min(np.abs(choices - point)))
    result = timeline[["campaign_step", "attack_time", "tactic", "technique"]].copy()
    result["network_delta_s"] = [nearest(network, point) for point in steps]
    result["provenance_delta_s"] = [nearest(provenance, point) for point in steps]
    return result


def join_tolerance_sensitivity(alignment: pd.DataFrame, tolerances: Iterable[int]) -> pd.DataFrame:
    """Quantify evidence coverage as the declared clock tolerance changes."""
    rows = []
    for tolerance in tolerances:
        network = alignment.network_delta_s.le(tolerance)
        provenance = alignment.provenance_delta_s.le(tolerance)
        rows.append({"tolerance_seconds": tolerance, "campaign_steps": len(alignment),
                     "network_matched_steps": int(network.sum()), "provenance_matched_steps": int(provenance.sum()),
                     "both_matched_steps": int((network & provenance).sum()),
                     "neither_matched_steps": int((~network & ~provenance).sum())})
    return pd.DataFrame(rows)
