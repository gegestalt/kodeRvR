"""Run the detector-first experiments requested after the POMDP correction."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import data as D
from ips.actions import IpsAction
from ips.advanced_policies import PpoConfig, actor_policy, train_bandit, train_ppo
from ips.belief import observable_belief
from ips.analysis.evidence import evaluate_detailed
from ips.analysis.next_phase import build_locked_holdout_manifest, detector_temporal_benchmark, source_leakage_curve
from ips.adapters.nsl_kdd import NslIpsConfig, build_nsl_ips_evidence
from ips.research_adoptions import optimal_stopping_policy
from ips.adapters.cse_temporal import split_events_by_role

ROOT = Path(__file__).resolve().parents[4]
EVENTS = ROOT / "data/ips_events/cse_cic_ids2018_temporal_events.parquet"
OUTPUT = ROOT / "results/notebook_ips_lab/next_phase"
OLD_POLICY = ROOT / "results/notebook_ips_lab/cse_temporal_evidence/policy_five_seed_runs.csv"
SEEDS = (42, 43, 44, 45, 46)


def _rule(state: np.ndarray, mask: np.ndarray) -> IpsAction:
    score = state[0]
    target = IpsAction.DROP_FLOW if score >= .75 else IpsAction.RATE_LIMIT if score >= .55 else IpsAction.MONITOR
    return target if mask[int(target)] else IpsAction.MONITOR


def _optimal(state: np.ndarray, mask: np.ndarray) -> IpsAction:
    return optimal_stopping_policy(state, mask)


def _policy_rows(events: pd.DataFrame) -> pd.DataFrame:
    splits = split_events_by_role(events, max_events=128)
    rows = []
    if OLD_POLICY.exists():
        old = pd.read_csv(OLD_POLICY)
        for item in old[old.method.eq("Uniform masked Double-DQN")].to_dict("records"):
            rows.append({**item, "method": "Masked Double-DQN", "observation_contract": "belief_state_v2"})
    for seed in SEEDS:
        for method, policy in (("Rule threshold", _rule), ("Optimal stopping", _optimal)):
            started = time.perf_counter(); _, _, metrics = evaluate_detailed(policy, splits.test, seed=seed)
            rows.append({"method": method, "seed": seed, "training_wall_s": time.perf_counter()-started,
                         **metrics, "observation_contract": "belief_state_v2"})
        started = time.perf_counter(); bandit = train_bandit(splits.train, training_episodes=60, seed=seed)
        _, _, metrics = evaluate_detailed(lambda state, mask, b=bandit: b.choose(state, mask), splits.test, seed=seed)
        rows.append({"method": "LinUCB", "seed": seed, "training_wall_s": time.perf_counter()-started,
                     **metrics, "observation_contract": "belief_state_v2"})
        started = time.perf_counter(); model, history = train_ppo(
            splits.train, config=PpoConfig(training_episodes=60, hidden_dim=32, update_epochs=2, constrained=True), seed=seed)
        _, _, metrics = evaluate_detailed(actor_policy(model), splits.test, seed=seed)
        rows.append({"method": "Constrained PPO", "seed": seed, "training_wall_s": time.perf_counter()-started,
                     **metrics, "constraint_cost_last20": history.disruption_cost.tail(20).mean(),
                     "lagrange_final": history.lagrange.iloc[-1], "observation_contract": "belief_state_v2"})
    return pd.DataFrame(rows)


def run() -> dict[str, object]:
    if not EVENTS.exists():
        raise FileNotFoundError("run python -m experiments.ips.cse.build_events first")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(EVENTS)
    events = events.assign(split_role=events.split_role.replace({"final_test": "development_test"}))
    detector_temporal_benchmark(events).to_csv(OUTPUT / "detector_temporal_benchmark.csv", index=False)

    sampled = []
    nsl = build_nsl_ips_evidence(
        D.load_nsl_kdd("train"), D.load_nsl_kdd("test"),
        NslIpsConfig(max_train_rows=6_000, max_test_rows=3_000, episode_size=12, folds=3),
    ).test_events
    for source, rows in (("CSE-CIC-IDS2018", events), ("NSL-KDD", nsl)):
        part = rows.sample(min(2500, len(rows)), random_state=42).copy()
        beliefs = [observable_belief(float(r.threat_probability), float(r.anomaly_score), [float(r.threat_probability)]) for r in part.itertuples()]
        part["estimated_stage"] = [x[0] for x in beliefs]; part["estimated_compromise"] = [x[1] for x in beliefs]
        part["recent_score_rate"] = [x[2] for x in beliefs]; part["source_dataset"] = source
        sampled.append(part)
    source_frame = pd.concat(sampled, ignore_index=True)
    features = ["threat_probability", "anomaly_score", "estimated_stage", "estimated_compromise", "recent_score_rate", "critical_service"]
    source_leakage_curve(source_frame, features, folds=5).to_csv(OUTPUT / "source_leakage_curve.csv", index=False)

    policies = _policy_rows(events.assign(split_role=events.split_role.replace({"development_test": "final_test"})))
    policies.to_csv(OUTPUT / "pomdp_policy_five_seed_runs.csv", index=False)
    policies.groupby("method", as_index=False).agg(
        seeds=("seed", "nunique"), containment_mean=("containment_rate", "mean"), containment_std=("containment_rate", "std"),
        compromise_mean=("compromise_rate", "mean"), return_mean=("mean_return", "mean"), return_std=("mean_return", "std"),
        false_preventions_mean=("false_preventions_per_episode", "mean"), training_wall_mean_s=("training_wall_s", "mean"),
    ).to_csv(OUTPUT / "pomdp_policy_five_seed_summary.csv", index=False)

    old = pd.read_csv(ROOT / "results/notebook_ips_lab/cse_temporal_evidence/policy_five_seed_summary.csv")
    dqn = old[old.method.str.contains("DQN")]
    replay = pd.DataFrame([{
        "uniform_containment": dqn[dqn.method.str.contains("Uniform")].containment_mean.iloc[0],
        "per_containment": dqn[dqn.method.str.contains("Prioritized")].containment_mean.iloc[0],
        "absolute_difference": abs(dqn.containment_mean.diff().dropna().iloc[0]),
        "interpretation": "No observed policy-level benefit from PER; inspect sampled-index/TD-error telemetry before tuning",
        "decision": "RL complexity not justified while optimal stopping has better containment",
    }])
    replay.to_csv(OUTPUT / "replay_equivalence_diagnostic.csv", index=False)

    pd.DataFrame([{"finding": "uncontained attacks with zero compromise", "status": "ENVIRONMENT CONSEQUENCE GAP",
                   "reason": "counterfactual episode windows may truncate before delayed compromise",
                   "required_fix": "log terminal cause and calibrate transition hazards from observed intervention/cyber-range data"}]).to_csv(OUTPUT / "consequence_gap_audit.csv", index=False)
    manifest = build_locked_holdout_manifest(
        ROOT / "data/cse_cic_ids2018/LOCKED_FINAL_HOLDOUT.json",
        ["Monday-19-02-2018_TrafficForML_CICFlowMeter.csv", "Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"],
    )
    status = {"feb16_role": "DEVELOPMENT_TEST", "locked_holdout": manifest["status"],
              "policy_evidence": "counterfactual dataset replay", "observed_interventions": 0,
              "primary_decision": "fix detector evidence before policy optimization",
              "rl_necessity": "NOT DEMONSTRATED"}
    (OUTPUT / "next_phase_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
