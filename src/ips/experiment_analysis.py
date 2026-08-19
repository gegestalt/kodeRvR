"""Dynamic benchmarking and diagnostics for dataset-backed IPS policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd

from ips.dataset import EpisodeSplits
from ips.dataset_experiment import DatasetTrainConfig, train_one_seed
from ips.dqn import DqnConfig, QNetwork


@dataclass(frozen=True)
class Candidate:
    name: str
    hidden_dim: int
    gamma: float
    learning_rate: float
    batch_size: int
    target_update_steps: int
    episodes: int = 60


DEFAULT_CANDIDATES = (
    Candidate("DQN-small-fast", 16, 0.95, 1e-3, 16, 50),
    Candidate("DQN-small-long-horizon", 16, 0.99, 1e-3, 16, 50),
    Candidate("DQN-balanced", 32, 0.99, 1e-3, 32, 75),
    Candidate("DQN-low-lr", 32, 0.99, 3e-4, 32, 75),
    Candidate("DQN-large", 64, 0.99, 1e-3, 32, 100),
    Candidate("DQN-large-batch", 64, 0.99, 1e-3, 64, 100),
)


def factorial_candidates(episodes: int = 60) -> tuple[Candidate, ...]:
    """Complete 3×3 hidden-width × batch-size interaction grid."""
    return tuple(
        Candidate(f"DQN-h{hidden}-b{batch}", hidden, 0.99, 1e-3, batch, 75, episodes)
        for hidden in (16, 32, 64)
        for batch in (16, 32, 64)
    )


THEORETICAL_CANDIDATES = (
    {
        "model": "Masked PPO",
        "executed": False,
        "why": "stable policy-gradient comparison with the same discrete safety mask",
        "strength": "less sensitive than DQN to Q-value scale and replay distribution",
        "weakness": "less sample-efficient; needs a vectorized environment",
        "environment_fit": "realistic after dataset episodes and reward sensitivity stabilize",
    },
    {
        "model": "Constrained PPO",
        "executed": False,
        "why": "separates prevention reward from explicit disruption constraints",
        "strength": "directly models an availability/safety budget",
        "weakness": "more complex optimization and multiplier tuning",
        "environment_fit": "recommended final simulator/cyber-range policy",
    },
    {
        "model": "Contextual bandit",
        "executed": False,
        "why": "tests whether sequential credit assignment is genuinely necessary",
        "strength": "simple, fast, highly interpretable baseline",
        "weakness": "cannot value delayed containment or attack progression",
        "environment_fit": "strong additional baseline",
    },
)


def parameter_count(candidate: Candidate) -> int:
    model = QNetwork(7, 7, candidate.hidden_dim)
    return sum(parameter.numel() for parameter in model.parameters())


def benchmark_candidates(
    splits: EpisodeSplits,
    *,
    candidates: tuple[Candidate, ...] = DEFAULT_CANDIDATES,
    seed: int = 42,
    output_dir: Path,
) -> pd.DataFrame:
    """Actually train every candidate under one shared split and protocol."""
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        dqn = DqnConfig(
            hidden_dim=candidate.hidden_dim,
            gamma=candidate.gamma,
            learning_rate=candidate.learning_rate,
            batch_size=candidate.batch_size,
            replay_capacity=max(2_000, candidate.batch_size * 10),
            warmup_steps=candidate.batch_size * 2,
            target_update_steps=candidate.target_update_steps,
            epsilon_decay_steps=max(500, candidate.episodes * 10),
        )
        tracemalloc.start()
        started = time.perf_counter()
        run = train_one_seed(
            splits,
            seed=seed,
            dqn=dqn,
            training=DatasetTrainConfig(
                episodes=candidate.episodes,
                validation_interval=max(10, candidate.episodes // 3),
            ),
            output_dir=output_dir / candidate.name,
            evaluate_test=False,
        )
        runtime = time.perf_counter() - started
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Candidate discovery is validation-only. The final test is reserved for
        # the selected finalists in benchmark_finalists_multi_seed().
        metrics = run["validation"]
        rows.append(
            {
                **asdict(candidate),
                "seed": seed,
                "parameters": parameter_count(candidate),
                "quantization": "FP32",
                "runtime_s": runtime,
                "peak_python_memory_mb": peak_memory / 1024**2,
                "checkpoint_mb": Path(str(run["checkpoint"])).stat().st_size / 1024**2,
                "training_steps": run["steps"],
                "updates": run["updates"],
                "throughput_steps_s": run["steps"] / runtime,
                "evaluation_split": "validation",
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    frame["safety_quality"] = (
        frame["containment_rate"]
        - frame["compromise_rate"]
        - 0.05 * frame["false_preventions_per_episode"]
    )
    return frame


def benchmark_finalists_multi_seed(
    splits: EpisodeSplits,
    finalists: tuple[Candidate, ...],
    *,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    output_dir: Path,
) -> pd.DataFrame:
    """Repeat validation-selected finalists and evaluate final test once/seed."""
    if len(seeds) < 5:
        raise ValueError("finalist comparison requires at least five seeds")
    rows = []
    for candidate in finalists:
        for seed in seeds:
            dqn = DqnConfig(
                hidden_dim=candidate.hidden_dim,
                gamma=candidate.gamma,
                learning_rate=candidate.learning_rate,
                batch_size=candidate.batch_size,
                replay_capacity=max(2_000, candidate.batch_size * 10),
                warmup_steps=candidate.batch_size * 2,
                target_update_steps=candidate.target_update_steps,
                epsilon_decay_steps=max(500, candidate.episodes * 10),
            )
            run = train_one_seed(
                splits,
                seed=seed,
                dqn=dqn,
                training=DatasetTrainConfig(
                    episodes=candidate.episodes,
                    validation_interval=max(10, candidate.episodes // 3),
                ),
                output_dir=output_dir / candidate.name / f"seed_{seed}",
            )
            metrics = run["final_test"]
            rows.append(
                {
                    "name": candidate.name,
                    "seed": seed,
                    "parameters": parameter_count(candidate),
                    **metrics,
                    "safety_quality": metrics["containment_rate"]
                    - metrics["compromise_rate"]
                    - 0.05 * metrics["false_preventions_per_episode"],
                }
            )
    output = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "finalists_five_seed.csv", index=False)
    return output


def pareto_mask(
    frame: pd.DataFrame,
    *,
    maximize: tuple[str, ...] = ("safety_quality",),
    minimize: tuple[str, ...] = ("runtime_s", "checkpoint_mb"),
) -> pd.Series:
    """Return True for configurations not dominated on every objective."""
    values = frame[list(maximize) + list(minimize)].to_numpy(float, copy=True)
    values[:, len(maximize):] *= -1
    optimal = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        dominated = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        dominated[i] = False
        optimal[i] = not dominated.any()
    return pd.Series(optimal, index=frame.index, name="pareto_optimal")


def add_objective_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized trade-off scores without hiding their formula."""
    out = frame.copy()
    def normalize(column: str, higher: bool = True) -> pd.Series:
        values = out[column].astype(float)
        spread = values.max() - values.min()
        scaled = pd.Series(1.0, index=out.index) if spread == 0 else (values - values.min()) / spread
        return scaled if higher else 1.0 - scaled
    out["quality_norm"] = normalize("safety_quality")
    out["speed_norm"] = normalize("runtime_s", higher=False)
    out["memory_norm"] = normalize("checkpoint_mb", higher=False)
    out["overall_score"] = (
        0.60 * out["quality_norm"] + 0.25 * out["speed_norm"] + 0.15 * out["memory_norm"]
    )
    out["pareto_optimal"] = pareto_mask(out)
    return out


def parameter_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank monotonic parameter relationships using Spearman correlation."""
    parameters = ["hidden_dim", "gamma", "learning_rate", "batch_size", "target_update_steps"]
    outcomes = ["safety_quality", "runtime_s", "checkpoint_mb", "throughput_steps_s"]
    correlation = frame[parameters + outcomes].corr(method="spearman")
    rows = []
    for parameter in parameters:
        rows.append(
            {
                "parameter": parameter,
                **{f"rho_{outcome}": float(correlation.loc[parameter, outcome]) for outcome in outcomes},
                "importance": float(
                    correlation.loc[parameter, outcomes].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def interaction_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the most estimable interaction in the bounded candidate grid."""
    return frame.pivot_table(
        index="hidden_dim",
        columns="batch_size",
        values="safety_quality",
        aggfunc="mean",
    )


def detect_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag robust-MAD run anomalies and describe likely associated variables."""
    metrics = ["runtime_s", "peak_python_memory_mb", "safety_quality"]
    parameters = ["hidden_dim", "gamma", "learning_rate", "batch_size", "target_update_steps"]
    result = frame.copy()
    scores = pd.DataFrame(index=frame.index)
    for column in metrics:
        values = frame[column].astype(float)
        median = values.median()
        mad = np.median(np.abs(values - median))
        scores[column] = 0.0 if mad == 0 else 0.6745 * (values - median) / mad
    result["outlier_score"] = scores.abs().max(axis=1)
    result["is_outlier"] = result["outlier_score"] > 3.5
    explanations = []
    for index, row in result.iterrows():
        if not row["is_outlier"]:
            explanations.append("")
            continue
        metric = scores.loc[index].abs().idxmax()
        direction = "above" if scores.loc[index, metric] > 0 else "below"
        deviations = {}
        for parameter in parameters:
            scale = frame[parameter].std(ddof=0)
            deviations[parameter] = 0.0 if scale == 0 else abs(
                (float(row[parameter]) - frame[parameter].median()) / scale
            )
        likely = max(deviations, key=deviations.get)
        explanations.append(
            f"{metric} is robustly {direction} median; {likely} is the most unusual "
            "tested parameter. Association only—this grid does not prove causality."
        )
    result["outlier_explanation"] = explanations
    return result


def objective_winners(frame: pd.DataFrame) -> pd.DataFrame:
    scored = add_objective_scores(frame)
    choices = {
        "best_quality": scored["safety_quality"].idxmax(),
        "fastest": scored["runtime_s"].idxmin(),
        "lowest_checkpoint_memory": scored["checkpoint_mb"].idxmin(),
        "best_overall_60Q_25S_15M": scored["overall_score"].idxmax(),
    }
    return pd.DataFrame(
        [{"objective": objective, **scored.loc[index].to_dict()} for objective, index in choices.items()]
    )


def dynamic_recommendations(frame: pd.DataFrame) -> list[str]:
    scored = add_objective_scores(frame)
    winners = objective_winners(frame).set_index("objective")
    effects = parameter_effects(frame)
    best = winners.loc["best_overall_60Q_25S_15M"]
    messages = [
        f"Best observed overall trade-off: {best['name']} (score={best['overall_score']:.3f}).",
        f"Quality priority: {winners.loc['best_quality', 'name']} with safety_quality={winners.loc['best_quality', 'safety_quality']:.3f}.",
        f"Speed priority: {winners.loc['fastest', 'name']} at {winners.loc['fastest', 'runtime_s']:.3f}s.",
        f"Strongest measured monotonic association: {effects.iloc[0]['parameter']} (max |Spearman rho|={effects.iloc[0]['importance']:.2f}).",
        f"Pareto candidates: {', '.join(scored.loc[scored['pareto_optimal'], 'name'])}.",
    ]
    if frame["hidden_dim"].nunique() < 4:
        messages.append("Next experiment: add an intermediate/extended hidden-width point to resolve the quality-cost curve.")
    if frame["seed"].nunique() == 1:
        messages.append("Reliability gap: repeat Pareto finalists across at least five seeds before declaring a winner.")
    messages.append("Model-family gap: benchmark a contextual bandit and masked PPO only after the DQN reward-sensitivity check.")
    return messages
