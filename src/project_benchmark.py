"""One-command health report and benchmark showcase for the full project.

Quick report (no expensive retraining):
    .venv/bin/python src/project_benchmark.py

Full verification plus a bounded DQN smoke run:
    .venv/bin/python src/project_benchmark.py --full

The script inventories datasets, features, implemented experiment tracks, saved
results, test health, fixed IPS policies, and the optional learned IPS smoke
benchmark. Historical ML models are not silently retrained; their committed CSV
artifacts are summarized and labelled as saved results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

import data as D
import dataset_catalog as DC
from ips.dqn import DqnConfig
from ips.evaluate import evaluate_policy
from ips.policies import aggressive_policy, allow_policy, rule_based_policy
from ips.train_dqn import TrainConfig, train

RESULTS = D.REPO_ROOT / "results"

TRACKS: tuple[tuple[str, str, str], ...] = (
    ("Leakage-safe preprocessing", "PROVEN", "src/preprocess.py + tests"),
    ("Supervised reference models", "PROVEN", "results/reference_track.csv"),
    ("RF/LightGBM official split", "PROVEN", "results/metrics.md"),
    ("Weighted/unweighted MLP", "PROVEN", "results/stability.md"),
    ("Threshold tuning", "PROVEN", "results/threshold_ablation.csv"),
    ("Normal-only anomaly detection", "PROVEN", "results/anomaly_detection.csv"),
    ("Semi-supervised learning", "PROVEN", "results/semi_supervised.csv"),
    ("Online partial-fit proxy", "PARTIAL", "not chronological drift"),
    ("CICIoT2023 supervised dev", "PARTIAL", "random dev split, not full raw CSV"),
    ("Adaptive IPS environment", "PROVEN", "src/ips + safety tests"),
    ("Masked Double DQN", "IMPLEMENTED", "src/ips/train_dqn.py"),
    ("True temporal drift", "BLOCKED", "timestamped local data required"),
    ("Cross-dataset NetFlow transfer", "BLOCKED", "shared-schema data required"),
    ("Cyber-range IPS validation", "PLANNED", "contained testbed required"),
)

MODEL_TRAINING: tuple[dict[str, str], ...] = (
    {
        "dataset_task": "NSL-KDD binary + 5-class",
        "models": "Dummy, LogReg, RF, ExtraTrees, HistGB, LightGBM",
        "status": "TRAINED / SAVED",
        "fit_selection": "official train; class balancing; train-only validation/CV",
        "evaluation": "official KDDTest+; macro-F1 + per-class recall",
        "run": ".venv/bin/python src/reference_track.py",
    },
    {
        "dataset_task": "NSL-KDD binary + 5-class",
        "models": "MLP unweighted + class-weighted",
        "status": "TRAINED / 5-SEED",
        "fit_selection": "official train; internal early-stopping validation",
        "evaluation": "official KDDTest+; macro-F1 and rare-family recall",
        "run": ".venv/bin/python src/train_mlp.py",
    },
    {
        "dataset_task": "NSL-KDD anomaly",
        "models": "IsolationForest, LOF, KMeans distance",
        "status": "TRAINED / SAVED",
        "fit_selection": "normal training rows only; train-normal quantile threshold",
        "evaluation": "official KDDTest+ attack/family recall",
        "run": ".venv/bin/python src/anomaly_detection.py",
    },
    {
        "dataset_task": "CICIoT2023 dev binary + category",
        "models": "Dummy, LogReg, RF, HistGB, LightGBM",
        "status": "TRAINED / DEV ONLY",
        "fit_selection": "stratified 200k train sample; random dev partition",
        "evaluation": "full dev test partition; not raw/temporal evidence",
        "run": ".venv/bin/python src/ciciot2023_baselines.py",
    },
    {
        "dataset_task": "Adaptive IPS simulated episodes",
        "models": "Allow-only, rule-based, aggressive",
        "status": "RUN LIVE IN THIS REPORT",
        "fit_selection": "no fitting; identical seeded scenarios",
        "evaluation": "containment, compromise, disruption, false prevention, return",
        "run": ".venv/bin/python src/project_benchmark.py",
    },
    {
        "dataset_task": "Adaptive IPS simulated episodes",
        "models": "Masked Double DQN",
        "status": "IMPLEMENTED / SMOKE WITH --full",
        "fit_selection": "training seed range; disjoint validation seed range",
        "evaluation": "safety-first checkpoint ordering; simulator evidence only",
        "run": ".venv/bin/python src/project_benchmark.py --full",
    },
)

METRIC_DEFINITIONS: tuple[dict[str, str], ...] = (
    {"metric": "accuracy", "calculation": "correct predictions / all predictions", "use": "secondary under imbalance"},
    {"metric": "per-class recall", "calculation": "TP / (TP + FN) for that class", "use": "how many attacks of each family are caught"},
    {"metric": "macro-F1", "calculation": "unweighted mean of each class F1; F1=2PR/(P+R)", "use": "primary classifier comparison"},
    {"metric": "weighted-F1", "calculation": "class F1 weighted by class support", "use": "overall volume-weighted quality"},
    {"metric": "ROC-AUC", "calculation": "area under TPR versus FPR across thresholds", "use": "binary ranking; can flatter imbalance"},
    {"metric": "PR-AUC", "calculation": "area under precision versus recall", "use": "preferred binary ranking under imbalance"},
    {"metric": "containment rate", "calculation": "contained attack episodes / attack episodes", "use": "IPS prevention outcome"},
    {"metric": "compromise rate", "calculation": "compromised attack episodes / attack episodes", "use": "lower is better; first checkpoint criterion"},
    {"metric": "false prevention", "calculation": "disruptive actions in benign episodes / episodes", "use": "availability/safety cost"},
    {"metric": "mean return", "calculation": "mean sum of transition rewards per episode", "use": "RL objective; never interpreted alone"},
)

SAVED_BENCHMARKS: tuple[tuple[str, str], ...] = (
    ("NSL-KDD reference", "reference_track.csv"),
    ("CICIoT2023 dev", "ciciot2023_baselines.csv"),
    ("Threshold ablation", "threshold_ablation.csv"),
    ("Anomaly detection", "anomaly_detection.csv"),
    ("Semi-supervised", "semi_supervised.csv"),
    ("Online proxy", "online_learning.csv"),
    ("Neural ablation", "neural_ablation.csv"),
)


def dependency_versions() -> dict[str, str]:
    """Return versions of the runtime dependencies most relevant to the lab."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "lightgbm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    return versions


def dataset_rows() -> list[dict[str, str]]:
    """Use the canonical catalog so missing local data is reported honestly."""
    return [
        {
            "dataset": entry.name,
            "role": entry.role,
            "status": entry.current_status,
            "limitation": entry.blocked_claim,
        }
        for entry in DC.build_entries()
    ]


def feature_rows() -> list[dict[str, str | int]]:
    """Describe currently implemented feature/state representations."""
    rows: list[dict[str, str | int]] = [
        {
            "representation": "NSL-KDD raw flow",
            "features": len(D.FEATURE_NAMES),
            "details": (
                f"{len(D.CATEGORICAL_COLS)} categorical + "
                f"{len(D.NUMERIC_COLS)} numeric; difficulty excluded"
            ),
        },
        {
            "representation": "Adaptive IPS state",
            "features": 7,
            "details": (
                "threat probability, anomaly, attack stage, compromise, "
                "criticality, attack rate, response budget"
            ),
        },
    ]
    ciciot_train = D.REPO_ROOT / "data" / "ciciot2023" / "train.parquet"
    if ciciot_train.exists():
        frame = pd.read_parquet(ciciot_train)
        feature_count = len(
            [c for c in frame.columns if c not in {"Label", "attack_class", "label"}]
        )
        detail = "numeric dev-parquet features; labels/provenance excluded"
    else:
        feature_count, detail = 46, "published expected model features; local parquet absent"
    rows.append(
        {
            "representation": "CICIoT2023",
            "features": feature_count,
            "details": detail,
        }
    )
    return rows


def saved_benchmark_rows() -> list[dict[str, Any]]:
    """Summarize the best saved macro-F1 row in each compatible artifact."""
    rows: list[dict[str, Any]] = []
    for track, filename in SAVED_BENCHMARKS:
        path = RESULTS / filename
        row: dict[str, Any] = {
            "track": track,
            "artifact": f"results/{filename}",
            "status": "available" if path.exists() else "missing",
            "best_macro_f1": None,
            "best_method": None,
        }
        if path.exists():
            frame = pd.read_csv(path)
            if "macro_f1" in frame and frame["macro_f1"].notna().any():
                best = frame.loc[frame["macro_f1"].astype(float).idxmax()]
                row["best_macro_f1"] = float(best["macro_f1"])
                for candidate in ("model", "method", "configuration", "variant"):
                    if candidate in frame.columns:
                        row["best_method"] = str(best[candidate])
                        break
        rows.append(row)
    return rows


def ips_baseline_rows(episodes: int, seed: int) -> list[dict[str, Any]]:
    """Run identical seeded scenarios for required non-learning IPS baselines."""
    policies = {
        "allow_only": allow_policy,
        "rule_based": rule_based_policy,
        "aggressive": aggressive_policy,
    }
    return [
        {"policy": name, **evaluate_policy(policy, episodes=episodes, seed=seed).to_dict()}
        for name, policy in policies.items()
    ]


def run_pytest() -> dict[str, Any]:
    """Run the complete suite and capture the final status without shell parsing."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
        cwd=D.REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    return {"passed": completed.returncode == 0, "returncode": completed.returncode, "output": output}


def run_dqn_smoke(output_dir: Path, seed: int) -> dict[str, Any]:
    """Run a bounded training check; this is not a publishable final result."""
    result = train(
        DqnConfig(
            hidden_dim=32,
            batch_size=32,
            replay_capacity=2_000,
            warmup_steps=64,
            target_update_steps=100,
            epsilon_decay_steps=1_500,
        ),
        TrainConfig(episodes=75, validation_interval=25, validation_episodes=50),
        seed=seed,
        output_dir=output_dir / "dqn_smoke",
    )
    return {
        "status": "smoke_only",
        "total_steps": result.total_steps,
        "updates": result.updates,
        "best_validation": result.best_validation.to_dict(),
        "checkpoint": str(result.best_checkpoint),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value if value is not None else "—").replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise, human-readable companion to the machine JSON report."""
    lines = ["# Project Health and Benchmark Report", "", "## Runtime", ""]
    lines += _markdown_table(
        [{"dependency": k, "version": v} for k, v in report["dependencies"].items()],
        ["dependency", "version"],
    )
    lines += ["", "## Datasets", ""]
    lines += _markdown_table(report["datasets"], ["dataset", "status", "role", "limitation"])
    lines += ["", "## Feature representations", ""]
    lines += _markdown_table(report["features"], ["representation", "features", "details"])
    lines += ["", "## Experiment coverage", ""]
    lines += _markdown_table(report["tracks"], ["track", "status", "evidence"])
    lines += ["", "## Model training and evaluation status", ""]
    lines += _markdown_table(
        report["model_training"],
        ["dataset_task", "models", "status", "fit_selection", "evaluation", "run"],
    )
    lines += ["", "## How metrics are calculated", ""]
    lines += _markdown_table(
        report["metric_definitions"], ["metric", "calculation", "use"]
    )
    lines += ["", "## Saved model benchmarks", ""]
    lines += _markdown_table(
        report["saved_benchmarks"],
        ["track", "status", "best_macro_f1", "best_method", "artifact"],
    )
    lines += ["", "## Adaptive IPS policy benchmarks", ""]
    lines += _markdown_table(
        report["ips_baselines"],
        [
            "policy", "mean_return", "containment_rate", "compromise_rate",
            "false_preventions_per_episode", "disruptive_actions_per_episode",
        ],
    )
    tests = report.get("tests")
    if tests is not None:
        lines += ["", "## Test suite", "", f"Passed: **{tests['passed']}**", "", "```text"]
        lines += tests["output"].splitlines()
        lines += ["```"]
    if report.get("dqn_smoke") is not None:
        lines += ["", "## DQN smoke benchmark", "", "```json"]
        lines += json.dumps(report["dqn_smoke"], indent=2, sort_keys=True).splitlines()
        lines += ["```"]
    lines += [
        "",
        "## Comparison rules",
        "",
        "1. Compare models directly only when dataset, label task, split, and metric are identical.",
        "2. Do not rank CICIoT2023 random-dev scores against NSL-KDD official-shift scores.",
        "3. Do not treat anomaly detection, supervised classification, and IPS return as one leaderboard.",
        "4. A DQN checkpoint is ordered by lower compromise, higher containment, lower false prevention, then return.",
        "5. Simulator IPS metrics demonstrate algorithm behavior, not live-network prevention effectiveness.",
        "",
        "## Interpretation boundary",
        "",
        "Saved supervised metrics come from their documented dataset splits. IPS policy "
        "metrics currently come from the seeded simulator and are not live-network or "
        "cyber-range evidence. The DQN smoke run verifies training mechanics only.",
        "",
    ]
    return "\n".join(lines)


def build_report(*, full: bool, episodes: int, seed: int, output_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dependencies": dependency_versions(),
        "datasets": dataset_rows(),
        "features": feature_rows(),
        "tracks": [
            {"track": track, "status": status, "evidence": evidence}
            for track, status, evidence in TRACKS
        ],
        "model_training": list(MODEL_TRAINING),
        "metric_definitions": list(METRIC_DEFINITIONS),
        "saved_benchmarks": saved_benchmark_rows(),
        "ips_baselines": ips_baseline_rows(episodes, seed),
        "artifact_counts": {
            "figures": len(list((RESULTS / "figures").glob("*.png"))),
            "csv_results": len(list(RESULTS.glob("*.csv"))),
            "markdown_results": len(list(RESULTS.glob("*.md"))),
            "test_files": len(list((D.REPO_ROOT / "tests").glob("test_*.py"))),
        },
        "tests": run_pytest() if full else None,
        "dqn_smoke": run_dqn_smoke(output_dir, seed) if full else None,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run pytest and a bounded DQN smoke benchmark")
    parser.add_argument("--episodes", type=int, default=200, help="episodes per fixed IPS baseline")
    parser.add_argument("--seed", type=int, default=D.RANDOM_STATE)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "project_benchmark")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        full=args.full,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"\nWrote {json_path}")
    print(f"Wrote {markdown_path}")
    tests_ok = report["tests"] is None or report["tests"]["passed"]
    return 0 if tests_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
