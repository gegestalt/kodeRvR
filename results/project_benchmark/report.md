# Project Health and Benchmark Report

## Runtime

| dependency | version |
| --- | --- |
| python | 3.14.6 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| torch | 2.13.0 |
| lightgbm | 4.6.0 |

## Datasets

| dataset | status | role | limitation |
| --- | --- | --- | --- |
| NSL-KDD | missing | Controlled baseline for every new method. | None for current NSL-KDD baseline; temporal claims remain invalid. |
| CICIoT2023 dev parquet | missing | Fast modern IoT dev sample for quality checks and pilot modelling. | Do not treat dev-sample results as full official raw-release results. |
| CICIoT2023 raw CSV | blocked_missing_local_files | Primary modern IoT supervised dataset: binary, 8-category, fine-label. | No full raw CSV modelling or quality claim until CSV files are present. |
| TON_IoT | blocked_missing_local_files | Multimodal SOC/EDR-style track: network, IoT telemetry, Windows/Linux traces. | No multimodal fusion, host telemetry, or TON_IoT score is valid yet. |
| CSE-CIC-IDS2018 | blocked_missing_local_files | Enterprise-scale/day-based drift and chronological evaluation. | No enterprise/day-based drift result is valid yet. |
| Common NetFlow schema | blocked_missing_local_files | Cross-dataset generalization using shared NetFlow-style features. | No cross-dataset score is valid without a common feature schema. |

## Feature representations

| representation | features | details |
| --- | --- | --- |
| NSL-KDD raw flow | 41 | 3 categorical + 38 numeric; difficulty excluded |
| Adaptive IPS state | 7 | threat probability, anomaly, attack stage, compromise, criticality, attack rate, response budget |
| CICIoT2023 | 46 | published expected model features; local parquet absent |

## Experiment coverage

| track | status | evidence |
| --- | --- | --- |
| Leakage-safe preprocessing | PROVEN | src/preprocess.py + tests |
| Supervised reference models | PROVEN | results/reference_track.csv |
| RF/LightGBM official split | PROVEN | results/metrics.md |
| Weighted/unweighted MLP | PROVEN | results/stability.md |
| Threshold tuning | PROVEN | results/threshold_ablation.csv |
| Normal-only anomaly detection | PROVEN | results/anomaly_detection.csv |
| Semi-supervised learning | PROVEN | results/semi_supervised.csv |
| Online partial-fit proxy | PARTIAL | not chronological drift |
| CICIoT2023 supervised dev | PARTIAL | random dev split, not full raw CSV |
| Adaptive IPS environment | PROVEN | src/ips + safety tests |
| Masked Double DQN | IMPLEMENTED | src/ips/train_dqn.py |
| True temporal drift | BLOCKED | timestamped local data required |
| Cross-dataset NetFlow transfer | BLOCKED | shared-schema data required |
| Cyber-range IPS validation | PLANNED | contained testbed required |

## Model training and evaluation status

| dataset_task | models | status | fit_selection | evaluation | run |
| --- | --- | --- | --- | --- | --- |
| NSL-KDD binary + 5-class | Dummy, LogReg, RF, ExtraTrees, HistGB, LightGBM | TRAINED / SAVED | official train; class balancing; train-only validation/CV | official KDDTest+; macro-F1 + per-class recall | .venv/bin/python src/reference_track.py |
| NSL-KDD binary + 5-class | MLP unweighted + class-weighted | TRAINED / 5-SEED | official train; internal early-stopping validation | official KDDTest+; macro-F1 and rare-family recall | .venv/bin/python src/train_mlp.py |
| NSL-KDD anomaly | IsolationForest, LOF, KMeans distance | TRAINED / SAVED | normal training rows only; train-normal quantile threshold | official KDDTest+ attack/family recall | .venv/bin/python src/anomaly_detection.py |
| CICIoT2023 dev binary + category | Dummy, LogReg, RF, HistGB, LightGBM | TRAINED / DEV ONLY | stratified 200k train sample; random dev partition | full dev test partition; not raw/temporal evidence | .venv/bin/python src/ciciot2023_baselines.py |
| Adaptive IPS simulated episodes | Allow-only, rule-based, aggressive | RUN LIVE IN THIS REPORT | no fitting; identical seeded scenarios | containment, compromise, disruption, false prevention, return | .venv/bin/python src/project_benchmark.py |
| Adaptive IPS simulated episodes | Masked Double DQN | IMPLEMENTED / SMOKE WITH --full | training seed range; disjoint validation seed range | safety-first checkpoint ordering; simulator evidence only | .venv/bin/python src/project_benchmark.py --full |

## How metrics are calculated

| metric | calculation | use |
| --- | --- | --- |
| accuracy | correct predictions / all predictions | secondary under imbalance |
| per-class recall | TP / (TP + FN) for that class | how many attacks of each family are caught |
| macro-F1 | unweighted mean of each class F1; F1=2PR/(P+R) | primary classifier comparison |
| weighted-F1 | class F1 weighted by class support | overall volume-weighted quality |
| ROC-AUC | area under TPR versus FPR across thresholds | binary ranking; can flatter imbalance |
| PR-AUC | area under precision versus recall | preferred binary ranking under imbalance |
| containment rate | contained attack episodes / attack episodes | IPS prevention outcome |
| compromise rate | compromised attack episodes / attack episodes | lower is better; first checkpoint criterion |
| false prevention | disruptive actions in benign episodes / episodes | availability/safety cost |
| mean return | mean sum of transition rewards per episode | RL objective; never interpreted alone |

## Saved model benchmarks

| track | status | best_macro_f1 | best_method | artifact |
| --- | --- | --- | --- | --- |
| NSL-KDD reference | missing | — | — | results/reference_track.csv |
| CICIoT2023 dev | available | 0.8563 | RandomForest (balanced) | results/ciciot2023_baselines.csv |
| Threshold ablation | available | 0.8230 | HistGB_balanced | results/threshold_ablation.csv |
| Anomaly detection | available | 0.8413 | KMeans_distance | results/anomaly_detection.csv |
| Semi-supervised | available | 0.7542 | labelled_only_logreg | results/semi_supervised.csv |
| Online proxy | available | 0.7642 | SGD_log_loss_balanced | results/online_learning.csv |
| Neural ablation | available | 0.7759 | — | results/neural_ablation.csv |

## Adaptive IPS policy benchmarks

| policy | mean_return | containment_rate | compromise_rate | false_preventions_per_episode | disruptive_actions_per_episode |
| --- | --- | --- | --- | --- | --- |
| allow_only | -15.1200 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| rule_based | 3.4585 | 0.9821 | 0.0179 | 0.0000 | 1.1300 |
| aggressive | -30.8500 | 1.0000 | 0.0000 | 8.8000 | 9.6200 |

## Test suite

Passed: **True**

```text
..s..................sssss......sssss................................... [ 64%]
........................sssss..........                                  [100%]
95 passed, 16 skipped in 2.15s
```

## DQN smoke benchmark

```json
{
  "best_validation": {
    "attack_episodes": 20,
    "benign_episodes": 30,
    "compromise_rate": 0.0,
    "containment_rate": 1.0,
    "disruptive_actions_per_episode": 0.64,
    "episodes": 50,
    "false_preventions_per_episode": 0.0,
    "masked_actions_per_episode": 0.0,
    "mean_return": 3.08,
    "mean_steps": 30.64,
    "return_std": 3.7793650260328127
  },
  "checkpoint": "/Users/hexenmeister/Library/Mobile Documents/com~apple~CloudDocs/Desktop/don't delete update/codes/network-intrusion-detection-ml/results/project_benchmark/dqn_smoke/dqn_best.pt",
  "status": "smoke_only",
  "total_steps": 2113,
  "updates": 2050
}
```

## Comparison rules

1. Compare models directly only when dataset, label task, split, and metric are identical.
2. Do not rank CICIoT2023 random-dev scores against NSL-KDD official-shift scores.
3. Do not treat anomaly detection, supervised classification, and IPS return as one leaderboard.
4. A DQN checkpoint is ordered by lower compromise, higher containment, lower false prevention, then return.
5. Simulator IPS metrics demonstrate algorithm behavior, not live-network prevention effectiveness.

## Interpretation boundary

Saved supervised metrics come from their documented dataset splits. IPS policy metrics currently come from the seeded simulator and are not live-network or cyber-range evidence. The DQN smoke run verifies training mechanics only.
