# Dataset-backed IPS event generation

The adaptive IPS notebook reads `data/ips_events/events.parquet`. Generate that
file from timestamped CICIoT2023, CSE-CIC-IDS2018, or TON_IoT CSV/Parquet data
with `src/ips/real_data_adapter.py`.

The adapter requires explicit timestamp, label, and campaign/host/session group
columns. Group columns are never guessed: a wrong group definition can leak the
same attacker, host, or campaign into training and final test.

Example `data/ips_events/source_config.json`:

```json
{
  "source": "data/cse_cic_ids2018/Wednesday.csv",
  "adapter": {
    "timestamp_col": "Timestamp",
    "label_col": "Label",
    "group_cols": ["Source IP", "Destination IP"],
    "episode_col": null,
    "critical_col": null,
    "window_seconds": 300,
    "folds": 5,
    "max_rows": 200000,
    "seed": 42
  }
}
```

Or run directly:

```bash
.venv/bin/python -m ips.real_data_adapter \
  data/cse_cic_ids2018/Wednesday.csv \
  --timestamp-col Timestamp \
  --label-col Label \
  --group-cols "Source IP,Destination IP" \
  --max-rows 200000
```

For every GroupKFold holdout, the adapter fits a HistGradientBoosting classifier
on the other groups and writes out-of-fold attack probabilities. It also fits an
Isolation Forest on benign rows from the other groups and writes anomaly scores.
Labels and configured identifiers are excluded from detector features.

The resulting schema is:

```text
episode_id, group_id, timestamp, threat_probability, anomaly_score,
attack_present, attack_stage, critical_service, attack_family
```

`attack_stage` is an episode-position proxy unless the source supplies a
validated attack-stage column. Action outcomes remain counterfactual until a
contained cyber range records real interventions.
