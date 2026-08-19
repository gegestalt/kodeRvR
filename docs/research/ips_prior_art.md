# Adaptive IPS prior-art and data-source ledger

Reviewed 2026-08-19. External code was not copied. The project inherits methods
only after adapting them to its safety mask, leakage protocol, real-data evidence
labels, and reproducible multi-seed evaluation.

| Source | Type | Relevant contribution | Local decision |
| --- | --- | --- | --- |
| [Akamai AkaNAT](https://www.akamai.com/blog/security/how-akamai-uses-machine-learning-to-detect-shared-ips) | production engineering report | Aggregated IP fingerprints, shared-IP classification, repeated-positive detection and expiry | Adopted as a tested shared-IP action guardrail and identity-scoped hysteresis. No claim is made that our current datasets contain ground-truth NAT labels. |
| [Jayalaxmi et al., IoT IDS/IPS survey](https://doi.org/10.1109/ACCESS.2022.3220622) | IEEE Access survey | Taxonomy of ML/DL IoT detection/prevention methods, datasets, feasibility and real-time concerns | Used to broaden the dataset and evaluation checklist; not treated as empirical policy evidence. |
| [Schaul et al., Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) | ICLR paper | Proportional replay and importance weights | Adopted as masked PER-Double-DQN; compared with uniform replay under identical splits/seeds. |
| [Hammar & Stadler, Learning Intrusion Prevention Policies through Optimal Stopping](https://arxiv.org/abs/2106.07160) | CNSM paper | Intrusion prevention as a stopping problem | Adopted as a transparent threshold/optimal-stopping baseline, not claimed as a reproduction. |
| [Huang & Ontañón, Invalid Action Masking](https://arxiv.org/abs/2006.14171) | peer-reviewed paper + code | Theoretical and empirical basis for policy action masking | Existing mask retained; notebook adds proposed-invalid-action audit. |
| [Stable-Baselines3 Contrib MaskablePPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/ppo_mask/ppo_mask.py) | maintained implementation | Rollout-buffer and masked-policy reference | Used as a design cross-check; local lightweight PPO remains independently implemented and is not labelled SB3. |
| [Jiang & Li, Doubly Robust OPE](https://proceedings.mlr.press/v48/jiang16.html) | ICML paper | Safe off-policy evaluation foundations | Weighted-IS diagnostic adopted; doubly robust estimation remains blocked until genuine behavior propensities and an independently fitted outcome model exist. |
| [CyberBattleSim](https://github.com/microsoft/CyberBattleSim) | Microsoft research environment | Network topology, lateral movement, defender SLA and delayed actions | Recommended future environment bridge; not mixed with flow evidence in current results. |
| [CSLE](https://github.com/Kim-Hammar/csle) | cyber-range/simulation platform | Simulation–emulation separation, stopping games, trace collection | Architecture adopted conceptually; full platform integration deferred because it is a separate distributed stack. |
| [gym-idsgame](https://github.com/Kim-Hammar/gym-idsgame) | Markov-game environment | Self-play and attacker/defender evaluation | Tournament/adaptive-attacker roadmap adopted; current flow notebook does not claim self-play evidence. |
| [CMARL-ACD](https://github.com/cyb3rlab/CMARL-ACD) | multi-agent cyber-defense implementation | Worst-case N×N checkpoint cross-evaluation | Cross-policy evaluation idea adopted; MARL itself deferred. |
| [UNB CIC dataset catalog](https://www.unb.ca/cic/datasets/index.html) | official dataset source | CICIoT2023, CICIDS2017, CSE-CIC-IDS2018 provenance | Preferred source for scientific claims. |
| [Hugging Face CIC-IoT-2023-full](https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-full) | dataset mirror/card | Large Parquet access and schema inspection | Candidate development mirror; random split is unsuitable for temporal claims. Verify hashes/provenance before use. |
| [Kaggle CIC-IDS Collection](https://www.kaggle.com/datasets/dhoogla/cicidscollection) | cleaned derivative collection | Harmonized CIC labels and removal of known flawed features | Useful sensitivity source only; results must remain separate from official raw-data experiments and respect CC BY-NC-SA 4.0. |
| [Kaggle cleaned CICIDS2017](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed) | derivative dataset | Cleaning and feature-selection ideas | Not adopted as primary evidence because rare Heartbleed/Infiltration classes were removed. |
| [IoT-23](https://www.stratosphereips.org/datasets-iot23) | official dataset source | Scenario-separated benign and malware traffic with detailed behavior labels | High-priority external validation source; preserve scenario identity and split by capture, never random rows. |
| [Bot-IoT](https://research.unsw.edu.au/projects/bot-iot-dataset) | official dataset source | IoT botnet traffic and attack categories | Candidate stress-test source; use capture/time groups and retain severe imbalance. |
| [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets) | official dataset source | Heterogeneous network, IoT/IIoT telemetry and operating-system data | High-priority domain-shift source; evaluate each modality separately before any fusion. |

## Scientific guardrails

- GitHub popularity, Kaggle votes, and benchmark accuracy are not evidence of
  leakage-free generalization.
- Official time/group structure takes precedence over random pre-splits.
- Detector classification metrics and IPS intervention metrics are reported
  separately.
- External cleaned datasets are never silently pooled with official sources.
- An implementation is called a reproduction only when versions, preprocessing,
  splits, seeds, and reported metrics match the source protocol.
- Offline policy evaluation is not reported as valid without logged behavior
  propensities, overlap diagnostics, and adequate effective sample size.
