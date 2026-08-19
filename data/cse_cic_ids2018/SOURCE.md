# CSE-CIC-IDS2018 temporal subset provenance

Official source: <https://www.unb.ca/cic/datasets/ids-2018.html>

Public bucket: `s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/`

Downloaded 2026-08-19. This initial temporal evidence subset contains three
complete, consecutive generated-flow CSV days. Raw files are gitignored.

| Role | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| train | `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | 358223333 | `acff8bc61376ee031d80878ee6099e0b1a87a1bd711d8068298421418c9f8147` |
| validation | `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` | 375945899 | `fa2947a8256d81ee9103ae16139d62d0e17aa23e696ee80d9e76fb51c01c9c4b` |
| final test | `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` | 333723605 | `1a4919faa0c49c7af97230b0c2d076eba23ee6dd81103a3801d51ac316355d8b` |

The detector protocol uses hour-group out-of-fold probabilities for the first
day and a first-day-only fitted detector for both later days. Days are never
randomly mixed. This subset covers brute-force and DoS family shift; it is not a
claim about all ten CSE-CIC-IDS2018 days.
