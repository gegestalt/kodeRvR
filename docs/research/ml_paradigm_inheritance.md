# ML paradigm inheritance for the adaptive IPS lab

## Source

Deepak Tomar, Kismat Chhillar, and Saurabh Shrivastava, "AI Enhanced
Intrusion Detection and Prevention Systems (IDS/IPS)," *IRE Journals*, volume
9, issue 9, March 2026, paper 1714847.

Source: <https://www.irejournals.com/formatedpaper/1714847.pdf>

## What this project inherits

The paper is used as a review scaffold for three detector paradigms:

| Paradigm | Role in this repository | Implemented evidence |
| --- | --- | --- |
| Supervised | Known-class chronological discrimination | Logistic Regression, Random Forest and Histogram Gradient Boosting on CICAPT flow rows |
| Normal-only anomaly | Novel-deviation screening without attack labels | Isolation Forest in CICAPT; Isolation Forest, LOF and K-means distance in the NSL-KDD anomaly track |
| Semi-supervised | Label-budget experiment | Labelled-only versus self-training Logistic Regression on NSL-KDD |

The notebook also maps the paper's operational concerns to measured project
evidence: class imbalance, concept drift, false-alert burden, inference latency,
streaming constraints and analyst-facing explanations.

## What this project does not inherit as a claim

The review's broad statements about deep-learning superiority are not treated
as experimental evidence. Architecture comparisons require identical features,
splits, calibration, seeds, resources and evaluation units. The paper's cited
bibliography also includes several QoS/networking references that do not
directly support IDS benchmark claims, so the project relies on its own saved
experiments and protocol-specific primary sources for numerical conclusions.

Federated learning, CNN/LSTM detectors and streaming retraining remain future
research candidates. They are added only after a dataset role, leakage-safe
protocol, operational baseline and falsifiable hypothesis are defined.
