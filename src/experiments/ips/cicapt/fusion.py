"""Build and evaluate CICAPT network/provenance late fusion and tactic beliefs."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ips.adapters.cicapt_fusion import build_multimodal_windows
from ips.analysis.belief_state import TacticBeliefEstimator
from ips.analysis.fusion import evaluate_modalities
from ips.workspace import ProjectPaths

TACTICS=("initial_access","command_and_control","persistence","credential_access","discovery","lateral_movement","collection","exfiltration","defence_evasion","cleanup")


def run() -> dict[str,object]:
    project=ProjectPaths.discover(); output=project.results/"cicapt_iiot2024"; artifacts=project.cicapt_primary_artifacts()
    network=pd.read_parquet(output/"phase2_chronological_events.parquet")
    provenance=pd.read_csv(artifacts["phase2_provenance"],low_memory=False)
    event_manifest=json.loads((output/"event_manifest.json").read_text())
    windows,manifest=build_multimodal_windows(network,provenance,event_manifest["feature_columns"],window_seconds=60)
    # Provenance-only windows inherit chronological role directly from time.
    day=pd.to_datetime(windows.timestamp,unit="s",utc=True).dt.strftime("%Y-%m-%d")
    windows["split_role"]=windows.split_role.fillna(day.map(event_manifest["day_roles"]))
    feature_sets={"network_only":manifest["network_features"],"provenance_only":manifest["provenance_features"],"late_fusion":manifest["feature_columns"]}
    metrics,_=evaluate_modalities(windows,feature_sets)
    metrics.to_csv(output/"fusion_modality_metrics.csv",index=False)

    fit=windows.split_role.isin(["train","validation"]); development=windows.split_role.eq("development_test")
    estimator=TacticBeliefEstimator(tactics=TACTICS,seed=42).fit(windows.loc[fit,manifest["feature_columns"]],windows.loc[fit,"attack_tactic"])
    beliefs=[]; previous=None
    for index,row in windows.loc[development].sort_values("timestamp").iterrows():
        belief=estimator.predict_one(windows.loc[[index],manifest["feature_columns"]],previous=previous)
        previous=belief.probabilities
        beliefs.append({"timestamp":row.timestamp,"true_tactic":row.attack_tactic,"normal_probability":belief.normal_probability,
                        "uncertainty":belief.uncertainty,**{f"p_{k}":v for k,v in belief.probabilities.items()}})
    belief_frame=pd.DataFrame(beliefs)
    probability_columns=[f"p_{name}" for name in TACTICS]
    belief_frame["predicted_tactic"]=np.where(belief_frame[probability_columns].max(axis=1)>belief_frame.normal_probability,
        belief_frame[probability_columns].idxmax(axis=1).str.removeprefix("p_"),"normal")
    belief_frame.to_parquet(output/"development_tactic_beliefs.parquet",index=False)
    attacked=belief_frame.true_tactic.ne("normal")
    belief_metrics={"development_windows":len(belief_frame),"attack_windows":int(attacked.sum()),
                    "top1_accuracy":accuracy_score(belief_frame.true_tactic,belief_frame.predicted_tactic),
                    "attack_macro_f1":f1_score(belief_frame.loc[attacked,"true_tactic"],belief_frame.loc[attacked,"predicted_tactic"],average="macro",zero_division=0),
                    "mean_uncertainty":float(belief_frame.uncertainty.mean()),"locked_final_rows_scored":0,
                    "fit_roles":["train","validation"],"evaluation_role":"development_test"}
    (output/"tactic_belief_metrics.json").write_text(json.dumps(belief_metrics,indent=2)+"\n",encoding="utf-8")
    manifest.update({"feature_sets":feature_sets,"locked_final_rows_scored":0})
    (output/"fusion_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return {"modality_rows":len(metrics),**belief_metrics}


if __name__=="__main__": print(json.dumps(run(),indent=2))
