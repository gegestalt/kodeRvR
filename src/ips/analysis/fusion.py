"""Chronological network/provenance/late-fusion evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,precision_score,recall_score
from sklearn.pipeline import make_pipeline

from ips.analysis.next_phase import _ece


def evaluate_modalities(windows:pd.DataFrame,feature_sets:dict[str,list[str]],*,seed:int=42) -> tuple[pd.DataFrame,dict[str,object]]:
    """Compare modalities identically; fit calibration on validation only."""
    train=windows.split_role.eq("train"); validation=windows.split_role.eq("validation")
    development=windows.split_role.eq("development_test")
    if windows.split_role.eq("locked_final_holdout").sum()==0: raise ValueError("locked holdout role is required")
    rows=[]; fitted={}
    for modality,features in feature_sets.items():
        model=make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(max_iter=120,max_depth=4,min_samples_leaf=2,class_weight="balanced",random_state=seed))
        model.fit(windows.loc[train,features],windows.loc[train,"attack_present"].astype(int)); fitted[modality]=model
        val_score=model.predict_proba(windows.loc[validation,features])[:,1]
        calibrator=LogisticRegression(random_state=seed).fit(val_score.reshape(-1,1),windows.loc[validation,"attack_present"].astype(int))
        for role,selected in (("validation",validation),("development_test",development)):
            y=windows.loc[selected,"attack_present"].astype(int).to_numpy(); raw=model.predict_proba(windows.loc[selected,features])[:,1]
            candidates={"raw":raw}
            if role=="development_test": candidates["platt_validation_fit"]=calibrator.predict_proba(raw.reshape(-1,1))[:,1]
            for calibration,score in candidates.items():
                prediction=score>=.5
                rows.append({"modality":modality,"calibration":calibration,"evaluation_role":role,"rows":len(y),"attacks":int(y.sum()),
                             "pr_auc":average_precision_score(y,score),"precision":precision_score(y,prediction,zero_division=0),
                             "recall":recall_score(y,prediction,zero_division=0),"brier":brier_score_loss(y,score),"ece":_ece(y,score)})
    return pd.DataFrame(rows),fitted
