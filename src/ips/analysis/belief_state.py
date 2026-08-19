"""Calibrated tactic-belief state for partially observable APT response."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class TacticBelief:
    probabilities: dict[str,float]
    normal_probability: float
    uncertainty: float


class TacticBeliefEstimator:
    def __init__(self, *, tactics: tuple[str,...], history_weight: float=.25, seed: int=42) -> None:
        if not tactics or not 0<=history_weight<1: raise ValueError("invalid tactics/history_weight")
        self.tactics=tactics; self.history_weight=history_weight; self.seed=seed
        self.model=make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(max_iter=100,class_weight="balanced",random_state=seed))

    def fit(self,X:pd.DataFrame,y:pd.Series) -> "TacticBeliefEstimator":
        self.model.fit(X,y.astype(str)); return self

    def predict_one(self,X:pd.DataFrame,*,previous:dict[str,float]|None=None) -> TacticBelief:
        raw=self.model.predict_proba(X)[0]; classes=list(self.model.classes_)
        values={tactic:(float(raw[classes.index(tactic)]) if tactic in classes else 0.0) for tactic in self.tactics}
        if previous:
            values={name:(1-self.history_weight)*value+self.history_weight*float(previous.get(name,0)) for name,value in values.items()}
        normal=float(raw[classes.index("normal")]) if "normal" in classes else max(0.0,1-sum(values.values()))
        known_mass=min(1.0,normal+sum(values.values())); uncertainty=max(0.0,1-max([normal,*values.values()]))
        if known_mass>1: values={key:value/known_mass for key,value in values.items()}; normal/=known_mass
        return TacticBelief(values,normal,uncertainty)
