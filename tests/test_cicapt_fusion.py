import numpy as np
import pandas as pd

from ips.adapters.cicapt_fusion import build_multimodal_windows
from ips.analysis.belief_state import TacticBeliefEstimator


def test_multimodal_windows_keep_hidden_truth_out_of_features():
    network = pd.DataFrame({"timestamp":[1,2,301],"x":[1.,2.,3.],"attack_present":[False,True,False],
                            "attack_tactic":["0","collection","0"],"split_role":["train","train","validation"]})
    provenance = pd.DataFrame({"type":["Process","Used"],"time":[1.,302.],"seen time":[np.nan,np.nan],
                               "start time":[np.nan,np.nan],"pid":[1,1],"exe":["a","a"],"operation":["x","y"],
                               "label":[0,1],"subLabel":["0","discovery"]})
    windows, manifest = build_multimodal_windows(network, provenance, ["x"], window_seconds=300)
    assert {"net_x_mean","prov_type_process","prov_type_used"} <= set(windows)
    assert not set(manifest["feature_columns"]) & {"attack_present","attack_tactic","split_role"}
    assert windows.attack_present.any()


def test_tactic_belief_has_fixed_classes_uncertainty_and_history():
    X = pd.DataFrame({"a":[0,0.1,1,1.1,2,2.1],"b":[0,0,1,1,2,2]})
    y = pd.Series(["normal","normal","collection","collection","discovery","discovery"])
    estimator = TacticBeliefEstimator(tactics=("collection","discovery","exfiltration"), seed=42).fit(X,y)
    belief = estimator.predict_one(X.iloc[[2]], previous={"collection":.2,"discovery":.1,"exfiltration":0.0})
    assert set(belief.probabilities) == {"collection","discovery","exfiltration"}
    assert 0 <= belief.uncertainty <= 1
    assert sum(belief.probabilities.values()) <= 1
