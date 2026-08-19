"""Audit CICAPT-supplied scripts and compare their schema with downloaded CSV."""

from __future__ import annotations

import json
import pandas as pd

from ips.analysis.source_audit import audit_python_sources
from ips.workspace import ProjectPaths


def run() -> dict[str, object]:
    project=ProjectPaths.discover(); source=project.cicapt_source()
    if source is None: raise FileNotFoundError("CICAPT download not found")
    scripts=sorted((source/"reference_code").glob("*.py")); audit=audit_python_sources(scripts)
    header=set(pd.read_csv(project.cicapt_primary_artifacts()["phase2_network"],nrows=0).columns.str.strip())
    expected={"ts","flow_duration","Header_Length","Rate","Srate","Drate","Tot sum","IAT","Magnitue","Radius","Variance","Weight"}
    schema={"expected_reference_features":sorted(expected),"present_in_phase2":sorted(expected & header),
            "missing_from_phase2":sorted(expected-header),"phase2_columns":len(header)}
    output=project.results/"cicapt_iiot2024"; output.mkdir(parents=True,exist_ok=True)
    serial=audit.copy()
    for column in ("imports","classes","functions","risks"): serial[column]=serial[column].map(json.dumps)
    serial.to_csv(output/"downloaded_source_audit.csv",index=False)
    (output/"downloaded_feature_schema_audit.json").write_text(json.dumps(schema,indent=2)+"\n",encoding="utf-8")
    return {"scripts":len(audit),"unsafe_to_execute":int((~audit.safe_to_execute).sum()),
            "schema_features_present":len(schema["present_in_phase2"]),"schema_features_missing":schema["missing_from_phase2"]}


if __name__ == "__main__": print(json.dumps(run(),indent=2))
