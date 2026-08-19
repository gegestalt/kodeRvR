"""Static audit for untrusted dataset-supplied Python reference code."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


def audit_python_sources(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        imports=[]; classes=[]; functions=[]; risks=set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
                if any(alias.name == "*" for alias in node.names): risks.add("wildcard_import")
            elif isinstance(node, ast.ClassDef): classes.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): functions.append(node.name)
            elif isinstance(node, ast.ExceptHandler) and node.type is None: risks.add("broad_except")
            elif isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    name = f"{node.func.value.id}.{node.func.attr}"
                if name in {"os.system", "subprocess.run", "subprocess.call", "subprocess.Popen"}: risks.add("shell_execution")
                if name in {"os.remove", "os.unlink", "shutil.rmtree"}: risks.add("file_deletion")
        rows.append({"file":path.name,"imports":sorted(set(imports)),"classes":sorted(classes),
                     "functions":sorted(functions),"risks":sorted(risks),"safe_to_execute":not risks,
                     "reuse_policy":"schema/formula reference only" if risks else "reviewed functions may be independently ported"})
    return pd.DataFrame(rows)
