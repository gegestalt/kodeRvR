"""Language-aware lexical, structural, and repository-behaviour features."""

from __future__ import annotations

import ast
from collections import Counter
import io
import keyword
import math
import re
import tokenize

import numpy as np

from code_provenance.schema import CodeSample


FEATURE_NAMES = (
    "lines", "nonblank_lines", "comment_fraction", "mean_line_length",
    "line_length_std", "identifier_count", "identifier_diversity",
    "mean_identifier_length", "string_literal_fraction", "numeric_literal_fraction",
    "max_indent", "mean_indent", "function_count", "class_count", "branch_count",
    "exception_count", "import_count", "docstring_count", "mean_function_lines",
    "message_length", "message_line_count", "files_changed", "additions", "deletions",
    "change_balance", "code_token_entropy",
)


_COMMENT_PATTERNS = {
    "python": re.compile(r"#.*$", re.MULTILINE),
    "javascript": re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL),
    "typescript": re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL),
    "java": re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL),
    "c": re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL),
    "cpp": re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL),
}


def _entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = np.asarray(list(Counter(tokens).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _python_structure(code: str) -> dict[str, float]:
    output = {name: 0.0 for name in (
        "function_count", "class_count", "branch_count", "exception_count",
        "import_count", "docstring_count", "mean_function_lines",
    )}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return output
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    output["function_count"] = float(len(functions))
    output["class_count"] = float(sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree)))
    output["branch_count"] = float(sum(isinstance(node, (ast.If, ast.For, ast.While, ast.Match)) for node in ast.walk(tree)))
    output["exception_count"] = float(sum(isinstance(node, (ast.Try, ast.Raise, ast.ExceptHandler)) for node in ast.walk(tree)))
    output["import_count"] = float(sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)))
    output["docstring_count"] = float(sum(
        ast.get_docstring(node, clean=False) is not None
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ))
    lengths = [max(1, getattr(node, "end_lineno", node.lineno) - node.lineno + 1) for node in functions]
    output["mean_function_lines"] = float(np.mean(lengths)) if lengths else 0.0
    return output


def extract_features(sample: CodeSample) -> dict[str, float]:
    """Extract interpretable features without using repository or author IDs."""
    lines = sample.code.splitlines()
    nonblank = [line for line in lines if line.strip()]
    lengths = [len(line) for line in nonblank]
    indents = [len(line) - len(line.lstrip()) for line in nonblank]
    comment_pattern = _COMMENT_PATTERNS.get(sample.language.lower(), re.compile(r"#.*$", re.MULTILINE))
    comments = comment_pattern.findall(sample.code)
    tokens = re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s\w]", sample.code)
    identifiers = [token for token in tokens if re.fullmatch(r"[A-Za-z_]\w*", token) and not keyword.iskeyword(token)]
    strings = re.findall(r"(['\"]).*?\1", sample.code, re.DOTALL)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", sample.code)
    features = {
        "lines": float(len(lines)),
        "nonblank_lines": float(len(nonblank)),
        "comment_fraction": sum(len(item) for item in comments) / max(len(sample.code), 1),
        "mean_line_length": float(np.mean(lengths)) if lengths else 0.0,
        "line_length_std": float(np.std(lengths)) if lengths else 0.0,
        "identifier_count": float(len(identifiers)),
        "identifier_diversity": len(set(identifiers)) / max(len(identifiers), 1),
        "mean_identifier_length": float(np.mean([len(item) for item in identifiers])) if identifiers else 0.0,
        "string_literal_fraction": len(strings) / max(len(tokens), 1),
        "numeric_literal_fraction": len(numbers) / max(len(tokens), 1),
        "max_indent": float(max(indents, default=0)),
        "mean_indent": float(np.mean(indents)) if indents else 0.0,
        "message_length": float(len(sample.commit_message)),
        "message_line_count": float(len(sample.commit_message.splitlines())),
        "files_changed": float(sample.files_changed),
        "additions": float(sample.additions),
        "deletions": float(sample.deletions),
        "change_balance": (sample.additions - sample.deletions) / max(sample.additions + sample.deletions, 1),
        "code_token_entropy": _entropy(tokens),
    }
    structure = _python_structure(sample.code) if sample.language.lower() == "python" else {
        "function_count": float(len(re.findall(r"\bfunction\b|\bdef\b", sample.code))),
        "class_count": float(len(re.findall(r"\bclass\b", sample.code))),
        "branch_count": float(len(re.findall(r"\b(if|for|while|switch)\b", sample.code))),
        "exception_count": float(len(re.findall(r"\b(try|catch|throw|raise|except)\b", sample.code))),
        "import_count": float(len(re.findall(r"\b(import|require|include)\b", sample.code))),
        "docstring_count": 0.0,
        "mean_function_lines": 0.0,
    }
    features.update(structure)
    return {name: float(features[name]) for name in FEATURE_NAMES}
