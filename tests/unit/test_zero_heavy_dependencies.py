"""Automated Zero-Heavy-Dependencies Audit Test.

Validates that Chokkhu codebase strictly adheres to mathematical sovereignty:
Zero banned external heavy dependencies (torch, tensorflow, sklearn, transformers, etc.).
"""

import ast
import os

BANNED_MODULES = {
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "tf",
    "keras",
    "sklearn",
    "scikit_learn",
    "transformers",
    "diffusers",
    "xgboost",
    "lightgbm",
    "catboost",
    "imbalanced_learn",
    "imblearn",
    "shap",
    "lime",
    "statsmodels",
    "gym",
    "gymnasium",
    "faiss",
    "gensim",
    "spacy",
    "networkx",
    "lifelines",
    "survivalsvm",
    "fairlearn",
    "deap",
}


def test_zero_banned_dependencies_in_source():
    """Scan all Python source files in src/chokkhu to verify zero banned imports."""
    src_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "chokkhu")
    )
    assert os.path.exists(src_dir), f"Source directory not found: {src_dir}"

    violations = []

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                try:
                    tree = ast.parse(content, filename=file_path)
                except SyntaxError:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            root_pkg = alias.name.split(".")[0]
                            if root_pkg in BANNED_MODULES:
                                violations.append(
                                    f"{rel_path}:{node.lineno} imports '{alias.name}'"
                                )
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.level == 0:
                            root_pkg = node.module.split(".")[0]
                            if root_pkg in BANNED_MODULES:
                                violations.append(
                                    f"{rel_path}:{node.lineno} imports from '{node.module}'"
                                )

    assert (
        len(violations) == 0
    ), f"Found {len(violations)} banned imports:\n" + "\n".join(violations)
