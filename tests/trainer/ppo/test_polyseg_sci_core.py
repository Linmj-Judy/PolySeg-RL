"""Regression checks for the dependency-free PolySeg-Sci core boundary."""

from __future__ import annotations

import ast
from pathlib import Path


def test_core_has_no_verl_imports() -> None:
    source = Path(__file__).resolve().parents[3] / "polyseg_sci" / "core.py"
    tree = ast.parse(source.read_text())
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    assert not [name for name in imported_modules if name == "verl" or name.startswith("verl.")]


def test_verl_path_is_a_thin_registry_shim() -> None:
    from verl.trainer.ppo import polyseg_sci, polyseg_sci_core

    assert polyseg_sci.run_science_round is polyseg_sci_core.run_science_round
    assert polyseg_sci.PolynomialVectorPredictor is polyseg_sci_core.PolynomialVectorPredictor
    assert polyseg_sci.RESTORE_EVIDENCE_KINDS == polyseg_sci_core.RESTORE_EVIDENCE_KINDS

