"""Tools used by the real GSM8K PolySeg agentic experiment."""

from __future__ import annotations

import ast
import hashlib
import os
import operator as op

from verl.tools.function_tool import function_tool


@function_tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: Python-style arithmetic expression, for example ``(3+4)*5``.

    The AST whitelist makes the tool deterministic and prevents arbitrary
    Python execution while preserving the arithmetic operations used by
    GSM8K solutions.
    """

    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))
        raise ValueError(f"unsupported expression node: {ast.dump(node)}")

    try:
        value = evaluate(ast.parse(expression, mode="eval").body)
        # Robustness ablations can inject a reproducible arithmetic corruption
        # without changing the default tool contract.  Hashing the expression
        # keeps repeated calls deterministic across Ray workers and processes.
        probability = float(os.getenv("POLYSEG_CALCULATOR_NOISE_P", "0"))
        if probability < 0.0 or probability > 1.0:
            raise ValueError("POLYSEG_CALCULATOR_NOISE_P must be in [0, 1]")
        seed = os.getenv("POLYSEG_CALCULATOR_NOISE_SEED", "0")
        digest = hashlib.sha256(f"{seed}:{expression}".encode("utf-8")).digest()
        draw = int.from_bytes(digest[:8], "big") / float(2**64)
        if probability and draw < probability:
            if isinstance(value, (int, float)):
                value = value + (1 if value >= 0 else -1)
            return f"NOISY_RESULT: {value}"
        return str(value)
    except Exception as exc:
        return f"ERROR: {exc}"
