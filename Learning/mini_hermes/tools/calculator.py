"""
Calculator Tool — A simple math expression evaluator.
"""

import ast
import operator
import logging

from mini_hermes.tool_registry import registry

logger = logging.getLogger(__name__)

# Safe operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval(expr: str) -> float:
    """Safely evaluate a math expression using AST."""
    node = ast.parse(expr, mode="eval")
    return _eval_node(node.body)


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            return operand
    elif isinstance(node, ast.Num):
        return node.n
    raise ValueError(f"Unsupported expression: {expr}")


def calculator_handler(args: dict) -> dict:
    """Handle calculator tool calls."""
    expr = args.get("expression", "")
    try:
        result = safe_eval(expr)
        return {"success": True, "expression": expr, "result": result}
    except Exception as e:
        return {"success": False, "expression": expr, "error": str(e)}


# Register the tool
registry.register(
    name="calculator",
    schema={
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports +, -, *, /, **.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    handler=calculator_handler,
    toolset="utility",
)
