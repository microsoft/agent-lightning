# Copyright (c) Microsoft. All rights reserved.

"""
Calculator Tool for Math Agent

A simple but robust calculator tool that safely evaluates mathematical expressions.
Demonstrates best practices for tool implementation in Agent-Lightning.
"""

import ast
import operator
from typing import Union


class SafeCalculator:
    """
    A safe calculator that evaluates mathematical expressions without using eval().
    
    Supports:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Parentheses for order of operations
    - Floating point and integer numbers
    - Unary operations: -x, +x
    
    Does not support:
    - Variable assignments
    - Function calls (except built-in math operations)
    - Importing modules
    - File operations or other dangerous code
    """
    
    # Allowed operations
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def eval_node(self, node: ast.AST) -> Union[int, float]:
        """
        Recursively evaluate an AST node.
        
        Args:
            node: An AST node representing part of the expression
            
        Returns:
            The numerical result of evaluating the node
            
        Raises:
            ValueError: If the expression contains unsupported operations
        """
        if isinstance(node, ast.Constant):
            # Literal number
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant type: {type(node.value)}")
            return node.value
        
        elif isinstance(node, ast.BinOp):
            # Binary operation (e.g., 5 + 3)
            left = self.eval_node(node.left)
            right = self.eval_node(node.right)
            op_type = type(node.op)
            
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            
            return self.OPERATORS[op_type](left, right)
        
        elif isinstance(node, ast.UnaryOp):
            # Unary operation (e.g., -5)
            operand = self.eval_node(node.operand)
            op_type = type(node.op)
            
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            
            return self.OPERATORS[op_type](operand)
        
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")
    
    def calculate(self, expression: str) -> Union[int, float]:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: A string containing a mathematical expression
            
        Returns:
            The numerical result
            
        Raises:
            ValueError: If the expression is invalid or contains unsupported operations
            SyntaxError: If the expression has invalid syntax
        """
        # Parse the expression into an AST
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise SyntaxError(f"Invalid expression syntax: {str(e)}")
        
        # Evaluate the AST
        result = self.eval_node(tree.body)
        
        # Round to reasonable precision to avoid floating point artifacts
        if isinstance(result, float):
            # Round to 10 decimal places
            result = round(result, 10)
            # Convert to int if it's a whole number
            if result.is_integer():
                result = int(result)
        
        return result


# Global calculator instance
_calculator = SafeCalculator()


def calculator_tool(expression: str) -> str:
    """
    Calculator tool for the math agent.
    
    Evaluates mathematical expressions safely without executing arbitrary code.
    
    Args:
        expression: A mathematical expression as a string
                   Examples: "5 + 3", "24 * 7 + 15", "(10 + 5) * 2"
    
    Returns:
        String representation of the result, or an error message if evaluation fails
    
    Examples:
        >>> calculator_tool("5 + 3")
        '8'
        >>> calculator_tool("24 * 7 + 15")
        '183'
        >>> calculator_tool("(10 + 5) * 2")
        '30'
        >>> calculator_tool("10 / 3")
        '3.3333333333'
    """
    try:
        result = _calculator.calculate(expression)
        return str(result)
    except (ValueError, SyntaxError) as e:
        return f"Error: {str(e)}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: Unexpected error - {str(e)}"


# Tool definition for OpenAI function calling format
calculator_tool_definition = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluates mathematical expressions. Supports basic arithmetic "
            "operations (+, -, *, /, //, %, **) and parentheses. "
            "Use this for any calculation in the problem."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical expression to evaluate. "
                        "Example: '5 + 3 * 2' or '(10 + 5) / 3'"
                    )
                }
            },
            "required": ["expression"]
        }
    }
}


if __name__ == "__main__":
    """
    Test the calculator tool with various expressions.
    Run this file directly to verify the calculator works correctly.
    """
    print("Testing Calculator Tool")
    print("=" * 60)
    
    test_cases = [
        "5 + 3",
        "24 * 7",
        "100 / 4",
        "2 ** 10",
        "(10 + 5) * 2",
        "48 / 2 * 9 + 9 - 20",
        "15 % 4",
        "10 // 3",
        "-5 + 10",
        "invalid expression",  # Should produce error
        "1 / 0",  # Should produce division by zero error
    ]
    
    for expression in test_cases:
        result = calculator_tool(expression)
        print(f"{expression:30s} = {result}")
    
    print("=" * 60)
    print("Calculator tool test complete!")