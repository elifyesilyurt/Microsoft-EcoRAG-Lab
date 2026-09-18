"""
dynamic_math_engine.py — Generalized Program-of-Thoughts (PoT) / Dynamic PAL Engine
-----------------------------------------------------------------------------------
Çok terimli ondalıklı aritmetik, oran, yüzde puanı ve karşılaştırmalı hesaplamaları
LLM'in zihinsel tahminine bırakmadan, izole ve güvenli bir Python çalışma zamanında
(AST-safe execution) %100 kesinlikle çözen dinamik matematik motoru.
"""

import ast
import operator
import re
from typing import Any, Dict, Optional

# İzin verilen güvenli matematiksel operatörler
_ALLOWED_OPERATORS = {
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

_SAFE_FUNCTIONS = {
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "len": len,
    "float": float,
    "int": int,
}


class SafeMathEvaluator:
    """Güvenli AST tabanlı Python matematiksel ifade değerlendiricisi."""

    @classmethod
    def eval_expr(cls, expr: str, context: Optional[Dict[str, Any]] = None) -> Any:
        context = context or {}
        tree = ast.parse(expr, mode="eval")
        return cls._eval_node(tree.body, context)

    @staticmethod
    def _tuple_to_number(val: Any) -> Any:
        if isinstance(val, tuple) and all(isinstance(x, (int, float)) for x in val):
            s = "".join(f"{int(x):03d}" if idx > 0 else str(int(x)) for idx, x in enumerate(val))
            try:
                return float(s) if '.' in s else int(s)
            except ValueError:
                return val
        return val

    @classmethod
    def _eval_node(cls, node: ast.AST, context: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            elif node.id in _SAFE_FUNCTIONS:
                return _SAFE_FUNCTIONS[node.id]
            raise ValueError(f"Tanımsız değişken veya fonksiyon: '{node.id}'")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError(f"İzin verilmeyen operatör: {op_type}")
            left = cls._tuple_to_number(cls._eval_node(node.left, context))
            right = cls._tuple_to_number(cls._eval_node(node.right, context))
            return _ALLOWED_OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError(f"İzin verilmeyen unar operatör: {op_type}")
            operand = cls._tuple_to_number(cls._eval_node(node.operand, context))
            return _ALLOWED_OPERATORS[op_type](operand)
        elif isinstance(node, ast.List):
            return [cls._tuple_to_number(cls._eval_node(elem, context)) for elem in node.elts]
        elif isinstance(node, ast.Tuple):
            raw_tup = tuple(cls._eval_node(elem, context) for elem in node.elts)
            return cls._tuple_to_number(raw_tup)
        elif isinstance(node, ast.Call):
            func = cls._eval_node(node.func, context)
            if func not in _SAFE_FUNCTIONS.values():
                raise ValueError(f"Güvenli olmayan fonksiyon çağrısı: {node.func}")
            args = [cls._eval_node(arg, context) for arg in node.args]
            return func(*args)
        else:
            raise ValueError(f"Desteklenmeyen sözdizimi: {type(node)}")


class DynamicMathExecutor:
    """
    Modelden gelen Program-of-Thoughts kod satırlarını veya
    sayısal formülleri çalıştıran güvenli motor.
    """

    @staticmethod
    def execute_code_lines(code_str: str) -> Dict[str, Any]:
        """
        Güvenli değişken atamalarını ve matematiksel ifadeleri sırayla çalıştırır.
        Örnek girdi:
            cat1 = 36.23
            cat2 = 38.24
            top2 = cat1 + cat2
            others = [3.39, 1.99, 0.05, 0.81, 1.22, 0.45, 14.05, 0.03, 0.05]
            top_others = sum(others)
            diff = top2 - top_others
        """
        # Kod bloğu işaretlerini temizle
        clean_code = re.sub(r'```(?:python)?|```', '', code_str).strip()
        env: Dict[str, Any] = {}
        execution_trace = []

        lines = [line.strip() for line in clean_code.split('\n') if line.strip() and not line.strip().startswith('#')]

        for line in lines:
            if '=' in line:
                var_name, expr = line.split('=', 1)
                var_name = var_name.strip()
                expr = expr.strip()
                # Değişken adı geçerli Python tanımlayıcısı olmalı
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
                    try:
                        val = SafeMathEvaluator.eval_expr(expr, env)
                        env[var_name] = val
                        execution_trace.append(f"{var_name} = {val}")
                    except Exception as e:
                        execution_trace.append(f"{var_name} = HATA ({e})")
            else:
                # İsimsiz tekil ifade ise
                try:
                    res = SafeMathEvaluator.eval_expr(line, env)
                    env["_result"] = res
                    execution_trace.append(f"_result = {res}")
                except Exception as e:
                    execution_trace.append(f"HATA ({e})")

        return {
            "environment": env,
            "trace": execution_trace,
            "success": len(env) > 0
        }


# Program-of-Thoughts Sistem İstemi
POT_EXTRACTION_SYSTEM_PROMPT = """You are a Program-Aided Language (PAL) Data Extraction & Math Engine.
Your task is to extract exact numbers from the provided report context and output EXECUTABLE PYTHON CODE to answer the question mathematically.

STRICT PROTOCOL:
1. Extract exact numbers directly from the report text or tables. Do not alter any numbers.
2. DO NOT calculate any results in your head. Write Python variable assignments and arithmetic formulas instead.
3. When summing multiple listed items/categories, explicitly put all extracted values into a Python list and use sum():
   e.g.: others = [3.39, 1.99, 0.05, 0.81, 1.22, 0.45, 14.05, 0.03, 0.05]
         others_total = sum(others)
4. Allowed operations & functions: +, -, *, /, sum(), min(), max(), abs(), round(), len().
5. End with the final answer variables clearly defined (e.g. group1_total = ..., group2_total = ..., difference_points = ...).

Example format:
```python
# Extracted numbers from report
cat1 = 36.23
cat2 = 38.24
top2_total = cat1 + cat2

# Sum of other listed categories
others = [3.39, 1.99, 0.05, 0.81, 1.22, 0.45, 14.05, 0.03, 0.05]
others_total = sum(others)

# Difference in percentage points
difference_points = top2_total - others_total
```
Respond ONLY with the Python code block."""


def is_mathematical_query(query: str) -> bool:
    """Sorgunun matematiksel hesaplama, karşılaştırma veya toplama gerektirip gerektirmediğini tespit eder."""
    q = query.lower()
    math_signals = [
        r"\btoplam\b", r"\btoplamı\b", r"\btoplam kaç\b", r"\bfark\b", r"\bfarkı\b", r"\bkaç puan\b", r"\byüzde puan\b",
        r"\bartış\b", r"\bazalış\b", r"\boran\b", r"\boranı\b", r"\byüzdesi\b", r"\byüzde kaç\b", r"\bkatı\b", r"\bdeğişim\b",
        r"\bhesapla\b", r"\bkarşılaştır\b", r"\bsum\b", r"\btotal\b", r"\bdifference\b", r"\bdelta\b", r"\bratio\b",
        r"\bpercentage point\b", r"\bpercentage points\b", r"\bincrease\b", r"\bdecrease\b", r"\bcompare\b",
        r"\bhigher\b", r"\blower\b", r"\bhow much more\b", r"\bhow much less\b"
    ]
    return any(bool(re.search(pat, q)) for pat in math_signals)

