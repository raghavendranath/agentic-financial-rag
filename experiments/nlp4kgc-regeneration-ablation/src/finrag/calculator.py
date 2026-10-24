"""Bounded arithmetic tool. Never evaluates Python, source text, or reference labels."""
from __future__ import annotations

import math
import re

OPS = {"add", "subtract", "multiply", "divide", "exp", "greater"}
STEP = re.compile(r"\s*(add|subtract|multiply|divide|exp|greater)\(\s*([^(),]+)\s*,\s*([^(),]+)\s*\)\s*")
LITERAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?\Z")


def literal(text):
    text = text.strip()
    if text.startswith("const_"):
        text = text[6:]
        if text == "m1":
            text = "-1"
    if not LITERAL.fullmatch(text):
        raise ValueError("Invalid arithmetic literal")
    value = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    if not math.isfinite(value) or abs(value) > 1e18:
        raise ValueError("Arithmetic literal out of bounds")
    return value


def execute(program):
    if not isinstance(program, str) or not program or len(program) > 2000:
        raise ValueError("Missing or oversized arithmetic program")
    position, values, tokens = 0, [], []
    while position < len(program):
        match = STEP.match(program, position)
        if not match or len(values) >= 24:
            raise ValueError("Invalid arithmetic program")
        op, a, b = (s.strip() for s in match.groups())
        args = []
        for arg in (a, b):
            if re.fullmatch(r"#\d+", arg):
                i = int(arg[1:])
                if i >= len(values) or isinstance(values[i], str):
                    raise ValueError("Invalid arithmetic reference")
                args.append(values[i])
            else:
                args.append(literal(arg))
        a_value, b_value = args
        if op == "add":
            result = a_value + b_value
        elif op == "subtract":
            result = a_value - b_value
        elif op == "multiply":
            result = a_value * b_value
        elif op == "divide":
            if b_value == 0:
                raise ValueError("Division by zero")
            result = a_value / b_value
        elif op == "exp":
            if abs(b_value) > 8:
                raise ValueError("Exponent out of bounds")
            result = a_value ** b_value
        else:
            result = "yes" if a_value > b_value else "no"
        if not isinstance(result, str) and (not isinstance(result, (int, float)) or
                                           not math.isfinite(result) or abs(result) > 1e24):
            raise ValueError("Arithmetic result out of bounds")
        values.append(result)
        tokens.extend([op + "(", a, b, ")"])
        position = match.end()
        if position < len(program):
            if not program.startswith(",", position):
                raise ValueError("Invalid step separator")
            position += 1
            if not program[position:].strip():
                raise ValueError("Trailing separator")
    result = values[-1]
    return {"value": result if isinstance(result, str) else repr(result),
            "program_tokens": tokens + ["EOF"], "steps": len(values)}
