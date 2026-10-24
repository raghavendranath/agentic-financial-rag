import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finrag.calculator import execute
from finrag.experiment import draft, validate
from test_experiment import FakeClient, answer, verdict


class CalculatorTests(unittest.TestCase):
    def test_matches_official_execution(self):
        spec = importlib.util.spec_from_file_location("finqa_official", ROOT / "provenance/finqa-official-evaluate.py")
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        programs = ["divide(5758, 7975)", "add(91640, 2638), subtract(#0, 8349)",
                    "subtract(11.02, 9.52), divide(#0, 9.52)", "multiply(90000, 149.28)",
                    "subtract(35%, 32.9%)", "exp(1.05, 3)", "greater(5, 6)",
                    "add(-2.5, 0)", "multiply(100, const_1000000)", "add(1, const_m1)"]
        for program in programs:
            result = execute(program)
            invalid, expected = official.eval_program(result["program_tokens"], [])
            self.assertEqual(invalid, 0)
            actual = result["value"]
            if actual not in ("yes", "no"):
                actual = round(float(actual), 5)
            self.assertEqual(actual, expected, program)
        import random
        rng = random.Random(1819)
        for _ in range(100):
            a, b, c = (rng.randint(1, 100000) for _ in range(3))
            program = f"subtract({a}, {b}), divide(#0, {c})"
            result = execute(program)
            invalid, expected = official.eval_program(result["program_tokens"], [])
            self.assertEqual(invalid, 0)
            self.assertEqual(round(float(result["value"]), 5), expected)

    def test_rejects_code_and_unsafe_or_invalid_arithmetic(self):
        for program in ["__import__('os')", "divide(1, 0)", "add(#0, 1)", "exp(2, 999999)",
                        "multiply(1e999, 2)", "add(nan, 0)", "add(1,2),", "add(1,2) garbage", ""]:
            with self.assertRaises((ValueError, ArithmeticError), msg=program):
                execute(program)

    def test_calculator_corrects_stated_value_before_validation(self):
        proposal = {**answer(85829), "program": "add(91640, 2638), subtract(#0, 8349)"}
        audit = {**verdict(True), "audit_program": proposal["program"]}
        client = FakeClient([proposal, audit])
        item = {"question": "net?", "filename": "co/page", "id": "q"}
        attempt = draft(client, item, [{"id": "table_1", "text": "91640 2638 8349"}], "draft")
        self.assertEqual(float(attempt["answer"]["value"]), 85929)
        self.assertEqual(attempt["calculator"]["model_stated_value"], "85829")
        checked = validate(client, item, attempt, "check")
        self.assertTrue(checked["accepted"])

    def test_unknown_cannot_be_accepted_even_if_validator_says_yes(self):
        proposal = {**answer(1), "value": "unknown", "program": ""}
        client = FakeClient([proposal, verdict(True)])
        item = {"question": "net?", "filename": "co/page", "id": "q"}
        attempt = draft(client, item, [{"id": "table_1", "text": "1"}], "draft")
        checked = validate(client, item, attempt, "check")
        self.assertFalse(checked["accepted"])


if __name__ == "__main__":
    unittest.main()
