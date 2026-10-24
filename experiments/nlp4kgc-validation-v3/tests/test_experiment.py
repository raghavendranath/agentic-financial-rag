import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finrag.core import (correct, number, parse_answer, parse_verdict, public_item,
                         digest, select, write_json, append_jsonl, prompt_hash, valid_citations)
from finrag.experiment import draft, validate, trajectory, run_experiment
from finrag.report import make_report
from finrag.client import OpenRouterClient, RateLimited


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    def complete(self, messages, tag):
        self.messages.append(messages)
        return {"content": json.dumps(next(self.replies)), "request_hash": str(len(self.messages)),
                "finish_reason": "stop", "elapsed_seconds": 1.0}


class FakeIndex:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, top_k=10, filename=None):
        self.queries.append(query)
        return [{"id": "table_1", "text": "2023 revenue 100; 2024 revenue 110"}]


def answer(value):
    return {"value": str(value), "program": f"add({value}, 0)",
            "citations": ["table_1"], "explanation": "110 - 100"}


def verdict(accept):
    return {"accept": accept, "reason": "Check years and subtraction", "retrieval_query": "revenue 2023 2024",
            "checks": {"target": True, "inputs": True, "periods": True, "units": True},
            "audit_program": "subtract(110, 100)", "audit_citations": ["table_1"]}


class ScoringTests(unittest.TestCase):
    def test_numeric_edge_cases(self):
        for text, value in [("6.4748e+06", 6474800), ("(1,234.50)", -1234.5),
                            ("12%", 0.12), ("−0.5", -0.5), ("yes", "yes")]:
            self.assertEqual(number(text), value)
        for text in ["NaN", "infinity", "Answer: 42", "2023: 42", "1,2", "", "unknown", True]:
            self.assertIsNone(number(text))
        self.assertFalse(correct("12", 0.12))
        self.assertTrue(correct("12%", 0.12))
        self.assertFalse(correct("yes", 1))
        self.assertTrue(correct("6.4748e6", 6474800))

    def test_strict_and_relaxed_are_separate(self):
        self.assertFalse(correct(100.2, 100))
        self.assertTrue(correct(100.2, 100, relaxed=True))
        self.assertFalse(correct(0.001, 0))

    def test_schema_validation(self):
        self.assertIsNone(parse_verdict('{"accept":"false","reason":"x","retrieval_query":"x"}'))
        self.assertIsNone(parse_answer('{"value":12,"citations":"table_1","explanation":"x"}'))
        self.assertIsNotNone(parse_answer(json.dumps(answer(10))))

    def test_accepted_unused_query_can_be_null(self):
        data = verdict(True)
        data["retrieval_query"] = None
        result = parse_verdict(json.dumps(data))
        self.assertEqual(result["retrieval_query"], "")
        self.assertIsNone(parse_verdict('{"accept":false,"reason":"missing","retrieval_query":null}'))

    def test_nested_citation_requires_real_parent_and_row(self):
        evidence = [{"id": "report::0", "text": "[text_3] The amount is 246."}]
        self.assertTrue(valid_citations(["report::0"], evidence))
        self.assertTrue(valid_citations(["report::0:text_3"], evidence))
        self.assertFalse(valid_citations(["report::0:text_99"], evidence))
        self.assertFalse(valid_citations(["other::0:text_3"], evidence))
        self.assertFalse(valid_citations(["text_3"], evidence))

    def test_gold_never_enters_public_data(self):
        raw = {"id": "x", "filename": "co/2024/page1", "pre_text": ["revenue 100"], "table": [],
               "qa": {"question": "what revenue?", "exe_ans": 999.123456789,
                      "gold_inds": {"secret": "do not send"}, "program": "gold_program_marker"}}
        public = public_item(raw)
        text = json.dumps(public)
        self.assertNotIn("999.123456789", text)
        self.assertNotIn("gold_program_marker", text)
        self.assertNotIn("gold_inds", text)

    def test_sample_has_no_duplicate_report_pages(self):
        rows = [{"id": str(i), "filename": f"co/page{i//2}", "pre_text": ["x"],
                 "qa": {"exe_ans": 1, "question": "q"}} for i in range(10)]
        a, b = select(rows, 4, 18), select(rows[::-1], 4, 18)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
        self.assertEqual(len({r["filename"] for r in a}), 4)


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.item = {"id": "q", "filename": "co/2024/page1", "question": "Revenue change?"}
        self.index = FakeIndex()
        self.config = {"top_k": 10, "max_attempts": 3, "fixed_retrieval_ablation": True}

    def test_rejected_draft_is_refined_then_stops(self):
        client = FakeClient([answer(20), verdict(False), answer(10), verdict(True)])
        initial = draft(client, self.item, self.index.retrieve("q"), "a")
        first = validate(client, self.item, initial, "v")
        result = trajectory(client, self.index, self.item, first, self.config, True)
        self.assertEqual(number(result["final"]["value"]), 10)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(len(client.messages), 4)
        self.assertIn("revenue 2023 2024", self.index.queries[-1])

    def test_accepted_first_draft_is_unchanged(self):
        client = FakeClient([answer(10), verdict(True)])
        first = validate(client, self.item, draft(client, self.item, self.index.retrieve("q"), "a"), "v")
        result = trajectory(client, self.index, self.item, first, self.config, True)
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(len(client.messages), 2)

    def test_never_exceeds_three_attempts(self):
        client = FakeClient([answer(20), verdict(False)] * 3)
        first = validate(client, self.item, draft(client, self.item, self.index.retrieve("q"), "a"), "v")
        result = trajectory(client, self.index, self.item, first, self.config, True)
        self.assertEqual(len(result["attempts"]), 3)
        self.assertFalse(result["accepted"])

    def test_fixed_retrieval_does_not_search_again(self):
        client = FakeClient([answer(20), verdict(False), answer(10), verdict(True)])
        first = validate(client, self.item, draft(client, self.item, self.index.retrieve("q"), "a"), "v")
        trajectory(client, self.index, self.item, first, self.config, False)
        self.assertEqual(len(self.index.queries), 1)

    def test_fabricated_citation_cannot_pass(self):
        bad = answer(10)
        bad["citations"] = ["not_retrieved"]
        client = FakeClient([bad, verdict(True)])
        first = validate(client, self.item, draft(client, self.item, self.index.retrieve("q"), "a"), "v")
        self.assertFalse(first["accepted"])


class BudgetTests(unittest.TestCase):
    def test_explicit_rate_limit_retries_are_bounded_and_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"model": "test", "provider": "test", "budget_usd": 5,
                      "temperature": 0, "seed": 18, "max_tokens": 100}
            client = OpenRouterClient(config, folder, "secret", {"pricing": {"prompt": .000001, "completion": .000001}, "endpoint": {}})
            reply = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {"cost": .001}}
            with patch("finrag.client.fetch_json", side_effect=[RateLimited(), reply]) as network, patch("finrag.client.time.sleep") as wait:
                result = client.complete([{"role": "user", "content": "same"}], "test")
                self.assertEqual(result["usage"]["cost"], .001)
                self.assertEqual(network.call_count, 2)
                wait.assert_called_once_with(20)
            self.assertTrue((Path(folder) / "transport-events.jsonl").exists())

    def test_repeated_rate_limit_stops_after_three_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"model": "test", "provider": "test", "budget_usd": 5,
                      "temperature": 0, "seed": 18, "max_tokens": 100}
            client = OpenRouterClient(config, folder, "secret", {"pricing": {"prompt": .000001, "completion": .000001}, "endpoint": {}})
            with patch("finrag.client.fetch_json", side_effect=RateLimited()) as network, patch("finrag.client.time.sleep") as wait:
                with self.assertRaisesRegex(RuntimeError, "three bounded retries"):
                    client.complete([{"role": "user", "content": "same"}], "test")
                self.assertEqual(network.call_count, 4)
                self.assertEqual(wait.call_count, 3)

    def test_concurrent_identical_requests_have_one_charge(self):
        from concurrent.futures import ThreadPoolExecutor
        import time
        with tempfile.TemporaryDirectory() as folder:
            config = {"model": "test", "provider": "test", "budget_usd": 5,
                      "temperature": 0, "seed": 18, "max_tokens": 100}
            client = OpenRouterClient(config, folder, "secret", {"pricing": {"prompt": .000001, "completion": .000001}, "endpoint": {}})
            def reply(*args, **kwargs):
                time.sleep(.02)
                return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {"cost": .001}}
            with patch("finrag.client.fetch_json", side_effect=reply) as network:
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(client.complete, [{"role": "user", "content": "same"}], str(i)) for i in range(3)]
                    results = [f.result() for f in futures]
                self.assertEqual(network.call_count, 1)
                self.assertTrue(all(r["request_hash"] == results[0]["request_hash"] for r in results))

    def test_concurrent_distinct_requests_share_budget_guard(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as folder:
            config = {"model": "test", "provider": "test", "budget_usd": .002,
                      "temperature": 0, "seed": 18, "max_tokens": 100}
            client = OpenRouterClient(config, folder, "secret", {"pricing": {"prompt": .000001, "completion": .000001}, "endpoint": {}})
            reply = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {"cost": .001}}
            with patch("finrag.client.fetch_json", return_value=reply) as network:
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(client.complete, [{"role": "user", "content": str(i)}], str(i)) for i in range(3)]
                    successes = 0
                    for future in futures:
                        try:
                            future.result()
                            successes += 1
                        except ValueError:
                            pass
                self.assertEqual(successes, 1)
                self.assertEqual(network.call_count, 1)

    def test_budget_blocks_request_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"model": "test", "provider": "test", "budget_usd": .000001,
                      "temperature": 0, "seed": 18, "max_tokens": 100}
            client = OpenRouterClient(config, folder, "never-written", {"pricing": {"prompt": .1, "completion": .1}, "endpoint": {}})
            with patch("finrag.client.fetch_json") as network:
                with self.assertRaises(ValueError):
                    client.complete([{"role": "user", "content": "test"}], "test")
                network.assert_not_called()

    def test_successful_calls_cached_without_secret_in_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"model": "test", "provider": "test", "budget_usd": 5,
                      "temperature": 0, "seed": 18, "max_tokens": 100}
            client = OpenRouterClient(config, folder, "SECRET_SENTINEL", {"pricing": {"prompt": .000001, "completion": .000001}, "endpoint": {}})
            reply = {"id": "r", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {"cost": .001}}
            with patch("finrag.client.fetch_json", return_value=reply) as network:
                one = client.complete([{"role": "user", "content": "test"}], "a")
                two = client.complete([{"role": "user", "content": "test"}], "b")
                self.assertEqual(one, two)
                self.assertEqual(network.call_count, 1)
            for p in Path(folder).glob("*.jsonl"):
                self.assertNotIn("SECRET_SENTINEL", p.read_text())

    def test_ambiguous_billing_blocks_automatic_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"model": "test", "provider": "test", "budget_usd": 5,
                      "temperature": 0, "seed": 18, "max_tokens": 100}
            client = OpenRouterClient(config, folder, "secret", {"pricing": {"prompt": .000001, "completion": .000001}, "endpoint": {}})
            messages = [{"role": "user", "content": "test"}]
            with patch("finrag.client.fetch_json", side_effect=RuntimeError("timeout")) as network:
                with self.assertRaises(RuntimeError):
                    client.complete(messages, "a")
                with self.assertRaisesRegex(RuntimeError, "ambiguous billing"):
                    client.complete(messages, "b")
                self.assertEqual(network.call_count, 1)


class IntegrationTests(unittest.TestCase):
    def test_full_offline_run_reports_improvement_and_damage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, run = root / "data", root / "run"
            inputs = [{"id": "a", "filename": "firm1/2024/page1", "question": "change?"},
                      {"id": "b", "filename": "firm2/2024/page1", "question": "change?"}]
            labels = {key: {"gold": 10} for key in ("a", "b")}
            data_manifest = {"splits": {"evaluation": {"ids": ["a", "b"], "inputs_hash": digest(inputs), "labels_hash": digest(labels)}}}
            write_json(data / "manifest.json", data_manifest)
            write_json(data / "evaluation.inputs.json", inputs)
            write_json(data / "evaluation.labels.json", labels)
            write_json(data / "index/manifest.json", {"test_fixture": True})
            client = FakeClient([answer(20), verdict(False), answer(10), verdict(True), answer(10), verdict(True),
                                 answer(10), verdict(False), answer(20), {**verdict(True), "audit_program": "add(20, 0)"}, answer(10), verdict(True)])
            config = {"top_k": 10, "max_attempts": 3, "fixed_retrieval_ablation": True}
            run_experiment(client, FakeIndex(), data, "evaluation", run, config, {"test_fixture": True})
            report = make_report(data, "evaluation", run)
            self.assertEqual(report["paired_transitions"]["agentic"]["wrong_to_right"], 1)
            self.assertEqual(report["paired_transitions"]["agentic"]["right_to_wrong"], 1)
            self.assertEqual(report["paired_transitions"]["agentic"]["accuracy_difference"], 0)
            self.assertEqual(report["validator_reference_evaluation"]["true_reject_wrong"], 1)
            self.assertEqual(report["validator_reference_evaluation"]["false_reject_correct"], 1)
            before = len(client.messages)
            run_experiment(client, FakeIndex(), data, "evaluation", run, config, {"test_fixture": True})
            self.assertEqual(len(client.messages), before)
            (data / "evaluation.labels.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "labels changed"):
                make_report(data, "evaluation", run)

    def test_token_chunk_overlap(self):
        from finrag.dense import DenseIndex
        class CharacterTokenizer:
            def __call__(self, text, **kwargs):
                return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}
        index = object.__new__(DenseIndex)
        index.tokenizer = CharacterTokenizer()
        chunks = index.chunk("report", "0123456789", 6, 2)
        self.assertEqual([c["text"] for c in chunks], ["012345", "456789"])
        self.assertEqual(chunks[0]["token_end"] - chunks[1]["token_start"], 2)


if __name__ == "__main__":
    unittest.main()
