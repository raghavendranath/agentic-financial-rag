import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finrag.core import resolve_citations, valid_citations
from finrag.experiment import draft, validate, trajectory
from test_experiment import FakeClient, FakeIndex, answer, verdict


class RefinementRegressionTests(unittest.TestCase):
    def test_observed_row_aliases_resolve_only_to_real_unique_retrieved_row(self):
        evidence = [{"id": "co/page.pdf::0", "filename": "co/page.pdf", "text": "[table_2] 10 20"}]
        for citation in ("co/page.pdf::table_2", "co/page.pdf::0::table_2", "co/page.pdf::0:table_2"):
            resolved, problems = resolve_citations([citation], evidence)
            self.assertEqual(resolved, ["co/page.pdf::0:table_2"])
            self.assertEqual(problems, [])
        self.assertFalse(valid_citations(["co/page.pdf::table_99"], evidence))
        self.assertFalse(valid_citations(["other/page.pdf::table_2"], evidence))
        duplicate = {**evidence[0], "id": "co/page.pdf::1300"}
        self.assertFalse(valid_citations(["co/page.pdf::table_2"], evidence + [duplicate]))
        self.assertTrue(valid_citations(["co/page.pdf::0:table_2"], evidence + [duplicate]))

    def test_validator_accept_cannot_override_disagreeing_audit(self):
        client = FakeClient([answer(20), verdict(True)])
        item = {"id": "q", "filename": "co/page", "question": "change?"}
        attempt = validate(client, item, draft(client, item, FakeIndex().retrieve("q"), "a"), "v")
        self.assertFalse(attempt["accepted"])
        self.assertTrue(any("disagrees" in issue for issue in attempt["operational_issues"]))

    def test_failed_unit_check_cannot_be_accepted(self):
        audit = verdict(True)
        audit["checks"]["units"] = False
        client = FakeClient([answer(10), audit])
        item = {"id": "q", "filename": "co/page", "question": "change?"}
        attempt = validate(client, item, draft(client, item, FakeIndex().retrieve("q"), "a"), "v")
        self.assertFalse(attempt["accepted"])

    def test_retry_gets_actual_operational_failure_instead_of_acceptance_only(self):
        bad = answer(10)
        bad["citations"] = ["not-a-real-citation"]
        client = FakeClient([bad, verdict(True), answer(10), verdict(True)])
        index = FakeIndex()
        item = {"id": "q", "filename": "co/page", "question": "change?"}
        first = validate(client, item, draft(client, item, index.retrieve("q"), "a"), "v")
        result = trajectory(client, index, item, first, {"max_attempts": 3, "top_k": 10}, True)
        payload = json.loads(client.messages[2][1]["content"])
        feedback = payload["previous_validation_feedback"]
        self.assertTrue(any("citations" in issue for issue in feedback["operational_issues"]))
        self.assertTrue(feedback["evidence_set_unchanged"])
        self.assertEqual(feedback["allowed_chunk_ids"], ["table_1"])
        self.assertTrue(result["accepted"])

    def test_frozen_v2_loaded_separately_and_preserved(self):
        from finrag.baseline import baseline_hash, policy
        from finrag.core import read_json
        self.assertEqual(baseline_hash(), read_json(ROOT / "provenance/v2-main-freeze.json")["source_hash"])
        module = policy()
        client = FakeClient([answer(20), verdict(True)])
        item = {"id": "q", "filename": "co/page", "question": "change?"}
        first = module.validate(client, item, module.draft(client, item, FakeIndex().retrieve("q"), "a"), "v")
        # v2 does not run the newly introduced independent audit-program agreement gate.
        self.assertTrue(first["accepted"])

    def test_paired_run_reuses_exact_initial_answer_for_both_policies(self):
        import tempfile
        from finrag.core import digest, write_json, jsonl
        from finrag.experiment import run_experiment
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, run = root / 'data', root / 'run'
            inputs = [{'id':'q','filename':'co/page','question':'change?'}]
            write_json(data / 'manifest.json', {'splits':{'pilot':{'ids':['q'],'inputs_hash':digest(inputs)}}})
            write_json(data / 'pilot.inputs.json', inputs)
            write_json(data / 'index/manifest.json', {'fixture':True})
            client = FakeClient([answer(10), verdict(True), verdict(True)])
            config = {'top_k':10,'max_attempts':3,'fixed_retrieval_ablation':False,'paired_v2_baseline':True}
            run_experiment(client, FakeIndex(), data, 'pilot', run, config, {})
            arms = jsonl(run / 'results.jsonl')[0]['arms']
            initial = arms['single_shot']['attempts'][0]
            for name in ('agentic','baseline_v2'):
                first = arms[name]['attempts'][0]
                for key in ('answer_call','answer_raw','calculator','evidence'):
                    self.assertEqual(first[key], initial[key])
            self.assertEqual(len(client.messages), 3)


if __name__ == "__main__":
    unittest.main()
