"""Regression checks for focused missing-fact retrieval without model calls."""
import copy
import unittest

from finrag.fb_retrieval import refinement_query, retrieve_refinement


class RecordingIndex:
    def __init__(self):
        self.calls = []
        self.evidence = [{"id": "REPORT/report::1300", "text": "source evidence"}]

    def retrieve(self, query, top_k, filename=None):
        self.calls.append((query, top_k, filename))
        return self.evidence


class FocusedRefinementTests(unittest.TestCase):
    def setUp(self):
        self.item = {"question": "Long ratio question with already known numerator",
                     "filename": "REPORT/report"}

    def test_missing_operand_query_replaces_original_and_preserves_filter(self):
        index = RecordingIndex()
        verdict = {"retrieval_query": "  REPORT fiscal year total current liabilities  "}
        before = copy.deepcopy((self.item, verdict))
        result = retrieve_refinement(index, self.item, verdict)
        self.assertEqual(index.calls, [("REPORT fiscal year total current liabilities", 10,
                                        "REPORT/report")])
        self.assertIs(result, index.evidence)
        self.assertEqual((self.item, verdict), before)

    def test_absent_unusable_or_empty_verdict_query_falls_back_once(self):
        for verdict in (None, {}, {"retrieval_query": " \n"}, {"retrieval_query": None},
                        {"retrieval_query": ["not a validated query"]}):
            with self.subTest(verdict=verdict):
                self.assertEqual(refinement_query(self.item, verdict), self.item["question"])

    def test_no_known_document_does_not_allow_cross_report_search(self):
        index = RecordingIndex()
        with self.assertRaisesRegex(ValueError, "known document"):
            retrieve_refinement(index, {"question": "question"}, {"retrieval_query": "fact"})
        self.assertEqual(index.calls, [])

    def test_no_query_or_question_fails_before_search(self):
        index = RecordingIndex()
        with self.assertRaisesRegex(ValueError, "nonempty question"):
            retrieve_refinement(index, {"filename": "REPORT/report", "question": "  "}, None)
        self.assertEqual(index.calls, [])


if __name__ == "__main__":
    unittest.main()
