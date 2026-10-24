"""FinanceBench refinement searches for the validator's missing fact.

This policy deliberately leaves embedding, chunking, top-k and known-document
filtering to the existing index. The original question remains in the answer and
validation prompts; repeating it in a focused retrieval query can obscure the
missing operand. No labels or reference pages participate in query selection.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def refinement_query(item: Mapping[str, Any], verdict: Mapping[str, Any] | None) -> str:
    """Use a usable standalone validator query, otherwise the original question."""
    query = verdict.get("retrieval_query") if isinstance(verdict, Mapping) else None
    if isinstance(query, str) and query.strip():
        return query.strip()
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Refinement requires a nonempty question or retrieval query")
    return question.strip()


def retrieve_refinement(index: Any, item: Mapping[str, Any],
                        verdict: Mapping[str, Any] | None, top_k: int = 10) -> list:
    """Retrieve with the focused query while preserving the known-report filter.

    Previous evidence is not silently added, unioned or reranked. Such policies
    need their own prospective evaluation and can displace a missing operand
    under the fixed context budget.
    """
    filename = item.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("FinanceBench refinement requires the known document")
    return index.retrieve(refinement_query(item, verdict), top_k, filename=filename)
