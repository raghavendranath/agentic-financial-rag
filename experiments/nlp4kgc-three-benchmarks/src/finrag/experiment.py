from __future__ import annotations

import platform
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .core import (answer_messages, append_jsonl, digest, jsonl, parse_answer,
                   parse_verdict, prompt_hash, read_json, valid_citations, validator_messages, write_json,
                   resolve_citations, correct)
from .data import load_inputs
from .calculator import execute


def source_hash():
    return digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})


def draft(client, item, evidence, tag, feedback=None):
    response = client.complete(answer_messages(item, evidence, feedback), tag)
    parsed = parse_answer(response["content"]) if response.get("finish_reason") == "stop" else None
    calculator = {"status": "no_answer"}
    if parsed:
        if parsed["value"].strip().lower() == "unknown" and not parsed["program"].strip():
            calculator = {"status": "abstained"}
        else:
            try:
                result = execute(parsed["program"])
                calculator = {"status": "executed", "model_stated_value": parsed["value"], **result}
                parsed = {**parsed, "value": result["value"]}
            except (ValueError, ArithmeticError):
                calculator = {"status": "invalid_program"}
                parsed = {**parsed, "value": "unknown"}
    return {"answer": parsed, "answer_raw": response["content"], "evidence": evidence,
            "calculator": calculator,
            "answer_call": response["request_hash"], "answer_seconds": response["elapsed_seconds"],
            "answer_finish_reason": response.get("finish_reason")}


def validate(client, item, attempt, tag):
    original = attempt["answer"]
    resolved, citation_problems = resolve_citations(original["citations"] if original else [], attempt["evidence"])
    proposed = ({**original, "citations": resolved if not citation_problems else original["citations"]}
                if original else {"malformed_reply": attempt["answer_raw"]})
    response = client.complete(validator_messages(item, attempt["evidence"], proposed), tag)
    verdict = parse_verdict(response["content"]) if response.get("finish_reason") == "stop" else None
    cited = proposed["citations"] if original else []
    citation_ids_valid = valid_citations(cited, attempt["evidence"])
    answerable = bool(attempt["answer"] and attempt["calculator"]["status"] == "executed")
    issues = []
    if not original:
        issues.append("Answer JSON is malformed; return the required value/program/citations/explanation fields.")
    if not citation_ids_valid:
        issues.append("Invalid or missing citations. Use actual evidence IDs. Problems: " + str(citation_problems))
    if not answerable:
        issues.append("No executable answer: " + attempt["calculator"]["status"] + ". Provide a valid program only if the evidence supports the requested quantity.")
    audit = {"status": "invalid_verdict"}
    if verdict:
        failed = [key for key, passed in verdict["checks"].items() if not passed]
        if failed:
            issues.append("Failed financial checks: " + ", ".join(failed))
        audit_citations, audit_problems = resolve_citations(verdict["audit_citations"], attempt["evidence"])
        try:
            audit = {"status": "executed", **execute(verdict["audit_program"]),
                     "citations": audit_citations, "citation_problems": audit_problems}
            if not audit_citations or audit_problems:
                issues.append("Audit inputs lack valid retrieved citations.")
            if not original or not correct(audit["value"], original["value"]):
                issues.append("Source-derived audit result " + audit["value"] + " disagrees with the candidate; re-check target, formula, period and output units.")
        except (ValueError, ArithmeticError):
            audit = {"status": "invalid_program"}
            issues.append("Validator audit program is missing/invalid; no arithmetic agreement established.")
        if not verdict["accept"]:
            issues.append("Validator rejected: " + verdict["reason"])
    else:
        issues.append("Validator JSON is invalid; the source audit could not be completed.")
    accepted = bool(verdict and verdict["accept"] and not issues)
    return {**attempt, "answer": proposed if original else None, "original_answer": original,
            "verdict": verdict, "accepted": accepted, "operational_issues": issues,
            "audit": audit, "citation_resolution": {"original": original["citations"] if original else [],
                                                       "resolved": resolved, "problems": citation_problems},
            "citation_ids_valid": citation_ids_valid,
            "validator_call": response["request_hash"], "validator_seconds": response["elapsed_seconds"],
            "validator_raw": response["content"], "validator_finish_reason": response.get("finish_reason")}


def trajectory(client, index, item, first, config, refine):
    attempts = [first]
    for n in range(1, config["max_attempts"]):
        previous = attempts[-1]
        if previous["accepted"]:
            break
        verdict = previous["verdict"]
        query = (verdict["retrieval_query"].strip() if verdict else "") or item["question"]
        if refine:
            # Retain the original question to avoid losing the company/metric context.
            evidence = index.retrieve(item["question"] + "\n" + query, config["top_k"], filename=item["filename"])
        else:
            evidence = first["evidence"]
        previous_ids = {e["id"] for e in previous["evidence"]}
        unchanged = previous_ids == {e["id"] for e in evidence}
        feedback = {"reason": verdict["reason"] if verdict else "Invalid validator output; re-check the evidence.",
                    "previous_answer": previous["answer"], "calculator": previous["calculator"],
                    "operational_issues": previous["operational_issues"], "audit": previous["audit"],
                    "allowed_chunk_ids": [e["id"] for e in evidence],
                    "evidence_set_unchanged": unchanged,
                    "history": [{"program": (a["answer"] or {}).get("program"),
                                 "value": (a["answer"] or {}).get("value"),
                                 "issues": a["operational_issues"]} for a in attempts],
                    "operational_feedback": "Fix the stated failed check. Do not repeat a rejected formula without explaining a supported correction. If only citation syntax failed, preserve the formula and fix the citation. If evidence is still insufficient, abstain rather than invent missing inputs."}
        tag = item["id"] + (":full:" if refine else ":fixed:") + str(n + 1)
        attempt = draft(client, item, evidence, tag + ":answer", feedback)
        attempts.append(validate(client, item, attempt, tag + ":validate"))
    return {"attempts": attempts, "final": attempts[-1]["answer"], "accepted": attempts[-1]["accepted"]}


def run_experiment(client, index, data_dir, split, run_dir, config, metadata, workers=1):
    folder = Path(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    # The lock covers calls, budget and results for the full run; no concurrent writer.
    import fcntl
    with (folder / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is already writing to this run") from None
        inputs, data_manifest = load_inputs(data_dir, split)
        index_manifest = read_json(Path(data_dir) / "index/manifest.json")
        from .baseline import baseline_hash
        frozen = {"config": config, "split": split, "workers": workers, "data_hash": digest(data_manifest),
                  "baseline_v2_source_hash": baseline_hash(),
                  "index_hash": digest(index_manifest), "prompt_hash": prompt_hash(), "source_hash": source_hash()}
        path = folder / "manifest.json"
        if path.exists():
            if read_json(path)["frozen"] != frozen:
                raise ValueError("Run configuration/code/data changed. Use a new run directory; old results remain immutable.")
        else:
            write_json(path, {"frozen": frozen, "preflight": metadata, "python": platform.python_version(),
                              "platform": platform.platform(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "status": "in_progress", "inference_has_no_label_access": True})
        existing = {r["id"]: r for r in jsonl(folder / "results.jsonl")}
        stop = threading.Event()
        index_lock = threading.Lock()
        class LockedIndex:
            def retrieve(self, *args, **kwargs):
                with index_lock:
                    return index.retrieve(*args, **kwargs)
        locked_index = LockedIndex()
        def one_question(i, item):
            if stop.is_set():
                return None
            print(f"Question {i}/{len(inputs)}: {item['id']}", flush=True)
            evidence = locked_index.retrieve(item["question"], config["top_k"], filename=item["filename"])
            initial = draft(client, item, evidence, item["id"] + ":initial")
            first = validate(client, item, initial, item["id"] + ":initial:validate")
            arms = {"single_shot": {"final": initial["answer"], "attempts": [initial]},
                    "agentic": trajectory(client, locked_index, item, first, config, refine=True)}
            if config.get("paired_v2_baseline"):
                from .baseline import policy
                baseline = policy()
                old_first = baseline.validate(client, item, initial, item["id"] + ":v2:validate")
                arms["baseline_v2"] = baseline.trajectory(client, locked_index, item, old_first, config, refine=True)
            if config["fixed_retrieval_ablation"]:
                arms["fixed_retrieval"] = trajectory(client, locked_index, item, first, config, refine=False)
            return {"id": item["id"], "filename": item["filename"], "question": item["question"], "arms": arms}
        errors = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(one_question, i, item) for i, item in enumerate(inputs, 1) if item["id"] not in existing]
            for future in as_completed(pending):
                try:
                    row = future.result()
                    if row is not None:
                        append_jsonl(folder / "results.jsonl", row)
                except Exception as exc:
                    stop.set()
                    errors.append(exc)
        if errors:
            raise errors[0]
        manifest = read_json(path)
        manifest["status"] = "inference_complete"
        write_json(path, manifest)
    return folder
