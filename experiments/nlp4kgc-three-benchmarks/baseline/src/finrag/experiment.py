from __future__ import annotations

import platform
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .core import (answer_messages, append_jsonl, digest, jsonl, parse_answer,
                   parse_verdict, prompt_hash, read_json, valid_citations, validator_messages, write_json)
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
    proposed = attempt["answer"] or {"malformed_reply": attempt["answer_raw"]}
    response = client.complete(validator_messages(item, attempt["evidence"], proposed), tag)
    verdict = parse_verdict(response["content"]) if response.get("finish_reason") == "stop" else None
    cited = attempt["answer"]["citations"] if attempt["answer"] else []
    citation_ids_valid = valid_citations(cited, attempt["evidence"])
    answerable = bool(attempt["answer"] and attempt["calculator"]["status"] == "executed")
    accepted = bool(verdict and verdict["accept"] and citation_ids_valid and answerable)
    return {**attempt, "verdict": verdict, "accepted": accepted,
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
        feedback = {"reason": verdict["reason"] if verdict else "Invalid validator output; re-check the evidence.",
                    "previous_answer": previous["answer"], "calculator": previous["calculator"],
                    "operational_feedback": "Need valid citations and executable arithmetic; unknown cannot be accepted."}
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
        frozen = {"config": config, "split": split, "workers": workers, "data_hash": digest(data_manifest),
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
