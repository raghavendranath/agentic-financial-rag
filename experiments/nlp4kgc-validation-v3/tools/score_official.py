"""Execute saved model programs with pinned upstream FinQA code, entirely offline."""
import argparse
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finrag.core import digest, jsonl, number, read_json, write_json
from finrag.calculator import execute
from finrag.report import cluster_interval


def score(data, folder):
    data, folder = Path(data), Path(folder)
    manifest = read_json(folder / "manifest.json")
    dataset = read_json(data / "manifest.json")
    if manifest["status"] != "inference_complete" or manifest["frozen"]["data_hash"] != digest(dataset):
        raise ValueError("Incomplete run or changed dataset")
    split = manifest["frozen"]["split"]
    labels = read_json(data / (split + ".labels.json"))
    if digest(labels) != dataset["splits"][split]["labels_hash"]:
        raise ValueError("Reference labels changed")
    rows = jsonl(folder / "results.jsonl")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(dataset["splits"][split]["ids"]):
        raise ValueError("Run not complete")
    order = {key: i for i, key in enumerate(dataset["splits"][split]["ids"])}
    rows.sort(key=lambda row: order[row["id"]])
    script = ROOT / "provenance/finqa-official-evaluate.py"
    spec = importlib.util.spec_from_file_location("finqa_official", script)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    arms, details, outcomes = {}, [], {}
    for arm in rows[0]["arms"]:
        hits, numeric_hits, corrected_stated = [], [], 0
        for row in rows:
            result = row["arms"][arm]
            final = result["attempts"][-1]
            calc = final["calculator"]
            hit = False
            if calc["status"] == "executed":
                # Arithmetic programs contain no table operations: [] is intentional.
                invalid, value = upstream.eval_program(calc["program_tokens"], [])
                hit = invalid == 0 and value == labels[row["id"]]["gold"]
                local = execute(final["answer"]["program"])
                local_value = number(local["value"])
                local_value = round(local_value, 5) if isinstance(local_value, (int, float)) else local_value
                if invalid or value != local_value:
                    raise ValueError("Local arithmetic and upstream execution disagree")
                if local["program_tokens"] != calc["program_tokens"]:
                    raise ValueError("Saved program tokens differ from raw model program")
                stated = number(calc["model_stated_value"])
                corrected_stated += stated != number(calc["value"])
            else:
                value = None
            hits.append(int(hit))
            if isinstance(number(labels[row["id"]]["gold"]), (int, float)):
                numeric_hits.append(int(hit))
            details.append({"id": row["id"], "arm": arm, "official_execution_value": value,
                            "reference": labels[row["id"]]["gold"], "correct": bool(hit),
                            "calculator_status": calc["status"],
                            "model_stated_value": calc.get("model_stated_value"),
                            "final_program": (result["final"] or {}).get("program")})
        outcomes[arm] = hits
        arms[arm] = {"n": len(hits), "correct": sum(hits),
                     "official_program_execution_accuracy": sum(hits) / len(hits),
                     "accuracy_95ci": cluster_interval(rows, hits),
                     "scalar_numerical_f1": sum(numeric_hits) / len(numeric_hits) if numeric_hits else None,
                     "numerical_n": len(numeric_hits), "numerical_correct": sum(numeric_hits),
                     "final_answers_with_calculator_value_different_from_model_stated_value": corrected_stated}
    differences = [a - b for a, b in zip(outcomes["agentic"], outcomes["single_shot"])]
    report = {"evaluation": "Upstream FinQA execution on predicted arithmetic programs",
              "upstream_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
              "arms": arms, "paired_agentic_minus_single_shot": {
                  "difference": sum(differences) / len(differences),
                  "95ci": cluster_interval(rows, differences),
                  "corrected": sum(d == 1 for d in differences), "corrupted": sum(d == -1 for d in differences)},
              "limitations": ["Public processed-page subset, known-filing retrieval; not original 1200-question run.",
                  "Numerical F1 on one final number equals numeric-item accuracy; not recovered historical scorer.",
                  "Symbolic program accuracy not claimed; this checks execution only.",
                  "Calculator validates arithmetic, not financial interpretation or evidence grounding.",
                  "No label edits, scale forgiveness, sign forgiveness or tolerance changes."]}
    write_json(folder / "OFFICIAL-SCORING.json", report)
    write_json(folder / "official-item-scores.json", details)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fresh50")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(score(args.data, args.run))
