from __future__ import annotations

import csv
import random
from pathlib import Path
from .core import correct, digest, jsonl, number, read_json, write_json


def fraction(a, b):
    return a / b if b else None


def cluster_interval(rows, values, repetitions=2000):
    groups = {}
    for row, value in zip(rows, values):
        filing = row["filename"].rsplit("/", 1)[0]
        groups.setdefault(filing, []).append(value)
    groups = list(groups.values())
    if len(groups) < 2:
        return None
    rng = random.Random(18)
    draws = []
    for _ in range(repetitions):
        selected = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(sum(sum(g) for g in selected) / sum(len(g) for g in selected))
    draws.sort()
    return [draws[int(0.025 * repetitions)], draws[min(repetitions - 1, int(0.975 * repetitions))]]


def make_report(data_dir, split, run_dir):
    folder, data = Path(run_dir), Path(data_dir)
    manifest = read_json(folder / "manifest.json")
    data_manifest = read_json(data / "manifest.json")
    if manifest["frozen"]["data_hash"] != digest(data_manifest):
        raise ValueError("Report dataset differs from inference dataset")
    if manifest["frozen"]["split"] != split:
        raise ValueError("Report split differs from run")
    labels = read_json(data / (split + ".labels.json"))
    if digest(labels) != data_manifest["splits"][split]["labels_hash"]:
        raise ValueError("Reference labels changed")
    rows = jsonl(folder / "results.jsonl")
    ids = [r["id"] for r in rows]
    expected = data_manifest["splits"][split]["ids"]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Inference incomplete or duplicated; do not report a selected subset as the full experiment")
    order = {key: i for i, key in enumerate(expected)}
    rows.sort(key=lambda row: order[row["id"]])
    names = list(rows[0]["arms"])
    outcomes, summary = {}, {}
    for arm in names:
        hits, relaxed, invalid, attempts, elapsed, scalar_f1 = [], [], 0, [], [], []
        for row in rows:
            result = row["arms"][arm]
            answer = result["final"]
            value = answer["value"] if answer else None
            gold = labels[row["id"]]["gold"]
            hits.append(int(correct(value, gold)))
            if isinstance(number(gold), (int, float)):
                # One final numerical scalar: precision=recall=1 iff the scalar matches.
                # Boolean questions are excluded from this numerical-only measure.
                scalar_f1.append(int(correct(value, gold)))
            relaxed.append(int(correct(value, gold, relaxed=True)))
            invalid += number(value) is None
            attempts.append(len(result["attempts"]))
            elapsed.append(sum(a["answer_seconds"] + a.get("validator_seconds", 0) for a in result["attempts"]))
        outcomes[arm] = hits
        summary[arm] = {"n": len(rows), "strict_numerical_accuracy": sum(hits) / len(rows),
                        "normalized_exact_match": sum(hits) / len(rows),
                        "accuracy_95ci_filing_bootstrap": cluster_interval(rows, hits),
                        "relaxed_0.5pct_accuracy": sum(relaxed) / len(rows),
                        "unparseable_or_unknown": invalid, "mean_attempts": sum(attempts) / len(rows),
                        "numerical_f1_final_scalar": fraction(sum(scalar_f1), len(scalar_f1)),
                        "numerical_f1_n": len(scalar_f1),
                        "mean_stored_api_seconds_per_trajectory": sum(elapsed) / len(rows)}
    transitions = {}
    for arm in names:
        if arm == "single_shot":
            continue
        base, final = outcomes["single_shot"], outcomes[arm]
        changes = [f - b for f, b in zip(final, base)]
        transitions[arm] = {"wrong_to_right": sum(b == 0 and f == 1 for b, f in zip(base, final)),
                            "right_to_wrong": sum(b == 1 and f == 0 for b, f in zip(base, final)),
                            "accuracy_difference": sum(changes) / len(rows),
                            "difference_95ci_filing_bootstrap": cluster_interval(rows, changes)}
    # Independent benchmark reference scoring of first-draft validator decisions.
    # This is not a human audit of evidence support or a proof of financial truth.
    tp = fp = tn = fn = malformed = first_accepts = 0
    for row, hit in zip(rows, outcomes["single_shot"]):
        first = row["arms"]["agentic"]["attempts"][0]
        rejected = not first["accepted"]
        first_accepts += not rejected
        malformed += first["verdict"] is None
        tp += rejected and not hit
        fp += rejected and bool(hit)
        tn += not rejected and bool(hit)
        fn += not rejected and not hit
    validation = {"evaluated": len(rows), "true_reject_wrong": tp, "false_reject_correct": fp,
                  "true_accept_correct": tn, "false_accept_wrong": fn,
                  "error_detection_precision": fraction(tp, tp + fp),
                  "error_detection_recall": fraction(tp, tp + fn),
                  "false_acceptance_rate_among_wrong": fraction(fn, tp + fn),
                  "false_rejection_rate_among_correct": fraction(fp, fp + tn),
                  "first_attempt_consistency_score": first_accepts / len(rows),
                  "malformed_validator_outputs": malformed}
    calls = jsonl(folder / "calls.jsonl")
    costs = [c.get("usage", {}).get("cost") for c in calls]
    report = {"experiment": "New reconstruction; not recovered original results", "split": split,
              "arms": summary, "paired_transitions": transitions, "validator_reference_evaluation": validation,
              "billing": {"unique_calls": len(calls), "known_reported_cost_usd": sum(float(c) for c in costs if c is not None),
                          "calls_missing_cost": sum(c is None for c in costs)},
              "limitations": ["Small exploratory sample; no guaranteed improvement.",
                  "FinQA report-page corpus, not complete PDFs or 500 EDGAR filings.",
                  "Hosted API latency/cost are not local A100 latency or self-hosted deployment evidence.",
                  "Numerical F1 uses final numerical scalars (excluding Boolean items); original metric code was lost, so exact historical metric equivalence is unverified.",
                  "Benchmark labels may contain errors; no independent human grounding audit is claimed.",
                  "Same-model verification can accept incorrect answers and reject correct answers.",
                  "No ConvFinQA histories, injected errors, dependency graphs or CL result reuse."]}
    write_json(folder / "summary.json", report)
    def fmt(value):
        return "undefined (zero denominator)" if value is None else f"{value:.3f}"
    lines = ["# Reconstructed FinQA experiment", "", "New run; not a reproduction of the lost numerical results.", "",
             "| Method | n | Normalized EM | Scalar Numerical F1 | Relaxed accuracy | Mean attempts |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in summary.items():
        lines.append(f"| {name} | {result['n']} | {fmt(result['strict_numerical_accuracy'])} | {fmt(result['numerical_f1_final_scalar'])} | {fmt(result['relaxed_0.5pct_accuracy'])} | {result['mean_attempts']:.2f} |")
    lines += ["", "## Paired changes", ""]
    for name, result in transitions.items():
        lines.append(f"- {name}: corrected {result['wrong_to_right']}; corrupted {result['right_to_wrong']}; accuracy difference {result['accuracy_difference']:+.3f}; filing-bootstrap 95% interval {result['difference_95ci_filing_bootstrap']}.")
    lines += ["", "## Validator against reference answers", ""]
    for name, value in validation.items():
        lines.append(f"- {name}: {fmt(value)}" if isinstance(value, float) or value is None else f"- {name}: {value}")
    lines += ["", "## Limitations", ""] + ["- " + t for t in report["limitations"]]
    lines += ["", "Accuracy counts missing, malformed, truncated and unknown answers as incorrect. Percent signs explicitly convert to decimals; bare factor-of-100 mismatches are not accepted. Each answer is evaluated on the same question and frozen reference. Confidence intervals resample filings, with seed 18 and 2,000 draws.",
              "", "API trajectory time sums stored call durations, including a shared first draft. It excludes embedding/index construction and is not an end-to-end latency benchmark.",
              "", "Known API charge: $" + f"{report['billing']['known_reported_cost_usd']:.4f}"]
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    with (folder / "audit.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "reference", "single_shot", "agentic", "baseline_correct", "agentic_correct", "human_grounding_review", "notes"])
        for row, baseline, final in zip(rows, outcomes["single_shot"], outcomes["agentic"]):
            values = [(row["arms"][arm]["final"] or {}).get("value", "") for arm in ("single_shot", "agentic")]
            writer.writerow([row["id"], row["question"], labels[row["id"]]["gold"], *values, baseline, final, "NOT REVIEWED", ""])
    # Export every actual refinement case, including unsuccessful ones; never select only successes.
    with (folder / "refinement-traces.jsonl").open("w") as f:
        for row in rows:
            if len(row["arms"]["agentic"]["attempts"]) > 1:
                from .core import canonical
                f.write(canonical(row) + "\n")
    return report


def compare_runs(run_dirs, output):
    loaded = [(Path(p), read_json(Path(p) / "manifest.json"), read_json(Path(p) / "summary.json")) for p in run_dirs]
    if len(loaded) < 2:
        raise ValueError("Provide at least two completed model runs")
    first = loaded[0][1]["frozen"]
    for path, manifest, summary in loaded:
        frozen = manifest["frozen"]
        if manifest.get("status") != "inference_complete" or frozen["split"] != "evaluation":
            raise ValueError("Compare complete evaluation runs, not pilots")
        for field in ("data_hash", "index_hash", "prompt_hash", "source_hash"):
            if frozen[field] != first[field]:
                raise ValueError(f"Runs differ in {field}; do not present them as controlled model comparisons")
        for field in ("chunk_tokens", "overlap_tokens", "top_k", "max_attempts", "seed", "temperature", "max_tokens", "reasoning"):
            if frozen["config"][field] != first["config"][field]:
                raise ValueError(f"Runs differ in {field}; label and resolve the confound before comparing")
    lines = ["# Models on the same frozen FinQA evaluation", "",
             "New experiment results; no historical DeepSeek scores reused.", "",
             "| Model | Provider | n | Single-shot | Agentic | Fixed retrieval | Corrected | Corrupted |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for path, manifest, summary in loaded:
        config = manifest["frozen"]["config"]
        arms, change = summary["arms"], summary["paired_transitions"]["agentic"]
        fixed = arms.get("fixed_retrieval", {}).get("strict_numerical_accuracy")
        fixed_text = "n/a" if fixed is None else f"{fixed:.3f}"
        lines.append(f"| {config['model']} | {config['provider']} | {arms['single_shot']['n']} | {arms['single_shot']['strict_numerical_accuracy']:.3f} | {arms['agentic']['strict_numerical_accuracy']:.3f} | {fixed_text} | {change['wrong_to_right']} | {change['right_to_wrong']} |")
    lines += ["", "Values are normalized five-decimal numerical accuracy. Model family, size, training and hosting quantization remain model-level differences. Refer to each model's report for filing-cluster intervals, scalar Numerical F1, Consistency Score, validator errors and limitations. A larger point estimate alone is not proof of superiority."]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n")
