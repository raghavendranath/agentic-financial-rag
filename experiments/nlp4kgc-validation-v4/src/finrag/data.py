from __future__ import annotations

import hashlib
from pathlib import Path
from .client import fetch_json
from .core import digest, public_item, read_json, select, write_json

FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
FINQA_ROOT = "https://raw.githubusercontent.com/czyssrs/FinQA/" + FINQA_COMMIT


def document(row):
    item = public_item(row)
    # Preserve row labels with headers; strip QA, gold evidence, and all supplied retrieval annotations.
    text = "\n\n".join("[" + c["id"] + "] " + c["text"] for c in item["chunks"])
    return {"filename": row["filename"], "text": text}


def prepare(folder, count=50, pilot_count=5, seed=18):
    if count <= 0 or pilot_count <= 0:
        raise ValueError("Sample sizes must be positive")
    folder = Path(folder)
    if (folder / "manifest.json").exists():
        raise ValueError("Dataset already frozen; use a new folder to change the sample")
    folder.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ("dev", "test"):
        url = FINQA_ROOT + "/dataset/" + split + ".json"
        raw = fetch_json(url)
        splits[split] = raw
        write_json(folder / (split + ".source.json"), raw)
    pilot = select(splits["dev"], pilot_count, seed)
    # Keep entire filings (not just pages) disjoint between the pilot and evaluation sample.
    pilot_filings = {r["filename"].rsplit("/", 1)[0] for r in pilot}
    eligible = [r for r in splits["test"] if r["filename"].rsplit("/", 1)[0] not in pilot_filings]
    evaluation = select(eligible, count, seed)
    selected_filings = {r["filename"].rsplit("/", 1)[0] for r in pilot + evaluation}
    by_file = {}
    for split in ("dev", "test"):
        for row in splits[split]:
            if row["filename"].rsplit("/", 1)[0] in selected_filings:
                doc = document(row)
                if row["filename"] in by_file and by_file[row["filename"]] != doc:
                    raise ValueError("Same report page has conflicting source text; resolve explicitly")
                by_file[row["filename"]] = doc
    corpus = sorted(by_file.values(), key=lambda d: d["filename"])
    write_json(folder / "corpus.json", corpus)
    manifest = {"dataset": "FinQA", "commit": FINQA_COMMIT, "seed": seed,
                "sampling": "SHA256(seed,id), one question per report page; pilot/eval filings disjoint",
                "corpus_scope": "All available dev/test report pages for selected filings, with no QA labels; retrieval scoped to known filing; not complete 10-Ks",
                "corpus_hash": digest(corpus), "files": len(corpus), "splits": {}}
    for name, rows, source_split in (("pilot", pilot, "dev"), ("evaluation", evaluation, "test")):
        public = [{k: v for k, v in public_item(r).items() if k != "chunks"} for r in rows]
        labels = {r["id"]: {"gold": r["qa"]["exe_ans"], "display_answer": r["qa"]["answer"],
                               "gold_program": r["qa"]["program"]} for r in rows}
        write_json(folder / (name + ".inputs.json"), public)
        write_json(folder / (name + ".labels.json"), labels)
        manifest["splits"][name] = {"source_split": source_split, "n": len(rows),
                                     "inputs_hash": digest(public), "labels_hash": digest(labels),
                                     "ids": [r["id"] for r in rows]}
    manifest["source_sha256"] = {split: hashlib.sha256((folder / (split + ".source.json")).read_bytes()).hexdigest()
                                   for split in ("dev", "test")}
    write_json(folder / "manifest.json", manifest)
    return manifest


def load_inputs(folder, split):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    inputs = read_json(folder / (split + ".inputs.json"))
    if digest(inputs) != manifest["splits"][split]["inputs_hash"]:
        raise ValueError("Frozen evaluation inputs changed")
    return inputs, manifest
