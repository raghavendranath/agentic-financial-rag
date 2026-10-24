"""Freeze fresh questions; exclude every v1 pilot/evaluation filing, not just IDs."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finrag.core import digest, number, public_item, read_json, write_json
from finrag.data import document, FINQA_COMMIT


def prepare():
    original = ROOT.parent / "nlp4kgc-reconstruction" / "data/finqa50-v1"
    folder = ROOT / "data/fresh50"
    if (folder / "manifest.json").exists():
        raise ValueError("Fresh sample already frozen")
    source = {s: read_json(original / (s + ".source.json")) for s in ("dev", "test")}
    old_inputs = sum((read_json(original / (s + ".inputs.json")) for s in ("pilot", "evaluation")), [])
    excluded = {r["filename"].rsplit("/", 1)[0] for r in old_inputs}
    used = set(excluded)
    previous = ROOT.parent / "nlp4kgc-validation-v2/data/fresh100"
    for name in ("pilot", "evaluation"):
        excluded.update(r["filename"].rsplit("/", 1)[0] for r in read_json(previous / (name + ".inputs.json")))
    used = set(excluded)
    chosen = {}
    for split, count in (("dev", 20), ("test", 50)):
        selected = []
        for row in sorted(source[split], key=lambda r: digest([1820, r["id"]])):
            filing = row["filename"].rsplit("/", 1)[0]
            if filing in used or number(row["qa"].get("exe_ans")) is None or not public_item(row)["chunks"]:
                continue
            selected.append(row)
            used.add(filing)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError("Not enough disjoint filings")
        chosen[split] = selected
    filings = used - excluded
    documents = {}
    for rows in source.values():
        for row in rows:
            if row["filename"].rsplit("/", 1)[0] in filings:
                doc = document(row)
                if row["filename"] in documents and documents[row["filename"]] != doc:
                    raise ValueError("Conflicting page text")
                documents[row["filename"]] = doc
    corpus = sorted(documents.values(), key=lambda d: d["filename"])
    write_json(folder / "corpus.json", corpus)
    manifest = {"dataset": "FinQA", "commit": FINQA_COMMIT, "seed": 1820,
                "sampling": "SHA256(1820,id); one question per filing; exclude all v1/v2 observed filings; 20 dev and 50 test",
                "excluded_v1_filings": sorted(excluded), "corpus_hash": digest(corpus),
                "corpus_scope": "Processed dev/test pages, known-filing retrieval; not full PDFs",
                "files": len(corpus), "splits": {},
                "source_sha256": {s: hashlib.sha256((original / (s + ".source.json")).read_bytes()).hexdigest()
                                  for s in source}}
    for name, split in (("pilot", "dev"), ("evaluation", "test")):
        rows = chosen[split]
        inputs = [{k: v for k, v in public_item(r).items() if k != "chunks"} for r in rows]
        labels = {r["id"]: {"gold": r["qa"]["exe_ans"], "display_answer": r["qa"]["answer"],
                             "gold_program": r["qa"]["program"]} for r in rows}
        write_json(folder / (name + ".inputs.json"), inputs)
        write_json(folder / (name + ".labels.json"), labels)
        manifest["splits"][name] = {"source_split": split, "n": len(rows),
                                    "inputs_hash": digest(inputs), "labels_hash": digest(labels),
                                    "ids": [r["id"] for r in rows]}
    write_json(folder / "manifest.json", manifest)
    print(f"Frozen 20 development + 50 fresh test questions in {len(corpus)} pages; all v1 filings excluded")


if __name__ == "__main__":
    prepare()
