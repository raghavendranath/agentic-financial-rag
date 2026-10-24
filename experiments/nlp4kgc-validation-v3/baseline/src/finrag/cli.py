from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from .client import OpenRouterClient, preflight, read_key
from .core import read_json, write_json
from .data import prepare


def config_at(path):
    config = read_json(path)
    for key in ("chunk_tokens", "top_k", "max_attempts", "max_tokens"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_attempts"] > 3:
        raise ValueError("Paper reconstruction allows at most three attempts")
    if not 0 <= config["overlap_tokens"] < config["chunk_tokens"]:
        raise ValueError("Invalid chunk overlap")
    if config["budget_usd"] <= 0:
        raise ValueError("Budget must be positive")
    if config["embedding_model"] != "BAAI/bge-m3":
        raise ValueError("This paper-aligned reconstruction requires BGE-M3")
    return config


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned BGE-M3/FAISS financial RAG experiment")
    parser.add_argument("--config", default="configs/qwen-paper-model.json")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Download and freeze FinQA inputs and separate evaluation labels")
    p.add_argument("--data", default="data/finqa50-v1")
    p.add_argument("--questions", type=int, default=50)
    p.add_argument("--pilot-questions", type=int, default=5)
    p = commands.add_parser("index", help="Build the real BGE-M3 1024d FAISS index")
    p.add_argument("--data", default="data/finqa50-v1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p = commands.add_parser("preflight", help="Check exact model/provider and prices; no inference charge")
    p.add_argument("--key-file", default=None)
    p = commands.add_parser("run", help="Run paired inference; reads no evaluation reference labels")
    p.add_argument("--data", default="data/finqa50-v1")
    p.add_argument("--split", choices=["pilot", "evaluation"], default="pilot")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    p.add_argument("--key-file", default=".secrets/openrouter.key")
    p.add_argument("--device", default="cpu")
    p = commands.add_parser("report", help="Score frozen outputs against reference answers, offline")
    p.add_argument("--data", default="data/finqa50-v1")
    p.add_argument("--split", choices=["pilot", "evaluation"], default="pilot")
    p.add_argument("--run", required=True)
    p = commands.add_parser("compare", help="Combine completed models only when data, prompts and architecture match")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="runs/MODEL-COMPARISON.md")
    p = commands.add_parser("ingest-pdfs", help="Create a reusable corpus from local PDFs, without modifying them")
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    config = config_at(args.config)
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "prepare":
        result = prepare(args.data, args.questions, args.pilot_questions, config["seed"])
        print(f"Frozen {result['splits']['evaluation']['n']} evaluation questions and {result['files']} report pages")
    elif args.command == "index":
        from .dense import DenseIndex
        data = Path(args.data)
        manifest = read_json(data / "manifest.json")
        from .core import digest
        corpus = read_json(data / "corpus.json")
        if digest(corpus) != manifest["corpus_hash"]:
            raise ValueError("Frozen source corpus changed")
        existing = data / "index/manifest.json"
        revision = read_json(existing)["parameters"]["revision"] if existing.exists() else None
        index = DenseIndex(config["embedding_model"], revision=revision, device=args.device)
        result = index.build(corpus, data / "index", config["chunk_tokens"], config["overlap_tokens"], args.batch_size)
        print(f"Index ready: {result['chunks']} chunks, {result['dimension']} dimensions, revision {result['parameters']['revision']}")
    elif args.command == "preflight":
        key = read_key(args.key_file) if args.key_file else None
        result = preflight(config, key)
        print(f"Available: {result['model']} on {result['provider']}; prices per token {result['pricing']}")
    elif args.command == "run":
        from .dense import DenseIndex
        from .experiment import run_experiment
        key = read_key(args.key_file)
        metadata = preflight(config, key)
        write_json(Path(args.out) / "latest-preflight.json", metadata)
        index_manifest = read_json(Path(args.data) / "index/manifest.json")
        index = DenseIndex(config["embedding_model"], revision=index_manifest["parameters"]["revision"], device=args.device)
        index.load(Path(args.data) / "index")
        client = OpenRouterClient(config, args.out, key, metadata)
        run_experiment(client, index, args.data, args.split, args.out, config, metadata, workers=args.workers)
        print("Inference complete. Run the offline report command to score it.")
    elif args.command == "report":
        from .report import make_report
        result = make_report(args.data, args.split, args.run)
        print(f"Report ready for {result['arms']['single_shot']['n']} paired questions in {args.run}")
    elif args.command == "compare":
        from .report import compare_runs
        compare_runs(args.runs, args.out)
        print(f"Model comparison saved to {args.out}")
    elif args.command == "ingest-pdfs":
        import pdfplumber
        from pypdf import PdfReader
        docs = []
        for source in args.paths:
            source = Path(source)
            reader = PdfReader(source)
            with pdfplumber.open(source) as pdf:
                for i, (page, layout_page) in enumerate(zip(reader.pages, pdf.pages), 1):
                    text = page.extract_text() or ""
                    tables = layout_page.extract_tables()
                    table_text = "\n\n".join("\n".join(" | ".join(str(c or "") for c in row) for row in t) for t in tables)
                    docs.append({"filename": source.stem + "/page_" + str(i), "text": text + "\n\n" + table_text})
        write_json(args.out, docs)
        print(f"Created corpus with {len(docs)} PDF pages. Extraction needs visual checking before research use.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
