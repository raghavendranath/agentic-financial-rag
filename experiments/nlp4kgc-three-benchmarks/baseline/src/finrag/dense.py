from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
from .core import digest, read_json, write_json


class DenseIndex:
    """Paper architecture: BGE-M3 dense (1024d), normalized FAISS IndexFlatIP."""

    def __init__(self, model="BAAI/bge-m3", revision=None, device=None):
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import model_info
        # Keep FAISS native libraries out of the PyTorch process. macOS wheels
        # otherwise load competing OpenMP runtimes; do not bypass their safety check.
        self.np = np
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        self.model_id = model
        revision = revision or model_info(model).sha
        self.encoder = SentenceTransformer(model, revision=revision, device=device, trust_remote_code=False)
        self.tokenizer = self.encoder.tokenizer
        dimension_method = getattr(self.encoder, "get_embedding_dimension", None) or self.encoder.get_sentence_embedding_dimension
        self.dimension = dimension_method()
        if model == "BAAI/bge-m3" and self.dimension != 1024:
            raise ValueError("BGE-M3 dimension must be 1024")
        self.resolved_revision = getattr(self.encoder[0].auto_model.config, "_commit_hash", None)
        if not self.resolved_revision:
            self.resolved_revision = revision
        if not self.resolved_revision:
            raise ValueError("Could not identify embedding model revision; freeze it explicitly")
        self.index = None
        self.count = 0
        self.chunks = []

    def worker(self, operation, *paths):
        env = os.environ.copy()
        env.pop("OPENROUTER_API_KEY", None)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        result = subprocess.run([sys.executable, "-m", "finrag.faiss_worker", operation, *map(str, paths)],
                                text=True, capture_output=True, env=env)
        if result.returncode:
            raise RuntimeError("Isolated FAISS worker failed: " + result.stderr[-2000:])
        return json.loads(result.stdout)

    def chunk(self, filename, text, chunk_tokens=1500, overlap_tokens=200):
        if not 0 <= overlap_tokens < chunk_tokens:
            raise ValueError("Overlap must be smaller than chunk size")
        # Offset mapping preserves original table labels, numeric formatting and text.
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offsets = encoded["offset_mapping"]
        out = []
        for start in range(0, len(offsets), chunk_tokens - overlap_tokens):
            stop = min(start + chunk_tokens, len(offsets))
            a, b = offsets[start][0], offsets[stop - 1][1]
            out.append({"id": filename + "::" + str(start), "filename": filename,
                        "text": text[a:b], "token_start": start, "token_end": stop})
            if stop == len(offsets):
                break
        return out

    def build(self, documents, folder, chunk_tokens=1500, overlap_tokens=200, batch_size=2):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        source_hash = digest(documents)
        parameters = {"source_hash": source_hash, "model": self.model_id,
                      "revision": self.resolved_revision, "chunk_tokens": chunk_tokens,
                      "overlap_tokens": overlap_tokens, "normalized": True, "index": "IndexFlatIP"}
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["parameters"] != parameters:
                raise ValueError("Index inputs/settings changed; use a new index folder")
            self.chunks = read_json(folder / "chunks.json")
            index_bytes = (folder / "index.faiss").read_bytes()
            import hashlib
            if (digest(self.chunks) != manifest["chunks_hash"] or
                    hashlib.sha256(index_bytes).hexdigest() != manifest["index_sha256"]):
                raise ValueError("Index integrity check failed")
            self.index = folder / "index.faiss"
            self.count = len(self.chunks)
            return manifest
        self.chunks = [c for d in documents for c in self.chunk(d["filename"], d["text"], chunk_tokens, overlap_tokens)]
        if not self.chunks:
            raise ValueError("Empty corpus")
        batches = []
        for start in range(0, len(self.chunks), batch_size):
            batch = self.chunks[start:start + batch_size]
            vectors = self.encoder.encode([c["text"] for c in batch], normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
            batches.append(self.np.ascontiguousarray(vectors, dtype="float32"))
            if start % (10 * batch_size) == 0:
                print(f"Embedded {min(start + batch_size, len(self.chunks))}/{len(self.chunks)} chunks", flush=True)
        with tempfile.TemporaryDirectory(prefix="finrag-faiss-") as temp:
            vector_path = Path(temp) / "vectors.npy"
            self.np.save(vector_path, self.np.concatenate(batches))
            self.worker("build", vector_path, folder / "index.faiss")
        self.index = folder / "index.faiss"
        self.count = len(self.chunks)
        write_json(folder / "chunks.json", self.chunks)
        import hashlib
        manifest = {"parameters": parameters, "dimension": self.dimension, "chunks": len(self.chunks),
                    "chunks_hash": digest(self.chunks),
                    "index_sha256": hashlib.sha256((folder / "index.faiss").read_bytes()).hexdigest()}
        write_json(manifest_path, manifest)
        return manifest

    def load(self, folder):
        folder = Path(folder)
        manifest = read_json(folder / "manifest.json")
        if manifest["parameters"]["model"] != self.model_id or manifest["parameters"]["revision"] != self.resolved_revision:
            raise ValueError("Query encoder does not match stored index")
        import hashlib
        self.chunks = read_json(folder / "chunks.json")
        if digest(self.chunks) != manifest["chunks_hash"] or hashlib.sha256((folder / "index.faiss").read_bytes()).hexdigest() != manifest["index_sha256"]:
            raise ValueError("Index integrity check failed")
        self.index = folder / "index.faiss"
        info = self.worker("info", self.index)
        self.count = info["count"]
        if info["dimension"] != self.dimension or self.count != len(self.chunks):
            raise ValueError("Index dimensionality/count mismatch")
        return manifest

    def retrieve(self, query, top_k=10, filename=None):
        if self.index is None:
            raise ValueError("Index not loaded")
        vector = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)
        k = len(self.chunks) if filename else min(top_k, len(self.chunks))
        with tempfile.TemporaryDirectory(prefix="finrag-faiss-query-") as temp:
            vector_path = Path(temp) / "query.npy"
            self.np.save(vector_path, self.np.ascontiguousarray(vector, dtype="float32"))
            found = self.worker("search", self.index, vector_path, k)
        result = [{**self.chunks[int(i)], "score": float(s)} for s, i in found if i >= 0]
        if filename:
            filing = filename.rsplit("/", 1)[0]
            result = [c for c in result if c["filename"].rsplit("/", 1)[0] == filing]
        return result[:top_k]
