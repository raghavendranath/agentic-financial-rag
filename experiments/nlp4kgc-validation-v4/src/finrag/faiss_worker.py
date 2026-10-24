"""FAISS only, no torch imports: isolate native OpenMP runtimes on macOS."""
import json
import sys
import faiss
import numpy as np


def main():
    faiss.omp_set_num_threads(1)
    operation, *args = sys.argv[1:]
    if operation == "build":
        vectors = np.load(args[0], allow_pickle=False)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding matrix")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        faiss.write_index(index, args[1])
        print(json.dumps({"count": index.ntotal, "dimension": index.d}))
    elif operation == "info":
        index = faiss.read_index(args[0])
        print(json.dumps({"count": index.ntotal, "dimension": index.d}))
    elif operation == "search":
        index = faiss.read_index(args[0])
        query = np.load(args[1], allow_pickle=False)
        if query.shape != (1, index.d):
            raise ValueError("Query dimension mismatch")
        scores, positions = index.search(np.ascontiguousarray(query, dtype="float32"), int(args[2]))
        print(json.dumps([[float(s), int(i)] for s, i in zip(scores[0], positions[0])]))
    else:
        raise ValueError("Unknown FAISS operation")


if __name__ == "__main__":
    main()
