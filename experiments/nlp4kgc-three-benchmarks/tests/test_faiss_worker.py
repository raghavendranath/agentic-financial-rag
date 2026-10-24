import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finrag.dense import DenseIndex


class NativeFaissTests(unittest.TestCase):
    def test_native_index_persists_exact_inner_products(self):
        # Real FAISS, not a retrieval mock; validates normalized IP and file round-trip.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            vectors = np.array([[1, 0], [0, 1], [.6, .8]], dtype="float32")
            query = np.array([[1, 0]], dtype="float32")
            np.save(folder / "vectors.npy", vectors)
            np.save(folder / "query.npy", query)
            index = object.__new__(DenseIndex)
            info = index.worker("build", folder / "vectors.npy", folder / "index.faiss")
            self.assertEqual(info, {"count": 3, "dimension": 2})
            self.assertEqual(index.worker("info", folder / "index.faiss"), info)
            found = index.worker("search", folder / "index.faiss", folder / "query.npy", 3)
            self.assertEqual([i for _, i in found], [0, 2, 1])
            np.testing.assert_allclose([s for s, _ in found], [1, .6, 0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
