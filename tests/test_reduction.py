import unittest
from unittest.mock import patch

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

from app.pipeline.reduction import Float32IncrementalPCA


class Float32IncrementalPCATests(unittest.TestCase):
    def test_reduced_dtype_changes_do_not_break_kmeans(self):
        rng = np.random.default_rng(42)
        reducer = Float32IncrementalPCA(n_components=4)
        clusterer = MiniBatchKMeans(n_clusters=3, random_state=42)
        # Reproduce the boundary condition regardless of installed sklearn version.
        for dtype in (np.float32, np.float64, np.float32):
            reduced = np.asfortranarray(rng.normal(size=(32, 4)), dtype=dtype)
            with patch.object(IncrementalPCA, "transform", return_value=reduced):
                actual = reducer.transform(np.zeros((32, 8)))
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(actual.flags.c_contiguous)
            np.testing.assert_allclose(actual, reduced, rtol=1e-6)
            clusterer.partial_fit(actual)
            self.assertEqual(clusterer.predict(actual).shape, (32,))

    def test_multiple_real_batches_then_prediction(self):
        rng = np.random.default_rng(7)
        batches = [rng.normal(size=(32, 8)).astype(np.float32) for _ in range(4)]
        reducer = Float32IncrementalPCA(n_components=4)
        clusterer = MiniBatchKMeans(n_clusters=3, random_state=42)
        for batch in batches:
            reducer.partial_fit(batch)
            reduced = reducer.transform(batch)
            self.assertEqual(reduced.dtype, np.float32)
            clusterer.partial_fit(reduced)
            self.assertEqual(clusterer.cluster_centers_.dtype, np.float32)
        for batch in batches:
            labels = clusterer.predict(reducer.transform(batch))
            self.assertEqual(labels.shape, (len(batch),))

    def test_fit_transform_preserves_pca_result(self):
        data = np.random.default_rng(9).normal(size=(32, 8))
        expected = IncrementalPCA(n_components=4).fit_transform(data)
        actual = Float32IncrementalPCA(n_components=4).fit_transform(data)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
