"""Incremental dimensionality reduction with a stable clustering input dtype."""

import numpy as np
from sklearn.decomposition import IncrementalPCA


class Float32IncrementalPCA(IncrementalPCA):
    """Keep reduced batches compatible with MiniBatchKMeans across updates."""

    def transform(self, X):
        # PCA accumulation can promote later batches to float64 while KMeans
        # retains float32 centers initialized from the first batch.
        return np.ascontiguousarray(super().transform(X), dtype=np.float32)
