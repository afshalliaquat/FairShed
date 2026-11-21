"""Unit tests for clustering and fairness metrics."""

import pytest
import numpy as np
from src.unsupervised_learning import ScratchKMeans
from src.fairness_metrics import (
    compute_jains_fairness_index,
    compute_gini_coefficient,
)


def test_scratch_kmeans_clustering():
    np.random.seed(42)
    # Generate 3 distinct clusters
    c1 = np.random.randn(20, 2) + np.array([0, 0])
    c2 = np.random.randn(20, 2) + np.array([10, 10])
    c3 = np.random.randn(20, 2) + np.array([-10, 10])
    X = np.vstack([c1, c2, c3])

    kmeans = ScratchKMeans(n_clusters=3, seed=42)
    kmeans.fit(X)
    assert kmeans.centroids_.shape == (3, 2)
    assert len(np.unique(kmeans.labels_)) == 3
    assert kmeans.inertia_ > 0.0


def test_jains_fairness_and_gini_indices():
    # Perfectly fair distribution: all areas get identical outage hours
    perfect_equal = [3.0, 3.0, 3.0, 3.0]
    assert compute_jains_fairness_index(perfect_equal) == pytest.approx(1.0, rel=1e-3)
    assert compute_gini_coefficient(perfect_equal) == pytest.approx(0.0, abs=1e-3)

    # Highly unfair distribution: 1 area takes all outages
    unfair = [12.0, 0.0, 0.0, 0.0]
    assert compute_jains_fairness_index(unfair) == pytest.approx(1.0 / 4.0, rel=1e-3)
    assert compute_gini_coefficient(unfair) > 0.70
