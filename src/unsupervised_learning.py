"""Task 8 (Lab 10): Unsupervised Learning - K-Means Clustering for Feeder & Demand Profiling.

Implements pure Python & NumPy K-Means clustering from scratch (Lloyd's Algorithm + K-Means++ init).
Applies clustering to segment city electrical sectors into archetypal operational tiers:
- Tier 0: Critical Infrastructure (Hospitals, 0 outage)
- Tier 1: Heavy Industrial Engines (High economic impact)
- Tier 2: Urban Commercial & Dense Residential (Standard shedding candidates)
- Tier 3: Suburban & Flexible Agricultural (Peak evening/night shedding tolerance)
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from src.city_grid import CityGridNetwork


@dataclass
class ClusteringProfileResult:
    """Diagnostic outcome of unsupervised K-Means clustering."""

    k: int
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    inertia: float
    feature_names: List[str]
    feeder_profiles_df: pd.DataFrame


class ScratchKMeans:
    """K-Means Clustering implemented from scratch using Lloyd's Algorithm.

    Includes K-Means++ smart centroid initialization and within-cluster sum of squares (inertia).
    """

    def __init__(
        self,
        n_clusters: int = 4,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
    ):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0

    def _init_centroids_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ initialization: Selects seeds with probability proportional to D(x)^2."""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))

        # First centroid chosen uniformly at random
        first_idx = self.rng.integers(0, n_samples)
        centroids[0] = X[first_idx]

        for c_idx in range(1, self.k):
            # Compute squared distances to nearest existing centroid
            dist_sq = np.min(
                [np.sum((X - centroids[j]) ** 2, axis=1) for j in range(c_idx)],
                axis=0,
            )
            total_dist = np.sum(dist_sq)
            if total_dist == 0:
                probs = np.full(n_samples, 1.0 / n_samples)
            else:
                probs = dist_sq / total_dist

            next_idx = self.rng.choice(n_samples, p=probs)
            centroids[c_idx] = X[next_idx]

        return centroids

    def fit(self, X: np.ndarray) -> "ScratchKMeans":
        """Iterative Expectation-Maximization loop."""
        n_samples = X.shape[0]
        self.centroids_ = self._init_centroids_kmeans_plus_plus(X)

        for _ in range(self.max_iter):
            # 1. Assignment Step: assign each sample to closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
            new_labels = np.argmin(distances, axis=1)

            # 2. Update Step: compute new centroids as mean of assigned points
            new_centroids = np.zeros_like(self.centroids_)
            for j in range(self.k):
                cluster_points = X[new_labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = np.mean(cluster_points, axis=0)
                else:
                    # If empty cluster, re-seed randomly
                    new_centroids[j] = X[self.rng.integers(0, n_samples)]

            # Check convergence
            shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            self.labels_ = new_labels

            if shift < self.tol:
                break

        # Compute final inertia (Within-Cluster Sum of Squares)
        assert self.centroids_ is not None and self.labels_ is not None
        self.inertia_ = 0.0
        for j in range(self.k):
            cluster_pts = X[self.labels_ == j]
            if len(cluster_pts) > 0:
                self.inertia_ += float(np.sum((cluster_pts - self.centroids_[j]) ** 2))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.centroids_ is not None, "Model not fitted."
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(distances, axis=1)


def cluster_city_feeders(
    grid_network: Optional[CityGridNetwork] = None,
    k_clusters: int = 4,
    seed: int = 42,
) -> ClusteringProfileResult:
    """Applies scratch K-Means to profile and cluster city distribution feeders."""
    grid = grid_network if grid_network is not None else CityGridNetwork()
    feeders = grid.feeders

    # Build feature matrix for each feeder
    feature_names = [
        "base_load_mw",
        "disruption_cost_per_hour",
        "is_critical_flag",
        "max_daily_quota_hours",
        "priority_tier",
    ]

    records = []
    for f in feeders:
        records.append(
            {
                "area_id": f.area_id,
                "name": f.name,
                "category": f.category,
                "base_load_mw": f.base_load_mw,
                "disruption_cost_per_hour": f.disruption_cost_per_hour,
                "is_critical_flag": 1.0 if f.is_critical_exempt else 0.0,
                "max_daily_quota_hours": float(f.max_daily_outage_hours),
                "priority_tier": float(f.priority_tier),
            }
        )

    df = pd.DataFrame(records)
    X_raw = df[feature_names].values

    # Normalize features before clustering.
    feature_mean = np.mean(X_raw, axis=0)
    feature_scale = np.std(X_raw, axis=0)
    feature_scale = np.where(feature_scale == 0.0, 1.0, feature_scale)
    X_scaled = (X_raw - feature_mean) / feature_scale

    kmeans = ScratchKMeans(n_clusters=min(k_clusters, len(feeders)), seed=seed)
    kmeans.fit(X_scaled)

    df["cluster_id"] = kmeans.labels_

    # Map cluster numbers to meaningful human labels
    cluster_labels = []
    for cid in kmeans.labels_:
        cluster_rows = df[df["cluster_id"] == cid]
        if cluster_rows["is_critical_flag"].max() > 0.5:
            cluster_labels.append("Critical Emergency Infrastructure (Exempt)")
        elif cluster_rows["disruption_cost_per_hour"].mean() > 300.0:
            cluster_labels.append("Strategic Industrial & Economic Engine")
        elif cluster_rows["base_load_mw"].mean() > 280.0:
            cluster_rows_name = "Urban Commercial / Dense Residential"
            cluster_labels.append(cluster_rows_name)
        else:
            cluster_labels.append("Suburban & Flexible Agricultural")

    df["profile_archetype"] = cluster_labels

    return ClusteringProfileResult(
        k=kmeans.k,
        cluster_labels=kmeans.labels_,
        cluster_centers=kmeans.centroids_,
        inertia=round(kmeans.inertia_, 2),
        feature_names=feature_names,
        feeder_profiles_df=df,
    )
