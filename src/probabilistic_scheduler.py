"""Discrete Bayesian and Markov models for shortfall-risk-aware scheduling."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.city_grid import CityGridNetwork

SHORTFALL_STATE_NAMES = ("none", "low", "medium", "high")
SHORTFALL_THRESHOLDS_MW = (15.0, 800.0, 1800.0)
SHORTFALL_STATE_FALLBACK_MW = np.array([0.0, 407.5, 1300.0, 2300.0])
TRAINING_CAPACITY_RATIOS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
BASE_CAPACITY_RATIO = 0.82


def _shortfall_states(shortfall_mw: np.ndarray) -> np.ndarray:
    return np.digitize(shortfall_mw, SHORTFALL_THRESHOLDS_MW).astype(int)


@dataclass
class ProbabilisticShortfallForecast:
    """Hourly shortfall expectations and their Markov/Bayesian state probabilities."""

    expected_shortfall_mw: np.ndarray
    state_probabilities: np.ndarray
    markov_prior_probabilities: np.ndarray

    def summary(self) -> Dict[str, float]:
        bayesian_probability = np.sum(self.state_probabilities[:, 1:], axis=1)
        markov_probability = np.sum(self.markov_prior_probabilities[:, 1:], axis=1)
        return {
            "bayesian_shortfall_probability_pct": round(
                float(np.mean(bayesian_probability) * 100.0), 1
            ),
            "markov_shortfall_probability_pct": round(
                float(np.mean(markov_probability) * 100.0), 1
            ),
            "mean_expected_shortfall_mw": round(
                float(np.mean(self.expected_shortfall_mw)), 1
            ),
            "peak_expected_shortfall_mw": round(
                float(np.max(self.expected_shortfall_mw)), 1
            ),
        }


class MarkovShortfallModel:
    """First-order Markov chain over discretized hourly shortfall states."""

    def __init__(self, n_states: int = len(SHORTFALL_STATE_NAMES), alpha: float = 1.0):
        if n_states != len(SHORTFALL_STATE_NAMES):
            raise ValueError(f"n_states must be {len(SHORTFALL_STATE_NAMES)}.")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        self.n_states = n_states
        self.alpha = alpha
        self.transition_matrix: Optional[np.ndarray] = None
        self.last_state: Optional[int] = None

    def fit(self, shortfall_mw: Sequence[float]) -> "MarkovShortfallModel":
        shortfall = np.asarray(shortfall_mw, dtype=float)
        if shortfall.ndim != 1 or len(shortfall) < 2:
            raise ValueError("At least two one-dimensional shortfall samples are required.")
        if not np.all(np.isfinite(shortfall)) or np.any(shortfall < 0.0):
            raise ValueError("Shortfall samples must be finite and nonnegative.")

        states = _shortfall_states(shortfall)
        counts = np.full((self.n_states, self.n_states), self.alpha, dtype=float)
        np.add.at(counts, (states[:-1], states[1:]), 1.0)
        self.transition_matrix = counts / counts.sum(axis=1, keepdims=True)
        self.last_state = int(states[-1])
        return self

    def predict_priors(self, steps: int) -> np.ndarray:
        if self.transition_matrix is None or self.last_state is None:
            raise RuntimeError("The Markov shortfall model must be fitted before prediction.")
        if steps < 1:
            raise ValueError("steps must be positive.")

        current = np.zeros(self.n_states, dtype=float)
        current[self.last_state] = 1.0
        priors = np.zeros((steps, self.n_states), dtype=float)
        for step in range(steps):
            current = current @ self.transition_matrix
            priors[step] = current
        return priors


class BayesianShortfallNetwork:
    """Naive-Bayes-structured network with shortfall state as the hidden parent.

    The learned graph is ShortfallState -> {Hour, TemperatureBand, DemandBand,
    CapacityMarginBand}. Conditional probability tables use Laplace smoothing.
    """

    FEATURE_CARDINALITIES = (24, 3, 4, 4)

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        self.alpha = alpha
        self.demand_edges: Optional[np.ndarray] = None
        self.conditional_probabilities: Optional[List[np.ndarray]] = None
        self.state_mean_shortfalls: Optional[np.ndarray] = None

    def fit(
        self,
        historical_data: pd.DataFrame,
        grid: CityGridNetwork,
    ) -> "BayesianShortfallNetwork":
        required = {"timestamp", "energy_consumption_mw", "temperature_c"}
        missing = required - set(historical_data.columns)
        if missing:
            raise ValueError(f"Missing Bayesian-network training columns: {', '.join(sorted(missing))}.")

        data = historical_data.sort_values("timestamp")
        demand = data["energy_consumption_mw"].to_numpy(dtype=float)
        temperature = data["temperature_c"].to_numpy(dtype=float)
        timestamps = pd.to_datetime(data["timestamp"])
        hours = timestamps.dt.hour.to_numpy(dtype=int)
        if len(demand) < 2:
            raise ValueError("At least two historical rows are required to fit the Bayesian network.")
        if not np.all(np.isfinite(demand)) or not np.all(np.isfinite(temperature)):
            raise ValueError("Bayesian-network training inputs must be finite.")
        if np.any(demand <= 0.0):
            raise ValueError("Historical demand must be positive.")

        self.demand_edges = np.unique(np.quantile(demand, [0.25, 0.50, 0.75]))
        evidence_rows: List[np.ndarray] = []
        shortfall_rows: List[np.ndarray] = []

        for index, capacity_ratio in enumerate(TRAINING_CAPACITY_RATIOS):
            capacity = grid.simulate_generation_capacity(
                demand,
                average_shortfall_ratio=1.0 - capacity_ratio,
                seed=42 + index,
            )
            shortfall = grid.calculate_shortfall(demand, capacity)
            evidence_rows.append(
                self._encode_evidence(hours, temperature, demand, capacity)
            )
            shortfall_rows.append(shortfall)

        evidence = np.concatenate(evidence_rows, axis=0)
        shortfall = np.concatenate(shortfall_rows)
        states = _shortfall_states(shortfall)

        conditional_probabilities = []
        for feature_index, cardinality in enumerate(self.FEATURE_CARDINALITIES):
            counts = np.full(
                (len(SHORTFALL_STATE_NAMES), cardinality),
                self.alpha,
                dtype=float,
            )
            np.add.at(counts, (states, evidence[:, feature_index]), 1.0)
            conditional_probabilities.append(
                counts / counts.sum(axis=1, keepdims=True)
            )

        state_means = SHORTFALL_STATE_FALLBACK_MW.copy()
        for state in range(len(SHORTFALL_STATE_NAMES)):
            state_values = shortfall[states == state]
            if len(state_values):
                state_means[state] = float(np.mean(state_values))

        self.conditional_probabilities = conditional_probabilities
        self.state_mean_shortfalls = state_means
        return self

    def _encode_evidence(
        self,
        hours: np.ndarray,
        temperature_c: np.ndarray,
        demand_mw: np.ndarray,
        capacity_mw: np.ndarray,
    ) -> np.ndarray:
        if self.demand_edges is None:
            raise RuntimeError("The Bayesian shortfall network must be fitted before encoding evidence.")
        return np.column_stack(
            (
                np.asarray(hours, dtype=int) % 24,
                np.digitize(temperature_c, [14.0, 22.0]),
                np.digitize(demand_mw, self.demand_edges),
                np.digitize(
                    capacity_mw / np.maximum(demand_mw, 1.0),
                    [0.75, 0.85, 0.95],
                ),
            )
        ).astype(int)

    def posterior(
        self,
        prior: np.ndarray,
        evidence: np.ndarray,
    ) -> np.ndarray:
        if self.conditional_probabilities is None:
            raise RuntimeError("The Bayesian shortfall network must be fitted before prediction.")
        prior = np.asarray(prior, dtype=float)
        evidence = np.asarray(evidence, dtype=int)
        if prior.shape != (len(SHORTFALL_STATE_NAMES),):
            raise ValueError("prior must have one probability per shortfall state.")
        if evidence.shape != (len(self.FEATURE_CARDINALITIES),):
            raise ValueError("evidence must contain hour, temperature, demand, and capacity-margin bins.")
        if not np.all(np.isfinite(prior)) or np.any(prior < 0.0) or not np.isclose(prior.sum(), 1.0):
            raise ValueError("prior must be a normalized probability distribution.")

        log_posterior = np.log(np.maximum(prior, 1e-300))
        for feature_index, value in enumerate(evidence):
            cardinality = self.FEATURE_CARDINALITIES[feature_index]
            if value < 0 or value >= cardinality:
                raise ValueError(f"Evidence value for feature {feature_index} is out of range.")
            likelihood = self.conditional_probabilities[feature_index][:, value]
            log_posterior += np.log(likelihood)

        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        return posterior / posterior.sum()


class ProbabilisticShortfallModel:
    """Combines Markov state prediction with Bayesian evidence for scheduling."""

    def __init__(self, alpha: float = 1.0):
        self.markov = MarkovShortfallModel(alpha=alpha)
        self.bayesian_network = BayesianShortfallNetwork(alpha=alpha)
        self._training_rows = 0

    def fit(
        self,
        historical_data: pd.DataFrame,
        grid: CityGridNetwork,
    ) -> "ProbabilisticShortfallModel":
        required = {"timestamp", "energy_consumption_mw"}
        missing = required - set(historical_data.columns)
        if missing:
            raise ValueError(f"Missing Markov training columns: {', '.join(sorted(missing))}.")
        data = historical_data.sort_values("timestamp")
        demand = data["energy_consumption_mw"].to_numpy(dtype=float)
        if len(demand) < 2 or not np.all(np.isfinite(demand)) or np.any(demand <= 0.0):
            raise ValueError("At least two finite positive historical demand samples are required.")

        base_capacity = grid.simulate_generation_capacity(
            demand,
            average_shortfall_ratio=1.0 - BASE_CAPACITY_RATIO,
            seed=42,
        )
        historical_shortfall = grid.calculate_shortfall(demand, base_capacity)
        self.markov.fit(historical_shortfall)
        self.bayesian_network.fit(data, grid)
        self._training_rows = len(data)
        return self

    def predict(
        self,
        demand_mw: Sequence[float],
        temperature_c: Sequence[float],
        capacity_mw: Sequence[float],
        hours: Optional[Sequence[int]] = None,
    ) -> ProbabilisticShortfallForecast:
        if self.markov.transition_matrix is None or self.markov.last_state is None:
            raise RuntimeError("The probabilistic shortfall model must be fitted before prediction.")
        if self.bayesian_network.state_mean_shortfalls is None:
            raise RuntimeError("The probabilistic shortfall model must be fitted before prediction.")

        demand = np.asarray(demand_mw, dtype=float)
        temperature = np.asarray(temperature_c, dtype=float)
        capacity = np.asarray(capacity_mw, dtype=float)
        if demand.ndim != 1 or len(demand) == 0:
            raise ValueError("demand_mw must be a non-empty one-dimensional sequence.")
        if temperature.shape != demand.shape or capacity.shape != demand.shape:
            raise ValueError("Demand, temperature, and capacity must have matching shapes.")
        if np.any(demand <= 0.0) or np.any(capacity < 0.0):
            raise ValueError("Demand must be positive and capacity must be nonnegative.")
        if not np.all(np.isfinite(demand)) or not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(capacity)):
            raise ValueError("Prediction inputs must be finite.")
        hour_values = (
            np.arange(len(demand), dtype=int) % 24
            if hours is None
            else np.asarray(hours, dtype=int)
        )
        if hour_values.shape != demand.shape:
            raise ValueError("hours must have the same shape as demand_mw.")

        evidence = self.bayesian_network._encode_evidence(
            hour_values, temperature, demand, capacity
        )
        markov_priors = np.zeros(
            (len(demand), len(SHORTFALL_STATE_NAMES)),
            dtype=float,
        )
        posteriors = np.zeros_like(markov_priors)
        expected_shortfall = np.zeros(len(demand), dtype=float)
        state_prior = np.zeros(len(SHORTFALL_STATE_NAMES), dtype=float)
        state_prior[self.markov.last_state] = 1.0

        for index in range(len(demand)):
            state_prior = state_prior @ self.markov.transition_matrix
            markov_priors[index] = state_prior
            posterior = self.bayesian_network.posterior(state_prior, evidence[index])
            posteriors[index] = posterior
            expected_shortfall[index] = float(
                posterior @ self.bayesian_network.state_mean_shortfalls
            )
            state_prior = posterior

        return ProbabilisticShortfallForecast(
            expected_shortfall_mw=expected_shortfall,
            state_probabilities=posteriors,
            markov_prior_probabilities=markov_priors,
        )

    @property
    def training_rows(self) -> int:
        return self._training_rows
