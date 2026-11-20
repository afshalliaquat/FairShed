"""Task 6 (Lab 7): Dynamic Bayesian Networks, Hidden Markov Models (HMM) & Kalman Filter.

Implements pure Python / NumPy probabilistic temporal models without external libraries:
1. State-Space Kalman Filter (KF):
   - Discrete-time 2D state estimator (Load MW, Rate of Change MW/h).
   - Recursively estimates true power demand from noisy SCADA sensor telemetries.
2. Hidden Markov Model (HMM):
   - Hidden States: {0: 'Normal Demand', 1: 'Stressed High-Load', 2: 'Grid Emergency'}
   - Observation Emissions: {0: 'Low <12GW', 1: 'Nominal 12-16GW', 2: 'Peak 16-19GW', 3: 'Critical >19GW'}
   - Forward Algorithm for state belief monitoring.
   - Viterbi Algorithm for decoding the most probable sequence of hidden grid operating states.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class KalmanFilterDiagnostics:
    """Telemetry diagnostic output from the Kalman Filter."""

    raw_noisy_measurements: np.ndarray
    filtered_estimates: np.ndarray
    estimated_load_rates: np.ndarray
    kalman_gains: np.ndarray
    error_covariances: np.ndarray
    raw_rmse: float
    filtered_rmse: float
    noise_reduction_percentage: float


class PowerGridKalmanFilter:
    """Discrete-time 2D Kalman Filter for electricity grid telemetry denoising.

    State vector: x_k = [Demand_MW, Rate_of_Change_MW_per_hour]^T
    System Dynamics:
        x_k = A * x_{k-1} + w_k,   w_k ~ N(0, Q)
        z_k = H * x_k + v_k,       v_k ~ N(0, R)
    """

    def __init__(
        self,
        dt_hours: float = 1.0,
        process_noise_q_pos: float = 5000.0,
        process_noise_q_vel: float = 120000.0,
        measurement_noise_r: float = 10000.0,  # Sensor variance (std = 100 MW)
    ):
        self.dt = dt_hours

        # State Transition Matrix A (constant velocity load model)
        self.A = np.array([[1.0, self.dt], [0.0, 1.0]], dtype=float)

        # Measurement Matrix H (we observe only the load demand, not the rate)
        self.H = np.array([[1.0, 0.0]], dtype=float)

        # Process Noise Covariance Q
        self.Q = np.array(
            [
                [process_noise_q_pos, 0.0],
                [0.0, process_noise_q_vel],
            ],
            dtype=float,
        )

        # Measurement Noise Covariance R (scalar variance)
        self.R = np.array([[measurement_noise_r]], dtype=float)

        # Identity matrix
        self.I = np.eye(2, dtype=float)

    def filter_sequence(
        self,
        noisy_measurements: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        initial_load_estimate: Optional[float] = None,
    ) -> KalmanFilterDiagnostics:
        """Runs the predict-update loop across an entire time-series sequence."""
        n = len(noisy_measurements)
        filtered_loads = np.zeros(n, dtype=float)
        estimated_rates = np.zeros(n, dtype=float)
        kalman_gains = np.zeros(n, dtype=float)
        error_covariances = np.zeros(n, dtype=float)

        # Initial state: [first measurement, initial estimated velocity]
        init_val = initial_load_estimate if initial_load_estimate is not None else float(noisy_measurements[0])
        init_rate = float(noisy_measurements[1] - noisy_measurements[0]) if n > 1 else 0.0
        x = np.array([[init_val], [init_rate]], dtype=float)
        P = np.array([[self.R[0, 0], 0.0], [0.0, 500.0 ** 2]], dtype=float)

        for k in range(n):
            z_k = np.array([[noisy_measurements[k]]], dtype=float)

            # --- PREDICT STEP ---
            # Prior state estimate
            x_pred = self.A @ x
            # Prior error covariance
            P_pred = (self.A @ P @ self.A.T) + self.Q

            # --- UPDATE STEP ---
            # Measurement residual (innovation)
            y_tilde = z_k - (self.H @ x_pred)
            # Innovation covariance
            S = float((self.H @ P_pred @ self.H.T + self.R)[0, 0])
            # Kalman Gain
            K = (P_pred @ self.H.T) / S

            # Posterior state estimate
            x = x_pred + (K * y_tilde[0, 0])
            # Posterior error covariance (Joseph form for numerical stability)
            P = (self.I - K @ self.H) @ P_pred

            filtered_loads[k] = float(x[0, 0])
            estimated_rates[k] = float(x[1, 0])
            kalman_gains[k] = float(K[0, 0])
            error_covariances[k] = float(P[0, 0])

        # Compute RMSE compared to ground truth if provided
        raw_rmse = 0.0
        filtered_rmse = 0.0
        noise_reduction = 0.0

        if ground_truth is not None and len(ground_truth) == n:
            raw_rmse = float(np.sqrt(np.mean((noisy_measurements - ground_truth) ** 2)))
            filtered_rmse = float(np.sqrt(np.mean((filtered_loads - ground_truth) ** 2)))
            if raw_rmse > 0:
                noise_reduction = float((1.0 - (filtered_rmse / raw_rmse)) * 100.0)

        return KalmanFilterDiagnostics(
            raw_noisy_measurements=noisy_measurements,
            filtered_estimates=filtered_loads,
            estimated_load_rates=estimated_rates,
            kalman_gains=kalman_gains,
            error_covariances=error_covariances,
            raw_rmse=raw_rmse,
            filtered_rmse=filtered_rmse,
            noise_reduction_percentage=noise_reduction,
        )


@dataclass
class HMMDecodingResult:
    """Results from Hidden Markov Model inference."""

    hidden_state_names: List[str]
    decoded_state_indices: List[int]
    decoded_state_labels: List[str]
    state_probabilities_forward: np.ndarray
    observation_sequence: List[int]
    log_likelihood: float


class GridStateHMM:
    """Hidden Markov Model for identifying latent operating stress regimes in the power grid.

    States:
        0: 'Normal' (Sustainable reserve margin)
        1: 'Stressed' (Substations approaching thermal limits)
        2: 'Emergency' (Critical shortfall risk / tripping hazard)
    Observations:
        0: 'Low Demand' (< 12,500 MW)
        1: 'Moderate Demand' (12,500 - 15,500 MW)
        2: 'High Peak Demand' (15,500 - 18,000 MW)
        3: 'Critical Overload' (> 18,000 MW)
    """

    STATE_NAMES = ["Normal", "Stressed", "Emergency"]
    OBSERVATION_NAMES = ["Low (<12.5k)", "Moderate (12.5-15.5k)", "High (15.5-18k)", "Critical (>18k)"]

    def __init__(self):
        # Initial state probabilities pi
        self.pi = np.array([0.70, 0.25, 0.05], dtype=float)

        # State transition probability matrix A (rows: from state, cols: to state)
        # Highly persistent states with realistic switching dynamics
        self.A = np.array(
            [
                [0.85, 0.13, 0.02],  # Normal -> [Normal, Stressed, Emergency]
                [0.15, 0.75, 0.10],  # Stressed -> [Normal, Stressed, Emergency]
                [0.05, 0.35, 0.60],  # Emergency -> [Normal, Stressed, Emergency]
            ],
            dtype=float,
        )

        # Emission probability matrix B (rows: hidden state, cols: observation)
        self.B = np.array(
            [
                [0.55, 0.40, 0.05, 0.00],  # Normal emits mostly Low/Moderate
                [0.05, 0.35, 0.50, 0.10],  # Stressed emits mostly Moderate/High
                [0.00, 0.05, 0.45, 0.50],  # Emergency emits mostly High/Critical
            ],
            dtype=float,
        )

    def discretize_observations(self, continuous_demands: np.ndarray) -> List[int]:
        """Maps continuous MW load readings into discrete emission symbols [0, 1, 2, 3]."""
        obs = []
        for mw in continuous_demands:
            if mw < 12500.0:
                obs.append(0)
            elif mw < 15500.0:
                obs.append(1)
            elif mw < 18000.0:
                obs.append(2)
            else:
                obs.append(3)
        return obs

    def forward_algorithm(self, observations: List[int]) -> Tuple[np.ndarray, float]:
        """Forward Algorithm: Computes alpha_t(i) = P(O_1, ..., O_t, State_t = S_i).

        Returns:
            Tuple of (alpha_matrix shape [T, n_states], total_log_likelihood)
        """
        T = len(observations)
        n_states = len(self.STATE_NAMES)
        alpha = np.zeros((T, n_states), dtype=float)

        # 1. Initialization
        alpha[0, :] = self.pi * self.B[:, observations[0]]
        scale_factors = np.zeros(T)
        scale_factors[0] = np.sum(alpha[0, :])
        if scale_factors[0] > 0:
            alpha[0, :] /= scale_factors[0]

        # 2. Induction
        for t in range(1, T):
            for j in range(n_states):
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.A[:, j]) * self.B[j, observations[t]]
            scale_factors[t] = np.sum(alpha[t, :])
            if scale_factors[t] > 0:
                alpha[t, :] /= scale_factors[t]

        # Compute log likelihood
        log_likelihood = float(np.sum(np.log(np.maximum(scale_factors, 1e-12))))
        return alpha, log_likelihood

    def viterbi_decode(self, observations: List[int]) -> HMMDecodingResult:
        """Viterbi Algorithm: Decodes the mathematically most probable hidden state path.

        Solves argmax_{q_1..q_T} P(q_1..q_T | O_1..O_T).
        """
        T = len(observations)
        n_states = len(self.STATE_NAMES)

        # Log domain to prevent underflow: delta[t, state]
        delta = np.zeros((T, n_states), dtype=float)
        psi = np.zeros((T, n_states), dtype=int)  # Backpointers

        log_pi = np.log(np.maximum(self.pi, 1e-12))
        log_A = np.log(np.maximum(self.A, 1e-12))
        log_B = np.log(np.maximum(self.B, 1e-12))

        # Initialization
        delta[0, :] = log_pi + log_B[:, observations[0]]

        # Recursion
        for t in range(1, T):
            obs = observations[t]
            for j in range(n_states):
                # Transition candidate scores
                scores = delta[t - 1, :] + log_A[:, j]
                best_prev = int(np.argmax(scores))
                delta[t, j] = scores[best_prev] + log_B[j, obs]
                psi[t, j] = best_prev

        # Termination & Backtracking
        best_path_indices = [0] * T
        best_path_indices[-1] = int(np.argmax(delta[-1, :]))

        for t in range(T - 2, -1, -1):
            best_path_indices[t] = psi[t + 1, best_path_indices[t + 1]]

        decoded_labels = [self.STATE_NAMES[idx] for idx in best_path_indices]
        forward_probs, log_lik = self.forward_algorithm(observations)

        return HMMDecodingResult(
            hidden_state_names=self.STATE_NAMES,
            decoded_state_indices=best_path_indices,
            decoded_state_labels=decoded_labels,
            state_probabilities_forward=forward_probs,
            observation_sequence=observations,
            log_likelihood=log_lik,
        )
