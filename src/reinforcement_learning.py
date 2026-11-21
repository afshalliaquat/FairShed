"""Week 13 (Lab 11): Reinforcement Learning - Tabular Q-Learning for Grid Load Shedding.

Implements a Markov Decision Process (MDP) and Tabular Q-Learning Agent from scratch:
- State Space S: (hour_of_day, deficit_level_bucket, cumulative_disparity_bucket)
- Action Space A: Discrete feeder dispatch actions across non-exempt clusters
- Reward Function R(s, a, s'): Penalizes unmet deficit, penalizes overshedding, rewards fairness
- Q-Learning Update: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
- Epsilon-greedy exploration with decay schedule.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np

from src.city_grid import CityGridNetwork


@dataclass
class RLTrainingResult:
    """Diagnostics from Q-learning agent training."""

    episodes: int
    final_epsilon: float
    q_table_size: int
    episode_rewards: List[float]
    policy_schedule: np.ndarray  # Shape (n_feeders, 24)
    policy_outages: Dict[str, int]
    policy_violations: List[str]


class TabularQGridAgent:
    """Tabular Q-Learning Agent for fair hourly load-shedding dispatch."""

    # Discrete Action Archetypes (never touches hospital!)
    ACTION_NAMES = [
        "Idle / All Powered",
        "Shed Rural & Agricultural",
        "Shed Suburban A & B",
        "Shed Urban Residential B",
        "Shed Urban Residential A & Commercial",
        "Shed Industrial & Suburban",
        "Multi-Sector Shared Shed",
    ]

    def __init__(
        self,
        grid_network: CityGridNetwork,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        epsilon_initial: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.grid = grid_network
        self.feeders = grid_network.feeders
        self.n_feeders = len(self.feeders)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_initial
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Q-table: Dict[state_tuple, np.ndarray of shape [n_actions]]
        self.n_actions = len(self.ACTION_NAMES)
        self.q_table: Dict[Tuple[int, int, int], np.ndarray] = {}

    def _get_q_values(self, state: Tuple[int, int, int]) -> np.ndarray:
        if state not in self.q_table:
            # Optimistic initialization to encourage initial exploration
            self.q_table[state] = np.zeros(self.n_actions, dtype=float)
        return self.q_table[state]

    def _discretize_deficit(self, shortfall_mw: float) -> int:
        """Discretizes continuous shortfall into 4 discrete buckets."""
        if shortfall_mw <= 10.0:
            return 0  # None
        elif shortfall_mw <= 300.0:
            return 1  # Low
        elif shortfall_mw <= 650.0:
            return 2  # Medium
        else:
            return 3  # High

    def _discretize_disparity(self, cumulative_outages: Dict[str, int]) -> int:
        """Discretizes outage spread among non-critical feeders."""
        non_crit_vals = [
            cumulative_outages[f.area_id]
            for f in self.feeders
            if not f.is_critical_exempt
        ]
        if not non_crit_vals:
            return 0
        spread = max(non_crit_vals) - min(non_crit_vals)
        if spread <= 1:
            return 0  # Fair
        elif spread == 2:
            return 1  # Moderate disparity
        else:
            return 2  # High disparity

    def _action_to_feeder_ids(
        self, action: int, hour: int, is_ramadan: bool
    ) -> Set[str]:
        """Translates discrete action index into a set of feeder IDs to shed."""
        # Hospital and Iftar/Sehri are NEVER shed under any action
        if is_ramadan and hour in (4, 18, 19):
            return set()

        if action == 0:
            return set()
        elif action == 1:
            return {"F08_RUR"}
        elif action == 2:
            return {"F06_SUB_A", "F07_SUB_B"}
        elif action == 3:
            return {"F05_RES_B", "F10_RES_C"}
        elif action == 4:
            return {"F04_RES_A", "F09_COMM_B"}
        elif action == 5:
            return {"F02_IND", "F06_SUB_A"}
        elif action == 6:
            return {"F05_RES_B", "F07_SUB_B", "F08_RUR"}
        return set()

    def select_action(self, state: Tuple[int, int, int]) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q_vals = self._get_q_values(state)
        # Random tie breaking for identical Q-values
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(self.rng.choice(best_actions))

    def train(
        self,
        hourly_shortfall_24h: np.ndarray,
        area_demands_24h: Dict[str, np.ndarray],
        num_episodes: int = 150,
        is_ramadan: bool = True,
    ) -> RLTrainingResult:
        """Trains the Q-learning agent across simulation days."""
        episode_rewards = []

        for _ in range(num_episodes):
            total_ep_reward = 0.0
            cumulative_outages = {f.area_id: 0 for f in self.feeders}

            for hour in range(24):
                shortfall = hourly_shortfall_24h[hour]
                def_bucket = self._discretize_deficit(shortfall)
                disp_bucket = self._discretize_disparity(cumulative_outages)
                state = (hour, def_bucket, disp_bucket)

                # Select action
                action = self.select_action(state)
                shed_ids = self._action_to_feeder_ids(action, hour, is_ramadan)

                # Execute action
                total_shed = sum(
                    area_demands_24h[f_id][hour]
                    for f_id in shed_ids
                    if f_id in area_demands_24h
                )

                # Update cumulative outages
                for f_id in shed_ids:
                    cumulative_outages[f_id] += 1

                # Calculate reward
                reward = 0.0
                if shortfall > 10.0:
                    if total_shed >= shortfall:
                        reward += 20.0  # Successfully met shortfall
                        overshed = total_shed - shortfall
                        reward -= overshed * 0.02  # Slight penalty for overshed
                    else:
                        unmet = shortfall - total_shed
                        reward -= (unmet / 50.0) * 15.0  # Severe blackout risk
                else:
                    if total_shed > 0:
                        reward -= 10.0  # Unnecessary shedding when no deficit
                    else:
                        reward += 5.0

                # Fairness penalty based on disparity
                next_disp_bucket = self._discretize_disparity(cumulative_outages)
                reward -= next_disp_bucket * 8.0

                # Max quota penalty
                for f in self.feeders:
                    if cumulative_outages[f.area_id] > f.max_daily_outage_hours and not f.is_critical_exempt:
                        reward -= 20.0

                total_ep_reward += reward

                # Next state & Q-update
                next_hour = (hour + 1) % 24
                next_shortfall = hourly_shortfall_24h[next_hour] if hour < 23 else 0.0
                next_def_bucket = self._discretize_deficit(next_shortfall)
                next_state = (next_hour, next_def_bucket, next_disp_bucket)

                # Bellman equation
                current_q = self._get_q_values(state)[action]
                next_max_q = np.max(self._get_q_values(next_state))
                new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
                self.q_table[state][action] = new_q

            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(round(total_ep_reward, 1))

        # Evaluate greedy policy schedule
        schedule = np.zeros((self.n_feeders, 24), dtype=int)
        eval_cum_outages = {f.area_id: 0 for f in self.feeders}

        for h in range(24):
            shortfall = hourly_shortfall_24h[h]
            def_bucket = self._discretize_deficit(shortfall)
            disp_bucket = self._discretize_disparity(eval_cum_outages)
            state = (h, def_bucket, disp_bucket)

            # Greedy choice
            q_vals = self._get_q_values(state)
            greedy_action = int(np.argmax(q_vals))
            shed_set = self._action_to_feeder_ids(greedy_action, h, is_ramadan)

            for i, f in enumerate(self.feeders):
                if f.area_id in shed_set:
                    schedule[i, h] = 1
                    eval_cum_outages[f.area_id] += 1

        violations = []
        # Check hospital
        for i, f in enumerate(self.feeders):
            if f.is_critical_exempt and np.sum(schedule[i, :]) > 0:
                violations.append(f"Hospital {f.name} was shed by RL agent!")

        return RLTrainingResult(
            episodes=num_episodes,
            final_epsilon=round(self.epsilon, 4),
            q_table_size=len(self.q_table),
            episode_rewards=episode_rewards,
            policy_schedule=schedule,
            policy_outages=eval_cum_outages,
            policy_violations=violations,
        )
