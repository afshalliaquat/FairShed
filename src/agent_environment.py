"""Task 1: Types of Agents, Environments to Implement & Python Revision.

Implements:
1. PEAS Specification for the Smart Load-Shedding & Energy Dispatch System.
2. Formal Environment Properties Analysis (Observability, Determinism, etc.).
3. Simulation Environment: GridEnvironment with state transitions, telemetry, and rewards.
4. Four Classical Agent Architectures:
   - Simple Reflex Agent
   - Model-Based Reflex Agent
   - Goal-Based Agent
   - Utility-Based (Fairness-Optimizing) Agent
5. Pure Python implementation demonstrating idiomatic concepts:
   Abstract Base Classes, Dataclasses, Enums, Type Annotations, Dunder methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.city_grid import FeederArea, CityGridNetwork


class EnvironmentProperty(Enum):
    """Formal AI properties characterizing the power distribution environment."""

    OBSERVABILITY = "Partially Observable (Substation telemetry contains sensor noise & transmission losses)"
    DETERMINISM = "Stochastic (Unpredictable generator tripping, cloud cover on solar, consumer surges)"
    EPISODICITY = "Sequential (Current shedding decisions directly affect future consecutive outage constraints)"
    DYNAMISM = "Dynamic (Grid load changes continuously while the agent is computing actions)"
    DISCRETENESS = "Mixed (Discrete feeder breaker switching {0, 1}; Continuous MW power flows & temperatures)"
    AGENT_TYPE = "Single-Agent with Competitive / Cooperative sub-sectors"


@dataclass(frozen=True)
class PEASSpecification:
    """PEAS Framework definition for the Load-Shedding Agent."""

    performance_measure: str = (
        "Maximize Grid Reliability (0 unmet shortfall) + Maximize Equity (Jain's Fairness Index -> 1.0) "
        "+ Zero Hospital / Iftar Violations + Minimized Social Disruption Cost."
    )
    environment: str = (
        "Urban electrical distribution grid comprising 8 heterogeneous feeders (Hospitals, Industrial, "
        "Commercial, Residential, Peri-urban), fluctuating hourly demand, and constrained generation capacity."
    )
    actuators: str = (
        "Remote SCADA Breaker Commands: Binary control vector [X_1, ..., X_8] in {0, 1}^8 "
        "(0 = Energized, 1 = Load-Shed / Disconnected) per hourly time step."
    )
    sensors: str = (
        "Substation telemetry meters (MW demand), generation capacity monitor (MW), "
        "ambient temperature sensor (°C), and real-time astronomical clock (hour, Ramadan calendar)."
    )

    def describe(self) -> str:
        return (
            "==================== PEAS SPECIFICATION ====================\n"
            f"• Performance Measure : {self.performance_measure}\n"
            f"• Environment         : {self.environment}\n"
            f"• Actuators           : {self.actuators}\n"
            f"• Sensors             : {self.sensors}\n"
            "============================================================"
        )


@dataclass
class GridPercept:
    """Sensor readings received by the agent at each hourly time step."""

    hour: int  # 0 to 23
    total_demand_mw: float
    generation_capacity_mw: float
    shortfall_mw: float
    area_demands: Dict[str, float]
    is_ramadan: bool
    is_iftar_time: bool
    is_sehri_time: bool


@dataclass
class AgentAction:
    """Action taken by the agent: set of feeder IDs to shed at the current hour."""

    hour: int
    shed_area_ids: Set[str]
    total_shed_mw: float
    unmet_shortfall_mw: float
    justification: str


class GridEnvironment:
    """Simulated power distribution environment governing state updates and physics."""

    def __init__(
        self,
        grid_network: Optional[CityGridNetwork] = None,
        is_ramadan: bool = True,
        noise_std_mw: float = 20.0,
        seed: int = 42,
    ):
        self.grid = grid_network if grid_network is not None else CityGridNetwork(is_ramadan=is_ramadan)
        self.is_ramadan = is_ramadan
        self.noise_std_mw = noise_std_mw
        self.rng = np.random.default_rng(seed)

        # Environment state tracking
        self.current_step: int = 0
        self.cumulative_outage_hours: Dict[str, int] = {f.area_id: 0 for f in self.grid.feeders}
        self.consecutive_outage_hours: Dict[str, int] = {f.area_id: 0 for f in self.grid.feeders}
        self.violation_log: List[str] = []

    def reset(self) -> None:
        """Resets the environment state to midnight hour 0."""
        self.current_step = 0
        self.cumulative_outage_hours = {f.area_id: 0 for f in self.grid.feeders}
        self.consecutive_outage_hours = {f.area_id: 0 for f in self.grid.feeders}
        self.violation_log.clear()

    def get_percept(
        self,
        city_demand_mw: float,
        generation_capacity_mw: float,
        step: Optional[int] = None,
    ) -> GridPercept:
        """Constructs sensor percept for the agent, adding realistic sensor noise."""
        if step is not None:
            self.current_step = step
        hour = self.current_step % 24

        # Add zero-mean Gaussian sensor noise
        sensor_noise = self.rng.normal(0, self.noise_std_mw)
        noisy_demand = max(100.0, city_demand_mw + sensor_noise)
        shortfall = max(0.0, noisy_demand - generation_capacity_mw)

        # Decompose into area demands
        area_demands = {}
        for feeder in self.grid.feeders:
            base_share = self.grid.feeder_weights[feeder.area_id]
            area_demands[feeder.area_id] = noisy_demand * base_share

        return GridPercept(
            hour=hour,
            total_demand_mw=noisy_demand,
            generation_capacity_mw=generation_capacity_mw,
            shortfall_mw=shortfall,
            area_demands=area_demands,
            is_ramadan=self.is_ramadan,
            is_iftar_time=(hour in (18, 19)),
            is_sehri_time=(hour == 4),
        )

    def execute_action(
        self, action: AgentAction, area_demands: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """Executes the breaker switching action and assesses penalties / reward.

        Returns:
            Tuple of (step_penalty, list_of_violations_incurred)
        """
        step_violations: List[str] = []
        hour = action.hour

        # Update cumulative & consecutive outage tracking
        for feeder in self.grid.feeders:
            f_id = feeder.area_id
            if f_id in action.shed_area_ids:
                self.cumulative_outage_hours[f_id] += 1
                self.consecutive_outage_hours[f_id] += 1

                # Check hospital violation
                if feeder.is_critical_exempt:
                    msg = f"Hour {hour:02d}: CRITICAL VIOLATION - Hospital Feeder '{feeder.name}' was shed!"
                    step_violations.append(msg)
                    self.violation_log.append(msg)

                # Check Iftar / Sehri violation
                if self.is_ramadan and hour in (18, 19):
                    msg = f"Hour {hour:02d}: RAMADAN VIOLATION - Feeder '{feeder.name}' shed during Iftar!"
                    step_violations.append(msg)
                    self.violation_log.append(msg)
                elif self.is_ramadan and hour == 4:
                    msg = f"Hour {hour:02d}: RAMADAN VIOLATION - Feeder '{feeder.name}' shed during Sehri!"
                    step_violations.append(msg)
                    self.violation_log.append(msg)

                # Check consecutive outage limit
                if self.consecutive_outage_hours[f_id] > feeder.max_consecutive_outage_hours:
                    msg = (
                        f"Hour {hour:02d}: CONSECUTIVE VIOLATION - Feeder '{feeder.name}' shed for "
                        f"{self.consecutive_outage_hours[f_id]} consecutive hours (max {feeder.max_consecutive_outage_hours})!"
                    )
                    step_violations.append(msg)
                    self.violation_log.append(msg)

                # Check daily quota limit
                if self.cumulative_outage_hours[f_id] > feeder.max_daily_outage_hours:
                    msg = (
                        f"Hour {hour:02d}: QUOTA VIOLATION - Feeder '{feeder.name}' exceeded daily limit "
                        f"({self.cumulative_outage_hours[f_id]}h > {feeder.max_daily_outage_hours}h)!"
                    )
                    step_violations.append(msg)
                    self.violation_log.append(msg)
            else:
                # Restored power resets consecutive counter
                self.consecutive_outage_hours[f_id] = 0

        # Calculate penalty: unmet shortfall is catastrophic (triggers grid blackout)
        penalty = 0.0
        if action.unmet_shortfall_mw > 1.0:
            penalty += action.unmet_shortfall_mw * 50.0  # Heavy blackout risk penalty

        # Disruption cost
        for f_id in action.shed_area_ids:
            feeder_obj = self.grid.get_feeder_by_id(f_id)
            if feeder_obj:
                penalty += feeder_obj.disruption_cost_per_hour

        # Constraint violation penalties
        penalty += len(step_violations) * 5000.0

        self.current_step += 1
        return penalty, step_violations


class BaseGridAgent(ABC):
    """Abstract Base Class for all Smart Grid Agents."""

    def __init__(self, name: str, grid_network: CityGridNetwork):
        self.name = name
        self.grid = grid_network

    @abstractmethod
    def select_action(self, percept: GridPercept) -> AgentAction:
        """Determines the shedding action given current sensor percept."""
        pass


class SimpleReflexGridAgent(BaseGridAgent):
    """Simple Reflex Agent: Condition-Action rules based strictly on current percept.

    Lacks internal memory of past outages; sheds the highest-load non-critical feeders
    until shortfall is met. Vulnerable to consecutive outage violations and unfairness.
    """

    def select_action(self, percept: GridPercept) -> AgentAction:
        if percept.shortfall_mw <= 0:
            return AgentAction(percept.hour, set(), 0.0, 0.0, "No shortfall - all feeders powered.")

        shed_feeders: Set[str] = set()
        total_shed = 0.0

        # Candidate non-exempt feeders sorted by load descending
        candidates = [
            f for f in self.grid.feeders
            if not f.is_critical_exempt and not f.is_protected_at_hour(percept.hour, percept.is_ramadan)
        ]
        candidates.sort(key=lambda f: percept.area_demands[f.area_id], reverse=True)

        for f in candidates:
            if total_shed >= percept.shortfall_mw:
                break
            shed_feeders.add(f.area_id)
            total_shed += percept.area_demands[f.area_id]

        unmet = max(0.0, percept.shortfall_mw - total_shed)
        return AgentAction(
            hour=percept.hour,
            shed_area_ids=shed_feeders,
            total_shed_mw=total_shed,
            unmet_shortfall_mw=unmet,
            justification=f"Reflex greedy shed {len(shed_feeders)} feeders covering {total_shed:.1f} MW.",
        )


class ModelBasedGridAgent(BaseGridAgent):
    """Model-Based Reflex Agent: Maintains internal world model.

    Tracks cumulative outage hours and consecutive outage history to prevent
    exceeding quotas and violating consecutive outage constraints.
    """

    def __init__(self, name: str, grid_network: CityGridNetwork):
        super().__init__(name, grid_network)
        self.internal_cumulative_outages: Dict[str, int] = {f.area_id: 0 for f in grid_network.feeders}
        self.internal_consecutive_outages: Dict[str, int] = {f.area_id: 0 for f in grid_network.feeders}

    def select_action(self, percept: GridPercept) -> AgentAction:
        if percept.shortfall_mw <= 0:
            for f in self.grid.feeders:
                self.internal_consecutive_outages[f.area_id] = 0
            return AgentAction(percept.hour, set(), 0.0, 0.0, "Shortfall zero; internal model reset consecutive.")

        # Filter candidates based on internal model
        eligible_candidates = []
        for f in self.grid.feeders:
            if f.is_critical_exempt or f.is_protected_at_hour(percept.hour, percept.is_ramadan):
                continue
            # Model constraints check
            if self.internal_consecutive_outages[f.area_id] >= f.max_consecutive_outage_hours:
                continue
            if self.internal_cumulative_outages[f.area_id] >= f.max_daily_outage_hours:
                continue
            eligible_candidates.append(f)

        # Prioritize feeders with lowest cumulative outage hours (fairness heuristic)
        eligible_candidates.sort(key=lambda f: self.internal_cumulative_outages[f.area_id])

        shed_feeders: Set[str] = set()
        total_shed = 0.0

        for f in eligible_candidates:
            if total_shed >= percept.shortfall_mw:
                break
            shed_feeders.add(f.area_id)
            total_shed += percept.area_demands[f.area_id]

        # Update internal model
        for f in self.grid.feeders:
            if f.area_id in shed_feeders:
                self.internal_cumulative_outages[f.area_id] += 1
                self.internal_consecutive_outages[f.area_id] += 1
            else:
                self.internal_consecutive_outages[f.area_id] = 0

        unmet = max(0.0, percept.shortfall_mw - total_shed)
        return AgentAction(
            hour=percept.hour,
            shed_area_ids=shed_feeders,
            total_shed_mw=total_shed,
            unmet_shortfall_mw=unmet,
            justification=f"Model-based shed {len(shed_feeders)} feeders respecting consecutive limits.",
        )


class GoalBasedGridAgent(BaseGridAgent):
    """Goal-Based Agent: Evaluates actions based on whether they achieve the explicit goal:

    Goal: {Unmet Shortfall == 0.0 AND Hospital Shed == False AND Iftar Protected == True}.
    Searches combinations to reach the goal with minimal surplus shed.
    """

    def select_action(self, percept: GridPercept) -> AgentAction:
        if percept.shortfall_mw <= 0:
            return AgentAction(percept.hour, set(), 0.0, 0.0, "Goal achieved trivially (no deficit).")

        # Exclude protected feeders
        valid_feeders = [
            f for f in self.grid.feeders
            if not f.is_critical_exempt and not f.is_protected_at_hour(percept.hour, percept.is_ramadan)
        ]

        # Find subset of valid feeders that achieves goal (shed >= shortfall) with minimal over-shed
        best_subset: List[FeederArea] = []
        best_overshed = float("inf")

        # Simple combinatorial search over reasonable subset sizes (1 to 4)
        from itertools import combinations

        for r in range(1, min(5, len(valid_feeders) + 1)):
            for comb in combinations(valid_feeders, r):
                shed_mw = sum(percept.area_demands[f.area_id] for f in comb)
                if shed_mw >= percept.shortfall_mw:
                    overshed = shed_mw - percept.shortfall_mw
                    if overshed < best_overshed:
                        best_overshed = overshed
                        best_subset = list(comb)

        if not best_subset:
            # Fallback if no clean combo meets it
            best_subset = valid_feeders[:3]

        shed_ids = {f.area_id for f in best_subset}
        total_shed = sum(percept.area_demands[f_id] for f_id in shed_ids)
        unmet = max(0.0, percept.shortfall_mw - total_shed)

        return AgentAction(
            hour=percept.hour,
            shed_area_ids=shed_ids,
            total_shed_mw=total_shed,
            unmet_shortfall_mw=unmet,
            justification=f"Goal-based search satisfied deficit with {total_shed - percept.shortfall_mw:.1f} MW overshed.",
        )


class UtilityBasedFairGridAgent(BaseGridAgent):
    """Utility-Based Agent: Uses an explicit multi-attribute utility function:

    U(action) = - w_shortfall * (Unmet_MW)^2
                - w_overshed * (OverShed_MW)
                - w_fairness * (Variance of Cumulative Outage Hours)
                - w_cost * (Social Disruption Cost)
                - Violations_Penalty

    Selects the action that strictly maximizes expected utility.
    """

    def __init__(
        self,
        name: str,
        grid_network: CityGridNetwork,
        w_shortfall: float = 10.0,
        w_overshed: float = 0.5,
        w_fairness: float = 80.0,
        w_cost: float = 0.1,
    ):
        super().__init__(name, grid_network)
        self.w_shortfall = w_shortfall
        self.w_overshed = w_overshed
        self.w_fairness = w_fairness
        self.w_cost = w_cost
        self.cumulative_outages: Dict[str, int] = {f.area_id: 0 for f in grid_network.feeders}

    def compute_utility(
        self,
        candidate_shed_ids: Set[str],
        percept: GridPercept,
    ) -> float:
        """Calculates scalar utility score for a candidate shedding decision."""
        # Hard constraint violations yield catastrophic negative utility
        for f_id in candidate_shed_ids:
            feeder = self.grid.get_feeder_by_id(f_id)
            if feeder and (feeder.is_critical_exempt or feeder.is_protected_at_hour(percept.hour, percept.is_ramadan)):
                return -1_000_000.0

        total_shed = sum(percept.area_demands[f_id] for f_id in candidate_shed_ids)
        unmet = max(0.0, percept.shortfall_mw - total_shed)
        overshed = max(0.0, total_shed - percept.shortfall_mw)

        # Projected cumulative outages
        projected_outages = []
        for f in self.grid.feeders:
            if not f.is_critical_exempt:
                current_h = self.cumulative_outages[f.area_id]
                new_h = current_h + (1 if f.area_id in candidate_shed_ids else 0)
                projected_outages.append(new_h)

        fairness_var = float(np.var(projected_outages)) if projected_outages else 0.0

        # Disruption cost
        social_cost = sum(
            self.grid.get_feeder_by_id(f_id).disruption_cost_per_hour
            for f_id in candidate_shed_ids
            if self.grid.get_feeder_by_id(f_id)
        )

        utility = (
            - self.w_shortfall * (unmet ** 1.5)
            - self.w_overshed * overshed
            - self.w_fairness * fairness_var
            - self.w_cost * social_cost
        )
        return utility

    def select_action(self, percept: GridPercept) -> AgentAction:
        if percept.shortfall_mw <= 0:
            return AgentAction(percept.hour, set(), 0.0, 0.0, "Zero shortfall - maximal utility is idle.")

        valid_feeders = [
            f for f in self.grid.feeders
            if not f.is_critical_exempt and not f.is_protected_at_hour(percept.hour, percept.is_ramadan)
        ]

        from itertools import combinations

        best_utility = -float("inf")
        best_set: Set[str] = set()

        for r in range(1, min(4, len(valid_feeders) + 1)):
            for comb in combinations(valid_feeders, r):
                candidate_ids = {f.area_id for f in comb}
                u = self.compute_utility(candidate_ids, percept)
                if u > best_utility:
                    best_utility = u
                    best_set = candidate_ids

        total_shed = sum(percept.area_demands[f_id] for f_id in best_set)
        unmet = max(0.0, percept.shortfall_mw - total_shed)

        # Update cumulative internal state
        for f_id in best_set:
            self.cumulative_outages[f_id] += 1

        return AgentAction(
            hour=percept.hour,
            shed_area_ids=best_set,
            total_shed_mw=total_shed,
            unmet_shortfall_mw=unmet,
            justification=f"Utility-optimal action (Utility={best_utility:.1f}) balancing deficit & fairness.",
        )
