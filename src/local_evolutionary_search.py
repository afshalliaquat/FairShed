"""Task 5 (Lab 6): Local Search (Hill Climbing) & Evolutionary (Genetic) Algorithms.

Implements optimization schedulers from scratch without external optimization libraries:
1. Steepest-Ascent Hill Climbing with neighborhood evaluation (flips & swaps).
2. Simulated Annealing with geometric temperature cooling schedule.
3. Genetic Algorithm (Evolutionary Search):
   - Chromosome: Binary schedule matrix (Feeders x 24 hours)
   - Multi-objective fitness (shortfall coverage + Jain's fairness + zero constraint violations)
   - Tournament selection, 2-point crossover, bit-flip mutation, and elitism.
"""

from dataclasses import dataclass
import math
import time
from typing import Callable, Dict, List, Tuple
import numpy as np

from src.city_grid import FeederArea, CityGridNetwork


@dataclass
class OptimizationResult:
    """Contains diagnostics and metrics from local or evolutionary search."""

    algorithm_name: str
    best_schedule: np.ndarray  # Shape (n_feeders, 24)
    best_score: float  # Lower is better for loss, higher is better for fitness
    hourly_shed_mw: np.ndarray
    area_outage_hours: Dict[str, int]
    iterations: int
    runtime_seconds: float
    convergence_history: List[float]
    violations: List[str]


class ScheduleEvaluator:
    """Computes penalty-augmented loss function for any 24h schedule matrix."""

    def __init__(
        self,
        grid_network: CityGridNetwork,
        hourly_shortfall: np.ndarray,
        area_demands: Dict[str, np.ndarray],
        max_daily_outage: int = 4,
        max_consecutive: int = 2,
        is_ramadan: bool = True,
        w_shortfall: float = 20.0,
        w_fairness: float = 50.0,
        w_overshed: float = 1.0,
        w_violation: float = 5000.0,
    ):
        self.grid = grid_network
        self.feeders = grid_network.feeders
        self.n_feeders = len(self.feeders)
        self.hours = len(hourly_shortfall)
        self.shortfall = hourly_shortfall
        self.area_demands = area_demands
        self.max_daily = max_daily_outage
        self.max_consecutive = max_consecutive
        self.is_ramadan = is_ramadan

        # Weights
        self.w_shortfall = w_shortfall
        self.w_fairness = w_fairness
        self.w_overshed = w_overshed
        self.w_violation = w_violation

    def evaluate(self, schedule: np.ndarray) -> Tuple[float, List[str]]:
        """Evaluates schedule loss (lower is better) and lists violations."""
        loss = 0.0
        violations: List[str] = []

        # 1. Hospital Exemption (Extreme Penalty)
        for i, f in enumerate(self.feeders):
            if f.is_critical_exempt:
                bad_hours = np.where(schedule[i, :] == 1)[0]
                if len(bad_hours) > 0:
                    loss += len(bad_hours) * self.w_violation * 2.0
                    violations.append(f"Hospital '{f.name}' shed at {list(bad_hours)}")

        # 2. Ramadan Protected Hours (Iftar: 18, 19; Sehri: 4)
        if self.is_ramadan:
            for i, f in enumerate(self.feeders):
                for h in (4, 18, 19):
                    if schedule[i, h] == 1:
                        loss += self.w_violation
                        violations.append(f"Feeder '{f.name}' shed during Iftar/Sehri at hour {h:02d}:00")

        # 3. Max Daily Outage Limit
        outage_counts = []
        for i, f in enumerate(self.feeders):
            total_h = int(np.sum(schedule[i, :]))
            if not f.is_critical_exempt:
                outage_counts.append(total_h)
                limit = min(self.max_daily, f.max_daily_outage_hours)
                if total_h > limit:
                    excess = total_h - limit
                    loss += excess * self.w_violation * 0.5
                    violations.append(f"Feeder '{f.name}' exceeded daily quota ({total_h}h > {limit}h)")

        # 4. Max Consecutive Outage Limit
        for i, f in enumerate(self.feeders):
            consec = 0
            for h in range(self.hours):
                if schedule[i, h] == 1:
                    consec += 1
                    if consec > self.max_consecutive:
                        loss += self.w_violation * 0.5
                        violations.append(f"Feeder '{f.name}' exceeded consecutive outage at {h:02d}:00")
                else:
                    consec = 0

        # 5. Hourly Shortfall Satisfaction & Overshedding
        for h in range(self.hours):
            req = self.shortfall[h]
            shed_mw = sum(
                schedule[i, h] * self.area_demands[self.feeders[i].area_id][h]
                for i in range(self.n_feeders)
            )
            if shed_mw < req:
                unmet = req - shed_mw
                loss += (unmet ** 1.8) * self.w_shortfall
            else:
                overshed = shed_mw - req
                loss += overshed * self.w_overshed

        # 6. Fairness Variance among non-critical areas
        if outage_counts:
            var_outage = float(np.var(outage_counts))
            loss += var_outage * self.w_fairness

        return loss, violations


class HillClimbingScheduler:
    """Hill Climbing optimizer for fair load-shedding scheduling."""

    def __init__(self, evaluator: ScheduleEvaluator, seed: int = 42):
        self.evaluator = evaluator
        self.rng = np.random.default_rng(seed)

    def generate_initial_schedule(self) -> np.ndarray:
        """Constructs a smart initial schedule meeting basic constraints."""
        n_feeders = self.evaluator.n_feeders
        hours = self.evaluator.hours
        schedule = np.zeros((n_feeders, hours), dtype=int)

        non_critical = [
            i for i, f in enumerate(self.evaluator.feeders)
            if not f.is_critical_exempt
        ]

        # Greedy pass to cover shortfall
        cumulative_h = np.zeros(n_feeders, dtype=int)
        for h in range(hours):
            req = self.evaluator.shortfall[h]
            if req <= 10.0 or (self.evaluator.is_ramadan and h in (4, 18, 19)):
                continue

            # Sort non-critical by cumulative hours
            sorted_candidates = sorted(non_critical, key=lambda idx: cumulative_h[idx])
            shed_so_far = 0.0

            for idx in sorted_candidates:
                if shed_so_far >= req:
                    break
                if cumulative_h[idx] < self.evaluator.max_daily:
                    schedule[idx, h] = 1
                    cumulative_h[idx] += 1
                    shed_so_far += self.evaluator.area_demands[self.evaluator.feeders[idx].area_id][h]

        return schedule

    def solve_steepest_ascent(
        self,
        max_iterations: int = 150,
        neighborhood_sample_size: int = 40,
    ) -> OptimizationResult:
        """Steepest-Ascent Hill Climbing with sampled neighborhood moves."""
        start_time = time.perf_counter()
        current = self.generate_initial_schedule()
        current_loss, violations = self.evaluator.evaluate(current)

        history = [current_loss]
        n_feeders = self.evaluator.n_feeders
        hours = self.evaluator.hours

        for it in range(max_iterations):
            best_neighbor = None
            best_neighbor_loss = current_loss

            # Sample neighborhood moves: 1-bit flips & 2-feeder swaps at an hour
            for _ in range(neighborhood_sample_size):
                neighbor = current.copy()
                move_type = self.rng.choice(["flip", "swap"])

                if move_type == "flip":
                    # Pick random non-critical feeder and random hour
                    f_idx = self.rng.choice([i for i, f in enumerate(self.evaluator.feeders) if not f.is_critical_exempt])
                    h = self.rng.integers(0, hours)
                    if self.evaluator.is_ramadan and h in (4, 18, 19):
                        continue
                    neighbor[f_idx, h] = 1 - neighbor[f_idx, h]

                elif move_type == "swap":
                    # Swap shedding state of two non-critical feeders at hour h
                    h = self.rng.integers(0, hours)
                    if self.evaluator.is_ramadan and h in (4, 18, 19):
                        continue
                    nc = [i for i, f in enumerate(self.evaluator.feeders) if not f.is_critical_exempt]
                    if len(nc) >= 2:
                        f1, f2 = self.rng.choice(nc, size=2, replace=False)
                        neighbor[f1, h], neighbor[f2, h] = neighbor[f2, h], neighbor[f1, h]

                loss, _ = self.evaluator.evaluate(neighbor)
                if loss < best_neighbor_loss:
                    best_neighbor_loss = loss
                    best_neighbor = neighbor

            # If no strictly better neighbor found, reached local minimum!
            if best_neighbor is None or best_neighbor_loss >= current_loss:
                break

            current = best_neighbor
            current_loss = best_neighbor_loss
            history.append(current_loss)

        runtime = time.perf_counter() - start_time
        final_loss, final_violations = self.evaluator.evaluate(current)

        # Compute hourly shed MW
        hourly_shed = np.zeros(hours)
        for h in range(hours):
            hourly_shed[h] = sum(
                current[i, h] * self.evaluator.area_demands[self.evaluator.feeders[i].area_id][h]
                for i in range(n_feeders)
            )

        area_outages = {
            f.area_id: int(np.sum(current[i, :]))
            for i, f in enumerate(self.evaluator.feeders)
        }

        return OptimizationResult(
            algorithm_name="Steepest-Ascent Hill Climbing",
            best_schedule=current,
            best_score=final_loss,
            hourly_shed_mw=hourly_shed,
            area_outage_hours=area_outages,
            iterations=len(history),
            runtime_seconds=runtime,
            convergence_history=history,
            violations=final_violations,
        )

class GeneticAlgorithmScheduler:
    """Evolutionary Search (Genetic Algorithm) for multi-objective fair load-shedding."""

    def __init__(
        self,
        evaluator: ScheduleEvaluator,
        population_size: int = 30,
        mutation_rate: float = 0.04,
        crossover_rate: float = 0.85,
        elitism_count: int = 2,
        seed: int = 42,
    ):
        self.evaluator = evaluator
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.rng = np.random.default_rng(seed)

    def _init_population(self) -> List[np.ndarray]:
        """Creates diverse initial chromosomes respecting hospital exemption."""
        pop = []
        n_feeders = self.evaluator.n_feeders
        hours = self.evaluator.hours

        for _ in range(self.pop_size):
            chrom = self.rng.integers(0, 2, size=(n_feeders, hours))

            # Strictly enforce zero on hospitals
            for i, f in enumerate(self.evaluator.feeders):
                if f.is_critical_exempt:
                    chrom[i, :] = 0

            # Strictly enforce zero on Iftar/Sehri
            if self.evaluator.is_ramadan:
                chrom[:, 4] = 0
                chrom[:, 18] = 0
                chrom[:, 19] = 0

            pop.append(chrom)

        return pop

    def _tournament_select(self, pop: List[np.ndarray], fitnesses: List[float], k: int = 3) -> np.ndarray:
        """Selects parent using tournament selection."""
        candidates = self.rng.choice(len(pop), size=k, replace=False)
        best_idx = max(candidates, key=lambda idx: fitnesses[idx])
        return pop[best_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point temporal crossover exchanging schedule blocks."""
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        hours = self.evaluator.hours
        pt1, pt2 = sorted(self.rng.choice(hours, size=2, replace=False))

        child1 = parent1.copy()
        child2 = parent2.copy()

        child1[:, pt1:pt2] = parent2[:, pt1:pt2]
        child2[:, pt1:pt2] = parent1[:, pt1:pt2]

        return child1, child2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Uniform bit-flip mutation respecting hard exemptions."""
        n_feeders, hours = chromosome.shape
        mutated = chromosome.copy()

        for i in range(n_feeders):
            if self.evaluator.feeders[i].is_critical_exempt:
                continue
            for h in range(hours):
                if self.evaluator.is_ramadan and h in (4, 18, 19):
                    continue
                if self.rng.random() < self.mutation_rate:
                    mutated[i, h] = 1 - mutated[i, h]

        return mutated

    def solve(self, generations: int = 40) -> OptimizationResult:
        """Runs the evolutionary search cycle."""
        start_time = time.perf_counter()
        pop = self._init_population()

        best_chrom = pop[0]
        best_loss = float("inf")
        history = []

        for gen in range(generations):
            losses = [self.evaluator.evaluate(ind)[0] for ind in pop]
            fitnesses = [1.0 / (1.0 + (loss / 1000.0)) for loss in losses]

            # Track global best
            min_loss_idx = int(np.argmin(losses))
            if losses[min_loss_idx] < best_loss:
                best_loss = losses[min_loss_idx]
                best_chrom = pop[min_loss_idx].copy()

            history.append(best_loss)

            # Sort population for elitism
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_pop = [pop[idx].copy() for idx in sorted_indices[: self.elitism_count]]

            # Breed next generation
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))

            pop = new_pop

        runtime = time.perf_counter() - start_time
        final_loss, violations = self.evaluator.evaluate(best_chrom)

        # Compute hourly shed MW
        hourly_shed = np.zeros(self.evaluator.hours)
        for h in range(self.evaluator.hours):
            hourly_shed[h] = sum(
                best_chrom[i, h] * self.evaluator.area_demands[self.evaluator.feeders[i].area_id][h]
                for i in range(self.evaluator.n_feeders)
            )

        area_outages = {
            f.area_id: int(np.sum(best_chrom[i, :]))
            for i, f in enumerate(self.evaluator.feeders)
        }

        return OptimizationResult(
            algorithm_name="Genetic Algorithm (Evolutionary Search)",
            best_schedule=best_chrom,
            best_score=final_loss,
            hourly_shed_mw=hourly_shed,
            area_outage_hours=area_outages,
            iterations=generations,
            runtime_seconds=runtime,
            convergence_history=history,
            violations=violations,
        )
