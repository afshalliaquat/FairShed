"""Unit tests for CSP Backtracking and Optimization Schedulers (Tasks 4 & 5)."""

import numpy as np
from src.city_grid import CityGridNetwork
from src.csp_scheduler import FairLoadSheddingCSP
from src.local_evolutionary_search import (
    ScheduleEvaluator,
    HillClimbingScheduler,
    GeneticAlgorithmScheduler,
)


def test_csp_backtracking_hospital_and_ramadan_guarantee():
    grid = CityGridNetwork(is_ramadan=True)
    # 24-hour synthetic deficit
    hourly_shortfall = np.array([0.0]*10 + [400.0, 500.0, 600.0, 300.0] + [0.0]*4 + [700.0, 800.0] + [0.0]*4)
    demands = grid.calculate_hourly_area_demands(np.full(24, 15000.0))

    solver = FairLoadSheddingCSP(
        grid_network=grid,
        hourly_shortfall=hourly_shortfall,
        area_demands=demands,
        max_daily_outage_per_area=4,
        max_consecutive_hours=2,
        is_ramadan=True,
    )
    result = solver.solve()

    # 1. Hospital index 0 must have exactly 0 hours of outage
    assert result.area_outage_hours["F01_HOSP"] == 0

    # 2. Hours 18, 19 (Iftar) and Hour 4 (Sehri) must have zero outages
    assert np.all(result.schedule_matrix[:, 4] == 0)
    assert np.all(result.schedule_matrix[:, 18] == 0)
    assert np.all(result.schedule_matrix[:, 19] == 0)

    # 3. No area exceeds max daily quota
    for f in grid.feeders:
        assert result.area_outage_hours[f.area_id] <= 4


def test_hill_climbing_and_genetic_algorithm():
    grid = CityGridNetwork(is_ramadan=True)
    shortfall = np.array([0.0]*12 + [500.0]*4 + [0.0]*8)
    demands = grid.calculate_hourly_area_demands(np.full(24, 15000.0))

    evaluator = ScheduleEvaluator(grid, shortfall, demands, max_daily_outage=4, max_consecutive=2, is_ramadan=True)

    # Hill Climbing
    hc = HillClimbingScheduler(evaluator, seed=42)
    res_hc = hc.solve_steepest_ascent(max_iterations=30)
    assert res_hc.best_schedule.shape == (len(grid.feeders), 24)
    assert res_hc.area_outage_hours["F01_HOSP"] == 0

    # Genetic Algorithm
    ga = GeneticAlgorithmScheduler(evaluator, population_size=10, seed=42)
    res_ga = ga.solve(generations=10)
    assert res_ga.best_schedule.shape == (len(grid.feeders), 24)
    assert res_ga.area_outage_hours["F01_HOSP"] == 0
