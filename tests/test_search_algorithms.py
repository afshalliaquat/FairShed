"""Unit tests for Uninformed and Informed Searching Algorithms (Tasks 2 & 3)."""

from src.city_grid import CityGridNetwork
from src.uninformed_search import (
    breadth_first_search_shedding,
    depth_first_search_shedding,
    uniform_cost_search_shedding,
)
from src.informed_search import (
    greedy_best_first_search_shedding,
    a_star_search_shedding,
)


def test_uninformed_and_informed_search_coverage():
    grid = CityGridNetwork(is_ramadan=True)
    hour = 14
    shortfall = 500.0
    hour_demands = {f.area_id: f.base_load_mw for f in grid.feeders}

    # BFS
    res_bfs = breadth_first_search_shedding(grid, hour, shortfall, hour_demands)
    assert res_bfs.solution_found is True
    assert res_bfs.total_shed_mw >= shortfall
    assert "F01_HOSP" not in res_bfs.selected_feeder_ids

    # DFS
    res_dfs = depth_first_search_shedding(grid, hour, shortfall, hour_demands)
    assert res_dfs.solution_found is True
    assert res_dfs.total_shed_mw >= shortfall
    assert "F01_HOSP" not in res_dfs.selected_feeder_ids

    # UCS
    res_ucs = uniform_cost_search_shedding(grid, hour, shortfall, hour_demands)
    assert res_ucs.solution_found is True
    assert res_ucs.total_shed_mw >= shortfall
    assert "F01_HOSP" not in res_ucs.selected_feeder_ids

    # Greedy Best-First
    res_gbfs = greedy_best_first_search_shedding(grid, hour, shortfall, hour_demands)
    assert res_gbfs.solution_found is True
    assert res_gbfs.total_shed_mw >= shortfall
    assert "F01_HOSP" not in res_gbfs.selected_feeder_ids

    # A* Search
    res_astar = a_star_search_shedding(grid, hour, shortfall, hour_demands)
    assert res_astar.solution_found is True
    assert res_astar.total_shed_mw >= shortfall
    assert "F01_HOSP" not in res_astar.selected_feeder_ids
    # A* cost should be less than or equal to BFS cost
    assert res_astar.total_cost <= res_bfs.total_cost + 1e-5


def test_zero_shortfall_search_returns_empty():
    grid = CityGridNetwork(is_ramadan=True)
    hour = 12
    hour_demands = {f.area_id: f.base_load_mw for f in grid.feeders}

    res = a_star_search_shedding(grid, hour, target_shortfall_mw=0.0, area_demands=hour_demands)
    assert res.solution_found is True
    assert len(res.selected_feeder_ids) == 0
    assert res.total_shed_mw == 0.0
