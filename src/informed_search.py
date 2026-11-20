"""Task 3 (Lab 4): Informed (Heuristic) Searching Algorithms.

Implements pure Python heuristic search algorithms without external search libraries:
1. Greedy Best-First Search: Guided solely by heuristic distance to goal h(n).
2. A* Search (A-Star): Guided by evaluation function f(n) = g(n) + h(n) with
   admissible & consistent heuristic, proving optimal feeder shedding selection.
"""

from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional, Set, Tuple

from src.city_grid import FeederArea, CityGridNetwork
from src.uninformed_search import SearchResult, _get_eligible_feeders


def greedy_best_first_search_shedding(
    grid: CityGridNetwork,
    hour: int,
    target_shortfall_mw: float,
    area_demands: Dict[str, float],
    is_ramadan: bool = True,
) -> SearchResult:
    """Greedy Best-First Search: Prioritizes states minimizing heuristic distance to goal:

    h(n) = max(0, target_shortfall_mw - current_shed_mw)
    """
    if target_shortfall_mw <= 0.0:
        return SearchResult("Greedy Best-First", 0.0, [], 0.0, 0.0, 0.0, 0, True)

    eligible = _get_eligible_feeders(grid, hour, is_ramadan)
    # PQ tuple: (heuristic_h, counter, idx, current_mw, tuple_of_shed_ids, cost)
    pq: List[Tuple[float, int, int, float, Tuple[str, ...], float]] = []
    counter = 0

    init_h = target_shortfall_mw
    heapq.heappush(pq, (init_h, counter, 0, 0.0, (), 0.0))
    visited = set()
    nodes_expanded = 0

    while pq:
        h, _, idx, current_mw, shed_ids, current_cost = heapq.heappop(pq)
        nodes_expanded += 1

        if current_mw >= target_shortfall_mw:
            overshed = current_mw - target_shortfall_mw
            return SearchResult(
                algorithm_name="Greedy Best-First Search",
                target_shortfall_mw=target_shortfall_mw,
                selected_feeder_ids=list(shed_ids),
                total_shed_mw=current_mw,
                overshed_mw=overshed,
                total_cost=current_cost,
                nodes_expanded=nodes_expanded,
                solution_found=True,
            )

        state_key = (idx, round(current_mw, 1))
        if state_key in visited:
            continue
        visited.add(state_key)

        if idx >= len(eligible):
            continue

        feeder = eligible[idx]
        feeder_load = area_demands.get(feeder.area_id, feeder.base_load_mw)

        # Child 1: Shed this feeder
        new_mw = current_mw + feeder_load
        h1 = max(0.0, target_shortfall_mw - new_mw)
        counter += 1
        heapq.heappush(
            pq,
            (
                h1,
                counter,
                idx + 1,
                new_mw,
                shed_ids + (feeder.area_id,),
                current_cost + feeder.disruption_cost_per_hour,
            ),
        )

        # Child 2: Keep powered
        h2 = max(0.0, target_shortfall_mw - current_mw)
        counter += 1
        heapq.heappush(pq, (h2, counter, idx + 1, current_mw, shed_ids, current_cost))

    return SearchResult("Greedy Best-First Search", target_shortfall_mw, [], 0.0, 0.0, 0.0, nodes_expanded, False)


def a_star_search_shedding(
    grid: CityGridNetwork,
    hour: int,
    target_shortfall_mw: float,
    area_demands: Dict[str, float],
    is_ramadan: bool = True,
) -> SearchResult:
    """A* Search: Evaluates nodes using f(n) = g(n) + h(n).

    g(n) = Accumulated socio-economic disruption cost.
    h(n) = Admissible heuristic: Remaining deficit * min_cost_per_mw among remaining feeders.
           Since min_cost_per_mw <= actual_cost_per_mw for every combination, h(n) is
           strictly admissible (never overestimates the true remaining cost) and monotonic/consistent.
    Guarantees optimal solution with minimal node expansions.
    """
    if target_shortfall_mw <= 0.0:
        return SearchResult("A* Search", 0.0, [], 0.0, 0.0, 0.0, 0, True)

    eligible = _get_eligible_feeders(grid, hour, is_ramadan)

    # Calculate min unit cost across all eligible feeders for admissibility
    unit_costs = [
        f.disruption_cost_per_hour / max(1.0, area_demands.get(f.area_id, f.base_load_mw))
        for f in eligible
    ]
    min_unit_cost = min(unit_costs) if unit_costs else 0.0

    def compute_admissible_heuristic(remaining_shortfall: float) -> float:
        return max(0.0, remaining_shortfall) * min_unit_cost

    # PQ tuple: (f_score, counter, g_cost, idx, current_mw, tuple_of_shed_ids)
    pq: List[Tuple[float, int, float, int, float, Tuple[str, ...]]] = []
    counter = 0

    init_h = compute_admissible_heuristic(target_shortfall_mw)
    heapq.heappush(pq, (init_h, counter, 0.0, 0, 0.0, ()))
    visited = {}
    nodes_expanded = 0

    while pq:
        f_score, _, g_cost, idx, current_mw, shed_ids = heapq.heappop(pq)
        nodes_expanded += 1

        if current_mw >= target_shortfall_mw:
            overshed = current_mw - target_shortfall_mw
            return SearchResult(
                algorithm_name="A* Search",
                target_shortfall_mw=target_shortfall_mw,
                selected_feeder_ids=list(shed_ids),
                total_shed_mw=current_mw,
                overshed_mw=overshed,
                total_cost=g_cost,
                nodes_expanded=nodes_expanded,
                solution_found=True,
            )

        state_key = (idx, round(current_mw, 1))
        if state_key in visited and visited[state_key] <= g_cost:
            continue
        visited[state_key] = g_cost

        if idx >= len(eligible):
            continue

        feeder = eligible[idx]
        feeder_load = area_demands.get(feeder.area_id, feeder.base_load_mw)

        # Child 1: Shed feeder
        new_g1 = g_cost + feeder.disruption_cost_per_hour
        new_mw1 = current_mw + feeder_load
        h1 = compute_admissible_heuristic(target_shortfall_mw - new_mw1)
        counter += 1
        heapq.heappush(
            pq,
            (
                new_g1 + h1,
                counter,
                new_g1,
                idx + 1,
                new_mw1,
                shed_ids + (feeder.area_id,),
            ),
        )

        # Child 2: Keep powered
        new_g2 = g_cost
        h2 = compute_admissible_heuristic(target_shortfall_mw - current_mw)
        counter += 1
        heapq.heappush(pq, (new_g2 + h2, counter, new_g2, idx + 1, current_mw, shed_ids))

    return SearchResult("A* Search", target_shortfall_mw, [], 0.0, 0.0, 0.0, nodes_expanded, False)
