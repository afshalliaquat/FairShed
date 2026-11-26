"""Task 2 (Lab 3): Uninformed Searching Algorithms for Load-Shedding Sub-Grids.

Implements pure Python uninformed search algorithms without external search libraries:
1. Breadth-First Search (BFS): Minimal feeder disconnection count.
2. Depth-First Search (DFS): Rapid feasible subset exploration.
3. Uniform Cost Search (UCS / Dijkstra): Minimum socio-economic disruption cost.

Applied to find the optimal set of feeder disconnect switches that satisfies the
hourly MW deficit while strictly honoring hospital and Ramadan constraints.
"""

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Dict, List, Tuple

from src.city_grid import FeederArea, CityGridNetwork


@dataclass
class SearchResult:
    algorithm_name: str
    target_shortfall_mw: float
    selected_feeder_ids: List[str]
    total_shed_mw: float
    overshed_mw: float
    total_cost: float
    nodes_expanded: int
    solution_found: bool


def _get_eligible_feeders(
    grid: CityGridNetwork, hour: int, is_ramadan: bool
) -> List[FeederArea]:
    return [
        f for f in grid.feeders
        if not f.is_critical_exempt and not f.is_protected_at_hour(hour, is_ramadan)
    ]


def breadth_first_search_shedding(
    grid: CityGridNetwork,
    hour: int,
    target_shortfall_mw: float,
    area_demands: Dict[str, float],
    is_ramadan: bool = True,
) -> SearchResult:
    if target_shortfall_mw <= 0.0:
        return SearchResult("BFS", 0.0, [], 0.0, 0.0, 0.0, 0, True)

    eligible = _get_eligible_feeders(grid, hour, is_ramadan)
    queue = deque([(0, 0.0, (), 0.0)])
    visited = set()
    nodes_expanded = 0

    while queue:
        idx, current_mw, shed_ids, current_cost = queue.popleft()
        nodes_expanded += 1

        if current_mw >= target_shortfall_mw:
            overshed = current_mw - target_shortfall_mw
            return SearchResult(
                algorithm_name="Breadth-First Search (BFS)",
                target_shortfall_mw=target_shortfall_mw,
                selected_feeder_ids=list(shed_ids),
                total_shed_mw=current_mw,
                overshed_mw=overshed,
                total_cost=current_cost,
                nodes_expanded=nodes_expanded,
                solution_found=True,
            )

        if idx >= len(eligible):
            continue

        state_key = (idx, round(current_mw, 1))
        if state_key in visited:
            continue
        visited.add(state_key)

        feeder = eligible[idx]
        feeder_load = area_demands.get(feeder.area_id, feeder.base_load_mw)

        queue.append(
            (
                idx + 1,
                current_mw + feeder_load,
                shed_ids + (feeder.area_id,),
                current_cost + feeder.disruption_cost_per_hour,
            )
        )

        queue.append((idx + 1, current_mw, shed_ids, current_cost))

    return SearchResult("Breadth-First Search (BFS)", target_shortfall_mw, [], 0.0, 0.0, 0.0, nodes_expanded, False)


def depth_first_search_shedding(
    grid: CityGridNetwork,
    hour: int,
    target_shortfall_mw: float,
    area_demands: Dict[str, float],
    is_ramadan: bool = True,
) -> SearchResult:
    if target_shortfall_mw <= 0.0:
        return SearchResult("DFS", 0.0, [], 0.0, 0.0, 0.0, 0, True)

    eligible = _get_eligible_feeders(grid, hour, is_ramadan)
    stack = [(0, 0.0, (), 0.0)]
    visited = set()
    nodes_expanded = 0

    while stack:
        idx, current_mw, shed_ids, current_cost = stack.pop()
        nodes_expanded += 1

        if current_mw >= target_shortfall_mw:
            overshed = current_mw - target_shortfall_mw
            return SearchResult(
                algorithm_name="Depth-First Search (DFS)",
                target_shortfall_mw=target_shortfall_mw,
                selected_feeder_ids=list(shed_ids),
                total_shed_mw=current_mw,
                overshed_mw=overshed,
                total_cost=current_cost,
                nodes_expanded=nodes_expanded,
                solution_found=True,
            )

        if idx >= len(eligible):
            continue

        state_key = (idx, round(current_mw, 1))
        if state_key in visited:
            continue
        visited.add(state_key)

        feeder = eligible[idx]
        feeder_load = area_demands.get(feeder.area_id, feeder.base_load_mw)

        stack.append((idx + 1, current_mw, shed_ids, current_cost))
        stack.append(
            (
                idx + 1,
                current_mw + feeder_load,
                shed_ids + (feeder.area_id,),
                current_cost + feeder.disruption_cost_per_hour,
            )
        )

    return SearchResult("Depth-First Search (DFS)", target_shortfall_mw, [], 0.0, 0.0, 0.0, nodes_expanded, False)


def uniform_cost_search_shedding(
    grid: CityGridNetwork,
    hour: int,
    target_shortfall_mw: float,
    area_demands: Dict[str, float],
    is_ramadan: bool = True,
) -> SearchResult:
    if target_shortfall_mw <= 0.0:
        return SearchResult("UCS", 0.0, [], 0.0, 0.0, 0.0, 0, True)

    eligible = _get_eligible_feeders(grid, hour, is_ramadan)
    pq: List[Tuple[float, int, int, float, Tuple[str, ...]]] = []
    counter = 0
    heapq.heappush(pq, (0.0, counter, 0, 0.0, ()))
    visited = {}
    nodes_expanded = 0

    while pq:
        cost, _, idx, current_mw, shed_ids = heapq.heappop(pq)
        nodes_expanded += 1

        if current_mw >= target_shortfall_mw:
            overshed = current_mw - target_shortfall_mw
            return SearchResult(
                algorithm_name="Uniform Cost Search (UCS)",
                target_shortfall_mw=target_shortfall_mw,
                selected_feeder_ids=list(shed_ids),
                total_shed_mw=current_mw,
                overshed_mw=overshed,
                total_cost=cost,
                nodes_expanded=nodes_expanded,
                solution_found=True,
            )

        state_key = (idx, round(current_mw, 1))
        if state_key in visited and visited[state_key] <= cost:
            continue
        visited[state_key] = cost

        if idx >= len(eligible):
            continue

        feeder = eligible[idx]
        feeder_load = area_demands.get(feeder.area_id, feeder.base_load_mw)

        counter += 1
        heapq.heappush(
            pq,
            (
                cost + feeder.disruption_cost_per_hour,
                counter,
                idx + 1,
                current_mw + feeder_load,
                shed_ids + (feeder.area_id,),
            ),
        )

        counter += 1
        heapq.heappush(pq, (cost, counter, idx + 1, current_mw, shed_ids))

    return SearchResult("Uniform Cost Search (UCS)", target_shortfall_mw, [], 0.0, 0.0, 0.0, nodes_expanded, False)
