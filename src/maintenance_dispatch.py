"""Small maintenance-team dispatch feature using UCS routing."""

from dataclasses import dataclass
from datetime import datetime
from collections import deque
import heapq
from typing import Dict, Iterable, List, Optional, Tuple


AREA_IDS = [
    "F01_HOSP", "F02_IND", "F03_COMM_A", "F04_RES_A", "F05_RES_B",
    "F06_SUB_A", "F07_SUB_B", "F08_RUR", "F09_COMM_B", "F10_RES_C",
]

TEAM_IDS = ("TEAM_A", "TEAM_B", "TEAM_C")


@dataclass
class MaintenanceTeam:
    id: str
    name: str
    base: str
    status: str = "available"
    assigned_area_id: Optional[str] = None
    assigned_complaint_id: Optional[str] = None
    travel_minutes: Optional[int] = None


@dataclass
class ComplaintResult:
    complaint_id: str
    accepted: bool
    status: str
    area_id: str
    hour: int
    assigned_team_id: Optional[str] = None
    travel_minutes: Optional[int] = None
    message: str = ""


class MaintenanceDispatchService:
    """Tracks 3 demo teams and compares search algorithms on a fixed graph."""

    def __init__(self) -> None:
        self.teams: Dict[str, MaintenanceTeam] = {
            "TEAM_A": MaintenanceTeam("TEAM_A", "Team A", "BASE_A"),
            "TEAM_B": MaintenanceTeam("TEAM_B", "Team B", "BASE_B"),
            "TEAM_C": MaintenanceTeam("TEAM_C", "Team C", "BASE_C"),
        }
        self.complaints: Dict[str, dict] = {}
        self._complaint_counter = 0

        # Direct base-to-area travel times. Other area-to-area links form a
        # small road graph, so routes are resolved with search rather than lookup.
        self.base_to_area_minutes = {
            "TEAM_A": [8, 7, 8, 9, 10, 11, 12, 14, 9, 10],
            "TEAM_B": [10, 9, 5, 6, 7, 8, 9, 11, 7, 8],
            "TEAM_C": [12, 11, 10, 9, 8, 7, 6, 8, 12, 9],
        }

    def _build_graph(self) -> Dict[str, List[Tuple[str, int]]]:
        graph: Dict[str, List[Tuple[str, int]]] = {
            team.base: [] for team in self.teams.values()
        }
        graph.update({area_id: [] for area_id in AREA_IDS})

        for team in self.teams.values():
            times = self.base_to_area_minutes[team.id]
            for area_id, travel in zip(AREA_IDS, times):
                graph[team.base].append((area_id, travel))
                graph[area_id].append((team.base, travel))

        for index, area_id in enumerate(AREA_IDS):
            next_area = AREA_IDS[(index + 1) % len(AREA_IDS)]
            previous_area = AREA_IDS[(index - 1) % len(AREA_IDS)]
            graph[area_id].append((next_area, 3))
            graph[area_id].append((previous_area, 3))
        return graph

    def find_route_bfs(self, team_id: str, area_id: str) -> Optional[List[str]]:
        team = self.teams.get(team_id)
        if team is None or area_id not in AREA_IDS:
            return None

        graph = self._build_graph()
        frontier = deque([(team.base, [team.base])])
        visited = {team.base}
        while frontier:
            node, route = frontier.popleft()
            if node == area_id:
                return route
            for neighbor, _ in graph.get(node, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                frontier.append((neighbor, route + [neighbor]))
        return None

    def find_route_dfs(self, team_id: str, area_id: str) -> Optional[List[str]]:
        team = self.teams.get(team_id)
        if team is None or area_id not in AREA_IDS:
            return None

        graph = self._build_graph()
        frontier = [(team.base, [team.base])]
        visited = set()
        while frontier:
            node, route = frontier.pop()
            if node == area_id:
                return route
            if node in visited:
                continue
            visited.add(node)
            for neighbor, _ in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    frontier.append((neighbor, route + [neighbor]))
        return None

    def find_route_ucs(self, team_id: str, area_id: str) -> Optional[List[str]]:
        team = self.teams.get(team_id)
        if team is None or area_id not in AREA_IDS:
            return None
        graph = self._build_graph()
        start = team.base
        frontier: List[Tuple[int, str, List[str]]] = [(0, start, [start])]
        visited: Dict[str, int] = {}

        while frontier:
            cost, node, route = heapq.heappop(frontier)
            if node == area_id:
                return route
            if node in visited and visited[node] <= cost:
                continue
            visited[node] = cost
            for neighbor, edge_cost in graph.get(node, []):
                heapq.heappush(frontier, (cost + edge_cost, neighbor, route + [neighbor]))
        return None

    def find_route_astar(self, team_id: str, area_id: str) -> Optional[List[str]]:
        team = self.teams.get(team_id)
        if team is None or area_id not in AREA_IDS:
            return None

        graph = self._build_graph()
        start = team.base

        # The demo graph has no real coordinates, so 0 is a safe admissible
        # heuristic and produces the true minimum travel time.
        frontier: List[Tuple[int, int, str, List[str]]] = [(0, 0, start, [start])]
        visited: Dict[str, int] = {}

        while frontier:
            _, cost, node, route = heapq.heappop(frontier)
            if node == area_id:
                return route
            if node in visited and visited[node] <= cost:
                continue
            visited[node] = cost
            for neighbor, edge_cost in graph.get(node, []):
                next_cost = cost + edge_cost
                heapq.heappush(
                    frontier,
                    (next_cost, next_cost, neighbor, route + [neighbor]),
                )
        return None

    def submit_complaint(
        self,
        area_id: str,
        hour: int,
        scheduled_off: bool,
        now: Optional[datetime] = None,
    ) -> ComplaintResult:
        now = now or datetime.now()
        self._complaint_counter += 1
        complaint_id = f"CMP-{now.strftime('%Y%m%d')}-{self._complaint_counter:04d}"

        if area_id not in AREA_IDS:
            return ComplaintResult(
                complaint_id, False, "rejected", area_id, hour,
                message="Unknown area.",
            )
        if scheduled_off:
            record = {
                "id": complaint_id, "area_id": area_id, "hour": hour,
                "status": "planned_outage", "accepted": False,
                "team_id": None, "travel_minutes": None,
                "created_at": now.isoformat(),
            }
            self.complaints[complaint_id] = record
            return ComplaintResult(
                complaint_id, False, "planned_outage", area_id, hour,
                message="This area is already in the planned load-shedding schedule.",
            )

        record = {
            "id": complaint_id, "area_id": area_id, "hour": hour,
            "status": "pending_admin", "accepted": False,
            "team_id": None, "travel_minutes": None,
            "created_at": now.isoformat(),
        }
        self.complaints[complaint_id] = record
        return ComplaintResult(
            complaint_id, True, "pending_admin", area_id, hour,
            message="Complaint sent to admin for review. No team has been assigned yet.",
        )

    def get_recommended_team(self, area_id: str) -> Optional[Tuple[str, int]]:
        available: List[Tuple[int, str]] = []
        for team in self.teams.values():
            if team.status != "available":
                continue
            route = self.find_route_ucs(team.id, area_id)
            if route:
                available.append((self._route_cost(route), team.id))
        if not available:
            return None
        travel_minutes, team_id = min(available, key=lambda item: (item[0], item[1]))
        return team_id, travel_minutes

    def get_team_route_options(self, area_id: str) -> dict:
        options = []
        for team in self.teams.values():
            if team.status != "available":
                continue
            team_routes = {}
            for algorithm_key, route_method in (
                ("bfs", self.find_route_bfs),
                ("dfs", self.find_route_dfs),
                ("ucs", self.find_route_ucs),
                ("astar", self.find_route_astar),
            ):
                route = route_method(team.id, area_id)
                if route is not None:
                    team_routes[algorithm_key] = {
                        "route": route,
                        "travel_minutes": self._route_cost(route),
                    }
            route_minutes = [
                route["travel_minutes"] for route in team_routes.values()
            ]
            options.append({
                "team_id": team.id,
                "team_name": team.name,
                "bfs": team_routes.get("bfs"),
                "dfs": team_routes.get("dfs"),
                "ucs": team_routes.get("ucs"),
                "astar": team_routes.get("astar"),
                "best_travel_minutes": min(route_minutes) if route_minutes else None,
            })

        options.sort(
            key=lambda item: (
                item["best_travel_minutes"] is None,
                item["best_travel_minutes"] if item["best_travel_minutes"] is not None else 9999,
                item["team_id"],
            )
        )
        recommended_team_ids = {}
        for algorithm_key in ("bfs", "dfs", "ucs", "astar"):
            candidates = [
                item for item in options if item[algorithm_key] is not None
            ]
            recommended_team_ids[algorithm_key] = min(
                candidates,
                key=lambda item: (
                    item[algorithm_key]["travel_minutes"],
                    item["team_id"],
                ),
            )["team_id"] if candidates else None

        ucs_recommended_team_id = recommended_team_ids["ucs"]
        astar_recommended_team_id = recommended_team_ids["astar"]
        return {
            "recommended_team_id": ucs_recommended_team_id,
            "bfs_recommended_team_id": recommended_team_ids["bfs"],
            "dfs_recommended_team_id": recommended_team_ids["dfs"],
            "ucs_recommended_team_id": ucs_recommended_team_id,
            "astar_recommended_team_id": astar_recommended_team_id,
            "options": options,
        }

    def assign_complaint(
        self,
        complaint_id: str,
        team_id: str,
        algorithm: str = "UCS",
    ) -> Optional[dict]:
        complaint = self.complaints.get(complaint_id)
        team = self.teams.get(team_id)
        if complaint is None or team is None:
            return None
        if complaint["status"] != "pending_admin":
            return None
        if team.status != "available":
            return None

        normalized_algorithm = str(algorithm).strip().upper()
        if normalized_algorithm in ("A*", "ASTAR"):
            normalized_algorithm = "A*"
        route_methods = {
            "BFS": self.find_route_bfs,
            "DFS": self.find_route_dfs,
            "UCS": self.find_route_ucs,
            "A*": self.find_route_astar,
        }
        route_method = route_methods.get(normalized_algorithm, self.find_route_ucs)
        route = route_method(team_id, complaint["area_id"])
        if route is None:
            return None
        travel_minutes = self._route_cost(route)
        team.status = "busy"
        team.assigned_area_id = complaint["area_id"]
        team.assigned_complaint_id = complaint_id
        team.travel_minutes = travel_minutes
        complaint["status"] = "dispatched"
        complaint["accepted"] = True
        complaint["team_id"] = team_id
        complaint["travel_minutes"] = travel_minutes
        complaint["assignment_algorithm"] = (
            normalized_algorithm if normalized_algorithm in route_methods else "UCS"
        )
        return complaint

    def _route_cost(self, route: Iterable[str]) -> int:
        nodes = list(route)
        graph = self._build_graph()
        total = 0
        for start, end in zip(nodes, nodes[1:]):
            total += next(cost for node, cost in graph[start] if node == end)
        return total

    def complete_team(self, team_id: str) -> bool:
        team = self.teams.get(team_id)
        if team is None:
            return False
        if team.assigned_complaint_id:
            complaint = self.complaints.get(team.assigned_complaint_id)
            if complaint:
                complaint["status"] = "completed"
        team.status = "available"
        team.assigned_area_id = None
        team.assigned_complaint_id = None
        team.travel_minutes = None
        return True

    def status(self) -> dict:
        complaints = []
        for complaint in self.complaints.values():
            record = dict(complaint)
            if complaint["status"] == "pending_admin":
                record["dispatch_options"] = self.get_team_route_options(complaint["area_id"])
            complaints.append(record)
        return {
            "teams": [
                {
                    "id": team.id,
                    "name": team.name,
                    "status": team.status,
                    "assigned_area_id": team.assigned_area_id,
                    "assigned_complaint_id": team.assigned_complaint_id,
                    "travel_minutes": team.travel_minutes,
                }
                for team in self.teams.values()
            ],
            "complaints": complaints,
        }
