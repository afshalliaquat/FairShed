"""Unit tests for Smart Grid Agents and Environment (Task 1)."""

import pytest
import numpy as np
from src.city_grid import CityGridNetwork, get_default_city_feeders
from src.agent_environment import (
    GridEnvironment,
    PEASSpecification,
    SimpleReflexGridAgent,
    ModelBasedGridAgent,
    GoalBasedGridAgent,
    UtilityBasedFairGridAgent,
)


def test_peas_specification():
    peas = PEASSpecification()
    desc = peas.describe()
    assert "PEAS SPECIFICATION" in desc
    assert "Hospital" in desc
    assert "Breaker" in desc


def test_grid_environment_percept_and_action():
    grid = CityGridNetwork(is_ramadan=True)
    env = GridEnvironment(grid, is_ramadan=True, noise_std_mw=0.0)

    percept = env.get_percept(city_demand_mw=15000.0, generation_capacity_mw=13000.0, step=20)
    assert percept.hour == 20
    assert percept.shortfall_mw == 2000.0
    assert not percept.is_iftar_time  # 20 is after 19

    # Agent action test
    agent = UtilityBasedFairGridAgent("UtilityAgent", grid)
    action = agent.select_action(percept)
    assert len(action.shed_area_ids) > 0

    # Ensure hospital is NOT shed
    assert "F01_HOSP" not in action.shed_area_ids

    penalty, violations = env.execute_action(action, percept.area_demands)
    assert len(violations) == 0  # Hospital and Ramadan honored


def test_agent_architectures_at_iftar():
    grid = CityGridNetwork(is_ramadan=True)
    env = GridEnvironment(grid, is_ramadan=True, noise_std_mw=0.0)

    # Hour 18:00 (Iftar)
    percept = env.get_percept(city_demand_mw=16000.0, generation_capacity_mw=14000.0, step=18)
    assert percept.is_iftar_time is True

    agents = [
        SimpleReflexGridAgent("Reflex", grid),
        ModelBasedGridAgent("Model", grid),
        GoalBasedGridAgent("Goal", grid),
        UtilityBasedFairGridAgent("Utility", grid),
    ]

    for ag in agents:
        act = ag.select_action(percept)
        # No protected feeder should be shed
        for f_id in act.shed_area_ids:
            feeder = grid.get_feeder_by_id(f_id)
            assert not feeder.is_critical_exempt
            assert not feeder.is_protected_at_hour(18, is_ramadan=True)
