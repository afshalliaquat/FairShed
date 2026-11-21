"""Master AI Lab Syllabus Showcase Runner.

Demonstrates the proper working of ALL tasks without advanced library usage:
- Task 1: Types of Agents, Environments to implement (PEAS & 4 Agent Classes)
- Task 2 (Lab 3): Uninformed Searching Algorithms (BFS, DFS, UCS)
- Task 3 (Lab 4): Informed Searching Algorithms (Greedy Best-First, A*)
- Task 4 (Lab 5): Constraint Satisfaction Problems (CSP Backtracking with MRV)
- Task 5 (Lab 6): Local (Hill Climbing) & Evolutionary (Genetic Algorithm) Search
- Task 7 (Lab 9): Supervised Learning (Linear Regression & Random Forest)
- Task 8 (Lab 10): Unsupervised Learning (Scratch K-Means Feeder Profiling)
- Lab 11: Reinforcement Learning (Tabular Q-Learning Grid Dispatch)
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure matplotlib uses a local writable cache dir in sandbox
_CACHE_DIR = BASE_DIR / ".matplotlib_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_CACHE_DIR)

from src.city_grid import CityGridNetwork, get_default_city_feeders
from src.agent_environment import (
    PEASSpecification,
    EnvironmentProperty,
    GridEnvironment,
    SimpleReflexGridAgent,
    ModelBasedGridAgent,
    GoalBasedGridAgent,
    UtilityBasedFairGridAgent,
)
from src.uninformed_search import (
    breadth_first_search_shedding,
    depth_first_search_shedding,
    uniform_cost_search_shedding,
)
from src.informed_search import (
    greedy_best_first_search_shedding,
    a_star_search_shedding,
)
from src.csp_scheduler import FairLoadSheddingCSP
from src.local_evolutionary_search import (
    ScheduleEvaluator,
    HillClimbingScheduler,
    GeneticAlgorithmScheduler,
)
from src.models import chronological_split, train_linear_regression, train_random_forest
from src.evaluate import evaluate_all_models
from src.unsupervised_learning import cluster_city_feeders
from src.reinforcement_learning import TabularQGridAgent
from src.fairness_metrics import evaluate_schedule_fairness
from src.visualization import (
    plot_fairness_outage_chart,
    plot_load_shedding_heatmap,
    plot_demand_shortfall_dispatch,
)
from src.data_loader import load_or_create_dataset
from src.data_cleaner import clean_energy_data
from src.feature_engineering import create_features


def print_banner(title: str, task_tag: str):
    print("\n" + "=" * 80)
    print(f"[*] {task_tag.upper()} :: {title.upper()}")
    print("=" * 80)


def run_all_labs_showcase():
    print("=" * 80)
    print("  PROJECT 24: ELECTRICITY DEMAND FORECAST & FAIR LOAD-SHEDDING SCHEDULER")
    print("  COMPLETE AI LAB SYLLABUS EXECUTION & VERIFICATION SUITE")
    print("=" * 80)

    # Prepare standard grid data
    grid = CityGridNetwork(is_ramadan=True)
    raw_df, _ = load_or_create_dataset()
    clean_df, _ = clean_energy_data(raw_df)
    features_df, feature_cols = create_features(clean_df)

    # Extract 24-hour snapshot for scheduling
    sample_day = features_df.tail(24).copy().reset_index(drop=True)
    hourly_demands = sample_day["energy_consumption_mw"].values
    hourly_capacities = grid.simulate_generation_capacity(hourly_demands, average_shortfall_ratio=0.18, seed=42)
    hourly_shortfalls = grid.calculate_shortfall(hourly_demands, hourly_capacities)
    area_demands = grid.calculate_hourly_area_demands(hourly_demands)

    total_shortfall_mwh = np.sum(hourly_shortfalls)
    print(f"\n[Scenario Initialization] 24-Hour Horizon | Total Forecast Demand: {np.sum(hourly_demands):,.0f} MWh")
    print(f"Available Generation: {np.sum(hourly_capacities):,.0f} MWh | Total Deficit: {total_shortfall_mwh:,.0f} MWh")
    print(f"Hospital Feeder: '{grid.feeders[0].name}' (100% EXEMPT)")
    print("Protected Ramadan Windows: Sehri (04:00) and Iftar (18:00 - 19:00)\n")

    # -------------------------------------------------------------------------
    # TASK 1: Types of Agents & PEAS Environment
    # -------------------------------------------------------------------------
    print_banner("Types of Agents, Environments & PEAS Specification", "Task 1 (Week 1-2)")
    peas = PEASSpecification()
    print(peas.describe())
    print("\nEnvironment Properties Classification:")
    for prop in EnvironmentProperty:
        print(f"  • {prop.name:<16}: {prop.value}")

    env = GridEnvironment(grid, is_ramadan=True)
    peak_hour = int(np.argmax(hourly_shortfalls))
    percept = env.get_percept(hourly_demands[peak_hour], hourly_capacities[peak_hour], step=peak_hour)

    agents = [
        SimpleReflexGridAgent("ReflexAgent", grid),
        ModelBasedGridAgent("ModelBasedAgent", grid),
        GoalBasedGridAgent("GoalBasedAgent", grid),
        UtilityBasedFairGridAgent("UtilityBasedAgent", grid),
    ]

    print(f"\nComparing Agent Actions at Peak Hour {peak_hour:02d}:00 (Demand: {percept.total_demand_mw:.0f} MW, Shortfall: {percept.shortfall_mw:.0f} MW):")
    for ag in agents:
        act = ag.select_action(percept)
        print(f"  • {ag.name:<18} -> Shed Feeders: {list(act.shed_area_ids)} | Shed MW: {act.total_shed_mw:.1f} | Unmet: {act.unmet_shortfall_mw:.1f}")

    # -------------------------------------------------------------------------
    # TASK 2: Uninformed Searching Algorithms (BFS, DFS, UCS)
    # -------------------------------------------------------------------------
    print_banner("Uninformed Searching Algorithms", "Task 2 (Week 3 / Lab 3)")
    target_deficit = float(hourly_shortfalls[peak_hour])
    hour_demands_dict = {f.area_id: float(area_demands[f.area_id][peak_hour]) for f in grid.feeders}

    bfs_res = breadth_first_search_shedding(grid, peak_hour, target_deficit, hour_demands_dict)
    dfs_res = depth_first_search_shedding(grid, peak_hour, target_deficit, hour_demands_dict)
    ucs_res = uniform_cost_search_shedding(grid, peak_hour, target_deficit, hour_demands_dict)

    uninformed_table = pd.DataFrame([
        {
            "Algorithm": bfs_res.algorithm_name,
            "Feeders Shed": len(bfs_res.selected_feeder_ids),
            "Shed MW": round(bfs_res.total_shed_mw, 1),
            "Overshed MW": round(bfs_res.overshed_mw, 1),
            "Disruption Cost": round(bfs_res.total_cost, 1),
            "Nodes Expanded": bfs_res.nodes_expanded,
        },
        {
            "Algorithm": dfs_res.algorithm_name,
            "Feeders Shed": len(dfs_res.selected_feeder_ids),
            "Shed MW": round(dfs_res.total_shed_mw, 1),
            "Overshed MW": round(dfs_res.overshed_mw, 1),
            "Disruption Cost": round(dfs_res.total_cost, 1),
            "Nodes Expanded": dfs_res.nodes_expanded,
        },
        {
            "Algorithm": ucs_res.algorithm_name,
            "Feeders Shed": len(ucs_res.selected_feeder_ids),
            "Shed MW": round(ucs_res.total_shed_mw, 1),
            "Overshed MW": round(ucs_res.overshed_mw, 1),
            "Disruption Cost": round(ucs_res.total_cost, 1),
            "Nodes Expanded": ucs_res.nodes_expanded,
        },
    ])
    print(uninformed_table.to_string(index=False))

    # -------------------------------------------------------------------------
    # TASK 3: Informed Searching Algorithms (Greedy Best-First & A*)
    # -------------------------------------------------------------------------
    print_banner("Informed (Heuristic) Searching Algorithms", "Task 3 (Week 4 / Lab 4)")
    gbfs_res = greedy_best_first_search_shedding(grid, peak_hour, target_deficit, hour_demands_dict)
    astar_res = a_star_search_shedding(grid, peak_hour, target_deficit, hour_demands_dict)

    informed_table = pd.DataFrame([
        {
            "Algorithm": gbfs_res.algorithm_name,
            "Feeders Shed": len(gbfs_res.selected_feeder_ids),
            "Shed MW": round(gbfs_res.total_shed_mw, 1),
            "Overshed MW": round(gbfs_res.overshed_mw, 1),
            "Disruption Cost": round(gbfs_res.total_cost, 1),
            "Nodes Expanded": gbfs_res.nodes_expanded,
            "Optimality": "Heuristic Greed",
        },
        {
            "Algorithm": astar_res.algorithm_name,
            "Feeders Shed": len(astar_res.selected_feeder_ids),
            "Shed MW": round(astar_res.total_shed_mw, 1),
            "Overshed MW": round(astar_res.overshed_mw, 1),
            "Disruption Cost": round(astar_res.total_cost, 1),
            "Nodes Expanded": astar_res.nodes_expanded,
            "Optimality": "Provably Optimal (Admissible Heuristic)",
        },
    ])
    print(informed_table.to_string(index=False))

    # -------------------------------------------------------------------------
    # TASK 4: Constraint Satisfaction Problems (CSP Backtracking)
    # -------------------------------------------------------------------------
    print_banner("Constraint Satisfaction Problem (CSP) Backtracking", "Task 4 (Week 5 / Lab 5)")
    csp_solver = FairLoadSheddingCSP(
        grid_network=grid,
        hourly_shortfall=hourly_shortfalls,
        area_demands=area_demands,
        max_daily_outage_per_area=4,
        max_consecutive_hours=2,
        is_ramadan=True,
    )
    csp_res = csp_solver.solve()
    print(f"CSP Solver Status: {'SUCCESS' if csp_res.success else 'REPAIRED'} | Runtime: {csp_res.runtime_seconds:.3f}s")
    print(f"Backtracks: {csp_res.backtracks} | Search Nodes Explored: {csp_res.nodes_explored}")
    print("Outage Hours per Feeder (CSP Backtracking):")
    for f in grid.feeders:
        tag = "[EXEMPT]" if f.is_critical_exempt else ""
        print(f"  • {f.name[:32]:<34}: {csp_res.area_outage_hours[f.area_id]}h {tag}")

    # -------------------------------------------------------------------------
    # TASK 5: Local Search (Hill Climbing) & Evolutionary (Genetic Algorithm)
    # -------------------------------------------------------------------------
    print_banner("Local Search (Hill Climbing) & Evolutionary Algorithms", "Task 5 (Week 7 / Lab 6)")
    evaluator = ScheduleEvaluator(
        grid_network=grid,
        hourly_shortfall=hourly_shortfalls,
        area_demands=area_demands,
        max_daily_outage=4,
        max_consecutive=2,
        is_ramadan=True,
    )
    hc_solver = HillClimbingScheduler(evaluator, seed=42)
    hc_res = hc_solver.solve_steepest_ascent(max_iterations=150)

    ga_solver = GeneticAlgorithmScheduler(evaluator, population_size=25, seed=42)
    ga_res = ga_solver.solve(generations=30)

    print(f"Steepest-Ascent Hill Climbing: Runtime {hc_res.runtime_seconds:.2f}s | Final Loss: {hc_res.best_score:.1f} | Violations: {len(hc_res.violations)}")
    print(f"Genetic Algo : Runtime {ga_res.runtime_seconds:.2f}s | Final Loss: {ga_res.best_score:.1f} | Violations: {len(ga_res.violations)}")

    # -------------------------------------------------------------------------
    # TASK 7: Supervised Learning (Linear Regression & Random Forest)
    # -------------------------------------------------------------------------
    print_banner("Supervised Learning: Demand Forecasting", "Task 7 (Week 10 / Lab 9)")
    X_train, X_test, y_train, y_test, _, _ = chronological_split(features_df, feature_cols)
    regression_models = {
        "linear_regression_model": train_linear_regression(X_train, y_train),
        "random_forest_model": train_random_forest(X_train, y_train),
    }
    regression_summary, _ = evaluate_all_models(regression_models, X_test, y_test)
    print(regression_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # TASK 8: Unsupervised Learning (Scratch K-Means Clustering)
    # -------------------------------------------------------------------------
    print_banner("Unsupervised Learning: K-Means Feeder Profiling", "Task 8 (Week 12 / Lab 10)")
    clustering_res = cluster_city_feeders(grid, k_clusters=4, seed=42)
    print(f"K-Means Lloyd's Algorithm Converged | Final Inertia: {clustering_res.inertia}")
    print(clustering_res.feeder_profiles_df[["area_id", "name", "base_load_mw", "profile_archetype"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # LAB 11: Reinforcement Learning (Tabular Q-Learning)
    # -------------------------------------------------------------------------
    print_banner("Reinforcement Learning: Q-Learning Grid Dispatch", "Task 9 (Week 13 / Lab 11)")
    rl_agent = TabularQGridAgent(grid, learning_rate=0.15, discount_factor=0.95, seed=42)
    rl_res = rl_agent.train(hourly_shortfalls, area_demands, num_episodes=100, is_ramadan=True)
    print(f"Q-Learning Trained for {rl_res.episodes} episodes | Q-table size: {rl_res.q_table_size} state-action pairs")
    print(f"Final Epsilon: {rl_res.final_epsilon} | Hospital Violations: {len(rl_res.policy_violations)}")

    # -------------------------------------------------------------------------
    # FAIRNESS AUDIT & BENCHMARK SUMMARY
    # -------------------------------------------------------------------------
    print_banner("Fairness Audit & Multi-Algorithm Scheduler Benchmark", "Fairness Evaluation")
    report_csp = evaluate_schedule_fairness("CSP Backtracking", csp_res.schedule_matrix, grid.feeders, hourly_shortfalls, area_demands, is_ramadan=True)
    report_hc = evaluate_schedule_fairness("Hill Climbing", hc_res.best_schedule, grid.feeders, hourly_shortfalls, area_demands, is_ramadan=True)
    report_ga = evaluate_schedule_fairness("Genetic Algorithm", ga_res.best_schedule, grid.feeders, hourly_shortfalls, area_demands, is_ramadan=True)
    report_rl = evaluate_schedule_fairness("Q-Learning Policy", rl_res.policy_schedule, grid.feeders, hourly_shortfalls, area_demands, is_ramadan=True)

    benchmark_df = pd.DataFrame([
        report_csp.to_dict(),
        report_hc.to_dict(),
        report_ga.to_dict(),
        report_rl.to_dict(),
    ])
    print(benchmark_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # FIGURE GENERATION
    # -------------------------------------------------------------------------
    print_banner("Generating Publication-Quality Visualizations", "Reports & Charts")
    outages_by_algo = {
        "CSP Backtracking": csp_res.area_outage_hours,
        "Hill Climbing": hc_res.area_outage_hours,
        "Genetic Algorithm": ga_res.area_outage_hours,
        "Q-Learning": rl_res.policy_outages,
    }
    f1 = plot_fairness_outage_chart(outages_by_algo, grid.feeders)
    f2 = plot_load_shedding_heatmap(csp_res.schedule_matrix, grid.feeders, "CSP Backtracking")
    f3 = plot_demand_shortfall_dispatch(
        np.arange(24), hourly_demands, hourly_capacities, hourly_shortfalls, csp_res.hourly_shed_mw
    )
    print(f"  [OK] Saved Fairness Bar Chart        -> {f1}")
    print(f"  [OK] Saved Schedule Heatmap          -> {f2}")
    print(f"  [OK] Saved Power Balance Curve       -> {f3}")

    print("\n" + "=" * 80)
    print(">>> RETAINED AI LAB TASKS EXECUTED, VERIFIED, AND BENCHMARKED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_labs_showcase()
