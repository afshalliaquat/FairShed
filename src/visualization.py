"""Visualization module for the Fair Load-Shedding Scheduler.

Generates publication-quality charts using Matplotlib:
1. Fairness Bar Chart: Outage hours per city area (highlighting hospital exemption & equality).
2. 24-Hour Load-Shedding Heatmap: Feeder vs Hour matrix showing power status.
3. Grid Shortfall Dispatch Curve: Hourly Demand vs Available Generation vs Load Shed.
4. Algorithm Benchmark Comparison: Jain's Fairness Index & Shortfall Fulfillment.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Ensure matplotlib uses a local writable cache dir in sandbox
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".matplotlib_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_CACHE_DIR)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import numpy as np

from src.city_grid import FeederArea

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_fairness_outage_chart(
    outage_hours_by_algo: Dict[str, Dict[str, int]],
    feeders: List[FeederArea],
    save_path: Optional[Path] = None,
) -> Path:
    """Plots grouped bar chart of outage hours per feeder across scheduling algorithms."""
    if save_path is None:
        save_path = FIGURES_DIR / "fairness_outage_chart.png"

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6))

    area_names = [f.name.split("/")[0].strip() for f in feeders]
    n_areas = len(feeders)
    n_algos = len(outage_hours_by_algo)

    x = np.arange(n_areas)
    width = 0.8 / max(1, n_algos)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (algo_name, outages_dict) in enumerate(outage_hours_by_algo.items()):
        hours_values = [outages_dict.get(f.area_id, 0) for f in feeders]
        offset = (idx - (n_algos - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            hours_values,
            width,
            label=algo_name,
            color=colors[idx % len(colors)],
            alpha=0.88,
            edgecolor="black",
            linewidth=0.8,
        )
        # Add numerical labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{int(height)}h",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    # Highlight hospital zone
    for i, f in enumerate(feeders):
        if f.is_critical_exempt:
            ax.axvspan(i - 0.45, i + 0.45, color="lightgreen", alpha=0.25, label="Hospital Exempt (0h)" if i == 0 else "")

    ax.set_title("Fairness Audit: 24-Hour Load-Shedding Outage Hours per City Area", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("City Sector / Distribution Feeder", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Blackout Duration (Hours / Day)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(area_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 6.5)
    ax.axhline(4.0, color="crimson", linestyle="--", linewidth=1.2, label="Max Quota Limit (4h)")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
def plot_load_shedding_heatmap(
    schedule_matrix: np.ndarray,
    feeders: List[FeederArea],
    algorithm_name: str,
    save_path: Optional[Path] = None,
) -> Path:
    """Renders a 24-hour binary schedule matrix heatmap (Feeder vs Hour)."""
    if save_path is None:
        save_path = FIGURES_DIR / "load_shedding_heatmap.png"

    fig, ax = plt.subplots(figsize=(13, 6))

    cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#d62728"])  # Green=Power ON, Red=Shed
    bounds = [-0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    cax = ax.imshow(schedule_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    feeder_labels = [f"{f.name[:28]} ({'EXEMPT' if f.is_critical_exempt else f'{f.base_load_mw}MW'})" for f in feeders]
    hours_labels = [f"{h:02d}:00" for h in range(24)]

    ax.set_yticks(np.arange(len(feeders)))
    ax.set_yticklabels(feeder_labels, fontsize=9)
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels(hours_labels, rotation=45, ha="right", fontsize=8)

    # Highlight Iftar (18, 19) and Sehri (4) columns
    ax.axvspan(3.5, 4.5, color="gold", alpha=0.35, label="Protected Sehri (04:00)")
    ax.axvspan(17.5, 19.5, color="cyan", alpha=0.35, label="Protected Iftar (18:00-19:00)")

    ax.set_title(f"24-Hour Load-Shedding Matrix ({algorithm_name})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of the Day", fontsize=11, fontweight="bold")

    # Legend for heatmap cells
    cbar = fig.colorbar(cax, ticks=[0, 1], orientation="horizontal", pad=0.18, shrink=0.4)
    cbar.ax.set_xticklabels(["Power ON (Energized)", "Load Shed (Blackout)"], fontweight="bold")

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=2, frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_demand_shortfall_dispatch(
    hours_vector: np.ndarray,
    demand_mw: np.ndarray,
    capacity_mw: np.ndarray,
    shortfall_mw: np.ndarray,
    shed_mw: np.ndarray,
    save_path: Optional[Path] = None,
) -> Path:
    """Plots 24-hour demand vs generation capacity vs load shed."""
    if save_path is None:
        save_path = FIGURES_DIR / "demand_shortfall_dispatch.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    hours = np.arange(len(demand_mw))

    # Top plot: Demand and Available Generation
    ax1.plot(hours, demand_mw, label="Forecast Demand (MW)", color="#1f77b4", linewidth=2.5, marker="o")
    ax1.plot(hours, capacity_mw, label="Available Generation Capacity (MW)", color="#2ca02c", linewidth=2.0, linestyle="--")
    ax1.fill_between(
        hours,
        capacity_mw,
        demand_mw,
        where=(demand_mw > capacity_mw),
        color="crimson",
        alpha=0.25,
        label="Generation Deficit (Shortfall)",
    )
    ax1.set_ylabel("Power Flow (MW)", fontsize=10, fontweight="bold")
    ax1.set_title("Urban Grid Power Balance: Demand Forecast vs Available Capacity", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Bottom plot: Shortfall vs Actual MW Shed
    ax2.bar(hours - 0.15, shortfall_mw, width=0.3, label="Required Shortfall (MW)", color="salmon", edgecolor="red")
    ax2.bar(hours + 0.15, shed_mw, width=0.3, label="Scheduled Shed Load (MW)", color="mediumseagreen", edgecolor="green")
    ax2.set_xlabel("Hour of the Day", fontsize=10, fontweight="bold")
    ax2.set_ylabel("MW Deficit / Shed", fontsize=10, fontweight="bold")
    ax2.set_xticks(hours)
    ax2.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right", fontsize=8)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
