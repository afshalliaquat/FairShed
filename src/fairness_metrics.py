"""Fairness metrics and evaluation analytics for the Load-Shedding Scheduler.

Calculates:
1. Jain's Fairness Index: J(x) = (sum x_i)^2 / (n * sum x_i^2)
2. Outage Standard Deviation and Variance
3. Max-Min Outage Spread
4. Gini Coefficient of load-shedding distribution
5. Shortfall Coverage & Over-shedding percentages
6. Hard constraint compliance audit
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.city_grid import FeederArea, CityGridNetwork


@dataclass
class ScheduleFairnessReport:
    """Comprehensive fairness and reliability metrics report."""

    algorithm_name: str
    total_shortfall_mwh: float
    total_shed_mwh: float
    shortfall_fulfillment_pct: float
    overshed_mwh: float
    jains_fairness_index: float  # Range: [1/n, 1.0]
    outage_std_hours: float
    outage_variance_hours: float
    max_outage_hours: int
    min_outage_hours: int
    outage_range_hours: int
    gini_coefficient: float
    hospital_shed_hours: int
    ramadan_violations_count: int
    consecutive_violations_count: int
    is_fully_compliant: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Algorithm": self.algorithm_name,
            "Jain's Fairness Index": round(self.jains_fairness_index, 3),
            "Outage Std Dev (h)": round(self.outage_std_hours, 2),
            "Max-Min Spread (h)": self.outage_range_hours,
            "Gini Coefficient": round(self.gini_coefficient, 3),
            "Shortfall Covered (%)": round(self.shortfall_fulfillment_pct, 1),
            "Total Shed (MWh)": round(self.total_shed_mwh, 1),
            "Hospital Shed (h)": self.hospital_shed_hours,
            "Iftar/Sehri Violations": self.ramadan_violations_count,
            "Consecutive Violations": self.consecutive_violations_count,
            "Fully Compliant": "Yes" if self.is_fully_compliant else "No",
        }


def compute_jains_fairness_index(values: List[float]) -> float:
    """Calculates Jain's Fairness Index: J(x) = (sum x_i)^2 / (n * sum x_i^2).

    Returns 1.0 if perfectly equal; returns 1/n in extreme unfairness.
    """
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 1.0

    # If all values are 0, distribution is trivially fair
    sum_val = float(np.sum(arr))
    if sum_val == 0.0:
        return 1.0

    sum_sq = float(np.sum(arr ** 2))
    if sum_sq == 0.0:
        return 1.0

    index = (sum_val ** 2) / (n * sum_sq)
    return float(np.clip(index, 1.0 / n, 1.0))


def compute_gini_coefficient(values: List[float]) -> float:
    """Calculates Gini coefficient of inequality: 0 = perfect equality, 1 = complete inequality."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or np.sum(arr) == 0.0:
        return 0.0

    # Sort ascending
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_arr)) / (n * np.sum(sorted_arr)) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


def evaluate_schedule_fairness(
    algorithm_name: str,
    schedule_matrix: np.ndarray,
    feeders: List[FeederArea],
    hourly_shortfall: np.ndarray,
    area_demands: Dict[str, np.ndarray],
    is_ramadan: bool = True,
    max_consecutive_allowed: int = 2,
) -> ScheduleFairnessReport:
    """Computes comprehensive fairness, reliability, and constraint adherence metrics."""
    n_feeders, hours = schedule_matrix.shape

    # 1. Energy Metrics (MWh)
    hourly_shed = np.zeros(hours)
    for h in range(hours):
        hourly_shed[h] = sum(
            schedule_matrix[i, h] * area_demands[feeders[i].area_id][h]
            for i in range(n_feeders)
        )

    total_shortfall = float(np.sum(hourly_shortfall))
    total_shed = float(np.sum(hourly_shed))

    effective_covered = float(np.sum(np.minimum(hourly_shed, hourly_shortfall)))
    coverage_pct = (effective_covered / total_shortfall * 100.0) if total_shortfall > 0 else 100.0
    overshed = max(0.0, total_shed - effective_covered)

    # 2. Outage Distribution among non-critical feeders
    non_critical_outages = []
    hospital_shed_h = 0
    ramadan_violations = 0
    consecutive_violations = 0

    for i, f in enumerate(feeders):
        outage_count = int(np.sum(schedule_matrix[i, :]))

        if f.is_critical_exempt:
            hospital_shed_h += outage_count
        else:
            non_critical_outages.append(outage_count)

        # Check Ramadan violations
        if is_ramadan:
            for h in (4, 18, 19):
                if schedule_matrix[i, h] == 1:
                    ramadan_violations += 1

        # Check consecutive violations
        consec = 0
        for h in range(hours):
            if schedule_matrix[i, h] == 1:
                consec += 1
                if consec > max_consecutive_allowed:
                    consecutive_violations += 1
            else:
                consec = 0

    # Fairness calculations on non-critical pool
    jains = compute_jains_fairness_index([float(x) for x in non_critical_outages])
    std_h = float(np.std(non_critical_outages)) if non_critical_outages else 0.0
    var_h = float(np.var(non_critical_outages)) if non_critical_outages else 0.0
    max_h = max(non_critical_outages) if non_critical_outages else 0
    min_h = min(non_critical_outages) if non_critical_outages else 0
    spread_h = max_h - min_h
    gini = compute_gini_coefficient([float(x) for x in non_critical_outages])

    is_compliant = (
        hospital_shed_h == 0
        and ramadan_violations == 0
        and consecutive_violations == 0
        and coverage_pct >= 90.0
    )

    return ScheduleFairnessReport(
        algorithm_name=algorithm_name,
        total_shortfall_mwh=round(total_shortfall, 1),
        total_shed_mwh=round(total_shed, 1),
        shortfall_fulfillment_pct=round(coverage_pct, 1),
        overshed_mwh=round(overshed, 1),
        jains_fairness_index=jains,
        outage_std_hours=std_h,
        outage_variance_hours=var_h,
        max_outage_hours=max_h,
        min_outage_hours=min_h,
        outage_range_hours=spread_h,
        gini_coefficient=gini,
        hospital_shed_hours=hospital_shed_h,
        ramadan_violations_count=ramadan_violations,
        consecutive_violations_count=consecutive_violations,
        is_fully_compliant=is_compliant,
    )
