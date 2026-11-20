"""City power grid and feeder definitions for the Pakistani Load-Shedding Scheduler.

Models urban distribution networks (e.g., LESCO/K-Electric) across distinct
functional sectors:
- Critical Infrastructure (Hospitals & Emergency Services: strictly exempt)
- Industrial Estates (High economic impact)
- Commercial Districts (Market & business hours)
- Dense Urban Residential
- Suburban Residential
- Peri-Urban & Agricultural Feeders

Provides shortfall calculations, protected timing windows (Iftar, Sehri),
and hourly area demand decomposition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class FeederArea:
    """Represents an electrical distribution feeder / urban sector."""

    area_id: str
    name: str
    category: str  # 'hospital', 'industrial', 'commercial', 'residential', 'suburban', 'rural'
    base_load_mw: float
    is_critical_exempt: bool = False
    max_daily_outage_hours: int = 4
    max_consecutive_outage_hours: int = 2
    priority_tier: int = 3  # 0=Critical (never cut), 1=Industrial, 2=Commercial, 3=Residential, 4=Flexible/Rural
    disruption_cost_per_hour: float = 100.0  # Socio-economic cost weight for search algorithms

    def is_protected_at_hour(self, hour: int, is_ramadan: bool = True) -> bool:
        """Determines if this feeder is strictly protected at a given hour."""
        # 1. Hospitals are 100% protected 24/7
        if self.is_critical_exempt:
            return True

        # 2. Ramadan Iftar (18:00 - 19:59) and Sehri (04:00 - 04:59) protection
        if is_ramadan:
            if hour in (18, 19):  # Iftar meal & Maghrib prayer
                return True
            if hour == 4:  # Sehri meal before dawn
                return True

        return False


def get_default_city_feeders() -> List[FeederArea]:
    """Returns the standard benchmark set of 10 city feeder zones."""
    return [
        FeederArea(
            area_id="F01_HOSP",
            name="Allama Iqbal Med / Jinnah Hospital Feeder",
            category="hospital",
            base_load_mw=120.0,
            is_critical_exempt=True,
            max_daily_outage_hours=0,
            max_consecutive_outage_hours=0,
            priority_tier=0,
            disruption_cost_per_hour=10000.0,
        ),
        FeederArea(
            area_id="F02_IND",
            name="Kot Lakhpat / SITE Industrial Zone",
            category="industrial",
            base_load_mw=450.0,
            is_critical_exempt=False,
            max_daily_outage_hours=4,
            max_consecutive_outage_hours=2,
            priority_tier=1,
            disruption_cost_per_hour=350.0,
        ),
        FeederArea(
            area_id="F03_COMM_A",
            name="Gulberg / Saddar Commercial District",
            category="commercial",
            base_load_mw=380.0,
            is_critical_exempt=False,
            max_daily_outage_hours=4,
            max_consecutive_outage_hours=2,
            priority_tier=2,
            disruption_cost_per_hour=250.0,
        ),
        FeederArea(
            area_id="F04_RES_A",
            name="DHA / Clifton Urban Residential Feeder",
            category="residential",
            base_load_mw=320.0,
            is_critical_exempt=False,
            max_daily_outage_hours=4,
            max_consecutive_outage_hours=2,
            priority_tier=3,
            disruption_cost_per_hour=150.0,
        ),
        FeederArea(
            area_id="F05_RES_B",
            name="Johar Town / Gulshan Residential Feeder",
            category="residential",
            base_load_mw=290.0,
            is_critical_exempt=False,
            max_daily_outage_hours=4,
            max_consecutive_outage_hours=2,
            priority_tier=3,
            disruption_cost_per_hour=140.0,
        ),
        FeederArea(
            area_id="F06_SUB_A",
            name="Model Town / Nazimabad Suburban Feeder",
            category="suburban",
            base_load_mw=240.0,
            is_critical_exempt=False,
            max_daily_outage_hours=4,
            max_consecutive_outage_hours=2,
            priority_tier=4,
            disruption_cost_per_hour=120.0,
        ),
        FeederArea(
            area_id="F07_SUB_B",
            name="Shahdara / Malir Peri-Urban Feeder",
            category="suburban",
            base_load_mw=210.0,
            is_critical_exempt=False,
            max_daily_outage_hours=4,
            max_consecutive_outage_hours=2,
            priority_tier=4,
            disruption_cost_per_hour=110.0,
        ),
        FeederArea(
            area_id="F08_RUR",
            name="Raiwind / Gadap Rural & Agri Feeder",
            category="rural",
            base_load_mw=180.0,
            is_critical_exempt=False,
            max_daily_outage_hours=5,
            max_consecutive_outage_hours=2,
            priority_tier=5,
            disruption_cost_per_hour=90.0,
        ),
        FeederArea(
            area_id="F09_COMM_B",
            name="Tariq Road / Blue Area Commercial Hub",
            category="commercial",
            base_load_mw=260.0,
            is_critical_exempt=False,
            max_daily_outage_hours=4,
            max_consecutive_outage_hours=2,
            priority_tier=2,
            disruption_cost_per_hour=230.0,
        ),
        FeederArea(
            area_id="F10_RES_C",
            name="Gulshan-e-Iqbal / F-8 Residential Feeder",
            category="residential",
            base_load_mw=220.0,
            is_critical_exempt=False,
            max_daily_outage_hours=4,
            max_consecutive_outage_hours=2,
            priority_tier=3,
            disruption_cost_per_hour=130.0,
        ),
    ]


class CityGridNetwork:
    """Manages the full city grid simulation, area loads, and generation capacity."""

    def __init__(self, feeders: Optional[List[FeederArea]] = None, is_ramadan: bool = True):
        self.feeders = feeders if feeders is not None else get_default_city_feeders()
        self.is_ramadan = is_ramadan
        self.total_base_mw = sum(f.base_load_mw for f in self.feeders)
        self.feeder_weights = {f.area_id: f.base_load_mw / self.total_base_mw for f in self.feeders}

    def get_feeder_by_id(self, area_id: str) -> Optional[FeederArea]:
        for f in self.feeders:
            if f.area_id == area_id:
                return f
        return None

    def calculate_hourly_area_demands(
        self, city_total_demands: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Distributes the city-wide hourly demand vector across all feeders."""
        area_demands = {}
        for feeder in self.feeders:
            # Baseline share with small category-specific diurnal modifiers
            base_share = self.feeder_weights[feeder.area_id]
            demands = np.zeros(len(city_total_demands), dtype=float)

            for h_idx, total_mw in enumerate(city_total_demands):
                hour = h_idx % 24
                modifier = 1.0

                # Commercial peaks 11am - 8pm
                if feeder.category == "commercial":
                    modifier = 1.25 if (11 <= hour <= 20) else 0.70
                # Industrial runs flat with slight night dip
                elif feeder.category == "industrial":
                    modifier = 1.15 if (8 <= hour <= 18) else 0.85
                # Residential peaks early morning (6-8) and evening (17-23)
                elif feeder.category in ("residential", "suburban"):
                    modifier = 1.30 if (17 <= hour <= 23) else (0.80 if 1 <= hour <= 5 else 1.0)
                # Rural / Agri operates night tube-wells
                elif feeder.category == "rural":
                    modifier = 1.20 if (hour <= 6 or hour >= 21) else 0.85

                demands[h_idx] = total_mw * base_share * modifier

            area_demands[feeder.area_id] = demands

        # Normalize across feeders at each hour so sum strictly equals total_mw
        normalized_demands = {f.area_id: np.zeros_like(city_total_demands) for f in self.feeders}
        for h_idx in range(len(city_total_demands)):
            raw_sum = sum(area_demands[f.area_id][h_idx] for f in self.feeders)
            if raw_sum > 0:
                scale = city_total_demands[h_idx] / raw_sum
                for f in self.feeders:
                    normalized_demands[f.area_id][h_idx] = area_demands[f.area_id][h_idx] * scale

        return normalized_demands

    def simulate_generation_capacity(
        self,
        city_total_demands: np.ndarray,
        average_shortfall_ratio: float = 0.18,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulates available generation capacity with realistic thermal & hydro fluctuations.

        In Pakistan, generation capacity falls short during peak summer hours (fuel shortages,
        transmission bottle-necks, and circular debt constraints).
        """
        rng = np.random.default_rng(seed)
        n = len(city_total_demands)
        peak_demand = np.max(city_total_demands)
        mean_demand = np.mean(city_total_demands)

        # Base available capacity set to cover ~82% of peak demand on average
        nominal_capacity = peak_demand * (1.0 - average_shortfall_ratio)
        capacity = np.full(n, nominal_capacity, dtype=float)

        for i in range(n):
            hour = i % 24
            # Solar peak adds some capacity between 10 AM and 4 PM
            solar_boost = 600.0 * np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0
            # Evening thermal de-rating due to ambient heat
            thermal_penalty = -400.0 if (17 <= hour <= 22) else 0.0
            # Small random fluctuation (generator trips / frequency changes)
            random_swing = rng.normal(0, 150.0)

            capacity[i] = nominal_capacity + solar_boost + thermal_penalty + random_swing

        return np.maximum(capacity, 5000.0)

    def calculate_shortfall(
        self, demand: np.ndarray, capacity: np.ndarray
    ) -> np.ndarray:
        """Calculates hourly deficit in MW. Returns 0 if capacity >= demand."""
        return np.maximum(0.0, demand - capacity)
