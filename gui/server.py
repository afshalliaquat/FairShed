"""FairShed Full-Featured Local Web Server.

Serves the modern FairShed HTML5/Tailwind/Chart.js GUI using Python's built-in
standard library HTTP server. Provides complete REST API endpoints powering:
1. EDA & Time Series Filtering
2. Next-Day Demand Forecasting (Random Forest, Linear Regression)
3. Live Fair Load-Shedding Scheduler (CSP Backtracking, Hill Climbing, Genetic Algorithm, Q-Learning)
4. AI Syllabus Experiments (Search BFS/DFS/A*, K-Means)
5. Interactive What-If Scenario Simulator
"""

import json
import hashlib
import hmac
import os
import secrets
import sys
import webbrowser
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.city_grid import CityGridNetwork, get_default_city_feeders
from src.csp_scheduler import FairLoadSheddingCSP
from src.local_evolutionary_search import ScheduleEvaluator, HillClimbingScheduler, GeneticAlgorithmScheduler
from src.fairness_metrics import evaluate_schedule_fairness
from src.probabilistic_scheduler import (
    ProbabilisticShortfallForecast,
    ProbabilisticShortfallModel,
    SHORTFALL_STATE_NAMES,
)
from src.uninformed_search import breadth_first_search_shedding, depth_first_search_shedding, uniform_cost_search_shedding
from src.informed_search import greedy_best_first_search_shedding, a_star_search_shedding
from src.unsupervised_learning import cluster_city_feeders
from src.maintenance_dispatch import MaintenanceDispatchService

GUI_DIR = BASE_DIR / "gui"
DATA_PATH = BASE_DIR / "data" / "processed" / "energy_features.csv"
MODELS_DIR = BASE_DIR / "models"
PORT = 8000

# Cache loaded data and models
_CACHED_DF = None
_CACHED_MODELS = {}
_CACHED_PROBABILISTIC_MODEL: Optional[ProbabilisticShortfallModel] = None
_ADMIN_SESSIONS = set()
_PUBLIC_STATUS_CACHE = None
_MAINTENANCE_SERVICE = MaintenanceDispatchService()
ADMIN_USERNAME = os.environ.get("FAIRSHED_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("FAIRSHED_ADMIN_PASSWORD", "admin123")


def get_cached_df():
    global _CACHED_DF
    if _CACHED_DF is None:
        if DATA_PATH.exists():
            _CACHED_DF = pd.read_csv(DATA_PATH)
            _CACHED_DF["timestamp"] = pd.to_datetime(_CACHED_DF["timestamp"])
        else:
            from src.data_loader import load_or_create_dataset
            from src.data_cleaner import clean_energy_data
            from src.feature_engineering import create_features
            raw_df, _ = load_or_create_dataset()
            clean_df, _ = clean_energy_data(raw_df)
            _CACHED_DF, _ = create_features(clean_df)
    return _CACHED_DF


def get_cached_models():
    global _CACHED_MODELS
    if not _CACHED_MODELS:
        try:
            from src.models import load_saved_models
            models, cols = load_saved_models()
            _CACHED_MODELS = {"models": models, "cols": cols}
        except Exception as e:
            print("Warning: Could not load saved models:", e)
            _CACHED_MODELS = {"models": {}, "cols": []}
    return _CACHED_MODELS


def get_cached_probabilistic_model():
    global _CACHED_PROBABILISTIC_MODEL
    if _CACHED_PROBABILISTIC_MODEL is None:
        model = ProbabilisticShortfallModel()
        model.fit(get_cached_df(), CityGridNetwork())
        _CACHED_PROBABILISTIC_MODEL = model
    return _CACHED_PROBABILISTIC_MODEL


def forecast_probabilistic_shortfall(
    demands: np.ndarray,
    temperatures: np.ndarray,
    capacities: np.ndarray,
    hours: np.ndarray,
) -> ProbabilisticShortfallForecast:
    return get_cached_probabilistic_model().predict(
        demands,
        temperatures,
        capacities,
        hours,
    )


def build_next_day_forecast():
    df = get_cached_df().sort_values("timestamp").reset_index(drop=True)
    models_meta = get_cached_models()
    models = models_meta.get("models", {})
    feature_cols = models_meta.get("cols", [])
    if not feature_cols or not models:
        raise RuntimeError("Trained forecasting models are unavailable.")

    missing_models = {"random_forest_model", "linear_regression_model"} - set(models)
    if missing_models:
        raise RuntimeError(f"Missing forecasting models: {', '.join(sorted(missing_models))}.")

    target_col = "energy_consumption_mw"
    if len(df) < 168 or target_col not in df or "temperature_c" not in df:
        raise RuntimeError("Insufficient historical data for a 24-hour forecast.")

    history = list(df[target_col].astype(float).tail(168))
    temperatures = list(df["temperature_c"].astype(float).tail(24))
    last_timestamp = pd.Timestamp(df["timestamp"].iloc[-1])
    forecasts = {"random_forest": [], "linear_regression": []}
    timestamps = []

    for offset in range(1, 25):
        timestamp = last_timestamp + pd.Timedelta(hours=offset)
        temperature = temperatures[(offset - 1) % 24]
        row = {
            "temperature_c": temperature,
            "hour": timestamp.hour,
            "dayofweek": timestamp.dayofweek,
            "dayofmonth": timestamp.day,
            "month": timestamp.month,
            "quarter": timestamp.quarter,
            "is_weekend": int(timestamp.dayofweek >= 5),
            "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24.0),
            "month_sin": np.sin(2 * np.pi * timestamp.month / 12.0),
            "month_cos": np.cos(2 * np.pi * timestamp.month / 12.0),
            "lag_1h": history[-1],
            "lag_2h": history[-2],
            "lag_24h": history[-24],
            "lag_168h": history[-168],
            "rolling_mean_24h": np.mean(history[-24:]),
            "rolling_std_24h": np.std(history[-24:], ddof=1),
            "rolling_min_24h": np.min(history[-24:]),
            "rolling_max_24h": np.max(history[-24:]),
            "cooling_degrees": max(0.0, temperature - 22.0),
            "heating_degrees": max(0.0, 14.0 - temperature),
            "temp_lag_1h": temperatures[(offset - 2) % 24],
            "temp_lag_24h": temperatures[(offset - 1) % 24],
        }
        input_row = pd.DataFrame([row])
        for column in feature_cols:
            if column not in input_row:
                input_row[column] = 0.0
        input_row = input_row[feature_cols]

        rf_prediction = float(models["random_forest_model"].predict(input_row)[0])
        linear_prediction = float(models["linear_regression_model"].predict(input_row)[0])
        forecasts["random_forest"].append(round(rf_prediction, 1))
        forecasts["linear_regression"].append(round(linear_prediction, 1))
        history.append(rf_prediction)
        timestamps.append(timestamp.isoformat())

    return {"timestamps": timestamps, **forecasts}


class FairShedHandler(SimpleHTTPRequestHandler):
    """Custom request handler serving static files and complete simulation APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(GUI_DIR), **kwargs)

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _is_authenticated(self):
        cookie = self.headers.get("Cookie", "")
        session_id = next(
            (part.split("=", 1)[1] for part in cookie.split("; ") if part.startswith("fairshed_session=")),
            None,
        )
        return session_id in _ADMIN_SESSIONS

    def _require_authentication(self):
        if self._is_authenticated():
            return True
        self._send_json({"error": "Admin login required."}, status=401)
        return False

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        global _PUBLIC_STATUS_CACHE

        if parsed.path == "/api/public-status":
            grid = CityGridNetwork(is_ramadan=True)
            sample_day = get_cached_df().tail(24).copy().reset_index(drop=True)
            demands = sample_day["energy_consumption_mw"].to_numpy(dtype=float)
            temperatures = sample_day["temperature_c"].to_numpy(dtype=float)
            hours = sample_day["timestamp"].dt.hour.to_numpy(dtype=int)
            capacities = grid.simulate_generation_capacity(
                demands,
                average_shortfall_ratio=0.18,
                seed=42,
            )
            try:
                probabilistic_forecast = forecast_probabilistic_shortfall(
                    demands,
                    temperatures,
                    capacities,
                    hours,
                )
            except (RuntimeError, ValueError) as error:
                self._send_json(
                    {"error": f"Probabilistic shortfall model unavailable: {error}"},
                    status=503,
                )
                return
            shortfalls = probabilistic_forecast.expected_shortfall_mw
            area_demands = grid.calculate_hourly_area_demands(demands)
            result = FairLoadSheddingCSP(
                grid,
                shortfalls,
                area_demands,
            ).solve()
            public_status = {
                "updated_at": pd.Timestamp.now().isoformat(),
                "hours": [f"{hour:02d}:00" for hour in range(24)],
                "feeders": [
                    {
                        "id": feeder.area_id,
                        "name": feeder.name,
                        "category": feeder.category,
                        "exempt": feeder.is_critical_exempt,
                        "outage_hours": int(result.area_outage_hours.get(feeder.area_id, 0)),
                    }
                    for feeder in grid.feeders
                ],
                "schedule_matrix": result.schedule_matrix.tolist(),
                "risk_summary": probabilistic_forecast.summary(),
                "maintenance": _MAINTENANCE_SERVICE.status(),
            }
            _PUBLIC_STATUS_CACHE = public_status
            self._send_json(public_status)
            return

        if parsed.path.startswith("/api/") and parsed.path != "/api/auth-status":
            if not self._require_authentication():
                return

        if parsed.path == "/api/auth-status":
            self._send_json({"authenticated": self._is_authenticated()})
            return

        # 1. API: Initial Application State
        if parsed.path == "/api/initial-data":
            df = get_cached_df()
            feeders = get_default_city_feeders()
            sample_day = df.tail(24).copy().reset_index(drop=True)
            demands = sample_day["energy_consumption_mw"].to_numpy(dtype=float)

            grid = CityGridNetwork(is_ramadan=True)
            capacities = grid.simulate_generation_capacity(
                demands,
                average_shortfall_ratio=0.18,
                seed=42,
            )
            actual_shortfalls = grid.calculate_shortfall(demands, capacities)
            try:
                probabilistic_forecast = forecast_probabilistic_shortfall(
                    demands,
                    sample_day["temperature_c"].to_numpy(dtype=float),
                    capacities,
                    sample_day["timestamp"].dt.hour.to_numpy(dtype=int),
                )
            except (RuntimeError, ValueError) as error:
                self._send_json(
                    {"error": f"Probabilistic shortfall model unavailable: {error}"},
                    status=503,
                )
                return
            shortfalls = probabilistic_forecast.expected_shortfall_mw

            # Pre-compute initial CSP schedule
            area_demands = grid.calculate_hourly_area_demands(demands)
            csp_solver = FairLoadSheddingCSP(grid, shortfalls, area_demands)
            csp_res = csp_solver.solve()
            report = evaluate_schedule_fairness("CSP Backtracking", csp_res.schedule_matrix, grid.feeders, shortfalls, area_demands)

            data = {
                "total_records": len(df),
                "date_min": df["timestamp"].min().strftime("%Y-%m-%d"),
                "date_max": df["timestamp"].max().strftime("%Y-%m-%d"),
                "avg_demand": round(float(df["energy_consumption_mw"].mean()), 1),
                "peak_demand": round(float(df["energy_consumption_mw"].max()), 1),
                "min_demand": round(float(df["energy_consumption_mw"].min()), 1),
                "feeders": [
                    {
                        "id": f.area_id,
                        "name": f.name,
                        "category": f.category,
                        "base_load_mw": f.base_load_mw,
                        "exempt": f.is_critical_exempt,
                        "max_outage": f.max_daily_outage_hours,
                        "disruption_cost": f.disruption_cost_per_hour,
                        "outage_hours": csp_res.area_outage_hours.get(f.area_id, 0),
                    }
                    for f in feeders
                ],
                "hours_24": [f"{h:02d}:00" for h in range(24)],
                "demands_24h": demands.tolist(),
                "capacities_24h": capacities.tolist(),
                "shortfalls_24h": shortfalls.tolist(),
                "actual_shortfalls_24h": actual_shortfalls.tolist(),
                "hourly_shed_24h": csp_res.hourly_shed_mw.tolist(),
                "shortfall_state_names": list(SHORTFALL_STATE_NAMES),
                "shortfall_state_probabilities": probabilistic_forecast.state_probabilities.tolist(),
                "markov_prior_probabilities": probabilistic_forecast.markov_prior_probabilities.tolist(),
                "csp_schedule": csp_res.schedule_matrix.tolist(),
                "fairness_report": report.to_dict(),
                "risk_summary": probabilistic_forecast.summary(),
            }
            self._send_json(data)
            return

        if parsed.path == "/api/forecast":
            try:
                self._send_json(build_next_day_forecast())
            except RuntimeError as error:
                self._send_json({"error": str(error)}, status=503)
            return

        # Serve static assets
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            params = json.loads(body)
        except Exception:
            params = {}

        if parsed.path == "/api/login":
            username = str(params.get("username", "") or params.get("email", "")).strip()
            password = str(params.get("password", ""))
            user_handle = username.split("@")[0].strip() if "@" in username else username
            valid_username = hmac.compare_digest(username.lower(), ADMIN_USERNAME.lower()) or hmac.compare_digest(user_handle.lower(), ADMIN_USERNAME.lower())
            valid_password = hmac.compare_digest(
                hashlib.sha256(password.encode("utf-8")).digest(),
                hashlib.sha256(ADMIN_PASSWORD.encode("utf-8")).digest(),
            )
            if not (valid_username and valid_password):
                self._send_json({"error": "Invalid email or password. Please try again."}, status=401)
                return
            session_id = secrets.token_urlsafe(32)
            _ADMIN_SESSIONS.add(session_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Set-Cookie", f"fairshed_session={session_id}; HttpOnly; SameSite=Strict; Path=/")
            self.end_headers()
            self.wfile.write(json.dumps({"authenticated": True}).encode("utf-8"))
            return

        if parsed.path == "/api/logout":
            cookie = self.headers.get("Cookie", "")
            session_id = next(
                (part.split("=", 1)[1] for part in cookie.split("; ") if part.startswith("fairshed_session=")),
                None,
            )
            if session_id:
                _ADMIN_SESSIONS.discard(session_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Set-Cookie", "fairshed_session=; Max-Age=0; HttpOnly; SameSite=Strict; Path=/")
            self.end_headers()
            self.wfile.write(b'{"authenticated": false}')
            return

        # Public customer complaints: no administrator login is required.
        if parsed.path == "/api/complaints":
            area_id = str(params.get("area_id", "") or "").strip()
            hour = int(params.get("hour", datetime.now().hour))

            global _PUBLIC_STATUS_CACHE
            if _PUBLIC_STATUS_CACHE is None:
                df = get_cached_df()
                sample_day = df.tail(24).copy().reset_index(drop=True)
                demands = sample_day["energy_consumption_mw"].to_numpy(dtype=float)
                temperatures = sample_day["temperature_c"].to_numpy(dtype=float)
                hours = sample_day["timestamp"].dt.hour.to_numpy(dtype=int)
                grid = CityGridNetwork(is_ramadan=True)
                capacities = grid.simulate_generation_capacity(demands, average_shortfall_ratio=0.18, seed=42)
                forecast = forecast_probabilistic_shortfall(demands, temperatures, capacities, hours)
                area_demands = grid.calculate_hourly_area_demands(demands)
                result = FairLoadSheddingCSP(grid, forecast.expected_shortfall_mw, area_demands).solve()
                _PUBLIC_STATUS_CACHE = {
                    "feeders": [{"id": feeder.area_id} for feeder in grid.feeders],
                    "schedule_matrix": result.schedule_matrix.tolist(),
                }

            scheduled_off = False
            for index, feeder in enumerate(_PUBLIC_STATUS_CACHE["feeders"]):
                if feeder["id"] == area_id:
                    scheduled_off = bool(_PUBLIC_STATUS_CACHE["schedule_matrix"][index][hour % 24])
                    break

            result = _MAINTENANCE_SERVICE.submit_complaint(
                area_id=area_id,
                hour=hour % 24,
                scheduled_off=scheduled_off,
            )
            if _PUBLIC_STATUS_CACHE is not None:
                _PUBLIC_STATUS_CACHE["maintenance"] = _MAINTENANCE_SERVICE.status()
            status_code = 200 if result.accepted else 409
            self._send_json(
                {
                    "complaint_id": result.complaint_id,
                    "accepted": result.accepted,
                    "status": result.status,
                    "area_id": result.area_id,
                    "hour": result.hour,
                    "assigned_team_id": result.assigned_team_id,
                    "travel_minutes": result.travel_minutes,
                    "message": result.message,
                    "maintenance": _MAINTENANCE_SERVICE.status(),
                },
                status=status_code,
            )
            return

        if parsed.path == "/api/maintenance/complete":
            if not self._require_authentication():
                return
            team_id = str(params.get("team_id", "") or "").strip()
            completed = _MAINTENANCE_SERVICE.complete_team(team_id)
            if _PUBLIC_STATUS_CACHE is not None:
                _PUBLIC_STATUS_CACHE["maintenance"] = _MAINTENANCE_SERVICE.status()
            self._send_json(
                {
                    "completed": completed,
                    "team_id": team_id,
                    "maintenance": _MAINTENANCE_SERVICE.status(),
                },
                status=200 if completed else 404,
            )
            return

        if parsed.path == "/api/maintenance/assign":
            if not self._require_authentication():
                return
            complaint_id = str(params.get("complaint_id", "") or "").strip()
            team_id = str(params.get("team_id", "") or "").strip()
            algorithm = str(params.get("algorithm", "UCS") or "UCS").strip()
            assigned = _MAINTENANCE_SERVICE.assign_complaint(complaint_id, team_id, algorithm)
            if _PUBLIC_STATUS_CACHE is not None:
                _PUBLIC_STATUS_CACHE["maintenance"] = _MAINTENANCE_SERVICE.status()
            self._send_json(
                {
                    "assigned": assigned is not None,
                    "complaint": assigned,
                    "maintenance": _MAINTENANCE_SERVICE.status(),
                },
                status=200 if assigned is not None else 409,
            )
            return

        if parsed.path == "/api/feeder-clusters":
            if not self._require_authentication():
                return
            k = int(params.get("k", 4))
            k = max(2, min(k, 10))
            clust = cluster_city_feeders(CityGridNetwork(), k_clusters=k, seed=42)
            profiles = []
            for row in clust.feeder_profiles_df.to_dict(orient="records"):
                profiles.append({
                    "area_id": row["area_id"],
                    "name": row["name"],
                    "category": row["category"],
                    "base_load_mw": round(float(row["base_load_mw"]), 1),
                    "disruption_cost_per_hour": round(float(row["disruption_cost_per_hour"]), 1),
                    "max_daily_quota_hours": int(row["max_daily_quota_hours"]),
                    "priority_tier": int(row["priority_tier"]),
                    "is_critical": bool(row["is_critical_flag"]),
                    "cluster_id": int(row["cluster_id"]),
                    "profile_archetype": row["profile_archetype"],
                })
            self._send_json({
                "k": clust.k,
                "inertia": clust.inertia,
                "feature_names": clust.feature_names,
                "profiles": profiles,
            })
            return

        if parsed.path.startswith("/api/") and not self._require_authentication():
            return

        # 2. API: EDA Filtering & Diurnal Analysis
        if parsed.path == "/api/eda":
            df = get_cached_df()
            start_date = params.get("startDate", "2024-01-01")
            end_date = params.get("endDate", "2024-12-31")

            mask = (df["timestamp"].dt.date >= pd.to_datetime(start_date).date()) & (
                df["timestamp"].dt.date <= pd.to_datetime(end_date).date()
            )
            filtered = df.loc[mask].copy()
            if filtered.empty:
                self._send_json({"error": "No demand data found for the selected date range."}, status=404)
                return

            # Downsample for smooth charting if large
            step = max(1, len(filtered) // 400)
            chart_df = filtered.iloc[::step]

            # Diurnal hourly profile
            diurnal = filtered.groupby("hour")["energy_consumption_mw"].agg(["mean", "min", "max", "std"]).reset_index()

            # Monthly summary
            monthly = filtered.groupby("month")["energy_consumption_mw"].agg(["mean", "max"]).reset_index()

            data = {
                "filtered_count": len(filtered),
                "timeseries": {
                    "timestamps": chart_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
                    "demands": chart_df["energy_consumption_mw"].round(1).tolist(),
                    "temperatures": chart_df["temperature_c"].round(1).tolist() if "temperature_c" in chart_df else [],
                },
                "diurnal": {
                    "hours": diurnal["hour"].tolist(),
                    "mean": diurnal["mean"].round(1).tolist(),
                    "min": diurnal["min"].round(1).tolist(),
                    "max": diurnal["max"].round(1).tolist(),
                },
                "monthly": {
                    "months": [
                        pd.Timestamp(year=2000, month=int(month), day=1).strftime("%b")
                        for month in monthly["month"]
                    ],
                    "mean": monthly["mean"].round(1).tolist(),
                    "max": monthly["max"].round(1).tolist(),
                },
                "stats": {
                    "mean": round(float(filtered["energy_consumption_mw"].mean()), 1),
                    "max": round(float(filtered["energy_consumption_mw"].max()), 1),
                    "min": round(float(filtered["energy_consumption_mw"].min()), 1),
                    "std": round(float(filtered["energy_consumption_mw"].std()), 1),
                }
            }
            self._send_json(data)
            return

        # 3. API: Live Load-Shedding Scheduler
        if parsed.path == "/api/schedule":
            df = get_cached_df()
            sample_day = df.tail(24).copy().reset_index(drop=True)
            demands = sample_day["energy_consumption_mw"].to_numpy(dtype=float)

            is_ramadan = bool(params.get("is_ramadan", True))
            exclude_hospitals = bool(params.get("exclude_hospitals", True))
            max_daily_quota = int(params.get("max_daily_quota", 4))
            max_consecutive = int(params.get("max_consecutive", 2))
            capacity_ratio = float(params.get("gen_capacity_ratio", 0.82))
            algo = params.get("algorithm", "csp")

            grid = CityGridNetwork(is_ramadan=is_ramadan)
            if not exclude_hospitals:
                for f in grid.feeders:
                    if f.is_critical_exempt:
                        f.is_critical_exempt = False
                        f.max_daily_outage_hours = max_daily_quota

            capacities = grid.simulate_generation_capacity(demands, average_shortfall_ratio=(1.0 - capacity_ratio), seed=42)
            actual_shortfalls = grid.calculate_shortfall(demands, capacities)
            try:
                probabilistic_forecast = forecast_probabilistic_shortfall(
                    demands,
                    sample_day["temperature_c"].to_numpy(dtype=float),
                    capacities,
                    sample_day["timestamp"].dt.hour.to_numpy(dtype=int),
                )
            except (RuntimeError, ValueError) as error:
                self._send_json(
                    {"error": f"Probabilistic shortfall model unavailable: {error}"},
                    status=503,
                )
                return
            shortfalls = probabilistic_forecast.expected_shortfall_mw
            area_demands = grid.calculate_hourly_area_demands(demands)

            evaluator = ScheduleEvaluator(grid, shortfalls, area_demands, max_daily_quota, max_consecutive, is_ramadan)

            if algo == "hill_climbing":
                hc = HillClimbingScheduler(evaluator, seed=42)
                res = hc.solve_steepest_ascent(max_iterations=120)
                sched_matrix = res.best_schedule
                algo_title = "Hill Climbing (Steepest Ascent)"
                runtime = res.runtime_seconds
            elif algo == "genetic":
                ga = GeneticAlgorithmScheduler(evaluator, population_size=20, seed=42)
                res = ga.solve(generations=25)
                sched_matrix = res.best_schedule
                algo_title = "Genetic Algorithm"
                runtime = res.runtime_seconds
            else:
                csp = FairLoadSheddingCSP(grid, shortfalls, area_demands, max_daily_quota, max_consecutive, is_ramadan)
                res = csp.solve()
                sched_matrix = res.schedule_matrix
                algo_title = "CSP Backtracking"
                runtime = res.runtime_seconds

            report = evaluate_schedule_fairness(algo_title, sched_matrix, grid.feeders, shortfalls, area_demands, is_ramadan, max_consecutive)

            hourly_shed = [
                float(sum(sched_matrix[i, h] * area_demands[grid.feeders[i].area_id][h] for i in range(len(grid.feeders))))
                for h in range(24)
            ]

            report_data = report.to_dict()
            report_data.update(
                {
                    "jains_fairness_index": report.jains_fairness_index,
                    "outage_std_hours": report.outage_std_hours,
                    "outage_spread_hours": report.outage_range_hours,
                }
            )

            data = {
                "algorithm": algo_title,
                "runtime_seconds": round(runtime, 4),
                "schedule_matrix": sched_matrix.tolist(),
                "hourly_shed": [round(x, 1) for x in hourly_shed],
                "hourly_shortfall": [round(x, 1) for x in shortfalls.tolist()],
                "hourly_actual_shortfall": [round(x, 1) for x in actual_shortfalls.tolist()],
                "hourly_demand": [round(x, 1) for x in demands.tolist()],
                "hourly_capacity": [round(x, 1) for x in capacities.tolist()],
                "shortfall_state_probabilities": probabilistic_forecast.state_probabilities.tolist(),
                "markov_prior_probabilities": probabilistic_forecast.markov_prior_probabilities.tolist(),
                "shortfall_state_names": list(SHORTFALL_STATE_NAMES),
                "risk_summary": probabilistic_forecast.summary(),
                "outages_per_area": {f.area_id: int(np.sum(sched_matrix[i, :])) for i, f in enumerate(grid.feeders)},
                "report": report_data,
            }
            self._send_json(data)
            return

        # 4. API: Interactive Scenario Playground Predictions
        if parsed.path == "/api/predict-scenario":
            models_meta = get_cached_models()
            models_dict = models_meta.get("models", {})
            feature_cols = models_meta.get("cols", [])

            sim_hour = int(params.get("hour", 14))
            sim_temp = float(params.get("temp", 32.0))
            sim_dayofweek = int(params.get("dayofweek", 1))
            sim_month = int(params.get("month", 7))
            sim_lag_1h = float(params.get("lag_1h", 15200.0))
            sim_lag_2h = float(params.get("lag_2h", 14800.0))
            sim_lag_24h = float(params.get("lag_24h", 15500.0))
            sim_lag_168h = float(params.get("lag_168h", 15400.0))
            sim_rolling_mean = float(params.get("rolling_mean", 14900.0))

            sample_dict = {
                "temperature_c": sim_temp,
                "hour": sim_hour,
                "dayofweek": sim_dayofweek,
                "dayofmonth": 15,
                "month": sim_month,
                "quarter": (sim_month - 1) // 3 + 1,
                "is_weekend": 1 if sim_dayofweek >= 5 else 0,
                "hour_sin": np.sin(2 * np.pi * sim_hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * sim_hour / 24.0),
                "month_sin": np.sin(2 * np.pi * sim_month / 12.0),
                "month_cos": np.cos(2 * np.pi * sim_month / 12.0),
                "lag_1h": sim_lag_1h,
                "lag_2h": sim_lag_2h,
                "lag_24h": sim_lag_24h,
                "lag_168h": sim_lag_168h,
                "rolling_mean_24h": sim_rolling_mean,
                "rolling_std_24h": 650.0,
                "rolling_min_24h": sim_rolling_mean - 1200.0,
                "rolling_max_24h": sim_rolling_mean + 1200.0,
                "cooling_degrees": max(0.0, sim_temp - 22.0),
                "heating_degrees": max(0.0, 14.0 - sim_temp),
                "temp_lag_1h": sim_temp - 0.5,
                "temp_lag_24h": sim_temp + 1.0,
            }

            input_row = pd.DataFrame([sample_dict])
            if feature_cols:
                for c in feature_cols:
                    if c not in input_row.columns:
                        input_row[c] = 0.0
                input_row = input_row[feature_cols]

            rf_pred = float(models_dict["random_forest_model"].predict(input_row)[0]) if "random_forest_model" in models_dict else sim_lag_24h + 250.0
            linear_pred = float(models_dict["linear_regression_model"].predict(input_row)[0]) if "linear_regression_model" in models_dict else sim_lag_24h + 180.0
            data = {
                "random_forest": round(rf_pred, 1),
                "linear_regression": round(linear_pred, 1),

                "cooling_degrees": round(max(0.0, sim_temp - 22.0), 1),
                "heating_degrees": round(max(0.0, 14.0 - sim_temp), 1),
            }
            self._send_json(data)
            return

        # 5. API: AI Lab Experiments Runner
        if parsed.path == "/api/run-experiment":
            exp_type = params.get("type", "search")
            grid = CityGridNetwork()

            if exp_type == "search":
                hour = 20
                deficit = 3650.0
                hour_demands = {f.area_id: f.base_load_mw for f in grid.feeders}
                bfs = breadth_first_search_shedding(grid, hour, deficit, hour_demands)
                dfs = depth_first_search_shedding(grid, hour, deficit, hour_demands)
                ucs = uniform_cost_search_shedding(grid, hour, deficit, hour_demands)
                gbfs = greedy_best_first_search_shedding(grid, hour, deficit, hour_demands)
                astar = a_star_search_shedding(grid, hour, deficit, hour_demands)

                data = {
                    "results": [
                        {"algo": "Breadth-First Search (BFS)", "nodes": bfs.nodes_expanded, "shed_mw": round(bfs.total_shed_mw, 1), "cost": bfs.total_cost, "feeders": bfs.selected_feeder_ids},
                        {"algo": "Depth-First Search (DFS)", "nodes": dfs.nodes_expanded, "shed_mw": round(dfs.total_shed_mw, 1), "cost": dfs.total_cost, "feeders": dfs.selected_feeder_ids},
                        {"algo": "Uniform Cost Search (UCS)", "nodes": ucs.nodes_expanded, "shed_mw": round(ucs.total_shed_mw, 1), "cost": ucs.total_cost, "feeders": ucs.selected_feeder_ids},
                        {"algo": "Greedy Best-First Search", "nodes": gbfs.nodes_expanded, "shed_mw": round(gbfs.total_shed_mw, 1), "cost": gbfs.total_cost, "feeders": gbfs.selected_feeder_ids},
                        {"algo": "A* Search (Provably Optimal)", "nodes": astar.nodes_expanded, "shed_mw": round(astar.total_shed_mw, 1), "cost": astar.total_cost, "feeders": astar.selected_feeder_ids},
                    ]
                }
                self._send_json(data)
                return

            elif exp_type == "kmeans":
                k = int(params.get("k", 4))
                clust = cluster_city_feeders(grid, k_clusters=k)
                data = {
                    "inertia": clust.inertia,
                    "profiles": clust.feeder_profiles_df[["area_id", "name", "base_load_mw", "profile_archetype"]].to_dict(orient="records"),
                }
                self._send_json(data)
                return

        self._send_json({"error": "Unknown endpoint"}, status=404)


def run_server(port: int = PORT, open_browser: bool = True):
    server_address = ("", port)
    httpd = HTTPServer(server_address, FairShedHandler)
    url = f"http://localhost:{port}"

    print("=" * 70)
    print("FairShed Full-Stack AI Platform running!")
    print(f"Local URL : {url}")
    print(f"Root Dir  : {GUI_DIR}")
    print("=" * 70)

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping FairShed server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port=port, open_browser=True)
