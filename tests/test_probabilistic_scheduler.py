import numpy as np
import pytest

from src.city_grid import CityGridNetwork
from src.csp_scheduler import FairLoadSheddingCSP
from src.data_loader import generate_synthetic_energy_data
from src.probabilistic_scheduler import (
    MarkovShortfallModel,
    ProbabilisticShortfallModel,
)


@pytest.fixture(scope="module")
def fitted_shortfall_model():
    history = generate_synthetic_energy_data(
        start_date="2023-01-01 00:00:00",
        end_date="2023-04-30 23:00:00",
        seed=7,
    )
    grid = CityGridNetwork()
    model = ProbabilisticShortfallModel().fit(history, grid)
    return history, grid, model


def test_markov_model_learns_normalized_transition_priors():
    shortfall = np.array([0, 0, 100, 100, 900, 900, 2500, 2500, 0], dtype=float)
    model = MarkovShortfallModel().fit(shortfall)

    priors = model.predict_priors(24)

    assert model.transition_matrix.shape == (4, 4)
    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)
    assert priors.shape == (24, 4)
    assert np.allclose(priors.sum(axis=1), 1.0)


def test_bayesian_markov_forecast_is_normalized_and_responds_to_capacity(
    fitted_shortfall_model,
):
    _, _, model = fitted_shortfall_model
    demand = np.full(24, 16000.0)
    temperature = np.full(24, 30.0)
    hours = np.arange(24)

    low_capacity = model.predict(demand, temperature, demand * 0.70, hours)
    high_capacity = model.predict(demand, temperature, demand * 0.95, hours)

    assert low_capacity.expected_shortfall_mw.shape == (24,)
    assert low_capacity.state_probabilities.shape == (24, 4)
    assert low_capacity.markov_prior_probabilities.shape == (24, 4)
    assert np.allclose(low_capacity.state_probabilities.sum(axis=1), 1.0)
    assert np.allclose(low_capacity.markov_prior_probabilities.sum(axis=1), 1.0)
    assert np.all(np.isfinite(low_capacity.expected_shortfall_mw))
    assert (
        low_capacity.summary()["bayesian_shortfall_probability_pct"]
        > high_capacity.summary()["bayesian_shortfall_probability_pct"]
    )


def test_probabilistic_shortfall_can_drive_a_csp_schedule(fitted_shortfall_model):
    history, grid, model = fitted_shortfall_model
    sample = history.tail(24).reset_index(drop=True)
    demand = sample["energy_consumption_mw"].to_numpy(dtype=float)
    temperature = sample["temperature_c"].to_numpy(dtype=float)
    hours = sample["timestamp"].dt.hour.to_numpy(dtype=int)
    capacity = grid.simulate_generation_capacity(demand, seed=42)
    forecast = model.predict(demand, temperature, capacity, hours)
    area_demands = grid.calculate_hourly_area_demands(demand)

    result = FairLoadSheddingCSP(
        grid,
        forecast.expected_shortfall_mw,
        area_demands,
    ).solve()

    assert result.schedule_matrix.shape == (len(grid.feeders), 24)
    assert np.allclose(result.hourly_shortfall, forecast.expected_shortfall_mw)
