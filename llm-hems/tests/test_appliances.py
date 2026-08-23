"""Tests for `hems.environment.appliances`.

Responsibility (BUILD_SPEC.md §7): verify the HVAC RC thermal model step
equation, the synthetic outdoor-temperature curve, and the shiftable-load
duration/deadline/non-interruptibility semantics.
"""

from __future__ import annotations

import pytest

from hems.environment.appliances import (
    HVACParams,
    WasherParams,
    WasherState,
    hvac_in_band,
    hvac_step,
    synthetic_outdoor_temp_c,
    washer_done,
    washer_step,
)


@pytest.fixture
def hvac_params() -> HVACParams:
    return HVACParams(
        rated_kw=2.0,
        cop=3.0,
        r=2.0,
        c=2.5,
        setpoint_c=24.0,
        tolerance_band_c=2.0,
        outdoor_min_c=22.0,
        outdoor_max_c=32.0,
        outdoor_peak_hour=15.0,
    )


def test_hvac_step_off_drifts_toward_outdoor(hvac_params: HVACParams):
    new_temp, energy = hvac_step(hvac_params, indoor_temp_c=24.0, outdoor_temp_c=32.0, on=False)
    # delta = (0.5 / (2.0*2.5)) * (32 - 24) = 0.1 * 8 = 0.8
    assert new_temp == pytest.approx(24.8)
    assert energy == 0.0


def test_hvac_step_on_draws_rated_power_and_cools(hvac_params: HVACParams):
    new_temp, energy = hvac_step(hvac_params, indoor_temp_c=26.0, outdoor_temp_c=32.0, on=True)
    # gain = (0.5/(2*2.5)) * (32-26) = 0.1*6 = 0.6
    # cooling = (cop*rated_kw*dt)/c = (3.0*2.0*0.5)/2.5 = 1.2
    assert new_temp == pytest.approx(26.0 + 0.6 - 1.2)
    assert energy == pytest.approx(1.0)  # rated_kw * dt = 2.0 * 0.5, independent of COP


def test_hvac_in_band(hvac_params: HVACParams):
    assert hvac_in_band(hvac_params, 24.0) is True
    assert hvac_in_band(hvac_params, 22.0) is True  # lower bound inclusive
    assert hvac_in_band(hvac_params, 26.0) is True  # upper bound inclusive
    assert hvac_in_band(hvac_params, 21.9) is False
    assert hvac_in_band(hvac_params, 26.1) is False


def test_synthetic_outdoor_temp_extremes():
    # min ~12h before peak_hour, max at peak_hour
    peak = synthetic_outdoor_temp_c(t=30, min_c=22.0, max_c=32.0, peak_hour=15.0)  # t=30 -> 15:00
    trough = synthetic_outdoor_temp_c(t=6, min_c=22.0, max_c=32.0, peak_hour=15.0)  # t=6 -> 03:00
    assert peak == pytest.approx(32.0, abs=1e-9)
    assert trough == pytest.approx(22.0, abs=1e-9)


def test_synthetic_outdoor_temp_is_bounded():
    for t in range(48):
        temp = synthetic_outdoor_temp_c(t, min_c=22.0, max_c=32.0, peak_hour=15.0)
        assert 22.0 <= temp <= 32.0


@pytest.fixture
def washer_params() -> WasherParams:
    return WasherParams(duration_steps=4, power_kw=0.8, released_step=14, deadline_step=42)


def test_washer_runs_for_exactly_duration_steps(washer_params: WasherParams):
    state = WasherState()
    for _ in range(washer_params.duration_steps):
        state, energy, on = washer_step(washer_params, state, requested_on=True)
        assert on is True
        assert energy == pytest.approx(0.4)  # 0.8kW * 0.5h
    assert washer_done(washer_params, state)
    assert state.steps_completed == washer_params.duration_steps


def test_washer_is_non_interruptible_once_started(washer_params: WasherParams):
    state = WasherState()
    state, _, on = washer_step(washer_params, state, requested_on=True)
    assert on is True
    # Agent tries to stop mid-cycle -- environment must ignore it.
    state, energy, on = washer_step(washer_params, state, requested_on=False)
    assert on is True
    assert energy == pytest.approx(0.4)
    assert state.steps_completed == 2


def test_washer_stays_off_until_requested(washer_params: WasherParams):
    state = WasherState()
    state, energy, on = washer_step(washer_params, state, requested_on=False)
    assert on is False
    assert energy == 0.0
    assert state.steps_completed == 0


def test_washer_no_further_energy_after_completion(washer_params: WasherParams):
    state = WasherState()
    for _ in range(washer_params.duration_steps):
        state, _, _ = washer_step(washer_params, state, requested_on=True)
    state, energy, on = washer_step(washer_params, state, requested_on=True)
    assert on is False
    assert energy == 0.0


def test_washer_total_energy_matches_energy_kwh_property(washer_params: WasherParams):
    state = WasherState()
    total = 0.0
    for _ in range(washer_params.duration_steps):
        state, energy, _ = washer_step(washer_params, state, requested_on=True)
        total += energy
    assert total == pytest.approx(washer_params.energy_kwh)


def test_washer_latest_start_step(washer_params: WasherParams):
    assert washer_params.latest_start_step == 38
