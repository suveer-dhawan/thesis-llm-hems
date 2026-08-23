"""Tests for `hems.environment.simulator`.

Responsibility (BUILD_SPEC.md §4): verify the 30-min orchestration loop
produces a correct `RunResult` (base/flexible/total load curves, cost,
decision log) for a real SGSC household, using the stub agent (always
defers) and a trivial always-on agent as two contrasting cases.
"""

from __future__ import annotations

import datetime as dt

import pytest

from hems.agents.base import Decision, DecisionContext
from hems.agents.stub_agent import StubAgent
from hems.environment.appliances import HVACParams, WasherParams
from hems.environment.household import Household, HouseholdParams
from hems.environment.simulator import STEPS_PER_DAY, run_simulation
from hems.environment.tariff import Tariff

HOUSEHOLD_ID = 8596070
HOT_DAY = dt.date(2013, 1, 18)  # Friday, sustained ~2kWh/half-hour afternoon AC draw


class AlwaysOnAgent:
    def decide(self, ctx: DecisionContext) -> Decision:
        return Decision(run=True, reason="always on")

    def name(self) -> str:
        return "always_on"


@pytest.fixture
def household() -> Household:
    return Household(
        HouseholdParams(
            household_id=HOUSEHOLD_ID,
            thermal_sensitivity="moderate",
            price_sensitivity="moderate",
            deadline="21:00",
            occupied_hours_start="07:00",
            occupied_hours_end="22:00",
        )
    )


@pytest.fixture
def tariff() -> Tariff:
    return Tariff.from_yaml("config/tariff.yaml")


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


@pytest.fixture
def washer_params() -> WasherParams:
    return WasherParams(duration_steps=4, power_kw=0.8, released_step=14, deadline_step=42)


def test_stub_agent_never_draws_flexible_energy(household, tariff, hvac_params, washer_params):
    result = run_simulation(household, tariff, hvac_params, washer_params, StubAgent(), HOT_DAY)

    assert len(result.steps) == STEPS_PER_DAY
    assert result.flexible_kwh_total == pytest.approx(0.0)
    assert result.total_kwh_total == pytest.approx(result.base_kwh_total)
    assert result.washer_completed is False


def test_stub_agent_base_load_matches_sgsc_curve(household, tariff, hvac_params, washer_params):
    result = run_simulation(household, tariff, hvac_params, washer_params, StubAgent(), HOT_DAY)
    # Known from data_profile.md-style inspection of the hot day for 8596070.
    assert result.base_kwh_total == pytest.approx(57.844, abs=1e-6)


def test_stub_agent_hvac_drifts_out_of_band_on_hot_day(
    household, tariff, hvac_params, washer_params
):
    result = run_simulation(household, tariff, hvac_params, washer_params, StubAgent(), HOT_DAY)
    # With HVAC never on and outdoor temp exceeding the comfort band for hours,
    # the household must drift out of band -- proves the hard band is not a
    # no-op.
    assert result.comfort_violation_steps > 0


def test_always_on_agent_runs_hvac_and_washer(household, tariff, hvac_params, washer_params):
    result = run_simulation(household, tariff, hvac_params, washer_params, AlwaysOnAgent(), HOT_DAY)

    assert result.washer_completed is True
    # Washer released at step 14, so it must complete by step 17 if run continuously.
    assert result.washer_completed_by_step == 17
    # HVAC runs every step -> rated_kw * dt * 48 steps.
    expected_hvac_kwh = hvac_params.rated_kw * hvac_params.dt_hours * STEPS_PER_DAY
    expected_washer_kwh = washer_params.energy_kwh
    assert result.flexible_kwh_total == pytest.approx(expected_hvac_kwh + expected_washer_kwh)
    # Full-power cooling every step is oversized relative to heat gain, so it
    # overshoots *below* the band rather than respecting it -- confirms the
    # RC model actually responds to `on`, not that always-on is well-behaved.
    assert result.steps[-1].indoor_temp_c < hvac_params.setpoint_c


def test_total_equals_base_plus_flexible_every_step(household, tariff, hvac_params, washer_params):
    result = run_simulation(household, tariff, hvac_params, washer_params, AlwaysOnAgent(), HOT_DAY)
    for s in result.steps:
        assert s.total_kwh == pytest.approx(s.base_kwh + s.flexible_kwh)
        assert s.total_cost == pytest.approx(s.total_kwh * s.rate)


def test_flexible_share_within_target_band(household, tariff, hvac_params, washer_params):
    # Sizing target (BUILD_SPEC.md "Base load vs flexible layer"): flexible
    # energy under normal operation should be a realistic 10-25% of the
    # household's daily total. AlwaysOnAgent over-runs HVAC (never lets the
    # thermostat cycle off) so this is a loose sanity bound, not the tuned
    # figure -- the MILP run (Part 1b) is the realistic operating point.
    result = run_simulation(household, tariff, hvac_params, washer_params, AlwaysOnAgent(), HOT_DAY)
    assert 0.0 < result.flexible_share < 0.6


def test_washer_never_starts_before_release_step(household, tariff, hvac_params, washer_params):
    result = run_simulation(household, tariff, hvac_params, washer_params, AlwaysOnAgent(), HOT_DAY)
    pre_release_steps = [s for s in result.steps if s.t < washer_params.released_step]
    assert all(s.washer_energy_kwh == 0.0 for s in pre_release_steps)
