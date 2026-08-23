"""Tests for `hems.agents.milp_agent` and `hems.agents.noisy_milp_agent`.

Responsibility (BUILD_SPEC.md §6.1, §6.2, §16 P1): verify the MILP keeps
HVAC inside the hard comfort band every step, completes the washer within
its window, beats a price-blind naive baseline on cost, and that
noisy-MILP staggers its schedule relative to the clean MILP while staying
feasible.
"""

from __future__ import annotations

import datetime as dt

import pytest

from hems.agents.base import Decision, DecisionContext
from hems.agents.milp_agent import MILPAgent
from hems.agents.noisy_milp_agent import NoisyMILPAgent
from hems.environment.appliances import HVACParams, WasherParams
from hems.environment.household import Household, HouseholdParams
from hems.environment.simulator import run_simulation
from hems.environment.tariff import Tariff

HOUSEHOLD_ID = 8596070
HOT_DAY = dt.date(2013, 1, 18)


class NaiveThermostatAgent:
    """Price-blind reactive control: bang-bang HVAC, washer run ASAP.

    A naive-but-comfort-respecting baseline (not committed to `agents/`,
    since it isn't part of the agent set the thesis compares) to check that
    the MILP actually saves money by shifting load into cheaper periods,
    rather than just trivially never running anything (the stub agent).
    """

    def decide(self, ctx: DecisionContext) -> Decision:
        if ctx.appliance["type"] == "hvac":
            temp = ctx.appliance["indoor_temp_c"]
            hi = ctx.appliance["setpoint_c"] + ctx.appliance["tolerance_band_c"]
            lo = ctx.appliance["setpoint_c"] - ctx.appliance["tolerance_band_c"]
            mid = ctx.appliance["setpoint_c"]
            return Decision(run=temp >= hi or (lo < temp < hi and temp > mid))
        return Decision(run=ctx.t >= ctx.appliance["released_step"])

    def name(self) -> str:
        return "naive"


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


def test_milp_keeps_hvac_in_band(household, tariff, hvac_params, washer_params):
    agent = MILPAgent(household, tariff, hvac_params, washer_params, HOT_DAY)
    result = run_simulation(household, tariff, hvac_params, washer_params, agent, HOT_DAY)
    assert result.comfort_violation_steps == 0


def test_milp_completes_washer_by_deadline(household, tariff, hvac_params, washer_params):
    agent = MILPAgent(household, tariff, hvac_params, washer_params, HOT_DAY)
    result = run_simulation(household, tariff, hvac_params, washer_params, agent, HOT_DAY)
    assert result.washer_completed is True
    assert result.washer_completed_by_step <= washer_params.deadline_step


def test_milp_beats_naive_price_blind_baseline(household, tariff, hvac_params, washer_params):
    milp = MILPAgent(household, tariff, hvac_params, washer_params, HOT_DAY)
    milp_result = run_simulation(household, tariff, hvac_params, washer_params, milp, HOT_DAY)
    naive_result = run_simulation(
        household, tariff, hvac_params, washer_params, NaiveThermostatAgent(), HOT_DAY
    )
    # Both respect the same hard constraints -- MILP must not cost more.
    assert naive_result.comfort_violation_steps == 0
    assert naive_result.washer_completed is True
    assert milp_result.total_cost < naive_result.total_cost


def test_milp_flexible_share_within_target_band(household, tariff, hvac_params, washer_params):
    agent = MILPAgent(household, tariff, hvac_params, washer_params, HOT_DAY)
    result = run_simulation(household, tariff, hvac_params, washer_params, agent, HOT_DAY)
    assert 0.10 <= result.flexible_share <= 0.25


def test_noisy_milp_keeps_hvac_in_its_own_band(household, tariff, hvac_params, washer_params):
    agent = NoisyMILPAgent(household, tariff, hvac_params, washer_params, HOT_DAY, seed=7)
    result = run_simulation(
        household, tariff, agent.jittered_hvac_params, agent.jittered_washer_params, agent, HOT_DAY
    )
    assert result.comfort_violation_steps == 0
    assert result.washer_completed is True


def test_noisy_milp_staggers_relative_to_clean_milp(household, tariff, hvac_params, washer_params):
    clean = MILPAgent(household, tariff, hvac_params, washer_params, HOT_DAY)
    noisy = NoisyMILPAgent(household, tariff, hvac_params, washer_params, HOT_DAY, seed=1)
    hvac_diff_steps = sum(1 for a, b in zip(clean.hvac_on, noisy._milp.hvac_on) if a != b)
    assert hvac_diff_steps > 0


def test_noisy_milp_seed_is_deterministic(household, tariff, hvac_params, washer_params):
    a = NoisyMILPAgent(household, tariff, hvac_params, washer_params, HOT_DAY, seed=99)
    b = NoisyMILPAgent(household, tariff, hvac_params, washer_params, HOT_DAY, seed=99)
    assert a._milp.hvac_on == b._milp.hvac_on
    assert a._milp.washer_on == b._milp.washer_on
