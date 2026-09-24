"""Tests for `hems.analysis.elasticity` (BUILD_SPEC.md §10).

Checks the arc-elasticity formula against hand-computed quantity/price pairs,
its NaN edge cases, per-component window quantities on a real simulated
baseline/CPP pair, and that pooling sums Q before taking the ratio.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

from hems.agents.stub_agent import StubAgent
from hems.analysis.elasticity import (
    ElasticityRecord,
    arc_elasticity,
    elasticity_table,
    household_elasticity,
    in_conway_range,
    pooled_elasticity,
)
from hems.environment.appliances import HVACParams, WasherParams
from hems.environment.household import Household, HouseholdParams
from hems.environment.simulator import run_simulation
from hems.environment.tariff import Tariff

DATE = dt.date(2013, 1, 18)


def test_spec_formula_hand_computed():
    # Q 10 -> 8 (-20%), P 0.55 -> 1.10 (+100%)  => -0.2
    assert arc_elasticity(10, 8, 0.55, 1.10) == pytest.approx(-0.2)


def test_midpoint_formula_hand_computed():
    # dQ/Qmid = -2/9, dP/Pmid = 0.55/0.825
    expected = (-2 / 9) / (0.55 / 0.825)
    assert arc_elasticity(10, 8, 0.55, 1.10, "midpoint") == pytest.approx(expected)


def test_full_elimination_hits_floor():
    # Removing all demand: elasticity = -1 / %dP
    assert arc_elasticity(5, 0, 0.55, 1.20) == pytest.approx(-0.55 / 0.65)


def test_undefined_cases_are_nan():
    assert math.isnan(arc_elasticity(0, 0, 0.55, 1.20))  # no baseline demand
    assert math.isnan(arc_elasticity(5, 4, 0.55, 0.55))  # no price change
    with pytest.raises(ValueError):
        arc_elasticity(5, 4, 0.55, 1.2, "bogus")


def test_conway_range():
    assert in_conway_range(-0.2)
    assert not in_conway_range(-0.8)
    assert not in_conway_range(-0.01)
    assert not in_conway_range(float("nan"))


def _rec(hid, qb, qc, archetype="A", agent="llm"):
    comp = lambda q: {"hvac": q, "washer": 0.0, "flexible": q, "total": q + 10.0}  # noqa: E731
    return ElasticityRecord(
        hid, agent, 42, archetype, "17:00", "20:00", 0.55, 1.20, comp(qb), comp(qc)
    )


def test_pooling_sums_quantities_before_ratio():
    recs = [_rec(1, 1.0, 0.0), _rec(2, 9.0, 9.0)]
    # pooled flexible: 10 -> 9 (-10%); mean of ratios would be -50%
    expected = (-0.1) / (0.65 / 0.55)
    assert pooled_elasticity(recs, "flexible") == pytest.approx(expected)


def test_table_groups_by_archetype():
    recs = [_rec(1, 4.0, 2.0, "A"), _rec(2, 4.0, 4.0, "C")]
    table = elasticity_table(recs, "archetype").set_index("archetype")
    assert table.loc["C", "elasticity_flexible"] == pytest.approx(0.0)
    assert table.loc["A", "elasticity_flexible"] < 0


class _HVACOnlyOffDuringCPP:
    """Runs HVAC every step except CPP steps; never runs the washer."""

    def decide(self, ctx):
        from hems.agents.base import Decision

        run = ctx.appliance["type"] == "hvac" and not ctx.is_cpp
        return Decision(run=run)

    def name(self):
        return "hvac_off_in_cpp"


def test_household_elasticity_on_simulated_pair():
    tariff = Tariff.from_yaml("config/tariff.yaml")
    base_t = tariff.with_cpp_events([])
    cpp_t = tariff.with_cpp_events([tariff.make_cpp_event(DATE, "17:00", "20:00", "moderate")])
    hh = Household(HouseholdParams(8596070, "moderate", "moderate", "21:00", "07:00", "22:00"))
    hvac = HVACParams.from_yaml("config/appliances.yaml")
    washer = WasherParams(duration_steps=4, power_kw=0.8, released_step=14, deadline_step=42)
    agent = _HVACOnlyOffDuringCPP()
    base = run_simulation(hh, base_t, hvac, washer, agent, DATE)
    cpp = run_simulation(hh, cpp_t, hvac, washer, agent, DATE)

    rec = household_elasticity(base, cpp, seed=42, archetype="C")
    assert (rec.window_start, rec.window_end) == ("17:00", "20:00")
    assert rec.p_base == pytest.approx(0.55) and rec.p_cpp == pytest.approx(1.20)
    assert rec.q_base["hvac"] == pytest.approx(6 * 1.0)  # 6 steps x 2 kW x 0.5 h
    assert rec.q_cpp["hvac"] == 0.0
    assert rec.q_base["total"] - rec.q_cpp["total"] == pytest.approx(6.0)  # base load cancels
    assert rec.elasticity("hvac") == pytest.approx(-0.55 / 0.65)
    assert math.isnan(rec.elasticity("washer"))
    assert -0.846 < rec.elasticity("total") < 0


def test_household_elasticity_needs_cpp_run():
    tariff = Tariff.from_yaml("config/tariff.yaml").with_cpp_events([])
    hh = Household(HouseholdParams(8596070, "moderate", "moderate", "21:00", "07:00", "22:00"))
    hvac = HVACParams.from_yaml("config/appliances.yaml")
    washer = WasherParams(duration_steps=4, power_kw=0.8, released_step=14, deadline_step=42)
    r = run_simulation(hh, tariff, hvac, washer, StubAgent(), DATE)
    with pytest.raises(ValueError):
        household_elasticity(r, r, seed=42)
