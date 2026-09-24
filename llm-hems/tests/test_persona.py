"""Tests for `hems.environment.persona` -- burden bucketing and persona rules."""

from __future__ import annotations

import pandas as pd
import pytest

from hems.environment.appliances import HVACParams, WasherParams
from hems.environment.persona import (
    apply_persona,
    bucket_burden,
    build_persona_table,
    load_assignment,
    load_persona_config,
    persona_household_params,
    time_weighted_average_rate,
)
from hems.environment.tariff import Tariff


def test_terciles_are_balanced_and_ordered():
    burden = pd.Series([0.09, 0.01, 0.05, 0.02, 0.07, 0.03])
    out = bucket_burden(burden, "terciles", ["A", "B", "C"])
    assert out.value_counts().to_dict() == {"A": 2, "B": 2, "C": 2}
    assert list(out) == ["C", "A", "B", "A", "C", "B"]


def test_terciles_break_ties_without_unbalancing():
    out = bucket_burden(pd.Series([0.05] * 6), "terciles", ["A", "B", "C"])
    assert out.value_counts().to_dict() == {"A": 2, "B": 2, "C": 2}


def test_fixed_thresholds_are_left_closed():
    burden = pd.Series([0.029, 0.03, 0.059, 0.06, 0.2])
    out = bucket_burden(burden, "fixed", ["A", "B", "C"], [0.03, 0.06])
    assert list(out) == ["A", "B", "B", "C", "C"]


def test_bucketing_rejects_bad_config():
    with pytest.raises(ValueError):
        bucket_burden(pd.Series([0.1]), "fixed", ["A", "B", "C"], [0.03])
    with pytest.raises(ValueError):
        bucket_burden(pd.Series([0.1]), "nope", ["A", "B", "C"])


def test_time_weighted_rate_is_between_offpeak_and_peak():
    rate = time_weighted_average_rate(Tariff.from_yaml("config/tariff.yaml"), 2013)
    assert 0.22 < rate < 0.55


def _manifest():
    return pd.DataFrame(
        {
            "customer_ID": [1, 2, 3],
            "cluster": [2, 9, 5],
            "income_tier": ["LOW", "MED", "HI"],
            "aircon": ["N", "SplitSystem", "Ducted"],
            "has_children": ["Y", "N", "N"],
        }
    )


def test_build_persona_table_burden_and_rules():
    cfg = load_persona_config()
    kwh = pd.Series({1: 6240.0, 2: 1040.0, 3: 7280.0})  # burdens 10%, 1%, 5%
    t = build_persona_table(_manifest(), kwh, 0.5, cfg).set_index("customer_ID")
    # burden = kWh x rate / (weekly midpoint x 52)
    assert t.loc[1, "burden"] == pytest.approx(6240 * 0.5 / (600 * 52))
    assert t.loc[3, "burden"] == pytest.approx(7280 * 0.5 / (1400 * 52))
    assert list(t["archetype"]) == ["C", "A", "B"]  # highest, lowest, middle burden
    assert t.loc[1, "tolerance_band_c"] == 1.5  # children beats no-aircon
    assert t.loc[2, "tolerance_band_c"] == 2.0  # split
    assert t.loc[3, "tolerance_band_c"] == 1.5  # ducted
    assert t.loc[1, "schedule_profile"] == "morning"
    assert t.loc[2, "schedule_profile"] == "night"
    assert t.loc[3, "schedule_profile"] == "evening"
    assert t.loc[1, "price_sensitivity"] == "high"
    # comfort orientation comes from the same rule as the band
    assert t.loc[1, "comfort_orientation"] == "comfort_prioritising"  # children
    assert t.loc[2, "comfort_orientation"] == "balanced"  # split
    assert t.loc[3, "comfort_orientation"] == "comfort_prioritising"  # ducted
    assert t.loc[1, "archetype_income_tier"] == "C"


def test_no_aircon_no_children_is_comfort_flexible():
    m = _manifest()
    m.loc[0, "has_children"] = "N"
    t = build_persona_table(m, pd.Series({1: 1.0, 2: 1.0, 3: 1.0}), 0.3, load_persona_config())
    row = t.set_index("customer_ID").loc[1]
    assert (row["tolerance_band_c"], row["comfort_orientation"]) == (2.5, "comfort_flexible")


def test_unknown_income_tier_raises():
    m = _manifest()
    m.loc[0, "income_tier"] = "DeclinedToAnswer"
    with pytest.raises(ValueError):
        build_persona_table(m, pd.Series({1: 1.0, 2: 1.0, 3: 1.0}), 0.3, load_persona_config())


def test_committed_assignment_wires_into_simulator_params():
    assignment = load_assignment()
    assert len(assignment) == 24
    params = persona_household_params(8596070, assignment)
    hvac, washer = apply_persona(
        params,
        HVACParams.from_yaml("config/appliances.yaml"),
        WasherParams.from_yaml("config/appliances.yaml", "config/household.yaml"),
    )
    assert params.archetype == "C" and params.price_sensitivity == "high"
    assert params.comfort_orientation == "balanced"
    assert hvac.tolerance_band_c == 2.0
    assert (washer.released_step, washer.deadline_step) == (32, 44)  # 16:00 -> 22:00
