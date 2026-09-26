"""Seeded per-household thermal heterogeneity (R, C, initial indoor temp).

Same seed -> identical params; disabled -> byte-identical to the homogeneous
runs (regression against the committed `results/phase3/rule_based/` curves);
the MILP plans with each household's own physics; the comfort metrics.
"""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from hems.agents.milp_agent import MILPAgent
from hems.analysis.metrics import degree_hours_above_ceiling, max_exceedance_c
from hems.environment.persona import (
    ThermalHeterogeneity,
    load_assignment,
    load_persona_environment,
)
from hems.environment.simulator import run_simulation
from hems.environment.tariff import Tariff
from hems.phase3 import Phase3Config, Phase3Runner

HET = ThermalHeterogeneity(
    enabled=True, r_range=(0.8, 1.2), c_range=(0.8, 1.2), t0_range=(-1.0, 1.0), seed=42
)
IDS = [int(i) for i in load_assignment()["customer_ID"]]
DATE = dt.date(2013, 1, 18)


def test_same_seed_gives_identical_params():
    again = ThermalHeterogeneity(**{**HET.__dict__})
    assert [HET.draws(h) for h in IDS] == [again.draws(h) for h in IDS]
    env_a = load_persona_environment(IDS[0], load_assignment(), thermal=HET)[1]
    env_b = load_persona_environment(IDS[0], load_assignment(), thermal=again)[1]
    assert env_a == env_b


def test_draws_depend_on_seed_and_household_not_run_order():
    other_seed = replace(HET, seed=43)
    assert HET.draws(IDS[0]) != other_seed.draws(IDS[0])
    assert HET.draws(IDS[0]) != HET.draws(IDS[1])
    forward = {h: HET.draws(h) for h in IDS}
    assert {h: HET.draws(h) for h in reversed(IDS)} == forward


def test_draws_respect_ranges_and_are_not_degenerate():
    d = pd.DataFrame([HET.draws(h) for h in IDS])
    assert d["r_mult"].between(0.8, 1.2).all() and d["c_mult"].between(0.8, 1.2).all()
    assert d["t0_offset_c"].between(-1.0, 1.0).all()
    assert d.nunique().min() == len(IDS)
    assert abs(d["r_mult"].corr(d["c_mult"])) < 0.6  # independent draws, loose check


def test_apply_scales_r_c_and_sets_initial_temp_only():
    base = load_persona_environment(IDS[0], load_assignment())[1]
    het = HET.apply(base, IDS[0])
    d = HET.draws(IDS[0])
    assert het.r == pytest.approx(base.r * d["r_mult"])
    assert het.c == pytest.approx(base.c * d["c_mult"])
    assert het.start_temp_c == pytest.approx(base.setpoint_c + d["t0_offset_c"])
    # global params untouched
    for f in ("rated_kw", "cop", "setpoint_c", "tolerance_band_c", "outdoor_min_c",
              "outdoor_max_c", "outdoor_peak_hour"):  # fmt: skip
        assert getattr(het, f) == getattr(base, f)


def test_disabled_is_identity():
    off = replace(HET, enabled=False)
    base = load_persona_environment(IDS[0], load_assignment())[1]
    assert off.apply(base, IDS[0]) is base
    assert load_persona_environment(IDS[0], load_assignment(), thermal=off)[1] == base
    assert base.initial_indoor_temp_c is None and base.start_temp_c == base.setpoint_c


def test_phase3_runner_applies_config_to_every_agent_env():
    cfg = Phase3Config.from_yaml(dry_run=True)
    assert cfg.thermal.enabled  # config default for this project
    runner = Phase3Runner(cfg)
    assert runner.env(IDS[0])[1] == cfg.thermal.apply(
        load_persona_environment(IDS[0], load_assignment())[1], IDS[0]
    )


def test_disabled_reproduces_committed_rule_based_run_exactly():
    """enabled=false -> the committed homogeneous run's aggregate curves, to the digit."""
    sys.path.insert(0, str(Path("experiments").resolve()))
    import run_rule_based as rrb

    committed = pd.read_csv("results/phase3/rule_based/aggregate_kw.csv")
    cfg = Phase3Config.from_yaml()
    runner = Phase3Runner(replace(cfg, thermal=replace(cfg.thermal, enabled=False)))
    params = rrb.RuleBasedParams.from_yaml()
    hids = sorted(IDS)
    for agent in rrb.AGENTS:
        runs = [rrb.simulate(runner, params, h, agent, 2.0, "cpp", 42) for h in hids]
        kw = [round(v * 2, 4) for v in rrb.aggregate(runs)]
        assert kw == committed[f"fixed|{agent}|cpp"].tolist(), agent


def test_milp_plans_with_the_households_own_physics():
    """Wrong R/C/T0 in the plan would push the simulated indoor temp out of band."""
    tariff = Tariff.from_yaml("config/tariff.yaml").with_cpp_events([])
    for hid in IDS[:6]:
        household, hvac, washer = load_persona_environment(hid, load_assignment(), thermal=HET)
        agent = MILPAgent(household, tariff, hvac, washer, DATE)
        assert agent._initial_temp == hvac.start_temp_c != hvac.setpoint_c
        result = run_simulation(household, tariff, hvac, washer, agent, DATE)
        assert result.comfort_violation_steps == 0, hid
        assert degree_hours_above_ceiling(result) == 0.0


def test_comfort_metrics():
    household, hvac, washer = load_persona_environment(IDS[0], load_assignment())
    tariff = Tariff.from_yaml("config/tariff.yaml").with_cpp_events([])

    class Off:
        def decide(self, ctx):
            from hems.agents.base import Decision

            return Decision(run=False)

        def name(self):
            return "off"

    result = run_simulation(household, tariff, hvac, washer, Off(), DATE)
    over = [max(0.0, s.indoor_temp_c - hvac.band_hi_c) for s in result.steps]
    assert degree_hours_above_ceiling(result) == pytest.approx(sum(over) * 0.5)
    assert max_exceedance_c(result) == pytest.approx(max(over)) and max(over) > 0
