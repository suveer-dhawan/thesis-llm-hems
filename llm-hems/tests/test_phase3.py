"""Phase 3 runner pieces that don't need an LLM: split, scenario verdict, dry-run pair."""

from __future__ import annotations

import pandas as pd

from hems.phase3 import Phase3Config, Phase3Runner, stratified_split


def test_stratified_split_keeps_archetypes_on_both_sides():
    assignment = pd.read_csv("config/persona_assignment.csv")
    ids = assignment["customer_ID"].tolist()
    cal, held = stratified_split(assignment, ids, 0.5, seed=42)
    assert set(cal).isdisjoint(held) and len(cal) + len(held) == 24
    arch = assignment.set_index("customer_ID")["archetype"]
    for a in "ABC":
        assert (arch[cal] == a).sum() == 4 and (arch[held] == a).sum() == 4
    assert stratified_split(assignment, ids, 0.5, seed=42) == (cal, held)  # reproducible


def test_dry_run_pair_is_memoised_and_spend_free():
    runner = Phase3Runner(Phase3Config.from_yaml(dry_run=True))
    date = runner.cfg.dates[0]
    base, cpp = runner.run_pair(8596070, "llm", date, seed=42, wtp_scale=1.0)
    assert runner.run_pair(8596070, "llm", date, seed=42, wtp_scale=1.0) == (base, cpp)
    assert not any(s.is_cpp for s in base.result.steps)
    assert sum(s.is_cpp for s in cpp.result.steps) == 6
    assert base.live_calls == cpp.live_calls == 0 and base.prompted > 0
    # identical base load on both days -- only the tariff differs
    assert base.result.base_load_curve() == cpp.result.base_load_curve()


def test_wtp_factor_follows_archetype_config():
    runner = Phase3Runner(Phase3Config.from_yaml(dry_run=True))
    factors = runner.cfg.archetype_factors
    assert factors == {"A": 1.3, "B": 1.0, "C": 0.7}
    assert runner.wtp_factor(8596070) == factors["C"]
    assert runner.wtp_factor(9911140) == factors["A"]
    date = runner.cfg.dates[0]
    run = runner.run_day(8596070, "llm", date, "base", seed=42, wtp_scale=2.0)
    prompted = run.llm_log[~run.llm_log["forced"]]
    assert (prompted["wtp_factor"] == 0.7).all() and (prompted["wtp_scale"] == 2.0).all()
