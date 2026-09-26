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


def test_curve_derived_elasticity_matches_household_elasticity():
    """The summaries' records (rebuilt from curves.parquet rows) == the direct computation."""
    from hems.phase3_report import curves_from_runs, records_from_curves

    runner = Phase3Runner(Phase3Config.from_yaml(dry_run=True))
    ids = [8596070, 9911140]
    direct = {r.household_id: r for r in runner.elasticity_records(ids, "rule_based", 42)}
    runs = [r for r in runner.runs if r.agent == "rule_based"]
    curves = curves_from_runs(runs, runner.archetype, lambda h: "x")
    rebuilt = {r.household_id: r for r in records_from_curves(curves)[("rule_based", 42)]}
    for hid in ids:
        a, b = direct[hid], rebuilt[hid]
        assert (a.window_start, a.window_end) == (b.window_start, b.window_end)
        assert abs(a.p_base - b.p_base) < 1e-12 and abs(a.p_cpp - b.p_cpp) < 1e-12
        for c in ("hvac", "washer", "flexible", "total"):
            assert abs(a.q_base[c] - b.q_base[c]) < 1e-9 and abs(a.q_cpp[c] - b.q_cpp[c]) < 1e-9


def test_rule_based_gets_the_same_scale_and_factor_as_llm():
    runner = Phase3Runner(Phase3Config.from_yaml(dry_run=True))
    date = runner.cfg.dates[0]
    for hid, factor in ((8596070, 0.7), (9911140, 1.3)):
        rb = runner.run_day(hid, "rule_based", date, "cpp", seed=42, wtp_scale=1.0)
        llm = runner.run_day(hid, "llm", date, "cpp", seed=42, wtp_scale=1.0)
        assert (rb.rule_log["wtp_factor"] == factor).all()
        assert (rb.rule_log["wtp_scale"] == 1.0).all()
        prompted = llm.llm_log[~llm.llm_log["forced"]]
        assert (prompted["wtp_factor"] == factor).all()


def test_band_mode_applies_to_every_agent_env():
    fixed = Phase3Runner(Phase3Config.from_yaml(dry_run=True, band_mode="fixed"))
    persona = Phase3Runner(Phase3Config.from_yaml(dry_run=True, band_mode="persona"))
    assert fixed.cfg.band_mode == "fixed" and fixed.cfg.fixed_band_c == 2.0
    hid = 8211127  # comfort_prioritising: persona band 1.5
    assert persona.env(hid)[1].tolerance_band_c == 1.5
    assert fixed.env(hid)[1].tolerance_band_c == 2.0
    # the orientation (LLM persona, rule-based k) is the persona's in both modes
    assert fixed.env(hid)[0].params.comfort_orientation == "comfort_prioritising"
    assert fixed.rule_params.k_by_orientation != persona.rule_params.k_by_orientation


def test_seeds_per_agent_follow_config():
    cfg = Phase3Config.from_yaml(dry_run=True)
    assert cfg.seeds_for("llm") == [42]
    assert cfg.seeds_for("noisy_milp") == [42, 43, 44]
    assert cfg.seeds_for("milp") == cfg.seeds_for("rule_based") == [42]
    assert not cfg.calibration_enabled and cfg.global_scale == 1.0
    resolved = cfg.resolved()
    for key in ("thermal_heterogeneity", "rule_based", "band", "seeds", "wtp_scaling", "phase3"):
        assert key in resolved
