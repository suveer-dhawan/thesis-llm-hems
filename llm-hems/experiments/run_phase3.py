"""Phase 3 (elasticity) harness -- BUILD_SPEC.md §10-13.

Four agents on every household: MILP, noisy-MILP, the rule-based satisficer
(the LLM agent's halt with a formula WTP) and the LLM. Baseline day (no CPP)
vs CPP day (`phase3.cpp`), same date, same household.

Everything comes from `config/experiment.yaml` (`phase3:`, `llm:`,
`thermal_heterogeneity:`, `rule_based:`):

  band      `phase3.band_mode` fixed (+/-fixed_band_c, primary) | persona;
            --band-mode overrides
  scaling   global_scale (1.0) x archetype factor (A 1.3 / B 1.0 / C 0.7),
            identical for LLM and rule-based; calibration disabled
  seeds     LLM `llm_seeds` ([42], T=0); noisy-MILP `noisy_milp_seeds`
            ([42, 43, 44]); MILP + rule-based deterministic

Steps (--steps, default all but calibrate):

  elasticity  every agent, primary seed: short-run arc elasticity on the CPP
              window, per household / appliance / aggregate / archetype
  overlay     load-curve overlay (SGSC base + four agents) per household +
              population; PAR, peak kW, time-of-peak, rebound peak
  scenario    discriminating scenario: one CPP spike + near washer deadline
              (each agent at its own seeds); dropped when
              phase3.washer_enabled is false (the final run is HVAC-only)
  seeds       noisy-MILP (and LLM) at every configured seed
  calibrate   only when phase3.calibration.enabled (it is not, for the final run)

Every run writes raw per-household artifacts (curves.parquet, decision logs,
scenario.csv, run_meta.json) and derives every summary from them via
`hems.phase3_report.finalize` -- summary.json/csv (per agent: peak kW + time,
PAR, 20:00-20:30 vs CPP-window load, degree-hours + max exceedance, flexible
elasticity, daily flexible kWh change, whole-household elasticity),
elasticity_*.csv, load_metrics.csv, seed_spread*.csv, report.md -- plus
config_resolved.yaml and provenance.json.

Sharding: run disjoint --households sets into separate --out folders (and,
live, separate --cache-path files), then `experiments/merge_shards.py`
combines them into the same summaries as one unsharded run.

Modes:

  --dry-run   deterministic stub LLM, zero spend (default: all 24 households);
              also writes llm_projection.json -- projected live calls against
              the configured cache
  --budget    = --dry-run over all households into results/phase3/budget
  (live)      real LLM through the disk cache; a run over more than
              live_slice.max_live_households refuses to start without
              --approve-budget

Examples:
  python experiments/run_phase3.py --dry-run --out results/phase3/dryrun
  python experiments/run_phase3.py --dry-run --households 8596070,9911140 --out /tmp/s1
  python experiments/run_phase3.py --households <12 ids> \\
      --cache-path results/phase3/cache_s1.jsonl --out results/phase3/final/shard1 --approve-budget
  python experiments/merge_shards.py results/phase3/final/shard1 results/phase3/final/shard2 \\
      --out results/phase3/final/merged
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from hems.analysis.plots import plot_four_way
from hems.environment.appliances import _clock_to_step
from hems.environment.tariff import step_to_clock
from hems.phase3 import (
    AGENTS,
    Phase3Config,
    Phase3Runner,
    fit_scale,
    run_scenario,
    stratified_split,
)
from hems.phase3_report import (
    SERIES,
    Artifacts,
    curves_from_runs,
    finalize,
    logs_from_runs,
)
from hems.provenance import write_provenance

print = functools.partial(print, flush=True)  # noqa: A001 -- progress for long runs

ALL_STEPS = ["calibrate", "elasticity", "overlay", "scenario", "seeds"]
DEFAULT_STEPS = ["elasticity", "overlay", "scenario", "seeds"]


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.values]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------
# calibration (disabled for the final run; kept for the pre-registered record)
# --------------------------------------------------------------------------


def step_calibrate(runner: Phase3Runner, ids: list[int], out: Path) -> tuple[float, list[int]]:
    cal = runner.cfg.calibration
    cal_ids, held_ids = stratified_split(
        runner.assignment, ids, float(cal["fraction"]), int(cal["split_seed"])
    )
    fit = fit_scale(runner, cal_ids)
    fit.grid.to_csv(out / "calibration_grid.csv", index=False)
    summary = {
        "pre_registered": {
            "target_elasticity": cal["target_elasticity"],
            "conway_range": list(runner.cfg.conway_range),
            "scale_grid": cal["scale_grid"],
            "fraction": cal["fraction"],
            "split_seed": cal["split_seed"],
            "fit_seed": cal["fit_seed"],
        },
        "calibration_ids": cal_ids,
        "held_out_ids": held_ids,
        "frozen_wtp_scale": fit.scale,
        "calibration_elasticity_total": fit.elasticity,
        "fit_inside_range": fit.in_range,
    }
    (out / "calibration.json").write_text(json.dumps(summary, indent=2))
    print(f"[calibrate] frozen wtp_scale {fit.scale} (inside range: {fit.in_range})")
    return fit.scale, held_ids or ids


# --------------------------------------------------------------------------
# simulation -> raw artifacts
# --------------------------------------------------------------------------


def _washer_prompt_bound(runner: Phase3Runner, hid: int, label: str) -> int:
    """Max washer prompts in one day: every step from release to latest start."""
    washer = runner.env(hid)[2]
    if washer.duration_steps == 0:  # phase3.washer_enabled: false -- never prompted
        return 0
    variants = runner.cfg.raw["scenario"]["variants"]
    if label in variants:
        v = variants[label]
        release, deadline = (
            _clock_to_step(v["washer_release"]),
            _clock_to_step(v["washer_deadline"]),
        )
    else:
        release, deadline = washer.released_step, washer.deadline_step
    return max(0, deadline - washer.duration_steps - release + 1)


def household_overlay(runner: Phase3Runner, hid: int, scale: float, out: Path) -> None:
    cfg = runner.cfg
    date, seed = cfg.dates[0], cfg.primary_seed
    curves = {"base": {}, "cpp": {}}
    for agent in AGENTS:
        base, cpp = runner.run_pair(hid, agent, date, seed, scale)
        curves["base"]["SGSC base"] = base.result.base_load_curve()
        curves["cpp"]["SGSC base"] = cpp.result.base_load_curve()
        curves["base"][SERIES[agent]] = base.result.total_load_curve()
        curves["cpp"][SERIES[agent]] = cpp.result.total_load_curve()
    plot_four_way(
        curves["base"],
        curves["cpp"],
        [step_to_clock(t) for t in range(48)],
        cfg.cpp_steps,
        out / f"overlay_{hid}.png",
        f"household {hid} (archetype {runner.archetype(hid)}) | {date} | band {cfg.band_mode} | "
        f"seed {seed} | global_scale {scale}",
    )


def simulate(runner: Phase3Runner, ids: list[int], steps: list[str], scale: float,
             out: Path) -> Artifacts:  # fmt: skip
    cfg = runner.cfg
    primary = cfg.primary_seed
    scenario_rows = []
    t0 = time.time()
    for i, hid in enumerate(ids, 1):
        if "elasticity" in steps or "overlay" in steps:
            for agent in AGENTS:
                for date in cfg.dates:
                    runner.run_pair(hid, agent, date, primary, scale)
        if "seeds" in steps:
            for agent in ("noisy_milp", "llm"):
                for seed in cfg.seeds_for(agent):
                    for date in cfg.dates:
                        runner.run_pair(hid, agent, date, seed, scale)
        if "overlay" in steps:
            household_overlay(runner, hid, scale, out)
        if "scenario" in steps:
            scenario_rows.append(
                run_scenario(runner, [hid], scale, {a: cfg.seeds_for(a) for a in AGENTS})
            )
        stats = runner.llm_call_stats()
        print(
            f"[{i}/{len(ids)}] household {hid} done -- LLM prompted "
            f"{stats['prompted_decisions']}, live {stats['live_calls_this_process']} "
            f"({time.time() - t0:.0f}s)"
        )

    idset = set(ids)
    runs = [
        r
        for r in runner.runs
        if r.household_id in idset
        and (r.agent not in ("llm", "rule_based") or r.wtp_scale == scale)
    ]
    day_runs = [r for r in runs if r.label in ("base", "cpp")]
    orientation = runner.assignment.set_index("customer_ID")["comfort_orientation"].to_dict()
    llm_runs = [r for r in runs if r.agent == "llm"]
    stats = runner.llm_call_stats()
    meta = {
        "households": ids,
        "steps": steps,
        "llm_prompted_decisions": sum(r.prompted for r in llm_runs),
        "llm_live_calls": sum(r.live_calls for r in llm_runs),
        "llm_model_versions": {
            str(k): v for k, v in (stats.get("model_versions_by_seed") or {}).items()
        },
        "llm_prompt_upper_bound": sum(
            48 + _washer_prompt_bound(runner, r.household_id, r.label) for r in llm_runs
        ),
        "llm_client_stats_by_seed": {
            str(k): v for k, v in (stats.get("client_stats_by_seed") or {}).items()
        },
    }
    return Artifacts(
        curves=curves_from_runs(day_runs, runner.archetype, orientation.get),
        personas=runner.assignment[runner.assignment["customer_ID"].isin(ids)],
        meta=meta,
        scenario=pd.concat(scenario_rows, ignore_index=True) if scenario_rows else None,
        llm_log=logs_from_runs(runs, "llm_log"),
        rule_log=logs_from_runs(runs, "rule_log"),
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def resolved_config(cfg: Phase3Config, runner: Phase3Runner, ids, steps, scale, out) -> dict:
    res = cfg.resolved()
    res["wtp_scaling"]["global_scale"] = scale
    res["run"] = {
        "households": ids,
        "steps": steps,
        "out": str(out),
        "cache_paths": [cfg.raw["llm_cache_path"]],
    }
    if not cfg.dry_run:
        # the settings each seed's client actually resolved (retries, backoff, rpm, ...)
        res["llm_clients"] = {s: runner.client(s).settings for s in cfg.llm_seeds}
    return res


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--dry-run", action="store_true", help="stub LLM, zero spend")
    parser.add_argument(
        "--budget", action="store_true", help="dry run over all households -> results/phase3/budget"
    )
    parser.add_argument("--households", default=None, help="comma-separated customer_IDs")
    parser.add_argument(
        "--steps",
        default=None,
        help=f"comma-separated subset of {ALL_STEPS} (default {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--band-mode", choices=["fixed", "persona"], default=None, help="override phase3.band_mode"
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="override phase3.llm_cache_path (one file per live shard)",
    )
    parser.add_argument(
        "--no-thermal-heterogeneity",
        action="store_true",
        help="override thermal_heterogeneity.enabled to false",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--approve-budget",
        action="store_true",
        help="allow a live run over more than live_slice.max_live_households",
    )
    args = parser.parse_args(argv)
    if args.budget:
        args.dry_run = True
        args.households = None
        args.out = args.out or "results/phase3/budget"

    cfg = Phase3Config.from_yaml(
        args.config,
        dry_run=args.dry_run,
        band_mode=args.band_mode,
        cache_path=args.cache_path,
        thermal_enabled=False if args.no_thermal_heterogeneity else None,
    )
    runner = Phase3Runner(cfg)
    live = cfg.raw["live_slice"]
    if args.households:
        ids = sorted(int(x) for x in args.households.split(","))
        unknown = set(ids) - set(int(i) for i in runner.assignment["customer_ID"])
        if unknown:
            parser.error(f"households without a persona assignment: {sorted(unknown)}")
    elif args.dry_run:
        ids = sorted(int(i) for i in runner.assignment["customer_ID"])
    else:
        ids = sorted(int(i) for i in live["households"])
    default_steps = [s for s in DEFAULT_STEPS if cfg.washer_enabled or s != "scenario"]
    steps = args.steps.split(",") if args.steps else default_steps
    if unknown := set(steps) - set(ALL_STEPS):
        parser.error(f"unknown steps {sorted(unknown)}")
    if "scenario" in steps and not cfg.washer_enabled:
        parser.error("scenario (washer) step requested but phase3.washer_enabled is false")
    if "calibrate" in steps and not cfg.calibration_enabled:
        parser.error("calibrate requested but phase3.calibration.enabled is false")
    steps = [s for s in ALL_STEPS if s in steps]

    if not args.dry_run and len(ids) > int(live["max_live_households"]) and not args.approve_budget:
        print(
            f"[budget gate] refusing a live run over {len(ids)} households "
            f"(> live_slice.max_live_households={live['max_live_households']}). "
            "Run --dry-run first (llm_projection.json) and re-run with --approve-budget.",
            file=sys.stderr,
        )
        sys.exit(2)

    out = Path(args.out or ("results/phase3/dryrun" if args.dry_run else "results/phase3/live"))
    out.mkdir(parents=True, exist_ok=True)
    write_provenance(out)
    scale = cfg.global_scale
    t0 = time.time()
    eval_ids = ids
    if "calibrate" in steps:
        scale, eval_ids = step_calibrate(runner, ids, out)

    resolved = resolved_config(cfg, runner, eval_ids, steps, scale, out)
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    print(
        f"[run_phase3] {'DRY RUN' if cfg.dry_run else 'LIVE'} | {len(eval_ids)} households | "
        f"steps {steps} | band {cfg.band_mode} | scale {scale} | cache {cfg.raw['llm_cache_path']}"
    )
    artifacts = simulate(runner, eval_ids, steps, scale, out)
    artifacts.meta["wall_seconds"] = round(time.time() - t0, 1)
    finalize(out, artifacts, resolved)
    print((out / "report.md").read_text())
    print(f"[run_phase3] wrote {out}/ ({artifacts.meta['wall_seconds']}s)")
    return out


if __name__ == "__main__":
    main()
