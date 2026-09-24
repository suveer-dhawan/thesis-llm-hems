"""Phase 3 (elasticity) harness -- BUILD_SPEC.md §10-13.

Steps (select with --steps; all read `phase3:` in config/experiment.yaml):

  calibrate   stratified calibration/held-out split, fit one global wtp_scale
              on the calibration subset (pre-registered rule), freeze it
  elasticity  short-run arc elasticity on the CPP window for LLM, MILP and
              noisy-MILP on the evaluation set (held-out if `calibrate` ran,
              else every --households id): per household, per appliance,
              aggregate, and by archetype (tests C > B > A)
  overlay     four-way load-curve overlay (SGSC base / MILP / noisy-MILP / LLM)
              per household + population sum; PAR, peak kW, time-of-peak,
              rebound peak
  scenario    discriminating scenario: one CPP spike + near washer deadline
  seeds       LLM + noisy-MILP elasticity across phase3.seeds (run-to-run spread)

Modes:

  --dry-run   deterministic stub LLM, zero spend (default: all 24 households,
              all steps)
  (live)      real LLM through the shared disk cache; defaults to the
              phase3.live_slice households and `elasticity,overlay`. A live run
              over more than live_slice.max_live_households households refuses
              to start without --approve-budget.
  --budget    no LLM calls: dry-run the full Phase 3 plan over all households,
              count prompted decisions and distinct prompts, subtract what the
              disk cache already holds, and report projected live calls and
              wall-clock at the configured rate limit.

Examples:
  python experiments/run_phase3.py --dry-run --households 8596070,9911140,8632083 \
      --out results/phase3/dryrun_smoke
  python experiments/run_phase3.py --out results/phase3/live_slice      # live slice
  python experiments/run_phase3.py --budget --out results/phase3/budget
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from hems.analysis.elasticity import (
    CONWAY_RANGE,
    elasticity_table,
    records_to_frame,
)
from hems.analysis.metrics import curve_metrics, rebound_peak_kw
from hems.analysis.plots import plot_four_way
from hems.environment.tariff import step_to_clock
from hems.phase3 import (
    AGENTS,
    Phase3Config,
    Phase3Runner,
    fit_scale,
    run_scenario,
    stratified_split,
)

ALL_STEPS = ["calibrate", "elasticity", "overlay", "scenario", "seeds"]
LIVE_DEFAULT_STEPS = ["elasticity", "overlay"]
SERIES = {"milp": "MILP", "noisy_milp": "noisy-MILP", "llm": "LLM"}


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.values]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


def step_calibrate(runner: Phase3Runner, ids: list[int], out: Path, report: list[str]):
    cal = runner.cfg.calibration
    cal_ids, held_ids = stratified_split(
        runner.assignment, ids, float(cal["fraction"]), int(cal["split_seed"])
    )
    fit = fit_scale(runner, cal_ids)
    fit.grid.to_csv(out / "calibration_grid.csv", index=False)
    summary = {
        "pre_registered": {
            "target_elasticity": cal["target_elasticity"],
            "conway_range": cal["conway_range"],
            "rule": "grid value with pooled total elasticity inside the range and closest "
            "to target on the calibration subset; closest overall if none inside (flagged)",
            "scale_grid": cal["scale_grid"],
            "fraction": cal["fraction"],
            "split_seed": cal["split_seed"],
            "fit_seed": cal["fit_seed"],
        },
        "calibration_ids": cal_ids,
        "held_out_ids": held_ids,
        "calibration_archetypes": {h: runner.archetype(h) for h in cal_ids},
        "held_out_archetypes": {h: runner.archetype(h) for h in held_ids},
        "frozen_wtp_scale": fit.scale,
        "calibration_elasticity_total": fit.elasticity,
        "fit_inside_range": fit.in_range,
    }
    (out / "calibration.json").write_text(json.dumps(summary, indent=2))
    report += [
        "## Calibration (pre-registered)",
        "",
        f"- target: {cal['target_elasticity']} inside Conway {cal['conway_range']}; "
        f"split fraction {cal['fraction']} per archetype, split_seed {cal['split_seed']}, "
        f"fit_seed {cal['fit_seed']}",
        f"- calibration: {cal_ids}",
        f"- held-out: {held_ids}",
        f"- **frozen wtp_scale = {fit.scale}** (calibration total elasticity "
        f"{fit.elasticity:.3f}, inside range: {fit.in_range})",
        "",
        _md_table(fit.grid),
        "",
    ]
    return fit.scale, held_ids


def step_elasticity(runner, ids, seed, scale, out: Path, report: list[str]) -> dict:
    method = runner.cfg.raw["elasticity_method"]
    records = {a: runner.elasticity_records(ids, a, seed, scale) for a in AGENTS}
    per_hh = pd.concat([records_to_frame(r, method) for r in records.values()], ignore_index=True)
    per_hh.to_csv(out / "elasticity_per_household.csv", index=False)

    agg = pd.concat(
        [elasticity_table(r, "agent", method) for r in records.values()], ignore_index=True
    )
    agg_mid = pd.concat(
        [elasticity_table(r, "agent", "midpoint") for r in records.values()], ignore_index=True
    )
    by_arch = pd.concat(
        [elasticity_table(r, "archetype", method).assign(agent=a) for a, r in records.items()],
        ignore_index=True,
    ).sort_values(["agent", "archetype"])
    agg.to_csv(out / "elasticity_aggregate.csv", index=False)
    by_arch.to_csv(out / "elasticity_by_archetype.csv", index=False)

    llm_arch = by_arch[by_arch["agent"] == "llm"].set_index("archetype")["elasticity_total"]
    ordering = {
        "values": llm_arch.to_dict(),
        # "more elastic" = more negative
        "C_more_elastic_than_B": bool(
            llm_arch.get("C", float("nan")) < llm_arch.get("B", float("nan"))
        ),
        "B_more_elastic_than_A": bool(
            llm_arch.get("B", float("nan")) < llm_arch.get("A", float("nan"))
        ),
        "C_more_elastic_than_A": bool(
            llm_arch.get("C", float("nan")) < llm_arch.get("A", float("nan"))
        ),
    }
    summary = {
        "households": ids,
        "seed": seed,
        "wtp_scale": scale,
        "method": method,
        "window": [records["llm"][0].window_start, records["llm"][0].window_end] if ids else None,
        "p_base": records["llm"][0].p_base if ids else None,
        "p_cpp": records["llm"][0].p_cpp if ids else None,
        "conway_range": list(CONWAY_RANGE),
        "aggregate": agg.to_dict(orient="records"),
        "aggregate_midpoint": agg_mid.to_dict(orient="records"),
        "by_archetype": by_arch.to_dict(orient="records"),
        "llm_archetype_ordering": ordering,
    }
    (out / "elasticity_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    show = ["household_id", "agent", "archetype"] + [
        f"elasticity_{c}" for c in ("hvac", "washer", "flexible", "total")
    ]
    report += [
        f"## Elasticity (seed {seed}, wtp_scale {scale}, method `{method}`)",
        "",
        f"Window {summary['window']}; price {summary['p_base']:.3f} -> {summary['p_cpp']:.3f} $/kWh. "
        f"Conway range {CONWAY_RANGE}. Per-appliance columns are pooled over households; "
        "`total` is whole-household (base + flexible) and is what the criterion is judged on.",
        "",
        "### Aggregate by agent (discriminant check)",
        "",
        _md_table(agg),
        "",
        "Midpoint-formula robustness:",
        "",
        _md_table(agg_mid),
        "",
        "### By archetype (prediction: C more elastic than B than A)",
        "",
        _md_table(
            by_arch[
                ["agent", "archetype", "n"]
                + [f"elasticity_{c}" for c in ("hvac", "washer", "flexible", "total")]
            ]
        ),
        "",
        f"LLM ordering: {json.dumps(ordering)}",
        "",
        "### Per household",
        "",
        _md_table(per_hh[show].sort_values(["agent", "household_id"])),
        "",
    ]
    return summary


def step_overlay(runner, ids, seed, scale, out: Path, report: list[str]) -> pd.DataFrame:
    cpp_start, cpp_end = runner.cfg.cpp_steps
    reb = runner.cfg.rebound_steps
    clocks = [step_to_clock(t) for t in range(48)]
    date = runner.cfg.dates[0]
    pop = {"base": {}, "cpp": {}}
    rows = []

    def add_metrics(scope, label, day, curve):
        m = curve_metrics(curve, clocks)
        m.update(
            scope=scope,
            series=label,
            day=day,
            rebound_peak_kw=round(rebound_peak_kw(curve, cpp_end, reb), 3),
        )
        rows.append(m)

    for hid in ids:
        curves = {"base": {}, "cpp": {}}
        for agent in AGENTS:
            base, cpp = runner.run_pair(hid, agent, date, seed, scale)
            curves["base"]["SGSC base"] = base.result.base_load_curve()
            curves["cpp"]["SGSC base"] = cpp.result.base_load_curve()
            curves["base"][SERIES[agent]] = base.result.total_load_curve()
            curves["cpp"][SERIES[agent]] = cpp.result.total_load_curve()
        for day in ("base", "cpp"):
            for label, curve in curves[day].items():
                add_metrics(str(hid), label, day, curve)
                acc = pop[day].setdefault(label, [0.0] * 48)
                pop[day][label] = [a + b for a, b in zip(acc, curve)]
        plot_four_way(
            curves["base"],
            curves["cpp"],
            clocks,
            (cpp_start, cpp_end),
            out / f"overlay_{hid}.png",
            f"household {hid} (archetype {runner.archetype(hid)}) | {date} | seed {seed} | "
            f"wtp_scale {scale}",
        )
    for day in ("base", "cpp"):
        for label, curve in pop[day].items():
            add_metrics("population", label, day, curve)
    plot_four_way(
        pop["base"],
        pop["cpp"],
        clocks,
        (cpp_start, cpp_end),
        out / "overlay_population.png",
        f"population sum ({len(ids)} households) | {date} | seed {seed} | wtp_scale {scale}",
    )
    metrics = pd.DataFrame(rows)
    piv = metrics.pivot_table(index=["scope", "series"], columns="day", values="rebound_peak_kw")
    metrics = metrics.merge(
        (piv["cpp"] - piv["base"]).rename("rebound_delta_kw").reset_index(), on=["scope", "series"]
    )
    metrics = metrics[
        [
            "scope",
            "series",
            "day",
            "par",
            "peak_kw",
            "time_of_peak",
            "rebound_peak_kw",
            "rebound_delta_kw",
        ]
    ]
    metrics.to_csv(out / "load_metrics.csv", index=False)
    report += [
        "## Load-curve metrics (four-way)",
        "",
        f"Rebound = peak kW in the {reb} steps after the CPP window ends; delta = CPP day - baseline day.",
        "",
        _md_table(metrics[metrics["scope"] == "population"]),
        "",
        "Figures: `overlay_population.png`, `overlay_<household>.png`.",
        "",
    ]
    return metrics


def step_scenario(runner, ids, scale, out: Path, report: list[str]) -> pd.DataFrame:
    df = run_scenario(runner, ids, runner.cfg.seeds, scale)
    df.to_csv(out / "scenario.csv", index=False)
    summary = df.groupby(["agent", "variant"])["coherent"].agg(["mean", "count"]).reset_index()
    sc = runner.cfg.raw["scenario"]
    report += [
        "## Discriminating scenario (single CPP spike + near washer deadline)",
        "",
        f"Spike {sc['spike']['start']}-{sc['spike']['end']} ({sc['spike']['severity']}); variants "
        + "; ".join(
            f"{k}: release {v['washer_release']}, deadline {v['washer_deadline']}"
            for k, v in sc["variants"].items()
        )
        + ". Coherence is judged against the scenario's true deadline.",
        "",
        _md_table(summary.rename(columns={"mean": "coherent_share"})),
        "",
        _md_table(df),
        "",
    ]
    return df


def step_seeds(runner, ids, scale, out: Path, report: list[str]) -> pd.DataFrame:
    from hems.analysis.elasticity import pooled_elasticity

    method = runner.cfg.raw["elasticity_method"]
    rows = []
    for agent in ("llm", "noisy_milp"):
        for seed in runner.cfg.seeds:
            recs = runner.elasticity_records(ids, agent, seed, scale)
            rows.append(
                {
                    "agent": agent,
                    "seed": seed,
                    "elasticity_total": pooled_elasticity(recs, "total", method),
                    "elasticity_flexible": pooled_elasticity(recs, "flexible", method),
                    "elasticity_hvac": pooled_elasticity(recs, "hvac", method),
                    "elasticity_washer": pooled_elasticity(recs, "washer", method),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "seed_spread.csv", index=False)
    spread = df.groupby("agent")[["elasticity_total", "elasticity_flexible"]].agg(
        ["min", "max", "std"]
    )
    spread.columns = [f"{a}_{b}" for a, b in spread.columns]
    spread = spread.reset_index()
    spread.to_csv(out / "seed_spread_summary.csv", index=False)
    report += [
        f"## Multi-seed spread (seeds {runner.cfg.seeds}, households {ids})",
        "",
        _md_table(df),
        "",
        _md_table(spread),
        "",
    ]
    return df


# --------------------------------------------------------------------------
# budget gate
# --------------------------------------------------------------------------


def budget(cfg_path: str, out: Path) -> dict:
    """Projected live calls for the full Phase 3 plan -- no LLM calls made."""
    runner = Phase3Runner(Phase3Config.from_yaml(cfg_path, dry_run=True))
    cfg = runner.cfg
    ids = [int(i) for i in runner.assignment["customer_ID"]]
    cal_ids, held_ids = stratified_split(
        runner.assignment,
        ids,
        float(cfg.calibration["fraction"]),
        int(cfg.calibration["split_seed"]),
    )
    t0 = time.time()
    fit = fit_scale(runner, cal_ids)
    for seed in cfg.seeds:
        runner.elasticity_records(ids, "llm", seed, fit.scale)
    run_scenario(runner, ids, [cfg.fit_seed], fit.scale)
    sim_s = time.time() - t0

    llm_runs = [r for r in runner.runs if r.agent == "llm"]
    prompted = sum(r.prompted for r in llm_runs)
    by_appliance = {"hvac": 0, "washer": 0}
    unique: set[tuple[int, str]] = set()
    for r in llm_runs:
        p = r.llm_log[~r.llm_log["forced"]]
        for app, n in p["appliance"].value_counts().items():
            by_appliance[app] += int(n)
        unique.update((r.seed, prompt) for prompt in p["prompt"])

    already_cached = None
    cache_note = ""
    try:
        live_runner = Phase3Runner(Phase3Config.from_yaml(cfg_path, dry_run=False))
        already_cached = sum(
            1 for seed, prompt in unique if live_runner.client(seed).is_cached(prompt)
        )
    except Exception as exc:  # noqa: BLE001 -- no key etc: report without the cache credit
        cache_note = f"cache check skipped: {exc}"

    days = 2 * len(cfg.dates)
    n_grid = len(cfg.calibration["scale_grid"])
    n_var = len(cfg.raw["scenario"]["variants"])
    per_day = 2 * 48
    upper = {
        "calibration": per_day * days * n_grid * len(cal_ids),
        "evaluation_all_seeds": per_day * days * len(cfg.seeds) * len(ids),
        "scenario": per_day * n_var * len(ids),
    }
    upper_total = sum(upper.values())
    est_live = len(unique) - (already_cached or 0)
    rpm = float((cfg.llm_cfg.get("gemini") or {}).get("rate_limit_rpm", 5))
    result = {
        "plan": {
            "households": len(ids),
            "calibration_households": len(cal_ids),
            "held_out_households": len(held_ids),
            "dates": [d.isoformat() for d in cfg.dates],
            "simulated_days_per_household_per_seed": days,
            "seeds": cfg.seeds,
            "scale_grid": cfg.calibration["scale_grid"],
            "scenario_variants": n_var,
            "appliances": 2,
            "steps_per_day": 48,
        },
        "formula_upper_bound": {
            "formula": "2 appliances x 48 steps x (days x grid x n_cal + days x seeds x N + variants x N)",
            "terms": upper,
            "total": upper_total,
        },
        "dry_run_projection": {
            "prompted_decisions": prompted,
            "prompted_by_appliance": by_appliance,
            "distinct_prompts": len(unique),
            "already_in_disk_cache": already_cached,
            "cache_note": cache_note,
            "projected_live_calls": est_live,
            "caveat": "counted on the dry-run stub's trajectories; the real LLM's decisions "
            "change which states (and so which prompts) occur. Treat as an estimate; the "
            "formula bound is the hard ceiling.",
        },
        "wall_clock_at_rate_limit": {
            "rate_limit_rpm": rpm,
            "projected_hours": round(est_live / rpm / 60, 1),
            "upper_bound_hours": round(upper_total / rpm / 60, 1),
        },
        "dry_run_simulation_seconds": round(sim_s, 1),
        "dry_run_fitted_scale_used": fit.scale,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "budget.json").write_text(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--dry-run", action="store_true", help="stub LLM, zero spend")
    parser.add_argument(
        "--budget", action="store_true", help="report projected live calls and exit"
    )
    parser.add_argument("--households", default=None, help="comma-separated customer_IDs")
    parser.add_argument("--steps", default=None, help=f"comma-separated subset of {ALL_STEPS}")
    parser.add_argument("--scale", type=float, default=1.0, help="wtp_scale when not calibrating")
    parser.add_argument("--seed", type=int, default=None, help="seed for elasticity/overlay")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--approve-budget",
        action="store_true",
        help="allow a live run over more than live_slice.max_live_households",
    )
    args = parser.parse_args()

    if args.budget:
        out = Path(args.out or "results/phase3/budget")
        print(json.dumps(budget(args.config, out), indent=2))
        return

    cfg = Phase3Config.from_yaml(args.config, dry_run=args.dry_run)
    runner = Phase3Runner(cfg)
    live = cfg.raw["live_slice"]
    if args.households:
        ids = [int(x) for x in args.households.split(",")]
    elif args.dry_run:
        ids = [int(i) for i in runner.assignment["customer_ID"]]
    else:
        ids = [int(i) for i in live["households"]]
    steps = (
        args.steps.split(",") if args.steps else (ALL_STEPS if args.dry_run else LIVE_DEFAULT_STEPS)
    )
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        parser.error(f"unknown steps {sorted(unknown)}")

    if not args.dry_run and len(ids) > int(live["max_live_households"]) and not args.approve_budget:
        print(
            f"[budget gate] refusing a live run over {len(ids)} households "
            f"(> live_slice.max_live_households={live['max_live_households']}). "
            "Run `--budget` first and re-run with --approve-budget once approved.",
            file=sys.stderr,
        )
        sys.exit(2)

    out = Path(args.out or ("results/phase3/dryrun" if args.dry_run else "results/phase3/live"))
    out.mkdir(parents=True, exist_ok=True)
    seed = (
        args.seed
        if args.seed is not None
        else (cfg.fit_seed if args.dry_run else int(live["seed"]))
    )
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "phase3": cfg.raw,
                "llm": cfg.llm_cfg,
                "dry_run": args.dry_run,
                "households": ids,
                "steps": steps,
                "seed": seed,
                "scale_arg": args.scale,
            },
            sort_keys=False,
        )
    )
    runner.assignment[runner.assignment["customer_ID"].isin(ids)].to_csv(
        out / "personas_used.csv", index=False
    )

    t0 = time.time()
    report = [
        f"# Phase 3 run -- {'DRY RUN (stub LLM, numbers are pipeline checks only)' if args.dry_run else 'LIVE'}",
        "",
        f"- households: {ids}",
        f"- steps: {steps}",
        f"- provider/model: {cfg.llm_cfg.get('provider')} / "
        f"{(cfg.llm_cfg.get(cfg.llm_cfg.get('provider', ''), {}) or {}).get('model')}",
        "",
    ]
    scale, eval_ids = args.scale, ids
    if "calibrate" in steps:
        scale, eval_ids = step_calibrate(runner, ids, out, report)
        if not eval_ids:
            eval_ids = ids
            report.append("_held-out subset empty for this household set -- evaluating on all._\n")
    if "elasticity" in steps:
        step_elasticity(runner, eval_ids, seed, scale, out, report)
    if "overlay" in steps:
        step_overlay(runner, eval_ids, seed, scale, out, report)
    if "scenario" in steps:
        step_scenario(runner, ids, scale, out, report)
    if "seeds" in steps:
        step_seeds(runner, ids, scale, out, report)

    stats = runner.llm_call_stats()
    stats["wall_seconds"] = round(time.time() - t0, 1)
    (out / "call_stats.json").write_text(json.dumps(stats, indent=2))
    logs = []
    for r in runner.runs:
        if r.llm_log is not None:
            logs.append(
                r.llm_log.assign(
                    household_id=r.household_id,
                    seed=r.seed,
                    run_label=r.label,
                    run_date=r.date.isoformat(),
                    run_wtp_scale=r.wtp_scale,
                )
            )
    if logs:
        pd.concat(logs, ignore_index=True).to_parquet(out / "llm_decision_log.parquet")
    report += ["## LLM call accounting", "", "```", json.dumps(stats, indent=2), "```", ""]
    (out / "report.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"[run_phase3] wrote {out}/")


if __name__ == "__main__":
    main()
