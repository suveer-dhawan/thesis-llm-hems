"""Comfort-orientation validation sweep (Phase 3 mechanism correction).

The Phase 3 live slice showed price-sensitivity persona wording does not
move the (price-blind) LLM, so the LLM-facing persona is now the household's
comfort orientation and cost sensitivity is a deterministic Python factor:
`effective_WTP = LLM_WTP x global_scale x archetype_factor`. This is the
cheap check that the LLM is still doing real work -- NOT a population run.

Six households, two per comfort orientation, each pair one burden-archetype
B and one C so the comfort groups are compared at the same archetype mix
(the sample has no C in comfort_flexible on the evening profile and no A in
comfort_prioritising; schedule profile is irrelevant here since the washer
is switched off). HVAC only, seed 42, one date, baseline + CPP day.

  premise    the pre-sweep check: 3 personas x 3 fixed states (indoor 24.0 /
             26.5 / 28.0C, band 22-26C for everyone), static queries, no
             simulation. Prints the WTP table and the contested-state reasons.
  grid   (a) WTP by comfort orientation at fixed, matched states (clock
             17:30, indoor 25.0-28.0C). Two band modes: `common` holds the
             band at 22-26C for every persona, so only the persona sentence
             differs; `own` uses each persona's real band (the production
             prompt). Balanced's own band == common, so it is reused from
             cache.
  sweep  (b)/(c) simulate each household on the baseline and CPP day at each
             global_scale; report flexible-layer (= HVAC, washer off)
             elasticity per comfort group and the pooled whole-household
             elasticity against Conway's range.

Every call goes through the shared Phase 3 JSONL cache
(`phase3.llm_cache_path`), so re-running is free; live-call counts are
reported per step. `--max-live` aborts between runs if exceeded.

Usage (from llm-hems/):
  PYTHONPATH=src .venv/bin/python experiments/sweep_comfort_scale.py grid
  PYTHONPATH=src .venv/bin/python experiments/sweep_comfort_scale.py sweep --scales 0.1,0.2,0.4,0.8,1.6
  PYTHONPATH=src .venv/bin/python experiments/sweep_comfort_scale.py sweep --dry-run
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import pandas as pd

from hems.agents.base import Decision, DecisionContext
from hems.agents.llm_agent import LLMAgent
from hems.analysis.elasticity import (
    CONWAY_RANGE,
    household_elasticity,
    in_conway_range,
    pooled_elasticity,
)
from hems.environment.simulator import run_simulation
from hems.phase3 import Phase3Config, Phase3Runner

print = functools.partial(print, flush=True)  # noqa: A001 -- unbuffered for background monitoring

OUT = Path("results/phase3/comfort_sweep")
SEED = 42
ORIENTATIONS = ("comfort_prioritising", "balanced", "comfort_flexible")
HOUSEHOLDS = {
    "comfort_prioritising": [8211127, 8568499],  # B (ducted + children), C (split + children)
    "balanced": [8201999, 8596070],  # B, C (split, no children)
    "comfort_flexible": [10444433, 8208263],  # B, C (no aircon)
}
GRID_CLOCK, GRID_T = "17:30", 35  # inside the CPP window
GRID_TEMPS = [25.0, 26.0, 26.5, 27.0, 27.5, 28.0]
PREMISE_STATES = {"comfortable": 24.0, "mild": 26.5, "contested": 28.0}
COMMON_BAND_C = 2.0


class HVACOnlyAgent:
    """The LLM agent for HVAC; the washer is never started (no energy, no prompt)."""

    def __init__(self, inner: LLMAgent):
        self.inner = inner

    def decide(self, ctx: DecisionContext) -> Decision:
        if ctx.appliance["type"] == "washer":
            return Decision(run=False, wtp=None, reason="washer disabled (HVAC-only sweep)")
        return self.inner.decide(ctx)

    def name(self) -> str:
        return "llm"


def _runner(dry_run: bool) -> Phase3Runner:
    runner = Phase3Runner(Phase3Config.from_yaml(dry_run=dry_run))
    a = runner.assignment.set_index("customer_ID")
    for orientation, ids in HOUSEHOLDS.items():
        for hid in ids:
            assert a.loc[hid, "comfort_orientation"] == orientation, hid
    return runner


def _live(runner: Phase3Runner) -> int:
    return 0 if runner.cfg.dry_run else runner.client(SEED).stats["live_calls"]


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# (a) controlled-state grid
# --------------------------------------------------------------------------


def step_grid(
    runner: Phase3Runner, temps: list[float] = GRID_TEMPS, modes: tuple = ("common", "own")
) -> pd.DataFrame:
    hvac = runner.env(HOUSEHOLDS["balanced"][0])[1]
    agent = (
        LLMAgent(dry_run=True) if runner.cfg.dry_run else LLMAgent(client=runner.client(SEED))
    )
    live0 = _live(runner)
    rows = []
    for mode in modes:
        for orientation in ORIENTATIONS:
            hid = HOUSEHOLDS[orientation][0]
            band = COMMON_BAND_C if mode == "common" else runner.env(hid)[1].tolerance_band_c
            for temp in temps:
                ctx = DecisionContext(
                    t=GRID_T,
                    clock=GRID_CLOCK,
                    day_index=0,
                    appliance={
                        "type": "hvac",
                        "indoor_temp_c": temp,
                        "outdoor_temp_c": 30.0,
                        "setpoint_c": hvac.setpoint_c,
                        "tolerance_band_c": band,
                        "rated_kw": hvac.rated_kw,
                    },
                    cost_now=0.0,
                    rate=0.0,
                    is_cpp=False,
                    household={"comfort_orientation": orientation},
                )
                agent.decide(ctx)
                e = agent.log[-1]
                rows.append(
                    {
                        "band_mode": mode,
                        "comfort_orientation": orientation,
                        "band": f"{hvac.setpoint_c - band:.1f}-{hvac.setpoint_c + band:.1f}",
                        "indoor_c": temp,
                        "over_ceiling_c": round(temp - (hvac.setpoint_c + band), 2),
                        "wtp": e.wtp,
                        "cached": e.cached,
                        "misreads": ",".join(e.misreads),
                        "reason": e.reason,
                    }
                )
                print(f"[grid] {mode:6s} {orientation:21s} {temp:.1f}C wtp={e.wtp:.3f}")
    df = pd.DataFrame(rows)
    df.attrs["live_calls"] = _live(runner) - live0
    return df


def grid_tables(df: pd.DataFrame) -> list[str]:
    out = []
    for mode in ("common", "own"):
        sub = df[df["band_mode"] == mode]
        wide = sub.pivot(index="indoor_c", columns="comfort_orientation", values="wtp")[
            list(ORIENTATIONS)
        ]
        wide["ordered (P>B>F)"] = [
            r["comfort_prioritising"] > r["balanced"] > r["comfort_flexible"]
            for _, r in wide.iterrows()
        ]
        bands = sub.groupby("comfort_orientation")["band"].first()
        out += [
            f"#### band mode `{mode}` -- bands: "
            + ", ".join(f"{o} {bands[o]}" for o in ORIENTATIONS),
            "",
            _md(wide.reset_index()),
            "",
        ]
    return out


# --------------------------------------------------------------------------
# (b)/(c) global_scale sweep
# --------------------------------------------------------------------------


def step_sweep(runner: Phase3Runner, scales: list[float], max_live: int) -> tuple[list, pd.DataFrame]:
    date = runner.cfg.dates[0]
    records, logs = [], []
    live0 = _live(runner)
    for scale in scales:
        for orientation, ids in HOUSEHOLDS.items():
            for hid in ids:
                pair = {}
                for label in ("base", "cpp"):
                    if _live(runner) - live0 > max_live:
                        raise SystemExit(f"[sweep] aborting: > {max_live} live calls")
                    household, hvac, washer = runner.env(hid)
                    inner = (
                        LLMAgent(dry_run=True, wtp_scale=scale, wtp_factor=runner.wtp_factor(hid))
                        if runner.cfg.dry_run
                        else LLMAgent(
                            client=runner.client(SEED),
                            wtp_scale=scale,
                            wtp_factor=runner.wtp_factor(hid),
                        )
                    )
                    t0 = time.time()
                    result = run_simulation(
                        household, runner.tariff_for(date, label), hvac, washer,
                        HVACOnlyAgent(inner), date,
                    )  # fmt: skip
                    pair[label] = result
                    log = inner.log_to_dataframe().assign(
                        household_id=hid,
                        comfort_orientation=orientation,
                        archetype=runner.archetype(hid),
                        band_hi_c=hvac.setpoint_c + hvac.tolerance_band_c,
                        run_label=label,
                        global_scale=scale,
                    )
                    logs.append(log)
                    print(
                        f"[sweep] scale={scale:<5} {orientation:21s} {hid} {label:4s} "
                        f"live={int((~log['cached'] & ~log['dry_run']).sum()):3d} "
                        f"hvac_kwh={sum(s.hvac_energy_kwh for s in result.steps):.1f} "
                        f"total_live={_live(runner) - live0} ({time.time() - t0:.0f}s)"
                    )
                rec = household_elasticity(pair["base"], pair["cpp"], SEED, runner.archetype(hid))
                rec.wtp_scale = scale
                rec.comfort_orientation = orientation  # ad-hoc tag for grouping
                records.append(rec)
    log_df = pd.concat(logs, ignore_index=True)
    log_df.attrs["live_calls"] = _live(runner) - live0
    return records, log_df


def sweep_tables(records: list, method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scales = sorted({r.wtp_scale for r in records})
    b_rows, c_rows, hh_rows = [], [], []
    for s in scales:
        at = [r for r in records if r.wtp_scale == s]
        row = {"global_scale": s}
        for o in ORIENTATIONS:
            grp = [r for r in at if r.comfort_orientation == o]
            row[f"{o} e_flex"] = pooled_elasticity(grp, "flexible", method)
            row[f"{o} kWh base->cpp"] = (
                f"{sum(r.q_base['flexible'] for r in grp):.1f}->"
                f"{sum(r.q_cpp['flexible'] for r in grp):.1f}"
            )
        b_rows.append(row)
        e_total = pooled_elasticity(at, "total", method)
        c_rows.append(
            {
                "global_scale": s,
                "e_total (pooled, 6 hh)": e_total,
                "e_flexible (pooled)": pooled_elasticity(at, "flexible", method),
                "in_conway": in_conway_range(e_total),
                **{f"{o} e_total": pooled_elasticity(
                    [r for r in at if r.comfort_orientation == o], "total", method
                ) for o in ORIENTATIONS},
            }
        )  # fmt: skip
        for r in at:
            hh_rows.append(
                {
                    "global_scale": s,
                    "household_id": r.household_id,
                    "comfort_orientation": r.comfort_orientation,
                    "archetype": r.archetype,
                    "hvac_kwh_base": r.q_base["hvac"],
                    "hvac_kwh_cpp": r.q_cpp["hvac"],
                    "e_flexible": r.elasticity("flexible", method),
                    "e_total": r.elasticity("total", method),
                }
            )
    return pd.DataFrame(b_rows), pd.DataFrame(c_rows), pd.DataFrame(hh_rows)


def trajectory_wtp_table(log: pd.DataFrame) -> pd.DataFrame:
    """Raw LLM WTP on the simulated trajectories, by degrees over the persona's own ceiling."""
    p = log[~log["forced"] & (log["appliance"] == "hvac")].copy()
    p["indoor_c"] = p["prompt"].str.extract(r"indoor temperature \(C\)\s+: ([\d.]+)")[0].astype(float)
    p["over_c"] = p["indoor_c"] - p["band_hi_c"]
    bins = [-99, -1.0, -0.5, 0.0, 0.5, 1.0, 99]
    labels = ["< -1.0", "-1.0..-0.5", "-0.5..0", "0..0.5", "0.5..1.0", "> 1.0"]
    p["over_ceiling_bin"] = pd.cut(p["over_c"], bins=bins, labels=labels, right=False)
    # one row per distinct prompt, so repeated cache hits don't weight the mean
    p = p.drop_duplicates("prompt")
    t = p.pivot_table(
        index="over_ceiling_bin",
        columns="comfort_orientation",
        values="wtp",
        aggfunc=["mean", "count"],
        observed=False,
    )
    out = pd.DataFrame(index=t.index)
    for o in ORIENTATIONS:
        if ("mean", o) in t.columns:
            out[f"{o} mean_wtp"] = t[("mean", o)]
            out[f"{o} n"] = t[("count", o)].fillna(0).astype(int)
    return out.reset_index()


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["premise", "grid", "sweep"])
    parser.add_argument("--scales", default="0.25,0.5,1,2,4", help="comma-separated global_scale values")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-live", type=int, default=800)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runner = _runner(args.dry_run)
    out = Path(args.out or (OUT / ("dryrun" if args.dry_run else "live")))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.step == "premise":
        df = step_grid(runner, list(PREMISE_STATES.values()), ("common",))
        state = {v: k for k, v in PREMISE_STATES.items()}
        df["state"] = df["indoor_c"].map(state)
        df.to_csv(out / "premise_wtp.csv", index=False)
        wide = df.pivot(index="comfort_orientation", columns="state", values="wtp")
        wide = wide.loc[list(ORIENTATIONS), list(PREMISE_STATES)]
        report = [
            f"### Premise check: raw LLM WTP ($), clock {GRID_CLOCK}, setpoint 24, band 22-26, "
            f"T=0 seed {SEED}",
            "",
            _md(wide.reset_index()),
            "",
            "Contested-state (28.0C) reasons:",
            "",
            *[
                f"- **{r['comfort_orientation']}** (wtp {r['wtp']}): {r['reason']}"
                for _, r in df[df["state"] == "contested"].iterrows()
            ],
            "",
            f"misreads: {int((df['misreads'] != '').sum())} of {len(df)}; "
            f"live calls: {df.attrs['live_calls']} of {len(df)} prompts; {time.time() - t0:.0f}s",
        ]
        (out / "premise_report.md").write_text("\n".join(report) + "\n")
        print("\n".join(report))
        return

    if args.step == "grid":
        df = step_grid(runner)
        df.to_csv(out / "grid_wtp.csv", index=False)
        report = [
            "### (a) WTP by comfort orientation at matched states "
            f"(clock {GRID_CLOCK}, seed {SEED}, raw LLM WTP $)",
            "",
            *grid_tables(df),
            f"misreads: {int((df['misreads'] != '').sum())} of {len(df)} calls; "
            f"live calls this step: {df.attrs['live_calls']} "
            f"(of {len(df)} prompts; rest from cache); {time.time() - t0:.0f}s",
        ]
        (out / "grid_report.md").write_text("\n".join(report) + "\n")
        print("\n".join(report))
        return

    scales = [float(s) for s in args.scales.split(",")]
    method = runner.cfg.raw["elasticity_method"]
    records, log = step_sweep(runner, scales, args.max_live)
    log.to_parquet(out / "sweep_llm_log.parquet")
    b, c, hh = sweep_tables(records, method)
    traj = trajectory_wtp_table(log)
    for name, df in (("b_elasticity_by_group", b), ("c_total", c), ("per_household", hh),
                     ("a_trajectory_wtp", traj)):  # fmt: skip
        df.to_csv(out / f"{name}.csv", index=False)
    prompted = log[~log["forced"]]
    report = [
        f"### Sweep: HVAC only, seed {SEED}, {runner.cfg.dates[0]}, CPP "
        f"{runner.cfg.raw['cpp']['start']}-{runner.cfg.raw['cpp']['end']}, method `{method}`, "
        f"archetype factors {runner.cfg.archetype_factors}",
        "",
        "#### (a') raw LLM WTP on the simulated trajectories (distinct prompts), by C over own ceiling",
        "",
        _md(traj),
        "",
        "#### (b) flexible-layer (HVAC) elasticity by comfort group vs global_scale",
        "",
        _md(b),
        "",
        f"#### (c) whole-household elasticity (Conway range {CONWAY_RANGE})",
        "",
        _md(c),
        "",
        f"prompted decisions: {len(prompted)}; distinct prompts: {prompted['prompt'].nunique()}; "
        f"live calls this step: {log.attrs['live_calls']}; {time.time() - t0:.0f}s",
    ]
    (out / "sweep_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
