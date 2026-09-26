"""Comfort-orientation validation sweep (Phase 3 mechanism correction).

The Phase 3 live slice showed price-sensitivity persona wording does not
move the (price-blind) LLM, so the LLM-facing persona is now the household's
comfort orientation and cost sensitivity is a deterministic Python factor:
`effective_WTP = LLM_WTP x global_scale x archetype_factor`. This is the
cheap check that the LLM is still doing real work -- NOT a population run.

Six real households, two per comfort orientation, with their real bands,
base loads and burden archetypes, chosen so each archetype A/B/C appears
twice (the sample has no A among comfort_prioritising households, so the
pairs are prioritising B+C, balanced A+C, flexible A+B -- the same skew as
the full sample, where comfort_flexible is mostly A). HVAC only (washer never
started), seed 42, one date, baseline + CPP day.

  premise  3 personas x 3 fixed states (indoor 24.0 / 26.5 / 28.0C, band
           22-26C for everyone), static queries, no simulation.
  grid     WTP by comfort orientation at fixed states (clock 17:30, indoor
           25.0-28.0C), band held common and at each persona's own band.
  sweep    simulate each household on the baseline and CPP day at each
           global_scale with the graded archetype factors, pick the scale
           whose pooled whole-household elasticity is in Conway's range and
           closest to the calibration target, then re-run that scale with
           uniform factors (1.0/1.0/1.0) -- the decomposition of how much
           heterogeneity the LLM persona + band produce on their own.

The LLM is only consulted inside `LLM_WINDOW` (14:00-21:00: the 17:00-20:00
CPP window with 3h lead-in, vs the ~1h RC time constant) to keep the sweep
cheap. Outside it the HVAC runs a price-blind thermostat (on iff indoor is
above the band ceiling), identical on both days; the final population run uses all 48 steps.

Every call goes through the shared Phase 3 JSONL cache
(`phase3.llm_cache_path`), so re-running is free. `sweep --project` runs the
whole sweep on the dry-run stub and counts the prompts that are not yet in
the live cache -- a projection of live calls (the stub's trajectories are not
the LLM's, so it is an estimate; the structural upper bound is reported too).
`--max-live` aborts a live sweep between runs once exceeded.

Usage (from llm-hems/):
  PYTHONPATH=src .venv/bin/python experiments/sweep_comfort_scale.py premise
  PYTHONPATH=src .venv/bin/python experiments/sweep_comfort_scale.py sweep --project
  PYTHONPATH=src .venv/bin/python experiments/sweep_comfort_scale.py sweep --max-live 400

`--fixed-band 2.0` gives every household the same +/-2.0C band (the orientation
sentence is unchanged), isolating the orientation wording from the persona-linked
band; outputs go to comfort_sweep/fixed_band_<C>/. `--no-decomp` skips the
uniform-factor decomposition run. `--orientations` restricts the households
simulated (groups left out show NaN).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import time
from pathlib import Path

import pandas as pd

from hems.agents.base import Decision, DecisionContext
from hems.agents.llm_agent import SCHEMA_BY_APPLIANCE, LLMAgent
from hems.analysis.elasticity import (
    ElasticityRecord,
    household_elasticity,
    in_conway_range,
    pooled_elasticity,
)
from hems.environment.appliances import _clock_to_step
from hems.environment.simulator import run_simulation
from hems.phase3 import Phase3Config, Phase3Runner
from hems.provenance import write_provenance

print = functools.partial(print, flush=True)  # noqa: A001 -- unbuffered for background monitoring

OUT = Path("results/phase3/comfort_sweep")
SEED = 42
ORIENTATIONS = ("comfort_prioritising", "balanced", "comfort_flexible")
HOUSEHOLDS = {
    "comfort_prioritising": [8211127, 8568499],  # B (ducted + children), C (split + children)
    "balanced": [9911140, 8596070],  # A, C (split, no children)
    "comfort_flexible": [10278838, 10444433],  # A, B (no aircon)
}
GRID_CLOCK, GRID_T = "17:30", 35  # inside the CPP window
GRID_TEMPS = [25.0, 26.0, 26.5, 27.0, 27.5, 28.0]
PREMISE_STATES = {"comfortable": 24.0, "mild": 26.5, "contested": 28.0}
COMMON_BAND_C = 2.0
LLM_WINDOW = ("14:00", "21:00")
DEFAULT_SCALES = "0.3,0.5,0.7,1.0"


class WindowedHVACAgent:
    """LLM for HVAC inside [start, end); a thermostat outside it; washer never started."""

    def __init__(self, inner: LLMAgent, start_step: int, end_step: int):
        self.inner = inner
        self.start_step, self.end_step = start_step, end_step

    def decide(self, ctx: DecisionContext) -> Decision:
        a = ctx.appliance
        if a["type"] == "washer":
            return Decision(run=False, wtp=None, reason="washer disabled (HVAC-only sweep)")
        if not self.start_step <= ctx.t < self.end_step:
            above = a["indoor_temp_c"] > a["setpoint_c"] + a["tolerance_band_c"]
            return Decision(run=above, wtp=None, reason="outside LLM window: thermostat")
        return self.inner.decide(ctx)

    def name(self) -> str:
        return "llm"


def _runner(dry_run: bool) -> Phase3Runner:
    # persona: --fixed-band is applied by this script on top of runner.env()
    runner = Phase3Runner(Phase3Config.from_yaml(dry_run=dry_run, band_mode="persona"))
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
    agent = LLMAgent(dry_run=True) if runner.cfg.dry_run else LLMAgent(client=runner.client(SEED))
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
# global_scale sweep + uniform/graded decomposition
# --------------------------------------------------------------------------

ORIENTATION_OF = {hid: o for o, ids in HOUSEHOLDS.items() for hid in ids}


class Sweep:
    """Memoised baseline/CPP pairs per (household, global_scale, condition).

    condition `graded` uses `phase3.wtp_scaling.archetype_factors`;
    `uniform` sets every household's factor to 1.0.
    """

    def __init__(
        self,
        runner: Phase3Runner,
        project: bool = False,
        max_live: int | None = None,
        fixed_band: float | None = None,
        orientations: tuple[str, ...] = ORIENTATIONS,
    ):
        self.runner = runner
        self.fixed_band = fixed_band
        self.households = [hid for hid, o in ORIENTATION_OF.items() if o in orientations]
        self.project = project
        self.max_live = max_live
        # The client the live sweep calls; in project mode it is only asked
        # whether a prompt is already cached (no API call).
        self.client = runner.client(SEED) if (project or not runner.cfg.dry_run) else None
        self.window = (_clock_to_step(LLM_WINDOW[0]), _clock_to_step(LLM_WINDOW[1]))
        self.records: dict[tuple, ElasticityRecord] = {}
        self.logs: list[pd.DataFrame] = []
        self.uncached: set[str] = set()
        self.stats0 = dict(self.client.stats) if self.client else {}

    @property
    def live_calls(self) -> int:
        if self.runner.cfg.dry_run:
            return 0
        return self.client.stats["live_calls"] - self.stats0.get("live_calls", 0)

    @property
    def cache_hits(self) -> int:
        if self.runner.cfg.dry_run:
            return 0
        return self.client.stats["cache_hits"] - self.stats0.get("cache_hits", 0)

    def factor(self, hid: int, condition: str) -> float:
        return 1.0 if condition == "uniform" else self.runner.wtp_factor(hid)

    def pair(self, hid: int, scale: float, condition: str) -> ElasticityRecord:
        key = (hid, scale, condition)
        if key in self.records:
            return self.records[key]
        date = self.runner.cfg.dates[0]
        household, hvac, washer = self.runner.env(hid)
        if self.fixed_band is not None:
            hvac = dataclasses.replace(hvac, tolerance_band_c=self.fixed_band)
        factor = self.factor(hid, condition)
        results = {}
        for label in ("base", "cpp"):
            if self.max_live is not None and self.live_calls > self.max_live:
                raise SystemExit(f"[sweep] budget gate: > {self.max_live} live calls -- paused")
            inner = (
                LLMAgent(dry_run=True, wtp_scale=scale, wtp_factor=factor)
                if self.runner.cfg.dry_run
                else LLMAgent(client=self.client, wtp_scale=scale, wtp_factor=factor)
            )
            t0, live0 = time.time(), self.live_calls
            results[label] = run_simulation(
                household, self.runner.tariff_for(date, label), hvac, washer,
                WindowedHVACAgent(inner, *self.window), date,
            )  # fmt: skip
            log = inner.log_to_dataframe().assign(
                household_id=hid,
                comfort_orientation=ORIENTATION_OF[hid],
                archetype=self.runner.archetype(hid),
                band_hi_c=hvac.setpoint_c + hvac.tolerance_band_c,
                run_label=label,
                global_scale=scale,
                condition=condition,
            )
            if self.project:
                self.uncached.update(
                    p
                    for p in log["prompt"]
                    if not self.client.is_cached(p, SCHEMA_BY_APPLIANCE["hvac"])
                )
            self.logs.append(log)
            print(
                f"[sweep] {condition:7s} scale={scale:<4} {ORIENTATION_OF[hid]:21s} {hid:>8} "
                f"{label:4s} live={self.live_calls - live0:3d} total_live={self.live_calls} "
                f"hvac_kwh={sum(s.hvac_energy_kwh for s in results[label].steps):.1f} "
                f"({time.time() - t0:.0f}s)"
            )
        rec = household_elasticity(
            results["base"], results["cpp"], SEED, self.runner.archetype(hid)
        )
        rec.wtp_scale = scale
        self.records[key] = rec
        return rec

    def at(self, scale: float, condition: str) -> list[ElasticityRecord]:
        return [self.pair(hid, scale, condition) for hid in self.households]

    def upper_bound_prompts(self) -> int:
        """Distinct prompts if no two trajectories shared a state after they could diverge.

        A trajectory is fixed by (orientation, global_scale x factor) -- the
        two households in a group see identical HVAC physics and prompts. Per
        trajectory: every LLM-window step on the baseline day, plus the CPP
        day's steps from the CPP start to the window end (before the CPP
        start both days are identical, so those prompts are shared).
        """
        cpp_start = self.runner.cfg.cpp_steps[0]
        per = (self.window[1] - self.window[0]) + (self.window[1] - cpp_start)
        combos = {
            (ORIENTATION_OF[hid], round(scale * self.factor(hid, cond), 9))
            for (hid, scale, cond) in self.records
        }
        return per * len(combos)


def _pooled(recs: list[ElasticityRecord], component: str, method: str) -> float:
    return pooled_elasticity(recs, component, method)


def _ordered(e_most: float, e_mid: float, e_least: float) -> bool:
    """True if elasticities are ordered most-elastic (most negative) first; NaN -> False."""
    vals = [e_most, e_mid, e_least]
    return all(v == v for v in vals) and e_most <= e_mid <= e_least and e_most < e_least


SHORT = {"comfort_prioritising": "prio", "balanced": "bal", "comfort_flexible": "flex"}


def _group_row(recs: list[ElasticityRecord], component: str, method: str) -> dict:
    """Pooled elasticity per comfort orientation and per burden archetype, plus orderings."""
    row: dict = {"pooled": _pooled(recs, component, method)}
    eo = {
        o: _pooled([r for r in recs if ORIENTATION_OF[r.household_id] == o], component, method)
        for o in ORIENTATIONS
    }
    ea = {a: _pooled([r for r in recs if r.archetype == a], component, method) for a in "ABC"}
    row.update({SHORT[o]: v for o, v in eo.items()})
    row["flex>bal>prio"] = _ordered(
        eo["comfort_flexible"], eo["balanced"], eo["comfort_prioritising"]
    )
    row.update({f"arch {a}": v for a, v in ea.items()})
    row["C>B>A"] = _ordered(ea["C"], ea["B"], ea["A"])
    return row


def scale_tables(
    sweep: Sweep, scales: list[float], method: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(flexible-layer table, whole-household table, calibration summary) over the grid."""
    flex_rows, total_rows, c_rows = [], [], []
    conway = sweep.runner.cfg.conway_range
    for s in scales:
        recs = sweep.at(s, "graded")
        flex_rows.append(
            {
                "global_scale": s,
                **_group_row(recs, "flexible", method),
                "HVAC kWh base->cpp": f"{sum(r.q_base['flexible'] for r in recs):.0f}->"
                f"{sum(r.q_cpp['flexible'] for r in recs):.0f}",
            }
        )
        total_rows.append({"global_scale": s, **_group_row(recs, "total", method)})
        e_total = _pooled(recs, "total", method)
        c_rows.append(
            {
                "global_scale": s,
                "e_total (pooled)": e_total,
                "e_flexible (pooled)": _pooled(recs, "flexible", method),
                "in_conway": in_conway_range(e_total, conway),
            }
        )
    return pd.DataFrame(flex_rows), pd.DataFrame(total_rows), pd.DataFrame(c_rows)


def choose_scale(c: pd.DataFrame, cal: dict) -> tuple[float, bool]:
    """Calibration rule: in Conway's range and closest to the target; else closest overall."""
    target = float(cal["target_elasticity"])
    c = c.assign(dist=(c["e_total (pooled)"] - target).abs().fillna(float("inf")))
    pool = c[c["in_conway"]] if c["in_conway"].any() else c
    best = pool.sort_values(["dist", "global_scale"]).iloc[0]
    return float(best["global_scale"]), bool(best["in_conway"])


def decomposition_table(sweep: Sweep, scale: float, method: str) -> pd.DataFrame:
    rows = []
    for cond in ("uniform", "graded"):
        recs = sweep.at(scale, cond)
        row: dict = {"condition": cond}
        eo = {}
        for o in ORIENTATIONS:
            eo[o] = _pooled(
                [r for r in recs if ORIENTATION_OF[r.household_id] == o], "flexible", method
            )
            row[f"{o} e_flex"] = eo[o]
        finite = [v for v in eo.values() if v == v]
        row["comfort spread (max-min)"] = max(finite) - min(finite) if finite else float("nan")
        ea = {}
        for a in "ABC":
            ea[a] = _pooled([r for r in recs if r.archetype == a], "flexible", method)
            row[f"arch {a} e_flex"] = ea[a]
        row["C>B>A elastic"] = _ordered(ea["C"], ea["B"], ea["A"])
        row["e_total (pooled)"] = _pooled(recs, "total", method)
        rows.append(row)
    return pd.DataFrame(rows)


def per_household_table(sweep: Sweep, method: str) -> pd.DataFrame:
    rows = []
    for (hid, scale, cond), r in sorted(
        sweep.records.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])
    ):
        rows.append(
            {
                "condition": cond,
                "global_scale": scale,
                "household_id": hid,
                "comfort_orientation": ORIENTATION_OF[hid],
                "archetype": r.archetype,
                "wtp_factor": sweep.factor(hid, cond),
                "hvac_kwh_base": r.q_base["hvac"],
                "hvac_kwh_cpp": r.q_cpp["hvac"],
                "e_flexible": r.elasticity("flexible", method),
                "e_total": r.elasticity("total", method),
            }
        )
    return pd.DataFrame(rows)


def trajectory_wtp_table(log: pd.DataFrame) -> pd.DataFrame:
    """Raw LLM WTP on the simulated trajectories, by degrees over the persona's own ceiling."""
    p = log[~log["forced"]].copy()
    p["indoor_c"] = (
        p["prompt"].str.extract(r"indoor temperature \(C\)\s+: ([\d.]+)")[0].astype(float)
    )
    p["over_c"] = p["indoor_c"] - p["band_hi_c"]
    bins = [-99, -1.0, -0.5, 0.0, 0.5, 1.0, 99]
    labels = ["< -1.0", "-1.0..-0.5", "-0.5..0", "0..0.5", "0.5..1.0", "> 1.0"]
    p["over_ceiling_bin"] = pd.cut(p["over_c"], bins=bins, labels=labels, right=False)
    p = p.drop_duplicates("prompt")  # one row per distinct prompt
    t = p.pivot_table(
        index="over_ceiling_bin", columns="comfort_orientation", values="wtp",
        aggfunc=["mean", "count"], observed=False,
    )  # fmt: skip
    out = pd.DataFrame(index=t.index)
    for o in ORIENTATIONS:
        if ("mean", o) in t.columns:
            out[f"{o} mean_wtp"] = t[("mean", o)]
            out[f"{o} n"] = t[("count", o)].fillna(0).astype(int)
    return out.reset_index()


def run_sweep(args, out: Path) -> None:
    project = args.project
    runner = _runner(dry_run=args.dry_run or project)
    sweep = Sweep(
        runner,
        project=project,
        max_live=None if project else args.max_live,
        fixed_band=args.fixed_band,
        orientations=tuple(args.orientations.split(",")),
    )
    scales = [float(s) for s in args.scales.split(",")]
    method = runner.cfg.raw["elasticity_method"]
    t0 = time.time()

    a, a_total, c = scale_tables(sweep, scales, method)
    best, in_range = choose_scale(c, runner.cfg.calibration)
    decomp = None if args.no_decomp else decomposition_table(sweep, best, method)
    log = pd.concat(sweep.logs, ignore_index=True)
    prompted = log[~log["forced"]]
    wall = time.time() - t0

    if project:
        n_uncached = len(sweep.uncached)
        ub = sweep.upper_bound_prompts()
        rpm = int((runner.cfg.llm_cfg.get("gemini") or {}).get("rate_limit_rpm", 5))
        proj = {
            "households": {hid: ORIENTATION_OF[hid] for hid in sweep.households},
            "scales": scales,
            "decomposition_scale_on_stub": best,
            "llm_window": LLM_WINDOW,
            "prompted_decisions": len(prompted),
            "distinct_prompts_on_stub_trajectories": int(prompted["prompt"].nunique()),
            "projected_live_calls": n_uncached,
            "projected_cache_hit_rate": round(1 - n_uncached / len(prompted), 3),
            "structural_upper_bound_live_calls": ub,
            "projected_wall_minutes_at_rpm": round(n_uncached / rpm, 1),
            "upper_bound_wall_minutes_at_rpm": round(ub / rpm, 1),
            "rate_limit_rpm": rpm,
            "note": "projection runs the dry-run stub; live trajectories differ (estimate)",
        }
        (out / "projection.json").write_text(json.dumps(proj, indent=2, default=str))
        print(json.dumps(proj, indent=2, default=str))
        return

    log.to_parquet(out / "sweep_llm_log.parquet")
    traj = trajectory_wtp_table(log)
    hh = per_household_table(sweep, method)
    for name, df in (("a_flexible_by_group", a), ("a_total_by_group", a_total),
                     ("b_decomposition", decomp), ("c_total_vs_scale", c), ("per_household", hh),
                     ("trajectory_wtp", traj)):  # fmt: skip
        if df is not None:
            df.to_csv(out / f"{name}.csv", index=False)
    stats = {
        "prompted_decisions": len(prompted),
        "distinct_prompts": int(prompted["prompt"].nunique()),
        "live_calls": sweep.live_calls,
        "cache_hits": sweep.cache_hits,
        "cache_hit_rate": (
            None if runner.cfg.dry_run else round(1 - sweep.live_calls / len(prompted), 3)
        ),
        "misread_calls": int(prompted["misreads"].map(len).gt(0).sum()),
        "wall_seconds": round(wall, 1),
        "dry_run": runner.cfg.dry_run,
    }
    (out / "call_stats.json").write_text(json.dumps(stats, indent=2))
    report = [
        f"### Comfort-orientation sweep -- {'DRY RUN (stub)' if runner.cfg.dry_run else 'LIVE'}",
        "",
        f"HVAC only, seed {SEED}, {runner.cfg.dates[0]}, CPP {runner.cfg.raw['cpp']['start']}-"
        f"{runner.cfg.raw['cpp']['end']} (0.55 -> 1.20 $/kWh), LLM window {LLM_WINDOW[0]}-"
        f"{LLM_WINDOW[1]} (thermostat outside), method `{method}`. Households: "
        + ", ".join(f"{hid} ({o}, {runner.archetype(hid)})" for hid, o in ORIENTATION_OF.items()
                   if hid in sweep.households),  # fmt: skip
        "",
        "Bands: "
        + (
            f"fixed at +/-{args.fixed_band}C for every household (orientation sentence unchanged)"
            if args.fixed_band is not None
            else "each persona's own"
        )
        + ".",
        "",
        f"Graded archetype factors {runner.cfg.archetype_factors}. Elasticity is pooled "
        "per group (sum kWh, then ratio); more negative = more elastic. Ordering flags need "
        "strict separation end to end; NaN (no baseline HVAC in the window) -> False.",
        "",
        "#### (a1) flexible-layer (HVAC) elasticity vs global_scale",
        "",
        _md(a),
        "",
        "#### (a2) whole-household elasticity vs global_scale",
        "",
        _md(a_total),
        "",
        *(
            []
            if decomp is None
            else [
                f"#### (b) decomposition at global_scale {best}: uniform (1.0/1.0/1.0) vs "
                "graded factors",
                "",
                _md(decomp),
                "",
            ]
        ),
        f"#### (c) whole-household elasticity vs global_scale "
        f"(Conway {list(runner.cfg.conway_range)}, "
        f"target {runner.cfg.calibration['target_elasticity']})",
        "",
        _md(c),
        "",
        f"chosen scale: {best} ({'in range' if in_range else 'NONE in range -- closest shown'})",
        "",
        "#### raw LLM WTP on the trajectories (distinct prompts), by C over own ceiling",
        "",
        _md(traj),
        "",
        "#### (d) call accounting",
        "",
        "```",
        json.dumps(stats, indent=2),
        "```",
    ]
    (out / "sweep_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["premise", "grid", "sweep"])
    parser.add_argument(
        "--scales", default=DEFAULT_SCALES, help="comma-separated global_scale values"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project", action="store_true", help="sweep: project live calls, no API")
    parser.add_argument("--max-live", type=int, default=400, help="sweep: live-call budget gate")
    parser.add_argument(
        "--fixed-band", type=float, default=None,
        help="sweep: same +/-C comfort band for every household (orientation sentence kept)",
    )  # fmt: skip
    parser.add_argument(
        "--orientations", default=",".join(ORIENTATIONS),
        help="sweep: comma-separated comfort orientations to simulate (default all)",
    )  # fmt: skip
    parser.add_argument("--no-decomp", action="store_true", help="sweep: skip uniform decomposition")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sub = "projection" if args.project else ("dryrun" if args.dry_run else "live")
    base = OUT if args.fixed_band is None else OUT / f"fixed_band_{args.fixed_band:g}"
    out = Path(args.out or base / sub)
    out.mkdir(parents=True, exist_ok=True)
    write_provenance(out)
    if args.step == "sweep":
        run_sweep(args, out)
        return

    runner = _runner(args.dry_run)
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


if __name__ == "__main__":
    main()
