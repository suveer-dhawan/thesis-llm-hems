"""Phase 3 summaries, computed only from per-household raw artifacts.

Responsibility: the one place every Phase 3 number is derived. A run (or a
shard of one) writes raw, per-household artifacts:

  curves.parquet             every baseline/CPP day-run, one row per step
                             (household x agent x seed x date x day x t)
  scenario.csv               discriminating-scenario verdicts (if run)
  llm_decision_log.parquet   every LLM decision (prompt, wtp, model version, ...)
  rule_decision_log.parquet  every rule-based satisficer decision
  personas_used.csv          the households' persona rows
  run_meta.json              households, steps, LLM call counts, prompt bound

`finalize(out, artifacts, resolved)` turns those into every summary file
and `report.md`. `experiments/run_phase3.py` calls it on its own artifacts;
`experiments/merge_shards.py` concatenates several shards' artifacts and
calls the same function -- so a merged run and an unsharded run are the same
computation on the same (sorted) rows.

Per agent (and per seed for seeded agents) every summary carries: aggregate
peak kW + time and PAR (both days), the load in the first post-CPP
half-hour (20:00-20:30) vs the CPP-window mean, comfort degree-hours above
the band ceiling + max exceedance, pooled flexible-layer elasticity, the
daily flexible kWh change (CPP day - baseline day), and pooled
whole-household elasticity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hems.analysis.elasticity import (
    ElasticityRecord,
    elasticity_table,
    in_conway_range,
    pooled_elasticity,
    records_to_frame,
)
from hems.analysis.metrics import curve_metrics, rebound_peak_kw
from hems.analysis.plots import plot_four_way
from hems.environment.appliances import _clock_to_step
from hems.environment.tariff import step_to_clock

SERIES = {
    "milp": "MILP",
    "noisy_milp": "noisy-MILP",
    "rule_based": "rule-based",
    "llm": "LLM",
}
CURVE_SORT = ["agent", "seed", "household_id", "date", "day", "t"]
LOG_SORT = ["household_id", "run_label", "seed", "run_date", "run_wtp_scale", "t", "appliance"]
DT_HOURS = 0.5


@dataclass
class Artifacts:
    """The raw per-household artifacts of a run or a merge of shards."""

    curves: pd.DataFrame
    personas: pd.DataFrame
    meta: dict
    scenario: pd.DataFrame | None = None
    llm_log: pd.DataFrame | None = None
    rule_log: pd.DataFrame | None = None

    def sorted(self) -> "Artifacts":
        def srt(df, cols):
            if df is None or df.empty:
                return df
            return df.sort_values([c for c in cols if c in df.columns], kind="mergesort")

        return Artifacts(
            curves=srt(self.curves, CURVE_SORT).reset_index(drop=True),
            personas=srt(self.personas, ["customer_ID"]).reset_index(drop=True),
            meta=self.meta,
            scenario=None
            if self.scenario is None
            else srt(self.scenario, ["variant", "household_id", "agent", "seed"]).reset_index(
                drop=True
            ),
            llm_log=None
            if self.llm_log is None
            else srt(self.llm_log, LOG_SORT).reset_index(drop=True),
            rule_log=None
            if self.rule_log is None
            else srt(self.rule_log, LOG_SORT).reset_index(drop=True),
        )

    def write(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.curves.to_parquet(out / "curves.parquet", index=False)
        self.personas.to_csv(out / "personas_used.csv", index=False)
        (out / "run_meta.json").write_text(json.dumps(self.meta, indent=2))
        if self.scenario is not None:
            self.scenario.to_csv(out / "scenario.csv", index=False)
        if self.llm_log is not None and not self.llm_log.empty:
            self.llm_log.to_parquet(out / "llm_decision_log.parquet", index=False)
        if self.rule_log is not None and not self.rule_log.empty:
            self.rule_log.to_parquet(out / "rule_decision_log.parquet", index=False)

    @classmethod
    def read(cls, d: Path) -> "Artifacts":
        def opt(name, reader):
            return reader(d / name) if (d / name).exists() else None

        return cls(
            curves=pd.read_parquet(d / "curves.parquet"),
            personas=pd.read_csv(d / "personas_used.csv"),
            meta=json.loads((d / "run_meta.json").read_text()),
            scenario=opt("scenario.csv", pd.read_csv),
            llm_log=opt("llm_decision_log.parquet", pd.read_parquet),
            rule_log=opt("rule_decision_log.parquet", pd.read_parquet),
        )


# --------------------------------------------------------------------------
# raw artifacts from a runner's day-runs
# --------------------------------------------------------------------------


def curves_from_runs(runs, archetype_of, orientation_of) -> pd.DataFrame:
    """One row per step of every baseline/CPP `DayRun` in `runs`."""
    rows = []
    for r in runs:
        res = r.result
        done = res.washer_completed
        for s in res.steps:
            rows.append(
                {
                    "household_id": r.household_id,
                    "archetype": archetype_of(r.household_id),
                    "comfort_orientation": orientation_of(r.household_id),
                    "agent": r.agent,
                    "seed": r.seed,
                    "wtp_scale": r.wtp_scale,
                    "date": r.date.isoformat(),
                    "day": r.label,
                    "t": s.t,
                    "clock": s.clock,
                    "rate": s.rate,
                    "is_cpp": s.is_cpp,
                    "base_kwh": s.base_kwh,
                    "hvac_kwh": s.hvac_energy_kwh,
                    "washer_kwh": s.washer_energy_kwh,
                    "flexible_kwh": s.flexible_kwh,
                    "total_kwh": s.total_kwh,
                    "indoor_temp_c": s.indoor_temp_c,
                    "band_hi_c": res.hvac_band_hi_c,
                    "in_band": s.in_band,
                    "washer_completed": done,
                }
            )
    return pd.DataFrame(rows)


def logs_from_runs(runs, attr: str) -> pd.DataFrame | None:
    frames = [
        getattr(r, attr).assign(
            household_id=r.household_id,
            seed=r.seed,
            run_label=r.label,
            run_date=r.date.isoformat(),
            run_wtp_scale=r.wtp_scale,
        )
        for r in runs
        if getattr(r, attr) is not None and not getattr(r, attr).empty
    ]
    return pd.concat(frames, ignore_index=True) if frames else None


# --------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------


def records_from_curves(curves: pd.DataFrame) -> dict[tuple[str, int], list[ElasticityRecord]]:
    """(agent, seed) -> one ElasticityRecord per household x date (CPP window from is_cpp)."""
    out: dict[tuple[str, int], list[ElasticityRecord]] = {}
    for (agent, seed, hid, date), g in curves.groupby(
        ["agent", "seed", "household_id", "date"], sort=True
    ):
        base = g[g["day"] == "base"].set_index("t")
        cpp = g[g["day"] == "cpp"].set_index("t")
        if base.empty or cpp.empty:
            continue
        window = sorted(cpp.index[cpp["is_cpp"]])

        def q(df):
            w = df.loc[window]
            hvac, washer, base_kwh = w["hvac_kwh"].sum(), w["washer_kwh"].sum(), w["base_kwh"].sum()
            return (
                {"hvac": hvac, "washer": washer, "flexible": hvac + washer,
                 "total": base_kwh + hvac + washer},
                w["rate"].mean(),
            )  # fmt: skip

        qb, pb = q(base)
        qc, pc = q(cpp)
        end = (window[-1] + 1) * 30
        rec = ElasticityRecord(
            household_id=int(hid),
            agent=str(agent),
            seed=int(seed),
            archetype=str(g["archetype"].iloc[0]),
            window_start=cpp.loc[window[0], "clock"],
            window_end=f"{end // 60:02d}:{end % 60:02d}",
            p_base=float(pb),
            p_cpp=float(pc),
            q_base={k: float(v) for k, v in qb.items()},
            q_cpp={k: float(v) for k, v in qc.items()},
            date=str(date),
            wtp_scale=float(g["wtp_scale"].iloc[0]) if agent in ("llm", "rule_based") else None,
        )
        out.setdefault((str(agent), int(seed)), []).append(rec)
    return out


def _pop_curve(g: pd.DataFrame, col: str = "total_kwh") -> list[float]:
    return g.groupby("t", sort=True)[col].sum().tolist()


def agent_summary(curves: pd.DataFrame, resolved: dict) -> pd.DataFrame:
    """One row per (agent, seed, date): the item-7 metrics, aggregated over households."""
    p3 = resolved["phase3"]
    method = p3["elasticity_method"]
    conway = tuple(float(x) for x in p3["conway_range"])
    cpp_lo, cpp_hi = _clock_to_step(p3["cpp"]["start"]), _clock_to_step(p3["cpp"]["end"])
    cpp_steps = range(cpp_lo, cpp_hi)
    post = cpp_hi  # first post-CPP half-hour (20:00-20:30 for a 17:00-20:00 event)
    post_label = f"{step_to_clock(post)}-{step_to_clock(post + 1)}"
    records = records_from_curves(curves)
    clocks = [step_to_clock(t) for t in range(48)]
    rows = []
    for (agent, seed, date), g in curves.groupby(["agent", "seed", "date"], sort=False):
        row = {
            "agent": agent,
            "seed": int(seed),
            "date": date,
            "n_households": int(g["household_id"].nunique()),
        }
        flex = {}
        for day in ("base", "cpp"):
            d = g[g["day"] == day]
            total = _pop_curve(d)
            m = curve_metrics(total, clocks)
            cpp_mean_kw = sum(total[t] for t in cpp_steps) * 2 / len(cpp_steps)
            post_kw = total[post] * 2
            over = (d["indoor_temp_c"] - d["band_hi_c"]).clip(lower=0)
            flex[day] = float(d["flexible_kwh"].sum())
            per_run = d.groupby("household_id", sort=True)
            row.update(
                {
                    f"{day}_peak_kw": m["peak_kw"],
                    f"{day}_time_of_peak": m["time_of_peak"],
                    f"{day}_par": m["par"],
                    f"{day}_cpp_window_mean_kw": round(cpp_mean_kw, 3),
                    f"{day}_post_{post_label}_kw": round(post_kw, 3),
                    f"{day}_post_over_cpp_ratio": round(post_kw / cpp_mean_kw, 3)
                    if cpp_mean_kw
                    else float("nan"),
                    f"{day}_degree_hours_above_ceiling": round(float(over.sum()) * DT_HOURS, 3),
                    f"{day}_max_exceedance_c": round(float(over.max()), 3),
                    f"{day}_flexible_kwh": round(flex[day], 3),
                    f"{day}_washer_completed": int(per_run["washer_completed"].first().sum()),
                }
            )
        recs = [r for r in records.get((agent, int(seed)), []) if r.date == date]
        e_total = pooled_elasticity(recs, "total", method)
        dflex = flex["cpp"] - flex["base"]
        row.update(
            {
                "elasticity_flexible": pooled_elasticity(recs, "flexible", method),
                "elasticity_flexible_midpoint": pooled_elasticity(recs, "flexible", "midpoint"),
                "elasticity_total": e_total,
                "elasticity_total_midpoint": pooled_elasticity(recs, "total", "midpoint"),
                "total_in_conway": in_conway_range(e_total, conway),
                "daily_flexible_kwh_change": round(dflex, 3),
                "daily_flexible_pct_change": round(100 * dflex / flex["base"], 2)
                if flex["base"]
                else float("nan"),
            }
        )
        rows.append(row)
    order = {a: i for i, a in enumerate(SERIES)}
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["agent", "seed", "date"], key=lambda c: c.map(order) if c.name == "agent" else c
    ).reset_index(drop=True)


def load_metrics(curves: pd.DataFrame, resolved: dict) -> pd.DataFrame:
    """PAR / peak / time-of-peak / rebound per household and population, primary seed."""
    p3 = resolved["phase3"]
    primary = int(resolved["seeds"]["primary"])
    cpp_end = _clock_to_step(p3["cpp"]["end"])
    reb = int(round(float(p3["rebound_hours"]) * 2))
    clocks = [step_to_clock(t) for t in range(48)]
    prim = curves[curves["seed"] == primary]
    rows = []

    def add(scope, series, day, curve):
        m = curve_metrics(curve, clocks)
        m.update(scope=scope, series=series, day=day,
                 rebound_peak_kw=round(rebound_peak_kw(curve, cpp_end, reb), 3))  # fmt: skip
        rows.append(m)

    for date, gd in prim.groupby("date", sort=True):
        for day in ("base", "cpp"):
            d = gd[gd["day"] == day]
            for hid, gh in d.groupby("household_id", sort=True):
                any_agent = gh[gh["agent"] == gh["agent"].iloc[0]]
                add(str(hid), "SGSC base", day, any_agent.sort_values("t")["base_kwh"].tolist())
                for agent in SERIES:
                    ga = gh[gh["agent"] == agent]
                    if not ga.empty:
                        add(str(hid), SERIES[agent], day, ga.sort_values("t")["total_kwh"].tolist())
            first = d[d["agent"] == d["agent"].iloc[0]]
            add("population", "SGSC base", day, _pop_curve(first, "base_kwh"))
            for agent in SERIES:
                ga = d[d["agent"] == agent]
                if not ga.empty:
                    add("population", SERIES[agent], day, _pop_curve(ga))
    metrics = pd.DataFrame(rows)
    piv = metrics.pivot_table(index=["scope", "series"], columns="day", values="rebound_peak_kw")
    metrics = metrics.merge(
        (piv["cpp"] - piv["base"]).rename("rebound_delta_kw").reset_index(), on=["scope", "series"]
    )
    cols = ["scope", "series", "day", "par", "peak_kw", "time_of_peak", "rebound_peak_kw",
            "rebound_delta_kw"]  # fmt: skip
    return metrics[cols]


def population_overlay(curves: pd.DataFrame, resolved: dict, path: Path, title: str) -> None:
    p3 = resolved["phase3"]
    primary = int(resolved["seeds"]["primary"])
    prim = curves[(curves["seed"] == primary) & (curves["date"] == curves["date"].min())]
    series = {"base": {}, "cpp": {}}
    for day in series:
        d = prim[prim["day"] == day]
        first = d[d["agent"] == d["agent"].iloc[0]]
        series[day]["SGSC base"] = _pop_curve(first, "base_kwh")
        for agent, label in SERIES.items():
            ga = d[d["agent"] == agent]
            if not ga.empty:
                series[day][label] = _pop_curve(ga)
    plot_four_way(
        series["base"],
        series["cpp"],
        [step_to_clock(t) for t in range(48)],
        (_clock_to_step(p3["cpp"]["start"]), _clock_to_step(p3["cpp"]["end"])),
        path,
        title,
    )


# --------------------------------------------------------------------------
# LLM call projection (dry run)
# --------------------------------------------------------------------------


def llm_projection(
    llm_log: pd.DataFrame | None,
    llm_cfg: dict,
    cache_paths: list[str],
    prompt_upper_bound: int,
) -> dict:
    """Projected live calls: distinct (seed, schema, prompt) keys not already cached.

    Keys are computed by a real `LLMClient` per seed (same identity: provider,
    model, temperature, seed, max_output_tokens, schema hash) built with a
    placeholder key -- it is only asked for cache keys, never called.
    """
    from hems.agents.llm_agent import SCHEMA_BY_APPLIANCE
    from hems.llm.client import _DiskCache, build_llm_client

    provider = llm_cfg.get("provider", "gemini")
    rpm = float((llm_cfg.get(provider) or {}).get("rate_limit_rpm") or 0) or None
    if llm_log is None or llm_log.empty:
        prompted = pd.DataFrame(columns=["seed", "appliance", "prompt"])
    else:
        prompted = llm_log[~llm_log["forced"]]
    cached_keys: set[str] = set()
    for p in cache_paths:
        if p and Path(p).exists():
            cached_keys |= set(_DiskCache(Path(p)).data)
    distinct: set[str] = set()
    for seed, g in prompted.groupby("seed"):
        client = build_llm_client(provider, llm_cfg, None, seed=int(seed),
                                  api_key="projection-only-no-calls")  # fmt: skip
        for appliance, prompt in zip(g["appliance"], g["prompt"]):
            distinct.add(client.cache_key(prompt, SCHEMA_BY_APPLIANCE[appliance]))
    by_app = prompted["appliance"].value_counts().to_dict() if len(prompted) else {}
    live = len(distinct - cached_keys)

    def hours(n):
        return round(n / rpm / 60, 2) if rpm else None

    return {
        "prompted_decisions": int(len(prompted)),
        "prompted_by_appliance": {k: int(v) for k, v in by_app.items()},
        "distinct_cache_keys": len(distinct),
        "already_in_cache": len(distinct & cached_keys),
        "cache_paths_checked": cache_paths,
        "projected_live_calls": live,
        "upper_bound_prompted_decisions": int(prompt_upper_bound),
        "rate_limit_rpm": rpm,
        "projected_hours_at_rpm": hours(live),
        "upper_bound_hours_at_rpm": hours(prompt_upper_bound),
        "caveat": "counted on the dry-run stub's trajectories. The HVAC decision count is "
        "exact (prompted every step), but which prompts repeat -- across a household's baseline, "
        "CPP and scenario days until they diverge, and across households that reach the same "
        "rendered state -- depends on the real LLM's decisions, so the distinct-key count is an "
        "estimate. The upper bound (every HVAC step + every washer step from release to latest "
        "start, no cache reuse) is the hard ceiling. Retries not counted.",
    }


# --------------------------------------------------------------------------
# finalize: every summary file + report.md
# --------------------------------------------------------------------------


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.values]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def finalize(out: Path, artifacts: Artifacts, resolved: dict) -> dict:
    """Write every summary + report.md from raw artifacts; returns summary.json's content."""
    a = artifacts.sorted()
    a.write(out)
    p3 = resolved["phase3"]
    steps = resolved["run"]["steps"]
    method = p3["elasticity_method"]
    conway = tuple(float(x) for x in p3["conway_range"])
    primary = int(resolved["seeds"]["primary"])
    ids = sorted(int(h) for h in a.curves["household_id"].unique())
    band = resolved["band"]
    scale = resolved["wtp_scaling"]["global_scale"]
    dry = resolved["dry_run"]
    provider = resolved["llm"].get("provider")
    model = (resolved["llm"].get(provider) or {}).get("model")

    summary_df = agent_summary(a.curves, resolved)
    summary_df.to_csv(out / "summary.csv", index=False)
    summary = {
        "households": ids,
        "n_households": len(ids),
        "dates": sorted(a.curves["date"].unique().tolist()),
        "band": band,
        "wtp_scaling": resolved["wtp_scaling"],
        "seeds": resolved["seeds"],
        "conway_range": list(conway),
        "elasticity_method": method,
        "by_agent_seed": json.loads(summary_df.to_json(orient="records")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    band_txt = (
        f"fixed +/-{band['fixed_band_c']} C for every household"
        if band["mode"] == "fixed"
        else "each household's persona band"
    )
    report = [
        "# Phase 3 run -- "
        + ("DRY RUN (stub LLM, numbers are pipeline checks only)" if dry else "LIVE"),
        "",
        f"- households ({len(ids)}): {ids}",
        f"- steps: {steps}",
        f"- band mode: `{band['mode']}` ({band_txt}); comfort orientation = persona's",
        f"- WTP scaling: global_scale {scale} x archetype factor "
        f"{resolved['wtp_scaling']['archetype_factors']}, identical for "
        f"{resolved['wtp_scaling']['applies_to']}; calibration enabled: "
        f"{resolved['wtp_scaling']['calibration_enabled']}",
        f"- seeds: primary {primary}; LLM {resolved['seeds']['llm']} (T=0); noisy-MILP "
        f"{resolved['seeds']['noisy_milp']}; MILP + rule-based deterministic",
        f"- thermal heterogeneity: {resolved['thermal_heterogeneity']}",
        f"- rule-based k ({resolved['rule_based']['band_mode_used']} band fit): "
        f"{resolved['rule_based']['hvac_k_used']}, "
        f"w_max {resolved['rule_based']['washer_w_max_used']}",
        f"- provider/model: {provider} / {model}",
        "- provenance: `provenance.json`; full config: `config_resolved.yaml`",
        "",
        "## Summary by agent (population aggregate)",
        "",
        f"Peak / PAR on the summed load of all households; post = first post-CPP half-hour vs "
        f"the CPP-window mean; degree-hours = sum over households of max(0, indoor - ceiling) x "
        f"0.5 h; elasticities pooled (`{method}`; midpoint alongside), total = whole household; "
        f"Conway {list(conway)}.",
        "",
    ]
    for day in ("base", "cpp"):
        cols = ["agent", "seed"] + [c for c in summary_df.columns if c.startswith(f"{day}_")]
        report += [f"### {'Baseline' if day == 'base' else 'CPP'} day", "",
                   _md_table(summary_df[cols]), ""]  # fmt: skip
    ecols = ["agent", "seed", "elasticity_flexible", "elasticity_flexible_midpoint",
             "elasticity_total", "elasticity_total_midpoint", "total_in_conway",
             "daily_flexible_kwh_change", "daily_flexible_pct_change"]  # fmt: skip
    report += ["### Elasticity and daily flexible change", "", _md_table(summary_df[ecols]), ""]

    records = records_from_curves(a.curves)
    if "elasticity" in steps:
        prim = {k: v for k, v in records.items() if k[1] == primary}
        per_hh = pd.concat([records_to_frame(r, method) for r in prim.values()], ignore_index=True)
        per_hh.to_csv(out / "elasticity_per_household.csv", index=False)
        agg = pd.concat(
            [elasticity_table(r, "agent", method, conway) for r in prim.values()], ignore_index=True
        )
        by_arch = pd.concat(
            [elasticity_table(r, "archetype", method, conway).assign(agent=k[0])
             for k, r in prim.items()],
            ignore_index=True,
        ).sort_values(["agent", "archetype"])  # fmt: skip
        agg.to_csv(out / "elasticity_aggregate.csv", index=False)
        by_arch.to_csv(out / "elasticity_by_archetype.csv", index=False)
        ordering = {}
        for agent in ("llm", "rule_based"):
            e = by_arch[by_arch["agent"] == agent].set_index("archetype")["elasticity_total"]
            ordering[agent] = {
                "values": e.to_dict(),
                "C_more_elastic_than_B": bool(e.get("C", float("nan")) < e.get("B", float("nan"))),
                "B_more_elastic_than_A": bool(e.get("B", float("nan")) < e.get("A", float("nan"))),
            }
        (out / "elasticity_summary.json").write_text(
            json.dumps(
                {
                    "households": ids,
                    "seed": primary,
                    "method": method,
                    "conway_range": list(conway),
                    "aggregate": agg.to_dict(orient="records"),
                    "by_archetype": by_arch.to_dict(orient="records"),
                    "archetype_ordering": ordering,
                },
                indent=2,
                default=float,
            )  # fmt: skip
        )
        comp = ["hvac", "washer", "flexible", "total"]
        report += [
            f"## Elasticity detail (seed {primary})",
            "",
            _md_table(agg),
            "",
            "### By archetype (prediction: C more elastic than B than A)",
            "",
            _md_table(by_arch[["agent", "archetype", "n"] + [f"elasticity_{c}" for c in comp]]),
            "",
            f"Archetype ordering: {json.dumps(ordering)}",
            "",
        ]
    if "overlay" in steps:
        lm = load_metrics(a.curves, resolved)
        lm.to_csv(out / "load_metrics.csv", index=False)
        population_overlay(
            a.curves, resolved, out / "overlay_population.png",
            f"population sum ({len(ids)} households) | band {band['mode']} | seed {primary} | "
            f"global_scale {scale}",
        )  # fmt: skip
        report += [
            "## Load-curve metrics (population, primary seed)",
            "",
            _md_table(lm[lm["scope"] == "population"]),
            "",
            "Figures: `overlay_population.png`, `overlay_<household>.png`.",
            "",
        ]
    if "seeds" in steps:
        spread = summary_df[summary_df["agent"].isin(["noisy_milp", "llm"])][
            ["agent", "seed", "elasticity_total", "elasticity_flexible", "cpp_peak_kw",
             "daily_flexible_kwh_change"]
        ]  # fmt: skip
        spread.to_csv(out / "seed_spread.csv", index=False)
        agg_spread = spread.groupby("agent")[["elasticity_total", "elasticity_flexible"]].agg(
            ["min", "max", "std"]
        )
        agg_spread.columns = [f"{x}_{y}" for x, y in agg_spread.columns]
        agg_spread = agg_spread.reset_index()
        agg_spread.to_csv(out / "seed_spread_summary.csv", index=False)
        report += ["## Seed spread", "", _md_table(spread), "", _md_table(agg_spread), ""]
    if "scenario" in steps and a.scenario is not None:
        sc = p3["scenario"]
        coh = a.scenario.groupby(["agent", "variant"])["coherent"].agg(["mean", "count"])
        report += [
            "## Discriminating scenario (single CPP spike + near washer deadline)",
            "",
            f"Spike {sc['spike']['start']}-{sc['spike']['end']} ({sc['spike']['severity']}). "
            "Coherence judged against the scenario's true deadline.",
            "",
            _md_table(coh.rename(columns={"mean": "coherent_share"}).reset_index()),
            "",
        ]
    meta = a.meta
    if dry:
        proj = llm_projection(
            a.llm_log, resolved["llm"], resolved["run"]["cache_paths"],
            int(meta.get("llm_prompt_upper_bound", 0)),
        )  # fmt: skip
        (out / "llm_projection.json").write_text(json.dumps(proj, indent=2))
        report += ["## Projected live LLM calls (from this dry run)", "", "```",
                   json.dumps(proj, indent=2), "```", ""]  # fmt: skip
    report += [
        "## LLM call accounting",
        "",
        "```",
        json.dumps(
            {
                k: meta.get(k)
                for k in ("llm_prompted_decisions", "llm_live_calls", "llm_model_versions")
            },
            indent=2,
        ),  # fmt: skip
        "```",
        "",
    ]
    (out / "report.md").write_text("\n".join(report))
    return summary
