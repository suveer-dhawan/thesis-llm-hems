"""No-LLM population run: MILP vs noisy-MILP vs rule-based satisficer.

All 24 persona households, the Phase 3 date, baseline day (no CPP) + CPP day
(`phase3.cpp`), in two band modes:

  persona  each household's own persona band (tolerance_band_c)
  fixed    every household at +/-2.0 C around the same setpoint

The comfort orientation (which selects the rule-based k) stays the persona's
in both modes; only the band changes, and it changes for every agent (MILP
constraint, noisy-MILP jitter centre, rule-based ceiling).

Rule-based runs at global_scale 1.0 and archetype factor 1.0 through the LLM
agent's own halt (`agents/rule_based_agent.py`), k / w_max from
`experiment.yaml` -> `rule_based:` (fitted by `experiments/fit_rule_based.py`).
Noisy-MILP uses one seed (`--seed`, default the Phase 3 fit seed 42) and, as in
`phase3.py`, its jittered params drive its own accounting. No LLM client is
constructed: zero API calls by construction.

Reported per band mode x agent: aggregate (24-household sum) peak kW and
time, PAR, aggregate load over the post-CPP steps (40-43) vs the CPP steps
(34-39) on both days, pooled flexible-layer arc elasticity over the CPP
window, and the daily flexible kWh change (CPP day - baseline day).

Usage (from llm-hems/):
  PYTHONPATH=src .venv/bin/python experiments/run_rule_based.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from hems.agents.milp_agent import MILPAgent  # noqa: E402
from hems.agents.noisy_milp_agent import NoisyMILPAgent  # noqa: E402
from hems.agents.rule_based_agent import RuleBasedParams, RuleBasedSatisficerAgent  # noqa: E402
from hems.analysis.elasticity import household_elasticity, pooled_elasticity  # noqa: E402
from hems.analysis.metrics import (  # noqa: E402
    curve_metrics,
    degree_hours_above_ceiling,
    max_exceedance_c,
)
from hems.environment.simulator import RunResult, run_simulation  # noqa: E402
from hems.phase3 import Phase3Config, Phase3Runner  # noqa: E402
from hems.provenance import write_provenance  # noqa: E402

OUT = Path("results/phase3/rule_based")
AGENTS = ("milp", "noisy_milp", "rule_based")
BAND_MODES = {"persona": None, "fixed": 2.0}  # None = the persona's own band
FIXED_BAND_C = 2.0
GLOBAL_SCALE, FACTOR = 1.0, 1.0
LABELS = {"milp": "MILP", "noisy_milp": "Noisy-MILP", "rule_based": "Rule-based satisficer"}
# Reference palette categorical slots 1-3 (dataviz skill, light mode).
COLORS = {"milp": "#2a78d6", "noisy_milp": "#eb6834", "rule_based": "#1baf7a"}


def simulate(runner: Phase3Runner, params: RuleBasedParams, hid: int, agent: str,
             band_c: float | None, label: str, seed: int) -> RunResult:  # fmt: skip
    date = runner.cfg.dates[0]
    household, hvac, washer = runner.env(hid)
    if band_c is not None:
        hvac = replace(hvac, tolerance_band_c=band_c)
    tariff = runner.tariff_for(date, label)
    if agent == "milp":
        a = MILPAgent(household, tariff, hvac, washer, date)
    elif agent == "noisy_milp":
        a = NoisyMILPAgent(household, tariff, hvac, washer, date, seed=seed,
                           jitter_config_path=runner.cfg.experiment_path)  # fmt: skip
        hvac, washer = a.jittered_hvac_params, a.jittered_washer_params
    elif agent == "rule_based":
        a = RuleBasedSatisficerAgent(params, wtp_scale=GLOBAL_SCALE, wtp_factor=FACTOR)
    else:
        raise ValueError(agent)
    return run_simulation(household, tariff, hvac, washer, a, date)


def aggregate(results: list[RunResult], attr: str = "total_kwh") -> list[float]:
    return [sum(getattr(r.steps[t], attr) for r in results) for t in range(len(results[0].steps))]


def window_stats(curve: list[float], steps: range) -> dict:
    kwh = sum(curve[t] for t in steps)
    return {"kwh": round(kwh, 3), "mean_kw": round(kwh * 2 / len(steps), 3)}


def distinct_trajectories(results: list[RunResult]) -> int:
    """Distinct indoor-temp trajectories (to 1e-6 C) across households."""
    return len({tuple(round(s.indoor_temp_c, 6) for s in r.steps) for r in results})


def agent_summary(runs: dict[str, list[RunResult]], cpp: range, post: range, seed: int,
                  archetype) -> dict:  # fmt: skip
    clocks = [s.clock for s in runs["base"][0].steps]
    first_post = range(post.start, post.start + 1)  # 20:00-20:30
    out: dict = {}
    for label in ("base", "cpp"):
        total = aggregate(runs[label])
        during, after = window_stats(total, cpp), window_stats(total, post)
        at_2000 = window_stats(total, first_post)
        out[f"{label}_day"] = {
            "aggregate": curve_metrics(total, clocks),
            "cpp_steps_34_39": during,
            "post_cpp_steps_40_43": after,
            "post_over_cpp_mean_kw_ratio": round(after["mean_kw"] / during["mean_kw"], 3),
            "step_40_2000_2030": at_2000,
            "step_40_over_cpp_mean_kw_ratio": round(at_2000["mean_kw"] / during["mean_kw"], 3),
            "distinct_hvac_trajectories": distinct_trajectories(runs[label]),
            "degree_hours_above_ceiling": round(
                sum(degree_hours_above_ceiling(r) for r in runs[label]), 3
            ),
            "max_exceedance_c": round(max(max_exceedance_c(r) for r in runs[label]), 3),
            "flexible_kwh": round(sum(aggregate(runs[label], "flexible_kwh")), 3),
            "washer_completed": sum(r.washer_completed for r in runs[label]),
            "comfort_violation_steps": sum(r.comfort_violation_steps for r in runs[label]),
        }
    b, c = out["base_day"], out["cpp_day"]
    out["post_cpp_rebound_kwh_cpp_minus_base"] = round(
        c["post_cpp_steps_40_43"]["kwh"] - b["post_cpp_steps_40_43"]["kwh"], 3
    )
    out["cpp_window_kwh_cpp_minus_base"] = round(
        c["cpp_steps_34_39"]["kwh"] - b["cpp_steps_34_39"]["kwh"], 3
    )
    recs = [
        household_elasticity(rb, rc, seed, archetype(rb.household_id))
        for rb, rc in zip(runs["base"], runs["cpp"])
    ]
    out["elasticity_flexible"] = {
        m: round(pooled_elasticity(recs, "flexible", m), 4) for m in ("base", "midpoint")
    }
    out["elasticity_total"] = {
        m: round(pooled_elasticity(recs, "total", m), 4) for m in ("base", "midpoint")
    }
    dflex = c["flexible_kwh"] - b["flexible_kwh"]
    out["daily_flexible_kwh_change"] = {
        "base": b["flexible_kwh"],
        "cpp": c["flexible_kwh"],
        "delta_kwh": round(dflex, 3),
        "delta_pct": round(100 * dflex / b["flexible_kwh"], 2) if b["flexible_kwh"] else None,
    }
    out["_records"] = recs
    return out


def plot(curves: dict, modes: list[str], cpp: range, post: range, path: Path) -> None:
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(len(modes), 2, figsize=(12, 1.0 + 3.25 * len(modes)),
                             sharex=True, sharey=True, squeeze=False)  # fmt: skip
    x = [t / 2 for t in range(48)]
    for i, mode in enumerate(modes):
        for j, label in enumerate(("base", "cpp")):
            ax = axes[i, j]
            ax.axvspan(cpp.start / 2, cpp.stop / 2, color="#d9d8d3", alpha=0.6, lw=0)
            ax.axvspan(post.start / 2, post.stop / 2, color="#efeeea", alpha=0.8, lw=0)
            for agent in AGENTS:
                kw = [v * 2 for v in curves[(mode, agent, label)]]
                ax.plot(x, kw, color=COLORS[agent], lw=2, label=LABELS[agent],
                        drawstyle="steps-post")  # fmt: skip
            day = "Baseline day" if label == "base" else "CPP day (17:00-20:00)"
            band = "persona bands" if mode == "persona" else f"fixed ±{FIXED_BAND_C} °C"
            ax.set_title(f"{day} · {band}", color=ink, fontsize=11, loc="left")
            ax.grid(axis="y", color=grid, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(muted)
            ax.tick_params(colors=muted, labelsize=9)
            ax.set_xticks(range(0, 25, 3))
            ax.set_xlim(0, 24)
            if j == 0:
                ax.set_ylabel("Aggregate load, 24 households (kW)", color=muted, fontsize=9)
            if i == len(modes) - 1:
                ax.set_xlabel("Hour of day", color=muted, fontsize=9)
    axes[0, 1].text(cpp.start / 2 + 0.1, 0.03, "CPP", transform=axes[0, 1].get_xaxis_transform(),
                    color=muted, fontsize=8, va="bottom")  # fmt: skip
    axes[0, 1].text(post.start / 2 + 0.1, 0.03, "post", transform=axes[0, 1].get_xaxis_transform(),
                    color=muted, fontsize=8, va="bottom")  # fmt: skip
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (1.0 + 3.25 * len(modes))))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def thermal_params_table(runner: Phase3Runner, hids: list[int]) -> pd.DataFrame:
    """The resolved per-household thermal params every agent in this run used."""
    rows = []
    for hid in hids:
        hvac = runner.env(hid)[1]
        rows.append({
            "household_id": hid,
            "comfort_orientation": runner.env(hid)[0].params.comfort_orientation,
            "heterogeneity_enabled": runner.cfg.thermal.enabled,
            **runner.cfg.thermal.draws(hid),
            "r": hvac.r,
            "c": hvac.c,
            "rc_hours": hvac.r * hvac.c,
            "setpoint_c": hvac.setpoint_c,
            "initial_indoor_temp_c": hvac.start_temp_c,
            "persona_band_c": hvac.tolerance_band_c,
        })  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=None, help="noisy-MILP seed (default fit_seed)")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--band-modes", default=",".join(BAND_MODES),
                        help="comma-separated subset of: " + ",".join(BAND_MODES))  # fmt: skip
    parser.add_argument("--no-thermal-heterogeneity", action="store_true",
                        help="override thermal_heterogeneity.enabled to false")  # fmt: skip
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_provenance(out)
    modes = args.band_modes.split(",")
    if unknown := set(modes) - set(BAND_MODES):
        parser.error(f"unknown band modes {sorted(unknown)}")

    # persona: this script applies each band mode itself on top of runner.env()
    cfg = Phase3Config.from_yaml(band_mode="persona")
    if args.no_thermal_heterogeneity:
        cfg = replace(cfg, thermal=replace(cfg.thermal, enabled=False))
    runner = Phase3Runner(cfg)
    seed = args.seed if args.seed is not None else runner.cfg.fit_seed
    params = RuleBasedParams.from_yaml(runner.cfg.experiment_path)
    cpp_lo, cpp_hi = runner.cfg.cpp_steps
    cpp, post = range(cpp_lo, cpp_hi), range(cpp_hi, cpp_hi + runner.cfg.rebound_steps)
    assert (cpp, post) == (range(34, 40), range(40, 44)), (cpp, post)
    hids = sorted(int(h) for h in runner.assignment["customer_ID"])

    t0 = time.time()
    summary: dict = {}
    curves: dict = {}
    rows = []
    for mode in modes:
        band_c = BAND_MODES[mode]
        summary[mode] = {}
        for agent in AGENTS:
            runs = {label: [simulate(runner, params, h, agent, band_c, label, seed) for h in hids]
                    for label in ("base", "cpp")}  # fmt: skip
            for label in runs:
                curves[(mode, agent, label)] = aggregate(runs[label])
            s = agent_summary(runs, cpp, post, seed, runner.archetype)
            for rec in s.pop("_records"):
                row = rec.to_row(runner.cfg.raw["elasticity_method"])
                rows.append({"band_mode": mode, **row})
            summary[mode][agent] = s
            print(f"[{mode:7s}] {agent:10s} peak cpp {s['cpp_day']['aggregate']} "
                  f"e_flex {s['elasticity_flexible']['base']}  "
                  f"dflex {s['daily_flexible_kwh_change']['delta_kwh']} kWh")  # fmt: skip

    result = {
        "date": runner.cfg.dates[0].isoformat(),
        "cpp": runner.cfg.raw["cpp"],
        "cpp_steps": [cpp.start, cpp.stop - 1],
        "post_cpp_steps": [post.start, post.stop - 1],
        "households": len(hids),
        "band_modes": {"persona": "own persona tolerance_band_c",
                       "fixed": f"+/-{FIXED_BAND_C} C for every household"},  # fmt: skip
        "noisy_milp_seed": seed,
        "thermal_heterogeneity": {
            **asdict(cfg.thermal),
            "per_household": "thermal_params.csv",
        },
        "rule_based": {
            "global_scale": GLOBAL_SCALE,
            "archetype_factor": FACTOR,
            "hvac_k": params.k_by_orientation,
            "washer_w_max": params.w_max,
            "fit": "experiments/fit_rule_based.py -> results/phase3/rule_based/fit.json",
        },
        "llm_calls": 0,
        "units": "aggregate = 24-household sum; kW from 30-min kWh x 2; elasticity pooled "
        "(sum kWh then ratio) over the CPP window",
        "wall_seconds": round(time.time() - t0, 1),
        "results": summary,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(rows).to_csv(out / "per_household.csv", index=False)
    pd.DataFrame(
        {f"{m}|{a}|{d}": [round(v * 2, 4) for v in c] for (m, a, d), c in curves.items()}
    ).rename_axis("step").to_csv(out / "aggregate_kw.csv")
    thermal_params_table(runner, hids).to_csv(out / "thermal_params.csv", index=False)
    plot(curves, modes, cpp, post, out / "aggregate_load.png")
    print(f"wrote {out}/summary.json, per_household.csv, aggregate_kw.csv, aggregate_load.png "
          f"({result['wall_seconds']}s)")  # fmt: skip


if __name__ == "__main__":
    main()
