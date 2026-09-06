"""Phase 2 diagnostics -- a quick, cheap sanity pass before building the
Phase 3 elasticity pipeline. Not part of the core agent/simulator pipeline;
re-runnable but one-off in purpose. Three checks:

1. Comfort-violation profile for the completed household 8596070 /
   2013-01-18 LLM run: reads the already-written decision log (no new API
   calls), reports out-of-band step count, max excess beyond the band, and
   the longest continuous excursion, and plots indoor temp vs. the band.
2. WTP price-invariance test: holds one fixed HVAC comfort state and one
   fixed washer slack state, and queries the LLM at four prices with
   everything else identical, to check whether `wtp` moves with price (it
   shouldn't -- BUILD_SPEC.md §6.3, the LLM must never do the cost
   comparison itself). 8 calls total.
3. Run-to-run variance check: since the Matilda API exposes no seed or
   temperature (see `README.md` / `config/experiment.yaml`), re-runs the
   full day 3x with caching disabled and reports spread in total cost,
   flexible share, comfort-violation count, and two individual WTP values
   at a state that is identical by construction across reps (t=0, before
   any agent decision has had a chance to diverge). 3 reps x 96 calls =
   288 calls -- the dominant cost of this script; done at the low end of
   the requested 3-5 reps to keep spend down.

Usage: `PYTHONPATH=src python experiments/phase2_diagnostics.py [--skip-variance]`
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from run_single import load_environment, run_agent

from hems.agents.base import DecisionContext
from hems.agents.llm_agent import LLMAgent

RESULTS_DIR = Path("results/phase2_diagnostics")
LLM_RUN_DIR = Path("results/phase2")
DATE = dt.date(2013, 1, 18)


# ---------------------------------------------------------------------------
# 1. Comfort-violation profile
# ---------------------------------------------------------------------------


def comfort_violation_profile(hvac_params) -> dict:
    df = pd.read_parquet(LLM_RUN_DIR / "llm_decision_log.parquet")
    band_lo, band_hi = hvac_params.band_lo_c, hvac_params.band_hi_c

    viol = df[~df["in_band"]].copy()
    viol["excess_c"] = viol["indoor_temp_c"].apply(lambda t: max(t - band_hi, band_lo - t, 0.0))

    runs, cur = [], 0
    for in_band in df["in_band"]:
        if not in_band:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)

    max_excess = round(viol["excess_c"].max(), 2) if len(viol) else 0.0
    longest_run_steps = max(runs) if runs else 0
    severity = "mild" if max_excess <= 2.0 else ("moderate" if max_excess <= 4.0 else "severe")

    profile = {
        "out_of_band_steps": int(len(viol)),
        "total_steps": int(len(df)),
        "max_excess_c": max_excess,
        "worst_step": (
            None
            if not len(viol)
            else {
                "t": int(viol.loc[viol["excess_c"].idxmax(), "t"]),
                "clock": viol.loc[viol["excess_c"].idxmax(), "clock"],
                "indoor_temp_c": round(
                    float(viol.loc[viol["excess_c"].idxmax(), "indoor_temp_c"]), 2
                ),
            }
        ),
        "longest_excursion_steps": int(longest_run_steps),
        "longest_excursion_minutes": int(longest_run_steps * 30),
        "excursion_run_lengths_steps": runs,
        "severity": severity,
        "band_lo_c": band_lo,
        "band_hi_c": band_hi,
    }

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["clock"], df["indoor_temp_c"], color="#4C72B0", label="indoor temp")
    band_label = f"comfort band [{band_lo}, {band_hi}]"
    ax.axhspan(band_lo, band_hi, color="#55A868", alpha=0.15, label=band_label)
    ax.axhline(band_lo, color="#55A868", linestyle="--", linewidth=1)
    ax.axhline(band_hi, color="#55A868", linestyle="--", linewidth=1)
    out_mask = ~df["in_band"]
    ax.scatter(
        df.loc[out_mask, "clock"],
        df.loc[out_mask, "indoor_temp_c"],
        color="#C44E52",
        zorder=5,
        label="out of band",
    )
    ax.set_xticks(df["clock"][::4])
    ax.set_xticklabels(df["clock"][::4], rotation=45, ha="right")
    ax.set_ylabel("indoor temp (C)")
    ax.set_xlabel("time of day")
    ax.set_title("LLM agent: indoor temperature vs. comfort band -- household 8596070, 2013-01-18")
    ax.legend()
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "comfort_profile.png", dpi=150)
    plt.close(fig)

    return profile


# ---------------------------------------------------------------------------
# 2. WTP price-invariance test
# ---------------------------------------------------------------------------

PRICES = [0.22, 0.55, 1.20, 2.50]


def price_invariance_test(hvac_params, washer_params, household) -> dict:
    agent = LLMAgent(cache_path=RESULTS_DIR / "price_invariance_cache.json")
    household_ctx = {
        "thermal_sensitivity": household.params.thermal_sensitivity,
        "price_sensitivity": household.params.price_sensitivity,
    }

    hvac_rows = []
    indoor = hvac_params.band_hi_c + 1.0  # 1C over the band ceiling, fixed across prices
    for rate in PRICES:
        ctx = DecisionContext(
            t=30,
            clock="15:00",
            day_index=0,
            appliance={
                "type": "hvac",
                "indoor_temp_c": indoor,
                "outdoor_temp_c": 32.0,
                "setpoint_c": hvac_params.setpoint_c,
                "tolerance_band_c": hvac_params.tolerance_band_c,
                "rated_kw": hvac_params.rated_kw,
            },
            cost_now=hvac_params.rated_kw * hvac_params.dt_hours * rate,
            rate=rate,
            is_cpp=False,  # held fixed so only `rate` varies (see report note)
            household=household_ctx,
        )
        agent.decide(ctx)
        entry = agent.log[-1]
        hvac_rows.append(
            {
                "rate": rate,
                "wtp": entry.wtp,
                "cost_now": entry.cost_now,
                "run": entry.run,
                "reason": entry.reason,
            }
        )

    washer_rows = []
    # fixed slack: not started, 4-step cycle, 10 steps until deadline -> 6 steps slack
    for rate in PRICES:
        ctx = DecisionContext(
            t=10,
            clock="05:00",
            day_index=0,
            appliance={
                "type": "washer",
                "steps_completed": 0,
                "duration_steps": washer_params.duration_steps,
                "released_step": washer_params.released_step,
                "deadline_step": 20,
                "interruptible": False,
                "rated_kw": washer_params.power_kw,
            },
            cost_now=washer_params.power_kw * washer_params.dt_hours * rate,
            rate=rate,
            is_cpp=False,
            household=household_ctx,
        )
        agent.decide(ctx)
        entry = agent.log[-1]
        washer_rows.append(
            {
                "rate": rate,
                "wtp": entry.wtp,
                "cost_now": entry.cost_now,
                "run": entry.run,
                "reason": entry.reason,
            }
        )

    hvac_wtps = [r["wtp"] for r in hvac_rows]
    washer_wtps = [r["wtp"] for r in washer_rows]
    return {
        "hvac": hvac_rows,
        "washer": washer_rows,
        "hvac_wtp_range": round(max(hvac_wtps) - min(hvac_wtps), 4),
        "washer_wtp_range": round(max(washer_wtps) - min(washer_wtps), 4),
    }


# ---------------------------------------------------------------------------
# 3. Run-to-run variance check
# ---------------------------------------------------------------------------


def variance_check(household, tariff, hvac_params, washer_params, n_reps: int) -> dict:
    reps = []
    for i in range(n_reps):
        result, agent = run_agent(
            "llm", household, tariff, hvac_params, washer_params, DATE, seed=42,
            dry_run=False, cache_path=None,
        )
        t0_hvac = next(e for e in agent.log if e.t == 0 and e.appliance == "hvac")
        t0_washer = next(e for e in agent.log if e.t == 0 and e.appliance == "washer")
        reps.append(
            {
                "rep": i,
                "total_cost": round(result.total_cost, 4),
                "flexible_share_pct": round(result.flexible_share * 100, 2),
                "comfort_violation_steps": result.comfort_violation_steps,
                "t0_hvac_wtp": t0_hvac.wtp,
                "t0_washer_wtp": t0_washer.wtp,
            }
        )
        print(f"[variance_check] rep {i}: {reps[-1]}")

    def spread(key):
        vals = [r[key] for r in reps]
        return {
            "values": vals,
            "min": min(vals),
            "max": max(vals),
            "range": round(max(vals) - min(vals), 4),
        }

    return {
        "reps": reps,
        "total_cost": spread("total_cost"),
        "flexible_share_pct": spread("flexible_share_pct"),
        "comfort_violation_steps": spread("comfort_violation_steps"),
        "t0_hvac_wtp": spread("t0_hvac_wtp"),
        "t0_washer_wtp": spread("t0_washer_wtp"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-variance", action="store_true", help="skip the 3x full-day variance check"
    )
    parser.add_argument(
        "--n-reps", type=int, default=3, help="number of full-day reps for the variance check"
    )
    args = parser.parse_args()

    household, tariff, hvac_params, washer_params = load_environment()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. comfort-violation profile ===")
    profile = comfort_violation_profile(hvac_params)
    print(json.dumps(profile, indent=2))

    print("\n=== 2. WTP price-invariance test ===")
    invariance = price_invariance_test(hvac_params, washer_params, household)
    print(json.dumps(invariance, indent=2))

    output = {"comfort_violation_profile": profile, "price_invariance": invariance}

    if not args.skip_variance:
        print(f"\n=== 3. run-to-run variance check ({args.n_reps} full-day reps) ===")
        variance = variance_check(household, tariff, hvac_params, washer_params, args.n_reps)
        print(json.dumps(variance, indent=2))
        output["variance_check"] = variance

    (RESULTS_DIR / "diagnostics_summary.json").write_text(json.dumps(output, indent=2))
    print(f"\n[phase2_diagnostics] wrote {RESULTS_DIR / 'diagnostics_summary.json'}")


if __name__ == "__main__":
    main()
