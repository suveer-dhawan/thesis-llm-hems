"""Model reliability battery -- Matilda vs Gemini head-to-head.

NOT Phase 3 / elasticity work, and does not touch the MILP or noisy-MILP
agents. Companion to `experiments/phase2_diagnostics.py`, which found the
(then Matilda-only, pre-hardened-prompt) LLM agent misreading its own input
state in 3 of 4 controlled calls: a wrong comfort band, a wrong indoor temp,
and an indoor/outdoor temperature mix-up (see
results/phase2_diagnostics/diagnostics_summary.json).

Before any elasticity work, this script:
  1. Exercises the hardened prompt + state-echo schema (`llm/schemas.py`
     `HVACWTPResponse`/`WasherWTPResponse`, `agents/llm_agent.py`
     `_detect_misreads`) that makes misreads programmatically detectable.
  2. Runs the identical battery against two providers (`matilda`, `gemini`)
     so a primary model can be chosen before Phase 3 starts.

**Price-blind prompt (post pre-Phase-3 fix):** `agents/prompts/{hvac,
shiftable}.txt` no longer state the rate or a CPP flag at all, and the
state-echo schemas dropped `echo_is_cpp` accordingly (only the comfort-state
echoes -- indoor temp/band for HVAC, deadline/slack for the washer --
remain, see `llm/schemas.py`). `DecisionContext.rate`/`is_cpp` are still
built below and still drive `cost_now` (Python-side accounting only, per
BUILD_SPEC.md §1) -- they are simply never rendered into the prompt, and a
misread on either is no longer a thing the model can report since it's
never asked to echo them. Part A's price x CPP grid is kept (it still
exercises the halt rule's own `cost_now` sensitivity and gives a cheap
eyeball check that `wtp` itself doesn't drift with `rate`/`is_cpp` now that
neither reaches the prompt), but the misread bookkeeping below only checks
what the model can still get wrong: the comfort-state echoes.

Two parts, run once per provider:

  A. State-reading battery (18 calls): synthetic HVAC contexts spanning
     indoor temp (in-band / mildly over / far over the ceiling) x CPP flag
     (on/off) x price level (off-peak/peak/CPP-typical) -- price/CPP only
     affect `cost_now` and the halt rule now, never the prompt. Reports a
     misread rate (fraction of calls with >=1 comfort-echo field wrong) and
     a per-field breakdown.

  B. WTP price-invariance re-check (BUILD_SPEC.md §6.3: the LLM must never
     do the cost comparison itself): fixed HVAC + washer comfort state,
     four prices, per provider (8 calls) -- same design as
     `phase2_diagnostics.price_invariance_test`, run per-provider here.

18 + 8 = 26 calls per provider, 52 total -- cheap, per the task's "tens of
calls per model" budget.

Both providers go through the same config path as the production pipeline
(`run_single.load_llm_config` / `build_llm_client`, reading
`config/experiment.yaml`'s `llm:` block) rather than constructing an
`LLMClient` with bare defaults, so a live (non `--dry-run`) `gemini` call
here gets the pinned `temperature: 0.0` / `seed: 42` the same way
`run_single.py --agent llm` does.

Usage:
  PYTHONPATH=src python experiments/reliability_battery.py
  PYTHONPATH=src python experiments/reliability_battery.py --providers matilda
  PYTHONPATH=src python experiments/reliability_battery.py --dry-run   # pipeline check, no spend
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from run_single import build_llm_client, load_environment, load_llm_config

from hems.agents.base import DecisionContext
from hems.agents.llm_agent import LLMAgent
from hems.provenance import write_provenance

RESULTS_DIR = Path("results/reliability_battery")

# -- Part A: state-reading battery -------------------------------------------

# (label, offset from band ceiling in C) -- "in_band" sits at the setpoint,
# safely inside; the other two are relative to the ceiling so they scale
# with whatever tolerance_band_c is configured.
INDOOR_LEVELS = [
    ("in_band", None),      # resolved to the setpoint (middle of the band)
    ("mild_over", 1.0),     # 1C over the ceiling
    ("far_over", 4.0),      # 4C over the ceiling
]
CPP_LEVELS = [False, True]
# off-peak / peak / CPP-typical -- deliberately independent of the is_cpp
# flag above (see module docstring).
PRICE_LEVELS = [0.22, 0.55, 1.20]

# -- Part B: price-invariance re-check (mirrors phase2_diagnostics.py) ------

INVARIANCE_PRICES = [0.22, 0.55, 1.20, 2.50]

ECHO_FIELDS_HVAC = ["indoor_temp_c", "band_lo_c", "band_hi_c"]
ECHO_FIELDS_WASHER = ["steps_until_deadline", "slack_steps"]


def household_ctx(household) -> dict:
    return {
        "thermal_sensitivity": household.params.thermal_sensitivity,
        "price_sensitivity": household.params.price_sensitivity,
    }


def build_llm_agent(dry_run: bool, cache_path, provider: str | None = None) -> LLMAgent:
    """Same provider/config resolution as `run_single.build_agent`'s `llm`
    branch -- reads `config/experiment.yaml`'s `llm:` block (provider,
    model, temperature, seed, ...) rather than constructing `LLMClient`
    with bare defaults, so a live call here is configured identically to
    `run_single.py --agent llm`."""
    if dry_run:
        return LLMAgent(dry_run=True)
    llm_cfg = load_llm_config()
    resolved_provider = provider or llm_cfg.get("provider", "matilda")
    client = build_llm_client(resolved_provider, llm_cfg, cache_path)
    return LLMAgent(client=client)


def build_hvac_context(hvac_params, hh_ctx, indoor_temp_c, rate, is_cpp, t=30, clock="15:00"):
    return DecisionContext(
        t=t,
        clock=clock,
        day_index=0,
        appliance={
            "type": "hvac",
            "indoor_temp_c": indoor_temp_c,
            "outdoor_temp_c": 33.0,
            "setpoint_c": hvac_params.setpoint_c,
            "tolerance_band_c": hvac_params.tolerance_band_c,
            "rated_kw": hvac_params.rated_kw,
        },
        cost_now=hvac_params.rated_kw * hvac_params.dt_hours * rate,
        rate=rate,
        is_cpp=is_cpp,
        household=hh_ctx,
    )


def build_washer_context(washer_params, hh_ctx, rate, is_cpp, t=10, clock="05:00", deadline_step=20):
    steps_completed = 0
    steps_remaining = washer_params.duration_steps - steps_completed
    steps_until_deadline = deadline_step - t
    # Python-computed, not derived by the LLM (BUILD_SPEC.md §8, the washer
    # slack fix) -- mirrors `environment/simulator.py`'s washer_ctx.
    slack_steps = steps_until_deadline - steps_remaining
    return DecisionContext(
        t=t,
        clock=clock,
        day_index=0,
        appliance={
            "type": "washer",
            "steps_completed": steps_completed,
            "duration_steps": washer_params.duration_steps,
            "released_step": washer_params.released_step,
            "deadline_step": deadline_step,
            "slack_steps": slack_steps,
            "interruptible": False,
            "rated_kw": washer_params.power_kw,
        },
        cost_now=washer_params.power_kw * washer_params.dt_hours * rate,
        rate=rate,
        is_cpp=is_cpp,
        household=hh_ctx,
    )


def run_state_battery(provider: str, hvac_params, hh_ctx, dry_run: bool) -> tuple[list[dict], LLMAgent]:
    cache_path = None if dry_run else RESULTS_DIR / f"{provider}_state_battery_cache.json"
    agent = build_llm_agent(dry_run, cache_path, provider=provider)

    rows = []
    for (label, offset), is_cpp, rate in itertools.product(INDOOR_LEVELS, CPP_LEVELS, PRICE_LEVELS):
        indoor = hvac_params.setpoint_c if offset is None else hvac_params.setpoint_c + hvac_params.tolerance_band_c + offset
        ctx = build_hvac_context(hvac_params, hh_ctx, indoor, rate, is_cpp)
        agent.decide(ctx)
        entry = agent.log[-1]
        rows.append(
            {
                "provider": provider,
                "indoor_level": label,
                "indoor_temp_c": indoor,
                "is_cpp": is_cpp,
                "rate": rate,
                "wtp": entry.wtp,
                "run": entry.run,
                "misreads": entry.misreads,
                "misread": len(entry.misreads) > 0,
            }
        )
        print(f"[state_battery:{provider}] {label} cpp={is_cpp} rate={rate} -> wtp={entry.wtp} misreads={entry.misreads}")
    return rows, agent


def summarize_state_battery(rows: list[dict]) -> dict:
    n = len(rows)
    misread_rate = round(sum(r["misread"] for r in rows) / n, 4) if n else None
    field_rates = {}
    for field in ECHO_FIELDS_HVAC:
        count = sum(1 for r in rows if field in r["misreads"])
        field_rates[field] = round(count / n, 4) if n else None
    return {"n_calls": n, "misread_rate": misread_rate, "field_misread_rates": field_rates}


# -- Part B ------------------------------------------------------------------


def run_price_invariance(provider: str, hvac_params, washer_params, hh_ctx, dry_run: bool) -> dict:
    cache_path = None if dry_run else RESULTS_DIR / f"{provider}_price_invariance_cache.json"
    agent = build_llm_agent(dry_run, cache_path, provider=provider)

    hvac_rows, washer_rows = [], []
    indoor = hvac_params.band_hi_c + 1.0  # 1C over the ceiling, fixed across prices
    for rate in INVARIANCE_PRICES:
        ctx = build_hvac_context(hvac_params, hh_ctx, indoor, rate, is_cpp=False)
        agent.decide(ctx)
        entry = agent.log[-1]
        hvac_rows.append({"rate": rate, "wtp": entry.wtp, "misreads": entry.misreads})

    for rate in INVARIANCE_PRICES:
        ctx = build_washer_context(washer_params, hh_ctx, rate, is_cpp=False)
        agent.decide(ctx)
        entry = agent.log[-1]
        washer_rows.append({"rate": rate, "wtp": entry.wtp, "misreads": entry.misreads})

    hvac_wtps = [r["wtp"] for r in hvac_rows]
    washer_wtps = [r["wtp"] for r in washer_rows]
    return {
        "hvac": hvac_rows,
        "washer": washer_rows,
        "hvac_wtp_range": round(max(hvac_wtps) - min(hvac_wtps), 4),
        "washer_wtp_range": round(max(washer_wtps) - min(washer_wtps), 4),
    }


# -- main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--providers", nargs="+", default=["matilda", "gemini"], choices=["matilda", "gemini"])
    parser.add_argument("--dry-run", action="store_true", help="pipeline check, no API spend")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_provenance(RESULTS_DIR)
    household, _tariff, hvac_params, washer_params = load_environment()
    hh_ctx = household_ctx(household)

    battery_raw: dict[str, list[dict]] = {}
    battery_summary: dict[str, dict] = {}
    invariance: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    overall: dict[str, dict] = {}

    for provider in args.providers:
        print(f"\n=== state-reading battery: {provider} ===")
        rows, agent = run_state_battery(provider, hvac_params, hh_ctx, args.dry_run)
        battery_raw[provider] = rows
        battery_summary[provider] = summarize_state_battery(rows)
        meta[provider] = {
            "provider": agent.provider,
            "model_id": agent.model_id,
            "sdk_version": agent.sdk_version,
        }
        print(json.dumps(battery_summary[provider], indent=2))

        print(f"\n=== price-invariance re-check: {provider} ===")
        invariance[provider] = run_price_invariance(provider, hvac_params, washer_params, hh_ctx, args.dry_run)
        print(
            f"[price_invariance:{provider}] hvac_wtp_range={invariance[provider]['hvac_wtp_range']} "
            f"washer_wtp_range={invariance[provider]['washer_wtp_range']}"
        )

        # Combined misread rate across all calls in the battery (state-reading
        # + price-invariance HVAC + price-invariance washer) -- the state
        # battery alone is HVAC-only; folding in the invariance calls gives
        # one headline number over every call made to this provider.
        all_calls = (
            [{"misreads": r["misreads"]} for r in rows]
            + invariance[provider]["hvac"]
            + invariance[provider]["washer"]
        )
        n_total = len(all_calls)
        n_misread = sum(1 for c in all_calls if c["misreads"])
        overall[provider] = {
            "n_calls": n_total,
            "misread_rate": round(n_misread / n_total, 4) if n_total else None,
        }

    output = {
        "meta": meta,
        "overall_misread_rate": overall,
        "state_battery_summary": battery_summary,
        "state_battery_raw": battery_raw,
        "price_invariance": invariance,
    }
    (RESULTS_DIR / "battery_results.json").write_text(json.dumps(output, indent=2))
    print(f"\n[reliability_battery] wrote {RESULTS_DIR / 'battery_results.json'}")


if __name__ == "__main__":
    main()
