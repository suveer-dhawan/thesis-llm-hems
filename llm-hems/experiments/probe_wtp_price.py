"""Diagnostic probe: does LLM WTP actually respond to price? (standalone, not production)

NOT part of the thesis pipeline. Does not touch `config/experiment.yaml`,
`agents/llm_agent.py`, `agents/prompts/*.txt`, or `llm/client.py` -- it only
*imports* them read-only (reusing the exact production prompt-building code
and the real appliance/tariff params from `config/`) to answer one question
before committing to a price-in-prompt vs. price-out-of-prompt ablation:
holding comfort state fixed, does the model's stated willingness-to-pay move
with the price it's shown, is that movement stable across repeated calls,
and does it differ by household persona?

Design (see module-level constants below for the exact values used):

- **Price grid**: the five rates actually configured in `config/tariff.yaml`
  (off-peak / shoulder / peak / CPP-moderate / CPP-severe), not an arbitrary
  grid -- this is what the agent will actually see in the simulator.
  `is_cpp` is set to match how each rate actually occurs in the tariff (the
  two CPP rates always co-occur with an active CPP flag in production;
  never invented independently here).
- **Fixed comfort state per appliance**: one representative state each for
  HVAC (indoor 1C above setpoint, inside the band) and the washer (not
  started, moderate deadline slack) -- held identical across every price/
  persona/seed cell, so price is the only thing that moves.
- **3 personas**: injected through the same `$thermal_sensitivity` /
  `$price_sensitivity` template placeholders the production prompt already
  exposes (`agents/prompts/hvac.txt`, `shiftable.txt`) -- that mechanism
  *is* wired in, just only ever populated from one household in
  `config/household.yaml` today. The probe supplies 3 descriptor sets
  through the same slots instead of adding new prompt text.
- **4 live seeds per cell**: `LLMClient.complete_json` caches on
  `(provider, prompt)`, which would make identical "seed" calls to an
  identical prompt collapse onto one cached response and hide any real
  sampling spread. This probe instead calls `_GeminiBackend.call(...)`
  directly (bypassing the cache entirely) while still routing through the
  *same* backend instance so its real rate-limit throttle (5 req/min free
  tier) is respected across all 120 calls.

Usage (from `llm-hems/`):
  PYTHONPATH=src python experiments/probe_wtp_price.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 -- unbuffered so background runs are monitorable

import numpy as np
from run_single import load_environment

from hems.agents.base import DecisionContext
from hems.agents.llm_agent import LLMAgent
from hems.llm.client import LLMClient
from hems.llm.schemas import HVACWTPResponse, WasherWTPResponse

RESULTS_DIR = Path("results")

# -- committed primary model, explicit (not experiment.yaml's default, which
# is still `matilda`) --------------------------------------------------------
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_RATE_LIMIT_RPM = 5
GEMINI_MAX_OUTPUT_TOKENS = 2048

# -- price grid: the real configured tariff.yaml rates -----------------------
# (label, rate $/kWh, is_cpp) -- is_cpp mirrors how each rate actually occurs
# in config/tariff.yaml: peak/shoulder/off_peak never co-occur with a CPP
# flag; the two `cpp.rates` levels always do.
PRICE_POINTS = [
    ("off_peak", 0.22, False),
    ("shoulder", 0.28, False),
    ("peak", 0.55, False),
    ("cpp_moderate", 1.20, True),
    ("cpp_severe", 2.50, True),
]

N_SEEDS = 4

# -- 3 personas, injected through the existing $thermal_sensitivity /
# $price_sensitivity template placeholders (hvac.txt / shiftable.txt) --------
PERSONAS = {
    "price_conscious": {
        "thermal_sensitivity": (
            "loose -- tolerates the room drifting away from the setpoint "
            "without much discomfort"
        ),
        "price_sensitivity": (
            "very price-conscious -- dislikes spending money on electricity and "
            "strongly prefers to wait for a cheaper interval, even if that means "
            "real discomfort or delay"
        ),
    },
    "comfort_first": {
        "thermal_sensitivity": (
            "tight -- strongly dislikes any deviation from the setpoint and wants "
            "the room comfortable at all times"
        ),
        "price_sensitivity": (
            "affluent and comfort-first -- money is not a real constraint for this "
            "household, it prioritises comfort and convenience over cost"
        ),
    },
    "neutral": {
        # identical to config/household.yaml's current placeholder values --
        # the baseline arm, not a new descriptor.
        "thermal_sensitivity": "moderate",
        "price_sensitivity": "moderate",
    },
}

# -- fixed comfort state per appliance (identical across every cell) ---------
# HVAC: 1C above setpoint, inside the +/-2C band (22.0-26.0C) -- some mild
# comfort pull, but not a band violation.
HVAC_T = 30
HVAC_CLOCK = "15:00"
HVAC_OUTDOOR_TEMP_C = 33.0

# Washer: not started, moderate deadline slack (2 of 4 steps' worth of
# slack = 1h beyond what's needed) -- neither indefinite (no urgency) nor
# zero (mandatory regardless of price).
WASHER_T = 10
WASHER_CLOCK = "05:00"
WASHER_DEADLINE_STEP = 16  # steps_until_deadline = 16-10 = 6; duration_steps = 4 -> slack = 2

# Verdict thresholds (heuristic, stated explicitly since the task specifies
# buckets but not numeric cutoffs):
#   - "reversal": mean WTP increases from one price point to the next
#     (allowing a $0.02 rounding/noise tolerance).
#   - "flat": total range of mean WTP across the 5 price points < $0.20.
#   - "stable": mean coefficient of variation (std/mean) across the 5 price
#     points' seed-groups < 0.30 (undefined/skipped when a cell's mean is 0).
#   - "erratic": mean CV > 0.50, or reversals with no consistent direction.
REVERSAL_TOLERANCE = 0.02
FLAT_RANGE_THRESHOLD = 0.20
STABLE_CV_THRESHOLD = 0.30
ERRATIC_CV_THRESHOLD = 0.50


def build_hvac_ctx(hvac_params, persona: dict, rate: float, is_cpp: bool) -> DecisionContext:
    indoor = hvac_params.setpoint_c + 1.0
    return DecisionContext(
        t=HVAC_T,
        clock=HVAC_CLOCK,
        day_index=0,
        appliance={
            "type": "hvac",
            "indoor_temp_c": indoor,
            "outdoor_temp_c": HVAC_OUTDOOR_TEMP_C,
            "setpoint_c": hvac_params.setpoint_c,
            "tolerance_band_c": hvac_params.tolerance_band_c,
            "rated_kw": hvac_params.rated_kw,
        },
        cost_now=hvac_params.rated_kw * hvac_params.dt_hours * rate,
        rate=rate,
        is_cpp=is_cpp,
        household=persona,
    )


def build_washer_ctx(washer_params, persona: dict, rate: float, is_cpp: bool) -> DecisionContext:
    return DecisionContext(
        t=WASHER_T,
        clock=WASHER_CLOCK,
        day_index=0,
        appliance={
            "type": "washer",
            "steps_completed": 0,
            "duration_steps": washer_params.duration_steps,
            "released_step": washer_params.released_step,
            "deadline_step": WASHER_DEADLINE_STEP,
            "interruptible": False,
            "rated_kw": washer_params.power_kw,
        },
        cost_now=washer_params.power_kw * washer_params.dt_hours * rate,
        rate=rate,
        is_cpp=is_cpp,
        household=persona,
    )


def call_live(backend, prompt: str, schema, max_retries: int = 3):
    """One genuinely-live call: bypasses `LLMClient.complete_json`'s cache
    entirely (so repeated "seed" calls to the same prompt are independent
    live samples, not a cache hit) while still going through the shared
    backend instance's real rate-limit throttle. Returns
    `(parsed_response, latency_s, was_rate_limited)`.
    """
    last_exc: Exception | None = None
    rate_limited = False
    for attempt in range(max_retries + 1):
        try:
            t0 = time.monotonic()
            raw = backend.call(prompt, schema)
            latency = time.monotonic() - t0
            return schema.model_validate(raw), latency, rate_limited
        except Exception as exc:  # noqa: BLE001 -- provider raises opaque SDK error types
            last_exc = exc
            is_rl = "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc)
            rate_limited = rate_limited or is_rl
            if attempt < max_retries:
                sleep_s = max(20, 2**attempt) if is_rl else min(2**attempt, 8)
                print(f"    [retry {attempt + 1}/{max_retries}] {exc!r} -- sleeping {sleep_s}s")
                time.sleep(sleep_s)
    raise RuntimeError(f"probe call failed after {max_retries + 1} attempts: {last_exc}") from last_exc


def linear_slope(rates: list[float], wtps: list[float]) -> float:
    """Rough linear fit slope (d(wtp)/d(rate)) over raw (not averaged) points."""
    if len(set(rates)) < 2:
        return float("nan")
    slope, _intercept = np.polyfit(rates, wtps, 1)
    return float(slope)


def summarize_cell(wtps: list[float]) -> dict:
    mean = float(np.mean(wtps))
    std = float(np.std(wtps, ddof=1)) if len(wtps) > 1 else 0.0
    cv = std / mean if mean > 1e-9 else float("nan")
    return {"mean": mean, "std": std, "cv": cv}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 persona x 2 prices x 1 seed x both appliances (4 calls) to verify connectivity/logic only",
    )
    args = parser.parse_args()

    personas = PERSONAS
    price_points = PRICE_POINTS
    n_seeds = N_SEEDS
    if args.smoke_test:
        personas = {"neutral": PERSONAS["neutral"]}
        price_points = [PRICE_POINTS[0], PRICE_POINTS[-1]]
        n_seeds = 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    household, tariff, hvac_params, washer_params = load_environment()

    client = LLMClient(
        provider="gemini",
        model=GEMINI_MODEL,
        rate_limit_rpm=GEMINI_RATE_LIMIT_RPM,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        cache_path=None,  # no persistence, no in-memory reuse across our own calls
    )
    agent = LLMAgent(client=client)  # reused only for its production prompt-builder methods
    backend = client._backend  # noqa: SLF001 -- read-only, to bypass complete_json's cache

    print(f"[probe_wtp_price] provider=gemini model_id={agent.model_id} sdk_version={agent.sdk_version}")
    print(f"[probe_wtp_price] rate_limit_rpm={GEMINI_RATE_LIMIT_RPM} n_seeds={N_SEEDS}")
    print(
        f"[probe_wtp_price] HVAC fixed state: indoor={hvac_params.setpoint_c + 1.0:.1f}C "
        f"band=[{hvac_params.band_lo_c:.1f},{hvac_params.band_hi_c:.1f}]C clock={HVAC_CLOCK}"
    )
    print(
        f"[probe_wtp_price] washer fixed state: steps_completed=0/{washer_params.duration_steps} "
        f"deadline_step={WASHER_DEADLINE_STEP} t={WASHER_T} clock={WASHER_CLOCK} "
        f"(steps_until_deadline={WASHER_DEADLINE_STEP - WASHER_T}, "
        f"slack={WASHER_DEADLINE_STEP - WASHER_T - washer_params.duration_steps} steps)"
    )

    rows: list[dict] = []
    n_calls = 0
    n_rate_limited_cells = 0
    t_start = time.monotonic()

    appliance_specs = [
        ("hvac", HVACWTPResponse, build_hvac_ctx, hvac_params),
        ("washer", WasherWTPResponse, build_washer_ctx, washer_params),
    ]

    fieldnames = [
        "appliance", "persona", "price_label", "rate", "is_cpp", "seed",
        "wtp", "reason", "cost_now", "utility", "run_decision", "latency_s",
    ]
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{timestamp}_probe_wtp_price.csv"
    csv_file = out_path.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for appliance_name, schema, ctx_builder, params in appliance_specs:
        for persona_name, persona in personas.items():
            for price_label, rate, is_cpp in price_points:
                ctx = ctx_builder(params, persona, rate, is_cpp)
                prompt = (
                    agent._build_hvac_prompt(ctx)  # noqa: SLF001
                    if appliance_name == "hvac"
                    else agent._build_washer_prompt(ctx)  # noqa: SLF001
                )
                for seed in range(n_seeds):
                    try:
                        resp, latency, rate_limited = call_live(backend, prompt, schema)
                    except RuntimeError as exc:
                        print(f"  FAILED {appliance_name}/{persona_name}/{price_label}/seed{seed}: {exc}")
                        continue
                    n_calls += 1
                    if rate_limited:
                        n_rate_limited_cells += 1
                    row = {
                        "appliance": appliance_name,
                        "persona": persona_name,
                        "price_label": price_label,
                        "rate": rate,
                        "is_cpp": is_cpp,
                        "seed": seed,
                        "wtp": resp.wtp,
                        "reason": resp.reason,
                        "cost_now": round(ctx.cost_now, 4),
                        "utility": round(resp.wtp - ctx.cost_now, 4),
                        "run_decision": (resp.wtp - ctx.cost_now) >= 0,
                        "latency_s": round(latency, 3),
                    }
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                    print(
                        f"  [{appliance_name}/{persona_name}/{price_label} rate={rate} "
                        f"seed={seed}] wtp={resp.wtp:.3f} ({latency:.2f}s)"
                    )

    csv_file.close()
    wall_clock_s = time.monotonic() - t_start

    # -- analysis -------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"total_calls={n_calls} wall_clock_s={wall_clock_s:.1f} model_id={agent.model_id} "
          f"temperature=<not set by client.py -- provider default> "
          f"rate_limited_calls={n_rate_limited_cells}")
    print(f"raw results written to {out_path}")

    for appliance_name, _schema, _builder, _params in appliance_specs:
        print(f"\n## {appliance_name.upper()}\n")
        for persona_name in personas:
            cell_rows = [r for r in rows if r["appliance"] == appliance_name and r["persona"] == persona_name]
            if not cell_rows:
                continue
            print(f"### {appliance_name} x {persona_name}\n")
            print("| price | rate ($/kWh) | mean WTP | std WTP | CV |")
            print("|---|---|---|---|---|")
            means = []
            all_rates, all_wtps = [], []
            for price_label, rate, _is_cpp in price_points:
                cell = [r["wtp"] for r in cell_rows if r["price_label"] == price_label]
                if not cell:
                    continue
                stats = summarize_cell(cell)
                means.append((rate, stats["mean"]))
                all_rates.extend([rate] * len(cell))
                all_wtps.extend(cell)
                cv_str = f"{stats['cv']:.2f}" if stats["cv"] == stats["cv"] else "n/a"
                print(f"| {price_label} | {rate} | {stats['mean']:.3f} | {stats['std']:.3f} | {cv_str} |")

            slope = linear_slope(all_rates, all_wtps)
            wtp_range = max(m for _, m in means) - min(m for _, m in means) if means else float("nan")
            reversals = [
                (means[i][0], means[i + 1][0])
                for i in range(len(means) - 1)
                if means[i + 1][1] > means[i][1] + REVERSAL_TOLERANCE
            ]
            cvs = [summarize_cell([r["wtp"] for r in cell_rows if r["price_label"] == pl])["cv"]
                   for pl, _, _ in price_points
                   if [r["wtp"] for r in cell_rows if r["price_label"] == pl]]
            cvs = [c for c in cvs if c == c]  # drop nan
            mean_cv = float(np.mean(cvs)) if cvs else float("nan")

            if wtp_range < FLAT_RANGE_THRESHOLD:
                verdict = "(B) essentially flat"
            elif mean_cv == mean_cv and mean_cv > ERRATIC_CV_THRESHOLD:
                verdict = "(C) erratic/unstable"
            elif not reversals and mean_cv == mean_cv and mean_cv < STABLE_CV_THRESHOLD and slope <= 0:
                verdict = "(A) monotone + stable"
            else:
                verdict = "(C) erratic/unstable"

            print(f"\nslope (d(WTP)/d($1/kWh)): {slope:.4f}")
            print(f"WTP range across price grid: ${wtp_range:.3f}")
            print(f"mean CV across price points: {mean_cv:.3f}" if mean_cv == mean_cv else "mean CV: n/a")
            print(f"reversals (price up, mean WTP up by >${REVERSAL_TOLERANCE}): {reversals if reversals else 'none'}")
            print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
