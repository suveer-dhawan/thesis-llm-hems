"""Final validation pass: temperature=0 + a genuinely contested comfort state.

v1/v2 (probe_wtp_price.py, probe_wtp_price_v2.py) found WTP essentially flat
or erratic in response to price. Two confounds were live throughout both:

  1. Default temperature (1.0, confirmed via `client.models.get(...)`) makes
     the WTP *number* sampling noise, decoupled from the model's own (fairly
     sensible) reasoning text -- seen directly in v2 where identical prompts
     produced wtp in {0.0, 0.5, 1.0} across seeds with near-identical reason
     strings.
  2. The fixed comfort state used throughout v1/v2 was low-stakes: HVAC
     indoor sat *inside* the comfort band, and the washer's "moderate slack"
     state was read by the model as "plenty of slack" every time -- so
     comfort/urgency correctly had ~nothing to trade against price, which by
     itself explains a flat WTP without implicating the price channel at all.

This probe fixes both to get a clean read: temperature explicitly pinned to
0.0 (deterministic greedy decode -- 1 call per cell, no seeds, with a
determinism check on 2 cells), and TWO comfort states per appliance so the
"low-stakes" and "genuinely contested" cases can be contrasted directly:

  - HVAC: (a) comfortable, indoor 25.0C (inside the 22-26C band, same as
    v1/v2) vs. (b) contested, indoor 27.5C (1.5C ABOVE the band ceiling --
    real, stated discomfort).
  - washer: (a) low-urgency, large slack (steps_until_deadline = 8, double
    the 4 steps_remaining) vs. (b) high-urgency, zero slack
    (steps_until_deadline == steps_remaining == 4, must start now).

CPP flag held OFF throughout (still decoupling price level from the flag,
per v2); the v2 anchored price framing is reused unchanged
(`probe_wtp_price_v2.inject_price_anchor`). All three personas are back
(price_conscious included -- its v1/v2 flatness may have been the same
low-stakes artifact, and at T=0 a recheck is nearly free).

Still diagnostic only -- no production prompt/config file is touched.
Temperature is set only in this probe's own direct `generate_content` call
(`call_live_temp`), never in `llm/client.py`.

Cells: 3 personas x 2 appliances x 2 comfort states x 5 prices x 1 call = 60,
plus a 2-cell determinism re-run (+2 calls) and an optional temperature=0.3
recheck on 2 contested cells x 5 prices (+10 calls) = 72 total.

Usage (from `llm-hems/`):
  PYTHONPATH=src python experiments/probe_wtp_price_v3.py
  PYTHONPATH=src python experiments/probe_wtp_price_v3.py --skip-optional
  PYTHONPATH=src python experiments/probe_wtp_price_v3.py --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import json
import time
from pathlib import Path

from probe_wtp_price import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_RATE_LIMIT_RPM,
    PERSONAS,
    RESULTS_DIR,
)
from probe_wtp_price_v2 import build_prompt
from run_single import load_environment

from hems.agents.base import DecisionContext
from hems.agents.llm_agent import LLMAgent
from hems.llm.client import LLMClient
from hems.llm.schemas import HVACWTPResponse, WasherWTPResponse

print = functools.partial(print, flush=True)  # noqa: A001 -- unbuffered for background monitoring

PRICES = [0.22, 0.28, 0.55, 1.20, 2.50]

HVAC_T = 30
HVAC_CLOCK = "15:00"
HVAC_OUTDOOR_TEMP_C = 33.0
HVAC_STATES = [("comfortable", 25.0), ("contested", 27.5)]  # indoor_temp_c

WASHER_T = 10
WASHER_CLOCK = "05:00"
WASHER_STATES = [("low_urgency", 18), ("high_urgency", 14)]  # deadline_step

# Determinism check: one HVAC cell, one washer cell, re-called a second time
# to confirm temperature=0.0 gives identical output.
DETERMINISM_CELLS = [
    ("hvac", "comfort_first", "contested", 0.55),
    ("washer", "neutral", "high_urgency", 1.20),
]

# Optional temperature=0.3 recheck: 2 contested cells across the full price
# grid, 1 call each (no seeds) -- cheap check for whether mild temperature
# preserves the T=0 signal.
OPTIONAL_TEMP = 0.3
OPTIONAL_CELLS = [
    ("hvac", "comfort_first", "contested"),
    ("washer", "comfort_first", "high_urgency"),
]

REVERSAL_TOLERANCE = 0.02


def build_hvac_ctx(hvac_params, persona: dict, rate: float, indoor_temp_c: float) -> DecisionContext:
    return DecisionContext(
        t=HVAC_T,
        clock=HVAC_CLOCK,
        day_index=0,
        appliance={
            "type": "hvac",
            "indoor_temp_c": indoor_temp_c,
            "outdoor_temp_c": HVAC_OUTDOOR_TEMP_C,
            "setpoint_c": hvac_params.setpoint_c,
            "tolerance_band_c": hvac_params.tolerance_band_c,
            "rated_kw": hvac_params.rated_kw,
        },
        cost_now=hvac_params.rated_kw * hvac_params.dt_hours * rate,
        rate=rate,
        is_cpp=False,
        household=persona,
    )


def build_washer_ctx(washer_params, persona: dict, rate: float, deadline_step: int) -> DecisionContext:
    return DecisionContext(
        t=WASHER_T,
        clock=WASHER_CLOCK,
        day_index=0,
        appliance={
            "type": "washer",
            "steps_completed": 0,
            "duration_steps": washer_params.duration_steps,
            "released_step": washer_params.released_step,
            "deadline_step": deadline_step,
            "interruptible": False,
            "rated_kw": washer_params.power_kw,
        },
        cost_now=washer_params.power_kw * washer_params.dt_hours * rate,
        rate=rate,
        is_cpp=False,
        household=persona,
    )


FIXED_SEED = 42  # only for temperature=0.0 calls -- see module docstring's determinism note


def call_live_temp(
    backend, prompt: str, schema, temperature: float, seed: int | None = None, max_retries: int = 3
):
    """Like probe_wtp_price.call_live, but pins `temperature` (and optionally
    `seed`) explicitly in the request -- `llm/client.py`'s
    `_GeminiBackend.call()` never sets either, so this bypasses it entirely
    rather than extending it, while still calling the shared backend's real
    `_throttle()` for rate-limit pacing and reusing its resolved
    api_key/model/max_output_tokens.

    IMPORTANT (discovered empirically before the full run): `temperature=0.0`
    ALONE is not sufficient for determinism on this model -- a manual repeat
    check on one contested-state prompt gave wtp=2.5 then wtp=12.5 with
    temperature=0.0 and no seed. Adding a fixed `seed` alongside
    `temperature=0.0` did produce identical output across 3 repeats (wtp=8.5
    all three times) on the same prompt. The main grid and determinism check
    both pass `seed=FIXED_SEED`; the optional temperature=0.3 recheck passes
    `seed=None` deliberately, since its purpose is to observe natural
    sampling variation.
    """
    from google import genai
    from google.genai import types

    last_exc: Exception | None = None
    rate_limited = False
    for attempt in range(max_retries + 1):
        try:
            backend._throttle()  # noqa: SLF001
            client = genai.Client(api_key=backend._api_key)  # noqa: SLF001
            t0 = time.monotonic()
            response = client.models.generate_content(
                model=backend._model,  # noqa: SLF001
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=schema.model_json_schema(),
                    max_output_tokens=backend._max_output_tokens,  # noqa: SLF001
                    temperature=temperature,
                    seed=seed,
                ),
            )
            latency = time.monotonic() - t0
            if not response.text:
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                raise RuntimeError(f"gemini returned no text (finish_reason={finish_reason})")
            raw = json.loads(response.text)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-optional", action="store_true", help="skip the temperature=0.3 recheck")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 persona x 1 appliance x 1 state x 2 prices (2 calls), no determinism/optional passes",
    )
    args = parser.parse_args()

    personas = PERSONAS
    prices = PRICES
    appliance_states = [
        ("hvac", HVACWTPResponse, build_hvac_ctx, HVAC_STATES),
        ("washer", WasherWTPResponse, build_washer_ctx, WASHER_STATES),
    ]
    if args.smoke_test:
        personas = {"neutral": PERSONAS["neutral"]}
        prices = [PRICES[0], PRICES[-1]]
        appliance_states = [(appliance_states[0][0], appliance_states[0][1], appliance_states[0][2], appliance_states[0][3][:1])]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    household, tariff, hvac_params, washer_params = load_environment()
    params_by_appliance = {"hvac": hvac_params, "washer": washer_params}

    client = LLMClient(
        provider="gemini",
        model=GEMINI_MODEL,
        rate_limit_rpm=GEMINI_RATE_LIMIT_RPM,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        cache_path=None,
    )
    agent = LLMAgent(client=client)
    backend = client._backend  # noqa: SLF001

    print(f"[probe_wtp_price_v3] provider=gemini model_id={agent.model_id} sdk_version={agent.sdk_version}")
    print("[probe_wtp_price_v3] temperature=0.0 (explicit, deterministic greedy decode)")
    print(
        f"[probe_wtp_price_v3] HVAC states: comfortable=25.0C, contested=27.5C "
        f"(band=[{hvac_params.band_lo_c:.1f},{hvac_params.band_hi_c:.1f}]C) clock={HVAC_CLOCK}"
    )
    print(
        f"[probe_wtp_price_v3] washer states: low_urgency deadline_step=18 (steps_until_deadline=8, "
        f"slack=4), high_urgency deadline_step=14 (steps_until_deadline=4, slack=0), t={WASHER_T}"
    )

    fieldnames = [
        "phase", "appliance", "persona", "comfort_state", "rate", "is_cpp", "temperature",
        "wtp", "reason", "cost_now", "utility", "run_decision", "latency_s",
    ]
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{timestamp}_probe_wtp_price_v3.csv"
    csv_file = out_path.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    rows: list[dict] = []
    prompts_by_key: dict[tuple, str] = {}
    n_calls = 0
    n_rate_limited = 0
    t_start = time.monotonic()

    print("\n=== MAIN GRID: temperature=0.0, flag OFF, anchored framing, 2 comfort states/appliance ===")
    for appliance_name, schema, ctx_builder, states in appliance_states:
        params = params_by_appliance[appliance_name]
        for persona_name, persona in personas.items():
            for state_label, state_value in states:
                for rate in prices:
                    ctx = ctx_builder(params, persona, rate, state_value)
                    prompt = build_prompt(agent, appliance_name, ctx)
                    prompts_by_key[(appliance_name, persona_name, state_label, rate)] = prompt
                    try:
                        resp, latency, rl = call_live_temp(backend, prompt, schema, temperature=0.0, seed=FIXED_SEED)
                    except RuntimeError as exc:
                        print(f"  FAILED main/{appliance_name}/{persona_name}/{state_label}/{rate}: {exc}")
                        continue
                    n_calls += 1
                    n_rate_limited += int(rl)
                    row = {
                        "phase": "main", "appliance": appliance_name, "persona": persona_name,
                        "comfort_state": state_label, "rate": rate, "is_cpp": False, "temperature": 0.0,
                        "wtp": resp.wtp, "reason": resp.reason, "cost_now": round(ctx.cost_now, 4),
                        "utility": round(resp.wtp - ctx.cost_now, 4), "run_decision": (resp.wtp - ctx.cost_now) >= 0,
                        "latency_s": round(latency, 3),
                    }
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                    print(
                        f"  [main/{appliance_name}/{persona_name}/{state_label} rate={rate}] "
                        f"wtp={resp.wtp:.3f} ({latency:.2f}s)"
                    )

    if not args.smoke_test:
        print("\n=== DETERMINISM CHECK: re-calling 2 cells a second time at T=0.0 ===")
        for appliance_name, persona_name, state_label, rate in DETERMINISM_CELLS:
            key = (appliance_name, persona_name, state_label, rate)
            prompt = prompts_by_key.get(key)
            if prompt is None:
                print(f"  SKIPPED (cell not in main grid): {key}")
                continue
            schema = HVACWTPResponse if appliance_name == "hvac" else WasherWTPResponse
            try:
                resp, latency, rl = call_live_temp(backend, prompt, schema, temperature=0.0, seed=FIXED_SEED)
            except RuntimeError as exc:
                print(f"  FAILED determinism/{key}: {exc}")
                continue
            n_calls += 1
            n_rate_limited += int(rl)
            original = next(r for r in rows if (r["appliance"], r["persona"], r["comfort_state"], r["rate"]) == key)
            identical = abs(resp.wtp - original["wtp"]) < 1e-9
            row = {
                "phase": "determinism_check", "appliance": appliance_name, "persona": persona_name,
                "comfort_state": state_label, "rate": rate, "is_cpp": False, "temperature": 0.0,
                "wtp": resp.wtp, "reason": resp.reason, "cost_now": original["cost_now"],
                "utility": round(resp.wtp - original["cost_now"], 4), "run_decision": (resp.wtp - original["cost_now"]) >= 0,
                "latency_s": round(latency, 3),
            }
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            print(
                f"  [determinism/{appliance_name}/{persona_name}/{state_label} rate={rate}] "
                f"original_wtp={original['wtp']:.3f} rerun_wtp={resp.wtp:.3f} identical={identical}"
            )

        if not args.skip_optional:
            print(f"\n=== OPTIONAL: temperature={OPTIONAL_TEMP} recheck on 2 contested cells, full price grid ===")
            for appliance_name, persona_name, state_label in OPTIONAL_CELLS:
                params = params_by_appliance[appliance_name]
                persona = personas[persona_name]
                ctx_builder = build_hvac_ctx if appliance_name == "hvac" else build_washer_ctx
                state_value = dict(HVAC_STATES if appliance_name == "hvac" else WASHER_STATES)[state_label]
                schema = HVACWTPResponse if appliance_name == "hvac" else WasherWTPResponse
                for rate in prices:
                    ctx = ctx_builder(params, persona, rate, state_value)
                    prompt = build_prompt(agent, appliance_name, ctx)
                    try:
                        resp, latency, rl = call_live_temp(backend, prompt, schema, temperature=OPTIONAL_TEMP, seed=None)
                    except RuntimeError as exc:
                        print(f"  FAILED optional/{appliance_name}/{persona_name}/{state_label}/{rate}: {exc}")
                        continue
                    n_calls += 1
                    n_rate_limited += int(rl)
                    row = {
                        "phase": f"temp_{OPTIONAL_TEMP}", "appliance": appliance_name, "persona": persona_name,
                        "comfort_state": state_label, "rate": rate, "is_cpp": False, "temperature": OPTIONAL_TEMP,
                        "wtp": resp.wtp, "reason": resp.reason, "cost_now": round(ctx.cost_now, 4),
                        "utility": round(resp.wtp - ctx.cost_now, 4), "run_decision": (resp.wtp - ctx.cost_now) >= 0,
                        "latency_s": round(latency, 3),
                    }
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                    print(
                        f"  [temp{OPTIONAL_TEMP}/{appliance_name}/{persona_name}/{state_label} rate={rate}] "
                        f"wtp={resp.wtp:.3f} ({latency:.2f}s)"
                    )

    csv_file.close()
    wall_clock_s = time.monotonic() - t_start

    print("\n" + "=" * 80)
    print("RUN METADATA")
    print("=" * 80)
    print(
        f"total_calls={n_calls} wall_clock_s={wall_clock_s:.1f} model_id={agent.model_id} "
        f"temperature=0.0 (main+determinism), {OPTIONAL_TEMP} (optional pass) -- explicit in this "
        f"probe's own call, not client.py rate_limited_calls={n_rate_limited}"
    )
    print(f"raw results written to {out_path}")


if __name__ == "__main__":
    main()
