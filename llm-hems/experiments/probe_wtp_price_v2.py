"""Follow-up diagnostic probe: fair-framing WTP(price) + isolated CPP-flag effect.

Extends `experiments/probe_wtp_price.py` (imports its shared prompt-builders,
persona/fixed-state constants, and cache-bypassing live-call helper) to
settle two questions raised by that run's results, before locking the
price-in-vs-out-of-prompt ablation decision:

  Q1 (fair framing): with the CPP flag held OFF and the raw price given a
     neutral reference anchor (so the model has some sense of what's cheap
     vs. expensive), does WTP move with price in the right direction
     (non-increasing), roughly monotonically, and stably across seeds?
  Q2 (flag effect, isolated): holding price FIXED at the two highest values,
     does flipping the CPP flag ON by itself push WTP UP (the wrong
     direction), independent of the price number?

Still diagnostic only -- no production prompt/config file is touched. The
anchor line is spliced into the already-rendered production prompt string
(`inject_price_anchor`), never into `agents/prompts/*.txt`.

Personas trimmed to the two that mattered in probe_wtp_price.py's results:
`comfort_first` (the only persona whose WTP moved at all) and `neutral`
(control) -- `price_conscious` is dropped, it sat pinned at its floor
regardless of price and added no signal.

Fixed comfort state is unchanged from probe_wtp_price.py: HVAC indoor
25.0C @ clock 15:00 (1C above the 24.0C setpoint, inside the 22-26C band);
washer 0/4 steps completed, deadline_step=16 at t=10 (steps_until_deadline=6,
slack=2 steps / 1h).

Two grids, one run:
  - MAIN GRID: flag OFF at all 5 real tariff.yaml rates (0.22/0.28/0.55/
    1.20/2.50), anchored framing -- 2 personas x 2 appliances x 5 prices x
    4 seeds = 80 calls. is_cpp=False even at 1.20/2.50 here on purpose,
    deliberately decoupling price level from the CPP flag (Q1).
  - FLAG PROBE: flag ON (original "CPP EVENT RIGHT NOW: ACTIVE" wording,
    still the declared source of truth), same anchored framing, only at
    1.20/2.50 -- 2 personas x 2 appliances x 2 prices x 4 seeds = 32 calls
    (Q2, compared cell-for-cell against the matching MAIN GRID flag-off
    rows at the same price).
  112 calls total, ~22 min at the shared backend's 5 req/min throttle.

Usage (from `llm-hems/`):
  PYTHONPATH=src python experiments/probe_wtp_price_v2.py
  PYTHONPATH=src python experiments/probe_wtp_price_v2.py --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import time
from pathlib import Path

from probe_wtp_price import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_RATE_LIMIT_RPM,
    PERSONAS as _ALL_PERSONAS,
    RESULTS_DIR,
    build_hvac_ctx,
    build_washer_ctx,
    call_live,
)
from run_single import load_environment

from hems.agents.llm_agent import LLMAgent
from hems.llm.client import LLMClient
from hems.llm.schemas import HVACWTPResponse, WasherWTPResponse

print = functools.partial(print, flush=True)  # noqa: A001 -- unbuffered for background monitoring

PERSONAS = {name: _ALL_PERSONAS[name] for name in ("comfort_first", "neutral")}

N_SEEDS = 4
MAIN_GRID_PRICES = [0.22, 0.28, 0.55, 1.20, 2.50]  # flag OFF at every one
FLAG_PROBE_PRICES = [1.20, 2.50]  # flag ON, same two prices for a cell-for-cell delta

# Reference anchor: the real off-peak / peak rates from config/tariff.yaml --
# factual, not evaluative (no "expensive"/"cheap" language, no instruction to
# act on it).
ANCHOR_LOW = 0.22
ANCHOR_HIGH = 0.55


def inject_price_anchor(prompt: str) -> str:
    """Splices a factual price-reference line into the already-rendered
    production prompt, directly after its "electricity price now" line --
    the only change this probe makes to prompt text (the source .txt files
    in agents/prompts/ are never touched).
    """
    anchor_line = (
        f"  price reference (context only) : off-peak here is about "
        f"${ANCHOR_LOW:.2f}/kWh, peak is about ${ANCHOR_HIGH:.2f}/kWh"
    )
    lines = prompt.split("\n")
    for i, line in enumerate(lines):
        if "electricity price now" in line:
            lines.insert(i + 1, anchor_line)
            break
    else:
        raise ValueError("could not find 'electricity price now' line to anchor")
    return "\n".join(lines)


def build_prompt(agent: LLMAgent, appliance_name: str, ctx) -> str:
    base = (
        agent._build_hvac_prompt(ctx)  # noqa: SLF001 -- reusing production builder read-only
        if appliance_name == "hvac"
        else agent._build_washer_prompt(ctx)  # noqa: SLF001
    )
    return inject_price_anchor(base)


def run_grid(
    agent: LLMAgent,
    backend,
    writer: csv.DictWriter,
    csv_file,
    rows: list[dict],
    appliance_specs,
    personas: dict,
    price_points: list[float],
    force_cpp: bool,
    grid_label: str,
    n_seeds: int,
) -> tuple[int, int]:
    n_calls = 0
    n_rate_limited = 0
    for appliance_name, schema, ctx_builder, params in appliance_specs:
        for persona_name, persona in personas.items():
            for rate in price_points:
                ctx = ctx_builder(params, persona, rate, force_cpp)
                prompt = build_prompt(agent, appliance_name, ctx)
                for seed in range(n_seeds):
                    try:
                        resp, latency, rate_limited = call_live(backend, prompt, schema)
                    except RuntimeError as exc:
                        print(f"  FAILED {grid_label}/{appliance_name}/{persona_name}/{rate}/seed{seed}: {exc}")
                        continue
                    n_calls += 1
                    if rate_limited:
                        n_rate_limited += 1
                    row = {
                        "grid": grid_label,
                        "appliance": appliance_name,
                        "persona": persona_name,
                        "rate": rate,
                        "is_cpp": force_cpp,
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
                        f"  [{grid_label}/{appliance_name}/{persona_name} rate={rate} "
                        f"cpp={force_cpp} seed={seed}] wtp={resp.wtp:.3f} ({latency:.2f}s)"
                    )
    return n_calls, n_rate_limited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 persona x 1 appliance x 2 prices x 1 seed, both grids (4 calls) -- connectivity/logic check only",
    )
    args = parser.parse_args()

    personas = PERSONAS
    main_prices = MAIN_GRID_PRICES
    flag_prices = FLAG_PROBE_PRICES
    n_seeds = N_SEEDS
    if args.smoke_test:
        personas = {"neutral": PERSONAS["neutral"]}
        main_prices = [MAIN_GRID_PRICES[0], MAIN_GRID_PRICES[-1]]
        flag_prices = [FLAG_PROBE_PRICES[-1]]
        n_seeds = 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    household, tariff, hvac_params, washer_params = load_environment()

    client = LLMClient(
        provider="gemini",
        model=GEMINI_MODEL,
        rate_limit_rpm=GEMINI_RATE_LIMIT_RPM,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        cache_path=None,
    )
    agent = LLMAgent(client=client)
    backend = client._backend  # noqa: SLF001 -- read-only, bypasses complete_json's cache

    print(f"[probe_wtp_price_v2] provider=gemini model_id={agent.model_id} sdk_version={agent.sdk_version}")
    print(f"[probe_wtp_price_v2] rate_limit_rpm={GEMINI_RATE_LIMIT_RPM} n_seeds={n_seeds}")
    print(f"[probe_wtp_price_v2] anchor: off-peak~${ANCHOR_LOW:.2f}/kWh, peak~${ANCHOR_HIGH:.2f}/kWh (factual, no framing)")
    print(
        f"[probe_wtp_price_v2] HVAC fixed state: indoor={hvac_params.setpoint_c + 1.0:.1f}C "
        f"band=[{hvac_params.band_lo_c:.1f},{hvac_params.band_hi_c:.1f}]C clock=15:00 (unchanged from v1)"
    )
    print(
        f"[probe_wtp_price_v2] washer fixed state: steps_completed=0/{washer_params.duration_steps} "
        f"deadline_step=16 t=10 clock=05:00 (steps_until_deadline=6, slack=2 steps) (unchanged from v1)"
    )

    fieldnames = [
        "grid", "appliance", "persona", "rate", "is_cpp", "seed",
        "wtp", "reason", "cost_now", "utility", "run_decision", "latency_s",
    ]
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{timestamp}_probe_wtp_price_v2.csv"
    csv_file = out_path.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    appliance_specs = [
        ("hvac", HVACWTPResponse, build_hvac_ctx, hvac_params),
        ("washer", WasherWTPResponse, build_washer_ctx, washer_params),
    ]
    if args.smoke_test:
        appliance_specs = appliance_specs[:1]

    rows: list[dict] = []
    t_start = time.monotonic()

    print("\n=== MAIN GRID: flag OFF, anchored framing ===")
    n_main, rl_main = run_grid(
        agent, backend, writer, csv_file, rows, appliance_specs, personas, main_prices,
        force_cpp=False, grid_label="main_flag_off", n_seeds=n_seeds,
    )

    print("\n=== FLAG PROBE: flag ON, anchored framing, high prices only ===")
    n_flag, rl_flag = run_grid(
        agent, backend, writer, csv_file, rows, appliance_specs, personas, flag_prices,
        force_cpp=True, grid_label="flag_on", n_seeds=n_seeds,
    )

    csv_file.close()
    wall_clock_s = time.monotonic() - t_start
    n_calls = n_main + n_flag
    n_rate_limited = rl_main + rl_flag

    print("\n" + "=" * 80)
    print("RUN METADATA")
    print("=" * 80)
    print(
        f"total_calls={n_calls} (main_grid={n_main}, flag_probe={n_flag}) "
        f"wall_clock_s={wall_clock_s:.1f} model_id={agent.model_id} "
        f"temperature=1.0 (provider default -- client.py sets none; confirmed via "
        f"`client.models.get(model='{GEMINI_MODEL}')`.temperature) "
        f"rate_limited_calls={n_rate_limited}"
    )
    print(f"raw results written to {out_path}")
    print(f"\n[probe_wtp_price_v2] wrote {len(rows)} rows -- use a separate analysis pass to build the report tables")


if __name__ == "__main__":
    main()
