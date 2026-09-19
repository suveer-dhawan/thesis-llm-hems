# Build Spec — LLM Satisficing HEMS Simulator

**Purpose:** implementation brief for the thesis simulator. Written to be handed to Claude Code (in VS Code) as the build contract. Scoped for an HD-grade, part-time build.

**Companion doc:** `decision-memo-semester2.md` (framing + decisions). Read that first.

> **CONFIRM tags:** anything marked `CONFIRM` is a sensible default so the build can start today, but should be checked against a real source (Ausgrid tariff, appliance nameplate data, current model IDs) before results are reported.

---

## 1. What we are building (one paragraph)

A discrete-time (30-min) simulator of a single household's appliance scheduling under a dynamic electricity tariff. A **deterministic Python core** owns all physics and all arithmetic (appliance state, energy, cost). A pluggable **decision agent** decides, each interval, whether to run each controllable appliance. Three agents are compared: a **MILP** cost-minimiser (hyper-rational), a **noisy-MILP** (heterogeneous strawman), and the **LLM satisficing agent** (proposes a willingness-to-pay; Python enforces the halt). The headline result is that the LLM's induced **price elasticity** falls in the empirical human range while it avoids the synchronised peak the MILP produces at scale.

## 2. Design principles (non-negotiable)

1. **All numbers live in Python.** The LLM never does arithmetic. It returns a subjective valuation; Python computes cost, compares, and decides. This is the core methodological claim — enforce it in code, not just in prose.
2. **Agents are interchangeable behind one interface.** The simulator loop does not know which agent it is running. Swapping MILP ↔ LLM changes one config line.
3. **Everything is config-driven and seeded.** No magic numbers in source. One `experiment.yaml` fully determines a run. Every run writes its resolved config + seed into its results folder.
4. **Every decision is logged.** For the LLM agent, log the prompt, the returned WTP, the reasoning string, the computed cost, and the run/defer decision — every appliance, every interval. Auditability is what defends "it satisfices" against "it's just noise."

## 3. Repository layout

```
llm-hems/
  README.md
  requirements.txt
  pyproject.toml                # or setup.cfg; ruff + pytest config
  config/
    appliances.yaml
    tariff.yaml
    household.yaml              # persona / comfort params
    household_sample.csv        # 24-household population manifest (Phase 0/4)
    experiment.yaml             # ties it all together for one run
  data/
    raw/                        # SGSC clean CSVs (gitignored)
    processed/                  # cached parquet after ingest
  src/hems/
    __init__.py
    environment/
      tariff.py                 # TOU + CPP cost calculator
      appliances.py             # HVAC thermal model + shiftable load
      household.py              # base load + physical state container
      simulator.py              # the 30-min orchestration loop
    agents/
      base.py                   # DecisionAgent interface
      milp_agent.py             # PuLP optimal
      noisy_milp_agent.py       # heterogeneous / jittered MILP
      llm_agent.py              # hybrid satisficing agent
      prompts/
        hvac.txt
        shiftable.txt
    llm/
      client.py                 # provider-agnostic wrapper
      schemas.py                # pydantic I/O contracts
    analysis/
      metrics.py                # PAR, peak kW, cost, load curves
      elasticity.py             # elasticity estimator
      plots.py
  experiments/
    run_single.py               # one household, one agent
    run_population.py           # loop over N households -> aggregate
    calibrate.py                # fit WTP scale on calibration subset
  results/                      # timestamped run folders (gitignored)
  tests/
    test_tariff.py
    test_appliances.py
    test_simulator.py
    test_elasticity.py
```

## 4. The simulation loop (the spine)

`simulator.py` runs one household over a horizon (default: one day = 48 steps; extend to multi-day for elasticity).

```
for t in 0..47:                       # 30-min steps
    tariff_state = tariff.rate_at(t)          # $/kWh, is_cpp flag
    for appliance in controllable:
        ctx = build_context(appliance, household, t, tariff_state)
        decision = agent.decide(ctx)          # run: bool  (agent-specific internals)
        if decision.run:
            energy = appliance.step(on=True)   # updates physical state
            cost   = energy * tariff_state.rate
        else:
            appliance.step(on=False)
        log(t, appliance, decision, cost, appliance.state)
    accumulate_load(t)
return RunResult(load_curve, total_cost, decision_log)
```

Base (uncontrollable) load for each interval comes from the SGSC profile so the household's aggregate curve is realistic; the controllable appliances are added on top.

### Base load vs. flexible layer (Phase 1 design decision)

SGSC is a **whole-home aggregate** meter reading — it cannot be disaggregated into per-appliance traces. Rather than pretend otherwise, the simulator treats each sampled household's SGSC curve as its **fixed, non-controllable base load** exactly as metered, and adds HVAC + the washer on top as an **explicit flexible layer** that the agent controls. This means the SGSC curve's own historical AC/appliance behaviour is *not* subtracted out before the synthetic flexible layer is added — the flexible layer is a deliberate modelling addition on top of the real aggregate, not a disaggregation of it. `total = base + flexible` at every step; `cost = total_kwh * rate` (the bill doesn't distinguish base from flexible — only the flexible portion is within the agent's control).

The flexible appliances are sized (config-driven, `config/appliances.yaml`) so that under normal operation flexible energy is a **realistic ~10–25% of the household's daily total** (`flexible / (base + flexible)`), printed by every run as `flexible_share_pct`. For the Phase 1 reference household (`8596070`, a hot summer weekday, see below), a bang-bang thermostat calibration against `r=2.0, c=2.5` (5h RC time constant) gives HVAC ≈ 10–14 kWh/day depending on control policy, plus a fixed 1.6 kWh washer cycle, landing flexible share at **~17–21%** of the day's total — inside the target band.

Because elasticity (Phase 3) is defined as a *whole-household* price response, `simulator.py` always tracks `base`, `flexible`, and `total` load (and cost) **separately at every step**, never collapsing them early — elasticity will be measured on `total`, matching a whole-household elasticity definition, while `flexible` remains available for per-appliance diagnostics.

Units, kept consistent everywhere in the codebase: SGSC cells and all appliance energy figures are **kWh per 30-min step**; `kW = kWh_per_step * 2`; `cost($) = energy(kWh) * rate($/kWh)`.

## 5. Agent interface

```python
# agents/base.py
class Decision(BaseModel):
    run: bool
    wtp: float | None = None      # LLM only; None for MILP
    reason: str | None = None

class DecisionAgent(Protocol):
    def decide(self, ctx: DecisionContext) -> Decision: ...
    def name(self) -> str: ...
```

`DecisionContext` (what every agent receives — Python-computed, appliance-agnostic where possible):
- `t`, `clock` (HH:MM), `day_index`
- `appliance`: type, semantics, rated_kw, and appliance-specific state
  - HVAC: `indoor_temp`, `outdoor_temp`, `setpoint`, `tolerance_band`
  - shiftable: `steps_remaining_to_complete`, `deadline_step`, `released_step`, `interruptible: false`
- `cost_now`: **exact $ to run this appliance this interval** (Python)
- `rate`, `is_cpp`, and (optional) `rate_forecast` for the rest of the horizon — **Python-only**: used for `cost_now` and the halt rule, never rendered into the LLM prompt (see §6.3)
- shiftable `slack_steps` (`steps_until_deadline - steps_remaining`, Python-computed): the deadline buffer, stated explicitly in the prompt so the LLM never has to derive it from two separate numbers
- `household`: comfort/persona params (see §8)

## 6. The three agents

### 6.1 MILP (hyper-rational baseline)
- Library: **PuLP** (CBC solver is fine).
- Objective: minimise total energy cost over the horizon.
- Constraints: shiftable load runs `duration` contiguous steps within `[released, deadline]`; HVAC keeps indoor temp inside a **hard** comfort band every step (linearised thermal dynamics — see §7).
- Output: the optimal on/off schedule → fed through the same `simulator` accounting for an apples-to-apples load curve.

### 6.2 Noisy-MILP (heterogeneous strawman) — *cheap, high value*
Same MILP, but per household draw random perturbations (seeded): jitter the comfort band width, the setpoint, and/or the effective deadline within tolerance; optionally add small random noise to the perceived price. Purpose: show whether trivial heterogeneity alone reproduces staggering + a bounded elasticity. The LLM must beat *this*, not just the single clean MILP.

### 6.3 LLM satisficing agent (the thesis)
Per appliance per interval:
1. Python builds the context and computes `cost_now`.
2. LLM is prompted (structured output) to return **`wtp`**: the dollar amount this household would willingly pay for the benefit of operating the appliance in *this* interval, given comfort state and time — plus a one-line `reason`. **No cost comparison is asked of the LLM.**
3. Python computes utility `U = wtp - cost_now` and applies the **halt rule: run iff `U >= 0`** (satisfice — run while good-enough, stop when the price friction exceeds the comfort valuation).
4. Log everything.

**The prompt is price-blind.** The LLM is never told the electricity rate or whether a CPP event is active — it sees only comfort/urgency state (indoor temp + comfort band for HVAC; cycle progress + the Python-computed deadline `slack_steps` for the washer) and the household persona. `DecisionContext.rate` / `is_cpp` / `cost_now` still exist and are used by Python for step 3's cost/halt arithmetic; they are simply never rendered into `agents/prompts/{hvac,shiftable}.txt`. This was a post-hoc fix (`experiments/probe_wtp_price*.py`): with price in the prompt, the WTP itself already encoded a price reaction, contaminating the price-elasticity measurement the halt rule is supposed to produce on its own (§1, §10).

Rationale for the split: LLMs are reliable at qualitative preference valuation, unreliable at arithmetic (macmillan2024ir); the halt is deterministic and reproducible because Python owns the comparison.

**Model config:** primary model + one ablation model, both current (`CONFIRM` exact IDs — GPT-4o / Claude 3.5 Sonnet as written in Part 1 are dated). Default provider is `gemini` (`config/experiment.yaml`'s `llm.provider`) — Matilda's API exposes no `temperature`/`seed` parameter, so it can't meet the determinism requirement below. **Temperature = 0.0, seed = 42** (`llm.temperature` / `llm.seed`, both passed into Gemini's `GenerateContentConfig`): `probe_wtp_price_v3.py` found temperature=0 alone does **not** make two identical prompts return an identical `wtp` — the seed is required as well. Run `n_seeds` (default 3) per scenario; report spread.

## 7. Appliance models (`CONFIRM` all params)

### HVAC (thermal friction)
First-order RC thermal model, per 30-min step `Δt`:
```
T_in[t+1] = T_in[t] + (Δt / (R*C)) * (T_out[t] - T_in[t]) - (COP * P_hvac * on[t] * Δt) / C
```
- `P_hvac` rated ≈ **2.0 kW** `CONFIRM`; `COP` ≈ **3.0** `CONFIRM`
- `R`, `C` chosen so a house drifts a few °C over an hour with HVAC off `CONFIRM` (calibrate to a plausible drift rate; document the choice)
- setpoint ≈ **24 °C**, comfort tolerance ± **2 °C** `CONFIRM`
- `T_out[t]`: use a synthetic summer day curve, or SGSC-region temperature if available `CONFIRM`
- **For MILP:** linearise (fix COP, treat as linear in `on[t]`) so it stays an MILP.

### Shiftable load — washing machine (temporal friction)
- Non-interruptible, **duration = 2 h = 4 steps** `CONFIRM`
- Power ≈ **0.8 kW** `CONFIRM`
- `released_step` (earliest start) and `deadline_step` (must finish by e.g. 21:00) from `household.yaml`
- Comfort semantics for the LLM: WTP to have laundry done rises as the deadline nears (urgency), and is ~0 long before it's needed.

Keep the appliance set to exactly these two for the core thesis. Do not add EV / battery unless Phase 3 finishes early.

## 8. Household / persona params (`household.yaml`)
Small set of interpretable comfort parameters the LLM prompt is conditioned on and the MILP bands are derived from — e.g. `thermal_sensitivity` (tight vs loose band), `price_sensitivity` (a qualitative descriptor, not a number the LLM does math on), `deadline` for the shiftable load, `occupied_hours`. Representative households come from the SGSC clusters (Phase 0 selects them).

## 9. Tariff (`tariff.yaml`)
Window structure (peak/shoulder/off-peak hours, and the seasonal peak) follows Ausgrid's current residential TOU network periods (source: "Ausgrid residential TOU network periods, 2025–26"); the $/kWh rates are representative retail values on top of that network tariff, `CONFIRM` against a current retail plan:
- Peak (15:00–21:00, every day, **summer Nov–Mar and winter Jun–Aug only**): `$0.55/kWh`. Apr–May and Sep–Oct have no peak season — that window falls back to the shoulder rate.
- Shoulder (07:00–15:00 and 21:00–22:00, every day): `$0.28/kWh`
- Off-peak (22:00–07:00, every day): `$0.22/kWh`
- **CPP events:** on flagged days, override a 2–4 h window with a spike. Two severities are configured for sensitivity: moderate `$1.20/kWh` (the default demo event, on the Phase 1 hot day `2013-01-18`) and severe `$2.50/kWh` `CONFIRM`. Schedule of CPP days is config.

`tariff.rate_at(t, date)` returns `(rate, is_cpp)`; the peak season and weekday both depend on `date`. Unit-tested against the table, including the seasonal peak gate and the CPP severities (`tests/test_tariff.py`).

## 10. Elasticity estimator (`analysis/elasticity.py`) — the key metric

Short-run own-price arc elasticity of demand for the priced window:
```
elasticity = (%ΔQ) / (%ΔP)
           = ((Q_cpp - Q_base) / Q_base) / ((P_cpp - P_base) / P_base)
```
where `Q` is energy consumed in the affected window, comparing a **baseline day** (no CPP) against a **CPP day**, holding the household fixed. Report per-appliance and aggregate. Expect negative values.

**Success criterion:** aggregate short-run elasticity falls within Conway 2020's Australian range **−0.026 to −0.47**, and — to show it isn't hand-tuned — the WTP scale is calibrated on a **calibration subset** of households (`calibrate.py`) and elasticity is then reported on a **held-out subset**. State the pre-registered target and split in the write-up.

## 11. Other metrics (`analysis/metrics.py`)
- **Peak-to-Average Ratio (PAR)** of the aggregate load curve.
- **Peak demand (kW)** and time-of-peak.
- **Total cost ($)** per household.
- **Load curves** for the four-way overlay: SGSC baseline / MILP / noisy-MILP / LLM.
- **Behavioural-coherence check** for the discriminating scenario (§12).

## 12. The discriminating scenario (separates satisficing from noise)
Design one scenario where structured reasoning and random jitter diverge — e.g. a **single unexpected CPP spike** with a **near deadline** on the shiftable load. A noisy-MILP responds only through its random draws; a satisficing agent should coherently trade the deadline urgency against the spike (defer if slack remains, run through the spike if the deadline forces it). Report whether the LLM's behaviour tracks the scenario logic where the noisy-MILP does not. This is the qualitative evidence that the LLM adds something.

## 13. Experiment protocol
1. **Phase 0:** ingest SGSC → select representative households → freeze config. Re-drawn to the **24 interview-complete households** from the sample audit (manifest: `config/household_sample.csv`, frozen into `data/processed/{households,load_curves}.parquet` by `experiments/build_household_sample.py`), replacing the original 20-household stratified sample; demo customer `8596070` is preserved across the re-draw.
2. **Phase 1:** run MILP + noisy-MILP on one household; sanity-check load curves and cost.
3. **Phase 2:** run LLM agent (primary + ablation model) on one household, both appliances.
4. **Phase 3 (single-household):** calibrate WTP on subset; report elasticity on held-out; four-way overlay; multi-seed spread; discriminating scenario.
5. **Phase 4 (population):** loop over N≈10–30 households → aggregate PAR / peak / rebound-peak. Droppable.

Every run: timestamped folder under `results/` with resolved config, seed, decision log (parquet), curves (png), and a `summary.json` of metrics.

## 14. Stack
- Python 3.11+, `pulp`, `pandas`, `numpy`, `pydantic`, `matplotlib`, `pyyaml`, `pytest`, `ruff`.
- LLM SDKs for the two chosen providers, behind `llm/client.py` so neither leaks into agent logic.
- Keep API keys in `.env` (gitignored); `client.py` reads them.

## 15. Cost / latency guard
LLM calls = appliances × steps × seeds × households × models. For one day: 2 × 48 × 3 × 1 × 2 ≈ 576 calls per household — trivial. Population of 30 ≈ 17k calls — still fine but cache aggressively (memoise identical contexts) and add a `--dry-run` stub agent for pipeline testing without spend.

## 16. Definition of done (per phase)
- **P0:** `run_single.py --agent stub` completes on a real SGSC household and writes a load curve.
- **P1:** MILP total cost ≤ noisy-MILP ≤ (expected) LLM cost; all pass unit tests.
- **P2:** LLM decision log shows WTP + reason + U + decision for every step.
- **P3:** held-out elasticity printed with the Conway range; four-way overlay figure produced.
- **P4:** aggregate PAR for MILP vs LLM over the household population.

---

### Open items to close before reporting results
- Exact current model IDs (primary + ablation).
- TOU *windows* now follow Ausgrid's current residential network periods (§9); the $/kWh rates on top of them are still representative retail placeholders — CONFIRM against a current retail plan. CPP severities ($1.20 / $2.50) are still illustrative.
- Appliance nameplate params (HVAC kW/COP, RC values, washer kW/duration).
- Final population size N — the household sample is now frozen at 24 (§13 Phase 0); Phase 4 may run a subset or all 24.
