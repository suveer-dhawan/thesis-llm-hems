# Build Spec — LLM Satisficing HEMS Simulator

**Status (2026-09-27):** describes the architecture **as built** at tag
`final-run-v1` / `final-run-v1-arm-a` and on `main`. The original build brief
(Phase 0–4 plan) is in git history. §14 lists every place the built system
departs from it and why. Code lives in `llm-hems/`; how to run and reproduce
it is in `llm-hems/README.md`, and every results folder is indexed in
`llm-hems/results/INDEX.md`. The final-run design is fixed by
`llm-hems/docs/prereg_final_run_v1.md`.

---

## 1. What the system does

A discrete-time (30-min, 48 steps/day) simulator of household
air-conditioning under a time-of-use tariff with one critical-peak-pricing
(CPP) event. A **deterministic Python core** owns all physics and arithmetic:
thermal state, energy and cost. A pluggable **decision agent** decides, each
step, whether to run the AC. Four agents run the same households, physics and
tariff:

| Agent | Where the decision comes from |
|---|---|
| **MILP** | PuLP/CBC cost minimiser over the whole day with perfect foresight of tariff and weather; hard comfort band. |
| **Noisy-MILP** | The same MILP with seeded per-household jitter of band width, setpoint and deadline (and optionally perceived price); seeds 42–44. |
| **Rule-based satisficer** | `WTP = k[orientation] × max(0, T_in − ceiling)`, through the **same** Python halt as the LLM. No LLM. |
| **LLM satisficer** | The LLM returns a price-blind WTP ($) for running the AC this half-hour; Python applies the halt. |

The research question (pre-registered): does this decoupled design produce
human-range price response, and how do its post-CPP rebound and comfort cost
compare with optimising and rule-based agents?

## 2. Design principles

1. **All numbers live in Python.** The LLM returns a subjective valuation and
   nothing else. Python computes cost, compares, and decides.
2. **The prompt is price-blind.** The LLM never sees a rate, a cost or a CPP
   flag, only comfort state, clock and persona. With price in the prompt, WTP
   itself reacted to price (`experiments/probe_wtp_price*.py`), which would
   double-count the response the halt is supposed to produce.
3. **Agents are interchangeable behind one interface.** The simulator loop
   does not know which agent it is running.
4. **Config-driven, seeded, provenance-stamped.** `config/experiment.yaml`
   fixes a run. Every results folder records `config_resolved.yaml` and
   `provenance.json` (commit, dirty flag, branch, argv, Python version).
5. **Every decision is logged**: prompt, raw WTP, reason, echoed state,
   `cost_now`, utility, run/defer, model version and cache hit, for every
   step.
6. **Reproducible from the cache.** At T = 0 with a fixed seed, the LLM
   response cache is the record. Offline replay reproduces a run exactly with
   zero API calls.

## 3. Repository layout (`llm-hems/`)

```
config/
  tariff.yaml, appliances.yaml, household.yaml
  persona.yaml                 # persona schema: burden terciles, thermal-tolerance rule
  persona_assignment.csv       # frozen per-household personas (experiments/build_personas.py)
  household_sample.csv         # the 24-household manifest
  experiment.yaml              # llm:, noisy_milp:, thermal_heterogeneity:, rule_based:, phase3:
data/processed/                # households.parquet, load_curves.parquet (committed); raw/ gitignored
docs/prereg_final_run_v1.md
src/hems/
  environment/ tariff.py, appliances.py (RC HVAC + washer), household.py,
               persona.py (personas + thermal heterogeneity), simulator.py
  agents/      base.py, stub_agent.py, milp_agent.py, noisy_milp_agent.py,
               rule_based_agent.py, llm_agent.py, prompts/{hvac,shiftable}.txt
  llm/         client.py (Gemini + Matilda backends, disk cache, retries, rate limit), schemas.py
  analysis/    metrics.py, elasticity.py, plots.py
  phase3.py          # final-run harness
  phase3_report.py   # per-household artifacts -> summaries + report.md
  provenance.py
experiments/   run_phase3.py, merge_shards.py, build_personas.py, fit_rule_based.py,
               analyze_final_v1.py, analyze_cross_model.py, analyze_noise_test.py,
               make_thesis_figures.py, plus earlier-phase scripts (run_single, sweeps, probes)
results/       indexed in results/INDEX.md
tests/         129 tests: tariff, appliances, simulator, elasticity, MILP, rule-based,
               LLM agent + client, persona, thermal heterogeneity, harness, sharding
```

## 4. Simulation loop (`environment/simulator.py`)

```
for t in 0..47:
    rate, is_cpp = tariff.rate_at(t, date)
    for appliance in (hvac, washer):
        ctx = DecisionContext(t, clock, appliance state, cost_now, rate, is_cpp, household)
        decision = agent.decide(ctx)
        step the appliance physics with decision.run; account energy and cost
        log the decision
    record base, flexible (HVAC + washer) and total kWh, indoor temperature, band
```

- **Base load:** the household's real SGSC whole-home profile for the date,
  used as-is (not disaggregated). The flexible layer (HVAC, washer) is added
  on top. Base, flexible and total are tracked separately at every step.
- **Units:** kWh per 30-min step everywhere; kW = kWh × 2; cost = kWh × rate.

## 5. Agent interface (`agents/base.py`)

```python
class Decision(BaseModel):
    run: bool
    wtp: float | None = None       # WTP agents only
    reason: str | None = None

class DecisionContext(BaseModel):
    t: int; clock: str; day_index: int
    appliance: dict                # HVAC: indoor_temp_c, setpoint_c, tolerance_band_c, ...
                                   # washer: steps_completed, duration_steps, slack, deadline, ...
    cost_now: float                # exact $ to run this appliance this step (Python)
    rate: float; is_cpp: bool; rate_forecast: list[float] | None
    household: dict                # persona incl. comfort_orientation, archetype

class DecisionAgent(Protocol):
    def decide(self, ctx: DecisionContext) -> Decision: ...
    def name(self) -> str: ...
```

**Halt rule, shared by both satisficers** (`llm_agent.satisficing_halt`):
`U = global_scale × archetype_factor × WTP − cost_now`, run iff `U ≥ 0`. In
the final runs `global_scale = 1.0` (the LLM's dollar WTP is taken literally)
and `archetype_factor` is A/B/C = 1.3 / 1.0 / 0.7.

**Forced washer outcomes** (`forced_washer_outcome`): when Python already
fixes the outcome (cycle complete, non-interruptible cycle in progress, or
before the release step), no agent is asked, so no prompt is sent.

## 6. Agents

### 6.1 MILP (`milp_agent.py`)
Minimises the day's energy cost with PuLP/CBC under linearised RC dynamics.
The comfort ceiling is a hard constraint. The floor limits only AC-induced
cooling: `floor_t = min(band_lo, AC-off free drift_t)`, so a night that drifts
below the floor by itself stays feasible. The MILP plans with each
household's own heterogeneous R, C and initial temperature.

### 6.2 Noisy-MILP (`noisy_milp_agent.py`)
The same MILP after seeded per-household jitter: band ±0.5 °C, setpoint
±0.5 °C, washer deadline ±1 step, perceived-price noise σ = 0.02. The
jittered parameters are the household's actual preferences and feed the
accounting too. It tests whether trivial heterogeneity alone reproduces
staggering and bounded elasticity.

### 6.3 Rule-based satisficer (`rule_based_agent.py`)
The LLM agent with the LLM replaced by a formula, going through the same halt
and forced-washer logic:

- HVAC: `WTP = k[orientation] × max(0, T_in − ceiling)`
- washer: `WTP = w_max × clamp(1 − slack / slack_at_release, 0, 1)`

`k` and `w_max` are fitted by least squares through the origin to raw LLM WTP
from the comfort-sweep logs (`experiments/fit_rule_based.py`), then frozen in
`experiment.yaml → rule_based:`. Fixed-band k = 3.791 / 0.620 / 0.660
(prioritising / balanced / flexible).

### 6.4 LLM satisficer (`llm_agent.py`, `llm/`)
One structured-output call per decision (identical prompts are served from
the cache). The HVAC prompt (`prompts/hvac.txt`) contains exactly:

- the persona sentence for the household's comfort orientation;
- the clock;
- indoor temperature (0.1 °C);
- the comfort band;
- one Python-computed comfort fact ("X °C OVER the ceiling", inside, or
  under).

It contains no price, no cost, no CPP flag and no outdoor temperature. The
response schema (`HVACWTPResponse`) returns `wtp ≥ 0`, a one-line `reason`,
and echoes of the indoor temperature and band bounds. The echoes are checked
against the real input and mismatches are logged as misreads (0 in both final
runs).

**Client** (`llm/client.py`): a provider-agnostic `LLMClient` with Gemini
(`google-genai`, model pinned in config) and Matilda backends. Settings are
T = 0.0 and seed 42 (T = 0 alone was not deterministic), bounded retries with
backoff, and a client-side rate limit. The disk cache is append-only JSONL,
keyed on provider, model, temperature, seed, max output tokens,
response-schema hash and exact prompt. Each entry stores the returned model
version and a timestamp.

**Models:** `gemini-3.5-flash-lite` (final run v1) and `gemini-3.8-flash`
(Arm A). Matilda was rejected after the 2026-09-06 reliability battery: it
misread 38.5% of state and exposes no model id, temperature or seed.

## 7. Physics (`environment/appliances.py`, `persona.py`)

**HVAC:** first-order RC model per 30-min step,
`T[t+1] = T[t] + Δt/(R·C) · (T_out[t] − T[t]) − COP · P · on[t] · Δt / C`.
P = 2.0 kW, COP = 3.0, R = 2.0 °C/kW and C = 2.5 kWh/°C (placeholders). The
setpoint is 24 °C. The outdoor temperature is a synthetic cosine day from
22 to 32 °C, peaking at 15:00.

**Thermal heterogeneity** (`thermal_heterogeneity:`): each household gets
seeded multipliers R × U(0.8, 1.2) and C × U(0.8, 1.2), and starts at
setpoint + U(−1, 1) °C. All agents use the same physics, and the MILP plans
with it.

**Washer:** non-interruptible 4 × 30 min at 0.8 kW with release and deadline
steps. It is disabled in the final runs (`phase3.washer_enabled: false`: a
zero-length cycle, so no prompts and no load), because the pilot showed a
deadline-driven valuation with zero price response by construction.

## 8. Households and personas

- **Households:** 24 interview-complete SGSC households
  (`config/household_sample.csv` → `data/processed/*.parquet`), stratified by
  load-shape cluster × income. Date 2013-01-18, a hot summer Friday.
- **Personas** (`experiments/build_personas.py` → `config/persona_assignment.csv`):
  - **Archetype A/B/C** is the energy-burden tercile: annual kWh ×
    time-weighted rate ÷ income-band midpoint. It acts **only in Python**, as
    the halt's WTP factor. The LLM never sees it.
  - **Comfort orientation** is derived from dwelling and occupancy. Children
    or ducted air conditioning → *comfort-prioritising*; split system →
    *balanced*; no air conditioning → *comfort-flexible*. This gives n =
    5 / 8 / 11. The orientation is the only persona the LLM sees: one
    sentence, with no money wording.
- **Comfort band:** fixed ±2.0 °C around 24 °C for everyone
  (`phase3.band_mode: fixed`, primary). Persona-linked bands were tested in
  the band-confound ablation and are not used in the final runs.

## 9. Tariff (`environment/tariff.py`, `config/tariff.yaml`)

The TOU windows follow Ausgrid's residential network periods (2025–26). Peak
is 15:00–21:00 in summer (Nov–Mar) and winter (Jun–Aug); shoulder is
07:00–15:00 and 21:00–22:00; off-peak is 22:00–07:00. The retail rates are
representative: peak 0.55, shoulder 0.28, off-peak 0.22 $/kWh. **CPP:** on
the CPP day, 17:00–20:00 is raised to 1.20 $/kWh (moderate; a severe level of
2.50 $/kWh is also configured). Each date is simulated twice, as a baseline
day and a CPP day, and only the tariff differs between them.

## 10. Harness (`phase3.py`, `experiments/run_phase3.py`)

- `Phase3Runner.run_pair(household, agent, date, seed, scale)` is the unit
  run: a baseline day plus a CPP day, memoised so no run is simulated or
  prompted twice.
- Steps: `elasticity` (all agents × households × pair), `overlay`
  (per-household and population load figures), and `seeds` (noisy-MILP
  42–44; the LLM uses seed 42 only, at T = 0). `calibration` and `scenario`
  exist but are disabled or excluded in the final runs.
- **Artifacts** (`phase3_report.finalize`): `curves.parquet` (per household ×
  agent × seed × day × step), both decision logs, and `personas_used.csv`.
  Every summary is derived from these: `summary.csv/json`, elasticity tables,
  `report.md`, overlays.
- **Controls:** `--dry-run` (stub LLM, `llm_projection.json`); a budget gate
  (a live run over more than 3 households needs `--approve-budget`); and
  sharding (`--households` subsets into separate `--out` and `--cache-path`,
  then `merge_shards.py`, identical to an unsharded run per
  `tests/test_sharding.py`).

## 11. Metrics

Per agent, on the population (sum of 24 households):

- peak kW and time of peak, and PAR;
- **post/CPP ratio**: 20:00–20:30 kW ÷ 17:00–20:00 mean kW on the CPP day
  (rebound);
- **degree-hours above the ceiling** (Σ max(0, T − ceiling) × 0.5 h) and max
  exceedance (comfort cost);
- **arc elasticity** of 17:00–20:00 energy, CPP vs base day:
  `((Q_cpp − Q_base)/Q_base) / ((1.20 − 0.55)/0.55)`, pooled over households.
  It is reported for the HVAC (flexible) layer and the whole household; the
  midpoint method is reported alongside;
- daily flexible-kWh change.

The whole-household elasticity is compared with Conway (2020), −0.026 to
−0.47. This is a validity check only, because base-load dilution makes it
weakly falsifiable.

## 12. Post-run analysis (offline, no API calls)

- `analyze_final_v1.py` replays every run offline from the cache. The client
  is patched to raise on any cache miss or write. It asserts exact equality
  with the saved curves, logs and summary, then computes:
  - H1 (WTP by orientation at matched °C over the ceiling, household-cluster
    bootstrap CIs);
  - per-orientation elasticity and comfort;
  - switch-on timing;
  - WTP resolution and monotonicity;
  - the **isotonic counterfactual** (the LLM's own WTP made monotone per
    orientation);
  - with `--level-matched`, a rule-based agent refitted to that run's own
    LLM WTP.
- `analyze_cross_model.py`: compares the two models (exploratory).
- `analyze_noise_test.py`: is the LLM's advantage reproducible by adding its
  own residuals to the level-matched rule, i.i.d. or household-shuffled, over
  50 seeds? Exploratory.
- `make_thesis_figures.py`: the thesis figures and booktabs tables, with
  every value asserted against the saved analysis. It is deterministic
  (`--check`).

## 13. Stack and environment

Python 3.12.4. Pinned in `requirements-lock.txt`: pulp, pandas, numpy,
pyarrow, pydantic, matplotlib, pyyaml, google-genai, matilda-client, pytest,
ruff. API keys go in `llm-hems/.env` (gitignored): `GEMINI_API_KEY`,
`MAINCODE_API_KEY`.

## 14. Departures from the original brief

| Original plan | As built | Why |
|---|---|---|
| Three agents | Four: a rule-based satisficer was added | A no-LLM control through the same halt. Without it, "the LLM satisfices" cannot be separated from "any threshold rule on temperature does". |
| Price/persona wording in the prompt | Price-blind prompt; comfort orientation is the only persona in the prompt; cost sensitivity is a Python factor | Price wording did not move WTP (live slice), comfort wording did (probe v3), and price in the prompt contaminated the elasticity. |
| Calibrate WTP scale on a subset, report on held-out | `global_scale` fixed at 1.0 a priori; calibration disabled | The pre-registration takes the LLM's dollar WTP literally, which removes a tuned parameter. |
| HVAC + washer | HVAC only | The washer's valuation is deadline-driven, with zero price response by construction (pilot). This is reported as a finding. |
| Persona-linked comfort bands | Fixed ±2.0 °C for all | The band-confound ablation: persona bands confound orientation with physics. |
| Homogeneous thermal physics | Seeded per-household R, C, initial temperature | So staggering is not an artefact of identical houses. |
| Primary + ablation model, three seeds | Two pinned Gemini models (final run + Arm A), T = 0 + seed 42, one LLM seed; noisy-MILP three seeds | T = 0 + seed is deterministic; the cache is the record. |
| Matilda or other providers | Gemini only for the final runs | Reliability battery (state misreads; no model, temperature or seed control). |
| Discriminating washer scenario (§12 of the brief) | Implemented (`run_scenario`), excluded from the final runs | The washer is excluded. |
| Phase 4 population loop (optional) | The final runs are the population runs (24 households) | — |

## 15. Open items (placeholders still marked `CONFIRM` in config)

- Retail $/kWh rates on top of the Ausgrid windows, and the CPP severities.
- Appliance nameplate and RC parameters (2 kW, COP 3, R, C). A uniform 2 kW
  AC is imposed on every household, including those without air
  conditioning. This is a pre-registered limitation.
- Single simulated day, synthetic weather, no demand feedback on the tariff
  (pre-registered limitations).
