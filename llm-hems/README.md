# llm-hems

A discrete-time (30-min) simulator of a single household's appliance scheduling
under a dynamic electricity tariff. A deterministic Python core owns all
physics and arithmetic (appliance state, energy, cost); a pluggable decision
agent decides, each interval, whether to run each controllable appliance.

Three agents are compared:

- **MILP** — a hyper-rational, PuLP-based cost-minimiser.
- **Noisy-MILP** — the same MILP with seeded per-household heterogeneity
  (jittered comfort bands, setpoints, deadlines), a strawman for "does trivial
  randomness alone explain staggering and bounded elasticity?"
- **LLM satisficing agent** (the thesis) — per appliance per interval, the LLM
  returns a willingness-to-pay (WTP) for running the appliance now; Python
  computes `U = wtp - cost_now` and applies a deterministic halt rule
  (`run iff U >= 0`). The LLM never does arithmetic.

The headline result: the LLM's induced price elasticity should fall in the
empirical human range (Conway 2020, Australia: −0.026 to −0.47) while avoiding
the synchronised peak the MILP produces at scale.

See [`BUILD_SPEC.md`](../BUILD_SPEC.md) for the full implementation brief and
`decision-memo-semester2.md` for the framing behind it. Values marked
`CONFIRM` throughout the config are sensible placeholders that must be checked
against a real source (Ausgrid tariff, appliance nameplate data, current model
IDs) before results are reported.

## Design principles

1. **All numbers live in Python.** The LLM never does arithmetic — it returns
   a subjective valuation; Python computes cost, compares, and decides.
2. **Agents are interchangeable behind one interface.** The simulator loop
   doesn't know which agent it's running; swapping MILP ↔ LLM is one config
   line.
3. **Everything is config-driven and seeded.** No magic numbers in source.
   `config/experiment.yaml` fully determines a run; every run writes its
   resolved config + seed into its results folder.
4. **Every decision is logged.** For the LLM agent: prompt, WTP, reasoning,
   computed cost, and run/defer decision — every appliance, every interval.

## Repository layout

```
config/          appliances.yaml, tariff.yaml, household.yaml, experiment.yaml
data/            raw/ (SGSC CSVs) and processed/ (cached parquet) — gitignored
src/hems/
  environment/   tariff, appliance physics, household state, simulator loop
  agents/        DecisionAgent interface + MILP / noisy-MILP / LLM / stub agents
  llm/           provider-agnostic client + pydantic I/O schemas
  analysis/      metrics, elasticity estimator, plots
experiments/     run_single.py, run_population.py, calibrate.py
results/         timestamped run folders — gitignored
tests/           pytest suite mirroring src/hems/environment + analysis
```

## Status

Phases 0-2 implemented: tariff/appliance/household physics, the simulator
loop, `stub`/`milp`/`noisy_milp`/`llm` agents, and analysis/plots all run
end-to-end. `analysis/elasticity.py` and the calibration split (Phase 3) are
still pending.

```bash
pip install -e .
python experiments/run_single.py --agent stub
python experiments/run_single.py --agent llm            # calls the live API
python experiments/run_single.py --agent llm --dry-run  # no API spend
```

## Reproducibility: the LLM agent

`llm/client.py` is a provider-agnostic client with two backends, selected by
`llm.provider` in `config/experiment.yaml` (`--provider` overrides it per
run):

- **matilda** -- Maincode's Matilda API, via the official `matilda-client`
  Python SDK (**v0.3.0**, confirmed working **2026-09-06**). Matilda owns
  model and provider routing internally -- its `chat.create_object` call
  exposes no `model`, `temperature`, or `seed` parameter, so **neither the
  model version nor the sampling behaviour behind a call is something this
  codebase can pin or reproduce**. Two calls with byte-identical prompts are
  not guaranteed to return the same `wtp`/`reason` (see the run-to-run
  variance check in `experiments/phase2_diagnostics.py`).
- **gemini** -- Google's `google-genai` SDK (**v2.22.0**) against the Gemini
  Developer API. Model id is pinned and configurable
  (`llm.gemini.model`, default `gemini-3.5-flash-lite` -- see the CONFIRM
  note in `config/experiment.yaml` for why the flagship `gemini-3.6-flash`
  isn't the default: its free tier caps at 20 requests/day per
  project+model, which the reliability battery below exhausts by itself).

Every `LLMAgent` run records `provider` + `model_id` + `sdk_version` per
decision-log row and in a `llm_meta.json` alongside the run's other outputs,
so a result can always be traced back to the exact backend that produced it.
For Matilda this is still just a fixed placeholder string (`model_id` is
genuinely unavailable from the API); for Gemini it's the real pinned model
id. Either way, anything reported from this agent should be read as *a*
sample from the provider's current model, not a fixed, replayable
computation -- `llm/client.py`'s disk cache (keyed on `(provider, prompt)`)
makes a single run's results replayable, but a fresh (uncached) call is not
guaranteed to match a prior one.

### Model reliability battery (2026-09-06)

Before any Phase 3 (elasticity) work, `experiments/phase2_diagnostics.py`
found the LLM agent misreading its own input state in 3 of 4 controlled
calls -- wrong comfort band, wrong indoor temp, an indoor/outdoor
temperature mix-up, and an invented active CPP event when `is_cpp=no` was
stated explicitly. In response: the prompt was hardened (state presented as
a labelled block, one explicit comfort fact, an unambiguous CPP line,
outdoor temperature dropped entirely), the response schema now requires the
model to echo back the state it says it's acting on
(`HVACWTPResponse`/`WasherWTPResponse` in `llm/schemas.py`), and
`experiments/reliability_battery.py` runs a 26-call battery (18 synthetic
HVAC state-reading contexts + the 8-call WTP price-invariance re-check) on
both providers head-to-head. Results in
`results/reliability_battery/battery_results.json`:

| | Matilda | Gemini (`gemini-3.5-flash-lite`) |
|---|---|---|
| model id | not exposed by the API | `gemini-3.5-flash-lite` |
| sdk version | `matilda-client==0.3.0` | `google-genai==2.22.0` |
| misread rate (26 calls) | **38.5%** (10/26) | **0%** (0/26) |
| fields that failed | indoor_temp_c 16.7%, comfort-band lower bound 16.7%, CPP flag 16.7% (state battery) | none |
| HVAC WTP price-invariance range | $0.00 | $0.10 |
| washer WTP price-invariance range | $2.00 (0 → 2.00 → 0 → 0 across the 4 prices -- non-monotonic, not a small residual) | $0.45 |

**Recommendation: Gemini (`gemini-3.5-flash-lite`) as the primary model for
Phase 3.** Matilda's misreads aren't random noise scattered across
unimportant fields -- they cluster on the comfort-band edge and the CPP flag,
which is exactly the state the discriminating scenario (BUILD_SPEC.md §12)
and the elasticity estimator depend on getting right. Matilda's washer WTP
also isn't price-invariant in practice ($2.00 swing on one price point),
which cuts against the core "the LLM never does the cost comparison"
assumption the halt rule relies on. Gemini had zero misreads across the
battery and a much tighter price-invariance range on both appliances, and
(unlike Matilda) its model id is actually pinned and reportable, which
matters for a thesis result that needs to be traceable to a specific model.
Caveat: `gemini-3.5-flash-lite` was used in place of the flagship
`gemini-3.6-flash` purely because of the latter's 20-req/day free quota, not
because it was chosen as materially better than the flagship -- both should
be spot-checked again before Phase 3 if this codebase later moves to a paid
tier or a multi-day run budget.

## Phase plan

| Phase | Scope | Definition of done |
|---|---|---|
| 0 | Ingest SGSC, select representative households, freeze config | `run_single.py --agent stub` completes on a real SGSC household and writes a load curve |
| 1 | MILP + noisy-MILP on one household | MILP cost ≤ noisy-MILP ≤ (expected) LLM cost; unit tests pass |
| 2 | LLM agent (primary + ablation model), both appliances | Decision log shows WTP + reason + U + decision for every step |
| 3 | Calibrate WTP on a subset; elasticity on held-out; four-way overlay; multi-seed spread; discriminating scenario (§12) | Held-out elasticity printed against the Conway range; four-way overlay figure produced |
| 4 (droppable) | Loop over N≈10–30 households → aggregate PAR / peak / rebound-peak | Aggregate PAR for MILP vs LLM over the population |

## Stack

Python 3.11+, `pulp`, `pandas`, `numpy`, `pydantic`, `matplotlib`, `pyyaml`,
`pytest`, `ruff`, plus LLM provider SDKs behind `llm/client.py`. See
[`requirements.txt`](requirements.txt).

## Open items before reporting results

- Exact current LLM model IDs (primary + ablation).
- Real Ausgrid residential TOU numbers + a defensible CPP schedule.
- Appliance nameplate params (HVAC kW/COP, RC values, washer kW/duration).
- Final population size N.
