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

Scaffolding only. `environment/*`, `agents/{milp,noisy_milp,llm}_agent.py`,
`llm/*`, and `analysis/*` are placeholder modules (docstring only — no tariff
math, appliance physics, MILP, or LLM calls yet). `agents/base.py` (the
`DecisionAgent` interface) and `agents/stub_agent.py` (an agent that always
defers) are implemented so the loop can run end-to-end:

```bash
pip install -e .
python experiments/run_single.py --agent stub
```

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
