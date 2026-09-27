# llm-hems

A 30-minute, discrete-time simulator of household air-conditioning under a
time-of-use tariff with a critical-peak-pricing (CPP) event. It asks one
question: **if an LLM supplies only a price-blind, persona-conditioned comfort
valuation, and Python applies the cost and a myopic "good-enough" halt, do the
resulting households respond to price like people do? And how do their
post-event rebound and comfort cost compare with optimising and rule-based
controllers?**

- The LLM returns a willingness-to-pay (WTP, $) for running the AC this
  half-hour. It never sees a price and never does arithmetic.
- Python computes `cost_now` and runs the appliance iff
  `WTP × global_scale × archetype_factor ≥ cost_now`.
- The same physics, tariff and households are run under four agents: MILP
  (perfect-foresight cost minimiser), noisy-MILP (seeded heterogeneity),
  a rule-based satisficer (WTP = k[orientation] × °C over the ceiling) and
  the LLM satisficer.

The architecture, agent contracts and metrics are in
[`../BUILD_SPEC.md`](../BUILD_SPEC.md). The pre-registration of the final run,
with hypotheses H1–H4, limitations and deviations, is in
[`docs/prereg_final_run_v1.md`](docs/prereg_final_run_v1.md).

## Final runs

| Run | Model | Tag (code) | Results | Analysis |
|---|---|---|---|---|
| **Final run v1** (pre-registered) | `gemini-3.5-flash-lite` | `final-run-v1` → `560d7fb` | `results/phase3/final_v1/` | `results/phase3/final_v1/analysis/analysis.md` |
| **Arm A** (pre-registered optional arm; model changed) | `gemini-3.8-flash` | `final-run-v1-arm-a` → `937907b` (run from `5851b05` on `arm-a`, same code) | `results/phase3/arm_a/` | `results/phase3/arm_a/analysis/analysis.md` |
| Cross-model comparison (exploratory) | both | `main` | `results/phase3/cross_model/` | `cross_model.md`, `noise_test.md` |
| Thesis figures and tables | both | `main` | `results/thesis_figures/` | `captions.md` |

Both runs use the same design: 24 SGSC households, HVAC only, a fixed ±2.0 °C
band around 24 °C, 2013-01-18, and a CPP event from 17:00 to 20:00
(0.55 → 1.20 $/kWh). The LLM runs at T = 0 with seed 42, and every response is
cached. Every results folder is listed with its commit and status
(final / pilot / ablation / superseded) in
[`results/INDEX.md`](results/INDEX.md).

## Reproducing the runs

There are two ways to reproduce the runs.

1. **Exact offline replay. No API calls; this is the recommended check.**
   The committed caches (`results/phase3/llm_cache_final.jsonl`, 978 entries;
   `results/phase3/llm_cache_arm_a.jsonl`, 1,034 entries) hold every LLM
   response the runs used. The analysis re-simulates every household, agent
   and day from the code. The LLM client is patched to raise on any cache miss
   or cache write. The replay then asserts that it reproduces `curves.parquet`,
   both decision logs and `summary.csv` **exactly**, before it derives anything.

   ```bash
   cd llm-hems
   PYTHONPATH=src .venv/bin/python experiments/analyze_final_v1.py        # final run v1
   PYTHONPATH=src .venv/bin/python experiments/analyze_final_v1.py \
       --run-dir results/phase3/arm_a --cache-path results/phase3/llm_cache_arm_a.jsonl \
       --cache-lines 1034 --label arm_a --level-matched                    # Arm A
   PYTHONPATH=src .venv/bin/python experiments/analyze_cross_model.py     # cross-model tables
   PYTHONPATH=src .venv/bin/python experiments/analyze_noise_test.py      # noise test
   PYTHONPATH=src .venv/bin/python experiments/make_thesis_figures.py --check
   ```

   `make_thesis_figures.py --check` builds everything twice and requires
   byte-identical output.

2. **Re-launch from the tag, exactly as the run was launched.** The launch
   scripts are archived in the results folders. They assume a separate clean
   clone at `$HOME/thesis-run/llm-hems`, and their pre-flight checks abort
   without side effects if the tree, cache size or output folder don't match.
   New LLM calls are billed and, although they run at T = 0 with a fixed
   seed, they are not guaranteed to match the cached responses. That is why
   the cache is the record.

   **Pinned environment:** Python 3.12.4 and `requirements-lock.txt`. The lock
   file was committed in `ea45192`, after the `final-run-v1` tag, with no code
   change. Prereg clarification 6 explains why. Keep the lock file outside the
   clone so the tree stays clean.

   ```bash
   git clone https://github.com/suveer-dhawan/thesis-llm-hems.git ~/thesis-run && cd ~/thesis-run/llm-hems
   git show main:llm-hems/requirements-lock.txt > ~/requirements-lock.txt
   python3.12 -m venv .venv                      # 3.12.4
   .venv/bin/pip install -r ~/requirements-lock.txt
   .venv/bin/pip install -e . --no-deps
   echo "GEMINI_API_KEY=..." > .env              # gitignored
   ```

   *Final run v1.* Check out the tag. At the tag, the cache holds the 191
   pilot entries the pre-flight expects.

   ```bash
   git checkout final-run-v1
   git show main:llm-hems/results/phase3/final_v1/launch_final_v1.sh > ~/launch_final_v1.sh
   bash ~/launch_final_v1.sh
   ```

   *Arm A.* Arm A was launched from branch `arm-a` at `5851b05`. There,
   `src/`, `experiments/` and `config/` are identical to tag
   `final-run-v1-arm-a` (the pre-flight checks this). That commit also holds
   the 167-entry pilot cache and the observe-only usage wrapper.

   ```bash
   git checkout -b arm-a 5851b05
   git show main:llm-hems/results/phase3/arm_a/launch_arm_a.sh > ~/launch_arm_a.sh
   bash ~/launch_arm_a.sh
   ```

   To re-launch **without any live calls**, run the same command with
   `--cache-path` pointing at a copy of the full cache from `main`. Every
   prompt then hits the cache. You have to skip the pre-flight's cache-size
   check, because the full cache is larger than the 191 or 167 entries it
   expects.

Each run folder records `provenance.json` (commit, dirty flag, branch, argv,
Python version), `config_resolved.yaml`, `run_meta.json`, the launch script
and the log.

## Repository layout

```
config/        tariff, appliances, household, persona, experiment.yaml (one file fixes a run)
data/          raw/ (SGSC CSVs, gitignored); processed/ (24-household parquet, committed)
docs/          prereg_final_run_v1.md
src/hems/
  environment/ tariff, RC thermal model + washer, household state, personas, simulator loop
  agents/      DecisionAgent interface; stub, MILP, noisy-MILP, rule-based, LLM agents; prompts/
  llm/         provider-agnostic client (Gemini, Matilda), disk cache, pydantic schemas
  analysis/    metrics (PAR, peak), elasticity, plots
  phase3.py    final-run harness (agents × households × base/CPP day × seeds, sharding)
  phase3_report.py  per-household artifacts → summaries, report.md
  provenance.py     commit / dirty / branch / argv into every results folder
experiments/   run_phase3.py (final-run entry point), merge_shards.py, analyze_*.py,
               make_thesis_figures.py, build_personas.py, fit_rule_based.py, earlier phase scripts
results/       see results/INDEX.md
tests/         pytest suite (physics, tariff, agents, client, harness, sharding)
```

## Setup and tests

```bash
cd llm-hems
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python experiments/run_phase3.py --dry-run --out results/phase3/dryrun   # stub LLM, zero spend
```

On macOS with Python 3.12, `import hems` can fail inside the venv because the
editable-install `.pth` file carries the `hidden` flag. Either keep
`PYTHONPATH=src` or run
`chflags nohidden .venv/lib/python3.12/site-packages/__editable__.hems-0.1.0.pth`.

## Cost and safety controls

- The LLM cache is append-only JSONL. Its key is provider, model,
  temperature, seed, max output tokens, response-schema hash and the exact
  prompt, so a re-run with the same cache makes zero live calls. Entries
  record the model version the API returned.
- `--dry-run` swaps in a stub LLM and writes `llm_projection.json`, with the
  projected live calls against the cache.
- A live run over more than `live_slice.max_live_households` households
  refuses to start without `--approve-budget`.
- The response schema makes the model echo back the state it acted on. Echo
  mismatches are logged as misreads: 0 in both final runs.

## Model choice (2026-09-06 reliability battery)

`experiments/reliability_battery.py` ran 26 calls: 18 synthetic HVAC
state-reading contexts plus 8 price-invariance checks. The results are in
`results/reliability_battery/battery_results.json`. Matilda misread its input
state in 38.5% of calls (comfort band edge, indoor temperature, CPP flag), and
its API exposes no model id, temperature or seed. `gemini-3.5-flash-lite`
misread 0 of 26 and can be pinned, so it became the primary model.
`gemini-3.8-flash` was then run as pre-registered Arm A.
