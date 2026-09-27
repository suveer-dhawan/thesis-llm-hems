# results/ — index

Every results folder and file, with the code it came from and its status. Paths
are relative to `llm-hems/results/`.

**Status**

- **final**: reported in the thesis, or a direct input to what is reported.
- **pilot**: a small or dry-run check of the final configuration, made before
  the final run.
- **ablation**: a deliberate variant used to make a design decision. It is not
  a headline result.
- **superseded**: an earlier stage whose code or mechanism has since changed.
  It is kept for the record only.

**Code** is the commit that produced the folder: `provenance.json` where the
folder has one (✱), otherwise the commit that added it to git. **Updated** is
the last commit that touched the folder, where that differs. *Local only* means
the path is gitignored and exists only in the original working copy.

Tags: `final-run-v1` → `560d7fb`; `final-run-v1-arm-a` → `937907b`.

## Final

| Path | Status | Code | Updated | What it is |
|---|---|---|---|---|
| `phase3/final_v1/` | **final** | `560d7fb` ✱ (tag `final-run-v1`, clean tree) | `2cd17a1` (analysis/) | Final run v1, pre-registered in `docs/prereg_final_run_v1.md`. 24 households, HVAC only, fixed ±2.0 °C, gemini-3.5-flash-lite, T = 0, seed 42. Four agents (MILP, noisy-MILP seeds 42–44, rule-based, LLM); base and CPP day. Includes `launch_final_v1.sh`, `final_v1.log`, `provenance.json`, `config_resolved.yaml`. |
| `phase3/final_v1/analysis/` | **final** | `64c6567`, `2cd17a1` | | Post-run analysis (`experiments/analyze_final_v1.py`): H1–H4 tables, isotonic counterfactual, offline replay check (`replay_check.json`). Each table is labelled pre-registered or exploratory. |
| `phase3/arm_a/` | **final** (pre-registered optional arm A; see Deviations) | `5851b05` ✱ (branch `arm-a`; src/experiments/config = tag `final-run-v1-arm-a`) | `96b9d85` (analysis/) | Arm A: identical design, model gemini-3.8-flash, new cache. Includes `launch_arm_a.sh`, the observe-only usage wrapper, per-attempt `usage_arm_a.jsonl` and `arm_a.log`. |
| `phase3/arm_a/analysis/` | **final** | `96b9d85` | | Same analysis as final_v1, plus `--level-matched` (a rule-based agent refitted to this run's own LLM WTP; exploratory). |
| `phase3/cross_model/` | **final** (all exploratory, post-hoc) | `96b9d85` | `8d3c7f0` | `cross_model.md`: gemini-3.5-flash-lite vs gemini-3.8-flash (`analyze_cross_model.py`). `noise_test*`: context-vs-dispersion noise test (`analyze_noise_test.py`). |
| `thesis_figures/` | **final** | `main` (this commit) | | Figures 1–8 (PDF + 300 dpi PNG) and LaTeX tables 1–4 from `experiments/make_thesis_figures.py`, with `captions.md`. Deterministic (`--check`). |
| `phase3/llm_cache_final.jsonl` | **final** | `b2f053d` | `6679230` | LLM response cache for final run v1: 978 entries (191 from the pilot + the run). Offline replay reproduces the run exactly from it. |
| `phase3/llm_cache_arm_a.jsonl` | **final** | `c19e132` | | Arm A cache: 1,034 entries (167 from the pilot + the run). |
| `phase3/persona/` | **final** (input) | `34c563a` | | Energy-burden persona report behind `config/persona_assignment.csv`: archetypes A/B/C and comfort orientation. |
| `data_profile/` | **final** (input) | `8622d73` | | Phase 0 SGSC profile: cluster shapes and representative curves. |
| `reliability_battery/` | **final** (model-selection evidence) | `6c979a2` | | 26-call Matilda-vs-Gemini state-reading and price-invariance battery that chose gemini-3.5-flash-lite. |

## Pilots and dry runs of the final configuration

| Path | Status | Code | Updated | What it is |
|---|---|---|---|---|
| `phase3/pilot_arm_a/` | pilot | `937907b` ✱ (tag `final-run-v1-arm-a`) | `c19e132` | Arm A live pilot: 3 households, elasticity step. Checked quota; 167 calls, 0 misreads, 0 truncations, 0 retries. Its responses seed `llm_cache_arm_a.jsonl`. |
| `phase3/pilot_live/`, `phase3/pilot_live.log` | pilot | `08bd170` ✱ (dirty) | `b2f053d` | Phase 3 live pilot: 3 held-out households, fixed ±2.0, all agents, washer on. Its responses seed `llm_cache_final.jsonl`. It showed the washer's zero price response, which is why the final run is HVAC only. |
| `phase3/pilot_dryrun/` | pilot | `08bd170` ✱ (dirty) | `b2f053d` | Stub-LLM dry run matching `pilot_live`. |
| `phase3/dryrun_hvac_only/`, `phase3/dryrun_hvac_only.log` | pilot | `08bd170` ✱ (dirty; committed as `1868556`) | | HVAC-only final-config dry run, all 24 households, stub LLM. Source of the call projection in the pre-registration. |
| `phase3/dryrun_final/` | superseded | `484b916` ✱ | `08bd170` | Final-config dry run with the washer still enabled. Replaced by `dryrun_hvac_only`. |

## Ablations (design decisions)

| Path | Status | Code | Updated | What it is |
|---|---|---|---|---|
| `phase3/comfort_sweep/live/` | ablation | `d649b48` | `e4a16e2` | Comfort-orientation sweep, live: 6 households, HVAC only, LLM window 14:00–21:00, global_scale 0.3/0.5/0.7/1.0. Checked the comfort-persona mechanism and the premise that WTP orders by persona. The persona-band rule-based k (`rule_based.hvac_k`) was fitted from its log. |
| `phase3/comfort_sweep/fixed_band_2/` | ablation | `c17ce62` | | Same sweep with the fixed ±2.0 band (prioritising and flexible re-run live; balanced reused because its band is already ±2.0), compared with the persona-linked bands in `live/comparison_vs_persona_band.md`. The frozen fixed-band k used in both final runs (3.791 / 0.620 / 0.660) was fitted from `live/sweep_llm_log.parquet` (`experiments/fit_rule_based.py`). |
| `phase3/comfort_sweep/projection/` | ablation | `1c7c3ad` | `e4a16e2` | Call projection for the sweep. |
| `phase3/comfort_sweep/fixed_band_2 2/`, `live 2/`, `projection 2/` | superseded (stray) | — | | Empty, untracked folders, probably Finder copies. They hold nothing and are safe to delete. |
| `phase3/rule_based/` | ablation | `b8573cd` | `85466bf` | Rule-based satisficer control (no LLM): k fit (`fit.json`), MILP / noisy-MILP / rule on 24 households, persona vs fixed bands; `thermal_het_fixed/`. |
| `phase3/llm_cache.jsonl` | superseded | `34c563a` | `c17ce62` | Cache used by the live slice and the comfort sweeps. |
| `20260919_*_probe_wtp_price{,_v2,_v3}.csv` | ablation (*local only*) | — | | WTP price-invariance probes that led to the price-blind prompt and to T = 0 + seed (`experiments/probe_wtp_price*.py`). |

## Superseded stages

| Path | Status | Code | Updated | What it is |
|---|---|---|---|---|
| `phase3/live_slice/`, `phase3/live_slice_rerun/` | superseded | `34c563a` | | Two-household live slice from the first Phase 3 harness, which used price-sensitivity wording in the prompt. It showed that the wording did not move WTP, which led to the comfort-orientation mechanism (`d649b48`). |
| `phase3/dryrun_smoke/` | superseded | `34c563a` | | First Phase 3 dry run: burden personas, with calibration still enabled. |
| `phase3/budget/` | superseded | `34c563a` | | Budget projection for the first Phase 3 harness. |
| `phase1/` | superseded | `3268384` | `089d1df` | Phase 1: MILP / noisy-MILP / stub on one household. It predates thermal heterogeneity and the MILP floor fix (`fe9e690`). Also holds the slide-6 chart. |
| `phase2/`, `phase2_dryrun/` | superseded (*local only*) | — | | Phase 2 single-household LLM agent (price in the prompt). |
| `phase2_diagnostics/` | superseded (*local only*) | — | | State-misread diagnostics that motivated the hardened prompt and the reliability battery. |
| `llm_cache.json` | superseded (*local only*) | — | | Phase 2 cache, keyed on (provider, prompt). |
