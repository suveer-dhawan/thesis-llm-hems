# Pre-registration — Final Run v1

**Written:** 2026-09-26, before the final run. **Code:** tag `final-run-v1` (exact commit recorded in the run's `provenance.json`; the run must start from a clean tree).
Any change after tagging → new tag + entry under Deviations.

## Research question
Does a decoupled architecture (LLM supplies a price-blind, persona-conditioned comfort valuation; Python applies costs and a myopic satisficing halt) produce human-range price response, and how does its post-CPP rebound and comfort cost compare with optimising and rule-based agents?

## Fixed design
- **Households:** 24 SGSC households (existing stratified sample). No calibration split — `global_scale` fixed a priori at 1.0 (LLM dollar WTP taken literally).
- **Appliance:** HVAC only (2 kW, COP 3). Washer excluded — see Limitations.
- **Physics:** first-order RC; per-household thermal heterogeneity R, C × U(0.8, 1.2), initial temp = setpoint ± U(1.0) °C, seeded. Synthetic summer outdoor curve (22–32 °C, peak 15:00). Base load: real SGSC profile, 2013-01-18.
- **Comfort band:** fixed ±2.0 °C around 24 °C for all households (primary). Persona-linked bands were tested in the band-confound ablation (Prompt C) and are not run here.
- **Tariff:** TOU peak 0.55 $/kWh; CPP day raises 17:00–20:00 to 1.20 $/kWh. Days: baseline + CPP.
- **Agents:**
  1. MILP — perfect foresight of the day's tariff and weather (stated asymmetry).
  2. Noisy-MILP — seeds 42, 43, 44.
  3. Rule-based satisficer — WTP = k[orientation] × °C over ceiling; k = 3.791 / 0.620 / 0.660 (prioritising / balanced / flexible), fitted to LLM WTP from the comfort-sweep logs before this run.
  4. LLM satisficer — gemini-3.5-flash-lite, T = 0, seed 42; price-blind prompt; one call per decision.
- **Halt (both satisficers):** run iff WTP × 1.0 × factor(A/B/C = 1.3/1.0/0.7) ≥ cost_now.
- **Washer:** disabled via `phase3.washer_enabled: false` (zero-length cycle; no prompts, no load). Washer scenario step excluded.
- **Projected live calls (dry run, commit 1868556):** 893 new (2,304 HVAC decisions; 1,016 distinct prompts, 123 already cached from the pilot); hard ceiling 2,304. Est. ~24 min single process, ~$0.25 in counted tokens (thinking tokens excluded). Budget cap $8.

## Metrics (per agent)
Peak kW + time; PAR; 20:00–20:30 load vs CPP-window mean; degree-hours above ceiling + max exceedance; flexible-layer (HVAC) elasticity; whole-household elasticity; daily HVAC kWh change.

## Hypotheses (every outcome is reported)
- **H1 — Persona valuation.** At matched indoor temperatures above the ceiling, LLM raw WTP orders prioritising > balanced ≥ flexible.
- **H2 — Validity.** Whole-household elasticity falls within Conway (2020) range (−0.026 to −0.47). Treated as a validity check only: base-load dilution makes it weakly falsifiable.
- **H3 — Rebound.** LLM post-CPP load (20:00–20:30, absolute and ratio to the CPP window) is lower than MILP's. Any satisficer rebound is interpreted as reactive snapback (accumulated discomfort), MILP's as anticipatory (pre-cooling + synchronised timing).
- **H4 — LLM vs rule-based.** No directional prediction. If equivalent on rebound and elasticity, report "matches a level-matched rule without hand-tuned parameters." Differences in comfort cost, ordering and WTP monotonicity reported descriptively.
- **Not predicted:** persona ordering of *elasticity* (the pilot showed it depends on where coarse WTP levels fall relative to peak/CPP costs).

## Analysis rules
No household exclusions. Parse failures and retries counted and reported. Per-persona results reported with n (imbalanced: 5 / 8 / 11) and bootstrap CIs over households.

## Known limitations (stated in advance)
Single simulated day; synthetic weather; uniform 2 kW AC imposed on all households (including no-aircon); fixed tariff with no demand feedback; MILP foresight vs satisficer myopia; coarse and sometimes non-monotone LLM WTP; comfort-sweep households overlap the 24 (they informed the band decision and rule-based k fit, not any tuned parameter); pilot responses (3 households) are reused from the cache; washer excluded because the pilot showed deadline-driven valuation with zero price response by construction — reported as a finding.

## Optional arms (pre-registered, reported separately)
- **A.** Second model (Gemini 3.6 or 3.8 Flash), identical config.
- **B.** 24 additional households (pool: 1,190 eligible, none reused), drawn with the original cluster × income stratification, no solar; dwelling matched where available (relax dwelling first if a cell is empty). Archetype (burden) cut-offs **held fixed at the current 24's values** so no existing household changes archetype. Identical config; reported both separately and pooled (N = 48).

## Pre-run clarifications (wording only; code at tag unchanged)
1. "One call per decision" means one valuation per decision; identical prompts are served from the deterministic cache (T = 0), so 2,304 decisions ≈ 893 live calls.
2. Retries are derived as call attempts − new cache entries. A call failing after 3 attempts halts the run; a restart from the same tag resumes from the cache with identical results. Any such restart is logged here.
3. Per-orientation results with n and bootstrap CIs are post-run analysis on the saved per-household outputs (the harness summarises by archetype).
4. The $8 cap is a billing limit, not enforced in code; expected spend ≈ $0.25–1.
5. Arms A and B each require a config/code change and therefore a new tag and a Deviations entry. Arm B needs the current 24's burden tercile cut-offs entered as fixed thresholds. Arm A requires confirming the second model's paid-tier quota first.
6. The run uses the pinned pilot environment (requirements-lock.txt; Python 3.12.4). requirements.txt at the tag omitted pyarrow and was unpinned; results do not depend on this.
7. Live mode defaults to a 2-household slice when --households is omitted; the final run passes all 24 IDs and all settings explicitly via ~/launch_final_v1.sh (copied into the results folder after the run). An initial launch on 2026-09-26 was aborted before producing results because of this default; its 3 cache entries were discarded and the cache restored to 191 entries.

## Deviations
- Arm A: model changed to gemini-3.8-flash; new cache; all else identical to final-run-v1.
