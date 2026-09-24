"""Fit the global WTP scale on a calibration subset; report held-out elasticity.

Responsibility (BUILD_SPEC.md §10, §13 Phase 3): split the households into
calibration vs held-out, stratified by burden archetype; fit ONE global
`wtp_scale` on the calibration subset so its pooled whole-household
short-run elasticity lands inside Conway 2020's range (-0.026..-0.47) and
closest to the pre-registered central target; freeze it; report elasticity
(LLM plus the MILP / noisy-MILP discriminant) on the held-out subset.

The pre-registered target, grid, split fraction and seeds live in
`config/experiment.yaml` -> `phase3.calibration` and are echoed into the
output (`calibration.json`, `report.md`). This is `run_phase3.py --steps
calibrate,elasticity`; same budget gate, same disk cache.

Usage:
  python experiments/calibrate.py --dry-run                 # zero spend
  python experiments/calibrate.py --approve-budget          # live, all 24 (after budget sign-off)
"""

from __future__ import annotations

import sys

import run_phase3

if __name__ == "__main__":
    if "--steps" not in sys.argv:
        sys.argv += ["--steps", "calibrate,elasticity"]
    if "--out" not in sys.argv:
        sys.argv += ["--out", "results/phase3/calibration"]
    run_phase3.main()
