#!/usr/bin/env bash
# Arm A (docs/prereg_final_run_v1.md, Deviations) -- live, HVAC-only, fixed +/-2.0, 24 households,
# model gemini-3.8-flash. Identical to ~/launch_final_v1.sh except model/cache (config at tag
# final-run-v1-arm-a) and the observe-only usage wrapper (logs per-attempt token usage,
# finish_reason, latency; no behavioural change). Every setting that differs between
# --dry-run and live is passed explicitly.
set -euo pipefail

REPO="$HOME/thesis-run/llm-hems"
cd "$REPO"

OUT=results/phase3/arm_a
CACHE=results/phase3/llm_cache_arm_a.jsonl
WRAPPER=results/phase3/pilot_arm_a/pilot_wrap.py   # committed copy used for the pilot
USAGE="$HOME/arm_a_usage.jsonl"
LOG="$HOME/arm_a.log"

# --- preflight: abort (no side effects) unless the tree matches the pre-registration ---
[ "$(git branch --show-current)" = "arm-a" ] || { echo "not on branch arm-a" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "working tree not clean" >&2; exit 1; }
git diff --quiet final-run-v1-arm-a HEAD -- src experiments config || { echo "src/experiments/config differ from tag final-run-v1-arm-a" >&2; exit 1; }
[ "$(wc -l < "$CACHE" | tr -d ' ')" = "167" ] || { echo "cache is not at 167 lines" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "$OUT already exists" >&2; exit 1; }
[ -f "$WRAPPER" ] || { echo "wrapper $WRAPPER missing" >&2; exit 1; }
[ ! -e "$USAGE" ] || { echo "$USAGE already exists (wrapper appends)" >&2; exit 1; }

# --- launch (through the observe-only usage wrapper; first arg = usage log path) ---
PYTHONPATH=src "$REPO/.venv/bin/python" "$WRAPPER" "$USAGE" \
  --config config/experiment.yaml \
  --households 8201999,8208263,8211127,8395524,8456189,8463253,8481835,8568499,8596070,8619039,8632083,8638181,9012382,9911140,9911204,9911250,9911254,10042536,10239118,10278838,10444433,10556929,10649333,11105177 \
  --steps elasticity,overlay,seeds \
  --band-mode fixed \
  --cache-path "$CACHE" \
  --out "$OUT" \
  --approve-budget \
  2>&1 | tee "$LOG"

# --- archive run inputs/telemetry next to the results ---
cp "$WRAPPER" "$OUT/usage_wrap.py"
cp "$USAGE" "$OUT/usage_arm_a.jsonl"
cp "$LOG" "$OUT/arm_a.log"
cp "$0" "$OUT/launch_arm_a.sh"
echo "[launch_arm_a] done -- artifacts archived in $OUT"
