#!/usr/bin/env bash
# Final run v1 (docs/prereg_final_run_v1.md) -- live, HVAC-only, fixed +/-2.0, 24 households.
# Every setting that differs between --dry-run and live is passed explicitly.
set -euo pipefail

REPO="$HOME/thesis-run/llm-hems"
cd "$REPO"

# --- preflight: abort (no side effects) unless the tree matches the pre-registration ---
[ "$(git rev-parse HEAD)" = "560d7fb316fc8f22f46e066a0d785002cee74a33" ] || { echo "HEAD is not final-run-v1" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "working tree not clean" >&2; exit 1; }
[ "$(wc -l < results/phase3/llm_cache_final.jsonl | tr -d ' ')" = "191" ] || { echo "cache is not at 191 lines" >&2; exit 1; }
[ ! -e results/phase3/final_v1 ] || { echo "results/phase3/final_v1 already exists" >&2; exit 1; }

# --- launch ---
PYTHONPATH=src "$REPO/.venv/bin/python" experiments/run_phase3.py \
  --config config/experiment.yaml \
  --households 8201999,8208263,8211127,8395524,8456189,8463253,8481835,8568499,8596070,8619039,8632083,8638181,9012382,9911140,9911204,9911250,9911254,10042536,10239118,10278838,10444433,10556929,10649333,11105177 \
  --steps elasticity,overlay,seeds \
  --band-mode fixed \
  --cache-path results/phase3/llm_cache_final.jsonl \
  --out results/phase3/final_v1 \
  --approve-budget \
  2>&1 | tee "$HOME/final_v1.log"
