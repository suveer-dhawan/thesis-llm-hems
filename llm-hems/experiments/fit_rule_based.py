"""Level-match the rule-based satisficer's k and w_max to the LLM's WTP.

Least squares through the origin (`rule_based_agent.fit_through_origin`) of
RAW LLM WTP (before global_scale / archetype factor) on each rule's
regressor, one row per DISTINCT prompt -- each distinct state counts once,
not once per trajectory that happened to visit it (visit counts depend on
the halt and on global_scale, not on the LLM's valuation).

  HVAC k[orientation]  <- `results/phase3/comfort_sweep/live/sweep_llm_log.parquet`
      (the live comfort sweep: 6 households, 2 per orientation, each at its
      own persona band, 14:00-21:00, all scales and both conditions). The
      regressor is max(0, indoor - band ceiling); indoor is parsed from the
      prompt, the ceiling is the logged `band_hi_c`.
  washer w_max         <- `results/phase3/live_slice/llm_decision_log.parquet`
      The comfort sweep is HVAC-only (washer never started), so it holds no
      washer WTP. The live slice (2 households, evening window, seed 42) is
      the only live washer WTP there is. Its washer prompt still carried the
      since-removed price-sensitivity persona line; the washer prompt is
      otherwise the current one. The regressor is the linear urgency
      1 - slack / slack_0, with slack parsed from the prompt and slack_0 from
      the household's persona window.

Writes `results/phase3/rule_based/fit.json`. The values used by runs are the
ones frozen in `config/experiment.yaml` -> `rule_based:` (copied from this
output, rounded to 3 dp); this script only re-derives and reports them.

Usage (from llm-hems/):
  PYTHONPATH=src .venv/bin/python experiments/fit_rule_based.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hems.agents.rule_based_agent import RuleBasedParams, fit_through_origin, washer_urgency
from hems.environment.appliances import _clock_to_step
from hems.environment.persona import load_assignment

SWEEP_LOG = Path("results/phase3/comfort_sweep/live/sweep_llm_log.parquet")
SLICE_LOG = Path("results/phase3/live_slice/llm_decision_log.parquet")
OUT = Path("results/phase3/rule_based/fit.json")
WASHER_DURATION_STEPS = 4  # config/appliances.yaml; the regressor needs slack_0


def _stats(x: pd.Series, y: pd.Series, b: float) -> dict:
    resid = y - b * x
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n_distinct_prompts": int(len(x)),
        "rmse": round(float(np.sqrt((resid**2).mean())), 4),
        "r2_vs_mean": round(1 - float((resid**2).sum()) / ss_tot, 4) if ss_tot else None,
        "mean_llm_wtp_at_zero_regressor": (
            round(float(y[x == 0].mean()), 4) if (x == 0).any() else None
        ),
    }


def fit_hvac(log: pd.DataFrame) -> dict:
    p = log[(log["appliance"] == "hvac") & ~log["forced"]].drop_duplicates("prompt").copy()
    indoor = p["prompt"].str.extract(r"indoor temperature \(C\)\s+: ([\d.]+)")[0].astype(float)
    p["x"] = (indoor - p["band_hi_c"]).clip(lower=0)
    out = {}
    for orientation, g in p.groupby("comfort_orientation"):
        k = fit_through_origin(g["x"], g["wtp"])
        out[orientation] = {"k": round(k, 6), **_stats(g["x"], g["wtp"], k)}
    return out


def fit_washer(log: pd.DataFrame, assignment: pd.DataFrame) -> dict:
    p = log[(log["appliance"] == "washer") & ~log["forced"]].drop_duplicates("prompt").copy()
    a = assignment.set_index("customer_ID")
    slack = p["prompt"].str.extract(r"buffer before deadline\s+: (-?\d+)")[0].astype(int)
    p["x"] = [
        washer_urgency(
            {
                "released_step": _clock_to_step(a.loc[hid, "washer_release"]),
                "deadline_step": _clock_to_step(a.loc[hid, "washer_deadline"]),
                "duration_steps": WASHER_DURATION_STEPS,
                "slack_steps": s,
            }
        )
        for hid, s in zip(p["household_id"], slack)
    ]
    w = fit_through_origin(p["x"], p["wtp"])
    return {
        "w_max": round(w, 6),
        "households": sorted(int(h) for h in p["household_id"].unique()),
        **_stats(p["x"], p["wtp"], w),
    }


def main() -> None:
    assignment = load_assignment()
    hvac = fit_hvac(pd.read_parquet(SWEEP_LOG))
    washer = fit_washer(pd.read_parquet(SLICE_LOG), assignment)
    frozen = RuleBasedParams.from_yaml()
    result = {
        "method": "least squares through the origin, raw LLM WTP on the rule regressor, "
        "one row per distinct prompt",
        "hvac": {"source": str(SWEEP_LOG), "by_orientation": hvac},
        "washer": {"source": str(SLICE_LOG), **washer},
        "frozen_in_config": {"hvac_k": frozen.k_by_orientation, "washer_w_max": frozen.w_max},
    }
    result["config_matches_fit_to_3dp"] = all(
        round(hvac[o]["k"], 3) == frozen.k_by_orientation.get(o) for o in hvac
    ) and round(washer["w_max"], 3) == frozen.w_max
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
