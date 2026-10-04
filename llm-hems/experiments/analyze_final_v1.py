"""Post-run analysis of Final Run v1 (docs/prereg_final_run_v1.md).

Reads the saved artifacts in results/phase3/final_v1/ (never writes there,
except into its `analysis/` subfolder) and regenerates, fully offline, what
they don't record:

  * The pre-decision indoor temperature of every LLM / rule-based decision.
    The decision logs carry the prompt (temperature rounded to 0.1 C) but not
    the exact state, and curves.parquet stores the *end-of-step* temperature.
    Every day is re-simulated from the tagged code with the LLM served from
    results/phase3/llm_cache_final.jsonl; the client is patched to raise on
    any cache miss or cache write, so no live call is possible. The replay is
    checked to reproduce curves.parquet, both decision logs and summary.csv
    exactly before anything is derived from it.
  * The counterfactual isotonic agent (item 7), which never existed in the run.

Every table is labelled "pre-registered" or "exploratory (post-hoc)".

    PYTHONPATH=src .venv/bin/python experiments/analyze_final_v1.py

Any other run of the same design (e.g. Arm A) is analysed with the same code:

    PYTHONPATH=src .venv/bin/python experiments/analyze_final_v1.py \
        --run-dir results/phase3/arm_a --cache-path results/phase3/llm_cache_arm_a.jsonl \
        --cache-lines 1034 --label arm_a --level-matched

The defaults reproduce final_v1's `analysis/` exactly. The model id is read from
the run's own `config_resolved.yaml`. For a run other than final_v1 the prose
summaries in analysis.md are generated from the data (final_v1 keeps its
hand-checked wording). `--level-matched` (exploratory) additionally refits the
rule-based k to this run's own LLM WTP and runs that agent offline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from hems.agents.base import Decision, DecisionContext  # noqa: E402
from hems.agents.llm_agent import LLMAgent, forced_washer_outcome, satisficing_halt  # noqa: E402
from hems.agents.rule_based_agent import RuleBasedParams, RuleBasedSatisficerAgent  # noqa: E402
from hems.environment.simulator import run_simulation  # noqa: E402
from hems.llm import client as llm_client  # noqa: E402
from hems.phase3 import AGENTS, DayRun, Phase3Config, Phase3Runner  # noqa: E402
from hems.phase3_report import agent_summary, curves_from_runs, logs_from_runs  # noqa: E402

# Set by configure() (defaults = final run v1).
RUN = Path("results/phase3/final_v1")
OUT = RUN / "analysis"
CACHE = Path("results/phase3/llm_cache_final.jsonl")
CACHE_LINES: int | None = 978
LABEL = "final_v1"
LEVEL_MATCHED = False
N_BOOT = 10_000
BOOT_SEED = 42
DT_H = 0.5
CPP_LO, CPP_HI = 34, 40  # 17:00-20:00
ORIENTATIONS = ["comfort_prioritising", "balanced", "comfort_flexible"]
ORI_SHORT = {"comfort_prioritising": "prioritising", "balanced": "balanced",
             "comfort_flexible": "flexible"}  # fmt: skip
ARCHETYPES = ["A", "B", "C"]
PRE = "pre-registered"
EXP = "exploratory (post-hoc)"

# Reference categorical palette (dataviz skill, slots 1-3 validated all-pairs);
# colour follows the entity: orientations and agents each keep a fixed hue.
ORI_COLOR = {"comfort_prioritising": "#2a78d6", "balanced": "#eb6834",
             "comfort_flexible": "#1baf7a"}  # fmt: skip
AGENT_COLOR = {"llm": "#2a78d6", "rule_based": "#eb6834", "milp": "#1baf7a",
               "isotonic": "#eda100", "rule_based_lm": "#e87ba4"}  # fmt: skip
AGENT_LABEL = {"llm": "LLM", "rule_based": "rule-based", "milp": "MILP",
               "isotonic": "isotonic (counterfactual)",
               "rule_based_lm": "rule-based, level-matched to this run"}  # fmt: skip
LM = "rule_based_lm"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------
# 0. offline replay
# --------------------------------------------------------------------------


class CacheMiss(RuntimeError):
    pass


def _no_live_calls() -> None:
    """Make any live LLM call (or cache write) raise."""
    orig = llm_client.LLMClient.complete_json

    def complete_json(self, prompt, schema):
        if not self.is_cached(prompt, schema):
            raise CacheMiss("cache miss during offline replay -- refusing a live call")
        return orig(self, prompt, schema)

    def refuse(*_a, **_k):
        raise CacheMiss("offline replay must not call the API or write the cache")

    llm_client.LLMClient.complete_json = complete_json
    llm_client._DiskCache.put = refuse
    llm_client._GeminiBackend.call = refuse
    llm_client._MatildaBackend.call = refuse


def _record_pre_decision_temp() -> None:
    """Log the indoor temperature each decision saw (NaN for washer decisions).

    Wraps `decide` / `log_to_dataframe` only to read the context; decisions
    are untouched (verified against the saved logs below).
    """
    for cls in (LLMAgent, RuleBasedSatisficerAgent):
        orig_decide, orig_df = cls.decide, cls.log_to_dataframe

        def decide(self, ctx, _orig=orig_decide):
            self.__dict__.setdefault("_pre_temp", []).append(
                ctx.appliance.get("indoor_temp_c", float("nan"))
            )
            return _orig(self, ctx)

        def log_to_dataframe(self, _orig=orig_df):
            df = _orig(self)
            df["pre_indoor_temp_c"] = self.__dict__.get("_pre_temp", [])
            return df

        cls.decide, cls.log_to_dataframe = decide, log_to_dataframe


def replay() -> tuple[Phase3Runner, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Re-simulate the whole run offline; return runner, curves, llm log, rule log, checks."""
    cache_hash = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    n_lines = len(CACHE.read_text().splitlines())
    assert CACHE_LINES is None or n_lines == CACHE_LINES, (
        f"cache has {n_lines} lines, expected {CACHE_LINES}"
    )
    _no_live_calls()
    _record_pre_decision_temp()

    resolved = yaml.safe_load((RUN / "config_resolved.yaml").read_text())
    cfg = Phase3Config.from_yaml("config/experiment.yaml", band_mode="fixed",
                                 cache_path=str(CACHE))  # fmt: skip
    # the model id is part of the cache key: take it from the run itself
    provider = resolved["llm"]["provider"]
    cfg.llm_cfg[provider]["model"] = resolved["llm"][provider]["model"]
    assert cfg.resolved()["phase3"] == resolved["phase3"], "config differs from the run's"
    assert cfg.resolved()["llm"] == resolved["llm"], "LLM config differs from the run's"
    runner = Phase3Runner(cfg)
    offline_client = {}

    def client(seed):  # dummy key: the backend is never reached
        if seed not in offline_client:
            offline_client[seed] = llm_client.build_llm_client(
                "gemini", cfg.llm_cfg, str(CACHE), seed=seed, api_key="offline-replay"
            )
        return offline_client[seed]

    runner.client = client
    ids = json.loads((RUN / "run_meta.json").read_text())["households"]
    for hid in ids:
        for agent in AGENTS:
            for seed in cfg.seeds_for(agent):
                for date in cfg.dates:
                    runner.run_pair(hid, agent, date, seed, cfg.global_scale)

    runs = runner.runs
    orientation = runner.assignment.set_index("customer_ID")["comfort_orientation"].to_dict()
    curves = curves_from_runs(runs, runner.archetype, orientation.get)
    llm_log = logs_from_runs(runs, "llm_log")
    rule_log = logs_from_runs(runs, "rule_log")
    checks = verify(curves, llm_log, rule_log, resolved)
    checks["cache_sha256_before"] = cache_hash
    checks["cache_sha256_after"] = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    checks["cache_unchanged"] = checks["cache_sha256_before"] == checks["cache_sha256_after"]
    checks["live_calls"] = sum(c.stats["live_calls"] for c in offline_client.values())
    checks["cache_hits"] = sum(c.stats["cache_hits"] for c in offline_client.values())
    assert checks["cache_unchanged"] and checks["live_calls"] == 0
    return runner, curves, llm_log, rule_log, checks


def _sort(df, cols):
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def verify(curves, llm_log, rule_log, resolved) -> dict:
    """Replay == saved run, exactly; raises otherwise."""
    from hems.phase3_report import CURVE_SORT, LOG_SORT

    saved_curves = _sort(pd.read_parquet(RUN / "curves.parquet"), CURVE_SORT)
    rc = _sort(curves, CURVE_SORT)[saved_curves.columns]
    pd.testing.assert_frame_equal(rc, saved_curves, check_exact=True, check_dtype=False)

    saved_llm = _sort(pd.read_parquet(RUN / "llm_decision_log.parquet"), LOG_SORT)
    rl = _sort(llm_log, LOG_SORT)
    same = ["t", "clock", "appliance", "prompt", "wtp", "reason", "cost_now", "utility", "run",
            "forced", "wtp_scale", "wtp_factor", "model_version", "response_timestamp",
            "household_id", "seed", "run_label"]  # fmt: skip
    pd.testing.assert_frame_equal(rl[same], saved_llm[same], check_exact=True, check_dtype=False)
    assert rl["misreads"].map(list).tolist() == saved_llm["misreads"].map(list).tolist()

    saved_rule = _sort(pd.read_parquet(RUN / "rule_decision_log.parquet"), LOG_SORT)
    rr = _sort(rule_log, LOG_SORT)[saved_rule.columns]
    pd.testing.assert_frame_equal(rr, saved_rule, check_exact=True, check_dtype=False)

    summ = agent_summary(curves, resolved)
    saved_summ = pd.read_csv(RUN / "summary.csv")
    for col in saved_summ.columns:
        a, b = summ[col].tolist(), saved_summ[col].tolist()
        if saved_summ[col].dtype.kind == "f":
            assert np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True), col
        else:
            assert [str(x) for x in a] == [str(x) for x in b], col
    return {
        "curves_rows": len(rc),
        "llm_log_rows": len(rl),
        "rule_log_rows": len(rr),
        "curves_identical": True,
        "llm_log_identical": True,
        "rule_log_identical": True,
        "summary_csv_identical": True,
        "replayed_llm_cached_share": float(rl["cached"].mean()),
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

DP = (1.20 - 0.55) / 0.55  # %dP of the CPP window (base-method arc elasticity)


def cluster_boot(n: int, stat, n_boot: int = N_BOOT) -> tuple[float, float]:
    """95% percentile CI of `stat(idx)` over households resampled with replacement.

    `stat` takes an (n_boot, n) index matrix and returns n_boot values. A fresh
    RNG (seed 42) per call keeps every CI independent of evaluation order.
    """
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    vals = stat(rng.integers(0, n, size=(n_boot, n)))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def boot_mean(values) -> tuple[float, float, float]:
    """Mean over households + cluster-bootstrap CI (one value per household)."""
    v = np.asarray(values, float)
    return float(v.mean()) if len(v) else float("nan"), *cluster_boot(
        len(v), lambda idx: v[idx].mean(1)
    )


def boot_ratio(num, den) -> tuple[float, float, float]:
    """sum(num)/sum(den) over households + CI (decision-weighted mean per bin)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if not len(num):
        return float("nan"), float("nan"), float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(num.sum() / den.sum()), *cluster_boot(
            len(num), lambda idx: num[idx].sum(1) / den[idx].sum(1)
        )


def pooled_e(qb, qc) -> float:
    qb, qc = float(np.sum(qb)), float(np.sum(qc))
    return (qc / qb - 1) / DP if qb else float("nan")


def boot_pooled_e(qb, qc) -> tuple[float, float, float, float]:
    """Pooled arc elasticity (sum Q, base method) + CI + share of undefined resamples."""
    qb, qc = np.asarray(qb, float), np.asarray(qc, float)
    if not len(qb):
        return (float("nan"),) * 4
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(qb), size=(N_BOOT, len(qb)))
    sb, sc = qb[idx].sum(1), qc[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(sb > 0, (sc / sb - 1) / DP, np.nan)
    lo, hi = (np.nanpercentile(e, [2.5, 97.5]) if np.isfinite(e).any() else (np.nan, np.nan))
    return pooled_e(qb, qc), float(lo), float(hi), float(np.isnan(e).mean())


def boot_diff(a_num, a_den, b_num, b_den) -> tuple[float, float, float]:
    """Difference of two decision-weighted means, households resampled within each group."""
    a_num, a_den, b_num, b_den = (np.asarray(x, float) for x in (a_num, a_den, b_num, b_den))
    if not len(a_num) or not len(b_num):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    ia = rng.integers(0, len(a_num), size=(N_BOOT, len(a_num)))
    ib = rng.integers(0, len(b_num), size=(N_BOOT, len(b_num)))
    d = a_num[ia].sum(1) / a_den[ia].sum(1) - b_num[ib].sum(1) / b_den[ib].sum(1)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(a_num.sum() / a_den.sum() - b_num.sum() / b_den.sum()), float(lo), float(hi)


def fmt(v, nd=3):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "–" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(v, nd) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def ci(m, lo, hi, nd=3) -> str:
    return "–" if np.isnan(m) else f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def clock(t: int) -> str:
    return f"{t // 2:02d}:{30 * (t % 2):02d}"


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------


def decisions(log: pd.DataFrame, curves: pd.DataFrame, personas: pd.DataFrame) -> pd.DataFrame:
    """HVAC decisions with the exact pre-decision state, orientation and archetype."""
    d = log[log["appliance"] == "hvac"].copy()
    ceiling = curves.groupby("household_id")["band_hi_c"].first()
    assert curves.loc[curves["agent"].isin(["llm", "rule_based"]), "band_hi_c"].nunique() == 1
    d["ceiling_c"] = d["household_id"].map(ceiling)
    d["over_c"] = d["pre_indoor_temp_c"] - d["ceiling_c"]
    p = personas.set_index("customer_ID")
    d["orientation"] = d["household_id"].map(p["comfort_orientation"])
    d["archetype"] = d["household_id"].map(p["archetype"])
    d["day"] = d["run_label"]
    k = np.ceil(np.round(d["over_c"] * 2, 9)).astype(int)
    d["bin_k"] = k
    d["bin"] = [f"({(i - 1) / 2:+.1f}, {i / 2:+.1f}]" for i in k]
    if "prompt" in d:
        # what the model was shown (0.1 C): the "X C OVER" figure, or indoor - ceiling
        shown_in = d["prompt"].str.extract(r"indoor temperature \(C\)\s*:\s*([\d.]+)")[0].astype(
            float
        )
        over = d["prompt"].str.extract(r"you are ([\d.]+)C OVER the ceiling")[0].astype(float)
        d["shown_over_c"] = over.fillna((shown_in - d["ceiling_c"]).round(1))
        err = (shown_in - d["pre_indoor_temp_c"]).abs().max()
        assert err <= 0.05 + 1e-9, f"prompt temperature disagrees with replay by {err}"
    return d.sort_values(["household_id", "day", "t"]).reset_index(drop=True)


def per_household_records(curves: pd.DataFrame) -> pd.DataFrame:
    """CPP-window (17:00-20:00) HVAC and total kWh, base vs CPP day, per household x agent."""
    w = curves[curves["t"].between(CPP_LO, CPP_HI - 1)]
    g = w.groupby(["agent", "seed", "household_id", "day"])[["hvac_kwh", "total_kwh"]].sum()
    g = g.unstack("day")
    g.columns = [f"q_{d}_{c.replace('_kwh', '')}" for c, d in g.columns]
    g = g.reset_index()
    meta = curves.groupby("household_id")[["comfort_orientation", "archetype"]].first()
    return g.join(meta, on="household_id")


# --------------------------------------------------------------------------
# 1. H1 -- LLM raw WTP by orientation at matched C above ceiling
# --------------------------------------------------------------------------


def _bin_hh(d: pd.DataFrame, ori: str, k: int):
    s = d[(d["orientation"] == ori) & (d["bin_k"] == k)].groupby("household_id")["wtp"]
    return s.sum().to_numpy(), s.count().to_numpy()


def h1_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in sorted(d["bin_k"].unique()):
        label = d.loc[d["bin_k"] == k, "bin"].iloc[0]
        row = {"C_above_ceiling": label, "side": "above" if k >= 1 else "at/below"}
        cell = {}
        for ori in ORIENTATIONS:
            num, den = _bin_hh(d, ori, k)
            cell[ori] = (num, den)
            m, lo, hi = boot_ratio(num, den)
            short = ORI_SHORT[ori]
            row[f"{short}_n_hh"] = len(num)
            row[f"{short}_n_dec"] = int(den.sum())
            row[f"{short}_mean_wtp"] = m
            row[f"{short}_ci"] = ci(m, lo, hi, 2)
            row[f"{short}_lo"], row[f"{short}_hi"] = lo, hi
        p_b = boot_diff(*cell["comfort_prioritising"], *cell["balanced"])
        b_f = boot_diff(*cell["balanced"], *cell["comfort_flexible"])
        row["prio_minus_bal"] = ci(*p_b, 2)
        row["bal_minus_flex"] = ci(*b_f, 2)
        means = [row[f"{ORI_SHORT[o]}_mean_wtp"] for o in ORIENTATIONS]
        row["H1_order_point"] = (
            bool(means[0] > means[1] >= means[2]) if not np.isnan(means).any() else None
        )
        row["prio>bal_CI_excl_0"] = bool(p_b[1] > 0) if not np.isnan(p_b[1]) else None
        rows.append(row)
    return pd.DataFrame(rows)


H1_COLS = ["C_above_ceiling", "prioritising_n_hh", "prioritising_n_dec", "prioritising_ci",
           "balanced_n_hh", "balanced_n_dec", "balanced_ci", "flexible_n_hh", "flexible_n_dec",
           "flexible_ci", "prio_minus_bal", "bal_minus_flex"]  # fmt: skip


def h1_verdict(h: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per above-ceiling bin with all three orientations: does prio > bal >= flex hold?"""
    means = ["prioritising_mean_wtp", "balanced_mean_wtp", "flexible_mean_wtp"]
    a = h[(h["side"] == "above")].dropna(subset=means)
    return pd.DataFrame({
        "variant": label, "bin": a["C_above_ceiling"],
        "n_hh (prio/bal/flex)": a["prioritising_n_hh"].astype(str) + "/"
        + a["balanced_n_hh"].astype(str) + "/" + a["flexible_n_hh"].astype(str),
        "prio_mean": a[means[0]], "bal_mean": a[means[1]], "flex_mean": a[means[2]],
        "order_holds_point": a["H1_order_point"].astype(bool),
        "prio-bal_CI_excl_0": a["prio>bal_CI_excl_0"].astype(bool),
        "bal-flex_CI_excl_0": a["bal_minus_flex"].str.extract(r"\[([-\d.]+),")[0].astype(float) > 0,
    })  # fmt: skip


def h1_summary_rows(d: pd.DataFrame) -> pd.DataFrame:
    """All above-ceiling decisions pooled per orientation (composition not matched)."""
    rows = []
    for ori in ORIENTATIONS:
        a = d[(d["orientation"] == ori) & (d["over_c"] > 0)].groupby("household_id")["wtp"]
        m, lo, hi = boot_ratio(a.sum(), a.count())
        mo = d[(d["orientation"] == ori) & (d["over_c"] > 0)].groupby("household_id")["over_c"]
        rows.append({"orientation": ori, "n_households_total": d.loc[d["orientation"] == ori,
                     "household_id"].nunique(), "n_hh_above": len(a), "n_dec_above": int(a.count().sum()),
                     "mean_wtp_above": m, "ci": ci(m, lo, hi, 2),
                     "mean_C_above_of_those": float(mo.sum().sum() / mo.count().sum())})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. per-orientation elasticity
# --------------------------------------------------------------------------


def elasticity_by(rec: pd.DataFrame, by: str, agents) -> pd.DataFrame:
    rows = []
    for agent in agents:
        ra = rec[rec["agent"] == agent]
        groups = [("all", ra)] + [(g, ra[ra[by] == g]) for g in sorted(ra[by].unique(),
                                  key=lambda x: ORIENTATIONS.index(x) if x in ORIENTATIONS else x)]  # fmt: skip
        for g, r in groups:
            row = {"agent": agent, by: g, "n": len(r)}
            for comp in ("hvac", "total"):
                e, lo, hi, nan_share = boot_pooled_e(r[f"q_base_{comp}"], r[f"q_cpp_{comp}"])
                row[f"e_{comp}"] = e
                row[f"e_{comp}_ci"] = ci(e, lo, hi)
                row[f"e_{comp}_lo"], row[f"e_{comp}_hi"] = lo, hi
                if comp == "hvac":
                    row["hvac_boot_undefined_share"] = nan_share
            row["q_base_hvac_kwh"] = r["q_base_hvac"].sum()
            row["q_cpp_hvac_kwh"] = r["q_cpp_hvac"].sum()
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. comfort per household
# --------------------------------------------------------------------------


def comfort_per_household(curves: pd.DataFrame) -> pd.DataFrame:
    c = curves.assign(over=(curves["indoor_temp_c"] - curves["band_hi_c"]).clip(lower=0))
    g = c.groupby(["agent", "seed", "household_id", "day"]).agg(
        degree_hours=("over", lambda x: x.sum() * DT_H),
        max_exceedance_c=("over", "max"),
        steps_above=("over", lambda x: int((x > 0).sum())),
        comfort_orientation=("comfort_orientation", "first"),
        archetype=("archetype", "first"),
    )
    return g.reset_index()


def comfort_by(ch: pd.DataFrame, by: str, agents) -> pd.DataFrame:
    rows = []
    for agent in agents:
        for day in ("base", "cpp"):
            a = ch[(ch["agent"] == agent) & (ch["day"] == day)]
            keys = ORIENTATIONS if by == "comfort_orientation" else ARCHETYPES
            for g in ["all"] + keys:
                r = a if g == "all" else a[a[by] == g]
                dh, max_e = boot_mean(r["degree_hours"]), boot_mean(r["max_exceedance_c"])
                rows.append({"agent": agent, "day": day, by: g, "n": len(r),
                             "degree_hours_mean": ci(*dh, 2), "max_exceedance_mean": ci(*max_e, 2),
                             "max_exceedance_max": r["max_exceedance_c"].max() if len(r) else np.nan,
                             "n_hh_max_gt_2.5": int((r["max_exceedance_c"] > 2.5).sum())})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. archetype B zero
# --------------------------------------------------------------------------


def ori_x_archetype(rec: pd.DataFrame, agents) -> pd.DataFrame:
    rows = []
    for agent in agents:
        r = rec[rec["agent"] == agent]
        for ori in ORIENTATIONS:
            for arch in ARCHETYPES:
                c = r[(r["comfort_orientation"] == ori) & (r["archetype"] == arch)]
                rows.append({"agent": agent, "orientation": ORI_SHORT[ori], "archetype": arch,
                             "n": len(c), "q_base_hvac": c["q_base_hvac"].sum(),
                             "q_cpp_hvac": c["q_cpp_hvac"].sum(),
                             "e_hvac": pooled_e(c["q_base_hvac"], c["q_cpp_hvac"]) if len(c) else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def halt_thresholds(d: pd.DataFrame, factors: dict) -> pd.DataFrame:
    """Where a raw WTP has to land for the halt to flip between baseline and CPP.

    Run iff WTP x factor >= 2 kW x 0.5 h x rate, so a decision in the CPP
    window responds to CPP only if WTP is in [0.55/factor, 1.20/factor).
    """
    rows = []
    for arch in ARCHETYPES:
        f = factors[arch]
        lo, hi = 1.0 * 0.55 / f, 1.0 * 1.20 / f
        w = d[(d["archetype"] == arch) & d["t"].between(CPP_LO, CPP_HI - 1)]
        vc = lambda x: ", ".join(f"{v:g}×{n}" for v, n in x.value_counts().sort_index().items())  # noqa: E731
        rows.append({"archetype": arch, "factor": f, "run_at_peak_if_wtp>=": round(lo, 3),
                     "run_at_cpp_if_wtp>=": round(hi, 3),
                     "llm_wtp_17-20_base_day": vc(w.loc[w["day"] == "base", "wtp"]),
                     "llm_wtp_17-20_cpp_day": vc(w.loc[w["day"] == "cpp", "wtp"]),
                     "base_day_decisions_in_responsive_band": int(
                         w.loc[w["day"] == "base", "wtp"].between(lo, hi, inclusive="left").sum())})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. rebound mechanism
# --------------------------------------------------------------------------


def switch_on(curves: pd.DataFrame, agents) -> pd.DataFrame:
    rows = []
    c = curves[curves["agent"].isin(agents)]
    for (agent, hid, day), g in c.groupby(["agent", "household_id", "day"]):
        g = g.sort_values("t")
        on = (g["hvac_kwh"].to_numpy() > 0)
        temp = g["indoor_temp_c"].to_numpy()
        first = next((t for t in range(40, 48) if on[t] and not on[t - 1]), None)
        first_on = next((t for t in range(40, 48) if on[t]), None)
        rows.append({"agent": agent, "household_id": hid, "day": day,
                     "orientation": g["comfort_orientation"].iloc[0], "archetype": g["archetype"].iloc[0],
                     "ac_19:00-21:30 (t38-43)": "".join("1" if on[t] else "0" for t in range(38, 44)),
                     "on_during_cpp_steps": int(on[CPP_LO:CPP_HI].sum()),
                     "indoor_at_20:00_c": round(float(temp[39]), 2),
                     "first_switch_on_after_20": clock(first) if first is not None else "none",
                     "first_switch_on_t": first if first is not None else -1,
                     "first_on_at_or_after_20": clock(first_on) if first_on is not None else "none",
                     "hvac_kw_20:00": float(g["hvac_kwh"].iloc[40] * 2),
                     "hvac_kwh_20-22": float(g["hvac_kwh"].iloc[40:44].sum())})  # fmt: skip
    return pd.DataFrame(rows)


def post_by_orientation(curves: pd.DataFrame, agents) -> pd.DataFrame:
    rows = []
    for agent in agents:
        for day in ("base", "cpp"):
            a = curves[(curves["agent"] == agent) & (curves["day"] == day)]
            tot_post = a.loc[a["t"] == 40, "total_kwh"].sum() * 2
            hv_post_all = a.loc[a["t"] == 40, "hvac_kwh"].sum() * 2
            for ori in ORIENTATIONS:
                o = a[a["comfort_orientation"] == ori]
                hv = o.loc[o["t"] == 40, "hvac_kwh"].sum() * 2
                rows.append({"agent": agent, "day": day, "orientation": ORI_SHORT[ori],
                             "n": o["household_id"].nunique(), "hvac_kw_20:00-20:30": hv,
                             "share_of_agent_post_hvac": hv / hv_post_all if hv_post_all else np.nan,
                             "share_of_agent_post_total": hv / tot_post,
                             "hvac_kwh_17-20": o.loc[o["t"].between(CPP_LO, CPP_HI - 1), "hvac_kwh"].sum(),
                             "hvac_kwh_20-22": o.loc[o["t"].between(40, 43), "hvac_kwh"].sum()})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. non-monotonicity
# --------------------------------------------------------------------------


def monotonicity(d: pd.DataFrame, xcol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consecutive steps (t, t+1) of a household-day where C above ceiling rises
    and both steps are above the ceiling: how often does WTP fall?"""
    per = []
    for (hid, day), g in d.groupby(["household_id", "day"]):
        g = g.sort_values("t")
        x, w = g[xcol].to_numpy(), g["wtp"].to_numpy()
        rise = (x[1:] > x[:-1] + 1e-12) & (x[:-1] > 0)
        per.append({"household_id": hid, "day": day, "orientation": g["orientation"].iloc[0],
                    "pairs_x_rising_above_ceiling": int(rise.sum()),
                    "wtp_fell": int((rise & (w[1:] < w[:-1])).sum()),
                    "wtp_same": int((rise & (w[1:] == w[:-1])).sum()),
                    "wtp_rose": int((rise & (w[1:] > w[:-1])).sum())})  # fmt: skip
    per = pd.DataFrame(per)
    per["fall_rate"] = per["wtp_fell"] / per["pairs_x_rising_above_ceiling"].replace(0, np.nan)
    rows = []
    for g in ["all"] + ORIENTATIONS:
        p = per if g == "all" else per[per["orientation"] == g]
        n = p["pairs_x_rising_above_ceiling"].to_numpy()
        f = p["wtp_fell"].to_numpy()
        m, lo, hi = boot_ratio(f, n)
        rows.append({"group": g, "household_days_with_pairs": int((n > 0).sum()), "pairs": int(n.sum()),
                     "wtp_fell": int(f.sum()), "wtp_same": int(p["wtp_same"].sum()),
                     "wtp_rose": int(p["wtp_rose"].sum()), "fall_rate": ci(m, lo, hi),
                     "household_days_with_any_fall": int((f > 0).sum())})  # fmt: skip
    return per, pd.DataFrame(rows)


def discordance(d: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional: over all pairs of above-ceiling decisions within an
    orientation with different C above ceiling, the share where the hotter
    decision got the lower WTP (0 for a monotone non-decreasing valuation)."""
    rows = []
    for ori in ORIENTATIONS:
        a = d[(d["orientation"] == ori) & (d["over_c"] > 0)]
        x, w = a["over_c"].to_numpy(), a["wtp"].to_numpy()
        dx, dw = x[:, None] - x[None, :], w[:, None] - w[None, :]
        hotter = dx > 1e-12
        rows.append({"orientation": ori, "decisions": len(a), "pairs_diff_temp": int(hotter.sum()),
                     "hotter_lower_wtp_share": float((hotter & (dw < 0)).sum() / hotter.sum()),
                     "hotter_equal_wtp_share": float((hotter & (dw == 0)).sum() / hotter.sum())})  # fmt: skip
    return pd.DataFrame(rows)


def wtp_values(d: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(d["wtp"], d["orientation"])[ORIENTATIONS]
    t.columns = [ORI_SHORT[c] for c in t.columns]
    t["all"] = t.sum(axis=1)
    t["share_all"] = t["all"] / t["all"].sum()
    above = d[d["over_c"] > 0]
    t["all_above_ceiling"] = above["wtp"].value_counts().reindex(t.index).fillna(0).astype(int)
    return t.reset_index().rename(columns={"wtp": "raw_wtp_$"})


# --------------------------------------------------------------------------
# 7. counterfactual isotonic agent
# --------------------------------------------------------------------------


def pava(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic (non-decreasing) least-squares fit; returns (unique x, fitted y)."""
    df = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["sum", "count"]).sort_index()
    xs, sums, cnts = df.index.to_numpy(float), df["sum"].to_numpy(float), df["count"].to_numpy(float)
    blocks: list[list[float]] = []  # [sum, count, n_points]
    for s, c in zip(sums, cnts):
        blocks.append([s, c, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2, n2 = blocks.pop()
            blocks[-1][0] += s2
            blocks[-1][1] += c2
            blocks[-1][2] += n2
    fitted = np.concatenate([np.full(int(n), s / c) for s, c, n in blocks])
    return xs, fitted


@dataclass
class IsotonicFit:
    xs: dict
    ys: dict

    def __call__(self, orientation: str, over_c: float) -> float:
        # linear interpolation between fitted points, clipped at the ends (sklearn's default)
        return float(np.interp(over_c, self.xs[orientation], self.ys[orientation]))


class IsotonicWTPAgent:
    """The LLM's own valuation, made monotone: WTP = isotonic_fit[orientation](C over ceiling),
    through the same Python halt (`satisficing_halt`) and archetype factor. No LLM."""

    def __init__(self, fit: IsotonicFit, wtp_factor: float, wtp_scale: float = 1.0):
        self.fit, self.wtp_factor, self.wtp_scale = fit, float(wtp_factor), float(wtp_scale)

    def decide(self, ctx: DecisionContext) -> Decision:
        a = ctx.appliance
        if a["type"] == "washer":
            forced = forced_washer_outcome(ctx)
            assert forced is not None, "washer is disabled in this run"
            return Decision(run=forced[0], wtp=None, reason=forced[1])
        over = a["indoor_temp_c"] - (a["setpoint_c"] + a["tolerance_band_c"])
        wtp = self.fit(ctx.household["comfort_orientation"], over)
        _, run = satisficing_halt(wtp, ctx.cost_now, self.wtp_scale, self.wtp_factor)
        return Decision(run=run, wtp=wtp, reason="isotonic counterfactual")

    def name(self) -> str:
        return "isotonic"


def fit_isotonic(d: pd.DataFrame) -> tuple[IsotonicFit, pd.DataFrame]:
    xs, ys, rows = {}, {}, []
    for ori in ORIENTATIONS:
        a = d[d["orientation"] == ori]
        xs[ori], ys[ori] = pava(a["over_c"].to_numpy(), a["wtp"].to_numpy())
        # compact step description: where the fitted level changes
        levels = pd.DataFrame({"x": xs[ori], "y": ys[ori].round(4)})
        steps = levels.groupby("y", sort=False)["x"].agg(["min", "max", "count"]).reset_index()
        for r in steps.itertuples():
            rows.append({"orientation": ori, "fitted_wtp": r.y, "from_C_over": r.min,
                         "to_C_over": r.max, "distinct_x": r.count})  # fmt: skip
    return IsotonicFit(xs, ys), pd.DataFrame(rows)


def run_isotonic(runner: Phase3Runner, fit: IsotonicFit, ids) -> pd.DataFrame:
    cfg = runner.cfg
    runs = []
    for hid in ids:
        household, hvac, washer = runner.env(hid)
        for date in cfg.dates:
            for label in ("base", "cpp"):
                agent = IsotonicWTPAgent(fit, runner.wtp_factor(hid), cfg.global_scale)
                res = run_simulation(household, runner.tariff_for(date, label), hvac, washer,
                                     agent, date)  # fmt: skip
                runs.append(DayRun(hid, "isotonic", cfg.primary_seed, cfg.global_scale, date,
                                   label, res))  # fmt: skip
    orientation = runner.assignment.set_index("customer_ID")["comfort_orientation"].to_dict()
    return curves_from_runs(runs, runner.archetype, orientation.get)


# --------------------------------------------------------------------------
# 7b. level-matched rule-based agent (exploratory fairness control)
# --------------------------------------------------------------------------


def _fit_rule_based_module():
    """experiments/fit_rule_based.py, imported for its exact fitting code."""
    path = Path(__file__).with_name("fit_rule_based.py")
    spec = importlib.util.spec_from_file_location("fit_rule_based", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_level_matched(llm_log: pd.DataFrame, curves: pd.DataFrame,
                      personas: pd.DataFrame) -> tuple[dict, pd.DataFrame]:  # fmt: skip
    """Refit k[orientation] to this run's own LLM WTP with the method that produced the
    frozen k (`fit_rule_based.fit_hvac`: least squares through the origin of raw WTP on
    max(0, indoor shown in the prompt - ceiling), one row per distinct prompt)."""
    ceiling = curves.groupby("household_id")["band_hi_c"].first()
    assert ceiling.nunique() == 1, "level-matching assumes the fixed band"
    log = llm_log.assign(
        band_hi_c=llm_log["household_id"].map(ceiling),
        comfort_orientation=llm_log["household_id"].map(
            personas.set_index("customer_ID")["comfort_orientation"]),
    )  # fmt: skip
    fit = _fit_rule_based_module().fit_hvac(log)
    frozen = RuleBasedParams.from_yaml(band_mode="fixed").k_by_orientation
    rows = [{"orientation": o, "k_level_matched": round(fit[o]["k"], 3), "k_exact": fit[o]["k"],
             "k_frozen_fixed_band": frozen[o], **{k: v for k, v in fit[o].items() if k != "k"}}
            for o in ORIENTATIONS]  # fmt: skip
    # rounded to 3 dp, as the frozen config values are
    return {o: round(fit[o]["k"], 3) for o in ORIENTATIONS}, pd.DataFrame(rows)


def run_rule_based(runner: Phase3Runner, k: dict, ids, name: str) -> pd.DataFrame:
    """The rule-based satisficer with the given k, all households, both days (no LLM)."""
    cfg = runner.cfg
    params = RuleBasedParams(k_by_orientation=dict(k), w_max=runner.rule_params.w_max)
    runs = []
    for hid in ids:
        household, hvac, washer = runner.env(hid)
        for date in cfg.dates:
            for label in ("base", "cpp"):
                agent = RuleBasedSatisficerAgent(params, cfg.global_scale, runner.wtp_factor(hid))
                res = run_simulation(household, runner.tariff_for(date, label), hvac, washer,
                                     agent, date)  # fmt: skip
                runs.append(DayRun(hid, name, cfg.primary_seed, cfg.global_scale, date, label,
                                   res))  # fmt: skip
    orientation = runner.assignment.set_index("customer_ID")["comfort_orientation"].to_dict()
    return curves_from_runs(runs, runner.archetype, orientation.get)


def level_matched(runner: Phase3Runner, llm_log, curves, personas, ids):
    """Fit + run the level-matched agent. The same code path with the frozen k must
    reproduce the saved rule-based curves exactly (checked here)."""
    from hems.phase3_report import CURVE_SORT

    frozen = run_rule_based(runner, runner.rule_params.k_by_orientation, ids, "rule_based")
    saved = _sort(curves[curves["agent"] == "rule_based"], CURVE_SORT)
    pd.testing.assert_frame_equal(_sort(frozen, CURVE_SORT)[saved.columns], saved,
                                  check_exact=True, check_dtype=False)  # fmt: skip
    k, fit = fit_level_matched(llm_log, curves, personas)
    return k, fit, run_rule_based(runner, k, ids, LM)


# --------------------------------------------------------------------------
# 8. rebound_delta_kw
# --------------------------------------------------------------------------


def rebound_check(curves: pd.DataFrame) -> pd.DataFrame:
    """Recompute rebound_peak_kw (20:00-22:00 peak of the population total) per day,
    the saved cpp-minus-base delta, and an energy-based alternative."""
    rows = []
    for agent in ["milp", "noisy_milp", "rule_based", "llm", "isotonic", LM]:
        a = curves[(curves["agent"] == agent) & (curves["seed"] == 42)]
        if a.empty:
            continue
        pop = a.groupby(["day", "t"])["total_kwh"].sum().unstack("day")
        hv = a.groupby(["day", "t"])["hvac_kwh"].sum().unstack("day")
        r = {"agent": agent}
        for day in ("base", "cpp"):
            w = pop.loc[40:43, day]
            r[f"{day}_rebound_peak_kw"] = round(w.max() * 2, 3)
            r[f"{day}_peak_step"] = clock(int(w.idxmax()))
            r[f"{day}_hvac_kwh_20-22"] = hv.loc[40:43, day].sum()
            r[f"{day}_hvac_kw_by_step_20-22"] = " / ".join(f"{v * 2:.0f}" for v in hv.loc[40:43, day])
        r["rebound_delta_kw (saved def.)"] = round(r["cpp_rebound_peak_kw"] - r["base_rebound_peak_kw"], 3)
        r["hvac_kwh_20-22_cpp_minus_base"] = r["cpp_hvac_kwh_20-22"] - r["base_hvac_kwh_20-22"]
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def compute() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    runner, curves, llm_log, rule_log, checks = replay()
    personas = pd.read_csv(RUN / "personas_used.csv")
    ids = sorted(int(h) for h in curves["household_id"].unique())
    factors = runner.cfg.archetype_factors
    T: dict = {"checks": checks}

    dl = decisions(llm_log, curves, personas)
    dr = decisions(rule_log, curves, personas)
    T["decisions_llm"], T["decisions_rule"] = dl, dr
    T["pretemp_vs_prompt_max_abs_err"] = float(
        (dl["prompt"].str.extract(r"indoor temperature \(C\)\s*:\s*([\d.]+)")[0].astype(float)
         - dl["pre_indoor_temp_c"]).abs().max()
    )  # fmt: skip

    # 7 first (its curves feed 2, 3, 5, 8)
    fit, fit_steps = fit_isotonic(dl)
    iso_curves = run_isotonic(runner, fit, ids)
    T["iso_fit"], T["iso_fit_steps"] = fit, fit_steps
    allc = pd.concat([curves, iso_curves], ignore_index=True)
    lm = []
    if LEVEL_MATCHED:  # exploratory fairness control
        T["lm_k"], T["lm_fit"], lm_curves = level_matched(runner, llm_log, curves, personas, ids)
        allc = pd.concat([allc, lm_curves], ignore_index=True)
        lm = [LM]
    prim = allc[allc["seed"] == runner.cfg.primary_seed]
    T["curves"] = allc

    rec = per_household_records(prim)
    saved = pd.read_csv(RUN / "elasticity_per_household.csv")
    chk = rec.merge(saved, on=["household_id", "agent"], suffixes=("", "_saved"))
    for c in ("q_base_hvac", "q_cpp_hvac", "q_base_total", "q_cpp_total"):
        assert np.allclose(chk[c], chk[f"{c}_saved"], atol=1e-9), c
    T["records"] = rec

    agents3 = ["llm", "rule_based", "milp"]
    T["h1"] = h1_table(dl)
    T["h1_pooled"] = h1_summary_rows(dl)
    # robustness: base day only, and one decision per distinct prompt (first occurrence,
    # ordered household -> base before cpp -> t), so pre-17:00 prompts shared by the two
    # days count once. Identical prompt => identical answer (T=0, cache), so this is exact.
    T["h1_base_only"] = h1_table(dl[dl["day"] == "base"])
    dedup = dl.sort_values(["household_id", "day", "t"]).drop_duplicates("prompt", keep="first")
    T["h1_dedup"] = h1_table(dedup)
    T["h1_dedup_n"] = {"decisions": len(dl), "distinct_prompts": len(dedup),
                       "distinct_across_households": int(
                           dl.groupby("prompt")["household_id"].nunique().gt(1).sum())}
    T["elasticity"] = elasticity_by(rec, "comfort_orientation", agents3 + ["isotonic"] + lm)
    ch = comfort_per_household(prim)
    T["comfort_hh"] = ch
    T["comfort_ori"] = comfort_by(ch, "comfort_orientation", ["llm", "rule_based", "isotonic"] + lm)
    T["comfort_arch"] = comfort_by(ch, "archetype", ["llm", "rule_based", "isotonic"] + lm)
    T["comfort_gt25"] = ch[(ch["max_exceedance_c"] > 2.5)].sort_values(
        ["agent", "day", "max_exceedance_c"], ascending=[True, True, False])
    T["ori_x_arch"] = ori_x_archetype(rec, agents3 + lm)
    T["thresholds"] = halt_thresholds(dl, factors)
    T["hvac_window_hh"] = rec[rec["agent"].isin(["llm", "rule_based"])][
        ["agent", "household_id", "comfort_orientation", "archetype", "q_base_hvac", "q_cpp_hvac"]
    ].sort_values(["agent", "archetype", "comfort_orientation", "household_id"])
    T["switch"] = switch_on(prim, ["llm", "rule_based", "isotonic"] + lm)
    T["post_ori"] = post_by_orientation(prim, ["llm", "rule_based", "milp", "isotonic"] + lm)
    T["mono_hh"], T["mono"] = monotonicity(dl, "over_c")
    T["mono_hh_shown"], T["mono_shown"] = monotonicity(dl, "shown_over_c")
    T["discord"] = discordance(dl)
    T["wtp_values"] = wtp_values(dl)
    T["rebound"] = rebound_check(allc)

    resolved = yaml.safe_load((RUN / "config_resolved.yaml").read_text())
    summ = agent_summary(allc, resolved)
    T["cf_summary"] = summ[(summ["seed"] == 42) & summ["agent"].isin(agents3 + ["isotonic"] + lm)]
    T["summary_all"] = summ

    csvs = {"h1_wtp_by_orientation_bin": T["h1"], "h1_pooled_above_ceiling": T["h1_pooled"],
            "h1_wtp_by_orientation_bin_base_day_only": T["h1_base_only"],
            "h1_wtp_by_orientation_bin_dedup_prompt": T["h1_dedup"],
            "elasticity_by_orientation": T["elasticity"], "comfort_per_household": ch,
            "comfort_by_orientation": T["comfort_ori"], "comfort_by_archetype": T["comfort_arch"],
            "comfort_max_exceedance_gt_2p5": T["comfort_gt25"],
            "orientation_x_archetype": T["ori_x_arch"], "halt_thresholds": T["thresholds"],
            "hvac_kwh_17_20_per_household": T["hvac_window_hh"], "switch_on_per_household": T["switch"],
            "post_cpp_by_orientation": T["post_ori"], "monotonicity_per_household_day": T["mono_hh"],
            "monotonicity_summary": T["mono"], "monotonicity_summary_shown_values": T["mono_shown"],
            "wtp_discordance": T["discord"], "wtp_distinct_values": T["wtp_values"],
            "isotonic_fit_steps": fit_steps, "counterfactual_summary": T["cf_summary"],
            "rebound_check": T["rebound"],
            "llm_decisions_with_state": dl.drop(columns=["prompt", "reason", "misreads"])}  # fmt: skip
    if LEVEL_MATCHED:
        csvs["level_matched_fit"] = T["lm_fit"]
    for name, df in csvs.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "replay_check.json").write_text(json.dumps(
        {**checks, "pretemp_vs_prompt_max_abs_err": T["pretemp_vs_prompt_max_abs_err"]}, indent=2))
    return T



# --------------------------------------------------------------------------
# 9. figures
# --------------------------------------------------------------------------


def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_wtp(T, path):
    d, fit, h1 = T["decisions_llm"], T["iso_fit"], T["h1"]
    rng = np.random.default_rng(0)  # display jitter only
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, ori in zip(axes, ORIENTATIONS):
        col, short = ORI_COLOR[ori], ORI_SHORT[ori]
        a = d[d["orientation"] == ori]
        ax.scatter(a["over_c"], a["wtp"] + rng.uniform(-0.03, 0.03, len(a)), s=9, color=col,
                   alpha=0.25, linewidths=0, label="LLM decision (raw WTP)")
        xs = np.linspace(d["over_c"].min(), d["over_c"].max(), 400)
        ax.plot(xs, [fit(ori, x) for x in xs], color=INK, lw=1.5, label="isotonic fit (item 7)")
        m = h1[h1[f"{short}_n_dec"] > 0]
        centers = m["C_above_ceiling"].map(lambda b: float(b[1:5]) + 0.25)
        means = m[f"{short}_mean_wtp"]
        ax.errorbar(centers, means, yerr=[means - m[f"{short}_lo"], m[f"{short}_hi"] - means], fmt="o", ms=6,
                    color=col, mec="white", mew=1.2, ecolor=col, elinewidth=2, capsize=0,
                    label="0.5 °C-bin mean, 95% CI")
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        n = a["household_id"].nunique()
        ax.set_title(f"{short} (n = {n} households)", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("°C above comfort ceiling (26 °C) at decision time", color=INK2, fontsize=9)
        _style(ax, "both")
    axes[0].set_ylabel("raw LLM WTP ($ per half-hour)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("LLM willingness-to-pay vs temperature above ceiling, by comfort orientation "
                 "(both days pooled)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_switch(T, path):
    s = T["switch"]
    slots = ["20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "none"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, day in zip(axes, ["cpp", "base"]):
        for j, agent in enumerate(["llm", "rule_based"]):
            c = s[(s["agent"] == agent) & (s["day"] == day)]["first_switch_on_after_20"]
            counts = c.value_counts().reindex(slots, fill_value=0)
            x = np.arange(len(slots)) + (j - 0.5) * 0.4
            ax.bar(x, counts, width=0.38, color=AGENT_COLOR[agent], label=AGENT_LABEL[agent])
            for xi, v in zip(x, counts):
                if v:
                    ax.text(xi, v + 0.3, str(v), ha="center", fontsize=8, color=INK2)
        ax.set_xticks(np.arange(len(slots)), slots)
        ax.set_title("CPP day" if day == "cpp" else "baseline day (reference)", color=INK,
                     fontsize=11, loc="left")
        ax.set_xlabel("first AC switch-on at/after 20:00 (off → on)", color=INK2, fontsize=9)
        _style(ax)
    axes[0].set_ylabel("households (of 24)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_elasticity(T, path):
    e = T["elasticity"]
    groups = ["all"] + ORIENTATIONS
    agents = ["llm", "rule_based", "milp", "isotonic"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, comp, title in zip(axes, ["hvac", "total"], ["HVAC (flexible layer)", "whole household"]):
        if comp == "total":
            ax.axhspan(-0.47, -0.026, color=GRID, alpha=0.6, lw=0)
            ax.text(3.45, -0.03, "Conway (2020) range", fontsize=8, color=INK2, ha="right", va="top")
        for j, agent in enumerate(agents):
            r = e[e["agent"] == agent].set_index("comfort_orientation").reindex(groups)
            x = np.arange(len(groups)) + (j - 1.5) * 0.18
            ax.errorbar(x, r[f"e_{comp}"], yerr=[r[f"e_{comp}"] - r[f"e_{comp}_lo"],
                        r[f"e_{comp}_hi"] - r[f"e_{comp}"]], fmt="o", ms=6, color=AGENT_COLOR[agent],
                        mec="white", mew=1.2, elinewidth=2, capsize=0, label=AGENT_LABEL[agent])
        n = e[e["agent"] == "llm"].set_index("comfort_orientation").reindex(groups)["n"]
        ax.set_xticks(range(len(groups)),
                      [f"{'all' if g == 'all' else ORI_SHORT[g]}\n(n = {k})" for g, k in zip(groups, n)])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{title} elasticity, CPP window 17:00–20:00", color=INK, fontsize=11, loc="left")
        _style(ax)
    axes[0].set_ylabel("pooled arc elasticity (95% household bootstrap CI)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_comfort(T, path):
    ch = T["comfort_hh"]
    cats = [(a, d) for a in ["llm", "rule_based", "isotonic"] for d in ["base", "cpp"]]
    rng = np.random.default_rng(0)  # display jitter only
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, col, lab in zip(axes, ["degree_hours", "max_exceedance_c"],
                            ["degree-hours above ceiling (°C·h)", "max exceedance above ceiling (°C)"]):
        for i, (agent, day) in enumerate(cats):
            c = ch[(ch["agent"] == agent) & (ch["day"] == day)]
            for ori in ORIENTATIONS:
                o = c[c["comfort_orientation"] == ori]
                ax.scatter(i + rng.uniform(-0.18, 0.18, len(o)), o[col], s=30, color=ORI_COLOR[ori],
                           edgecolor="white", linewidth=0.8,
                           label=ORI_SHORT[ori] if i == 0 else None)
            ax.hlines(c[col].median(), i - 0.3, i + 0.3, color=INK, lw=2)
        if col == "max_exceedance_c":
            ax.axhline(2.5, color=INK2, lw=0.8, ls="--")
            ax.text(len(cats) - 0.5, 2.55, "2.5 °C", fontsize=8, color=INK2, ha="right", va="bottom")
        ax.set_xticks(range(len(cats)), [f"{AGENT_LABEL[a].split(' ')[0]}\n{d}" for a, d in cats])
        ax.set_ylabel(lab, color=INK2, fontsize=9)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, title="orientation (bar = median)", title_fontsize=8)
    fig.suptitle("Comfort cost per household (MILP: 0 for every household)", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figures(T) -> dict:
    out = {"fig_wtp_vs_c_above_ceiling.png": fig_wtp, "fig_switch_on_histogram.png": fig_switch,
           "fig_elasticity_by_orientation.png": fig_elasticity,
           "fig_comfort_distribution.png": fig_comfort}  # fmt: skip
    for name, fn in out.items():
        fn(T, OUT / name)
    return out



# --------------------------------------------------------------------------
# 10. analysis.md
# --------------------------------------------------------------------------


def _details(summary: str, table: str) -> str:
    return f"<details><summary>{summary}</summary>\n\n{table}\n\n</details>"


def write_md(T) -> None:
    c = T["checks"]
    h1, e, cf, reb = T["h1"], T["elasticity"], T["cf_summary"].set_index("agent"), T["rebound"]
    mono = T["mono"].set_index("group")
    sw = T["switch"]
    cpp_sw = sw[sw["day"] == "cpp"]
    at20 = cpp_sw[cpp_sw["first_switch_on_after_20"] == "20:00"].groupby("agent").size()
    ev = e.set_index(["agent", "comfort_orientation"])
    h1_above = h1[h1["side"] == "above"]
    ok_bins = h1_above.dropna(subset=["prioritising_mean_wtp", "balanced_mean_wtp",
                                      "flexible_mean_wtp"])  # fmt: skip
    bf = h1_above.dropna(subset=["balanced_mean_wtp", "flexible_mean_wtp"])
    bf_pos = int((bf["balanced_mean_wtp"] > bf["flexible_mean_wtp"]).sum())
    gt = T["comfort_gt25"]
    wv = T["wtp_values"]
    top = wv.sort_values("all", ascending=False).head(3)
    mr = reb.set_index("agent")

    L = [
        "# Final Run v1 — post-run analysis",
        "",
        "Generated by `experiments/analyze_final_v1.py` from the saved artifacts in "
        "`results/phase3/final_v1/` (commit of the run: `560d7fb`, pre-registration "
        "`docs/prereg_final_run_v1.md`). No live LLM calls: every LLM answer is replayed "
        "from `results/phase3/llm_cache_final.jsonl`, with the client patched to raise on any "
        "cache miss or write. Nothing outside `analysis/` was written.",
        "",
        "Every result is labelled **pre-registered** or **exploratory (post-hoc)**. Bootstrap "
        f"CIs: households resampled with replacement, {N_BOOT:,} draws, seed {BOOT_SEED}, "
        "percentile 95% intervals; pooled quantities are recomputed on each resample. "
        "n per orientation is imbalanced (prioritising 5 / balanced 8 / flexible 11), and cells "
        "with 1–3 households give CIs that are narrow by construction, not precise.",
        "",
        "## 0. Inventory and offline regeneration",
        "",
        "**Saved per step** (`curves.parquet`, 13,824 rows = 24 households × 6 agent-seeds × "
        "2 days × 48 steps): rate, CPP flag, base / HVAC / washer / flexible / total kWh, "
        "**end-of-step** indoor temperature, band ceiling, in-band flag. "
        "**Saved per decision**: `llm_decision_log.parquet` (t, clock, appliance, full prompt, "
        "raw WTP, reason, cost_now, utility, run, cached, forced, scale/factor, model version, "
        "misread flags, household, seed, day) and `rule_decision_log.parquet` (same minus "
        "prompt/reason, plus the regressor = °C over ceiling, clipped at 0). **Per household**: "
        "`elasticity_per_household.csv` (CPP-window kWh per component, both days), "
        "`load_metrics.csv`, `personas_used.csv` (orientation, archetype).",
        "",
        "**Not saved:** the exact indoor temperature *at decision time* (the logs carry it only "
        "as 0.1 °C text inside the prompt; curves hold the post-step value), and the "
        "counterfactual agent of item 7. Both were regenerated deterministically: every "
        "household × agent × seed × day was re-simulated from the tagged code, with MILP / "
        "noisy-MILP / rule-based re-solved and the LLM replayed from the cache, and the "
        "pre-decision temperature recorded from the simulator's decision context.",
        "",
        "Replay check (all must hold before anything below is computed; they did):",
        "",
        md(pd.DataFrame([
            ("curves.parquet reproduced exactly (all agents, 13,824 rows)", c["curves_identical"]),
            ("llm_decision_log reproduced exactly (prompt, WTP, reason, run, utility, model "
             "version, timestamp; 4,608 rows)", c["llm_log_identical"]),
            ("rule_decision_log reproduced exactly", c["rule_log_identical"]),
            ("summary.csv recomputed from replay identical", c["summary_csv_identical"]),
            ("live LLM calls during replay", c["live_calls"]),
            ("cache hits during replay (= 2,304 prompted decisions)", c["cache_hits"]),
            ("cache file unchanged (sha256 before = after)", c["cache_unchanged"]),
            ("max |prompt temperature − replayed pre-decision temperature| (°C)",
             round(T["pretemp_vs_prompt_max_abs_err"], 4)),
        ], columns=["check", "result"])),
        "",
        "Per-decision state (exact °C over ceiling, the value shown to the model, "
        "orientation, archetype) is saved as `llm_decisions_with_state.csv`.",
        "",
        "## 1. H1 — LLM raw WTP by comfort orientation at matched temperature  *(pre-registered)*",
        "",
        "Raw LLM WTP (before archetype factor), both days pooled, binned on the exact "
        "pre-decision °C above the 26 °C ceiling in 0.5 °C bins (right-closed; a decision "
        "exactly at the ceiling falls in (−0.5, 0]). Mean is decision-weighted within the bin; "
        "CI resamples households within orientation. Differences resample each orientation "
        "independently. Note: base and CPP days share identical prompts before 17:00, so "
        "pre-17:00 decisions appear twice (as pre-specified, 'both days pooled').",
        "",
        md(h1[H1_COLS]),
        "",
        "### H1 robustness: base day only / one decision per distinct prompt  *(exploratory "
        "(post-hoc) robustness check of the pre-registered table; same bins, same bootstrap)*",
        "",
        "(a) Baseline day only — each household-step counted once; drops the CPP day's "
        "post-17:00 decisions.",
        "",
        md(T["h1_base_only"][H1_COLS]),
        "",
        f"(b) Deduplicated by distinct prompt — {T['h1_dedup_n']['distinct_prompts']:,} distinct "
        f"prompts from {T['h1_dedup_n']['decisions']:,} decisions; each kept once, at its first "
        f"occurrence (household, then baseline before CPP day, then time). Removes the "
        f"pre-17:00 double count and also collapses identical prompts across households of the "
        f"same orientation ({T['h1_dedup_n']['distinct_across_households']} prompts occurred "
        f"in more than one household); the kept copy's exact temperature sets the bin.",
        "",
        md(T["h1_dedup"][H1_COLS]),
        "",
        "Ordering check in bins where all three orientations are observed:",
        "",
        md(pd.concat([h1_verdict(h1, "pooled (pre-registered)"),
                      h1_verdict(T["h1_base_only"], "(a) base day only"),
                      h1_verdict(T["h1_dedup"], "(b) distinct prompts")], ignore_index=True), 2),
        "",
        "All above-ceiling decisions per orientation (not temperature-matched; shown for "
        "context only — prioritising households sit much closer to the ceiling):",
        "",
        md(T["h1_pooled"]),
        "",
        f"**Summary.** Below the ceiling the LLM gives essentially the same near-zero WTP "
        f"(~$0.05) to every orientation. Above it, in every bin where all three orientations are "
        f"observed ({', '.join(ok_bins['C_above_ceiling'])}), the pre-registered order "
        f"prioritising > balanced ≥ flexible holds, and both gaps have 95% CIs above zero. "
        f"Prioritising households never get more than "
        f"{T['decisions_llm'].query('orientation == \"comfort_prioritising\"')['over_c'].max():.2f} °C "
        f"over (their high WTP keeps the AC on), so above 1 °C only balanced vs flexible can be "
        f"compared: balanced > flexible in {bf_pos} of {len(bf)} bins. H1 is supported where it "
        f"can be tested; the comparison with prioritising rests on 5 households and ≤1 °C. "
        + (("Robustness: counting the baseline day only, or each distinct prompt once, "
            "leaves the ordering and both CI exclusions unchanged in every testable bin.")
           if all(h1_verdict(t, "")[["order_holds_point", "prio-bal_CI_excl_0",
                                      "bal-flex_CI_excl_0"]].all().all()
                  for t in (T["h1_base_only"], T["h1_dedup"]))
           else "Robustness: the ordering or a CI exclusion changes in at least one robustness "
                "variant — see the ordering check above."),
        "",
        "## 2. Per-orientation elasticity  *(exploratory, post-hoc)*",
        "",
        "Pooled arc elasticity (Σ CPP-window kWh, base method, %ΔP = (1.20−0.55)/0.55) over "
        "households in the group; primary seed. `isotonic` is the item-7 counterfactual.",
        "",
        md(e[["agent", "comfort_orientation", "n", "e_hvac_ci", "e_total_ci",
              "q_base_hvac_kwh", "q_cpp_hvac_kwh"]]),
        "",
        f"**Summary.** The LLM's HVAC response grows as comfort orientation loosens: "
        f"prioritising {ev.loc[('llm', 'comfort_prioritising'), 'e_hvac']:.2f}, balanced "
        f"{ev.loc[('llm', 'balanced'), 'e_hvac']:.2f}, flexible "
        f"{ev.loc[('llm', 'comfort_flexible'), 'e_hvac']:.2f}. Prioritising households do not "
        f"respond at all under either satisficer: their WTP (~$2–2.5 once over the ceiling) "
        f"clears even the CPP cost. The rule-based agent puts balanced "
        f"({ev.loc[('rule_based', 'balanced'), 'e_hvac']:.2f}) ahead of flexible "
        f"({ev.loc[('rule_based', 'comfort_flexible'), 'e_hvac']:.2f}). MILP responds about "
        f"equally in every group (≈ −0.65 to −0.72). Orientation is confounded with archetype "
        f"(7 of 11 flexible households are archetype A, and no prioritising household is), so "
        f"these are not clean orientation effects (see item 4).",
        "",
        "## 3. Comfort per household  *(metrics pre-registered; per-orientation / per-archetype "
        "breakdown and the > 2.5 °C list exploratory)*",
        "",
        "Degree-hours = Σ max(0, indoor − 26 °C) × 0.5 h per household-day (end-of-step "
        "temperatures, as in the run's summary); mean over households with bootstrap CI. MILP is "
        "0 for every household and day and is omitted. Per-household values: "
        "`comfort_per_household.csv`.",
        "",
        "By orientation:",
        "",
        md(T["comfort_ori"], 2),
        "",
        "By archetype:",
        "",
        md(T["comfort_arch"], 2),
        "",
        "Household-days with max exceedance > 2.5 °C:",
        "",
        md(gt[["agent", "household_id", "day", "comfort_orientation", "archetype",
               "max_exceedance_c", "degree_hours", "steps_above"]], 2),
        "",
        f"**Summary.** Under the LLM, comfort cost rises from prioritising (≈1.3 °C·h, "
        f"never above 0.74 °C) to flexible, which carries the worst cases. Among LLM "
        f"household-days, "
        f"{len(gt[gt['agent'] == 'llm'])} exceed 2.5 °C across "
        f"{gt[gt['agent'] == 'llm']['household_id'].nunique()} households "
        f"({', '.join(map(str, sorted(gt[gt['agent'] == 'llm']['household_id'].unique())))}). "
        f"Two of them (10239118 flexible/B, 9012382 balanced/C) exceed it on the baseline day "
        f"too: after the archetype factor their WTP is below the ordinary peak cost, so the "
        f"cause is not the CPP event. 8208263 (flexible/C) exceeds it only on the CPP day. The rule-based "
        f"agent's cost is spread more evenly (base-day max 1.81 °C) but is higher on average on "
        f"the CPP day, where balanced households take the largest hit.",
        "",
        "## 4. Why archetype B shows zero LLM elasticity  *(exploratory, post-hoc)*",
        "",
        "Orientation × archetype, pooled HVAC elasticity (n = households):",
        "",
        md(T["ori_x_arch"]),
        "",
        "The mechanism is the halt rule: run iff WTP × factor ≥ 2 kW × 0.5 h × rate. A "
        "decision in 17:00–20:00 responds to CPP only if raw WTP lands in "
        "[0.55 / factor, 1.20 / factor). LLM WTPs actually given in that window:",
        "",
        md(T["thresholds"]),
        "",
        _details("Per-household HVAC kWh, 17:00–20:00, baseline vs CPP (LLM and rule-based)",
                 md(T["hvac_window_hh"])),
        "",
        "**Summary.** The LLM's WTP takes a few fixed values (0.05, 0.45, 0.5, 1.25, 1.5, 2.5, "
        "…). For archetype B (factor 1.0) the band that responds to price is $0.55–1.20. The "
        "LLM gave a value in it once in 48 baseline-day window decisions (a single $0.75). "
        "Every other B decision is either below $0.55 (off at the ordinary peak price already) "
        "or at $1.25 or more (on even at CPP), so CPP changes nothing. Factors 1.3 (A) and 0.7 "
        "(C) move that band onto values the LLM does give: 0.45/0.5 for A, 1.25/1.5 for C. "
        "The zero comes from where the LLM's fixed values sit relative to the price levels, "
        "not from B households being price-insensitive. The rule-based agent's continuous WTP "
        "crosses the band, so its B elasticity is non-zero.",
        "",
        "## 5. Rebound mechanism, CPP day  *(exploratory, post-hoc)*",
        "",
        "AC state at steps 38–43 (19:00, 19:30, 20:00, 20:30, 21:00, 21:30; 1 = on), and first "
        "off→on switch at or after 20:00. Rate after the event: 0.55 until 21:00, then 0.28 "
        "(shoulder), 0.22 from 22:00.",
        "",
        _details("Per household, CPP day (LLM, rule-based, isotonic)",
                 md(cpp_sw.sort_values(["agent", "orientation", "household_id"])[
                     ["agent", "household_id", "orientation", "archetype", "ac_19:00-21:30 (t38-43)",
                      "indoor_at_20:00_c", "first_switch_on_after_20", "hvac_kw_20:00",
                      "hvac_kwh_20-22"]], 2)),
        "",
        "Switch-on time distribution (households; both days):",
        "",
        md(pd.crosstab([sw["agent"], sw["day"]], sw["first_switch_on_after_20"]).reset_index()),
        "",
        "Post-CPP contribution by orientation (HVAC kW in 20:00–20:30, share of the agent's "
        "HVAC and total load in that half-hour, HVAC kWh in the window and after):",
        "",
        md(T["post_ori"]),
        "",
        f"**Summary.** On the CPP day, {at20.get('rule_based', 0)} of 24 rule-based households "
        f"switch on in the same half-hour (20:00), versus {at20.get('llm', 0)} under the LLM; "
        "the LLM spreads switch-on over 20:00, 20:30, 21:00 and later. Most of the difference "
        "is flexible households: the LLM gives them roughly $0.45–0.6 whether they are 1 °C "
        "or 3 °C over. That is too low for some to pay the 0.55 rate after 20:00, and they "
        "wait for the 0.28 shoulder rate at 21:00. The rule-based WTP rises with temperature, "
        "so the heat built up during CPP pushes almost all of them over the threshold at 20:00. "
        "The LLM's lower post-CPP ratio therefore comes from its coarse, flat WTP interacting "
        "with the 21:00 tariff step. It does not look like foresight, and it costs comfort "
        "(item 3). Flexible households supply the largest share of every agent's post-CPP HVAC load.",
        "",
        "## 6. WTP non-monotonicity  *(pre-registered as descriptive under H4; rate definition "
        "post-hoc)*",
        "",
        "Consecutive steps of a household-day where °C above ceiling rises and both steps are "
        "above the ceiling; rate = share where WTP falls (household-day bootstrap CI). Upper table "
        "uses the exact temperature, lower the 0.1 °C value shown to the model.",
        "",
        md(T["mono"]),
        "",
        md(T["mono_shown"]),
        "",
        _details("Per household-day (rows with ≥ 1 fall)",
                 md(T["mono_hh"][T["mono_hh"]["wtp_fell"] > 0])),
        "",
        "Across all above-ceiling decisions within an orientation (every pair with different "
        "temperature): share where the hotter one got the lower WTP, and share with equal WTP.",
        "",
        md(T["discord"]),
        "",
        "Distinct raw WTP values (all 2,304 HVAC decisions):",
        "",
        md(wv, 3),
        "",
        f"**Summary.** The LLM uses {len(wv)} distinct WTP values; the top three "
        f"({', '.join(f'${v:g}' for v in top['raw_wtp_$'])}) make up "
        f"{top['share_all'].sum():.0%} of decisions. When the house gets hotter from one "
        f"half-hour to the next above the ceiling, WTP falls {mono.loc['all', 'wtp_fell']} "
        f"times in {mono.loc['all', 'pairs']} ({mono.loc['all', 'fall_rate']}). Almost all "
        f"of these are flexible households ({mono.loc['comfort_flexible', 'wtp_fell']} of "
        f"{mono.loc['comfort_flexible', 'pairs']}); balanced has "
        f"{mono.loc['balanced', 'wtp_fell']}, prioritising none. Using the rounded temperature "
        "the model saw gives the same picture, so this is not a rounding artefact. Comparing "
        "all pairs of decisions within an orientation, 12–18% have the hotter state valued "
        "lower, and 29–51% have equal WTP: the temperature-to-WTP mapping is a coarse step "
        "function with reversals rather than a rising curve.",
        "",
        "## 7. Counterfactual: monotone version of the LLM's own valuation  *(exploratory, "
        "post-hoc)*",
        "",
        "Per orientation, an isotonic (monotone non-decreasing, least-squares, pool-adjacent-"
        "violators) fit of raw LLM WTP on exact °C above ceiling, fitted on all 2,304 LLM "
        "HVAC decisions (both days, below-ceiling included), with linear interpolation between "
        "fitted points and clipping at the ends. It runs as an agent through the same halt, "
        "archetype factor, simulator, tariffs and 24 households on both days. No LLM. Caveat: "
        "it is fitted on states the LLM trajectories visited (in-sample), and it runs into "
        "hotter states than the LLM did.",
        "",
        md(cf.reset_index()[["agent", "cpp_post_20:00-20:30_kw", "cpp_cpp_window_mean_kw",
                             "cpp_post_over_cpp_ratio", "base_post_over_cpp_ratio",
                             "base_degree_hours_above_ceiling", "cpp_degree_hours_above_ceiling",
                             "base_max_exceedance_c", "cpp_max_exceedance_c",
                             "elasticity_flexible", "elasticity_total"]]),
        "",
        "(`elasticity_flexible` = HVAC here: washer disabled. Per-orientation elasticities with "
        "CIs for the isotonic agent are in item 2.)",
        "",
        _details("Fitted step function (isotonic_fit_steps.csv)", md(T["iso_fit_steps"], 4)),
        "",
        f"**Summary.** Making the LLM's valuation monotone brings the rebound back: post/CPP "
        f"ratio {cf.loc['isotonic', 'cpp_post_over_cpp_ratio']:.3f}, against "
        f"{cf.loc['llm', 'cpp_post_over_cpp_ratio']:.3f} for the LLM and "
        f"{cf.loc['rule_based', 'cpp_post_over_cpp_ratio']:.3f} for rule-based. Elasticity "
        f"rises (HVAC {cf.loc['isotonic', 'elasticity_flexible']:.3f} vs "
        f"{cf.loc['llm', 'elasticity_flexible']:.3f}; whole-household "
        f"{cf.loc['isotonic', 'elasticity_total']:.3f}, still in the Conway range). "
        f"Degree-hours are similar to the LLM's on the baseline day and higher on the CPP day, "
        f"and max exceedance is worse ({cf.loc['isotonic', 'cpp_max_exceedance_c']:.2f} °C, "
        f"flexible household 8208263). So the LLM's low post-CPP ratio depends on its "
        f"non-monotone and flat WTP, not only on the average level of its valuations. Once the "
        f"valuation is monotone, the result looks like the rule-based agent.",
        "",
        "## 8. `rebound_delta_kw`  *(exploratory, diagnostic)*",
        "",
        "**Definition.** `hems.analysis.metrics.rebound_peak_kw` "
        "([src/hems/analysis/metrics.py:82-89](../../../../src/hems/analysis/metrics.py#L82-L89)) "
        "= 2 × max half-hour kWh in the `rebound_hours` (= 2, `config/experiment.yaml:163`) "
        "after the CPP end, i.e. the peak kW of 20:00–22:00. `load_metrics` "
        "([src/hems/phase3_report.py:297-336](../../../../src/hems/phase3_report.py#L297-L336)) "
        "computes it per (scope, series, day), then at lines 330–333 pivots on day and "
        "merges `cpp − base` **back onto both day rows**. So `rebound_delta_kw` is one number "
        "per (scope, series) = rebound_peak(CPP day) − rebound_peak(baseline day), repeated "
        "on each day's row.",
        "",
        md(reb),
        "",
        f"**Why MILP shows 35.54 on both days.** It is a single cross-day difference printed "
        f"twice: {mr.loc['milp', 'cpp_rebound_peak_kw']:.3f} kW (CPP day, peak at "
        f"{mr.loc['milp', 'cpp_peak_step']}, when 22 of 24 MILP ACs switch on together: "
        f"44 kW of HVAC) − {mr.loc['milp', 'base_rebound_peak_kw']:.3f} kW (baseline day, "
        f"peak at {mr.loc['milp', 'base_peak_step']}, zero HVAC — pure SGSC base load) = 35.54. "
        "Nothing on the baseline day itself is 35.54.",
        "",
        "**Is it valid?** Only in a narrow sense. The arithmetic is correct: it is the increase "
        "in the highest 20:00–22:00 kW on the CPP day versus the same window on the baseline "
        "day. As a rebound measure it has four problems:",
        "",
        "1. The per-day rows are misleading; the value belongs to neither day.",
        "2. It compares two peaks that can fall at different times (20:00 vs 21:00), and a "
        "single half-hour decides it. With 2 kW on/off ACs it jumps in 2 kW steps.",
        "3. Each agent's baseline day already has its own post-20:00 peak from the 21:00 "
        "peak→shoulder price drop. For the satisficers that peak is large (rule-based 101.5 kW "
        "at 21:00), so their delta (5.54) is small even though 19 households switch on at 20:00. "
        "MILP's baseline window is flat, so its delta is large. The metric therefore does not "
        "put agents on a common scale.",
        "4. It is not a pre-registered metric. H3 is stated on the 20:00–20:30 load and its "
        "ratio to the CPP-window mean (`summary.csv`), which avoid 1 and 2.",
        "",
        "Use the pre-registered post-CPP metrics for H3. If a rebound quantity is wanted, the "
        "HVAC-energy difference (CPP − base) for 20:00–22:00 (last column) is more robust.",
        "",
        "## 9. Figures",
        "",
        "- `fig_wtp_vs_c_above_ceiling.png`: LLM WTP vs °C above ceiling, one panel per "
        "orientation (decisions, bin means with CI, isotonic fit).",
        "- `fig_switch_on_histogram.png`: first AC switch-on at/after 20:00, LLM vs rule-based, "
        "CPP day (baseline day for reference).",
        "- `fig_elasticity_by_orientation.png`: pooled HVAC and whole-household elasticity by "
        "orientation with 95% CIs; LLM, rule-based, MILP, isotonic counterfactual.",
        "- `fig_comfort_distribution.png`: per-household degree-hours and max exceedance by "
        "agent and day, coloured by orientation.",
        "",
        "![](fig_wtp_vs_c_above_ceiling.png)",
        "![](fig_switch_on_histogram.png)",
        "![](fig_elasticity_by_orientation.png)",
        "![](fig_comfort_distribution.png)",
        "",
    ]
    (OUT / "analysis.md").write_text("\n".join(L))


# --------------------------------------------------------------------------
# 10b. analysis.md for any other run: same tables, prose generated from the data
# --------------------------------------------------------------------------


def _rng(vals, nd=2) -> str:
    v = [x for x in vals if not np.isnan(x)]
    return f"{min(v):.{nd}f} to {max(v):.{nd}f}" if v else "–"


def _h1_robust_ok(T) -> bool:
    return all(h1_verdict(t, "")[["order_holds_point", "prio-bal_CI_excl_0",
                                  "bal-flex_CI_excl_0"]].all().all()
               for t in (T["h1_base_only"], T["h1_dedup"]))  # fmt: skip


def write_md_generic(T) -> None:
    c = T["checks"]
    h1, e, cf, reb = T["h1"], T["elasticity"], T["cf_summary"].set_index("agent"), T["rebound"]
    mono = T["mono"].set_index("group")
    sw = T["switch"]
    cpp_sw = sw[sw["day"] == "cpp"]
    at20 = cpp_sw[cpp_sw["first_switch_on_after_20"] == "20:00"].groupby("agent").size()
    ev = e.set_index(["agent", "comfort_orientation"])
    dl = T["decisions_llm"]
    resolved = yaml.safe_load((RUN / "config_resolved.yaml").read_text())
    model = resolved["llm"][resolved["llm"]["provider"]]["model"]
    prov = json.loads((RUN / "provenance.json").read_text())
    lm = LEVEL_MATCHED
    agents_c = ["llm", "rule_based", "isotonic"] + ([LM] if lm else [])

    # H1 facts
    ha = h1[h1["side"] == "above"]
    ok = ha.dropna(subset=["prioritising_mean_wtp", "balanced_mean_wtp", "flexible_mean_wtp"])
    v = h1_verdict(h1, "")
    bf = ha.dropna(subset=["balanced_mean_wtp", "flexible_mean_wtp"])
    bf_pos = int((bf["balanced_mean_wtp"] > bf["flexible_mean_wtp"]).sum())
    bf_ci = int((bf["bal_minus_flex"].str.extract(r"\[([-\d.]+),")[0].astype(float) > 0).sum())
    below = dl[dl["over_c"] <= 0].groupby("orientation")["wtp"].mean()
    prio_max = dl.loc[dl["orientation"] == "comfort_prioritising", "over_c"].max()
    # comfort facts
    ch = T["comfort_hh"]
    gt = T["comfort_gt25"]
    gl = gt[gt["agent"] == "llm"]
    base_gt = sorted(gl.loc[gl["day"] == "base", "household_id"].unique())
    cori = T["comfort_ori"]
    dh = lambda a, d, g: cori[(cori["agent"] == a) & (cori["day"] == d)  # noqa: E731
                              & (cori["comfort_orientation"] == g)]["degree_hours_mean"].iloc[0]
    # archetype facts
    rec = T["records"]
    e_arch = {(a, x): pooled_e(r["q_base_hvac"], r["q_cpp_hvac"])
              for (a, x), r in rec.groupby(["agent", "archetype"])}  # fmt: skip
    thr = T["thresholds"].set_index("archetype")
    personas = pd.read_csv(RUN / "personas_used.csv")
    flex_a = int(((personas["comfort_orientation"] == "comfort_flexible")
                  & (personas["archetype"] == "A")).sum())  # fmt: skip
    prio_a = int(((personas["comfort_orientation"] == "comfort_prioritising")
                  & (personas["archetype"] == "A")).sum())  # fmt: skip
    # rebound / WTP facts
    po = T["post_ori"]
    flex_share = po[(po["day"] == "cpp") & (po["orientation"] == "flexible")].set_index("agent")[
        "share_of_agent_post_hvac"]
    wv = T["wtp_values"]
    top = wv.sort_values("all", ascending=False).head(3)
    dis = T["discord"]
    mr = reb.set_index("agent")

    L = [
        f"# {LABEL} — post-run analysis",
        "",
        f"Generated by `experiments/analyze_final_v1.py --label {LABEL}` from the saved artifacts "
        f"in `{RUN}/` (model `{model}`; run commit `{prov['commit'][:7]}` on branch "
        f"`{prov['branch']}`). Same code, tables and bootstrap as the final_v1 analysis; the "
        "prose summaries here are generated from the numbers (the final_v1 analysis carries "
        f"hand-checked wording). No live LLM calls: every LLM answer is replayed from `{CACHE}`, "
        "with the client patched to raise on any cache miss or write. Nothing outside "
        "`analysis/` was written.",
        "",
        "Every result is labelled **pre-registered** or **exploratory (post-hoc)**. For a "
        "pre-registered optional arm (docs/prereg_final_run_v1.md, *Optional arms* and "
        "*Deviations*) the pre-registered metrics and H1 keep that label; everything the final_v1 "
        "analysis labelled exploratory stays exploratory. Bootstrap CIs: households resampled "
        f"with replacement, {N_BOOT:,} draws, seed {BOOT_SEED}, percentile 95% intervals. n per "
        "orientation is imbalanced (prioritising 5 / balanced 8 / flexible 11).",
        "",
        "## 0. Offline regeneration",
        "",
        "Every household × agent × seed × day was re-simulated from the current code with the "
        f"run's own model id (`{model}`, read from `config_resolved.yaml`) and the LLM replayed "
        "from the run's cache. All checks below must hold before anything else is computed.",
        "",
        md(pd.DataFrame([
            (f"curves.parquet reproduced exactly ({c['curves_rows']:,} rows)", c["curves_identical"]),
            (f"llm_decision_log reproduced exactly ({c['llm_log_rows']:,} rows)", c["llm_log_identical"]),
            ("rule_decision_log reproduced exactly", c["rule_log_identical"]),
            ("summary.csv recomputed from replay identical", c["summary_csv_identical"]),
            ("live LLM calls during replay", c["live_calls"]),
            ("cache hits during replay", c["cache_hits"]),
            ("cache file unchanged (sha256 before = after)", c["cache_unchanged"]),
            ("max |prompt temperature − replayed pre-decision temperature| (°C)",
             round(T["pretemp_vs_prompt_max_abs_err"], 4)),
        ], columns=["check", "result"])),
        "",
        "## 1. H1 — LLM raw WTP by comfort orientation at matched temperature  *(pre-registered)*",
        "",
        "Raw LLM WTP (before archetype factor), both days pooled, 0.5 °C bins of exact "
        "pre-decision °C above the 26 °C ceiling (right-closed). Decision-weighted bin means; CI "
        "resamples households within orientation; differences resample each orientation "
        "independently.",
        "",
        md(h1[H1_COLS]),
        "",
        "### H1 robustness  *(exploratory (post-hoc) robustness check; same bins and bootstrap)*",
        "",
        "(a) Baseline day only:",
        "",
        md(T["h1_base_only"][H1_COLS]),
        "",
        f"(b) One decision per distinct prompt ({T['h1_dedup_n']['distinct_prompts']:,} of "
        f"{T['h1_dedup_n']['decisions']:,}; first occurrence household → baseline → time):",
        "",
        md(T["h1_dedup"][H1_COLS]),
        "",
        "Ordering check in bins where all three orientations are observed:",
        "",
        md(pd.concat([h1_verdict(h1, "pooled (pre-registered)"),
                      h1_verdict(T["h1_base_only"], "(a) base day only"),
                      h1_verdict(T["h1_dedup"], "(b) distinct prompts")], ignore_index=True), 2),
        "",
        "All above-ceiling decisions per orientation (not temperature-matched; context only):",
        "",
        md(T["h1_pooled"]),
        "",
        f"**Summary.** At or below the ceiling mean WTP is "
        f"{below.get('comfort_prioritising', np.nan):.3f} (prioritising) / "
        f"{below.get('balanced', np.nan):.3f} (balanced) / "
        f"{below.get('comfort_flexible', np.nan):.3f} (flexible). Above it, all three "
        f"orientations are observed in {len(ok)} bin(s) ({', '.join(ok['C_above_ceiling'])}); "
        f"the order prioritising > balanced ≥ flexible holds at the point estimate in "
        f"{int(v['order_holds_point'].sum())} of {len(v)}, prioritising − balanced has a CI "
        f"above zero in {int(v['prio-bal_CI_excl_0'].sum())} and balanced − flexible in "
        f"{int(v['bal-flex_CI_excl_0'].sum())}. Prioritising households never get more than "
        f"{prio_max:.2f} °C over; across all {len(bf)} above-ceiling bins with balanced and "
        f"flexible, balanced > flexible in {bf_pos} (CI above zero in {bf_ci}). "
        + ("Robustness: base-day-only and distinct-prompt variants leave the ordering and both "
           "CI exclusions unchanged in every testable bin." if _h1_robust_ok(T) else
           "Robustness: the ordering or a CI exclusion changes in at least one robustness "
           "variant — see the ordering check."),
        "",
        "## 2. Per-orientation elasticity  *(exploratory, post-hoc)*",
        "",
        "Pooled arc elasticity (Σ CPP-window kWh, base method) over households in the group; "
        "primary seed. `isotonic` = item-7 counterfactual"
        + ("; `rule_based_lm` = item 7b." if lm else "."),
        "",
        md(e[["agent", "comfort_orientation", "n", "e_hvac_ci", "e_total_ci",
              "q_base_hvac_kwh", "q_cpp_hvac_kwh"]]),
        "",
        "**Summary.** LLM HVAC elasticity: " + ", ".join(
            f"{ORI_SHORT[o]} {ev.loc[('llm', o), 'e_hvac']:.2f}" for o in ORIENTATIONS)
        + f" (all {ev.loc[('llm', 'all'), 'e_hvac']:.2f}); rule-based: " + ", ".join(
            f"{ORI_SHORT[o]} {ev.loc[('rule_based', o), 'e_hvac']:.2f}" for o in ORIENTATIONS)
        + f" (all {ev.loc[('rule_based', 'all'), 'e_hvac']:.2f}); MILP "
        f"{_rng([ev.loc[('milp', o), 'e_hvac'] for o in ORIENTATIONS])} across orientations. "
        f"Orientation is confounded with archetype ({flex_a} of 11 flexible households are "
        f"archetype A, {prio_a} prioritising), so these are not clean orientation effects.",
        "",
        "## 3. Comfort per household  *(metrics pre-registered; breakdowns and the > 2.5 °C list "
        "exploratory)*",
        "",
        "Degree-hours = Σ max(0, indoor − 26 °C) × 0.5 h per household-day (end-of-step "
        "temperatures); mean over households with bootstrap CI. MILP is 0 everywhere.",
        "",
        "By orientation:",
        "",
        md(T["comfort_ori"], 2),
        "",
        "By archetype:",
        "",
        md(T["comfort_arch"], 2),
        "",
        "Household-days with max exceedance > 2.5 °C:",
        "",
        md(gt[["agent", "household_id", "day", "comfort_orientation", "archetype",
               "max_exceedance_c", "degree_hours", "steps_above"]], 2),
        "",
        "**Summary.** LLM mean degree-hours on the CPP day: " + ", ".join(
            f"{ORI_SHORT[o]} {dh('llm', 'cpp', o)}" for o in ORIENTATIONS)
        + f" (rule-based: " + ", ".join(
            f"{ORI_SHORT[o]} {dh('rule_based', 'cpp', o)}" for o in ORIENTATIONS) + "). "
        + " ".join(
            f"{AGENT_LABEL[a]}: {int((gt['agent'] == a).sum())} household-days > 2.5 °C "
            f"({int(((gt['agent'] == a) & (gt['day'] == 'base')).sum())} on the baseline day)."
            for a in agents_c)
        + (f" LLM households above 2.5 °C already on the baseline day (so not caused by the CPP "
           f"event): {', '.join(map(str, base_gt))}." if base_gt else ""),
        "",
        "## 4. Archetype decomposition and halt bands  *(exploratory, post-hoc)*",
        "",
        "Orientation × archetype, pooled HVAC elasticity (n = households):",
        "",
        md(T["ori_x_arch"]),
        "",
        "Halt: run iff WTP × factor ≥ 2 kW × 0.5 h × rate, so a 17:00–20:00 decision responds "
        "to CPP only if raw WTP is in [0.55 / factor, 1.20 / factor). LLM WTPs given in that "
        "window:",
        "",
        md(T["thresholds"]),
        "",
        _details("Per-household HVAC kWh, 17:00–20:00, baseline vs CPP (LLM and rule-based)",
                 md(T["hvac_window_hh"])),
        "",
        "**Summary.** Pooled LLM HVAC elasticity by archetype: " + ", ".join(
            f"{x} {e_arch.get(('llm', x), np.nan):.2f}" for x in ARCHETYPES)
        + " (rule-based: " + ", ".join(
            f"{x} {e_arch.get(('rule_based', x), np.nan):.2f}" for x in ARCHETYPES) + "). "
        "Baseline-day LLM decisions in 17:00–20:00 whose raw WTP falls in the price-responsive "
        "band: " + ", ".join(
            f"{x} {thr.loc[x, 'base_day_decisions_in_responsive_band']} "
            f"(band ${thr.loc[x, 'run_at_peak_if_wtp>=']:.2f}–{thr.loc[x, 'run_at_cpp_if_wtp>=']:.2f})"
            for x in ARCHETYPES)
        + ". An archetype responds only to the extent the LLM's WTP values land inside its band.",
        "",
        "## 5. Rebound mechanism, CPP day  *(exploratory, post-hoc)*",
        "",
        "AC state at steps 38–43 (19:00–21:30; 1 = on) and first off→on switch at or after "
        "20:00. Rate after the event: 0.55 until 21:00, then 0.28, 0.22 from 22:00.",
        "",
        _details("Per household, CPP day",
                 md(cpp_sw.sort_values(["agent", "orientation", "household_id"])[
                     ["agent", "household_id", "orientation", "archetype", "ac_19:00-21:30 (t38-43)",
                      "indoor_at_20:00_c", "first_switch_on_after_20", "hvac_kw_20:00",
                      "hvac_kwh_20-22"]], 2)),
        "",
        "Switch-on time distribution (households; both days):",
        "",
        md(pd.crosstab([sw["agent"], sw["day"]], sw["first_switch_on_after_20"]).reset_index()),
        "",
        "Post-CPP contribution by orientation:",
        "",
        md(T["post_ori"]),
        "",
        "**Summary.** On the CPP day, households switching on at 20:00: " + ", ".join(
            f"{AGENT_LABEL[a]} {int(at20.get(a, 0))}" for a in agents_c)
        + " (of 24). Flexible households' share of the agent's 20:00–20:30 HVAC load: " + ", ".join(
            f"{AGENT_LABEL[a]} {flex_share.get(a, np.nan):.0%}" for a in agents_c + ["milp"]) + ".",
        "",
        "## 6. WTP non-monotonicity  *(pre-registered as descriptive under H4; rate definition "
        "post-hoc)*",
        "",
        "Consecutive steps where °C above ceiling rises and both are above the ceiling; rate = "
        "share where WTP falls. Upper: exact temperature; lower: the 0.1 °C value shown.",
        "",
        md(T["mono"]),
        "",
        md(T["mono_shown"]),
        "",
        _details("Per household-day (rows with ≥ 1 fall)",
                 md(T["mono_hh"][T["mono_hh"]["wtp_fell"] > 0])),
        "",
        "Across all above-ceiling decisions within an orientation (pairs with different "
        "temperature): share where the hotter one got the lower WTP, and share with equal WTP.",
        "",
        md(T["discord"]),
        "",
        f"Distinct raw WTP values (all {len(dl):,} HVAC decisions):",
        "",
        md(wv, 3),
        "",
        f"**Summary.** {len(wv)} distinct WTP values; the top three "
        f"({', '.join(f'${x:g}' for x in top['raw_wtp_$'])}) make up "
        f"{top['share_all'].sum():.0%} of decisions. Above the ceiling, WTP falls as the house "
        f"heats {mono.loc['all', 'wtp_fell']} times in {mono.loc['all', 'pairs']} "
        f"({mono.loc['all', 'fall_rate']}); by orientation " + ", ".join(
            f"{ORI_SHORT[o]} {mono.loc[o, 'wtp_fell']}/{mono.loc[o, 'pairs']}" for o in ORIENTATIONS)
        + ". Shown-value rate: " + T["mono_shown"].set_index("group").loc["all", "fall_rate"]
        + f". Cross-sectionally, {_rng(dis['hotter_lower_wtp_share'] * 100, 0)}% of pairs have "
        f"the hotter state valued lower and {_rng(dis['hotter_equal_wtp_share'] * 100, 0)}% equal.",
        "",
        "## 7. Counterfactual: monotone version of the LLM's own valuation  *(exploratory, "
        "post-hoc)*",
        "",
        "Per orientation, an isotonic (pool-adjacent-violators) fit of raw LLM WTP on exact °C "
        "above ceiling over all LLM HVAC decisions, run as an agent through the same halt, "
        "factor, simulator, tariffs and households. No LLM. In-sample; it visits states the LLM "
        "did not.",
        "",
        md(cf.reset_index()[["agent", "cpp_post_20:00-20:30_kw", "cpp_cpp_window_mean_kw",
                             "cpp_post_over_cpp_ratio", "base_post_over_cpp_ratio",
                             "base_par", "cpp_par",
                             "base_degree_hours_above_ceiling", "cpp_degree_hours_above_ceiling",
                             "base_max_exceedance_c", "cpp_max_exceedance_c",
                             "elasticity_flexible", "elasticity_total"]]),
        "",
        _details("Fitted step function (isotonic_fit_steps.csv)", md(T["iso_fit_steps"], 4)),
        "",
        f"**Summary.** CPP-day post/CPP ratio: isotonic "
        f"{cf.loc['isotonic', 'cpp_post_over_cpp_ratio']:.3f}, LLM "
        f"{cf.loc['llm', 'cpp_post_over_cpp_ratio']:.3f}, rule-based "
        f"{cf.loc['rule_based', 'cpp_post_over_cpp_ratio']:.3f}, MILP "
        f"{cf.loc['milp', 'cpp_post_over_cpp_ratio']:.3f}. HVAC elasticity isotonic "
        f"{cf.loc['isotonic', 'elasticity_flexible']:.3f} vs LLM "
        f"{cf.loc['llm', 'elasticity_flexible']:.3f}; CPP-day degree-hours (Σ 24 households) isotonic "
        f"{cf.loc['isotonic', 'cpp_degree_hours_above_ceiling']:.1f} vs LLM "
        f"{cf.loc['llm', 'cpp_degree_hours_above_ceiling']:.1f}; CPP-day max exceedance "
        f"{cf.loc['isotonic', 'cpp_max_exceedance_c']:.2f} vs "
        f"{cf.loc['llm', 'cpp_max_exceedance_c']:.2f} °C.",
        "",
    ]
    if lm:
        fit = T["lm_fit"]
        L += [
            "## 7b. Level-matched rule-based agent  *(exploratory, post-hoc fairness control)*",
            "",
            "The run's rule-based agent uses k frozen before the run, fitted to "
            "gemini-3.5-flash-lite's WTP in the comfort sweeps. Here k is refitted to **this "
            "run's** LLM WTP by the same method (`experiments/fit_rule_based.py::fit_hvac`: least "
            "squares through the origin of raw WTP on max(0, indoor shown in the prompt − 26 °C), "
            "one row per distinct prompt, per orientation; fixed ±2.0 band; rounded to 3 dp as "
            "the frozen values are). The agent then runs offline on all 24 households, both "
            "days, through the same halt and archetype factors. The same code path with the "
            "frozen k reproduces the saved rule-based curves exactly (checked).",
            "",
            md(fit, 4),
            "",
            md(cf.reset_index().set_index("agent").loc[["llm", LM, "rule_based"]].reset_index()[
                ["agent", "cpp_post_over_cpp_ratio", "base_par", "cpp_par",
                 "cpp_degree_hours_above_ceiling", "cpp_max_exceedance_c",
                 "elasticity_flexible", "elasticity_total"]]),
            "",
            "Per-orientation elasticity and comfort for `rule_based_lm` are in items 2 and 3.",
            "",
            f"**Summary.** Level-matched k = " + " / ".join(
                f"{T['lm_k'][o]:.3f}" for o in ORIENTATIONS)
            + " (prioritising / balanced / flexible) vs frozen " + " / ".join(
                f"{x:.3f}" for x in fit["k_frozen_fixed_band"]) + ". "
            f"Against the LLM: post/CPP ratio {cf.loc[LM, 'cpp_post_over_cpp_ratio']:.3f} vs "
            f"{cf.loc['llm', 'cpp_post_over_cpp_ratio']:.3f}; HVAC elasticity "
            f"{cf.loc[LM, 'elasticity_flexible']:.3f} vs "
            f"{cf.loc['llm', 'elasticity_flexible']:.3f}; CPP-day degree-hours (Σ 24 households) "
            f"{cf.loc[LM, 'cpp_degree_hours_above_ceiling']:.1f} vs "
            f"{cf.loc['llm', 'cpp_degree_hours_above_ceiling']:.1f}; CPP-day max exceedance "
            f"{cf.loc[LM, 'cpp_max_exceedance_c']:.2f} vs "
            f"{cf.loc['llm', 'cpp_max_exceedance_c']:.2f} °C.",
            "",
        ]
    L += [
        "## 8. `rebound_delta_kw`  *(exploratory, diagnostic)*",
        "",
        "`rebound_delta_kw` = peak kW of 20:00–22:00 on the CPP day − the same on the baseline "
        "day (one number per agent, printed on both day rows of `load_metrics.csv`; see the "
        "final_v1 analysis, item 8, for why it is not a common-scale rebound measure). The "
        "HVAC-energy difference for 20:00–22:00 (last column) is the more robust quantity.",
        "",
        md(reb),
        "",
        f"MILP: {mr.loc['milp', 'cpp_rebound_peak_kw']:.3f} kW (CPP, "
        f"{mr.loc['milp', 'cpp_peak_step']}) − {mr.loc['milp', 'base_rebound_peak_kw']:.3f} kW "
        f"(baseline, {mr.loc['milp', 'base_peak_step']}); LLM: "
        f"{mr.loc['llm', 'cpp_rebound_peak_kw']:.3f} ({mr.loc['llm', 'cpp_peak_step']}) − "
        f"{mr.loc['llm', 'base_rebound_peak_kw']:.3f} ({mr.loc['llm', 'base_peak_step']}).",
        "",
        "## 9. Figures",
        "",
        "![](fig_wtp_vs_c_above_ceiling.png)",
        "![](fig_switch_on_histogram.png)",
        "![](fig_elasticity_by_orientation.png)",
        "![](fig_comfort_distribution.png)",
        "",
    ]
    (OUT / "analysis.md").write_text("\n".join(L))


def configure(argv=None) -> argparse.Namespace:
    """Point the module at a run (defaults = final run v1)."""
    global RUN, OUT, CACHE, CACHE_LINES, LABEL, LEVEL_MATCHED
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", default=str(RUN), help="saved run folder (read-only)")
    ap.add_argument("--cache-path", default=str(CACHE), help="the run's LLM cache (.jsonl)")
    ap.add_argument("--cache-lines", type=int, default=CACHE_LINES,
                    help="expected cache line count (asserted); -1 to skip")  # fmt: skip
    ap.add_argument("--label", default=LABEL, help="run label (final_v1 keeps its prose)")
    ap.add_argument("--out", default=None, help="output folder (default <run-dir>/analysis)")
    ap.add_argument("--level-matched", action="store_true",
                    help="exploratory: add a rule-based agent refit to this run's LLM WTP")  # fmt: skip
    a = ap.parse_args(argv)
    RUN, CACHE, LABEL = Path(a.run_dir), Path(a.cache_path), a.label
    OUT = Path(a.out) if a.out else RUN / "analysis"
    CACHE_LINES = None if a.cache_lines == -1 else a.cache_lines
    LEVEL_MATCHED = a.level_matched
    return a


if __name__ == "__main__":
    configure()
    T = compute()
    figures(T)
    (write_md if LABEL == "final_v1" and not LEVEL_MATCHED else write_md_generic)(T)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 80)
    for k in ["h1", "h1_pooled", "elasticity", "comfort_ori", "comfort_arch", "comfort_gt25",
              "ori_x_arch", "thresholds", "post_ori", "mono", "mono_shown", "discord", "wtp_values",
              "iso_fit_steps", "cf_summary", "rebound"]:
        print(f"\n=== {k}\n", T[k].to_string())
    print(T["switch"].groupby(["agent", "day", "first_switch_on_after_20"]).size())
