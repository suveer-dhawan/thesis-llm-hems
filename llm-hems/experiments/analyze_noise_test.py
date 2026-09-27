"""Context vs dispersion: is the LLM's rebound/comfort advantage just noise?  (EXPLORATORY)

For each model (final_v1 = gemini-3.5-flash-lite, Arm A = gemini-3.8-flash):

  1. Context check. Raw LLM WTP above the ceiling regressed on °C over the ceiling
     (as shown in the prompt) x orientation, then clock (hour of day), minutes
     above the ceiling so far that day, and the indoor-temperature change since the
     previous step. R² (in-sample, adjusted, household-grouped 5-fold CV) per added
     feature, plus a direct test of the one non-temperature field that varies in
     the prompt (the clock) inside cells of identical orientation x shown °C.
  2. Noise-matched rule. The run's level-matched rule (`rule_based_lm`, k refitted
     to the run's own LLM WTP) plus per-decision noise drawn i.i.d. from that run's
     empirical residuals (LLM WTP - rule WTP at the LLM's own states), resampled
     within orientation x side of the ceiling, WTP clipped at 0. 50 seeds, 24
     households, both days, offline. Sensitivity: resampled within orientation only.
  3. Household-shuffled residuals. Each household-day gets the whole residual
     sequence (in time order, residual at step t applied at step t) of a random
     other household of the same orientation on the same day label. 50 seeds.
  4. Clouds added to the rebound-vs-comfort scatter.

No live LLM calls: the client is patched to raise on any cache miss / cache write
(no LLM is used at all -- the noise agents are pure Python); both caches and both
run folders are hashed before and after and must be unchanged. Writes only
results/phase3/cross_model/noise_test*.

    PYTHONPATH=src .venv/bin/python experiments/analyze_noise_test.py

Everything here is exploratory (post-hoc); nothing is pre-registered.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from hems.agents.base import Decision, DecisionContext  # noqa: E402
from hems.agents.llm_agent import forced_washer_outcome, satisficing_halt  # noqa: E402
from hems.agents.rule_based_agent import hvac_over_ceiling_c  # noqa: E402
from hems.environment.simulator import run_simulation  # noqa: E402
from hems.phase3 import DayRun, Phase3Config, Phase3Runner  # noqa: E402
from hems.phase3_report import CURVE_SORT, agent_summary, curves_from_runs  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "analyze_final_v1", Path(__file__).with_name("analyze_final_v1.py")
)
A = importlib.util.module_from_spec(_spec)
sys.modules["analyze_final_v1"] = A  # dataclasses look their module up here
_spec.loader.exec_module(A)

OUT = Path("results/phase3/cross_model")
RUNS = {  # key -> (run dir, model, cache)
    "flash-lite": (Path("results/phase3/final_v1"), "gemini-3.5-flash-lite"),
    "3.8-flash": (Path("results/phase3/arm_a"), "gemini-3.8-flash"),
}
SEEDS = list(range(50))
N_FOLDS = 5
DAYS = ("base", "cpp")
ORIS = A.ORIENTATIONS
VARIANTS = {  # name -> (rng stream id, label)
    "iid_ori_side": (2, "noise-matched rule: i.i.d. residuals within orientation × side of ceiling"),
    "iid_ori": (4, "sensitivity: i.i.d. residuals within orientation only"),
    "hh_shuffled": (3, "household-shuffled residual sequences (time order kept)"),
}
SHORT = {"iid_ori_side": "i.i.d. noise", "iid_ori": "i.i.d. noise (orientation only)",
         "hh_shuffled": "household-shuffled noise"}  # fmt: skip
CLOUD_COLOR = {"iid_ori_side": "#4a3aa7", "hh_shuffled": "#e34948"}  # palette slots 7, 8
CLOUD_MARKER = {"iid_ori_side": "o", "hh_shuffled": "D"}
AGENT_COLOR = {**A.AGENT_COLOR, "noisy_milp": "#008300"}
INK, INK2 = A.INK, A.INK2
md = A.md
EXPL = "*(exploratory, post-hoc)*"


def _hash_tree(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_file():
                h.update(str(f).encode())
                h.update(f.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load(key: str) -> dict:
    run, model = RUNS[key]
    resolved = yaml.safe_load((run / "config_resolved.yaml").read_text())
    assert resolved["llm"]["gemini"]["model"] == model
    cfg = Phase3Config.from_yaml("config/experiment.yaml", band_mode="fixed",
                                 cache_path=resolved["phase3"]["llm_cache_path"])  # fmt: skip
    cfg.llm_cfg["gemini"]["model"] = model
    assert cfg.resolved()["phase3"] == resolved["phase3"]
    runner = Phase3Runner(cfg)
    curves = pd.read_parquet(run / "curves.parquet")
    llm_log = pd.read_parquet(run / "llm_decision_log.parquet")
    personas = pd.read_csv(run / "personas_used.csv")
    ids = sorted(int(h) for h in curves["household_id"].unique())
    # level-matched k, exactly as the cross-model analysis (which also checks that the
    # same code path with the frozen k reproduces the saved rule-based curves)
    k, fit, lm_curves = A.level_matched(runner, llm_log, curves, personas, ids)
    saved_k = pd.read_csv(OUT / "lm_fit.csv").query("model == @key").set_index("orientation")
    assert all(np.isclose(k[o], saved_k.loc[o, "k_level_matched"]) for o in ORIS)

    # decisions with their exact pre-decision state (from the verified replay)
    s = pd.read_csv(run / "analysis" / "llm_decisions_with_state.csv")
    s = s[s["appliance"] == "hvac"].sort_values(["household_id", "day", "t"]).reset_index(drop=True)
    assert len(s) == len(ids) * 2 * 48
    p = llm_log[llm_log["appliance"] == "hvac"][["household_id", "run_label", "t", "prompt", "wtp"]]
    s = s.merge(p.rename(columns={"run_label": "day", "wtp": "wtp_log"}),
                on=["household_id", "day", "t"], validate="1:1")  # fmt: skip
    assert (s["wtp"] == s["wtp_log"]).all()
    g = s.groupby(["household_id", "day"], sort=False)
    s["above"] = s["over_c"] > 0
    s["hour"] = s["t"] // 2
    s["min_above_so_far"] = 30 * (g["above"].cumsum() - s["above"])  # prior steps only
    s["dtemp_c"] = (s["pre_indoor_temp_c"] - g["pre_indoor_temp_c"].shift(1)).fillna(0.0)
    s["k"] = s["orientation"].map(k)
    s["rule_wtp"] = s["k"] * s["over_c"].clip(lower=0)  # what the rule emits at this state
    s["resid"] = s["wtp"] - s["rule_wtp"]
    return {"key": key, "run": run, "model": model, "resolved": resolved, "runner": runner,
            "ids": ids, "k": k, "lm_curves": lm_curves, "state": s,
            "llm_summary": pd.read_csv(run / "summary.csv").query("agent == 'llm'").iloc[0]}  # fmt: skip


# --------------------------------------------------------------------------
# 1. context check
# --------------------------------------------------------------------------


def _design(d: pd.DataFrame, feats: list[str]) -> np.ndarray:
    cols = [pd.get_dummies(d["orientation"]).astype(float)]  # orientation intercepts
    if "temp" in feats:
        cols.append(pd.get_dummies(d["orientation"]).astype(float).mul(d["shown_over_c"], axis=0)
                    .add_prefix("slope_"))  # fmt: skip
    if "temp_cells" in feats:  # saturated: one mean per orientation x shown °C
        cols = [pd.get_dummies(d["orientation"] + "|" + d["shown_over_c"].round(1).astype(str))
                .astype(float)]  # fmt: skip
    if "clock" in feats:
        cols.append(pd.get_dummies(d["hour"], prefix="h", drop_first=True).astype(float))
    if "min_above" in feats:
        cols.append(d[["min_above_so_far"]].astype(float))
    if "dtemp" in feats:
        cols.append(d[["dtemp_c"]].astype(float))
    return pd.concat(cols, axis=1).to_numpy(float)


def _folds(d: pd.DataFrame) -> np.ndarray:
    """Household-grouped folds, stratified by orientation (seeded)."""
    rng = np.random.default_rng(42)
    fold = {}
    for o in ORIS:
        hh = np.array(sorted(d.loc[d["orientation"] == o, "household_id"].unique()))
        rng.shuffle(hh)
        for i, h in enumerate(hh):
            fold[h] = i % N_FOLDS
    return d["household_id"].map(fold).to_numpy()


def _r2(y, pred) -> float:
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def fit_stats(d: pd.DataFrame, feats: list[str], cv: bool = True) -> dict:
    X, y = _design(d, feats), d["wtp"].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = _r2(y, X @ beta)
    p, n = np.linalg.matrix_rank(X), len(y)
    out = {"n": n, "params": int(p), "R2": r2, "adj_R2": 1 - (1 - r2) * (n - 1) / (n - p)}
    if cv:
        f, pred = _folds(d), np.empty(n)
        for i in range(N_FOLDS):
            tr, te = f != i, f == i
            b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            pred[te] = X[te] @ b
        out["CV_R2"] = _r2(y, pred)
    return out


STEPS = [("°C over ceiling (shown) × orientation", ["temp"]),
         ("+ clock (hour of day)", ["temp", "clock"]),
         ("+ minutes above ceiling so far", ["temp", "clock", "min_above"]),
         ("+ Δ indoor temp since last step", ["temp", "clock", "min_above", "dtemp"])]  # fmt: skip


def context(R: dict) -> dict:
    d = R["state"][R["state"]["above"]].reset_index(drop=True)
    rows, prev = [], None
    for label, feats in STEPS:
        st = fit_stats(d, feats)
        rows.append({"model": R["key"], "model spec": label, **st,
                     "ΔR2": st["R2"] - prev["R2"] if prev else np.nan,
                     "ΔCV_R2": st["CV_R2"] - prev["CV_R2"] if prev else np.nan})  # fmt: skip
        prev = st
    full = STEPS[-1][1]
    for f, name in [("clock", "clock"), ("min_above", "minutes above so far"), ("dtemp", "Δ temp")]:
        st = fit_stats(d, [x for x in full if x != f])
        rows.append({"model": R["key"], "model spec": f"full − {name} (unique contribution)",
                     **st, "ΔR2": st["R2"] - prev["R2"], "ΔCV_R2": st["CV_R2"] - prev["CV_R2"]})  # fmt: skip
    # nonlinear temperature ceiling, and what clock adds on top (in-sample only)
    sat = fit_stats(d, ["temp_cells"], cv=False)
    sat_c = fit_stats(d, ["temp_cells", "clock"], cv=False)
    rows.append({"model": R["key"], "model spec": "saturated: one mean per orientation × shown °C",
                 **sat, "ΔR2": np.nan, "ΔCV_R2": np.nan})  # fmt: skip
    rows.append({"model": R["key"], "model spec": "saturated + clock", **sat_c,
                 "ΔR2": sat_c["R2"] - sat["R2"], "ΔCV_R2": np.nan})  # fmt: skip

    # the prompt has only persona, clock and indoor °C: within cells of identical
    # orientation x shown °C seen at >= 2 clocks, does the answer change with the clock?
    dd = d.drop_duplicates("prompt")
    cells = dd.groupby(["orientation", "shown_over_c"])
    multi = cells.filter(lambda g: g["clock"].nunique() >= 2)
    cg = multi.groupby(["orientation", "shown_over_c"])
    within = (multi["wtp"] - cg["wtp"].transform("mean")).pow(2).sum()
    total = (multi["wtp"] - multi["wtp"].mean()).pow(2).sum()
    clock_row = {"model": R["key"], "distinct above-ceiling prompts": len(dd),
                 "cells (orientation × shown °C) seen at ≥2 clocks": cg.ngroups,
                 "prompts in those cells": len(multi),
                 "cells where WTP differs by clock": int((cg["wtp"].nunique() > 1).sum()),
                 "share of WTP variance in those cells that is within-cell (clock)":
                     within / total if total else np.nan}  # fmt: skip

    # direction of the clock effect: mean residual of the base model by time block
    X = _design(d, ["temp"])
    res = d["wtp"] - X @ np.linalg.lstsq(X, d["wtp"].to_numpy(float), rcond=None)[0]
    blocks = pd.cut(d["hour"], [-1, 11, 16, 19, 20, 23],
                    labels=["00–12", "12–17", "17–20", "20–21", "21–24"])
    blk = res.groupby(blocks, observed=False).agg(["mean", "count"])
    block_row = {"model": R["key"], **{f"{b} mean resid $ (n)": f"{m:+.3f} ({int(c)})"
                                       for b, (m, c) in blk.iterrows()}}  # fmt: skip
    return {"r2": pd.DataFrame(rows), "clock": clock_row, "blocks": block_row,
            "resid_20_21": float(blk.loc["20–21", "mean"])}


# --------------------------------------------------------------------------
# 2-3. noise agents
# --------------------------------------------------------------------------


class NoisyRuleAgent:
    """Level-matched rule WTP + a residual, clipped at 0, through the same Python halt
    and forced-washer handling as every other WTP agent. `eps(t, orientation, above)`."""

    def __init__(self, k: dict, eps, wtp_scale: float, wtp_factor: float):
        self.k, self.eps = k, eps
        self.wtp_scale, self.wtp_factor = float(wtp_scale), float(wtp_factor)
        self.trace: list[tuple] = []

    def decide(self, ctx: DecisionContext) -> Decision:
        a = ctx.appliance
        if a["type"] == "washer":
            forced = forced_washer_outcome(ctx)
            assert forced is not None, "washer is disabled in this run"
            return Decision(run=forced[0], wtp=None, reason=forced[1])
        ori = ctx.household["comfort_orientation"]
        x = hvac_over_ceiling_c(a)
        wtp = max(0.0, self.k[ori] * x + self.eps(ctx.t, ori, x > 0))
        _, run = satisficing_halt(wtp, ctx.cost_now, self.wtp_scale, self.wtp_factor)
        self.trace.append((ori, x, wtp))
        return Decision(run=run, wtp=wtp, reason="noise-matched rule")

    def name(self) -> str:
        return "noisy_rule"


def simulate(R: dict, variant: str | None, seed: int) -> tuple[pd.DataFrame, list]:
    runner, cfg, s = R["runner"], R["runner"].cfg, R["state"]
    ori_of = s.groupby("household_id")["orientation"].first()
    stream = VARIANTS[variant][0] if variant else 0
    pools = {(o, side): s.loc[(s["orientation"] == o) & (s["above"] == side), "resid"].to_numpy()
             for o in ORIS for side in (False, True)}  # fmt: skip
    pools_o = {o: s.loc[s["orientation"] == o, "resid"].to_numpy() for o in ORIS}
    series = {(h, d): g.sort_values("t")["resid"].to_numpy() for (h, d), g in
              s.groupby(["household_id", "day"])}  # fmt: skip
    runs, trace = [], []
    for hid in R["ids"]:
        household, hvac, washer = runner.env(hid)
        for date in cfg.dates:
            for di, label in enumerate(DAYS):
                rng = np.random.default_rng([stream, seed, hid, di])
                if variant is None:
                    eps = lambda t, o, up: 0.0  # noqa: E731  (must reproduce rule_based_lm)
                elif variant == "iid_ori_side":
                    eps = lambda t, o, up, r=rng: float(r.choice(pools[(o, up)]))  # noqa: E731
                elif variant == "iid_ori":
                    eps = lambda t, o, up, r=rng: float(r.choice(pools_o[o]))  # noqa: E731
                else:  # whole sequence of another household of the same orientation, same day
                    donors = [h for h in R["ids"] if h != hid and ori_of[h] == ori_of[hid]]
                    seq = series[(int(rng.choice(donors)), label)]
                    eps = lambda t, o, up, q=seq: float(q[t])  # noqa: E731
                agent = NoisyRuleAgent(R["k"], eps, cfg.global_scale, runner.wtp_factor(hid))
                res = run_simulation(household, runner.tariff_for(date, label), hvac, washer,
                                     agent, date)  # fmt: skip
                runs.append(DayRun(hid, variant or "zero_noise", seed, cfg.global_scale, date,
                                   label, res))  # fmt: skip
                trace += agent.trace
    orientation = runner.assignment.set_index("customer_ID")["comfort_orientation"].to_dict()
    return curves_from_runs(runs, runner.archetype, orientation.get), trace


def noise_runs(R: dict) -> pd.DataFrame:
    # zero noise must reproduce the level-matched rule exactly
    zc, _ = simulate(R, None, 42)
    lm = A._sort(R["lm_curves"], CURVE_SORT)
    zc = A._sort(zc.assign(agent=A.LM), CURVE_SORT)[lm.columns]
    pd.testing.assert_frame_equal(zc, lm, check_exact=True, check_dtype=False)
    R["lm_summary"] = agent_summary(R["lm_curves"], R["resolved"]).iloc[0]
    rows, wtp_check = [], []
    for v in VARIANTS:
        curves, traces = [], []
        for seed in SEEDS:
            c, tr = simulate(R, v, seed)
            curves.append(c)
            traces += tr
        summ = agent_summary(pd.concat(curves, ignore_index=True), R["resolved"])
        assert len(summ) == len(SEEDS)
        rows.append(summ.assign(model=R["key"], variant=v))
        tr = pd.DataFrame(traces, columns=["orientation", "over_c", "wtp"])
        for o in ORIS:
            a = tr[(tr["orientation"] == o) & (tr["over_c"] > 0)]
            ll = R["state"][(R["state"]["orientation"] == o) & R["state"]["above"]]
            wtp_check.append({"model": R["key"], "variant": v, "orientation": A.ORI_SHORT[o],
                              "mean WTP above ceiling, LLM": ll["wtp"].mean(),
                              "mean WTP above ceiling, noise agent": a["wtp"].mean(),
                              "SD WTP above ceiling, LLM": ll["wtp"].std(),
                              "SD WTP above ceiling, noise agent": a["wtp"].std(),
                              "share clipped at 0 (noise agent, above ceiling)":
                                  (a["wtp"] == 0).mean()})  # fmt: skip
    R["wtp_check"] = pd.DataFrame(wtp_check)
    return pd.concat(rows, ignore_index=True)


METRICS = [("cpp_post_over_cpp_ratio", "post/CPP ratio"),
           ("cpp_degree_hours_above_ceiling", "CPP degree-hours"),
           ("elasticity_flexible", "e HVAC")]  # fmt: skip


def pct_rank(cloud: np.ndarray, v: float) -> float:
    """Percentile of v in the cloud (share below, ties counted half), in %."""
    return 100 * (np.mean(cloud < v) + 0.5 * np.mean(cloud == v))


def summarise(R: dict, seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    llm, lm = R["llm_summary"], R["lm_summary"]
    for v in VARIANTS:
        a = seeds[(seeds["model"] == R["key"]) & (seeds["variant"] == v)]
        row = {"model": R["key"], "variant": SHORT[v], "seeds": len(a)}
        for c, name in METRICS:
            x = a[c].to_numpy(float)
            q5, q50, q95 = np.percentile(x, [5, 50, 95])
            row[f"{name} median [5–95%]"] = f"{q50:.3f} [{q5:.3f}, {q95:.3f}]"
            row[f"{name} LLM"] = llm[c]
            row[f"{name} LLM pct"] = pct_rank(x, llm[c])
            row[f"{name} rule_lm"] = lm[c]
        dom = (a["cpp_post_over_cpp_ratio"] <= llm["cpp_post_over_cpp_ratio"]) & (
            a["cpp_degree_hours_above_ceiling"] <= llm["cpp_degree_hours_above_ceiling"])
        row["seeds at least as good as LLM on both"] = f"{int(dom.sum())}/{len(a)}"
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. figure
# --------------------------------------------------------------------------


def fig(runs, seeds, path) -> None:
    pts = pd.read_csv(OUT / "scatter.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    names = {"milp": "MILP", "noisy_milp": "noisy-MILP", "rule_based": "rule-based (frozen k)",
             "llm": "LLM", "isotonic": "isotonic", A.LM: "rule-based, level-matched"}  # fmt: skip
    for ax, R in zip(axes, runs):
        for v in CLOUD_COLOR:
            a = seeds[(seeds["model"] == R["key"]) & (seeds["variant"] == v)]
            ax.scatter(a["cpp_degree_hours_above_ceiling"], a["cpp_post_over_cpp_ratio"], s=22,
                       marker=CLOUD_MARKER[v], color=CLOUD_COLOR[v], alpha=0.55, lw=0, zorder=2,
                       label=f"{SHORT[v]} (50 seeds)")  # fmt: skip
        p = pts[pts["model"].isin(["shared", R["key"]])]
        for q in p.itertuples(index=False):
            ax.scatter(q.x, q.y, s=110 if q.agent == "llm" else 80, marker="s" if q.model == "shared"
                       else "o", color=AGENT_COLOR[q.agent], edgecolor="white", linewidth=2,
                       zorder=4)  # fmt: skip
            if q.agent == "noisy_milp" and q.seed != 43:
                continue
            lab = names[q.agent] + (" (seeds 42–44)" if q.agent == "noisy_milp" else "")
            left = q.agent == "rule_based"  # keeps the label off the clouds
            ax.annotate(lab, (q.x, q.y), xytext=(-8 if left else 8, 4 if q.agent != "noisy_milp"
                        else -12), textcoords="offset points", ha="right" if left else "left",
                        fontsize=8, color=INK if q.agent == "llm" else INK2,
                        fontweight="bold" if q.agent == "llm" else None)  # fmt: skip
        ax.set_title(f"{R['model']} ({R['key']})", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("CPP-day degree-hours above ceiling (°C·h, Σ 24 households)", color=INK2,
                      fontsize=9)  # fmt: skip
        A._style(ax, "both")
        ax.margins(x=0.2, y=0.1)
        ax.set_xlim(left=-15)
    axes[0].set_ylabel("CPP-day post/CPP ratio (20:00–20:30 kW ÷ 17:00–20:00 mean kW)",
                       color=INK2, fontsize=9)  # fmt: skip
    fig.suptitle("Rebound vs comfort — noise-matched rule clouds vs the LLM (exploratory)",
                 color=INK, fontsize=12, x=0.01, ha="left")  # fmt: skip
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def verdict(R: dict, summ: pd.DataFrame) -> str:
    r = summ[summ["model"] == R["key"]].set_index("variant")
    llm = R["llm_summary"]
    out = []
    for v in ("iid_ori_side", "hh_shuffled"):
        x = r.loc[SHORT[v]]
        pr, pd_ = x["post/CPP ratio LLM pct"], x["CPP degree-hours LLM pct"]
        out.append(f"{SHORT[v]}: LLM ratio {llm['cpp_post_over_cpp_ratio']:.3f} sits at "
                   f"percentile {pr:.0f} of the cloud ({x['post/CPP ratio median [5–95%]']}), "
                   f"degree-hours {llm['cpp_degree_hours_above_ceiling']:.1f} at percentile {pd_:.0f} "
                   f"({x['CPP degree-hours median [5–95%]']}); seeds at least as good on both: "
                   f"{x['seeds at least as good as LLM on both']}")  # fmt: skip
    return "; ".join(out)


def plain(R: dict, summ: pd.DataFrame, ctx: dict, r2: pd.DataFrame) -> str:
    """Plain-language verdict, worded from the numbers."""
    r = summ[summ["model"] == R["key"]].set_index("variant")
    llm, lm = R["llm_summary"], R["lm_summary"]
    ra, dh = "cpp_post_over_cpp_ratio", "cpp_degree_hours_above_ceiling"
    iid, sh = r.loc[SHORT["iid_ori_side"]], r.loc[SHORT["hh_shuffled"]]

    def med(x, name):
        return float(x[f"{name} median [5–95%]"].split()[0])

    def lo(x, name):
        return float(x[f"{name} median [5–95%]"].split("[")[1].split(",")[0])

    gap = lm[ra] - llm[ra]
    c_iid = (lm[ra] - med(iid, "post/CPP ratio")) / gap
    c_sh = (lm[ra] - med(sh, "post/CPP ratio")) / gap
    iid_out = llm[ra] < lo(iid, "post/CPP ratio")
    sh_out = llm[ra] < lo(sh, "post/CPP ratio")
    dh_in = iid["CPP degree-hours LLM pct"] >= 5
    rr = r2[r2["model"] == R["key"]].set_index("model spec")
    clock_gain = rr.loc["+ clock (hour of day)", "ΔR2"]
    return (
        f"**{R['key']} — "
        + ("the LLM sits outside the i.i.d. noise cloud" if iid_out
           else "the i.i.d. noise cloud reaches the LLM")
        + (", but inside the household-shuffled cloud.** " if not sh_out
           else " and outside the household-shuffled cloud too.** ")
        + f"Adding the LLM's own residuals to its level-matched rule as independent "
        f"per-decision noise moves the rule's post/CPP ratio from {lm[ra]:.3f} to a median "
        f"{med(iid, 'post/CPP ratio'):.3f} ({c_iid:.0%} of the way to the LLM's {llm[ra]:.3f}) "
        f"and its CPP degree-hours from {lm[dh]:.1f} to {med(iid, 'CPP degree-hours'):.1f} "
        f"(LLM {llm[dh]:.1f}"
        + (", i.e. the noise cloud matches the LLM's comfort" if dh_in else "")
        + f"); {'no' if iid['post/CPP ratio LLM pct'] == 0 else 'few'} seeds reach the LLM's "
        f"ratio and {iid['seeds at least as good as LLM on both']} match it on both axes. "
        f"When the same residuals keep their time order (borrowed from another household of the "
        f"same orientation) the cloud moves to a median {med(sh, 'post/CPP ratio'):.3f} "
        f"({c_sh:.0%} of the gap) and the LLM falls at percentile "
        f"{sh['post/CPP ratio LLM pct']:.0f} on the ratio and "
        f"{sh['CPP degree-hours LLM pct']:.0f} on degree-hours — "
        + ("inside the cloud. " if not sh_out else "still below it. ")
        + "So random dispersion alone does **not** reproduce the LLM's rebound advantage; "
        + ("dispersion that keeps the LLM's time-ordered structure largely does, and that "
           "structure is shared across households of an orientation rather than tied to a "
           "household's identity. "
           if not sh_out else "even time-ordered residuals do not. ")
        + ("The context check agrees: " if abs(ctx["resid_20_21"]) >= 0.05 else
           "The context check gives only weak support for a clock mechanism here: ")
        + f"the clock is the only non-temperature field that varies "
        f"in the prompt; it adds little overall (R² +{clock_gain:.3f}; minutes-above-ceiling "
        f"and Δ temp add ≈ 0, as they must, since neither is in the prompt), but "
        f"{ctx['clock']['cells where WTP differs by clock']} of "
        f"{ctx['clock']['cells (orientation × shown °C) seen at ≥2 clocks']} identical "
        f"orientation × °C cells get different answers at different times, and at 20:00–21:00 "
        f"the LLM's WTP is {'+' if ctx['resid_20_21'] >= 0 else '−'}${abs(ctx['resid_20_21']):.3f} "
        f"relative to its temperature fit"
        + ("." if abs(ctx["resid_20_21"]) >= 0.05 else
           " — small, so for this model the time-ordered cloud's success may owe as much to "
           "the residuals following the temperature path (the linear rule's misfit along a "
           "trajectory) as to the clock.")
    )


def write_md(runs, T) -> None:
    ctx = T["context_r2"]
    num = ["n", "params", "R2", "adj_R2", "CV_R2", "ΔR2", "ΔCV_R2"]
    sm = T["summary"]
    L = [
        "# Context vs dispersion — is the LLM's advantage reproducible by noise?  *(EXPLORATORY)*",
        "",
        "**Everything in this file is exploratory (post-hoc); nothing here is pre-registered.** "
        "Generated by `experiments/analyze_noise_test.py` from the saved artifacts of "
        "`results/phase3/final_v1` (gemini-3.5-flash-lite) and `results/phase3/arm_a` "
        "(gemini-3.8-flash) and their `analysis/` folders. No LLM is called (the client is "
        "patched to raise on any cache miss or write; the noise agents are pure Python); both "
        "caches and both run folders were hashed before and after and are unchanged.",
        "",
        "## 1. Context check — what does the LLM's WTP depend on beyond temperature?  " + EXPL,
        "",
        "**What the prompt contains.** Every HVAC prompt is one of 6 templates (3 personas × "
        "within/over/under the band). The only fields that vary between decisions are the "
        "persona sentence (= orientation), the clock (`CURRENT STATE AT hh:mm`), the indoor "
        "temperature (0.1 °C) and the derived 'X °C OVER the ceiling' fact; the band is fixed "
        "(22–26 °C). There is **no price, no history, no minutes-above-ceiling and no previous "
        "temperature** in the prompt, and the answer is a deterministic function of the prompt "
        "(T = 0 + cache: every repeated prompt has exactly one WTP in both runs). So minutes "
        "above ceiling and Δ temperature can only matter through their correlation with the "
        "clock and the shown temperature — they are included as requested, as a check on that.",
        "",
        "Rows: HVAC decisions above the ceiling (exact pre-decision temperature > 26 °C), both "
        "days. Dependent variable: raw WTP ($). Base model: an intercept and a slope on shown "
        "°C over the ceiling per orientation. Clock = hour-of-day dummies. Minutes above "
        "ceiling so far = 30 × prior steps that day above the ceiling. Δ temp = pre-decision "
        "indoor °C minus the previous step's. CV R² = household-grouped 5-fold "
        "(orientation-stratified) out-of-fold R². Saturated = one mean per orientation × shown "
        "°C cell (the most any temperature-only function can explain; in-sample).",
        "",
        md(ctx[["model", "model spec"] + num], 3),
        "",
        "**Direct test of the clock.** Distinct prompts only; cells of identical orientation × "
        "shown °C that were seen at two or more clock times — any WTP difference inside a cell "
        "is due to the clock alone (the only other varying field).",
        "",
        md(pd.DataFrame(T["clock_cells"]), 3),
        "",
        "Mean residual of the base model (temperature × orientation) by time of day, $ (n "
        "decisions):",
        "",
        md(pd.DataFrame(T["clock_blocks"])),
        "",
        "## 2–3. Noise-matched rule and household-shuffled residuals  " + EXPL,
        "",
        "Each model's level-matched rule (`rule_based_lm`, k refitted to that run's own LLM "
        "WTP; " + "; ".join(f"{R['key']}: " + " / ".join(f"{R['k'][o]:.3f}" for o in ORIS)
                            for R in runs)
        + " for prioritising / balanced / flexible) emits WTP = max(0, k·(°C over ceiling) + "
        "ε) and goes through the same Python halt and archetype factor as the LLM. Residuals "
        "are LLM WTP − rule WTP at every HVAC decision of that model's run (the LLM's own "
        "states, both days, all 48 steps; the rule WTP uses the exact temperature as the rule "
        "agent does). With ε = 0 the code reproduces the saved `rule_based_lm` curves exactly "
        "(asserted).",
        "",
        "- **i.i.d. noise** (the noise-matched rule): ε drawn with replacement, independently "
        "per decision, from the residuals of the same orientation **and the same side of the "
        "ceiling** (above vs at/below). Stratifying by side keeps above-ceiling residuals "
        "(which can be ±$1–2 for prioritising) out of in-band states, where the LLM itself "
        "answers ≈ $0.05.",
        "- **i.i.d. noise (orientation only)** — sensitivity, the literal 'within orientation' "
        "pool with both sides mixed.",
        "- **household-shuffled noise**: every household-day takes the complete 48-step "
        "residual sequence of a randomly drawn *other* household of the same orientation on "
        "the same day label, residual at step t applied at step t (keeps the LLM's temporal "
        "structure, breaks household identity). Donors are drawn per seed.",
        "",
        "50 seeds each, 24 households, both days. Clouds: median [5th–95th percentile] over "
        "seeds. `LLM pct` = percentile of the actual LLM value within the cloud (share of seeds "
        "below it, ties half; 0 = better than every seed for ratio and degree-hours, since "
        "lower is better on both). `rule_lm` = the zero-noise level-matched rule.",
        "",
        md(sm[["model", "variant", "seeds"] + [c for c in sm.columns if c.startswith("post/CPP")]]),
        "",
        md(sm[["model", "variant"] + [c for c in sm.columns if c.startswith("CPP degree")]
              + ["seeds at least as good as LLM on both"]]),  # fmt: skip
        "",
        md(sm[["model", "variant"] + [c for c in sm.columns if c.startswith("e HVAC")]]),
        "",
        "Level check — the noise agents' realised WTP above the ceiling vs the LLM's (pooled "
        "over seeds; the noise agents visit their own states, so this need not match exactly):",
        "",
        md(T["wtp_check"], 3),
        "",
        "## 4. Rebound vs comfort with the noise clouds  " + EXPL,
        "",
        "![](fig_rebound_vs_comfort_noise.png)",
        "",
        "Per-seed values: `noise_test_seeds.csv`. Points are the same as in "
        "`fig_rebound_vs_comfort.png` (cross_model.md item 9); the orientation-only "
        "sensitivity cloud is in the tables but not drawn.",
        "",
        "## 5. Verdict  " + EXPL,
        "",
    ]
    L += T["verdict_text"]
    (OUT / "noise_test.md").write_text("\n".join(L) + "\n")


def main() -> None:
    guarded = [RUNS[k][0] for k in RUNS] + [Path("results/phase3/llm_cache_final.jsonl"),
                                            Path("results/phase3/llm_cache_arm_a.jsonl")]
    before = _hash_tree(guarded)
    A._no_live_calls()
    runs = [load(k) for k in RUNS]
    T: dict = {}
    ctx = [context(R) for R in runs]
    T["context_r2"] = pd.concat([c["r2"] for c in ctx], ignore_index=True)
    T["clock_cells"] = [c["clock"] for c in ctx]
    T["clock_blocks"] = [c["blocks"] for c in ctx]
    seeds = pd.concat([noise_runs(R) for R in runs], ignore_index=True)
    T["summary"] = pd.concat([summarise(R, seeds) for R in runs], ignore_index=True)
    T["wtp_check"] = pd.concat([R["wtp_check"] for R in runs], ignore_index=True)
    fig(runs, seeds, OUT / "fig_rebound_vs_comfort_noise.png")
    T["verdict_text"] = [plain(R, T["summary"], c, T["context_r2"]) + "\n"
                         for R, c in zip(runs, ctx)]
    T["verdict_text"] += ["Numbers behind the verdict:", ""] + [
        f"- **{R['key']}**: {verdict(R, T['summary'])}." for R in runs]
    T["verdict_text"] += ["", "Caveats: the residual pools include the linear rule's misfit "
                          "(LLM WTP is flatter in °C than k·x, notably for prioritising), so "
                          "i.i.d. resampling also scrambles that systematic part — the noise "
                          "agents' WTP SD above the ceiling is larger than the LLM's for "
                          "prioritising (level check above). The household-shuffled variant "
                          "keeps each donor's residual at its own clock time, so it carries "
                          "both the clock pattern and the donor's temperature path. One "
                          "LLM run per model (seed 42, T = 0): the LLM point has no seed "
                          "spread of its own."]
    keep = ["model", "variant", "seed", "cpp_post_over_cpp_ratio", "cpp_degree_hours_above_ceiling",
            "cpp_max_exceedance_c", "cpp_par", "base_par", "elasticity_flexible", "elasticity_total"]
    seeds[keep].to_csv(OUT / "noise_test_seeds.csv", index=False)
    T["summary"].to_csv(OUT / "noise_test_summary.csv", index=False)
    T["context_r2"].to_csv(OUT / "noise_test_context_r2.csv", index=False)
    pd.DataFrame(T["clock_cells"]).to_csv(OUT / "noise_test_clock_cells.csv", index=False)
    T["wtp_check"].to_csv(OUT / "noise_test_wtp_check.csv", index=False)
    write_md(runs, T)
    assert _hash_tree(guarded) == before, "a run folder or cache changed"
    print((OUT / "noise_test.md").read_text())


if __name__ == "__main__":
    main()
