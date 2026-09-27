"""Cross-model comparison: final run v1 (gemini-3.5-flash-lite) vs Arm A (gemini-3.8-flash).

Reads each run's saved artifacts and its `analysis/` folder (written by
experiments/analyze_final_v1.py) and writes results/phase3/cross_model/. The
only thing simulated here is the level-matched rule-based agent (rule-based k
refitted to each run's own LLM WTP, `analyze_final_v1.level_matched`), which
needs no LLM; the LLM client is still patched to refuse any live call or cache
write. Run folders are never written.

    PYTHONPATH=src .venv/bin/python experiments/analyze_cross_model.py

Every table is labelled: per-model results keep their pre-registered /
exploratory label; the model-vs-model comparison itself is exploratory (the
pre-registration reports Arm A separately and makes no cross-model prediction).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from hems.phase3 import Phase3Config, Phase3Runner  # noqa: E402
from hems.phase3_report import agent_summary  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "analyze_final_v1", Path(__file__).with_name("analyze_final_v1.py")
)
A = importlib.util.module_from_spec(_spec)
sys.modules["analyze_final_v1"] = A  # dataclasses look their module up here
_spec.loader.exec_module(A)

OUT = Path("results/phase3/cross_model")
RUNS = {  # key -> (run dir, short model name)
    "flash-lite": (Path("results/phase3/final_v1"), "gemini-3.5-flash-lite"),
    "3.8-flash": (Path("results/phase3/arm_a"), "gemini-3.8-flash"),
}
SHARED = ["milp", "noisy_milp", "rule_based"]  # no LLM: identical in both runs (asserted)
MARKER = {"flash-lite": "o", "3.8-flash": "^"}
AGENT_COLOR = {**A.AGENT_COLOR, "noisy_milp": "#008300"}  # slot 6
AGENT_LABEL = {**A.AGENT_LABEL, "noisy_milp": "noisy-MILP",
               "rule_based": "rule-based (frozen k)",
               "rule_based_lm": "rule-based, level-matched to own run"}  # fmt: skip
INK, INK2 = A.INK, A.INK2
md, fmt = A.md, A.fmt


# --------------------------------------------------------------------------
# per-run loading + level-matched agent
# --------------------------------------------------------------------------


def load_run(key: str) -> dict:
    run, model = RUNS[key]
    an = run / "analysis"
    resolved = yaml.safe_load((run / "config_resolved.yaml").read_text())
    assert resolved["llm"]["gemini"]["model"] == model, (key, resolved["llm"]["gemini"]["model"])
    R = {"key": key, "run": run, "model": model, "resolved": resolved}
    for name in ["h1_wtp_by_orientation_bin", "h1_wtp_by_orientation_bin_base_day_only",
                 "h1_wtp_by_orientation_bin_dedup_prompt", "wtp_distinct_values",
                 "monotonicity_summary", "monotonicity_summary_shown_values",
                 "elasticity_by_orientation", "counterfactual_summary",
                 "switch_on_per_household", "comfort_per_household"]:  # fmt: skip
        R[name] = pd.read_csv(an / f"{name}.csv")
    R["summary"] = pd.read_csv(run / "summary.csv")
    return R


def add_level_matched(R: dict) -> None:
    """Level-matched rule-based agent for this run, simulated offline (no LLM needed)."""
    run = R["run"]
    A._no_live_calls()
    cfg = Phase3Config.from_yaml("config/experiment.yaml", band_mode="fixed",
                                 cache_path=R["resolved"]["phase3"]["llm_cache_path"])  # fmt: skip
    cfg.llm_cfg["gemini"]["model"] = R["model"]
    assert cfg.resolved()["phase3"] == R["resolved"]["phase3"]
    runner = Phase3Runner(cfg)
    curves = pd.read_parquet(run / "curves.parquet")
    llm_log = pd.read_parquet(run / "llm_decision_log.parquet")
    personas = pd.read_csv(run / "personas_used.csv")
    ids = sorted(int(h) for h in curves["household_id"].unique())
    k, fit, lm = A.level_matched(runner, llm_log, curves, personas, ids)
    R["lm_k"], R["lm_fit"] = k, fit
    summ = agent_summary(lm, R["resolved"])
    R["lm_summary"] = summ
    rec = A.per_household_records(lm)
    R["lm_elasticity"] = A.elasticity_by(rec, "comfort_orientation", [A.LM])
    R["lm_switch"] = A.switch_on(lm, [A.LM])
    R["lm_comfort_hh"] = A.comfort_per_household(lm)
    # the run's own analysis (if it included the level-matched agent) must agree
    own = R["counterfactual_summary"]
    if (own["agent"] == A.LM).any():
        a = own[own["agent"] == A.LM].iloc[0]
        for c in ["cpp_post_over_cpp_ratio", "cpp_degree_hours_above_ceiling", "elasticity_flexible"]:
            assert np.isclose(a[c], summ.iloc[0][c], atol=1e-9), c
    # a run analysed with --level-matched already carries these rows: they must equal the
    # recomputed ones, then are dropped so every table uses the recomputed copy exactly once
    for name, mine in [("switch_on_per_household", R["lm_switch"]),
                       ("comfort_per_household", R["lm_comfort_hh"]),
                       ("elasticity_by_orientation", R["lm_elasticity"])]:  # fmt: skip
        df = R[name]
        if (df["agent"] == A.LM).any():
            theirs = df[df["agent"] == A.LM].reset_index(drop=True)
            num = theirs.select_dtypes("number").columns
            assert np.allclose(theirs[num].to_numpy(float), mine[num].to_numpy(float),
                               equal_nan=True), name  # fmt: skip
        R[name] = df[df["agent"] != A.LM].reset_index(drop=True)
    # every agent's pre-registered summary row, plus isotonic + level-matched
    iso = R["counterfactual_summary"].query("agent == 'isotonic'")
    R["agents"] = pd.concat([R["summary"], iso, summ], ignore_index=True)


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def h1_side_by_side(runs) -> pd.DataFrame:
    rows = []
    for R in runs:
        h = R["h1_wtp_by_orientation_bin"]
        for r in h[h["side"] == "above"].itertuples(index=False):
            d = r._asdict()
            rows.append({"model": R["key"], "bin (°C over)": d["C_above_ceiling"],
                         "n_hh p/b/f": f"{d['prioritising_n_hh']}/{d['balanced_n_hh']}/{d['flexible_n_hh']}",
                         "prioritising": d["prioritising_ci"], "balanced": d["balanced_ci"],
                         "flexible": d["flexible_ci"], "prio − bal": d["prio_minus_bal"],
                         "bal − flex": d["bal_minus_flex"]})  # fmt: skip
    return pd.DataFrame(rows)


def _verdict(h: pd.DataFrame, label: str) -> pd.DataFrame:
    h = h.copy()
    h["H1_order_point"] = h["H1_order_point"].map({True: True, False: False, "True": True,
                                                   "False": False})  # fmt: skip
    h["prio>bal_CI_excl_0"] = h["prio>bal_CI_excl_0"].map({True: True, False: False,
                                                           "True": True, "False": False})  # fmt: skip
    return A.h1_verdict(h, label)


def h1_verdicts(runs) -> pd.DataFrame:
    rows = []
    for R in runs:
        for name, label in [("h1_wtp_by_orientation_bin", "pooled (pre-registered)"),
                            ("h1_wtp_by_orientation_bin_base_day_only", "(a) base day only"),
                            ("h1_wtp_by_orientation_bin_dedup_prompt", "(b) distinct prompts")]:
            v = _verdict(R[name], label)
            rows.append({"model": R["key"], "variant": label,
                         "bins with all 3 orientations": ", ".join(v["bin"]),
                         "order holds (point)": f"{int(v['order_holds_point'].sum())}/{len(v)}",
                         "prio−bal CI > 0": f"{int(v['prio-bal_CI_excl_0'].sum())}/{len(v)}",
                         "bal−flex CI > 0": f"{int(v['bal-flex_CI_excl_0'].sum())}/{len(v)}"})  # fmt: skip
        h = R["h1_wtp_by_orientation_bin"]
        bf = h[(h["side"] == "above")].dropna(subset=["balanced_mean_wtp", "flexible_mean_wtp"])
        lo = bf["bal_minus_flex"].str.extract(r"\[([-\d.]+),")[0].astype(float)
        rows.append({"model": R["key"], "variant": "balanced vs flexible, all above-ceiling bins",
                     "bins with all 3 orientations": ", ".join(bf["C_above_ceiling"]),
                     "order holds (point)": f"{int((bf['balanced_mean_wtp'] > bf['flexible_mean_wtp']).sum())}/{len(bf)}",
                     "prio−bal CI > 0": "–", "bal−flex CI > 0": f"{int((lo > 0).sum())}/{len(bf)}"})  # fmt: skip
    return pd.DataFrame(rows)


def wtp_table(runs) -> pd.DataFrame:
    rows = []
    for R in runs:
        w = R["wtp_distinct_values"]
        top = w.sort_values("all", ascending=False).head(3)
        above = w[w["all_above_ceiling"] > 0]
        m, ms = (R["monotonicity_summary"].set_index("group"),
                 R["monotonicity_summary_shown_values"].set_index("group"))  # fmt: skip
        rows.append({"model": R["key"], "distinct WTP values": len(w),
                     "distinct above ceiling": len(above),
                     "top-3 values": ", ".join(f"${v:g}" for v in top["raw_wtp_$"]),
                     "top-3 share": top["share_all"].sum(),
                     "non-monotone rate (exact °C)": m.loc["all", "fall_rate"],
                     "falls / pairs": f"{m.loc['all', 'wtp_fell']}/{m.loc['all', 'pairs']}",
                     "non-monotone rate (shown °C)": ms.loc["all", "fall_rate"],
                     **{f"falls/pairs {A.ORI_SHORT[o]}": f"{m.loc[o, 'wtp_fell']}/{m.loc[o, 'pairs']}"
                        for o in A.ORIENTATIONS}})  # fmt: skip
    return pd.DataFrame(rows)


AGENT_ORDER = ["milp", "noisy_milp", "rule_based", "llm", "isotonic", A.LM]


def system_table(runs) -> pd.DataFrame:
    rows = []
    for R in runs:
        a = R["agents"]
        for agent in AGENT_ORDER:
            for _, d in a[a["agent"] == agent].iterrows():
                rows.append({
                    "model": R["key"] if agent not in SHARED else f"{R['key']} (shared)",
                    "agent": agent, "seed": d["seed"],
                    "post/CPP ratio (CPP day)": d["cpp_post_over_cpp_ratio"],
                    "post 20:00-20:30 kW": d["cpp_post_20:00-20:30_kw"],
                    "PAR base": d["base_par"], "PAR CPP": d["cpp_par"],
                    "PAR base→CPP": d["cpp_par"] - d["base_par"],
                    "degree-h base": d["base_degree_hours_above_ceiling"],
                    "degree-h CPP": d["cpp_degree_hours_above_ceiling"],
                    "max exceed. base": d["base_max_exceedance_c"],
                    "max exceed. CPP": d["cpp_max_exceedance_c"],
                    "e HVAC": d["elasticity_flexible"], "e total": d["elasticity_total"],
                })  # fmt: skip
    return pd.DataFrame(rows)


def elasticity_table(runs) -> pd.DataFrame:
    rows = []
    for R in runs:
        e = pd.concat([R["elasticity_by_orientation"], R["lm_elasticity"]], ignore_index=True)
        for agent in ["llm", "isotonic", A.LM, "rule_based", "milp"]:
            for r in e[e["agent"] == agent].itertuples(index=False):
                d = r._asdict()
                rows.append({"model": R["key"] if agent not in SHARED else f"{R['key']} (shared)",
                             "agent": agent, "group": A.ORI_SHORT.get(d["comfort_orientation"], "all"),
                             "n": d["n"], "e HVAC [95% CI]": d["e_hvac_ci"],
                             "e total [95% CI]": d["e_total_ci"]})  # fmt: skip
    return pd.DataFrame(rows)


def comfort_per_orientation(runs) -> pd.DataFrame:
    """Mean per-household CPP-day degree-hours / max exceedance, by orientation."""
    rows = []
    for R in runs:
        ch = pd.concat([R["comfort_per_household"], R["lm_comfort_hh"]], ignore_index=True)
        ch = ch[(ch["day"] == "cpp") & (ch["seed"] == 42)]
        for agent in ["llm", "isotonic", A.LM, "rule_based"]:
            a = ch[ch["agent"] == agent]
            row = {"model": R["key"] if agent not in SHARED else f"{R['key']} (shared)",
                   "agent": agent}
            for o in ["all"] + A.ORIENTATIONS:
                g = a if o == "all" else a[a["comfort_orientation"] == o]
                s = "all" if o == "all" else A.ORI_SHORT[o]
                row[f"degree-h/hh {s}"] = A.ci(*A.boot_mean(g["degree_hours"]), 2)
            row["max exceed. (max)"] = a["max_exceedance_c"].max()
            row["hh-days > 2.5 °C"] = int((a["max_exceedance_c"] > 2.5).sum())
            rows.append(row)
    return pd.DataFrame(rows)


SLOTS = ["20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "none"]


def switch_table(runs) -> pd.DataFrame:
    rows = []
    for R in runs:
        s = pd.concat([R["switch_on_per_household"], R["lm_switch"]], ignore_index=True)
        s = s[s["day"] == "cpp"]
        for agent in ["llm", "isotonic", A.LM, "rule_based"]:
            c = s.loc[s["agent"] == agent, "first_switch_on_after_20"].value_counts()
            rows.append({"model": R["key"] if agent not in SHARED else f"{R['key']} (shared)",
                         "agent": agent, **{k: int(c.get(k, 0)) for k in SLOTS}})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def fig_scatter(runs, path) -> pd.DataFrame:
    pts = []
    for R in runs:
        for _, d in R["agents"].iterrows():
            if d["agent"] in SHARED and R is not runs[0]:
                continue  # identical in both runs: draw once
            pts.append({"model": "shared" if d["agent"] in SHARED else R["key"],
                        "agent": d["agent"], "seed": d["seed"],
                        "x": d["cpp_degree_hours_above_ceiling"],
                        "y": d["cpp_post_over_cpp_ratio"]})  # fmt: skip
    pts = pd.DataFrame(pts)
    fig, ax = plt.subplots(figsize=(9, 6))
    for p in pts.itertuples(index=False):
        mk = MARKER.get(p.model, "s")
        ax.scatter(p.x, p.y, s=90, marker=mk, color=AGENT_COLOR[p.agent], edgecolor="white",
                   linewidth=2, zorder=3)  # fmt: skip
    # direct labels (several light hues are < 3:1 on white, so every point is labelled)
    offsets = {}
    for p in pts.itertuples(index=False):
        if p.agent == "noisy_milp" and p.seed != 43:
            continue  # one label for the three seeds
        name = {"milp": "MILP", "noisy_milp": "noisy-MILP (seeds 42–44)",
                "rule_based": "rule-based (frozen k)", "llm": "LLM", "isotonic": "isotonic",
                A.LM: "rule-based, level-matched"}[p.agent]  # fmt: skip
        label = name if p.model == "shared" else f"{name} — {p.model}"
        dy = offsets.setdefault((round(p.x, -1), round(p.y, 1)), 0)
        offsets[(round(p.x, -1), round(p.y, 1))] += 0.035
        ax.annotate(label, (p.x, p.y), xytext=(8, -3 - 900 * dy / 30), textcoords="offset points",
                    fontsize=8, color=INK2)  # fmt: skip
    for m, mk in [("flash-lite", "o"), ("3.8-flash", "^"), ("shared (no LLM)", "s")]:
        ax.scatter([], [], marker=mk, color=INK2, s=60, label=m)
    ax.legend(frameon=False, fontsize=8, title="marker = run", title_fontsize=8, loc="lower right")
    ax.set_xlabel("CPP-day degree-hours above ceiling (°C·h, Σ 24 households)", color=INK2, fontsize=9)
    ax.set_ylabel("CPP-day post/CPP ratio (20:00–20:30 kW ÷ 17:00–20:00 mean kW)", color=INK2,
                  fontsize=9)  # fmt: skip
    ax.set_title("Rebound vs comfort cost, every agent and variant, both runs "
                 "(colour = agent, marker = run)", color=INK, fontsize=11, loc="left")  # fmt: skip
    A._style(ax, "both")
    ax.margins(x=0.25, y=0.12)
    ax.set_xlim(left=-15)  # degree-hours are >= 0
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return pts


def fig_switch(sw: pd.DataFrame, path) -> None:
    agents = ["llm", "rule_based", A.LM]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for ax, key in zip(axes, RUNS):
        for j, agent in enumerate(agents):
            r = sw[(sw["agent"] == agent) & sw["model"].str.startswith(key)].iloc[0]
            counts = [r[s] for s in SLOTS]
            x = np.arange(len(SLOTS)) + (j - 1) * 0.28
            ax.bar(x, counts, width=0.26, color=AGENT_COLOR[agent], label=AGENT_LABEL[agent])
            for xi, v in zip(x, counts):
                if v:
                    ax.text(xi, v + 0.3, str(v), ha="center", fontsize=7, color=INK2)
        ax.set_xticks(np.arange(len(SLOTS)), SLOTS)
        ax.set_title(f"{RUNS[key][1]} — CPP day", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("first AC switch-on at/after 20:00 (off → on)", color=INK2, fontsize=9)
        A._style(ax)
    axes[0].set_ylabel("households (of 24)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_elasticity(runs, path) -> None:
    groups = ["all"] + A.ORIENTATIONS
    series = [("flash-lite", "llm"), ("3.8-flash", "llm"), ("flash-lite", A.LM),
              ("3.8-flash", A.LM), ("flash-lite", "rule_based")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    for ax, comp, title in zip(axes, ["hvac", "total"], ["HVAC", "whole household"]):
        if comp == "total":
            ax.axhspan(-0.47, -0.026, color=A.GRID, alpha=0.6, lw=0)
            ax.text(3.45, -0.03, "Conway (2020) range", fontsize=8, color=INK2, ha="right", va="top")
        for j, (key, agent) in enumerate(series):
            R = next(r for r in runs if r["key"] == key)
            e = pd.concat([R["elasticity_by_orientation"], R["lm_elasticity"]], ignore_index=True)
            r = e[e["agent"] == agent].set_index("comfort_orientation").reindex(groups)
            x = np.arange(len(groups)) + (j - 2) * 0.15
            lab = (AGENT_LABEL[agent] + ("" if agent in SHARED else f" — {key}"))
            ax.errorbar(x, r[f"e_{comp}"], yerr=[r[f"e_{comp}"] - r[f"e_{comp}_lo"],
                        r[f"e_{comp}_hi"] - r[f"e_{comp}"]], fmt=MARKER[key] if agent not in SHARED else "s",
                        ms=6, color=AGENT_COLOR[agent], mec="white", mew=1.2, elinewidth=2,
                        capsize=0, label=lab)  # fmt: skip
        ax.set_xticks(range(len(groups)), ["all" if g == "all" else A.ORI_SHORT[g] for g in groups])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{title} elasticity, 17:00–20:00", color=INK, fontsize=11, loc="left")
        A._style(ax)
    axes[0].set_ylabel("pooled arc elasticity (95% household bootstrap CI)", color=INK2, fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [load_run(k) for k in RUNS]
    for R in runs:
        add_level_matched(R)
    # agents without an LLM must be identical across the two runs
    s0, s1 = (R["summary"][R["summary"]["agent"].isin(SHARED)].reset_index(drop=True) for R in runs)
    pd.testing.assert_frame_equal(s0, s1)

    T = {
        "h1": h1_side_by_side(runs), "h1_verdicts": h1_verdicts(runs), "wtp": wtp_table(runs),
        "system": system_table(runs), "elasticity": elasticity_table(runs),
        "comfort_ori": comfort_per_orientation(runs), "switch": switch_table(runs),
        "lm_fit": pd.concat([R["lm_fit"].assign(model=R["key"]) for R in runs], ignore_index=True),
    }  # fmt: skip
    T["scatter"] = fig_scatter(runs, OUT / "fig_rebound_vs_comfort.png")
    fig_switch(T["switch"], OUT / "fig_switch_on_histogram.png")
    fig_elasticity(runs, OUT / "fig_elasticity_by_orientation.png")
    for name, df in T.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    write_md(runs, T)
    print((OUT / "cross_model.md").read_text())


def write_md(runs, T) -> None:
    sy = T["system"].copy()
    sy["model"] = sy["model"].str.replace(r"^flash-lite \(shared\)", "shared", regex=True)
    sy = sy[~sy["model"].str.startswith("3.8-flash (shared)")]
    g = {(R["key"], a): R["agents"][(R["agents"]["agent"] == a) & (R["agents"]["seed"] == 42)].iloc[0]
         for R in runs for a in ["llm", "isotonic", A.LM, "rule_based", "milp"]}  # fmt: skip
    lite, flash = "flash-lite", "3.8-flash"

    def v(k, a, c, nd=3):
        return f"{g[(k, a)][c]:.{nd}f}"

    wt = T["wtp"].set_index("model")
    lmf = T["lm_fit"]
    L = [
        "# Cross-model comparison — gemini-3.5-flash-lite (final_v1) vs gemini-3.8-flash (Arm A)",
        "",
        "Generated by `experiments/analyze_cross_model.py` from the two runs' saved artifacts "
        "and their `analysis/` folders (`experiments/analyze_final_v1.py`). No live LLM calls. "
        "Arm A is the pre-registered optional arm A (docs/prereg_final_run_v1.md: model changed, "
        "new cache, everything else identical). **Each model's own numbers keep the label they "
        "have in that run's analysis (pre-registered or exploratory); the model-vs-model "
        "comparison is exploratory (post-hoc)** — the pre-registration makes no cross-model "
        "prediction. The level-matched rule-based agent is exploratory throughout.",
        "",
        "MILP, noisy-MILP and the frozen-k rule-based agent never call the LLM; their "
        "summary rows are identical in the two runs (asserted) and are shown once as *shared*. "
        "The frozen k (3.791 / 0.620 / 0.660) was fitted before the run to "
        "gemini-3.5-flash-lite's WTP in the comfort sweeps, so it is level-matched to flash-lite "
        "by construction but not to 3.8-flash. To compare each model with a rule tuned to its "
        "own level, k was refitted to **each run's own** LLM WTP with the same method "
        "(`fit_rule_based.fit_hvac`: least squares through the origin, raw WTP on max(0, "
        "indoor shown − 26 °C), one row per distinct prompt, fixed ±2.0 band, 3 dp) and that "
        "agent (`rule_based_lm`) was simulated offline on all 24 households, both days. The "
        "same code path with the frozen k reproduces the saved rule-based curves exactly.",
        "",
        "## 1. H1 — WTP by orientation at matched °C above ceiling  *(per model: pre-registered; "
        "comparison: exploratory)*",
        "",
        "Above-ceiling bins, both days pooled, decision-weighted mean raw WTP with 95% "
        "household-bootstrap CI (full tables incl. below-ceiling bins in each run's analysis):",
        "",
        md(T["h1"]),
        "",
        "Does prioritising > balanced ≥ flexible hold (bins where all three orientations are "
        "observed; robustness variants exploratory)?",
        "",
        md(T["h1_verdicts"]),
        "",
        "## 2. WTP resolution and monotonicity  *(descriptive under H4; rate definition "
        "post-hoc; comparison exploratory)*",
        "",
        md(T["wtp"], 3),
        "",
        "## 3. System metrics per agent  *(metrics pre-registered; isotonic and level-matched "
        "agents exploratory)*",
        "",
        "CPP-day post/CPP ratio = 20:00–20:30 kW ÷ 17:00–20:00 mean kW. PAR = peak-to-average "
        "ratio of the population load. Degree-hours are Σ over the 24 households; max exceedance "
        "is the worst household-step. e = pooled arc elasticity (HVAC; whole household).",
        "",
        md(sy),
        "",
        "## 4. Elasticity, pooled and per orientation  *(pooled: pre-registered; per orientation "
        "and all comparisons: exploratory)*",
        "",
        md(T["elasticity"]),
        "",
        "![](fig_elasticity_by_orientation.png)",
        "",
        "## 5. Comfort by orientation, CPP day  *(exploratory)*",
        "",
        "Mean degree-hours per household [95% CI]; worst exceedance; household-days above 2.5 °C.",
        "",
        md(T["comfort_ori"], 2),
        "",
        "## 6. Switch-on after the event, CPP day  *(exploratory)*",
        "",
        "Households by first off→on AC switch at or after 20:00.",
        "",
        md(T["switch"]),
        "",
        "![](fig_switch_on_histogram.png)",
        "",
        "## 7. Isotonic (monotone) counterfactual vs raw LLM  *(exploratory)*",
        "",
        md(pd.DataFrame([{"model": k, "post/CPP LLM": g[(k, "llm")]["cpp_post_over_cpp_ratio"],
                          "post/CPP isotonic": g[(k, "isotonic")]["cpp_post_over_cpp_ratio"],
                          "e HVAC LLM": g[(k, "llm")]["elasticity_flexible"],
                          "e HVAC isotonic": g[(k, "isotonic")]["elasticity_flexible"],
                          "degree-h CPP LLM": g[(k, "llm")]["cpp_degree_hours_above_ceiling"],
                          "degree-h CPP isotonic": g[(k, "isotonic")]["cpp_degree_hours_above_ceiling"],
                          "max exceed. CPP LLM": g[(k, "llm")]["cpp_max_exceedance_c"],
                          "max exceed. CPP isotonic": g[(k, "isotonic")]["cpp_max_exceedance_c"]}
                         for k in RUNS])),  # fmt: skip
        "",
        "## 8. Each model vs its own level-matched rule-based agent  *(exploratory fairness "
        "control)*",
        "",
        "Refitted k per orientation (frozen k shown for reference):",
        "",
        md(lmf[["model", "orientation", "k_level_matched", "k_frozen_fixed_band",
                "n_distinct_prompts", "rmse", "r2_vs_mean"]], 3),
        "",
        md(pd.DataFrame([{"model": k, "agent": a,
                          "post/CPP ratio": g[(k, a)]["cpp_post_over_cpp_ratio"],
                          "PAR base": g[(k, a)]["base_par"], "PAR CPP": g[(k, a)]["cpp_par"],
                          "degree-h CPP": g[(k, a)]["cpp_degree_hours_above_ceiling"],
                          "max exceed. CPP": g[(k, a)]["cpp_max_exceedance_c"],
                          "e HVAC": g[(k, a)]["elasticity_flexible"],
                          "e total": g[(k, a)]["elasticity_total"]}
                         for k in RUNS for a in ["llm", A.LM, "rule_based"]])),  # fmt: skip
        "",
        "(`rule_based` = frozen k, identical in both runs.)",
        "",
        "## 9. Rebound vs comfort, every agent and variant  *(exploratory)*",
        "",
        "![](fig_rebound_vs_comfort.png)",
        "",
        md(T["scatter"].rename(columns={"x": "CPP degree-hours", "y": "CPP post/CPP ratio"})),
        "",
        "## 10. Summary  *(exploratory)*",
        "",
        f"- **H1 ordering.** " + " ".join(
            f"{r['model']}: order holds in {r['order holds (point)']} bins with all three "
            f"orientations ({r['bins with all 3 orientations']}), prio−bal CI > 0 in "
            f"{r['prio−bal CI > 0']}, bal−flex CI > 0 in {r['bal−flex CI > 0']}."
            for _, r in T["h1_verdicts"].query("variant == 'pooled (pre-registered)'").iterrows())
        + " Robustness variants: see table in item 1.",
        f"- **WTP resolution.** flash-lite uses {wt.loc[lite, 'distinct WTP values']} distinct "
        f"values (top-3 share {wt.loc[lite, 'top-3 share']:.0%}), 3.8-flash "
        f"{wt.loc[flash, 'distinct WTP values']} (top-3 share {wt.loc[flash, 'top-3 share']:.0%}). "
        f"Non-monotone rate above the ceiling: flash-lite {wt.loc[lite, 'non-monotone rate (exact °C)']}, "
        f"3.8-flash {wt.loc[flash, 'non-monotone rate (exact °C)']}.",
        f"- **Rebound (post/CPP ratio, CPP day).** LLM flash-lite {v(lite, 'llm', 'cpp_post_over_cpp_ratio')}, "
        f"3.8-flash {v(flash, 'llm', 'cpp_post_over_cpp_ratio')}; MILP "
        f"{v(lite, 'milp', 'cpp_post_over_cpp_ratio')}; frozen rule-based "
        f"{v(lite, 'rule_based', 'cpp_post_over_cpp_ratio')}; own level-matched rule "
        f"{v(lite, A.LM, 'cpp_post_over_cpp_ratio')} (flash-lite) / "
        f"{v(flash, A.LM, 'cpp_post_over_cpp_ratio')} (3.8-flash); isotonic "
        f"{v(lite, 'isotonic', 'cpp_post_over_cpp_ratio')} / {v(flash, 'isotonic', 'cpp_post_over_cpp_ratio')}.",
        f"- **Comfort (CPP-day degree-hours, Σ 24 households).** LLM flash-lite "
        f"{v(lite, 'llm', 'cpp_degree_hours_above_ceiling', 1)}, 3.8-flash "
        f"{v(flash, 'llm', 'cpp_degree_hours_above_ceiling', 1)}; frozen rule-based "
        f"{v(lite, 'rule_based', 'cpp_degree_hours_above_ceiling', 1)}; own level-matched "
        f"{v(lite, A.LM, 'cpp_degree_hours_above_ceiling', 1)} / "
        f"{v(flash, A.LM, 'cpp_degree_hours_above_ceiling', 1)}. Max exceedance LLM "
        f"{v(lite, 'llm', 'cpp_max_exceedance_c', 2)} / {v(flash, 'llm', 'cpp_max_exceedance_c', 2)} °C.",
        f"- **Elasticity (pooled HVAC; whole household).** LLM flash-lite "
        f"{v(lite, 'llm', 'elasticity_flexible')}; {v(lite, 'llm', 'elasticity_total')}, "
        f"3.8-flash {v(flash, 'llm', 'elasticity_flexible')}; {v(flash, 'llm', 'elasticity_total')} "
        f"(Conway range −0.47 to −0.026).",
        "- **Rebound vs comfort.** " + " ".join(
            f"{k}: the LLM has a {'lower' if g[(k, 'llm')]['cpp_post_over_cpp_ratio'] < min(g[(k, a)]['cpp_post_over_cpp_ratio'] for a in ('isotonic', A.LM)) else 'not lower'} "
            f"post/CPP ratio and {'fewer' if g[(k, 'llm')]['cpp_degree_hours_above_ceiling'] < min(g[(k, a)]['cpp_degree_hours_above_ceiling'] for a in ('isotonic', A.LM)) else 'not fewer'} "
            f"CPP-day degree-hours than both its isotonic counterfactual and its own level-matched rule."
            for k in RUNS)
        + f" MILP variants: zero degree-hours, ratios {T['scatter'].query('agent in [\"milp\", \"noisy_milp\"]')['y'].min():.3f}"
        f"–{T['scatter'].query('agent in [\"milp\", \"noisy_milp\"]')['y'].max():.3f}.",
        "",
    ]
    (OUT / "cross_model.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
