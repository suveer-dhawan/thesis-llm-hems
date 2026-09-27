"""Thesis figures and LaTeX tables, in one consistent style.

Reads the two final runs' saved artifacts (results/phase3/final_v1 =
gemini-3.5-flash-lite, results/phase3/arm_a = gemini-3.8-flash), their
`analysis/` folders and results/phase3/cross_model/, and writes only
results/thesis_figures/. Run folders and caches are never written.

The only thing simulated is what the runs don't store as curves: the isotonic
counterfactual and the level-matched rule-based agent, both pure Python
(`analyze_final_v1.run_isotonic` / `.level_matched`, the exact code that made
the saved analysis). The LLM client is patched to refuse any live call or
cache write. Every number drawn or tabulated is asserted against the saved
CSVs it corresponds to before anything is written.

    PYTHONPATH=src .venv/bin/python experiments/make_thesis_figures.py

Deterministic: fixed bootstrap seeds, fixed jitter seed, no timestamps in the
PDF/PNG metadata; `--check` runs twice and compares every output byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from hems.phase3 import Phase3Config, Phase3Runner  # noqa: E402
from hems.phase3_report import agent_summary  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "analyze_final_v1", Path(__file__).with_name("analyze_final_v1.py")
)
A = importlib.util.module_from_spec(_spec)
sys.modules["analyze_final_v1"] = A  # dataclasses look their module up here
_spec.loader.exec_module(A)

OUT = Path("results/thesis_figures")
CROSS = Path("results/phase3/cross_model")
RUNS = {  # key -> (run dir, model id)
    "flash-lite": (Path("results/phase3/final_v1"), "gemini-3.5-flash-lite"),
    "3.8-flash": (Path("results/phase3/arm_a"), "gemini-3.8-flash"),
}
LLM_OF = {"flash-lite": "llm_fl", "3.8-flash": "llm_38"}
RUN_TITLE = {"flash-lite": "gemini-3.5-flash-lite run", "3.8-flash": "gemini-3.8-flash run"}
PRIMARY_SEED = 42
DT_H = 0.5
CPP_START, CPP_END = 17.0, 20.0

# --------------------------------------------------------------------------
# style: one fixed colour + marker + line style per agent, in every figure
# --------------------------------------------------------------------------

# Okabe-Ito (colour-blind safe). MILP is black on purpose: it is the reference.
# Green/pink sit in the CVD 6-8 band, so every agent also has its own marker.
AGENTS = ["milp", "noisy_milp", "rule_based", "rule_based_lm", "isotonic", "llm_fl", "llm_38"]
COLOR = {"milp": "#000000", "noisy_milp": "#56B4E9", "rule_based": "#E69F00",
         "rule_based_lm": "#CC79A7", "isotonic": "#009E73", "llm_fl": "#0072B2",
         "llm_38": "#D55E00"}  # fmt: skip
MARKER = {"milp": "s", "noisy_milp": "P", "rule_based": "v", "rule_based_lm": "^",
          "isotonic": "D", "llm_fl": "o", "llm_38": "p"}  # fmt: skip
LINESTYLE = {"milp": (0, (1, 1)), "noisy_milp": (0, (3, 1, 1, 1)), "rule_based": (0, (4, 1.5)),
             "rule_based_lm": (0, (2, 1)), "isotonic": (0, (6, 1.5, 1, 1.5)),
             "llm_fl": "-", "llm_38": "-"}  # fmt: skip
LABEL = {"milp": "MILP", "noisy_milp": "noisy-MILP", "rule_based": "rule-based",
         "rule_based_lm": "level-matched rule", "isotonic": "isotonic",
         "llm_fl": "LLM (3.5-flash-lite)", "llm_38": "LLM (3.8-flash)"}  # fmt: skip
ORIS = A.ORIENTATIONS
ORI_LABEL = {"comfort_prioritising": "prioritising", "balanced": "balanced",
             "comfort_flexible": "flexible"}  # fmt: skip
ORI_MARKER = {"comfort_prioritising": "o", "balanced": "s", "comfort_flexible": "^"}
ORI_LS = {"comfort_prioritising": "-", "balanced": (0, (4, 1.5)), "comfort_flexible": (0, (1, 1))}
INK, INK2, GRID, SHADE = "#1a1a1a", "#555555", "#e2e2e2", "#ececec"
CLOUD = {"iid_ori_side": ("#b8b8b8", "o", "i.i.d. residual noise (50 seeds)"),
         "hh_shuffled": ("#6e6e6e", "+", "household-shuffled residuals (50 seeds)")}

WIDTH = 6.27  # A4 text width with 1 in margins (8.27 in - 2 in)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.4, "pdf.fonttype": 42, "legend.frameon": False,
    "savefig.dpi": 300, "figure.dpi": 100, "axes.titlelocation": "left",
})  # fmt: skip


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name: str) -> list[str]:
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    return [f"{name}.pdf", f"{name}.png"]


def agent_handles(agents, line=True):
    return [Line2D([], [], color=COLOR[a], marker=MARKER[a], ms=5,
                   ls=LINESTYLE[a] if line else "none", label=LABEL[a])
            for a in agents]  # fmt: skip


def shade_cpp(ax, label=True):
    ax.axvspan(CPP_START, CPP_END, color=SHADE, lw=0, zorder=0)
    if label:
        ax.text((CPP_START + CPP_END) / 2, 0.98, "CPP", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=INK2)  # fmt: skip


# --------------------------------------------------------------------------
# data: saved curves + offline isotonic / level-matched curves, all checked
# --------------------------------------------------------------------------


def load_run(key: str) -> dict:
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
    an = run / "analysis"
    dl = pd.read_csv(an / "llm_decisions_with_state.csv")
    dl = dl[dl["appliance"] == "hvac"].reset_index(drop=True)

    fit, steps = A.fit_isotonic(dl)
    saved_steps = pd.read_csv(an / "isotonic_fit_steps.csv")
    assert np.allclose(steps["fitted_wtp"], saved_steps["fitted_wtp"]), "isotonic fit differs"
    iso = A.run_isotonic(runner, fit, ids)
    k, _, lm = A.level_matched(runner, llm_log, curves, personas, ids)  # asserts frozen-k replay
    saved_k = pd.read_csv(CROSS / "lm_fit.csv").query("model == @key").set_index("orientation")
    assert all(np.isclose(k[o], saved_k.loc[o, "k_level_matched"]) for o in ORIS)

    c = pd.concat([curves, iso, lm], ignore_index=True)
    c["agent"] = c["agent"].replace({"llm": LLM_OF[key]})
    return {"key": key, "run": run, "model": model, "resolved": resolved, "curves": c,
            "dl": dl, "k": k, "an": an}  # fmt: skip


def check_summary(R: dict) -> pd.DataFrame:
    """agent_summary of every agent in this run == cross_model/system.csv."""
    s = agent_summary(R["curves"], R["resolved"])
    sysm = pd.read_csv(CROSS / "system.csv")
    sysm = sysm[sysm["model"].str.startswith(R["key"])]
    cols = {"post/CPP ratio (CPP day)": "cpp_post_over_cpp_ratio", "PAR base": "base_par",
            "PAR CPP": "cpp_par", "degree-h base": "base_degree_hours_above_ceiling",
            "degree-h CPP": "cpp_degree_hours_above_ceiling",
            "max exceed. CPP": "cpp_max_exceedance_c", "e HVAC": "elasticity_flexible",
            "e total": "elasticity_total",
            "post 20:00-20:30 kW": "cpp_post_20:00-20:30_kw"}  # fmt: skip
    for _, d in sysm.iterrows():
        agent = LLM_OF[R["key"]] if d["agent"] == "llm" else d["agent"]
        mine = s[(s["agent"] == agent) & (s["seed"] == d["seed"])]
        assert len(mine) == 1, (agent, d["seed"])
        for theirs, ours in cols.items():
            assert np.isclose(d[theirs], mine.iloc[0][ours], atol=1e-9), (agent, ours)
    assert len(sysm) == len(s), "system.csv and recomputed summary cover different agents"
    return s


def population_kw(c: pd.DataFrame, agent: str, day: str, seed: int = PRIMARY_SEED) -> np.ndarray:
    a = c[(c["agent"] == agent) & (c["day"] == day) & (c["seed"] == seed)]
    assert a["household_id"].nunique() == 24
    return a.groupby("t")["total_kwh"].sum().sort_index().to_numpy() / DT_H


def steps_xy(y: np.ndarray):
    return np.arange(len(y) + 1) * DT_H, np.append(y, y[-1])


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def fig1_population(R) -> list[str]:
    c = R["curves"]
    agents = ["milp", "noisy_milp", "rule_based", "rule_based_lm", "isotonic", "llm_fl"]
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.6), sharex=True, sharey=True)
    for ax, a in zip(axes.flat, agents):
        shade_cpp(ax)
        for day, ls, lw, col in [("base", (0, (3, 1.5)), 1.0, INK2), ("cpp", "-", 1.5, COLOR[a])]:
            ax.plot(*steps_xy(population_kw(c, a, day)), drawstyle="steps-post", ls=ls, lw=lw,
                    color=col)  # fmt: skip
        ax.set_title(LABEL[a] + (" (seed 42)" if a == "noisy_milp" else ""), color=INK)
        style(ax)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
    for ax in axes[-1]:
        ax.set_xlabel("time of day (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("population load (kW)")
    fig.legend([Line2D([], [], color=INK2, ls=(0, (3, 1.5)), lw=1.0),
                Line2D([], [], color=INK, lw=1.5)],
               ["base day (no event)", "CPP day (agent colour)"],
               loc="lower center", ncol=2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig01_population_load_base_vs_cpp")


def fig2_zoom(runs) -> list[str]:
    fl, f38 = runs
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    shade_cpp(ax)
    series = [("milp", fl), ("rule_based", fl), ("llm_fl", fl), ("llm_38", f38)]
    for a, R in series:
        x, y = steps_xy(population_kw(R["curves"], a, "cpp"))
        x, y = x[:47], np.append(y[:46], y[45])  # up to 23:00, no riser at the edge
        ax.plot(x, y, drawstyle="steps-post", color=COLOR[a], ls=LINESTYLE[a],
                lw=1.8 if a.startswith("llm") else 1.4)  # fmt: skip
        mid = np.arange(len(y) - 1) * DT_H + DT_H / 2
        ax.plot(mid, y[:-1], ls="none", marker=MARKER[a], ms=4, color=COLOR[a], mec="white",
                mew=0.6)  # fmt: skip
    ax.set_xlim(16, 23)
    ax.set_xticks(np.arange(16, 23.5, 1), [f"{h:02d}:00" for h in range(16, 24)])
    ax.set_xlabel("time of day, CPP day")
    ax.set_ylabel("population load (kW)")
    style(ax)
    fig.legend(handles=agent_handles([a for a, _ in series]), loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save(fig, "fig02_cpp_day_zoom_16_23")


def fig3_h1(runs) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3), sharey=True)
    for ax, R in zip(axes, runs):
        h = pd.read_csv(R["an"] / "h1_wtp_by_orientation_bin.csv")
        h = h[h["side"] == "above"]
        # bin label "(+a, +b]" -> midpoint
        edges = h["C_above_ceiling"].str.findall(r"[-+]?\d+\.\d+")
        mid = edges.map(lambda v: (float(v[0]) + float(v[1])) / 2).to_numpy()
        col = COLOR[LLM_OF[R["key"]]]
        for j, o in enumerate(ORIS):
            s = ORI_LABEL[o]
            m = h[f"{s}_mean_wtp"].to_numpy(float)
            ok = ~np.isnan(m)
            x = mid[ok] + (j - 1) * 0.06
            lo, hi = h[f"{s}_lo"].to_numpy(float)[ok], h[f"{s}_hi"].to_numpy(float)[ok]
            ax.errorbar(x, m[ok], yerr=[m[ok] - lo, hi - m[ok]], color=col, ls=ORI_LS[o],
                        marker=ORI_MARKER[o], ms=4.5, mfc=col if j == 0 else "white", mec=col,
                        lw=1.1, elinewidth=0.9, capsize=0)  # fmt: skip
        ax.set_title(LABEL[LLM_OF[R["key"]]], color=INK)
        ax.set_xlim(0, 4)
        ax.set_xlabel("°C above ceiling (0.5 °C bins)")
        style(ax)
    axes[0].set_ylabel("raw WTP (\\$, mean, 95% CI)")

    fig.legend(handles=[Line2D([], [], color=INK2, ls=ORI_LS[o], marker=ORI_MARKER[o], ms=4.5,
                               mfc=INK2 if o == ORIS[0] else "white", label=ORI_LABEL[o])
                        for o in ORIS], title="comfort orientation", title_fontsize=8.5,
               loc="lower center", ncol=3)  # fmt: skip
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return save(fig, "fig03_h1_wtp_vs_temperature")


def elasticity_rows(R) -> pd.DataFrame:
    c = R["curves"][R["curves"]["seed"] == PRIMARY_SEED]
    agents = ["milp", "noisy_milp", "rule_based", "rule_based_lm", "isotonic", LLM_OF[R["key"]]]
    e = A.elasticity_by(A.per_household_records(c), "comfort_orientation", agents)
    # every agent the run's own analysis covers must agree exactly
    saved = pd.read_csv(R["an"] / "elasticity_by_orientation.csv")
    saved["agent"] = saved["agent"].replace({"llm": LLM_OF[R["key"]]})
    m = e.merge(saved, on=["agent", "comfort_orientation"], suffixes=("", "_saved"))
    assert len(m) >= 16, len(m)
    for col in ("e_hvac", "e_hvac_lo", "e_hvac_hi", "e_total", "e_total_lo", "e_total_hi"):
        assert np.allclose(m[col], m[f"{col}_saved"], equal_nan=True, atol=1e-12), col
    return e


def fig4_elasticity(runs) -> tuple[list[str], dict]:
    groups = ["all"] + ORIS
    E = {R["key"]: elasticity_rows(R) for R in runs}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.6), sharex=True, sharey="row")
    for col, R in enumerate(runs):
        e = E[R["key"]]
        agents = ["milp", "noisy_milp", "rule_based", "rule_based_lm", "isotonic", LLM_OF[R["key"]]]
        for row, comp in enumerate(["hvac", "total"]):
            ax = axes[row, col]
            if comp == "total":
                ax.axhspan(-0.47, -0.026, color=SHADE, lw=0, zorder=0)
                ax.text(3.45, -0.03, "Conway (2020)", ha="right", va="top", fontsize=7.5,
                        color=INK2)  # fmt: skip
            for j, a in enumerate(agents):
                r = e[e["agent"] == a].set_index("comfort_orientation").reindex(groups)
                x = np.arange(len(groups)) + (j - 2.5) * 0.12
                v = r[f"e_{comp}"].to_numpy(float)
                ax.errorbar(x, v, yerr=[v - r[f"e_{comp}_lo"], r[f"e_{comp}_hi"] - v],
                            fmt=MARKER[a], ms=4, color=COLOR[a], mec="white", mew=0.5,
                            elinewidth=1.0, capsize=0)  # fmt: skip
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_xticks(range(len(groups)), ["all"] + [ORI_LABEL[o] for o in ORIS])
            style(ax)
            if row == 0:
                ax.set_title(RUN_TITLE[R["key"]], color=INK)
        axes[0, 0].set_ylabel("HVAC elasticity")
        axes[1, 0].set_ylabel("whole-household elasticity")
    fig.legend(handles=agent_handles(AGENTS, line=False), loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save(fig, "fig04_elasticity_by_orientation"), E


SLOTS = ["20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "none"]


def fig5_switch(runs) -> list[str]:
    sw_saved = pd.read_csv(CROSS / "switch.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
    for ax, R in zip(axes, runs):
        agents = [LLM_OF[R["key"]], "isotonic", "rule_based"]
        c = R["curves"][(R["curves"]["seed"] == PRIMARY_SEED) & (R["curves"]["day"] == "cpp")]
        s = A.switch_on(c, agents)
        for j, a in enumerate(agents):
            cnt = s.loc[s["agent"] == a, "first_switch_on_after_20"].value_counts()
            counts = [int(cnt.get(k, 0)) for k in SLOTS]
            ref = sw_saved[(sw_saved["agent"] == ("llm" if a.startswith("llm") else a))
                           & sw_saved["model"].str.startswith(R["key"])].iloc[0]  # fmt: skip
            assert counts == [int(ref[k]) for k in SLOTS], (R["key"], a)
            x = np.arange(len(SLOTS)) + (j - 1) * 0.27
            ax.bar(x, counts, width=0.25, color=COLOR[a], label=LABEL[a], lw=0)
        ax.set_xticks(range(len(SLOTS)), SLOTS, rotation=45,
                      ha="right", rotation_mode="anchor")  # fmt: skip
        ax.set_title(RUN_TITLE[R["key"]], color=INK)
        ax.set_xlabel("first AC switch-on at/after 20:00")
        style(ax)
    axes[0].set_ylabel("households (of 24)")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR[a], label=LABEL[a])
                        for a in ["llm_fl", "llm_38", "isotonic", "rule_based"]],
               loc="lower center", ncol=4)  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save(fig, "fig05_switch_on_after_event")


def fig6_scatter(runs, summ) -> list[str]:
    seeds = pd.read_csv(CROSS / "noise_test_seeds.csv")
    pts = pd.read_csv(CROSS / "scatter.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), sharey=True)
    for ax, R in zip(axes, runs):
        for v, (col, mk, lab) in CLOUD.items():
            a = seeds[(seeds["model"] == R["key"]) & (seeds["variant"] == v)]
            assert len(a) == 50
            ax.scatter(a["cpp_degree_hours_above_ceiling"], a["cpp_post_over_cpp_ratio"], s=10,
                       marker=mk, color=col, lw=0.8 if mk == "+" else 0, zorder=2)  # fmt: skip
        s = summ[R["key"]]
        agents = ["milp", "noisy_milp", "rule_based", "rule_based_lm", "isotonic", LLM_OF[R["key"]]]
        for a in agents:
            for r in s[s["agent"] == a].itertuples(index=False):
                x, y = r.cpp_degree_hours_above_ceiling, r.cpp_post_over_cpp_ratio
                key = "shared" if a in ("milp", "noisy_milp", "rule_based") else R["key"]
                p = pts[(pts["model"] == key) & (pts["agent"] == a.replace(LLM_OF[R["key"]], "llm"))
                        & (pts["seed"] == r.seed)]  # fmt: skip
                assert len(p) == 1
                assert np.isclose(p["x"].iloc[0], x) and np.isclose(p["y"].iloc[0], y)
                ax.scatter(x, y, s=46 if a.startswith("llm") else 34, marker=MARKER[a],
                           color=COLOR[a], edgecolor="white", lw=0.8, zorder=4)  # fmt: skip
        ax.set_title(RUN_TITLE[R["key"]], color=INK)
        ax.set_xlabel("CPP-day degree-hours above ceiling (°C h)")
        ax.set_xlim(-12, 320)
        style(ax, "both")
    axes[0].set_ylabel("post/CPP load ratio, CPP day")
    h = agent_handles(AGENTS, line=False) + [
        Line2D([], [], color=col, marker=mk, ls="none", ms=4, label=lab)
        for col, mk, lab in CLOUD.values()]  # fmt: skip
    fig.legend(handles=h, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    return save(fig, "fig06_rebound_vs_comfort_noise")


def fig7_comfort(runs) -> list[str]:
    fl, f38 = runs
    series = [("llm_fl", fl), ("llm_38", f38), ("rule_based", fl)]
    ch = pd.concat([A.comfort_per_household(R["curves"][(R["curves"]["agent"] == a)
                                                       & (R["curves"]["seed"] == PRIMARY_SEED)])
                    for a, R in series], ignore_index=True)  # fmt: skip
    # agrees with each run's own per-household comfort table
    for a, R in series:
        saved = pd.read_csv(R["an"] / "comfort_per_household.csv")
        name = "llm" if a.startswith("llm") else a
        saved = saved[(saved["agent"] == name) & (saved["seed"] == PRIMARY_SEED)]
        mine = ch[ch["agent"] == a].merge(saved, on=["household_id", "day"], suffixes=("", "_s"))
        assert len(mine) == 48 and np.allclose(mine["degree_hours"], mine["degree_hours_s"])
    rng = np.random.default_rng(7)  # jitter only
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3), sharey=True)
    for ax, day in zip(axes, ["base", "cpp"]):
        for i, o in enumerate(ORIS):
            for j, (a, _) in enumerate(series):
                v = ch[(ch["agent"] == a) & (ch["day"] == day)
                       & (ch["comfort_orientation"] == o)]["degree_hours"].to_numpy()  # fmt: skip
                x0 = i + (j - 1) * 0.26
                ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(v)), v, s=12, marker=MARKER[a],
                           color=COLOR[a], edgecolor="white", lw=0.4, zorder=3)  # fmt: skip
                ax.hlines(np.median(v), x0 - 0.1, x0 + 0.1, color=INK, lw=1.4, zorder=4)
        ax.set_xticks(range(3), [ORI_LABEL[o] for o in ORIS])
        ax.set_title("base day" if day == "base" else "CPP day", color=INK)
        style(ax)
    axes[0].set_ylabel("degree-hours above ceiling\nper household (°C h)")
    fig.legend(handles=agent_handles([a for a, _ in series], line=False)
               + [Line2D([], [], color=INK, lw=1.4, label="median")],
               loc="lower center", ncol=4)  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save(fig, "fig07_comfort_by_orientation")


def clock_residuals(R) -> pd.DataFrame:
    """Residual of the noise test's base model (intercept + slope on shown °C over ceiling,
    per orientation; above-ceiling decisions, both days), by hour of day."""
    d = R["dl"][R["dl"]["over_c"] > 0].reset_index(drop=True)
    cols = [pd.get_dummies(d["orientation"]).astype(float),
            pd.get_dummies(d["orientation"]).astype(float).mul(d["shown_over_c"], axis=0)]
    X = pd.concat(cols, axis=1).to_numpy(float)
    y = d["wtp"].to_numpy(float)
    d["resid"] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    d["hour"] = d["t"] // 2
    # must reproduce the time-block table in noise_test.md
    blocks = pd.cut(d["hour"], [-1, 11, 16, 19, 20, 23],
                    labels=["00–12", "12–17", "17–20", "20–21", "21–24"])
    blk = d["resid"].groupby(blocks, observed=False).agg(["mean", "count"])
    line = next(ln for ln in (CROSS / "noise_test.md").read_text().splitlines()
                if re.match(rf"\| {re.escape(R['key'])} \| [+-]\d\.\d{{3}} \(", ln))
    want = [(float(m), int(n)) for m, n in re.findall(r"([+-]\d\.\d{3}) \((\d+)\)", line)]
    got = [(round(float(m), 3), int(n)) for m, n in blk.itertuples(index=False)]
    assert got == want, (got, want)
    rows = []
    for hr, g in d.groupby("hour"):
        per = g.groupby("household_id")["resid"]
        m, lo, hi = A.boot_ratio(per.sum(), per.count())
        rows.append({"model": R["key"], "hour": int(hr), "n_decisions": len(g),
                     "n_households": per.ngroups, "mean_resid": m, "lo": lo, "hi": hi})  # fmt: skip
    return pd.DataFrame(rows)


def fig8_clock(runs) -> tuple[list[str], pd.DataFrame]:
    res = pd.concat([clock_residuals(R) for R in runs], ignore_index=True)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
    shade_cpp(ax, label=False)
    ax.text(18.5, 0.98, "CPP window\n(event on CPP day only)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color=INK2)  # fmt: skip
    ax.axhline(0, color=INK2, lw=0.6)
    for j, R in enumerate(runs):
        a = LLM_OF[R["key"]]
        r = res[(res["model"] == R["key"]) & (res["n_decisions"] >= 5)]
        x = r["hour"] + 0.5 + (j - 0.5) * 0.18  # hour-block centre
        ax.errorbar(x, r["mean_resid"], yerr=[r["mean_resid"] - r["lo"], r["hi"] - r["mean_resid"]],
                    color=COLOR[a], marker=MARKER[a], ms=4.5, mec="white", mew=0.5, ls=LINESTYLE[a],
                    lw=1.1, elinewidth=0.9, capsize=0, label=LABEL[a])  # fmt: skip
    ax.set_xlim(10, 24)
    ax.set_xticks(range(10, 25, 2))
    ax.set_ylim(-0.32, 0.4)
    ax.set_xlabel("time of day (h)")
    ax.set_ylabel("WTP residual vs temperature fit (\\$)")
    style(ax)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, "fig08_wtp_residual_by_clock"), res


# --------------------------------------------------------------------------
# LaTeX tables (booktabs)
# --------------------------------------------------------------------------


def tex_escape(s: str) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("$", r"\$"), ("°C", r"\textdegree C"), ("−", "--"),
                 ("–", "--"), ("×", r"$\times$"), ("≥", r"$\geq$")]:  # fmt: skip
        s = s.replace(a, b)
    return s


def num(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def ci_tex(s: str) -> str:
    """'-0.282 [-0.407, -0.166]' -> math minus signs."""
    if s in ("–", "-", "nan") or pd.isna(s):
        return "--"
    return re.sub(r"(?<![\d.])-(?=\d)", "$-$", str(s))


def write_table(name: str, header: list[str], rows: list, align: str, caption: str,
                label: str, groups_after: tuple = (), notes: str = "") -> str:  # fmt: skip
    """booktabs table. A row is a list of cells, or a str: a group sub-heading spanning
    every column (preceded by a midrule unless it is the first row)."""
    ncol = len(header)
    L = ["% generated by experiments/make_thesis_figures.py -- do not edit by hand",
         r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         rf"\begin{{tabular}}{{{align}}}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]  # fmt: skip
    for i, r in enumerate(rows):
        if isinstance(r, str):
            if i:
                L.append(r"\midrule")
            L.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{r}}}}} \\")
            continue
        assert len(r) == ncol, (name, r)
        L.append(" & ".join(r) + r" \\")
        if i in groups_after:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        L += [r"\par\smallskip", rf"\parbox{{\linewidth}}{{\footnotesize {notes}}}"]
    L += [r"\end{table}", ""]
    (OUT / f"{name}.tex").write_text("\n".join(L))
    return f"{name}.tex"


MODEL_HEAD = {"flash-lite": "gemini-3.5-flash-lite (final run v1)",
              "3.8-flash": "gemini-3.8-flash (Arm A)"}  # fmt: skip
SHORT_AGENT = {"milp": "MILP", "noisy_milp": "noisy-MILP", "rule_based": "rule-based",
               "rule_based_lm": "level-matched rule", "isotonic": "isotonic",
               "llm_fl": "LLM", "llm_38": "LLM"}  # fmt: skip


def tab_summary(summ) -> str:
    rows = []
    shared = "Shared agents (no LLM; identical in both runs)"
    order = [(shared, ["milp", "noisy_milp", "rule_based"], "flash-lite"),
             (MODEL_HEAD["flash-lite"], ["llm_fl", "isotonic", "rule_based_lm"], "flash-lite"),
             (MODEL_HEAD["3.8-flash"], ["llm_38", "isotonic", "rule_based_lm"],
              "3.8-flash")]  # fmt: skip
    for head, agents, key in order:
        rows.append(head)
        s = summ[key]
        for a in agents:
            for r in s[s["agent"] == a].sort_values("seed").itertuples(index=False):
                rows.append([tex_escape(SHORT_AGENT[a]), str(r.seed),
                             num(r.cpp_post_over_cpp_ratio), num(r.base_par), num(r.cpp_par),
                             num(r.base_degree_hours_above_ceiling, 1),
                             num(r.cpp_degree_hours_above_ceiling, 1),
                             num(r.cpp_max_exceedance_c, 2),
                             num(r.elasticity_flexible), num(r.elasticity_total)])  # fmt: skip
    return write_table(
        "tab_summary_by_agent",
        ["Agent", "Seed", "Post/CPP", r"PAR\textsubscript{base}", r"PAR\textsubscript{CPP}",
         r"DH\textsubscript{base}", r"DH\textsubscript{CPP}", r"Max exc.", r"$e_\text{HVAC}$",
         r"$e_\text{total}$"],
        rows, "lrrrrrrrrr",
        "System metrics by agent, both runs (24 households, fixed $\\pm$2.0\\,\\textdegree C "
        "band).",
        "tab:summary-by-agent", (),
        "Post/CPP = CPP-day 20:00--20:30 population kW $\\div$ 17:00--20:00 mean kW. PAR = "
        "peak-to-average "
        "ratio. DH = degree-hours above the comfort ceiling, $\\Sigma$ over households "
        "(\\textdegree C\\,h). "
        "Max exc.\\ = worst household-step exceedance on the CPP day (\\textdegree C). $e$ = "
        "pooled arc "
        "elasticity, 17:00--20:00 window. Isotonic and level-matched rows are exploratory "
        "counterfactuals fitted to each run's own LLM.")  # fmt: skip


def tab_h1() -> str:
    h = pd.read_csv(CROSS / "h1.csv")
    rows, prev = [], None
    for r in h.itertuples(index=False):
        v = list(r._asdict().values())
        if v[0] != prev:
            rows.append(MODEL_HEAD[v[0]])
            prev = v[0]
        rows.append([tex_escape(v[1]).replace("+", ""), v[2]] + [ci_tex(x) for x in v[3:8]])
    return write_table(
        "tab_h1_matched_bins",
        [r"\textdegree C over", r"$n_\text{hh}$ p/b/f", "Prioritising", "Balanced",
         "Flexible", "Prio $-$ bal", "Bal $-$ flex"],
        rows, "lcccccc",
        "H1: raw LLM WTP (\\$) by comfort orientation at matched temperature above the ceiling.",
        "tab:h1-matched-bins", (),
        "Decision-weighted mean with 95\\% household-cluster bootstrap CI (10{,}000 resamples); "
        "both days pooled; -- = orientation not observed in that bin. p/b/f = households "
        "prioritising / balanced / flexible. Per-model results are pre-registered; the cross-model "
        "comparison is exploratory.")  # fmt: skip


NOISE_SHORT = {"i.i.d. noise": "i.i.d.",
               "i.i.d. noise (orientation only)": "i.i.d., orientation only",
               "household-shuffled noise": "household-shuffled"}  # fmt: skip


def tab_noise() -> str:
    s = pd.read_csv(CROSS / "noise_test_summary.csv")
    rows, prev = [], None
    for r in s.itertuples(index=False):
        v = list(r._asdict().values())
        if v[0] != prev:
            rows.append(MODEL_HEAD[v[0]])
            prev = v[0]
        rows.append([tex_escape(NOISE_SHORT[v[1]]), ci_tex(v[3]), num(v[4]), f"{v[5]:.0f}",
                     ci_tex(v[7]), num(v[8], 1), f"{v[9]:.0f}", tex_escape(v[15])])  # fmt: skip
    return write_table(
        "tab_noise_test",
        ["Noise variant", "Post/CPP cloud", "LLM", "Pct", "DH\\textsubscript{CPP} cloud",
         "LLM", "Pct", "Dom."],
        rows, "lcrrcrrc",
        "Noise test: level-matched rule plus the LLM's own residuals (exploratory, post-hoc).",
        "tab:noise-test", (),
        "Cloud = median [5th--95th percentile] over 50 seeds. Pct = percentile of the LLM value "
        "within the cloud (0 = better than every seed; lower is better on both metrics). Dom.\\ = "
        "seeds at least as good as the LLM on both post/CPP ratio and CPP-day "
        "degree-hours.")  # fmt: skip


def tab_headline(summ, E) -> str:
    wtp = pd.read_csv(CROSS / "wtp.csv").set_index("model")
    ver = pd.read_csv(CROSS / "h1_verdicts.csv")
    noise = pd.read_csv(CROSS / "noise_test_summary.csv")
    keys = list(RUNS)

    def s(key, agent, col):
        x = summ[key]
        return x[(x["agent"] == agent) & (x["seed"] == PRIMARY_SEED)].iloc[0][col]

    def e_ci(key, comp):
        e = E[key]
        r = e[(e["agent"] == LLM_OF[key]) & (e["comfort_orientation"] == "all")].iloc[0]
        return ci_tex(r[f"e_{comp}_ci"])

    def v(key, variant, col):
        return ver[(ver["model"] == key) & (ver["variant"] == variant)].iloc[0][col]

    def nz(key, variant, col):
        return noise[(noise["model"] == key) & (noise["variant"] == variant)].iloc[0][col]

    spec = [
        ("H1 order holds (bins with all 3 orientations)",
         lambda k: v(k, "pooled (pre-registered)", "order holds (point)")),
        ("H1 prio $-$ bal CI $>0$", lambda k: v(k, "pooled (pre-registered)", "prio−bal CI > 0")),
        ("H1 bal $-$ flex CI $>0$", lambda k: v(k, "pooled (pre-registered)", "bal−flex CI > 0")),
        ("Distinct WTP values", lambda k: str(wtp.loc[k, "distinct WTP values"])),
        ("Top-3 WTP share", lambda k: num(wtp.loc[k, "top-3 share"], 2)),
        ("Non-monotone rate above ceiling",
         lambda k: ci_tex(wtp.loc[k, "non-monotone rate (exact °C)"])),
        (r"$e_\text{HVAC}$ [95\% CI]", lambda k: e_ci(k, "hvac")),
        (r"$e_\text{total}$ [95\% CI]", lambda k: e_ci(k, "total")),
        ("Post/CPP ratio, LLM", lambda k: num(s(k, LLM_OF[k], "cpp_post_over_cpp_ratio"))),
        ("Post/CPP ratio, level-matched rule",
         lambda k: num(s(k, "rule_based_lm", "cpp_post_over_cpp_ratio"))),
        ("Post/CPP ratio, isotonic", lambda k: num(s(k, "isotonic", "cpp_post_over_cpp_ratio"))),
        (r"PAR\textsubscript{CPP}, LLM", lambda k: num(s(k, LLM_OF[k], "cpp_par"))),
        (r"DH\textsubscript{CPP}, LLM (\textdegree C\,h)",
         lambda k: num(s(k, LLM_OF[k], "cpp_degree_hours_above_ceiling"), 1)),
        (r"DH\textsubscript{CPP}, level-matched rule",
         lambda k: num(s(k, "rule_based_lm", "cpp_degree_hours_above_ceiling"), 1)),
        ("Max exceedance, CPP day (\\textdegree C)",
         lambda k: num(s(k, LLM_OF[k], "cpp_max_exceedance_c"), 2)),
        ("LLM pct in i.i.d. noise cloud (ratio / DH)",
         lambda k: f"{nz(k, 'i.i.d. noise', 'post/CPP ratio LLM pct'):.0f} / "
                   f"{nz(k, 'i.i.d. noise', 'CPP degree-hours LLM pct'):.0f}"),
        ("LLM pct in household-shuffled cloud (ratio / DH)",
         lambda k: f"{nz(k, 'household-shuffled noise', 'post/CPP ratio LLM pct'):.0f} / "
                   f"{nz(k, 'household-shuffled noise', 'CPP degree-hours LLM pct'):.0f}"),
    ]  # fmt: skip
    rows = [[lab] + [f(k) for k in keys] for lab, f in spec]
    return write_table(
        "tab_cross_model_headline", ["", "3.5-flash-lite", "3.8-flash"], rows, "lcc",
        "Cross-model headline: gemini-3.5-flash-lite (final run v1) vs gemini-3.8-flash (Arm A).",
        "tab:cross-model-headline", (2, 7, 14),
        "Shared references: MILP post/CPP 1.555, frozen-$k$ rule-based 1.372 (164.4 "
        "\\textdegree C\\,h). "
        "Conway (2020) whole-household range: $-0.47$ to $-0.026$. H1 rows are pre-registered per "
        "model; everything else in the comparison is exploratory.")  # fmt: skip


# --------------------------------------------------------------------------
# captions
# --------------------------------------------------------------------------

CAPTIONS = [
    ("fig01_population_load_base_vs_cpp", "Figure 1",
     "Population load (24 households) under each agent in the gemini-3.5-flash-lite run, base day "
     "(dashed grey) vs CPP day (solid, agent colour). Shaded: CPP window 17:00–20:00. Noisy-MILP "
     "shown for seed 42; the isotonic and level-matched agents are exploratory counterfactuals.",
     "results/phase3/final_v1/curves.parquet (MILP, noisy-MILP, rule-based, LLM); isotonic and "
     "level-matched curves re-simulated offline (analyze_final_v1.run_isotonic / .level_matched)"),
    ("fig02_cpp_day_zoom_16_23", "Figure 2",
     "CPP-day population load, 16:00–23:00, for MILP, the rule-based satisficer and the LLM agent "
     "with gemini-3.5-flash-lite and gemini-3.8-flash. Shaded: CPP window. Markers sit at "
     "half-hour step centres.",
     "results/phase3/final_v1/curves.parquet, results/phase3/arm_a/curves.parquet"),
    ("fig03_h1_wtp_vs_temperature", "Figure 3",
     "H1: raw LLM willingness-to-pay vs indoor temperature above the comfort ceiling, by comfort "
     "orientation, for each model. Points are decision-weighted bin means (0.5 °C bins, both days "
     "pooled) with 95% household-bootstrap CIs; only bins in which an orientation occurs are "
     "shown.",
     "results/phase3/{final_v1,arm_a}/analysis/h1_wtp_by_orientation_bin.csv"),
    ("fig04_elasticity_by_orientation", "Figure 4",
     "Pooled arc elasticity of 17:00–20:00 load (HVAC, top; whole household, bottom) by comfort "
     "orientation for every agent, with 95% household-bootstrap CIs. Left: gemini-3.5-flash-lite "
     "run; right: gemini-3.8-flash run (MILP, noisy-MILP seed 42 and frozen rule-based "
     "identical in "
     "both). Shaded: Conway (2020) empirical range for whole-household elasticity.",
     "recomputed from curves (analyze_final_v1.elasticity_by), asserted equal to "
     "results/phase3/{final_v1,arm_a}/analysis/elasticity_by_orientation.csv"),
    ("fig05_switch_on_after_event", "Figure 5",
     "Timing of the first AC switch-on (off → on) at or after 20:00 on the CPP day, households of "
     "24, for the LLM, its isotonic counterfactual and the rule-based satisficer, per model.",
     "recomputed from curves (analyze_final_v1.switch_on), asserted equal to "
     "results/phase3/cross_model/switch.csv"),
    ("fig06_rebound_vs_comfort_noise", "Figure 6",
     "Rebound (CPP-day post/CPP load ratio) vs comfort cost (CPP-day degree-hours above "
     "ceiling) for "
     "every agent, with the noise-test clouds: the level-matched rule plus the LLM's own "
     "residuals, "
     "resampled i.i.d. (grey dots) or as whole household-shuffled sequences (dark crosses), 50 "
     "seeds "
     "each. Lower-left is better on both axes. Exploratory.",
     "results/phase3/cross_model/noise_test_seeds.csv; agent points recomputed and asserted equal "
     "to results/phase3/cross_model/scatter.csv"),
    ("fig07_comfort_by_orientation", "Figure 7",
     "Per-household degree-hours above the comfort ceiling by comfort orientation, base day and "
     "CPP "
     "day, for the LLM agent (both models) and the rule-based satisficer. Each point is a "
     "household; "
     "bars are medians (n = 5 / 8 / 11 households per orientation).",
     "recomputed from curves (analyze_final_v1.comfort_per_household), asserted equal to "
     "results/phase3/{final_v1,arm_a}/analysis/comfort_per_household.csv"),
    ("fig08_wtp_residual_by_clock", "Figure 8",
     "Time-of-day effect: mean residual of LLM WTP relative to a per-orientation linear fit on the "
     "°C above ceiling shown in the prompt (above-ceiling decisions, both days), by hour of day, "
     "with 95% household-bootstrap CIs; hours with fewer than 5 decisions omitted. The clock is "
     "the "
     "only non-temperature field that varies in the prompt. Exploratory.",
     "results/phase3/{final_v1,arm_a}/analysis/llm_decisions_with_state.csv; block means asserted "
     "equal to results/phase3/cross_model/noise_test.md §1; plotted values in "
     "fig08_wtp_residual_by_clock.csv"),
    ("tab_summary_by_agent", "Table 1",
     "System metrics by agent, both runs (post/CPP ratio, PAR, degree-hours, max exceedance, "
     "HVAC and whole-household elasticity).",
     "recomputed with phase3_report.agent_summary, asserted equal to "
     "results/phase3/cross_model/system.csv"),
    ("tab_h1_matched_bins", "Table 2",
     "H1: raw LLM WTP by comfort orientation at matched °C above ceiling, both models, with "
     "orientation differences and 95% household-bootstrap CIs.",
     "results/phase3/cross_model/h1.csv"),
    ("tab_noise_test", "Table 3",
     "Noise test: where the LLM falls within clouds of the level-matched rule plus its own "
     "residuals (i.i.d., orientation-only i.i.d., household-shuffled), 50 seeds each.",
     "results/phase3/cross_model/noise_test_summary.csv"),
    ("tab_cross_model_headline", "Table 4",
     "Cross-model headline: H1 verdict, WTP resolution, elasticity, rebound, comfort and "
     "noise-test "
     "position for gemini-3.5-flash-lite vs gemini-3.8-flash.",
     "results/phase3/cross_model/{wtp,h1_verdicts,noise_test_summary}.csv + recomputed summaries "
     "(asserted equal to system.csv)"),
]


def write_captions() -> str:
    L = ["# Thesis figures and tables — draft captions", "",
         "Generated by `experiments/make_thesis_figures.py` (offline; no LLM calls). Each entry: "
         "draft caption, then the data it is drawn from. Figures are PDF (vector) + PNG (300 dpi), "
         "6.3 in wide; tables are booktabs `.tex` (`\\usepackage{booktabs, amsmath}`).",
         "",
         "Agent colours (Okabe-Ito, fixed across every figure): "
         + ", ".join(f"{LABEL[a]} `{COLOR[a]}` ({MARKER[a]})" for a in AGENTS)
         + ".", ""]  # fmt: skip
    for name, tag, cap, src in CAPTIONS:
        ext = ".tex" if name.startswith("tab_") else ".pdf / .png"
        L += [f"## {tag} — `{name}{ext}`", "", f"**Caption.** {cap}", "",
              f"**Source.** {src}", ""]
    (OUT / "captions.md").write_text("\n".join(L))
    return "captions.md"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build(out: Path) -> list[str]:
    global OUT
    OUT = out
    OUT.mkdir(parents=True, exist_ok=True)
    A._no_live_calls()
    runs = [load_run(k) for k in RUNS]
    # agents without an LLM are identical in both runs
    for a in ("milp", "noisy_milp", "rule_based"):
        key = ["seed", "household_id", "day", "t"]
        c0, c1 = (A._sort(R["curves"][R["curves"]["agent"] == a], key) for R in runs)
        assert np.array_equal(c0["total_kwh"].to_numpy(), c1["total_kwh"].to_numpy()), a
    summ = {R["key"]: check_summary(R) for R in runs}

    files = []
    files += fig1_population(runs[0])
    files += fig2_zoom(runs)
    files += fig3_h1(runs)
    f, E = fig4_elasticity(runs)
    files += f
    files += fig5_switch(runs)
    files += fig6_scatter(runs, summ)
    files += fig7_comfort(runs)
    f, res = fig8_clock(runs)
    files += f
    res.to_csv(OUT / "fig08_wtp_residual_by_clock.csv", index=False, float_format="%.6f")
    files.append("fig08_wtp_residual_by_clock.csv")
    files += [tab_summary(summ), tab_h1(), tab_noise(), tab_headline(summ, E)]
    files.append(write_captions())
    return files


def digest(folder: Path, files) -> dict:
    return {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in files}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/thesis_figures")
    ap.add_argument("--check", action="store_true",
                    help="build a second time into a temp dir and require identical bytes")
    args = ap.parse_args(argv)
    out = Path(args.out)
    files = build(out)
    first = digest(out, files)
    print(f"wrote {len(files)} files to {out}/")
    if args.check:
        tmp = Path(tempfile.mkdtemp(prefix="thesis_figs_"))
        try:
            files2 = build(tmp)
            assert files2 == files
            second = digest(tmp, files2)
            diff = [f for f in files if first[f] != second[f]]
            assert not diff, f"non-deterministic outputs: {diff}"
            print(f"determinism check: {len(files)} files byte-identical across two builds")
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
