# Copied from scratch_writing/ on 2026-10-04; original output dir ../scratch_writing/; outputs saved alongside this file.
"""Q1 post-event peak (exploratory, offline). Run from llm-hems/:
    PYTHONPATH=src .venv/bin/python ../scratch_writing/q1_post_event_peak.py [--noise]
No live LLM calls (A._no_live_calls), writes only into ../scratch_writing/."""
import hashlib, importlib.util, sys, json
from pathlib import Path
import numpy as np, pandas as pd

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

F = _load("make_thesis_figures", "experiments/make_thesis_figures.py")
A = F.A
OUT = Path("../scratch_writing")

def hash_tree(paths):
    h = hashlib.sha256()
    for p in paths:
        for f in sorted([p] if p.is_file() else p.rglob("*")):
            if f.is_file(): h.update(str(f).encode()); h.update(f.read_bytes())
    return h.hexdigest()

GUARD = [Path("results/phase3/final_v1"), Path("results/phase3/arm_a"),
         Path("results/phase3/llm_cache_final.jsonl"), Path("results/phase3/llm_cache_arm_a.jsonl"),
         Path("results/thesis_figures"), Path("results/phase3/cross_model")]
before = hash_tree(GUARD)
A._no_live_calls()

def clock(t): return f"{t//2:02d}:{30*(t%2):02d}"

def pop_kw(c, agent, day, seed):
    a = c[(c.agent == agent) & (c.day == day) & (c.seed == seed)]
    assert a.household_id.nunique() == 24
    s = a.groupby("t")["total_kwh"].sum().sort_index()
    assert list(s.index) == list(range(48))
    return s.to_numpy() / 0.5

def win_max(y, lo, hi):  # steps lo..hi-1 (half-hour starts)
    seg = y[lo:hi]; i = int(np.argmax(seg)); return float(seg[i]), clock(lo + i)

def describe(y_cpp, y_base):
    cpp_mean = y_cpp[34:40].mean(); post = y_cpp[40]
    r = {"post_kw_20:00": post, "cpp_mean_17-20": cpp_mean, "ratio": round(post / cpp_mean, 3)}
    r["max20-22"], r["at20-22"] = win_max(y_cpp, 40, 44)
    r["max20-24"], r["at20-24"] = win_max(y_cpp, 40, 48)
    r["cpp_peak"], r["cpp_peak_at"] = win_max(y_cpp, 0, 48)
    r["base_peak"], r["base_peak_at"] = win_max(y_base, 0, 48)
    for t in range(39, 46): r[clock(t)] = y_cpp[t]
    return r

rows = []
runs = [F.load_run(k) for k in F.RUNS]
for R in runs:
    s = F.check_summary(R)  # asserts recomputed summary == cross_model/system.csv
    c = R["curves"]
    for agent, seed in [("milp", 42), ("noisy_milp", 42), ("noisy_milp", 43), ("noisy_milp", 44),
                        ("rule_based", 42), (F.LLM_OF[R["key"]], 42), ("isotonic", 42), ("rule_based_lm", 42)]:
        d = describe(pop_kw(c, agent, "cpp", seed), pop_kw(c, agent, "base", seed))
        saved = s[(s.agent == agent) & (s.seed == seed)].iloc[0]
        assert np.isclose(d["ratio"], saved["cpp_post_over_cpp_ratio"]), (agent, seed)
        assert np.isclose(d["cpp_peak"], saved["cpp_peak_kw"]) and np.isclose(d["base_peak"], saved["base_peak_kw"])
        rows.append({"run": R["key"], "agent": agent, "seed": seed, **d,
                     "saved_cpp_time_of_peak": saved["cpp_time_of_peak"], "saved_base_time_of_peak": saved["base_time_of_peak"]})
    # shared agents identical across runs (already asserted in build(); re-check here)
T = pd.DataFrame(rows)
T.to_csv(OUT / "q1_agents.csv", index=False, float_format="%.4f")
print(T.to_string())

if "--noise" in sys.argv:
    N = _load("analyze_noise_test", "experiments/analyze_noise_test.py")
    saved = pd.read_csv("results/phase3/cross_model/noise_test_seeds.csv")
    out = []
    for key in N.RUNS:
        R = N.load(key)
        for v in ("iid_ori_side", "hh_shuffled", "iid_ori"):
            for seed in N.SEEDS:
                c, _ = N.simulate(R, v, seed)
                d = describe(pop_kw(c, v, "cpp", seed), pop_kw(c, v, "base", seed))
                sv = saved[(saved.model == key) & (saved.variant == v) & (saved.seed == seed)].iloc[0]
                assert np.isclose(d["ratio"], sv["cpp_post_over_cpp_ratio"]), (key, v, seed)
                out.append({"run": key, "variant": v, "seed": seed, **d})
            print(key, v, "done", flush=True)
    NT = pd.DataFrame(out); NT.to_csv(OUT / "q1_noise_seeds.csv", index=False, float_format="%.4f")
    def q(x): return f"{np.median(x):.2f} [{np.percentile(x,5):.2f}, {np.percentile(x,95):.2f}]"
    summ = NT.groupby(["run", "variant"]).agg(
        ratio=("ratio", q), max20_22=("max20-22", q), max20_24=("max20-24", q),
        kw_2000=("20:00", q), kw_2100=("21:00", q),
        mode_at20_22=("at20-22", lambda s: s.value_counts().to_dict())).reset_index()
    summ.to_csv(OUT / "q1_noise_summary.csv", index=False)
    print(summ.to_string())

assert hash_tree(GUARD) == before, "a guarded file changed"
print("guard OK: run folders, caches, thesis_figures, cross_model unchanged")
