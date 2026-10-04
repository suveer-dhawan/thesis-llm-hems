# Copied from scratch_writing/ on 2026-10-04; original output dir ../scratch_writing/; outputs saved alongside this file.
"""Q1 extras: ties in window maxima, baseline (non-HVAC) vs HVAC kW decomposition. Offline."""
import importlib.util, sys
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("make_thesis_figures", "experiments/make_thesis_figures.py")
F = importlib.util.module_from_spec(spec); sys.modules["make_thesis_figures"] = F; spec.loader.exec_module(F)
F.A._no_live_calls()
def clock(t): return f"{t//2:02d}:{30*(t%2):02d}"
rows = []
for k in F.RUNS:
    R = F.load_run(k); c = R["curves"]
    for agent, seed in [("milp",42),("noisy_milp",42),("noisy_milp",43),("noisy_milp",44),("rule_based",42),(F.LLM_OF[k],42),("isotonic",42),("rule_based_lm",42)]:
        for day in ("cpp","base"):
            a = c[(c.agent==agent)&(c.day==day)&(c.seed==seed)]
            g = a.groupby("t")[["base_kwh","hvac_kwh","total_kwh"]].sum()/0.5
            y = g.total_kwh.to_numpy()
            def ties(lo,hi):
                m = y[lo:hi].max(); return ",".join(clock(lo+i) for i in np.flatnonzero(np.isclose(y[lo:hi], m)))
            rows.append({"run":k,"agent":agent,"seed":seed,"day":day,"ties20_22":ties(40,44),"ties20_24":ties(40,48),"ties_day":ties(0,48),
                         **{f"base_{clock(t)}":g.base_kwh[t] for t in range(39,46)},
                         **{f"nAC_{clock(t)}":int(g.hvac_kwh[t]*0.5) for t in range(39,46)}})
T = pd.DataFrame(rows); pd.set_option("display.width",400); print(T.to_string())
T.to_csv("../scratch_writing/q1_extra.csv", index=False, float_format="%.3f")
