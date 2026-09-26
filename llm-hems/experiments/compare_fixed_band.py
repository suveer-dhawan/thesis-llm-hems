"""Fixed-band (+/-2.0) vs persona-linked band at global_scale 1.0: elasticity + matched-temp WTP."""

import pandas as pd

R = "results/phase3/comfort_sweep"
ORI = ["comfort_prioritising", "balanced", "comfort_flexible"]
K = -0.4230769230769231 / -0.5  # base-method arc factor: e = K * dQ/Q (checked against pooled rows)

pl = pd.read_csv(f"{R}/live/per_household.csv")
pl = pl[(pl.condition == "graded") & (pl.global_scale == 1.0)].assign(src="persona-linked (on file)")
fb = pd.read_csv(f"{R}/fixed_band_2/live/per_household.csv").assign(src="fixed +/-2.0 (new)")
bal = pl[pl.comfort_orientation == "balanced"].assign(src="reused: persona-linked (band already +/-2.0)")
fb = pd.concat([fb, bal])

print("### per household (scale 1.0, graded)\n")
cols = ["comfort_orientation", "household_id", "archetype", "hvac_kwh_base", "hvac_kwh_cpp",
        "e_flexible", "e_total", "src"]
print(pd.concat([pl.assign(band="persona"), fb.assign(band="fixed")])
      .sort_values(["comfort_orientation", "household_id", "band"])[cols + ["band"]]
      .to_string(index=False, float_format="{:.3f}".format))

rows = []
for o in ORI:
    r = {"comfort_orientation": o}
    for name, df in (("persona", pl), ("fixed", fb)):
        g = df[df.comfort_orientation == o]
        qb, qc = g.hvac_kwh_base.sum(), g.hvac_kwh_cpp.sum()
        r[f"{name} HVAC kWh"] = f"{qb:.0f}->{qc:.0f}"
        r[f"{name} e_flex (pooled)"] = K * (qc - qb) / qb if qb else float("nan")
        r[f"{name} e_total (hh mean)"] = g.e_total.mean()
    r["fixed source"] = fb[fb.comfort_orientation == o].src.iloc[0]
    rows.append(r)
print("\n### comfort-axis (HVAC) elasticity per orientation, scale 1.0\n")
print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))


def prompts(path, **filt):
    d = pd.read_parquet(path)
    d = d[~d.forced]
    for k, v in filt.items():
        d = d[d[k] == v]
    d = d.assign(indoor=d.prompt.str.extract(r"indoor temperature \(C\)\s+: ([\d.]+)")[0].astype(float))
    return d.drop_duplicates("prompt")


pl_log = prompts(f"{R}/live/sweep_llm_log.parquet", condition="graded", global_scale=1.0)
pl_all = prompts(f"{R}/live/sweep_llm_log.parquet")
fb_log = pd.concat([prompts(f"{R}/fixed_band_2/live/sweep_llm_log.parquet"),
                    pl_log[pl_log.comfort_orientation == "balanced"]])

bins = [24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5]


def table(log, title):
    log = log.assign(indoor_bin=pd.cut(log.indoor, bins, right=False))
    t = log.pivot_table(index="indoor_bin", columns="comfort_orientation", values="wtp",
                        aggfunc=["mean", "count"], observed=False)
    out = pd.DataFrame(index=t.index.astype(str))
    for o in ORI:
        m, n = t[("mean", o)].values, t[("count", o)].fillna(0).astype(int).values
        out[o] = [f"{a:.2f} (n={c})" if c else "-" for a, c in zip(m, n)]
    print(f"\n### {title}\n")
    print(out.reset_index().to_string(index=False))


table(fb_log, "raw LLM WTP ($) by indoor temp -- FIXED +/-2.0 band (balanced reused from on-file 1.0)")
table(pl_log, "raw LLM WTP ($) by indoor temp -- PERSONA-LINKED band, scale 1.0 graded")
table(pl_all, "raw LLM WTP ($) by indoor temp -- PERSONA-LINKED band, all on-file scales (more coverage)")
