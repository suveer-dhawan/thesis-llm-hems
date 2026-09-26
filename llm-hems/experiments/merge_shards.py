"""Merge sharded run_phase3.py outputs into the summaries of one unsharded run.

Each shard is a `run_phase3.py --households <subset> --out <dir>` folder. The
shards must be the same experiment -- identical `config_resolved.yaml` apart
from the `run` block (households, out, cache path) and
`phase3.llm_cache_path` -- and cover disjoint households. Their raw
artifacts (curves.parquet, decision logs, scenario.csv, personas_used.csv,
run_meta.json) are concatenated and passed to the same
`hems.phase3_report.finalize` an unsharded run uses, so every summary
(summary.json/csv, elasticity_*, load_metrics, seed_spread*, report.md,
llm_projection.json) is what the unsharded run would have written. The
per-household overlay PNGs are copied across.

Usage (from llm-hems/):
  PYTHONPATH=src .venv/bin/python experiments/merge_shards.py SHARD_DIR [SHARD_DIR ...] --out DIR
"""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import pandas as pd
import yaml

from hems.phase3_report import Artifacts, finalize
from hems.provenance import write_provenance

RUN_SPECIFIC = ("run", "llm_clients")


def _comparable(resolved: dict) -> dict:
    c = copy.deepcopy({k: v for k, v in resolved.items() if k not in RUN_SPECIFIC})
    c["phase3"].pop("llm_cache_path", None)
    return c


def _sum_dicts(dicts: list[dict]) -> dict:
    out: dict = {}
    for d in dicts:
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                out[k] = _sum_dicts([out.get(k, {}), v])
            else:
                out[k] = out.get(k, 0) + v
    return out


def merge(shards: list[Path], out: Path) -> dict:
    resolved = [yaml.safe_load((s / "config_resolved.yaml").read_text()) for s in shards]
    base = _comparable(resolved[0])
    for s, r in zip(shards[1:], resolved[1:]):
        if _comparable(r) != base:
            raise SystemExit(f"{s}: config differs from {shards[0]} -- not shards of one run")
    steps = [r["run"]["steps"] for r in resolved]
    if any(st != steps[0] for st in steps):
        raise SystemExit(f"shards ran different steps: {steps}")

    parts = [Artifacts.read(s) for s in shards]
    seen: dict[int, Path] = {}
    for s, p in zip(shards, parts):
        for hid in p.meta["households"]:
            if hid in seen:
                raise SystemExit(f"household {hid} is in both {seen[hid]} and {s}")
            seen[hid] = s

    def cat(attr):
        frames = [getattr(p, attr) for p in parts if getattr(p, attr) is not None]
        return pd.concat(frames, ignore_index=True) if frames else None

    ids = sorted(seen)
    meta = {
        "households": ids,
        "steps": steps[0],
        "llm_prompted_decisions": sum(p.meta["llm_prompted_decisions"] for p in parts),
        "llm_live_calls": sum(p.meta["llm_live_calls"] for p in parts),
        "llm_model_versions": _sum_dicts([p.meta.get("llm_model_versions") for p in parts]),
        "llm_prompt_upper_bound": sum(p.meta["llm_prompt_upper_bound"] for p in parts),
        "llm_client_stats_by_seed": _sum_dicts(
            [p.meta.get("llm_client_stats_by_seed") for p in parts]
        ),
        "wall_seconds": sum(p.meta.get("wall_seconds", 0) for p in parts),
        "merged_from": [str(s) for s in shards],
    }
    merged = copy.deepcopy(resolved[0])
    merged["run"] = {
        "households": ids,
        "steps": steps[0],
        "out": str(out),
        "cache_paths": sorted({c for r in resolved for c in r["run"]["cache_paths"]}),
        "merged_from": [str(s) for s in shards],
    }
    merged.pop("llm_clients", None)
    if any("llm_clients" in r for r in resolved):
        merged["llm_clients_by_shard"] = {
            str(s): r.get("llm_clients") for s, r in zip(shards, resolved)
        }

    out.mkdir(parents=True, exist_ok=True)
    write_provenance(
        out,
        {"merged_from": [str(s) for s in shards],
         "shard_provenance": {str(s): yaml.safe_load((s / "provenance.json").read_text())
                              for s in shards if (s / "provenance.json").exists()}},
    )  # fmt: skip
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(merged, sort_keys=False))
    for s in shards:
        for png in s.glob("overlay_*.png"):
            if png.name != "overlay_population.png":
                shutil.copy2(png, out / png.name)
    artifacts = Artifacts(
        curves=cat("curves"),
        personas=cat("personas"),
        meta=meta,
        scenario=cat("scenario"),
        llm_log=cat("llm_log"),
        rule_log=cat("rule_log"),
    )
    return finalize(out, artifacts, merged)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    merge(args.shards, args.out)
    print(f"[merge_shards] {len(args.shards)} shards -> {args.out}/")
    return args.out


if __name__ == "__main__":
    main()
