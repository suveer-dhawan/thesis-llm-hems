"""Sharded run_phase3 dry runs merged by merge_shards == one unsharded dry run."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("experiments").resolve()))

import merge_shards  # noqa: E402
import run_phase3  # noqa: E402

SHARD_1 = [8596070, 9911140]
SHARD_2 = [8211127, 10278838]
# run-specific by design: households/out/cache per shard, timestamps, shard list
RUN_SPECIFIC = {"provenance.json", "config_resolved.yaml", "run_meta.json"}


def _dry(ids, out):
    run_phase3.main(["--dry-run", "--households", ",".join(map(str, ids)), "--out", str(out)])


def test_two_shards_merged_equal_unsharded(tmp_path):
    _dry(SHARD_1 + SHARD_2, tmp_path / "full")
    _dry(SHARD_1, tmp_path / "s1")
    _dry(SHARD_2, tmp_path / "s2")
    merge_shards.main([str(tmp_path / "s1"), str(tmp_path / "s2"), "--out", str(tmp_path / "m")])

    full = {p.name for p in (tmp_path / "full").iterdir()}
    merged = {p.name for p in (tmp_path / "m").iterdir()}
    assert full == merged
    assert {"summary.json", "summary.csv", "report.md", "curves.parquet"} <= full
    for name in sorted(full - RUN_SPECIFIC):
        a, b = tmp_path / "full" / name, tmp_path / "m" / name
        if name.endswith(".parquet"):
            pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))
        else:
            assert a.read_bytes() == b.read_bytes(), name


def test_merge_rejects_overlapping_shards(tmp_path):
    _dry(SHARD_1, tmp_path / "s1")
    _dry(SHARD_1[:1], tmp_path / "s2")
    try:
        merge_shards.main(
            [str(tmp_path / "s1"), str(tmp_path / "s2"), "--out", str(tmp_path / "m")]
        )
    except SystemExit as e:
        assert "both" in str(e)
    else:
        raise AssertionError("overlapping shards were merged")
