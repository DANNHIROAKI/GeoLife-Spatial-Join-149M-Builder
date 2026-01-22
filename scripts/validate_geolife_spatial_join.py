#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


LEVELS = {
    1: {"r_d_cm": 1000, "r_t_ms": 30000},
    2: {"r_d_cm": 2500, "r_t_ms": 150000},
    3: {"r_d_cm": 10000, "r_t_ms": 600000},
}


def expected_schema(dims: int) -> pa.Schema:
    base = [
        pa.field("rect_id", pa.int64()),
        pa.field("traj_id", pa.int64()),
        pa.field("user_id", pa.int32()),
        pa.field("point_idx", pa.int32()),
        pa.field("x_min_cm", pa.int64()),
        pa.field("x_max_cm", pa.int64()),
        pa.field("y_min_cm", pa.int64()),
        pa.field("y_max_cm", pa.int64()),
        pa.field("t_min_ms", pa.int64()),
        pa.field("t_max_ms", pa.int64()),
    ]
    if dims == 4:
        base += [pa.field("z_min_cm", pa.int64()), pa.field("z_max_cm", pa.int64())]
    return pa.schema(base)


def load_first_rows(group_dir: Path, n: int, columns: List[str]) -> pa.Table:
    dataset = ds.dataset(str(group_dir), format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=200_000)
    batches = []
    got = 0
    for b in scanner.to_batches():
        if got >= n:
            break
        take = min(n - got, b.num_rows)
        batches.append(b.slice(0, take))
        got += take
    if not batches:
        return pa.Table.from_pydict({c: pa.array([], type=pa.int64()) for c in columns})
    return pa.Table.from_batches(batches)


def check_schema_one_file(group_dir: Path, dims: int) -> None:
    files = sorted(group_dir.glob("part-*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet parts found under {group_dir}")
    t = pq.read_table(files[0])
    exp = expected_schema(dims)

    cols = set(t.schema.names)
    if dims == 3:
        if "z_min_cm" in cols or "z_max_cm" in cols:
            raise AssertionError(f"{group_dir}: dims=3 must NOT contain z_* columns")
    else:
        if "z_min_cm" not in cols or "z_max_cm" not in cols:
            raise AssertionError(f"{group_dir}: dims=4 must contain z_* columns")

    for f in exp:
        if f.name not in t.schema.names:
            raise AssertionError(f"{group_dir}: missing column {f.name}")
        if t.schema.field(f.name).type != f.type:
            raise AssertionError(
                f"{group_dir}: column {f.name} type mismatch: {t.schema.field(f.name).type} vs {f.type}"
            )

    if dims == 4:
        if t.column("z_min_cm").null_count != 0 or t.column("z_max_cm").null_count != 0:
            raise AssertionError(f"{group_dir}: dims=4 z columns must have no nulls")


def check_boundary_identities(sample: pa.Table, dims: int, level: int) -> None:
    r_d = LEVELS[level]["r_d_cm"]
    r_t = LEVELS[level]["r_t_ms"]

    def col(name: str) -> np.ndarray:
        return sample[name].to_numpy(zero_copy_only=False)

    x_min = col("x_min_cm"); x_max = col("x_max_cm")
    y_min = col("y_min_cm"); y_max = col("y_max_cm")
    t_min = col("t_min_ms"); t_max = col("t_max_ms")

    assert np.all(x_min <= x_max), "x_min <= x_max violated"
    assert np.all(y_min <= y_max), "y_min <= y_max violated"
    assert np.all(t_min <= t_max), "t_min <= t_max violated"

    assert np.all((x_max - x_min) == 2 * r_d), "x span != 2*r_d"
    assert np.all((y_max - y_min) == 2 * r_d), "y span != 2*r_d"
    assert np.all((t_max - t_min) == 2 * r_t), "t span != 2*r_t"

    if dims == 4:
        z_min = col("z_min_cm"); z_max = col("z_max_cm")
        assert np.all((z_max - z_min) == 2 * r_d), "z span != 2*r_d"


def naive_self_join_intersections(sample: pa.Table, dims: int) -> int:
    """
    Naive O(n^2) closed-interval intersection check within sample (self-join).
    Returns number of intersecting pairs (i<j).
    """
    x_min = sample["x_min_cm"].to_numpy(zero_copy_only=False)
    x_max = sample["x_max_cm"].to_numpy(zero_copy_only=False)
    y_min = sample["y_min_cm"].to_numpy(zero_copy_only=False)
    y_max = sample["y_max_cm"].to_numpy(zero_copy_only=False)
    t_min = sample["t_min_ms"].to_numpy(zero_copy_only=False)
    t_max = sample["t_max_ms"].to_numpy(zero_copy_only=False)

    if dims == 4:
        z_min = sample["z_min_cm"].to_numpy(zero_copy_only=False)
        z_max = sample["z_max_cm"].to_numpy(zero_copy_only=False)

    n = sample.num_rows
    cnt = 0
    for i in range(n):
        j = np.arange(i + 1, n)
        ok = (x_min[i] <= x_max[j]) & (x_max[i] >= x_min[j]) \
             & (y_min[i] <= y_max[j]) & (y_max[i] >= y_min[j]) \
             & (t_min[i] <= t_max[j]) & (t_max[i] >= t_min[j])
        if dims == 4:
            ok = ok & (z_min[i] <= z_max[j]) & (z_max[i] >= z_min[j])
        cnt += int(ok.sum())
    return cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--join_check", type=int, default=0, help="1 to run 10k naive O(n^2) self-join check (slow)")
    ap.add_argument("--sample_rows", type=int, default=50_000, help="Rows to sample from head of each group for checks")
    args = ap.parse_args()

    out = Path(args.output_root)

    # Schema checks
    for dims in (3, 4):
        for level in (1, 2, 3):
            group_dir = out / f"dims={dims}" / f"level={level}"
            check_schema_one_file(group_dir, dims)

    # Boundary identity checks on first sample_rows rows
    for dims in (3, 4):
        for level in (1, 2, 3):
            group_dir = out / f"dims={dims}" / f"level={level}"
            cols = expected_schema(dims).names
            sample = load_first_rows(group_dir, n=int(args.sample_rows), columns=cols)
            check_boundary_identities(sample, dims=dims, level=level)

    print("[OK] schema + boundary identity checks passed.")

    if args.join_check == 1:
        for dims in (3, 4):
            for level in (1, 2, 3):
                group_dir = out / f"dims={dims}" / f"level={level}"
                cols = expected_schema(dims).names
                sample = load_first_rows(group_dir, n=10_000, columns=cols)
                cnt = naive_self_join_intersections(sample, dims=dims)
                print(f"[JOIN-SELF] dims={dims} level={level}  n={sample.num_rows}  intersect_pairs={cnt}")

        print("[OK] naive self-join checks completed.")


if __name__ == "__main__":
    main()
