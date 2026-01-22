#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GeoLife-Spatial-Join Builder (GeoLife 1.3)

Implements the building spec described in the conversation:
- Deterministic scan order of .plt under input_root/Data
- Parse .plt skipping first 6 lines
- Point-level filter: lat/lon range + datetime parse success
- CRS: EPSG:4326 -> EPSG:3857
- Encoding: x/y in cm int (round half away from zero), t in ms, z in cm (round half away from zero)
- 3D: (x, y, t) using all points
- 4D: (x, y, z, t) using only valid-altitude points
- Levels: three encounter thresholds, expressed as axis-aligned boxes with L∞ equivalence
- IDs:
  - traj_id = xxhash64(seed=0) of traj_src UTF-8 bytes
  - rect_id = xxhash64(seed=0) of concat:
      traj_id(int64 little-endian) + point_idx(int32 little-endian) + dims(int8) + level(int8)
- Output:
  geolife_spatial_join/
    manifest.json
    dims=3/level=1/part-00000.parquet ...
    dims=4/level=1/part-00000.parquet ...
    dict/trajectories.parquet
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
from pyproj import Transformer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


# -------------------------
# Spec constants
# -------------------------

DATASET_NAME = "GeoLife-Spatial-Join"
GEOLIFE_VERSION = "1.3"

CRS_INPUT = "EPSG:4326"
CRS_OUTPUT = "EPSG:3857"

ALT_INVALID_FLAG_FT = -777.0
ALT_RANGE_M = (-500.0, 10000.0)

# Encounter levels (Δd meters, Δt seconds) and derived radii:
# r_d_cm = (Δd * 100)/2 ; r_t_ms = (Δt * 1000)/2
LEVELS: Dict[int, Dict[str, int]] = {
    1: {"delta_d_m": 20, "delta_t_s": 60, "r_d_cm": 1000, "r_t_ms": 30000},
    2: {"delta_d_m": 50, "delta_t_s": 300, "r_d_cm": 2500, "r_t_ms": 150000},
    3: {"delta_d_m": 200, "delta_t_s": 1200, "r_d_cm": 10000, "r_t_ms": 600000},
}

# Storage parameters (target sizes)
ROW_GROUP_BYTES = 128 * 1024 * 1024
TARGET_FILE_BYTES = 512 * 1024 * 1024
COMPRESSION = "zstd"


# -------------------------
# Utility: rounding & hashing
# -------------------------

def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """
    round_half_away_from_zero:
      sign(x) * floor(abs(x) + 0.5)
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def u64_to_i64(u: int) -> int:
    """Interpret unsigned 64-bit integer as signed int64 (two's complement)."""
    if u >= (1 << 63):
        return u - (1 << 64)
    return u


def xxh64_i64(data: bytes, seed: int = 0) -> int:
    """xxHash64(seed) => signed int64."""
    u = xxhash.xxh64(data, seed=seed).intdigest()
    return u64_to_i64(u)


def compute_traj_id(traj_src: str) -> int:
    """traj_id = xxHash64(seed=0) of UTF-8 bytes of traj_src."""
    return xxh64_i64(traj_src.encode("utf-8"), seed=0)


def compute_rect_ids_for_levels(
    traj_id: int,
    point_idx: np.ndarray,
    dims: int,
    levels: Iterable[int],
) -> Dict[int, np.ndarray]:
    """
    rect_id bytes layout (little-endian):
      traj_id: int64   (8 bytes)
      point_idx: int32 (4 bytes)
      dims: int8       (1 byte)
      level: int8      (1 byte)
    Total 14 bytes.
    """
    levels_list = list(levels)
    n = int(point_idx.shape[0])

    out: Dict[int, np.ndarray] = {lv: np.empty(n, dtype=np.int64) for lv in levels_list}

    buf = bytearray(14)
    struct.pack_into("<q", buf, 0, int(traj_id))  # int64 little-endian
    buf[12] = int(dims) & 0xFF  # dims int8

    for i in range(n):
        struct.pack_into("<i", buf, 8, int(point_idx[i]))  # point_idx int32 little-endian
        for lv in levels_list:
            buf[13] = int(lv) & 0xFF  # level int8
            u = xxhash.xxh64(buf, seed=0).intdigest()
            out[lv][i] = u64_to_i64(u)

    return out


# -------------------------
# PLT parsing
# -------------------------

def parse_datetime_utc(date_str: str, time_str: str) -> Optional[dt.datetime]:
    """Parse GeoLife .plt date+time. Interpreted as UTC."""
    try:
        y = int(date_str[0:4]); m = int(date_str[5:7]); d = int(date_str[8:10])
        hh = int(time_str[0:2]); mm = int(time_str[3:5]); ss = int(time_str[6:8])
        return dt.datetime(y, m, d, hh, mm, ss)
    except Exception:
        return None


def datetime_to_epoch_ms_utc(dtime: dt.datetime) -> int:
    """Convert naive datetime assumed UTC into epoch milliseconds (floor)."""
    return int(calendar.timegm(dtime.timetuple()) * 1000 + (dtime.microsecond // 1000))


@dataclass
class ParsedTrajectory:
    traj_src: str
    traj_id: int
    user_id: int

    point_idx: np.ndarray   # int64 during processing
    lon: np.ndarray         # float64
    lat: np.ndarray         # float64
    alt_ft: np.ndarray      # float64
    t_ms: np.ndarray        # int64


def parse_plt_file(data_dir: Path, plt_path: Path) -> Optional[ParsedTrajectory]:
    """Parse one .plt and apply §2.5 filter."""
    traj_src = plt_path.relative_to(data_dir).as_posix()  # normalized separator "/"

    try:
        user_dir = traj_src.split("/", 1)[0]
        user_id = int(user_dir, 10)
    except Exception:
        return None

    traj_id = compute_traj_id(traj_src)

    lat_list: List[float] = []
    lon_list: List[float] = []
    alt_list: List[float] = []
    t_list: List[int] = []
    idx_list: List[int] = []

    with plt_path.open("r", encoding="utf-8", errors="ignore") as f:
        # Skip header 6 lines
        for _ in range(6):
            _ = f.readline()

        for raw_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue

            try:
                lat = float(parts[0]); lon = float(parts[1])
                alt_ft = float(parts[3])
                date_str = parts[5].strip(); time_str = parts[6].strip()
            except Exception:
                continue

            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue

            dtime = parse_datetime_utc(date_str, time_str)
            if dtime is None:
                continue

            t_ms = datetime_to_epoch_ms_utc(dtime)

            lat_list.append(lat)
            lon_list.append(lon)
            alt_list.append(alt_ft)
            t_list.append(t_ms)
            idx_list.append(raw_idx)  # point_idx within the trajectory file after header

    if len(idx_list) == 0:
        return None

    return ParsedTrajectory(
        traj_src=traj_src,
        traj_id=traj_id,
        user_id=user_id,
        point_idx=np.asarray(idx_list, dtype=np.int64),
        lon=np.asarray(lon_list, dtype=np.float64),
        lat=np.asarray(lat_list, dtype=np.float64),
        alt_ft=np.asarray(alt_list, dtype=np.float64),
        t_ms=np.asarray(t_list, dtype=np.int64),
    )


# -------------------------
# Parquet writing with part files
# -------------------------

@dataclass
class FileStat:
    path: str
    rows: int


class PartParquetWriter:
    """
    Write Parquet into part-%05d.parquet files under a group directory, approximately matching:
      - compression: ZSTD
      - target file size: 512 MiB (approx by rows)
      - row group size: 128 MiB (approx by rows)
    """

    def __init__(self, output_root: Path, group_dir: Path, schema: pa.Schema, bytes_per_row: int):
        self.output_root = output_root
        self.group_dir = group_dir
        self.schema = schema
        self.bytes_per_row = int(bytes_per_row)

        self.row_group_rows = max(1, ROW_GROUP_BYTES // self.bytes_per_row)
        self.max_rows_per_file = max(1, TARGET_FILE_BYTES // self.bytes_per_row)

        self.part_idx = 0
        self.writer: Optional[pq.ParquetWriter] = None
        self.cur_rows = 0
        self.total_rows = 0
        self.file_stats: List[FileStat] = []

        self.group_dir.mkdir(parents=True, exist_ok=True)

    def _open_new(self) -> None:
        assert self.writer is None
        out_path = self.group_dir / f"part-{self.part_idx:05d}.parquet"
        self.writer = pq.ParquetWriter(
            where=str(out_path),
            schema=self.schema,
            compression=COMPRESSION,
            use_dictionary=False,
        )
        self.cur_rows = 0

    def _close_current(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        out_path = self.group_dir / f"part-{self.part_idx:05d}.parquet"
        rel_path = out_path.relative_to(self.output_root).as_posix()
        self.file_stats.append(FileStat(path=rel_path, rows=int(self.cur_rows)))
        self.writer = None

    def close(self) -> None:
        self._close_current()

    def write_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return

        if table.schema != self.schema:
            table = table.cast(self.schema)

        start = 0
        n = table.num_rows
        while start < n:
            if self.writer is None:
                self._open_new()

            remaining = self.max_rows_per_file - self.cur_rows
            if remaining <= 0:
                self._close_current()
                self.part_idx += 1
                continue

            take = min(remaining, n - start)
            chunk = table.slice(start, take)
            self.writer.write_table(chunk, row_group_size=int(self.row_group_rows))
            self.cur_rows += take
            self.total_rows += take
            start += take


# -------------------------
# Schemas
# -------------------------

def schema_rect(dims: int) -> pa.Schema:
    fields = [
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
        fields.extend([
            pa.field("z_min_cm", pa.int64()),
            pa.field("z_max_cm", pa.int64()),
        ])
    return pa.schema(fields)


TRAJ_DICT_SCHEMA = pa.schema([
    pa.field("traj_id", pa.int64()),
    pa.field("user_id", pa.int32()),
    pa.field("traj_src", pa.string()),
    pa.field("n_points_all", pa.int32()),
    pa.field("t_start_ms", pa.int64()),
    pa.field("t_end_ms", pa.int64()),
])


# -------------------------
# Main build
# -------------------------

def collect_plt_files(data_dir: Path) -> List[Path]:
    all_paths = list(data_dir.rglob("*.plt"))
    all_paths.sort(key=lambda p: p.relative_to(data_dir).as_posix())
    return all_paths


def build(input_root: Path, output_root: Path) -> None:
    data_dir = input_root / "Data"
    if not data_dir.is_dir():
        raise SystemExit(f"ERROR: input_root does not contain Data/: {data_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    dict_dir = output_root / "dict"
    dict_dir.mkdir(parents=True, exist_ok=True)

    # Writers for 6 groups
    writers: Dict[Tuple[int, int], PartParquetWriter] = {}

    # Uncompressed bytes-per-row estimate:
    # 3D: 2x int64 + 2x int32 + 6x int64 = 72 bytes
    # 4D: +2x int64 => 88 bytes
    bpr_3d = 72
    bpr_4d = 88

    for dims in (3, 4):
        for level in (1, 2, 3):
            group_dir = output_root / f"dims={dims}" / f"level={level}"
            sch = schema_rect(dims)
            writers[(dims, level)] = PartParquetWriter(
                output_root=output_root,
                group_dir=group_dir,
                schema=sch,
                bytes_per_row=(bpr_3d if dims == 3 else bpr_4d),
            )

    group_user_sets: Dict[Tuple[int, int], set] = {(d, l): set() for d in (3, 4) for l in (1, 2, 3)}
    group_traj_sets: Dict[Tuple[int, int], set] = {(d, l): set() for d in (3, 4) for l in (1, 2, 3)}

    traj_dict_rows: List[dict] = []

    transformer = Transformer.from_crs(CRS_INPUT, CRS_OUTPUT, always_xy=True)

    plt_files = collect_plt_files(data_dir)
    it = plt_files if tqdm is None else tqdm(plt_files, desc="Scanning .plt", unit="file")

    for plt_path in it:
        parsed = parse_plt_file(data_dir, plt_path)
        if parsed is None:
            continue

        traj_id = parsed.traj_id
        user_id = parsed.user_id

        n_points_all = int(parsed.point_idx.shape[0])
        traj_dict_rows.append({
            "traj_id": int(traj_id),
            "user_id": int(user_id),
            "traj_src": parsed.traj_src,
            "n_points_all": int(n_points_all),
            "t_start_ms": int(parsed.t_ms.min()),
            "t_end_ms": int(parsed.t_ms.max()),
        })

        # Project & encode XY (meters -> cm int)
        x_m, y_m = transformer.transform(parsed.lon, parsed.lat)
        x_m = np.asarray(x_m, dtype=np.float64)
        y_m = np.asarray(y_m, dtype=np.float64)
        x_cm = round_half_away_from_zero(x_m * 100.0).astype(np.int64)
        y_cm = round_half_away_from_zero(y_m * 100.0).astype(np.int64)

        t_ms = parsed.t_ms.astype(np.int64, copy=False)

        # point_idx for output columns is int32
        point_idx_i32 = parsed.point_idx.astype(np.int32, copy=False)

        # Altitude conversion & mask for 4D
        alt_ft = parsed.alt_ft.astype(np.float64, copy=False)
        alt_m = alt_ft * 0.3048
        z_cm = round_half_away_from_zero(alt_m * 100.0).astype(np.int64)
        valid_z = (alt_ft != ALT_INVALID_FLAG_FT) & (alt_m >= ALT_RANGE_M[0]) & (alt_m <= ALT_RANGE_M[1])

        # manifest sets
        for level in (1, 2, 3):
            group_user_sets[(3, level)].add(int(user_id))
            group_traj_sets[(3, level)].add(int(traj_id))
        if bool(valid_z.any()):
            for level in (1, 2, 3):
                group_user_sets[(4, level)].add(int(user_id))
                group_traj_sets[(4, level)].add(int(traj_id))

        # rect ids
        rect_ids_3d = compute_rect_ids_for_levels(traj_id=traj_id, point_idx=parsed.point_idx, dims=3, levels=(1, 2, 3))

        # Write dims=3 groups
        traj_id_col_3d = np.full(n_points_all, traj_id, dtype=np.int64)
        user_id_col_3d = np.full(n_points_all, user_id, dtype=np.int32)

        for level in (1, 2, 3):
            r_d_cm = LEVELS[level]["r_d_cm"]
            r_t_ms = LEVELS[level]["r_t_ms"]

            table = pa.Table.from_pydict({
                "rect_id": rect_ids_3d[level],
                "traj_id": traj_id_col_3d,
                "user_id": user_id_col_3d,
                "point_idx": point_idx_i32,
                "x_min_cm": x_cm - r_d_cm,
                "x_max_cm": x_cm + r_d_cm,
                "y_min_cm": y_cm - r_d_cm,
                "y_max_cm": y_cm + r_d_cm,
                "t_min_ms": t_ms - r_t_ms,
                "t_max_ms": t_ms + r_t_ms,
            }, schema=schema_rect(3))

            writers[(3, level)].write_table(table)

        # Write dims=4 groups (only valid altitude points)
        if bool(valid_z.any()):
            v_idx = parsed.point_idx[valid_z]
            v_n = int(v_idx.shape[0])

            v_point_idx_i32 = v_idx.astype(np.int32, copy=False)
            v_x_cm = x_cm[valid_z]
            v_y_cm = y_cm[valid_z]
            v_t_ms = t_ms[valid_z]
            v_z_cm = z_cm[valid_z]

            rect_ids_4d = compute_rect_ids_for_levels(traj_id=traj_id, point_idx=v_idx, dims=4, levels=(1, 2, 3))

            traj_id_col_4d = np.full(v_n, traj_id, dtype=np.int64)
            user_id_col_4d = np.full(v_n, user_id, dtype=np.int32)

            for level in (1, 2, 3):
                r_d_cm = LEVELS[level]["r_d_cm"]
                r_t_ms = LEVELS[level]["r_t_ms"]

                table = pa.Table.from_pydict({
                    "rect_id": rect_ids_4d[level],
                    "traj_id": traj_id_col_4d,
                    "user_id": user_id_col_4d,
                    "point_idx": v_point_idx_i32,
                    "x_min_cm": v_x_cm - r_d_cm,
                    "x_max_cm": v_x_cm + r_d_cm,
                    "y_min_cm": v_y_cm - r_d_cm,
                    "y_max_cm": v_y_cm + r_d_cm,
                    "t_min_ms": v_t_ms - r_t_ms,
                    "t_max_ms": v_t_ms + r_t_ms,
                    "z_min_cm": v_z_cm - r_d_cm,
                    "z_max_cm": v_z_cm + r_d_cm,
                }, schema=schema_rect(4))

                writers[(4, level)].write_table(table)

    # Close writers and collect file stats
    files_manifest: Dict[str, List[dict]] = {}
    counts_manifest: Dict[str, dict] = {}

    for (dims, level), w in writers.items():
        w.close()
        group_key = f"dims={dims}/level={level}"
        files_manifest[group_key] = [{"path": fs.path, "rows": fs.rows} for fs in w.file_stats]
        counts_manifest[group_key] = {
            "rows": int(w.total_rows),
            "users": int(len(group_user_sets[(dims, level)])),
            "trajectories": int(len(group_traj_sets[(dims, level)])),
        }

    # dict/trajectories.parquet
    traj_table = pa.Table.from_pylist(traj_dict_rows, schema=TRAJ_DICT_SCHEMA)
    pq.write_table(
        traj_table,
        where=str(dict_dir / "trajectories.parquet"),
        compression=COMPRESSION,
        use_dictionary=False,
    )

    # manifest.json
    manifest = {
        "dataset_name": DATASET_NAME,
        "geolife_version": GEOLIFE_VERSION,
        "build_time_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "input_root": str(input_root),
        "crs_input": CRS_INPUT,
        "crs_output": CRS_OUTPUT,
        "units": {"xy": "cm", "z": "cm", "t": "ms"},
        "levels": [
            {
                "level": lv,
                "delta_d_m": LEVELS[lv]["delta_d_m"],
                "delta_t_s": LEVELS[lv]["delta_t_s"],
                "r_d_cm": LEVELS[lv]["r_d_cm"],
                "r_t_ms": LEVELS[lv]["r_t_ms"],
            }
            for lv in (1, 2, 3)
        ],
        "altitude_validity": {
            "invalid_flag_ft": ALT_INVALID_FLAG_FT,
            "range_m": [ALT_RANGE_M[0], ALT_RANGE_M[1]],
        },
        "counts": counts_manifest,
        "files": files_manifest,
        "hashing": {"algo": "xxhash64", "seed": 0, "endianness": "little"},
    }

    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, help="GeoLife extracted root directory (must contain Data/)")
    ap.add_argument("--output_root", required=True, help="Output directory: geolife_spatial_join/")
    args = ap.parse_args()

    build(Path(args.input_root), Path(args.output_root))


if __name__ == "__main__":
    main()
