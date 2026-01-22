#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------------
# User-configurable options
# -------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

RAW_ROOT="${RAW_ROOT:-$ROOT_DIR/_raw}"
RAW_GEO_DIR="${RAW_GEO_DIR:-$RAW_ROOT/geolife}"
ZIP_PATH="${ZIP_PATH:-$RAW_GEO_DIR/Geolife_Trajectories_1.3.zip}"
EXTRACT_DIR="${EXTRACT_DIR:-$RAW_GEO_DIR/extracted}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/geolife_spatial_join}"

# Official Microsoft Download Center "Download" target for GeoLife Trajectories 1.3 (may change over time).
# If it changes, override by: GEOZIP_URL="..." ./run.sh
GEOZIP_URL="${GEOZIP_URL:-https://download.microsoft.com/download/f/4/8/f4894aa5-fdbc-481e-9285-d5f8c4c4f039/Geolife%20Trajectories%201.3.zip}"

# Build controls
FORCE="${FORCE:-0}"              # FORCE=1 => delete OUTPUT_DIR then rebuild
VALIDATE="${VALIDATE:-1}"        # VALIDATE=1 => run validations after build
VALIDATE_JOIN="${VALIDATE_JOIN:-0}"  # VALIDATE_JOIN=1 => run 10k O(n^2) self-join check (slow)

# -------------------------
# Helpers
# -------------------------
need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "ERROR: need curl or wget to download files." >&2
    exit 1
  fi
}

sanity_import_check() {
  "$VENV_DIR/bin/python" - <<'PY'
import sys
mods = ["numpy", "pyproj", "pyarrow", "xxhash"]
bad = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        bad.append((m, repr(e)))
if bad:
    print("ERROR: Python deps import failed:")
    for m, e in bad:
        print(f"  - {m}: {e}")
    sys.exit(1)
print("[deps] import check OK")
PY
}

# -------------------------
# Preflight
# -------------------------
need_cmd "$PYTHON_BIN"
need_cmd unzip

if [[ ! -f "$ROOT_DIR/requirements.txt" ]]; then
  echo "ERROR: requirements.txt not found at $ROOT_DIR/requirements.txt" >&2
  exit 1
fi

if [[ -d "$OUTPUT_DIR" && "$FORCE" != "1" ]]; then
  echo "ERROR: output directory already exists: $OUTPUT_DIR" >&2
  echo "       Set FORCE=1 to rebuild (this will delete the output directory)." >&2
  exit 1
fi

mkdir -p "$RAW_GEO_DIR"
mkdir -p "$EXTRACT_DIR"

# -------------------------
# Step 1: Create venv + install deps
# -------------------------
echo "[1/4] Setting up Python virtual environment..."
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "      Python: $("$VENV_DIR/bin/python" -V)"
echo "      Upgrading pip/setuptools/wheel..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
echo "      pip: $("$VENV_DIR/bin/python" -m pip --version)"

echo "      Installing requirements.txt..."
"$VENV_DIR/bin/python" -m pip install --upgrade -r "$ROOT_DIR/requirements.txt"

echo "      Sanity import check..."
sanity_import_check

# -------------------------
# Step 2: Download GeoLife zip
# -------------------------
if [[ ! -f "$ZIP_PATH" ]]; then
  echo "[2/4] Downloading GeoLife Trajectories 1.3 zip..."
  echo "      Source: Microsoft Download Center (override by GEOZIP_URL=... if needed)"
  download_file "$GEOZIP_URL" "$ZIP_PATH"
else
  echo "[2/4] GeoLife zip already exists: $ZIP_PATH"
fi

# -------------------------
# Step 3: Extract
# -------------------------
echo "[3/4] Extracting zip (keeping original directory structure)..."
MARKER="$EXTRACT_DIR/.extracted_ok"
if [[ ! -f "$MARKER" ]]; then
  rm -rf "$EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"
  unzip -q "$ZIP_PATH" -d "$EXTRACT_DIR"
  touch "$MARKER"
else
  echo "      Already extracted: $EXTRACT_DIR"
fi

# Detect dataset root that contains Data/
INPUT_ROOT=""
if [[ -d "$EXTRACT_DIR/Geolife Trajectories 1.3/Data" ]]; then
  INPUT_ROOT="$EXTRACT_DIR/Geolife Trajectories 1.3"
elif [[ -d "$EXTRACT_DIR/Data" ]]; then
  INPUT_ROOT="$EXTRACT_DIR"
else
  CAND="$(find "$EXTRACT_DIR" -maxdepth 3 -type d -name Data -print -quit || true)"
  if [[ -n "$CAND" ]]; then
    INPUT_ROOT="$(dirname "$CAND")"
  fi
fi

if [[ -z "$INPUT_ROOT" || ! -d "$INPUT_ROOT/Data" ]]; then
  echo "ERROR: Cannot find GeoLife 'Data/' directory under: $EXTRACT_DIR" >&2
  echo "       Please inspect the extracted content and set INPUT_ROOT manually by editing run.sh." >&2
  exit 1
fi

echo "      Detected GeoLife input_root: $INPUT_ROOT"

# -------------------------
# Step 4: Build dataset
# -------------------------
echo "[4/4] Building GeoLife-Spatial-Join dataset..."
if [[ -d "$OUTPUT_DIR" && "$FORCE" == "1" ]]; then
  rm -rf "$OUTPUT_DIR"
fi

"$VENV_DIR/bin/python" "$ROOT_DIR/scripts/build_geolife_spatial_join.py"   --input_root "$INPUT_ROOT"   --output_root "$OUTPUT_DIR"

if [[ "$VALIDATE" == "1" ]]; then
  echo "Running validations..."
  "$VENV_DIR/bin/python" "$ROOT_DIR/scripts/validate_geolife_spatial_join.py"     --output_root "$OUTPUT_DIR"     --join_check "$VALIDATE_JOIN"
fi

echo
echo "DONE."
echo "Output dataset directory: $OUTPUT_DIR"
echo "Manifest: $OUTPUT_DIR/manifest.json"
