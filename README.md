# How to Start

> #### One-Click Run
>
> ```bash
> chmod +x run.sh
> ./run.sh
> 
> ```
>
> #### Common Environment Variables
>
> **Force Rebuild** (will delete the output directory):
>
> ```bash
> FORCE=1 ./run.sh
> ```
>
> **Skip Validation**:
>
> ```bash
> VALIDATE=0 ./run.sh
> ```
>
> **Enable 10k  Self-Join Check**:
>
> ```bash
> VALIDATE_JOIN=1 ./run.sh
> ```
>
> **Override Download URL** (if the Microsoft direct link changes):
>
> ```bash
> GEOZIP_URL="https://..." ./run.sh
> ```
>
> #### Default output directory:
>
> ```text
> geolife_spatial_join/
>   manifest.json
>   dims=3/level=1/part-00000.parquet ...
>   dims=3/level=2/...
>   dims=3/level=3/...
>   dims=4/level=1/...
>   dims=4/level=2/...
>   dims=4/level=3/...
>   dict/trajectories.parquet
> ```

# Building Spec.

> ## 1. Dataset Definition and Intersection Semantics
>
> This specification constructs an **axis-aligned rectangle / hyperrectangle** dataset based on **GeoLife Trajectories 1.3** (GeoLife 1.3), for high-dimensional SpatialJoin benchmark evaluation of **rectangle–rectangle intersection join (intersection join)**.
>
> ### 1.1 Dimensions and Dataset Groups
>
> Construct six dataset groups:
>
> - **3D**: $(x, y, t)$
>   - `dims=3, level=1` (all points)
>   - `dims=3, level=2` (all points)
>   - `dims=3, level=3` (all points)
> - **4D**: $(x, y, z, t)$
>   - `dims=4, level=1` (valid-altitude points only)
>   - `dims=4, level=2` (valid-altitude points only)
>   - `dims=4, level=3` (valid-altitude points only)
>
> ### 1.2 Encounter Level Parameters (Thresholds)
>
> Define three encounter intensity levels using distance threshold $\Delta d$ and time threshold $\Delta t$:
>
> - Level 1: $\Delta d = 20$ m, $\Delta t = 60$ s  
> - Level 2: $\Delta d = 50$ m, $\Delta t = 300$ s  
> - Level 3: $\Delta d = 200$ m, $\Delta t = 1200$ s  
>
> ### 1.3 Intersection Join Equivalence ($L_\infty$)
>
> This dataset uses **only axis-aligned box intersection tests**, and guarantees equivalence to the thresholds under **$L_\infty$ semantics**. For the same level:
>
> - Construct a box centered at each point. The spatial half-width is $r_d=\Delta d/2$, and the temporal half-width is $r_t=\Delta t/2$;
>
> - Two boxes intersect **iff** the corresponding points satisfy:
> 
>   $$
>   |x_1-x_2|\le \Delta d,\quad |y_1-y_2|\le \Delta d,\quad |t_1-t_2|\le \Delta t
>   $$
> 
>   (For 4D, additionally include $|z_1-z_2|\le \Delta d$.)
>
> All dimensions use **closed intervals** $[min, max]$, and the intersection predicate is:
> `A.min <= B.max && A.max >= B.min` (must hold simultaneously for all dimensions).
>
> ------
>
> ## 2. Input Data and Point Record Extraction
>
> ### 2.1 Input Directory
>
> The input is the root directory after extracting GeoLife 1.3. The directory contains `Data/`, which is organized by user directories (e.g., `Data/000/...`), and trajectory files are `.plt`.
>
> ### 2.2 Trajectory File Scan Order
>
> The build process traverses `.plt` files in a deterministic order:
>
> 1. Recursively collect all `.plt` paths under `Data/`;
> 2. Convert each path to a path relative to `Data/`;
> 3. Normalize using `/` as the separator;
> 4. Sort by the relative path in lexicographic order;
> 5. Process each `.plt` sequentially in that order.
>
> ### 2.3 `.plt` Parsing Rules
>
> For each `.plt` file:
>
> - Skip the first 6 header lines; parse line by line starting from line 7;
> - Each line is comma-separated; read fields:
>   - `lat` (latitude, decimal degrees)
>   - `lon` (longitude, decimal degrees)
>   - the 3rd column is read but not used for geometric dimension construction
>   - `alt_ft` (altitude in feet; invalid altitude is marked as `-777`)
>   - combine the date and time fields into `datetime`
>
> ### 2.4 Point Record
>
> Each trajectory point generates one point record with the following fields:
>
> - `user_id`: parse the user directory name as a decimal integer (`"000" → 0`)
> - `traj_src`: normalized relative path (e.g., `000/Trajectory/20081023025304.plt`)
> - `traj_id`: derived from `traj_src` (see §4.2)
> - `point_idx`: the point index within the trajectory, starting at 0 and incrementing
> - `lat`, `lon`: float64
> - `alt_ft`: float64
> - `t_ms`: timestamp in milliseconds (see §3.3)
>
> ### 2.5 Point-Level Filtering
>
> A point record can enter downstream processing only if:
>
> - `lat ∈ [-90, 90]`
> - `lon ∈ [-180, 180]`
> - `datetime` is successfully parsed
>
> ------
>
> ## 3. Numeric Encoding (Space, Time, Altitude)
>
> ### 3.1 Coordinate Projection
>
> All spatial coordinates use:
>
> - Input: WGS84 (EPSG:4326)
> - Output: Web Mercator (EPSG:3857)
>
> The projection result is $(x_m, y_m)$ in meters.
>
> ### 3.2 Spatial Integer Encoding (Centimeters)
>
> Convert meter coordinates to centimeter integers:
>
> - $x_{cm} = \text{round\_away\_from\_zero}(x_m \times 100)$
> - $y_{cm} = \text{round\_away\_from\_zero}(y_m \times 100)$
>
> Definition of `round_away_from_zero`:
>
> - If the absolute fractional part is greater than 0.5, round to the nearest integer;
> - If the absolute fractional part equals 0.5, round to the integer **away from 0**;
> - Otherwise, round to the nearest integer (standard rounding to nearest).
>
> ### 3.3 Time Integer Encoding (Milliseconds)
>
> Interpret `datetime` as UTC time and convert it to Unix epoch milliseconds:
>
> - $t_{ms} = \lfloor \text{timestamp\_seconds} \times 1000 \rfloor$
>
> ### 3.4 Altitude Encoding (Centimeters) and Valid-Altitude Points
>
> Altitude input is in feet:
>
> - $alt_m = alt_{ft} \times 0.3048$
> - $z_{cm} = \text{round\_away\_from\_zero}(alt_m \times 100)$
>
> A point is considered a valid-altitude point if:
>
> - `alt_ft != -777`
> - `alt_m ∈ [-500, 10000]`
>
> The 4D datasets use only valid-altitude points; the 3D datasets use all points and do not introduce the $z$ dimension.
>
> ------
>
> ## 4. Box Construction, ID Rules, and Six Outputs
>
> ### 4.1 Radius Table (Fixed by Level)
>
> Spatial units use cm, and time units use ms. For each level:
>
> - $r_{d,cm} = (\Delta d_m \times 100)/2$
> - $r_{t,ms} = (\Delta t_s \times 1000)/2$
>
> | Level | $\Delta d$ (m) | $r_{d,cm}$ | $\Delta t$ (s) | $r_{t,ms}$ |
> | ----- | -------------- | ---------- | -------------- | ---------- |
> | 1     | 20             | 1000       | 60             | 30000      |
> | 2     | 50             | 2500       | 300            | 150000     |
> | 3     | 200            | 10000      | 1200           | 600000     |
>
> ### 4.2 Trajectory ID and Rectangle ID (Deterministic)
>
> To achieve consistency across languages and platforms, IDs are generated by hashing fixed byte sequences. The hash function is **xxHash64** with a fixed seed of 0.
>
> - `traj_id`: compute xxHash64 on the UTF-8 byte sequence of `traj_src`.
>   - `traj_src` is a normalized relative path, uses `/` as the separator, contains no drive letter, and contains no `.` or `..`.
>
> - `rect_id`: compute xxHash64 on the following byte sequence (concatenated in order):
>   1. `traj_id`: int64, little-endian
>   2. `point_idx`: int32, little-endian
>   3. `dims`: int8 (3 or 4)
>   4. `level`: int8 (1/2/3)
>
> ### 4.3 Box Boundary Computation
>
> For each point center $(x_{cm}, y_{cm}, t_{ms})$:
>
> **3D (x, y, t)**, generated for all points:
>
> - `x_min_cm = x_cm - r_d_cm`
> - `x_max_cm = x_cm + r_d_cm`
> - `y_min_cm = y_cm - r_d_cm`
> - `y_max_cm = y_cm + r_d_cm`
> - `t_min_ms = t_ms - r_t_ms`
> - `t_max_ms = t_ms + r_t_ms`
>
> **4D (x, y, z, t)**, generated only for valid-altitude points, additionally:
>
> - `z_min_cm = z_cm - r_d_cm`
> - `z_max_cm = z_cm + r_d_cm`
>
> ### 4.4 Definition of Six Output Groups
>
> Output six groups by `dims` and `level`. Each point generates one rectangle record for each level:
>
> - `dims=3/level=1`, `dims=3/level=2`, `dims=3/level=3`: input is all points
> - `dims=4/level=1`, `dims=4/level=2`, `dims=4/level=3`: input is valid-altitude points only
>
> ------
>
> ## 5. Storage Layout, Parquet Schema, and Manifest
>
> ### 5.1 Output Directory Structure
>
> The output root directory is `geolife_spatial_join/` with the following structure:
>
> ```
> 
> geolife_spatial_join/
> manifest.json
> dims=3/
> level=1/part-00000.parquet
> level=2/part-00000.parquet
> level=3/part-00000.parquet
> dims=4/
> level=1/part-00000.parquet
> level=2/part-00000.parquet
> level=3/part-00000.parquet
> dict/
> trajectories.parquet
> 
> ```
>
> ### 5.2 Parquet Write Parameters
>
> Write parameters are fixed as:
>
> - Compression: ZSTD
> - Row group size: 128 MiB
> - Target single-file size: 512 MiB
> - File naming: `part-%05d.parquet`, starting from `part-00000.parquet` and incrementing
>
> ### 5.3 Rectangle Record Schema
>
> All groups contain the following fields (except for $z$ bounds):
>
> Common metadata:
>
> - `rect_id: int64`
> - `traj_id: int64`
> - `user_id: int32`
> - `point_idx: int32`
>
> 3D bounds:
>
> - `x_min_cm: int64`
> - `x_max_cm: int64`
> - `y_min_cm: int64`
> - `y_max_cm: int64`
> - `t_min_ms: int64`
> - `t_max_ms: int64`
>
> Additional 4D bounds (only present in `dims=4` files):
>
> - `z_min_cm: int64`
> - `z_max_cm: int64`
>
> ### 5.4 Trajectory Dictionary Table (`dict/trajectories.parquet`)
>
> This table provides mapping from `traj_id` to the source trajectory file and statistics, for auditing and grouped experiments:
>
> - `traj_id: int64`
> - `user_id: int32`
> - `traj_src: string`
> - `n_points_all: int32` (number of points after §2.5 filtering)
> - `t_start_ms: int64` (minimum `t_ms` among points in the trajectory)
> - `t_end_ms: int64` (maximum `t_ms` among points in the trajectory)
>
> ### 5.5 `manifest.json` Content
>
> `manifest.json` records immutable build parameters and statistics as a JSON object, including:
>
> - `dataset_name`: `"GeoLife-Spatial-Join"`
> - `geolife_version`: `"1.3"`
> - `build_time_utc`: ISO 8601 time
> - `input_root`: input root directory string
> - `crs_input`: `"EPSG:4326"`
> - `crs_output`: `"EPSG:3857"`
> - `units`: `{ "xy": "cm", "z": "cm", "t": "ms" }`
> - `levels`: an array, with each entry containing the numeric values of $\Delta d,\Delta t,r_d,r_t$
> - `altitude_validity`: `{ "invalid_flag_ft": -777, "range_m": [-500, 10000] }`
> - `counts`: for each of the six groups, the number of rows, users, and trajectories
> - `files`: for each group, the list of Parquet files and the row count of each file
> - `hashing`: `{ "algo": "xxhash64", "seed": 0, "endianness": "little" }`
>
> ------
>
> ## 6. Build Validation and Consistency Requirements
>
> ### 6.1 Schema Validation
>
> For each group:
>
> - In `dims=3` files, `z_min_cm` and `z_max_cm` do not exist
> - In `dims=4` files, both `z_min_cm` and `z_max_cm` exist and contain no null values
> - All column types match §5.3
>
> ### 6.2 Boundary Consistency Validation
>
> For each group, sample records to validate the following identities:
>
> - `x_min_cm <= x_max_cm`, `y_min_cm <= y_max_cm`, `t_min_ms <= t_max_ms`
> - `x_max_cm - x_min_cm == 2 * r_d_cm`
> - `y_max_cm - y_min_cm == 2 * r_d_cm`
> - `t_max_ms - t_min_ms == 2 * r_t_ms`
> - For `dims=4`: `z_max_cm - z_min_cm == 2 * r_d_cm`
> - For `dims=4`: all records satisfy the valid-altitude conditions in §3.4
>
> ### 6.3 Sampled Validation of Intersection Semantics
>
> For each group, run one sampled join validation:
>
> 1. Fix random seed to 0, uniformly sample 10,000 rectangle records from the group;
> 2. Compute the set of intersecting pairs using a naive $O(n^2)$ algorithm (closed-interval intersection);
> 3. Compare set equality between the computed intersecting pair set and the intersecting pair set output by the join implementation under test (ignore output order).
>
> ### 6.4 Compliance Requirements
>
> The build script and data artifacts must comply with GeoLife 1.3 data usage license constraints. Released content includes the build program, configuration, and documentation of the `manifest.json` structure. Data files are generated and stored locally by users based on GeoLife 1.3.

> 
