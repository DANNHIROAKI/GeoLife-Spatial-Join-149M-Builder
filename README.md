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

> 
