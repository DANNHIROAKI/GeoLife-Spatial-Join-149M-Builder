# GeoLife-Spatial-Join 一站式构建工程包（GeoLife 1.3）

本工程包实现你提供的 **GeoLife-Spatial-Join Building Spec（GeoLife 1.3）**：

- 自动创建 Python venv（.venv）
- 自动安装依赖（numpy/pyproj/pyarrow/xxhash/tqdm）
- 自动下载 GeoLife Trajectories 1.3 zip（默认 Microsoft Download Center 直链，可覆盖）
- 自动解压并定位 `Data/`
- 自动生成 `geolife_spatial_join/` 六组 Parquet + `dict/trajectories.parquet` + `manifest.json`
- 自动执行基础校验（schema + 边界恒等式）
- 可选执行 10k 抽样自连接 O(n^2) 相交对检查（更慢）

## 一键运行

```bash
chmod +x run.sh
./run.sh
```

## 常用环境变量

- **强制重建**（会删除输出目录）：
```bash
FORCE=1 ./run.sh
```

- **跳过校验**：
```bash
VALIDATE=0 ./run.sh
```

- **启用 10k O(n^2) 自连接检查**：
```bash
VALIDATE_JOIN=1 ./run.sh
```

- **覆盖下载 URL**（当微软直链变动时）：
```bash
GEOZIP_URL="https://..." ./run.sh
```

## 输出

默认输出目录：

```
geolife_spatial_join/
  manifest.json
  dims=3/level=1/part-00000.parquet ...
  dims=3/level=2/...
  dims=3/level=3/...
  dims=4/level=1/...
  dims=4/level=2/...
  dims=4/level=3/...
  dict/trajectories.parquet
```
