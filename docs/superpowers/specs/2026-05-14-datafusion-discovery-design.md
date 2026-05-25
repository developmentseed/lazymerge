# DataFusion Source Discovery + Overview Selection

Extend lazymerge's source discovery to support zarr-datafusion-search metadata as an alternative to convention scanning, and add automatic overview selection using the zarr multiscales convention.

## Goals

- Use zarr-datafusion-search `/meta` group metadata (bbox in EPSG:4326) for scalable per-chunk source discovery via DataFusion SQL queries.
- Fall back to existing convention-based scanning when the store has no DataFusion metadata.
- Automatically select the best overview level from zarr multiscales convention to match the target resolution, avoiding unnecessary data reads.
- Keep the `ScanIndex` path unchanged for convention-only stores.

## Non-Goals

- SQL-based chunk-level filtering (Pass 2 always uses convention attrs).
- Generating multiscale pyramids.
- Supporting non-zarr metadata formats.

## Dependencies

Existing:
- `zarr` (v3), `cubed`, `pyproj`, `numpy`, `zarr-cm`

New (optional):
- `zarr-datafusion-search` — only required when `datafusion=True`

## Data Model

### Store layout with DataFusion metadata

```
store/
  meta/                          # DataFusion columnar metadata
    id            (StringDType)  # row ID = source group path
    bbox          (object/WKB)   # spatial extent in EPSG:4326
    proj:epsg     (int64)        # EPSG code
    transform_0..5 (float64)     # affine transform coefficients
    shape_x, shape_y (int64)     # array dimensions
    ...
  LC08_L2SP_044034_20251010_02_T1/   # source group (ID from meta row)
    red/                              # band group
      multiscales: { layout: [...] } # multiscales convention attr
      0/                             # finest resolution array (spatial:, proj: attrs)
      1/                             # overview level 1
      2/                             # overview level 2
    green/
      ...
```

**Key convention:** bbox values in `/meta` are always stored in EPSG:4326.

### Store layout without DataFusion metadata (current)

```
store/
  utm18n_tile_a   (zarr.Array with spatial:, proj: attrs)
  utm18n_tile_c   (zarr.Array with spatial:, proj: attrs)
  ...
```

## Architecture

### Two discovery paths in `merge()`

The `datafusion` parameter on `merge()` selects the discovery strategy:

- `datafusion=False` (default): Requires a pre-built `ScanIndex` (from `scan_store()`). Source discovery uses in-memory bbox intersection. Current behavior.
- `datafusion=True`: No `ScanIndex` needed (`source_index=None`). Each target chunk issues a DataFusion SQL query against the `/meta` group to find intersecting sources by bbox. Scales to large stores because only matching sources are returned per chunk.

```python
# Convention path — user builds index, passes it in
index = scan_store(root)
merge(source_index=index, ..., band=None)

# DataFusion path — merge handles discovery internally per-chunk
merge(..., band="red", datafusion=True)
```

### DataFusion per-chunk query flow

```
Target chunk bbox (in target CRS)
  |
  v
Reproject bbox to EPSG:4326 via pyproj
  |
  v
DataFusion SQL: SELECT id, proj:epsg, transform_0..5, shape_x, shape_y
                FROM meta
                WHERE ST_Intersects(bbox, <query_bbox_wkb>)
  |
  v
For each matching row:
  Build SourceEntry(path=id, spatial_attrs, proj_attrs)
  Navigate into source_group[band], select overview
  Pass 2: chunk-level filtering via convention attrs
  Read + warp matching chunks
```

### Overview selection flow (unchanged from v1)

After source discovery, before chunk filtering and data reading, select the best overview:

```
Source group (e.g. LC08_L2SP_044034_20251010_02_T1/red)
  |
  v
Read multiscales attr from band group
  |
  v
If multiscales present:
  Parse layout levels (asset paths + scale factors)
  Compute each level's pixel size from base spatial:transform * cumulative scale
  select_overview(): pick coarsest level with pixel size <= target resolution
  |
If no multiscales:
  Use the single available array
  |
  v
Selected array path + its spatial:/proj: attrs used for Pass 2
```

### `select_overview` logic (adapted from lazycogs)

```python
def select_overview(
    overviews: list[OverviewLevel],
    target_res: float,
    native_res: float,
) -> OverviewLevel | None:
    """Choose the coarsest overview whose resolution is <= target_res."""
    if not overviews:
        return None
    if target_res <= native_res:
        return None

    selected = None
    for overview in overviews:
        if overview.resolution <= target_res:
            selected = overview
        else:
            break
    return selected
```

## Module Changes

### `conventions.py` (already implemented)

- `OverviewLevel` dataclass: `path`, `scale`, `resolution`
- `read_multiscales(group)`: parses zarr multiscales convention, returns overview levels

### `sources.py`

Keep: `scan_store(root)` for convention scanning (remove the `backend` parameter — it's no longer needed).

Keep: `select_overview()`.

Remove: `_scan_datafusion()`, `scan_store(backend="datafusion")`. DataFusion discovery moves into `merge.py`.

Add: `query_datafusion_sources(root, bbox_4326)` — issues a DataFusion SQL query against the `/meta` group and returns matching `SourceEntry` objects. This is the per-chunk query function called from `_merge_block`.

```python
def query_datafusion_sources(
    root: zarr.Group,
    bbox_4326: tuple[float, float, float, float],
) -> list[SourceEntry]:
    """Query the /meta group via DataFusion for sources intersecting bbox_4326."""
```

### `merge.py`

Update `merge()` signature:

```python
def merge(
    source_index: ScanIndex | None,
    target: cubed.Array,
    target_spatial: SpatialAttrs,
    target_proj: ProjAttrs,
    store: Store,
    resampling: str = "nearest",
    band: str | None = None,
    datafusion: bool = False,
) -> tuple[cubed.Array, SpatialAttrs, ProjAttrs]:
```

Update `_merge_block`:
- When `datafusion=True`: reproject chunk bbox to 4326, call `query_datafusion_sources()`, then proceed with band navigation + overview selection + chunk filtering + warping.
- When `datafusion=False`: use `source_index.find_intersecting_sources()` as before.

### `__init__.py`

Export: `OverviewLevel`, `read_multiscales`, `select_overview`, `query_datafusion_sources`.

## What stays the same

- `ScanIndex` class interface for convention stores.
- `warp.py`: no changes.
- `target.py`: no changes.
- Convention-based chunk filtering (Pass 2) always uses `spatial:`/`proj:` attrs from the selected array.
- Overview selection logic.

## Resolved Questions

1. **Band selection**: `merge()` takes a `band` parameter (string) specifying which band group to read from within each source group (e.g. `band="red"`). Future work: support a band dimension on the target array so multiple bands merge in one pass, following the lazycogs approach.

2. **Chunk shape from DataFusion**: Not needed upfront. When `datafusion=True`, the actual array is opened per-chunk and chunk_shape is read directly from it.

3. **Bbox CRS in /meta**: By convention, bbox values in the `/meta` group are always stored in EPSG:4326. The query bbox is reprojected to 4326 before querying.
