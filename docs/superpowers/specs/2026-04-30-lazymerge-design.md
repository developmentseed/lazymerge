# lazymerge Design Spec

Pure Zarr + cubed array mosaic library. No GDAL, no xarray. Inspired by [lazycogs](https://developmentseed.org/lazycogs/) but built entirely on Zarr v3 conventions and cubed for lazy computation.

## Goals

- Demonstrate that Zarr spatial conventions (`spatial:`, `proj:`, `multiscales:`) are sufficient to describe, index, and mosaic geospatial arrays without format-specific readers.
- Provide a lazy cubed array that users can slice and compute on demand, with optional materialization to a convention-compliant Zarr store.
- Keep the warp module dependency-free (relative to lazymerge internals) so it can be extracted as a standalone reprojection library.
- Start with local/synthetic data and single-machine cubed execution. Architect so cloud stores and serverless executors are natural extensions.

## Non-Goals

- xarray integration (no xarray dependency).
- Reading COGs or any non-Zarr format (that's lazycogs' job).
- Multiscale pyramid generation in v1 (single resolution target only).
- Serverless execution (local cubed executor only for now).

## Dependencies

- `zarr` (v3) — array storage and metadata
- `cubed` — lazy chunked array computation and `map_blocks`
- `pyproj` — CRS transforms
- `numpy` — array operations and resampling

Optional / future:
- `zarr-datafusion-search` — large-store spatial indexing

## Architecture

```
lazymerge/
  conventions/    # Zarr convention metadata read/write
  sources/        # Spatial index of source arrays
  target/         # Target array creation and materialization
  warp/           # Standalone pyproj+numpy reprojection
  merge/          # cubed map_blocks orchestrator
```

Data flow:

```
Source Zarr arrays (with spatial:/proj: attrs)
  |
  v
Source Index (scan store or datafusion query)
  |
  v
merge() — cubed.map_blocks over target chunks:
  |  1. chunk_bbox() from target convention attrs
  |  2. find_intersecting_sources() — pass 1 filter
  |  3. find_intersecting_chunks() — pass 2 filter
  |  4. read source chunks from Zarr store
  |  5. warp_chunk() — reproject each to target grid
  |  6. aggregate overlapping results
  |
  v
Lazy cubed.Array (with spatial:/proj: attrs)
  |
  v (optional)
to_zarr() — materialize to Zarr v3 store with conventions
```

---

## Module 1: conventions/

Read and write Zarr v3 metadata following the [spatial](https://github.com/zarr-conventions/spatial), [geo:proj](https://github.com/zarr-conventions/geo-proj), and [multiscales](https://github.com/zarr-conventions/multiscales) conventions.

### Data Structures

**`SpatialAttrs`** — dataclass:
- `dimensions: list[str]` — e.g., `["y", "x"]`
- `transform: tuple[float, ...]` — affine coefficients `(a, b, c, d, e, f)` in rasterio/Affine order
- `bbox: tuple[float, ...]` — `(xmin, ymin, xmax, ymax)` in coordinate space
- `shape: tuple[int, ...]` — spatial dimension sizes `(height, width)`
- `registration: str` — `"pixel"` (default) or `"node"`

**`ProjAttrs`** — dataclass (at least one field required):
- `code: str | None` — e.g., `"EPSG:32618"`
- `wkt2: str | None`
- `projjson: dict | None`

**`MultiscalesAttrs`** — dataclass:
- `datasets: list[MultiscaleDataset]` — each with `path`, `spatial_attrs`
- Reserved for future use. Not implemented in v1.

### Functions

```python
def read_spatial(node: zarr.Group | zarr.Array) -> SpatialAttrs
def read_proj(node: zarr.Group | zarr.Array) -> ProjAttrs
def write_spatial(node: zarr.Group | zarr.Array, attrs: SpatialAttrs) -> None
def write_proj(node: zarr.Group | zarr.Array, attrs: ProjAttrs) -> None

def chunk_bbox(
    spatial_attrs: SpatialAttrs,
    chunk_coords: tuple[int, ...],
    chunk_shape: tuple[int, ...],
) -> tuple[float, float, float, float]
```

`chunk_bbox` computes the spatial bounding box of a specific chunk by applying the affine transform to the chunk's pixel offset and size. This is the key primitive for two-pass spatial filtering.

### Design Decisions

- Plain dataclasses + functions. No Zarr subclasses or plugins.
- Works with any `zarr.Group` or `zarr.Array` from zarr-python v3.
- Serialization uses the convention namespace prefixes (`spatial:`, `proj:`) directly in `.attrs`.

---

## Module 2: sources/

Builds a spatial index of available source arrays from their convention metadata.

### Data Structures

**`SourceEntry`** — dataclass:
- `path: str` — path within the Zarr store
- `spatial_attrs: SpatialAttrs`
- `proj_attrs: ProjAttrs`
- `chunk_shape: tuple[int, ...]` — the source array's chunk grid dimensions

**`SourceIndex`** — protocol:
```python
class SourceIndex(Protocol):
    def find_intersecting_sources(
        self, target_bbox: tuple[float, float, float, float], target_crs: str
    ) -> list[SourceEntry]: ...

    def find_intersecting_chunks(
        self, source: SourceEntry, target_chunk_bbox: tuple[float, float, float, float], target_crs: str
    ) -> list[tuple[str, tuple[int, ...]]]: ...
```

### Implementations

**`ScanIndex`** — small store path:
- `scan_store(root: zarr.Group) -> ScanIndex` — walks all child groups/arrays, reads `SpatialAttrs` and `ProjAttrs`, stores results as an in-memory list of `SourceEntry`.
- `find_intersecting_sources()` — reprojects each source's `spatial:bbox` to the target CRS via `pyproj.Transformer`, tests intersection. Caches reprojected bboxes.
- `find_intersecting_chunks()` — uses `chunk_bbox()` to enumerate source chunks, tests each against the target chunk bbox (in target CRS).

**`DataFusionIndex`** — large store path (future):
- Adapter over `zarr-datafusion-search`. Same `SourceIndex` protocol.
- Queries the 1D metadata arrays (including WKT bounding box arrays) via DataFusion SQL.
- Not implemented in v1. Defined as a protocol extension point.

### CRS Handling

All bbox comparisons reproject to the target CRS using `pyproj.Transformer`. Source bboxes are cached after first reprojection to avoid redundant transforms.

---

## Module 3: target/

Creates the user-facing lazy cubed array and handles materialization.

### Functions

```python
def create_target(
    crs: str,
    bbox: tuple[float, float, float, float],
    resolution: float,
    chunk_size: tuple[int, int] = (512, 512),
    dtype: str = "float32",
    fill_value: float = float("nan"),
) -> cubed.Array:
```

Returns a plain `cubed.Array` with:
- `shape` computed from `bbox` and `resolution`: `(ceil((ymax-ymin)/resolution), ceil((xmax-xmin)/resolution))`
- `chunks` set to `chunk_size`
- `dtype` and `fill_value` as specified
- Convention attributes stamped directly on the array's `.attrs`:
  - `spatial:dimensions`, `spatial:transform`, `spatial:bbox`, `spatial:shape`, `spatial:registration`
  - `proj:code` (or `proj:wkt2` / `proj:projjson`)

The affine transform is derived from bbox + resolution:
- `a = resolution` (pixel width)
- `b = 0` (no rotation)
- `c = xmin` (x origin)
- `d = 0` (no rotation)
- `e = -resolution` (pixel height, negative for north-up)
- `f = ymax` (y origin)

```python
def to_zarr(
    array: cubed.Array,
    store: str | zarr.Store,
    path: str | None = None,
) -> None:
```

Computes the cubed graph and writes the result to a Zarr v3 store. Copies convention attributes from the cubed array's `.attrs` onto the output Zarr array metadata using `write_spatial()` and `write_proj()`.

---

## Module 4: warp/

Standalone reprojection using pyproj + numpy. Zero imports from the rest of lazymerge. Designed for eventual extraction as an independent library.

### Function

```python
def warp_chunk(
    source_data: np.ndarray,
    source_transform: tuple[float, ...],
    source_crs: str,
    target_transform: tuple[float, ...],
    target_crs: str,
    target_shape: tuple[int, int],
    resampling: str = "nearest",
) -> np.ndarray:
```

### Algorithm

Note: `target_transform` and `target_shape` describe the specific target *chunk*, not the full target array. The caller (the merge block function) computes the chunk-local transform by adjusting the array-level transform's origin to the chunk's pixel offset.

1. **Build target pixel grid:** Generate a 2D grid of (row, col) indices for `target_shape`. Apply `target_transform` to get (x, y) coordinates in the target CRS for every target pixel.

2. **Reproject to source CRS:** Use `pyproj.Transformer.from_crs(target_crs, source_crs, always_xy=True)` to transform the target (x, y) coordinates into source CRS coordinates.

3. **Map to source pixel indices:** Apply the inverse of `source_transform` to convert source CRS coordinates to fractional (row, col) indices in the source array.

4. **Sample source data:**
   - `"nearest"`: Round to nearest integer indices. Mask out-of-bounds. Index into `source_data`.
   - `"bilinear"` (future): Interpolate from 4 nearest pixels using fractional indices.

5. **Return:** Array of shape `target_shape` with reprojected values. Out-of-bounds pixels are NaN.

### Design Decisions

- Input/output are plain numpy arrays and scalar parameters. No Zarr, no cubed, no dataclass imports.
- `pyproj.Transformer` is created per call. Caching transformers across calls is a future optimization.
- Only `"nearest"` resampling in v1.

---

## Module 5: merge/

The cubed `map_blocks` orchestrator that ties everything together.

### Function

```python
def merge(
    source_index: SourceIndex,
    target: cubed.Array,
    store: zarr.Store,
    resampling: str = "nearest",
    aggregation: str = "first",
) -> cubed.Array:
```

`store` is the Zarr store containing the source arrays (needed to read chunk data inside the block function).

Returns a new `cubed.Array` with the same shape, chunks, dtype, and convention attrs as `target`, but backed by a `map_blocks` computation graph.

### Block Function

The function passed to `cubed.map_blocks` receives each target chunk's position via `block_info` and executes:

1. **Compute target chunk bbox:** Use `chunk_bbox()` with the target's `spatial:transform` and the chunk's offset/shape from `block_info`.

2. **Pass 1 — find sources:** `source_index.find_intersecting_sources(chunk_bbox, target_crs)` returns source arrays whose bbox overlaps this chunk.

3. **Pass 2 — find source chunks:** For each matching source, `source_index.find_intersecting_chunks(source, chunk_bbox, target_crs)` returns specific chunk coordinates.

4. **Read source chunks:** Open the source array from `store` at the source entry's path, read the relevant chunks as numpy arrays.

5. **Warp:** Call `warp_chunk()` for each source chunk, reprojecting into the target chunk's grid (using the target chunk's transform and shape).

6. **Aggregate:** Combine warped results from all sources:
   - `"first"`: Iterate sources in order. For each target pixel, use the first non-NaN value.
   - `"last"`: Same, but last non-NaN wins.
   - `"mean"` / `"median"`: Future. Stack all warped arrays and reduce.

7. **Return:** The filled target chunk as a numpy array.

### Design Decisions

- The source index and store reference are passed as kwargs to `map_blocks`, making them available inside the block function.
- Aggregation is applied per-chunk, not globally. This means "first" ordering is per-chunk-consistent but depends on source index iteration order.
- Reading source data happens inside the block function (lazy). No data is read until cubed executes the graph.

---

## Demo Script

A top-level script (`demo.py`) exercising the full pipeline with synthetic data.

### Setup

1. **Generate synthetic sources:** Create 4-6 Zarr arrays in a temporary store. Each array:
   - Has a different CRS (e.g., 2-3 UTM zones, maybe one in EPSG:4326)
   - Covers a different but overlapping spatial extent
   - Is filled with a recognizable pattern (e.g., linear gradient based on coordinates, or a unique constant per array)
   - Has `spatial:` and `proj:` convention attributes stamped via `write_spatial()` and `write_proj()`
   - Has realistic chunk sizes (e.g., 256x256)

2. **Build source index:** `scan_store()` over the synthetic store.

### Execute

3. **Create target:** `create_target()` with a CRS and extent that spans multiple source arrays. Resolution and chunk size chosen to produce a manageable output (e.g., ~2048x2048 pixels).

4. **Merge:** `merge(source_index, target, store)` returns the lazy cubed array.

5. **Compute subset:** Slice a few chunks and call `.compute()` to verify values.

6. **Materialize:** `to_zarr()` writes the full result.

### Verify

7. **Read back:** Open the output Zarr store, confirm `spatial:` and `proj:` attributes are present and correct.

8. **Spot-check values:** For known coordinates, verify that pixel values match the expected source data after reprojection. Use the gradient/constant patterns to validate that the right source was selected and warped correctly.

No visualization dependency. Verification is numeric assertions. Optional matplotlib plotting can be added for visual inspection but is not part of the core demo.
