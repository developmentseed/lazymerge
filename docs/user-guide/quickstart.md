# Quickstart

This guide walks through merging multiple Zarr source arrays into a single
output grid.

## 1. Open a Zarr store and build a source index

lazymerge needs to know which source arrays exist and where they are
located in space. For a local Zarr store where each top-level array has
spatial and projection convention attributes, use `scan_store`:

```python
import zarr
from lazymerge import scan_store

root = zarr.open_group("my_data.zarr", mode="r")
index = scan_store(root)
```

## 2. Merge sources into a target grid

Define your output grid by CRS, bounding box, and resolution, then call
`merge`:

```python
from lazymerge import merge

result, spatial, proj, _ = merge(
    store=zarr.storage.LocalStore("my_data.zarr"),
    crs="EPSG:32618",
    bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
    resolution=10.0,
    chunk_size=(512, 512),
    source_index=index,
)
```

`result` is a lazy Cubed array. No data has been read yet.

## 3. Compute the result

```python
data = result.compute()
print(data.shape)  # (100, 200)
```

Only the source regions that intersect each output chunk are read during
compute.

## Using DataFusion for source discovery

For stores with a `/meta` group (see
[architecture](../architecture/conventions.md)), you can skip
the scan step and use SQL-based spatial filtering:

```python
from obstore.store import LocalStore

result, spatial, proj, _ = merge(
    store=LocalStore("my_data.zarr"),
    crs="EPSG:32618",
    bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
    resolution=10.0,
    datafusion=True,
    sortby="datetime",
)
```

## Multi-band merges

Request specific bands by name. A single band produces a 2D `(y, x)` output;
multiple bands produce a 3D `(band, y, x)` output:

```python
result, spatial, proj, _ = merge(
    store=store,
    crs="EPSG:32618",
    bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
    resolution=10.0,
    source_index=index,
    bands=["red", "green", "blue"],
)

data = result.compute()
print(data.shape)  # (3, 100, 200)
```

## Temporal grouping

Bucket source scenes by time period to produce a time dimension:

```python
result, spatial, proj, time_coords = merge(
    store=store,
    crs="EPSG:32618",
    bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
    resolution=10.0,
    datafusion=True,
    temporal_grouping="P1M",  # monthly composites
)

data = result.compute()
print(data.shape)       # (12, 100, 200)
print(time_coords)      # array of datetime64[D] values
```

Supported period strings: `P1D` (daily), `P1W` (weekly), `P1M` (monthly),
`P1Y` (yearly), `PnD` (fixed n-day windows), `PnW` (fixed n-week windows).

## Dry-run with explain

Inspect which source regions would be read without fetching any pixel data:

```python
from lazymerge import explain

plan = explain(
    store=store,
    crs="EPSG:32618",
    bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
    resolution=10.0,
    source_index=index,
)

print(plan.summary())
```
