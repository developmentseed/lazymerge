# Conventions

lazymerge uses a set of Zarr attribute conventions to describe the spatial
and projection metadata of source arrays. These conventions are managed by
the [`zarr-cm`](https://github.com/developmentseed/zarr-cm) library.

## Spatial attributes

Every source array (or its parent group) carries a `spatial:*` attribute set:

| Attribute | Type | Description |
|---|---|---|
| `spatial:dimensions` | `list[str]` | Dimension names, e.g. `["y", "x"]` |
| `spatial:transform` | `list[float]` | 6-element affine transform: `[a, b, c, d, e, f]` mapping `(col, row)` to `(x, y)` |
| `spatial:bbox` | `list[float]` | Bounding box: `[xmin, ymin, xmax, ymax]` |
| `spatial:shape` | `list[int]` | Array shape: `[height, width]` |
| `spatial:registration` | `str` | Pixel registration model, default `"pixel"` |

These attributes are read by `read_spatial()` and written by
`write_spatial()`.

## Projection attributes

CRS information is stored in `proj:*` attributes:

| Attribute | Type | Description |
|---|---|---|
| `proj:code` | `str \| null` | CRS authority code, e.g. `"EPSG:32618"` |
| `proj:wkt2` | `str \| null` | WKT2 CRS definition |
| `proj:projjson` | `object \| null` | PROJJSON CRS definition |

At least one of these must be present. `read_proj()` and `write_proj()`
handle serialization.

## Multiscale overviews

Source arrays may include a `multiscales` attribute on their parent group
describing an image pyramid:

```json
{
  "multiscales": {
    "layout": [
      {"asset": "0", "transform": {"scale": [1.0, 1.0], "translation": [0.0, 0.0]}},
      {"asset": "1", "derived_from": "0", "transform": {"scale": [2.0, 2.0], "translation": [0.5, 0.5]}},
      {"asset": "2", "derived_from": "1", "transform": {"scale": [2.0, 2.0], "translation": [0.5, 0.5]}}
    ]
  }
}
```

Each entry after the base level describes an overview with a cumulative scale
factor relative to the native resolution. `read_multiscales()` parses this
into a list of `OverviewLevel` objects. During merge, `select_overview()`
picks the coarsest level that avoids upsampling.

## DataFusion metadata group

When using `datafusion=True`, the store must contain a `/meta` group
compatible with
[zarr-datafusion-search](https://github.com/developmentseed/zarr-datafusion-search).
This group contains 1D columnar arrays that are queried via SQL. The
required columns are:

| Column | Type | Description |
|---|---|---|
| `id` | string | Source array path within the store |
| `proj:epsg` | int | EPSG code for the source CRS |
| `transform_0` -- `transform_5` | float | Affine transform components |
| `shape_x`, `shape_y` | int | Array dimensions |
| `bbox` | WKB geometry | Bounding box in EPSG:4326 |
| `datetime` | timestamp | Acquisition datetime (required for temporal grouping) |
