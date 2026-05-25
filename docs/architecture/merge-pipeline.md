# Merge Pipeline

lazymerge uses [Cubed](https://cubed-dev.github.io/cubed/) to lazily merge
geospatial Zarr arrays. The merge is expressed as a `cubed.map_blocks`
operation, meaning that each output chunk is computed independently and only
the source data needed for that chunk is read.

## Overview

The pipeline has three phases:

1. **Target grid construction** -- define the output array shape, CRS,
   resolution, and chunk layout.
2. **Per-chunk source discovery** -- for each output chunk, find which
   source arrays intersect its spatial extent.
3. **Per-chunk warp and composite** -- read the intersecting source regions,
   reproject them to the target grid, and composite them into the output
   chunk.

## Target grid

`merge()` creates a target grid from the user-provided CRS, bounding box,
and resolution. This produces a `SpatialAttrs` (affine transform, shape,
bbox) and a `ProjAttrs` (CRS code). A lazy Cubed array filled with NaN is
allocated at the target shape and chunk size.

When temporal grouping or multi-band output is requested, extra dimensions
are prepended to the target array: `(time, band, y, x)`, `(time, y, x)`,
or `(band, y, x)`.

## Source discovery

For each output chunk, the pipeline computes the chunk's bounding box in the
target CRS and finds intersecting sources using one of two strategies:

**ScanIndex** -- a pre-built in-memory index created by `scan_store()`. Each
source entry stores its spatial extent and CRS, and bounding box intersection
is checked after reprojecting to the target CRS.

**DataFusion** -- a SQL query against the `/meta` group in the Zarr store
(via `zarr-datafusion-search`). The chunk bbox is reprojected to EPSG:4326
and used in a `ST_Intersects` spatial filter. This path supports
`sortby` (controlling composite priority) and `sql_filter` (arbitrary
attribute predicates).

When temporal grouping is active, an additional SQL filter restricts sources
to the time window for the current time step.

## Overview selection

When a source has a multiscale pyramid (indicated by a `multiscales`
attribute on its Zarr group), the pipeline selects the coarsest overview
level whose pixel size does not exceed the target resolution. This avoids
reading full-resolution data when the output is at a coarser scale.

The target resolution is converted to the source CRS before comparison,
handling cases where the target and source use different CRS with different
units.

## Warp and composite

For each source intersecting a chunk:

1. **Pixel mapping** -- an affine + CRS transform maps every target pixel
   center to fractional source pixel coordinates. For cross-CRS merges,
   `pyproj.Transformer` handles the reprojection.

2. **Region read** -- the bounding pixel range in the source is computed
   from the mapped coordinates. Only this subregion is read from the Zarr
   array.

3. **Resampling** -- source pixels are sampled at the mapped coordinates
   using nearest-neighbor interpolation.

4. **NaN compositing** -- the warped source data fills NaN pixels in the
   output. Sources are processed in order (scan index order, or `sortby`
   order for DataFusion), and once all pixels in a chunk are filled, remaining
   sources are skipped (early stop optimization).

## Execution model

Because the merge is a `cubed.map_blocks` call, Cubed controls scheduling.
Each chunk is an independent task that can be executed in parallel. The
library reads source data via `zarr-python`, which supports local
filesystems, cloud object stores, and Icechunk repositories.
