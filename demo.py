# demo.py
"""End-to-end demo of lazymerge with synthetic data."""

import zarr

from lazymerge.conventions import (
    ProjAttrs,
    SpatialAttrs,
    read_proj,
    read_spatial,
    write_proj,
    write_spatial,
)
from lazymerge.merge import merge
from lazymerge.sources import scan_store
from lazymerge.target import to_zarr


def make_synthetic_store() -> tuple[zarr.storage.MemoryStore, zarr.Group]:
    """Create 4 synthetic source arrays: 2 in UTM 18N, 2 in UTM 17N."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    sources = [
        # UTM 18N tiles (left side)
        {
            "name": "utm18n_tile_a",
            "crs": "EPSG:32618",
            "bbox": (500000.0, 5990000.0, 505000.0, 5995000.0),
            "resolution": 10.0,
            "fill": 100.0,
        },
        {
            "name": "utm18n_tile_c",
            "crs": "EPSG:32618",
            "bbox": (500000.0, 5995000.0, 505000.0, 6000000.0),
            "resolution": 10.0,
            "fill": 300.0,
        },
        # UTM 17N tiles (right side — same geographic area as UTM 18N tiles would
        # cover at x=[503000,508000], but expressed in UTM zone 17N coordinates)
        {
            "name": "utm17n_tile_b",
            "crs": "EPSG:32617",
            "bbox": (895094.0, 6006920.0, 900511.0, 6012337.0),
            "resolution": 10.0,
            "fill": 200.0,
        },
        {
            "name": "utm17n_tile_d",
            "crs": "EPSG:32617",
            "bbox": (894669.0, 6011912.0, 900086.0, 6017329.0),
            "resolution": 10.0,
            "fill": 400.0,
        },
    ]

    for src in sources:
        xmin, ymin, xmax, ymax = src["bbox"]
        res = src["resolution"]
        width = int((xmax - xmin) / res)
        height = int((ymax - ymin) / res)

        arr = root.create_array(src["name"], shape=(height, width), dtype="f4", chunks=(256, 256))
        arr[:] = src["fill"]

        write_spatial(
            arr,
            SpatialAttrs(
                dimensions=["y", "x"],
                transform=(res, 0.0, xmin, 0.0, -res, ymax),
                bbox=(xmin, ymin, xmax, ymax),
                shape=(height, width),
            ),
        )
        write_proj(arr, ProjAttrs(code=src["crs"]))

    return store, root


def main() -> None:

    # Step 1: Create synthetic sources
    store, root = make_synthetic_store()
    for _name, item in root.members():
        if isinstance(item, zarr.Array):
            pass

    # Step 2: Build source index
    index = scan_store(root)
    for _entry in index.entries:
        pass

    # Step 3: Merge (UTM 17N tiles will be reprojected into UTM 18N)
    result_arr, result_spatial, result_proj = merge(
        store=store,
        crs="EPSG:32618",
        bbox=(500000.0, 5990000.0, 508000.0, 6000000.0),
        resolution=10.0,
        chunk_size=(256, 256),
        source_index=index,
    )

    # Step 5: Compute and verify
    data = result_arr.compute()

    # Step 6: Materialize to Zarr
    output_store = zarr.storage.MemoryStore()
    to_zarr(result_arr, result_spatial, result_proj, output_store, path="mosaic")

    # Step 7: Read back and verify
    out_root = zarr.open_group(output_store, mode="r")
    out_arr = out_root["mosaic"]
    read_spatial(out_arr)
    read_proj(out_arr)

    # Show zarr_conventions metadata written by zarr-cm
    conventions = out_arr.attrs.get("zarr_conventions", [])
    if conventions:
        for _conv in conventions:
            pass

    # Spot-check: pixel at (0, 0) is top-left of target = (500000, 6000000) in UTM 18N
    # This should be in tile_c (fill=300, EPSG:32618)
    val_top_left = data[0, 0]

    # Pixel at (995, 795) is near bottom-right in UTM 18N (a few pixels from edge
    # to account for reprojection boundary effects from UTM 17N -> 18N)
    # This should be in tile_b (fill=200, EPSG:32617 — reprojected into target)
    val_bottom_right = data[995, 795]

    assert val_top_left == 300.0, f"Expected 300.0, got {val_top_left}"
    assert val_bottom_right == 200.0, f"Expected 200.0, got {val_bottom_right}"



if __name__ == "__main__":
    main()
