from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import zarr

from lazymerge.conventions import ProjAttrs, SpatialAttrs, write_proj, write_spatial

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

    from lazymerge.sources import SourceEntry

logger = logging.getLogger(__name__)

_S3_HTTPS_RE = re.compile(r"^https://(.+?)\.s3[.\-][\w-]*\.amazonaws\.com/(.+)$")


def _https_to_s3(url: str) -> str:
    """Convert an HTTPS S3 URL to its ``s3://`` protocol equivalent.

    Handles the virtual-hosted style used by AWS
    (``https://<bucket>.s3.<region>.amazonaws.com/<key>``).
    Non-matching URLs are returned unchanged.
    """
    m = _S3_HTTPS_RE.match(url)
    if m:
        return f"s3://{m.group(1)}/{m.group(2)}"
    return url


def _write_multiscales(overviews: int, attrs: MutableMapping[str, Any]) -> None:
    """Write multiscales convention attribute for a band group."""
    layout = []
    factor = 2
    for overview in range(overviews):
        if overview == 0:
            config: dict[str, Any] = {"asset": str(overview)}
        else:
            scale = float(factor**overview)
            config = {
                "asset": str(overview),
                "derived_from": str(overview - 1),
                "factors": [factor, factor],
                "transform": {
                    "scale": [scale, scale],
                    "translation": [0.0, 0.0],
                },
            }
        layout.append(config)
    attrs["multiscales"] = {"layout": layout}


def default_virtualizer(
    registry: Any,
    overviews: list[int] | None = None,
) -> Callable[[str, SourceEntry, Any, list[str]], None]:
    """Create a virtualizer callback that uses VirtualTIFF to create virtual zarr references.

    The callback virtualizes COG assets into an Icechunk store using VirtualiZarr,
    then writes zarr conventions (spatial, proj, multiscales) so that merge() can
    discover and read the virtualized arrays.

    Args:
        registry: An ObjectStoreRegistry for resolving COG URLs to object stores.
        overviews: Optional list of overview level indices to also virtualize
            (e.g. [1, 2, 3] for the first three overview IFDs). Level 0 (full
            resolution) is always included.

    Returns:
        A callback compatible with merge(virtualize=...).

    """
    from rasterio.warp import transform_bounds  # noqa: PLC0415
    from shapely import wkb  # noqa: PLC0415
    from virtual_tiff import VirtualTIFF  # noqa: PLC0415
    from virtualizarr import open_virtual_dataset  # noqa: PLC0415

    levels = [0]
    if overviews is not None:
        levels.extend(overviews)
    num_levels = max(levels) + 1 if levels else 1

    def _virtualize(
        source_id: str,
        entry: SourceEntry,
        store: Any,
        bands: list[str],
    ) -> None:
        if entry.metadata is None:
            msg = f"Cannot virtualize {source_id}: SourceEntry has no metadata"
            raise ValueError(msg)

        zarr_store: Any = store
        if hasattr(store, "prefix") and not isinstance(store, zarr.abc.store.Store):  # type: ignore[attr-defined]
            zarr_store = store.prefix

        # Idempotency: check if already virtualized by looking for the first band
        first_band = bands[0] if bands else None
        if first_band is not None:
            try:
                check_group = zarr.open_group(
                    zarr_store,
                    path=f"/{source_id}/{first_band}/0",
                    mode="r",
                )
                if any(isinstance(m, zarr.Array) for _, m in check_group.members()):
                    return
            except (KeyError, FileNotFoundError):
                pass

        # Step 1: Virtualize COGs into the store
        for band in bands:
            asset_key = f"asset_{band}"
            href = entry.metadata.get(asset_key)
            if href is None:
                msg = f"No asset column '{asset_key}' in metadata for {source_id}"
                raise KeyError(msg)

            href = _https_to_s3(href)

            for level in levels:
                group_path = f"{source_id}/{band}/{level}"
                ds = open_virtual_dataset(
                    url=href,
                    registry=registry,
                    parser=VirtualTIFF(ifd=level),
                )
                ds.vz.to_icechunk(store=store, group=group_path)

        # Step 2: Write conventions (spatial, proj, multiscales)
        meta = entry.metadata
        epsg = meta.get("proj:epsg")
        if epsg is None:
            msg = f"No proj:epsg in metadata for {source_id}"
            raise KeyError(msg)

        proj_attrs = ProjAttrs(code=f"EPSG:{epsg}")

        # Compute projected bbox from geometry
        geom = wkb.loads(meta["bbox"])
        bbox_4326 = [*geom.bounds]
        projected_bbox = transform_bounds("EPSG:4326", proj_attrs.code, *bbox_4326)

        # Determine base shape and transform
        if "shape_y" in meta and meta["shape_y"] is not None:
            base_shape: tuple[int, int] = (int(meta["shape_y"]), int(meta["shape_x"]))
        else:
            # Read shape from the virtualized array
            level_group = zarr.open_group(
                zarr_store,
                path=f"/{source_id}/{bands[0]}/0",
                mode="r",
            )
            for _name, member in level_group.members():
                if isinstance(member, zarr.Array):
                    base_shape = (member.shape[0], member.shape[1])
                    break

        if "transform_0" in meta and meta["transform_0"] is not None:
            base_transform = (
                float(meta["transform_0"]),
                float(meta["transform_1"]),
                float(meta["transform_2"]),
                float(meta["transform_3"]),
                float(meta["transform_4"]),
                float(meta["transform_5"]),
            )
        else:
            minx, miny, maxx, maxy = projected_bbox
            height, width = base_shape
            res_x = (maxx - minx) / width
            res_y = -(maxy - miny) / height
            base_transform = (res_x, 0.0, minx, 0.0, res_y, maxy)

        # Write spatial+proj on the item group
        item_group = zarr.open_group(zarr_store, path=f"/{source_id}", zarr_format=3)
        write_spatial(
            item_group,
            SpatialAttrs(
                dimensions=["y", "x"],
                transform=base_transform,
                bbox=projected_bbox,
                shape=base_shape,
            ),
        )
        write_proj(item_group, proj_attrs)

        for band in bands:
            # Write multiscales convention on each band group
            band_group = zarr.open_group(
                zarr_store,
                path=f"/{source_id}/{band}",
                zarr_format=3,
            )
            _write_multiscales(overviews=num_levels, attrs=band_group.attrs)

            # Write spatial+proj on each overview array
            for level in levels:
                factor = 2**level
                ovr_transform = (
                    base_transform[0] * factor,
                    base_transform[1],
                    base_transform[2],
                    base_transform[3],
                    base_transform[4] * factor,
                    base_transform[5],
                )

                level_group = zarr.open_group(
                    zarr_store,
                    path=f"/{source_id}/{band}/{level}",
                    zarr_format=3,
                )
                for _name, member in level_group.members():
                    if isinstance(member, zarr.Array):
                        ovr_shape = member.shape
                        write_spatial(
                            member,
                            SpatialAttrs(
                                dimensions=["y", "x"],
                                transform=ovr_transform,
                                bbox=projected_bbox,
                                shape=ovr_shape,
                            ),
                        )
                        write_proj(member, proj_attrs)
                        break

        logger.info("Virtualized %s (%d bands, %d levels)", source_id, len(bands), len(levels))

    return _virtualize
