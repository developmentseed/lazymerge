from __future__ import annotations

import logging
import math
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


def _best_overview_level(
    native_res: float,
    target_res: float,
    max_overviews: int,
) -> int:
    """Return the best COG overview IFD index for a given target resolution.

    Picks the coarsest overview whose resolution does not exceed the target,
    assuming a standard 2x pyramid.  Returns 0 when full resolution is best.
    """
    if target_res <= native_res:
        return 0
    level = math.floor(math.log2(target_res / native_res))
    return min(max(level, 0), max_overviews)


def _write_multiscales(levels: list[int], attrs: MutableMapping[str, Any]) -> None:
    """Write multiscales convention attribute for the given levels."""
    layout = []
    factor = 2
    for level in levels:
        if level == 0:
            config: dict[str, Any] = {"asset": str(level)}
        else:
            scale = float(factor**level)
            config = {
                "asset": str(level),
                "derived_from": "0",
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
    target_resolution: float | None = None,
    max_overviews: int = 6,
) -> Callable[[str, SourceEntry, Any, list[str]], None]:
    """Create a virtualizer callback that uses VirtualTIFF to create virtual zarr references.

    The callback virtualizes COG assets into an Icechunk store using VirtualiZarr,
    then writes zarr conventions (spatial, proj, multiscales) so that merge() can
    discover and read the virtualized arrays.

    When ``target_resolution`` is provided, the callback automatically selects the
    single best COG overview level for that resolution (assuming a standard 2x
    pyramid) and only virtualizes level 0 plus that overview.  This avoids
    virtualizing overview levels that will never be read.

    Parameters
    ----------
    registry
        An ObjectStoreRegistry for resolving COG URLs to object stores.
    target_resolution
        The target pixel size passed to ``merge()``.  When set, only the best
        overview for this resolution is virtualized (along with level 0).
        When ``None``, only level 0 is virtualized.
    max_overviews
        Maximum overview IFD index to consider.  Ignored when
        ``target_resolution`` is ``None``.

    Returns
    -------
    Callable
        A callback compatible with ``merge(virtualize=...)``.

    """
    from rasterio.warp import transform_bounds  # noqa: PLC0415
    from shapely import wkb  # noqa: PLC0415
    from virtual_tiff import VirtualTIFF  # noqa: PLC0415
    from virtualizarr import open_virtual_dataset  # noqa: PLC0415

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

        meta = entry.metadata

        # Determine base transform early so we can pick the best overview
        epsg = meta.get("proj:epsg")
        if epsg is None:
            msg = f"No proj:epsg in metadata for {source_id}"
            raise KeyError(msg)
        proj_attrs = ProjAttrs(code=f"EPSG:{epsg}")

        geom = wkb.loads(meta["bbox"])
        bbox_4326 = [*geom.bounds]
        projected_bbox = transform_bounds("EPSG:4326", proj_attrs.code, *bbox_4326)

        if "shape_y" in meta and meta["shape_y"] is not None:
            base_shape: tuple[int, int] = (int(meta["shape_y"]), int(meta["shape_x"]))
        else:
            base_shape = (0, 0)  # resolved after virtualizing level 0

        if "transform_0" in meta and meta["transform_0"] is not None:
            base_transform = (
                float(meta["transform_0"]),
                float(meta["transform_1"]),
                float(meta["transform_2"]),
                float(meta["transform_3"]),
                float(meta["transform_4"]),
                float(meta["transform_5"]),
            )
        elif base_shape != (0, 0):
            minx, miny, maxx, maxy = projected_bbox
            height, width = base_shape
            res_x = (maxx - minx) / width
            res_y = -(maxy - miny) / height
            base_transform = (res_x, 0.0, minx, 0.0, res_y, maxy)
        else:
            base_transform = None  # type: ignore[assignment]

        # Step 1: Always virtualize level 0
        for band in bands:
            asset_key = f"asset_{band}"
            href = meta.get(asset_key)
            if href is None:
                msg = f"No asset column '{asset_key}' in metadata for {source_id}"
                raise KeyError(msg)

            href = _https_to_s3(href)

            group_path = f"{source_id}/{band}/0"
            ds = open_virtual_dataset(
                url=href,
                registry=registry,
                parser=VirtualTIFF(ifd=0),
            )
            ds.vz.to_icechunk(store=store, group=group_path)

        # Resolve base_shape and base_transform from the virtualized array if needed
        if base_shape == (0, 0) or base_transform is None:
            level_group = zarr.open_group(
                zarr_store,
                path=f"/{source_id}/{bands[0]}/0",
                mode="r",
            )
            for _name, member in level_group.members():
                if isinstance(member, zarr.Array):
                    base_shape = (member.shape[0], member.shape[1])
                    break
            if base_transform is None:
                minx, miny, maxx, maxy = projected_bbox
                height, width = base_shape
                res_x = (maxx - minx) / width
                res_y = -(maxy - miny) / height
                base_transform = (res_x, 0.0, minx, 0.0, res_y, maxy)

        # Step 2: Pick and virtualize the best overview level
        native_res = abs(base_transform[0])
        levels: list[int] = [0]

        if target_resolution is not None and target_resolution > native_res:
            best = _best_overview_level(native_res, target_resolution, max_overviews)
            if best > 0:
                levels.append(best)
                for band in bands:
                    asset_key = f"asset_{band}"
                    href = _https_to_s3(meta[asset_key])

                    group_path = f"{source_id}/{band}/{best}"
                    ds = open_virtual_dataset(
                        url=href,
                        registry=registry,
                        parser=VirtualTIFF(ifd=best),
                    )
                    ds.vz.to_icechunk(store=store, group=group_path)

        # Step 3: Write conventions
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
            band_group = zarr.open_group(
                zarr_store,
                path=f"/{source_id}/{band}",
                zarr_format=3,
            )
            _write_multiscales(levels=levels, attrs=band_group.attrs)

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

        logger.info(
            "Virtualized %s (%d bands, levels %s, native_res=%.1f)",
            source_id,
            len(bands),
            levels,
            native_res,
        )

    return _virtualize
