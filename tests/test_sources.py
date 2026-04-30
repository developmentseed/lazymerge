import zarr

from lazymerge.conventions import ProjAttrs, SpatialAttrs, write_proj, write_spatial
from lazymerge.sources import scan_store


def _make_source_store():
    """Create a store with two source arrays in the same CRS but different extents."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    # Source A: covers x=[500000, 501000], y=[5999000, 6000000] in EPSG:32618
    a = root.create_array("source_a", shape=(100, 100), dtype="f4", chunks=(50, 50))
    a[:] = 1.0
    write_spatial(
        a,
        SpatialAttrs(
            dimensions=["y", "x"],
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
            bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
            shape=(100, 100),
        ),
    )
    write_proj(a, ProjAttrs(code="EPSG:32618"))

    # Source B: covers x=[501000, 502000], y=[5999000, 6000000] in EPSG:32618
    b = root.create_array("source_b", shape=(100, 100), dtype="f4", chunks=(50, 50))
    b[:] = 2.0
    write_spatial(
        b,
        SpatialAttrs(
            dimensions=["y", "x"],
            transform=(10.0, 0.0, 501000.0, 0.0, -10.0, 6000000.0),
            bbox=(501000.0, 5999000.0, 502000.0, 6000000.0),
            shape=(100, 100),
        ),
    )
    write_proj(b, ProjAttrs(code="EPSG:32618"))

    return store, root


def test_scan_store_finds_all_sources():
    store, root = _make_source_store()
    index = scan_store(root)
    assert len(index.entries) == 2
    paths = {e.path for e in index.entries}
    assert paths == {"source_a", "source_b"}


def test_find_intersecting_sources_full_overlap():
    """A target bbox covering both sources should return both."""
    store, root = _make_source_store()
    index = scan_store(root)
    results = index.find_intersecting_sources(
        target_bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 2


def test_find_intersecting_sources_partial():
    """A target bbox covering only source_a's extent should return only source_a."""
    store, root = _make_source_store()
    index = scan_store(root)
    results = index.find_intersecting_sources(
        target_bbox=(500000.0, 5999000.0, 500500.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 1
    assert results[0].path == "source_a"


def test_find_intersecting_sources_none():
    """A target bbox far away should return nothing."""
    store, root = _make_source_store()
    index = scan_store(root)
    results = index.find_intersecting_sources(
        target_bbox=(0.0, 0.0, 1.0, 1.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 0


def test_find_intersecting_chunks_subset():
    """Only chunks overlapping the target chunk bbox should be returned."""
    store, root = _make_source_store()
    index = scan_store(root)
    source_a = [e for e in index.entries if e.path == "source_a"][0]

    results = index.find_intersecting_chunks(
        source=source_a,
        target_chunk_bbox=(500000.0, 5999000.0, 500500.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    col_indices = {c[1] for _, c in results}
    assert col_indices == {0}
    assert len(results) > 0


def test_find_intersecting_chunks_no_overlap():
    """Target chunk bbox outside source should return empty list."""
    store, root = _make_source_store()
    index = scan_store(root)
    source_a = [e for e in index.entries if e.path == "source_a"][0]

    results = index.find_intersecting_chunks(
        source=source_a,
        target_chunk_bbox=(600000.0, 5999000.0, 601000.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 0


# --- Mixed CRS tests ---


def _make_mixed_crs_store():
    """Create a store with source_a in EPSG:32618 and source_b in EPSG:32617.

    source_a covers x=[500000, 501000], y=[5999000, 6000000] in UTM 18N.
    source_b covers the equivalent geographic area of x=[501000, 502000], y=[5999000, 6000000]
    in UTM 18N, but stored in UTM 17N coordinates: x=[892673, 893756], y=[6015735, 6016818].
    """
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")

    a = root.create_array("source_a", shape=(100, 100), dtype="f4", chunks=(50, 50))
    a[:] = 1.0
    write_spatial(
        a,
        SpatialAttrs(
            dimensions=["y", "x"],
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),
            bbox=(500000.0, 5999000.0, 501000.0, 6000000.0),
            shape=(100, 100),
        ),
    )
    write_proj(a, ProjAttrs(code="EPSG:32618"))

    b_width = int((893756.0 - 892673.0) / 10.0)
    b_height = int((6016818.0 - 6015735.0) / 10.0)
    b = root.create_array("source_b", shape=(b_height, b_width), dtype="f4", chunks=(50, 50))
    b[:] = 2.0
    write_spatial(
        b,
        SpatialAttrs(
            dimensions=["y", "x"],
            transform=(10.0, 0.0, 892673.0, 0.0, -10.0, 6016818.0),
            bbox=(892673.0, 6015735.0, 893756.0, 6016818.0),
            shape=(b_height, b_width),
        ),
    )
    write_proj(b, ProjAttrs(code="EPSG:32617"))

    return store, root


def test_scan_store_mixed_crs():
    """scan_store should find sources regardless of CRS."""
    store, root = _make_mixed_crs_store()
    index = scan_store(root)
    assert len(index.entries) == 2
    crs_values = {e.proj_attrs.code for e in index.entries}
    assert crs_values == {"EPSG:32618", "EPSG:32617"}


def test_find_intersecting_sources_mixed_crs_both():
    """A target bbox in UTM 18N spanning both sources should find both,
    even though source_b is in UTM 17N."""
    store, root = _make_mixed_crs_store()
    index = scan_store(root)
    results = index.find_intersecting_sources(
        target_bbox=(500000.0, 5999000.0, 502000.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 2
    paths = {e.path for e in results}
    assert paths == {"source_a", "source_b"}


def test_find_intersecting_sources_mixed_crs_only_utm17n():
    """A target bbox that only overlaps the UTM 17N source should find only it."""
    store, root = _make_mixed_crs_store()
    index = scan_store(root)
    # This bbox in UTM 18N covers ~x=[501500, 502000] — only source_b territory
    results = index.find_intersecting_sources(
        target_bbox=(501500.0, 5999500.0, 502000.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 1
    assert results[0].path == "source_b"


def test_find_intersecting_sources_mixed_crs_none():
    """A target bbox far from both sources should find nothing."""
    store, root = _make_mixed_crs_store()
    index = scan_store(root)
    results = index.find_intersecting_sources(
        target_bbox=(600000.0, 5999000.0, 601000.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 0


def test_find_intersecting_chunks_mixed_crs():
    """find_intersecting_chunks should work when source and target are in different CRS."""
    store, root = _make_mixed_crs_store()
    index = scan_store(root)
    source_b = [e for e in index.entries if e.path == "source_b"][0]

    # Query with a target chunk bbox in UTM 18N that overlaps source_b
    results = index.find_intersecting_chunks(
        source=source_b,
        target_chunk_bbox=(501000.0, 5999000.0, 502000.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    # Should find at least some chunks from the UTM 17N source
    assert len(results) > 0
    # All results should reference source_b
    assert all(path == "source_b" for path, _ in results)


def test_find_intersecting_chunks_mixed_crs_no_overlap():
    """A target chunk bbox that doesn't overlap the UTM 17N source should return nothing."""
    store, root = _make_mixed_crs_store()
    index = scan_store(root)
    source_b = [e for e in index.entries if e.path == "source_b"][0]

    results = index.find_intersecting_chunks(
        source=source_b,
        target_chunk_bbox=(499000.0, 5999000.0, 500000.0, 6000000.0),
        target_crs="EPSG:32618",
    )
    assert len(results) == 0
