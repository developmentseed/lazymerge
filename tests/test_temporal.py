import numpy as np
import pytest

from lazymerge.temporal import grouper_from_period


class TestDayGrouper:
    def setup_method(self):
        self.grouper = grouper_from_period("P1D")

    def test_group_key(self):
        assert self.grouper.group_key("2024-06-15T10:30:00Z") == "2024-06-15"

    def test_group_key_strips_time(self):
        assert self.grouper.group_key("2024-01-01T23:59:59Z") == "2024-01-01"

    def test_datetime_filter(self):
        start, end = self.grouper.datetime_filter("2024-06-15")
        assert start == "2024-06-15T00:00:00Z"
        assert end == "2024-06-16T00:00:00Z"

    def test_to_datetime64(self):
        result = self.grouper.to_datetime64("2024-06-15")
        assert result == np.datetime64("2024-06-15", "D")

    def test_keys_sort_chronologically(self):
        keys = [
            self.grouper.group_key("2024-12-31T00:00:00Z"),
            self.grouper.group_key("2024-01-01T00:00:00Z"),
            self.grouper.group_key("2024-06-15T00:00:00Z"),
        ]
        assert sorted(keys) == ["2024-01-01", "2024-06-15", "2024-12-31"]


class TestWeekGrouper:
    def setup_method(self):
        self.grouper = grouper_from_period("P1W")

    def test_group_key_monday(self):
        assert self.grouper.group_key("2024-06-17T10:00:00Z") == "2024-W25"

    def test_group_key_sunday_same_week(self):
        assert self.grouper.group_key("2024-06-23T10:00:00Z") == "2024-W25"

    def test_group_key_year_boundary(self):
        assert self.grouper.group_key("2024-12-30T10:00:00Z") == "2025-W01"

    def test_datetime_filter(self):
        start, end = self.grouper.datetime_filter("2024-W25")
        assert start == "2024-06-17T00:00:00Z"
        assert end == "2024-06-24T00:00:00Z"

    def test_to_datetime64(self):
        result = self.grouper.to_datetime64("2024-W25")
        assert result == np.datetime64("2024-06-17", "D")


class TestMonthGrouper:
    def setup_method(self):
        self.grouper = grouper_from_period("P1M")

    def test_group_key(self):
        assert self.grouper.group_key("2024-06-15T10:00:00Z") == "2024-06"

    def test_group_key_first_day(self):
        assert self.grouper.group_key("2024-01-01T00:00:00Z") == "2024-01"

    def test_datetime_filter(self):
        start, end = self.grouper.datetime_filter("2024-06")
        assert start == "2024-06-01T00:00:00Z"
        assert end == "2024-07-01T00:00:00Z"

    def test_datetime_filter_december(self):
        start, end = self.grouper.datetime_filter("2024-12")
        assert start == "2024-12-01T00:00:00Z"
        assert end == "2025-01-01T00:00:00Z"

    def test_to_datetime64(self):
        result = self.grouper.to_datetime64("2024-06")
        assert result == np.datetime64("2024-06-01", "D")


class TestYearGrouper:
    def setup_method(self):
        self.grouper = grouper_from_period("P1Y")

    def test_group_key(self):
        assert self.grouper.group_key("2024-06-15T10:00:00Z") == "2024"

    def test_datetime_filter(self):
        start, end = self.grouper.datetime_filter("2024")
        assert start == "2024-01-01T00:00:00Z"
        assert end == "2025-01-01T00:00:00Z"

    def test_to_datetime64(self):
        result = self.grouper.to_datetime64("2024")
        assert result == np.datetime64("2024-01-01", "D")


class TestFixedDayGrouper:
    def setup_method(self):
        self.grouper = grouper_from_period("P16D")

    def test_group_key_epoch_start(self):
        key = self.grouper.group_key("2000-01-01T00:00:00Z")
        assert key == "000000"

    def test_group_key_same_bucket(self):
        key1 = self.grouper.group_key("2000-01-01T00:00:00Z")
        key2 = self.grouper.group_key("2000-01-16T00:00:00Z")
        assert key1 == key2

    def test_group_key_next_bucket(self):
        key = self.grouper.group_key("2000-01-17T00:00:00Z")
        assert key == "000001"

    def test_keys_sort_chronologically(self):
        keys = [
            self.grouper.group_key("2024-06-15T00:00:00Z"),
            self.grouper.group_key("2000-01-01T00:00:00Z"),
            self.grouper.group_key("2024-01-01T00:00:00Z"),
        ]
        assert sorted(keys) == [
            self.grouper.group_key("2000-01-01T00:00:00Z"),
            self.grouper.group_key("2024-01-01T00:00:00Z"),
            self.grouper.group_key("2024-06-15T00:00:00Z"),
        ]

    def test_datetime_filter(self):
        start, end = self.grouper.datetime_filter("000001")
        assert start == "2000-01-17T00:00:00Z"
        assert end == "2000-02-02T00:00:00Z"

    def test_to_datetime64(self):
        result = self.grouper.to_datetime64("000001")
        assert result == np.datetime64("2000-01-17", "D")


class TestGrouperFromPeriod:
    def test_p1d(self):
        g = grouper_from_period("P1D")
        assert g.group_key("2024-01-01T00:00:00Z") == "2024-01-01"

    def test_p1w(self):
        g = grouper_from_period("P1W")
        assert g.group_key("2024-06-17T00:00:00Z") == "2024-W25"

    def test_p1m(self):
        g = grouper_from_period("P1M")
        assert g.group_key("2024-06-15T00:00:00Z") == "2024-06"

    def test_p1y(self):
        g = grouper_from_period("P1Y")
        assert g.group_key("2024-06-15T00:00:00Z") == "2024"

    def test_p16d(self):
        g = grouper_from_period("P16D")
        assert g.group_key("2000-01-17T00:00:00Z") == "000001"

    def test_p2w_converts_to_fixed_14(self):
        g = grouper_from_period("P2W")
        key1 = g.group_key("2000-01-01T00:00:00Z")
        key2 = g.group_key("2000-01-14T00:00:00Z")
        assert key1 == key2

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            grouper_from_period("invalid")

    def test_unsupported_duration(self):
        with pytest.raises(ValueError):
            grouper_from_period("PT1H")


from unittest.mock import patch, MagicMock


class TestQueryTemporalGroups:
    def test_groups_and_deduplicates(self):
        """query_temporal_groups should bucket datetimes and return sorted unique keys."""
        from lazymerge.temporal import grouper_from_period
        from lazymerge.sources import query_temporal_groups

        grouper = grouper_from_period("P1M")

        # Mock _run_async to return raw datetime strings (bypassing DataFusion)
        with patch("lazymerge.sources._run_async", return_value=[
                "2024-06-15T10:00:00Z",
                "2024-06-20T10:00:00Z",  # same month as above — should dedup
                "2024-07-01T10:00:00Z",
                "2024-08-10T10:00:00Z",
            ]):
            groups = query_temporal_groups(
                store=MagicMock(),
                bbox_4326=(-180, -90, 180, 90),
                grouper=grouper,
            )

        assert groups == ["2024-06", "2024-07", "2024-08"]
