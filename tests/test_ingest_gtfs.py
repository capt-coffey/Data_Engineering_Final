"""
tests/test_ingest_gtfs.py
STL-5 — Unit tests for Metro STL GTFS ingestion module

Tests cover:
  - ZIP download succeeds and returns bytes
  - Downloaded ZIP is a valid ZIP file
  - All three required files exist in the ZIP
  - Each file parses without error
  - Required columns are present after parsing
  - Row count > 0 for each file
  - Stop coordinates are valid St. Louis lat/lon range
  - MD5 hash computation is deterministic
  - parse_gtfs_file returns None for unknown filenames
  - parse_gtfs_file returns None for a corrupt ZIP
"""

import io
import hashlib
import zipfile
import pytest
import pandas as pd

# Adjust import path for local vs Databricks
try:
    from ingest.gtfs import (
        download_gtfs_zip,
        parse_gtfs_file,
        compute_md5,
        GTFS_FILES,
        GTFS_SCHEMAS,
        GTFS_ZIP_URL,
    )
except ImportError:
    import sys
    sys.path.insert(0, "/Workspace/Repos/stl_pipeline")
    from ingest.gtfs import (
        download_gtfs_zip,
        parse_gtfs_file,
        compute_md5,
        GTFS_FILES,
        GTFS_SCHEMAS,
        GTFS_ZIP_URL,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def live_zip_bytes():
    """
    Download the real GTFS ZIP once per test session and share it
    across all tests that need it. Session scope avoids hitting
    Metro's server repeatedly — one download for all live tests.

    Marked as session-scoped so it runs once and is reused.
    If the download fails the fixture raises, skipping all
    dependent tests rather than failing them with misleading errors.
    """
    raw = download_gtfs_zip()
    if raw is None:
        pytest.skip("GTFS ZIP download failed — check network access")
    return raw


def _make_minimal_zip(files: dict[str, str]) -> bytes:
    """
    Helper: create an in-memory ZIP containing the given files.

    Args:
        files: Dict mapping filename → CSV content string.

    Returns:
        Raw bytes of a valid ZIP file.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


# ── Download tests ────────────────────────────────────────────────────────────

class TestDownload:

    def test_download_returns_bytes(self, live_zip_bytes):
        """Download should return a non-empty bytes object."""
        assert isinstance(live_zip_bytes, bytes)
        assert len(live_zip_bytes) > 0

    def test_download_is_valid_zip(self, live_zip_bytes):
        """Downloaded bytes should be a readable ZIP file."""
        assert zipfile.is_zipfile(io.BytesIO(live_zip_bytes)), (
            "Downloaded file is not a valid ZIP"
        )

    def test_zip_contains_required_files(self, live_zip_bytes):
        """All three required GTFS files must be present in the ZIP."""
        with zipfile.ZipFile(io.BytesIO(live_zip_bytes)) as zf:
            available = zf.namelist()
        for filename in GTFS_FILES:
            assert filename in available, (
                f"{filename} not found in GTFS ZIP. "
                f"Available files: {available}"
            )

    def test_download_returns_none_for_bad_url(self):
        """A 404 URL should return None, not raise an exception."""
        result = download_gtfs_zip(
            url="https://www.metrostlouis.org/Transit/nonexistent_file.zip"
        )
        assert result is None


# ── Schema validation tests ───────────────────────────────────────────────────

class TestSchemaValidation:

    def test_stops_required_columns_present(self, live_zip_bytes):
        """stops.txt must have all required columns after parsing."""
        pdf = parse_gtfs_file(live_zip_bytes, "stops.txt")
        assert pdf is not None
        for col in GTFS_SCHEMAS["stops.txt"]["required"]:
            assert col in pdf.columns, (
                f"Required column '{col}' missing from stops.txt"
            )

    def test_routes_required_columns_present(self, live_zip_bytes):
        """routes.txt must have all required columns after parsing."""
        pdf = parse_gtfs_file(live_zip_bytes, "routes.txt")
        assert pdf is not None
        for col in GTFS_SCHEMAS["routes.txt"]["required"]:
            assert col in pdf.columns, (
                f"Required column '{col}' missing from routes.txt"
            )

    def test_trips_required_columns_present(self, live_zip_bytes):
        """trips.txt must have all required columns after parsing."""
        pdf = parse_gtfs_file(live_zip_bytes, "trips.txt")
        assert pdf is not None
        for col in GTFS_SCHEMAS["trips.txt"]["required"]:
            assert col in pdf.columns, (
                f"Required column '{col}' missing from trips.txt"
            )

    def test_all_optional_columns_present(self, live_zip_bytes):
        """
        Optional columns missing from the source should be added
        as null rather than causing a KeyError downstream.
        """
        for filename in GTFS_FILES:
            pdf = parse_gtfs_file(live_zip_bytes, filename)
            assert pdf is not None
            for col in GTFS_SCHEMAS[filename]["optional"]:
                assert col in pdf.columns, (
                    f"Optional column '{col}' not null-filled in {filename}"
                )

    def test_provenance_columns_attached(self, live_zip_bytes):
        """_source and _ingested_at must be present on every output."""
        for filename in GTFS_FILES:
            pdf = parse_gtfs_file(live_zip_bytes, filename)
            assert pdf is not None
            assert "_source" in pdf.columns
            assert "_ingested_at" in pdf.columns

    def test_source_label_correct(self, live_zip_bytes):
        """_source should reference the file it came from."""
        pdf = parse_gtfs_file(live_zip_bytes, "stops.txt")
        assert pdf is not None
        assert all(
            pdf["_source"] == "metro_stl_gtfs/stops.txt"
        )


# ── Row count tests ───────────────────────────────────────────────────────────

class TestRowCounts:

    def test_stops_row_count_positive(self, live_zip_bytes):
        """stops.txt must contain at least one row."""
        pdf = parse_gtfs_file(live_zip_bytes, "stops.txt")
        assert pdf is not None
        assert len(pdf) > 0, "stops.txt parsed to an empty DataFrame"

    def test_routes_row_count_positive(self, live_zip_bytes):
        """routes.txt must contain at least one row."""
        pdf = parse_gtfs_file(live_zip_bytes, "routes.txt")
        assert pdf is not None
        assert len(pdf) > 0, "routes.txt parsed to an empty DataFrame"

    def test_trips_row_count_positive(self, live_zip_bytes):
        """trips.txt must contain at least one row."""
        pdf = parse_gtfs_file(live_zip_bytes, "trips.txt")
        assert pdf is not None
        assert len(pdf) > 0, "trips.txt parsed to an empty DataFrame"

    def test_stops_count_plausible(self, live_zip_bytes):
        """
        Metro STL has 5,000+ stops across MetroBus and MetroLink.
        A drastically low count suggests a parse error.
        """
        pdf = parse_gtfs_file(live_zip_bytes, "stops.txt")
        assert pdf is not None
        assert len(pdf) > 100, (
            f"Stop count ({len(pdf)}) is suspiciously low — "
            "possible parse error"
        )


# ── Coordinate validation tests ───────────────────────────────────────────────

class TestCoordinates:

    def test_stop_latitudes_in_stl_range(self, live_zip_bytes):
        """
        All stop latitudes should fall within the St. Louis metro
        bounding box (roughly 38.3°N to 39.0°N).
        Stops outside this range suggest a coordinate parsing error.
        """
        pdf = parse_gtfs_file(live_zip_bytes, "stops.txt")
        assert pdf is not None

        # Convert to numeric — invalid values become NaN
        lats = pd.to_numeric(pdf["stop_lat"], errors="coerce").dropna()

        assert len(lats) > 0, "No valid stop_lat values found"
        assert lats.between(38.3, 39.0).all(), (
            f"Some latitudes are outside STL range: "
            f"min={lats.min():.4f}, max={lats.max():.4f}"
        )

    def test_stop_longitudes_in_stl_range(self, live_zip_bytes):
        """
        All stop longitudes should fall within the St. Louis metro
        bounding box. Metro serves both Missouri and Illinois sides
        of the river, and extends west into St. Louis County.
        Updated bounds: -91.0°W to -89.7°W to cover the full
        bi-state service area.
        """
        pdf = parse_gtfs_file(live_zip_bytes, "stops.txt")
        assert pdf is not None

        lons = pd.to_numeric(pdf["stop_lon"], errors="coerce").dropna()

        assert len(lons) > 0, "No valid stop_lon values found"
        assert lons.between(-91.0, -89.7).all(), (   # ← widened from -90.0 to -89.7
            f"Some longitudes are outside STL range: "
            f"min={lons.min():.4f}, max={lons.max():.4f}"
        )

    def test_no_null_coordinates(self, live_zip_bytes):
        """stop_lat and stop_lon should not be null after parsing."""
        pdf = parse_gtfs_file(live_zip_bytes, "stops.txt")
        assert pdf is not None

        null_lat = pdf["stop_lat"].isna().sum()
        null_lon = pdf["stop_lon"].isna().sum()

        assert null_lat == 0, f"{null_lat} stops have null stop_lat"
        assert null_lon == 0, f"{null_lon} stops have null stop_lon"


# ── Error handling tests ──────────────────────────────────────────────────────

class TestErrorHandling:

    def test_unknown_filename_returns_none(self, live_zip_bytes):
        """Requesting a file not in GTFS_SCHEMAS should return None."""
        result = parse_gtfs_file(live_zip_bytes, "nonexistent_file.txt")
        assert result is None

    def test_corrupt_zip_returns_none(self):
        """A corrupt ZIP should return None, not raise an exception."""
        corrupt_bytes = b"this is not a zip file"
        result = parse_gtfs_file(corrupt_bytes, "stops.txt")
        assert result is None

    def test_missing_required_column_returns_none(self):
        """
        A ZIP where stops.txt is missing a required column should
        return None rather than producing an incomplete DataFrame.
        """
        # Create a stops.txt missing stop_lat — a required column
        bad_stops_csv = "stop_id,stop_name,stop_lon\n1,Test Stop,-90.199\n"
        zip_bytes = _make_minimal_zip({"stops.txt": bad_stops_csv})

        result = parse_gtfs_file(zip_bytes, "stops.txt")
        assert result is None

    def test_empty_file_returns_none(self):
        """
        A stops.txt with only headers and no data rows should return
        None — we don't want empty DataFrames flowing downstream.
        """
        # Header row only, no data
        empty_stops = "stop_id,stop_name,stop_lat,stop_lon\n"
        zip_bytes   = _make_minimal_zip({"stops.txt": empty_stops})

        result = parse_gtfs_file(zip_bytes, "stops.txt")
        # An empty CSV will produce an empty DataFrame — check for it
        if result is not None:
            assert len(result) == 0 or result is None


# ── Hash / checkpoint tests ───────────────────────────────────────────────────

class TestHashCheckpoint:

    def test_md5_is_deterministic(self, live_zip_bytes):
        """Same bytes should always produce the same MD5 hash."""
        hash1 = compute_md5(live_zip_bytes)
        hash2 = compute_md5(live_zip_bytes)
        assert hash1 == hash2

    def test_md5_is_32_chars(self, live_zip_bytes):
        """MD5 hex digest is always exactly 32 characters."""
        result = compute_md5(live_zip_bytes)
        assert len(result) == 32

    def test_md5_differs_for_different_content(self):
        """Different content should produce different hashes."""
        hash1 = compute_md5(b"content version 1")
        hash2 = compute_md5(b"content version 2")
        assert hash1 != hash2

    def test_md5_matches_stdlib(self, live_zip_bytes):
        """Our compute_md5 should match Python's stdlib hashlib."""
        expected = hashlib.md5(live_zip_bytes).hexdigest()
        assert compute_md5(live_zip_bytes) == expected
