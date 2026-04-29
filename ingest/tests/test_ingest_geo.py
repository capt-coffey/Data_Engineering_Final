import pytest
import os
import geopandas as gpd


OUTPUT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "ingest", "raw_data", "stl_neighborhoods.geojson")
)


def test_geojson_file_exists():
    assert os.path.exists(OUTPUT_PATH), f"GeoJSON file not found at {OUTPUT_PATH}"


def test_geojson_loads():
    """Test that the GeoJSON file can be loaded into a GeoDataFrame."""
    gdf = gpd.read_file(OUTPUT_PATH)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0

def test_crs_is_epsg4326():
    """Test that the GeoDataFrame is in the correct CRS (EPSG:4326)."""
    gdf = gpd.read_file(OUTPUT_PATH)
    assert gdf.crs is not None, "CRS is not defined in the GeoDataFrame"
    assert gdf.crs.to_epsg() == 4326, f"Expected CRS EPSG:4326 but got {gdf.crs.to_epsg()}" 

def test_neighborhood_count():
    """Test that the number of neighborhoods matches expected count."""
    gdf = gpd.read_file(OUTPUT_PATH)
    expected_count = 79  # Update this if the expected number of neighborhoods changes
    assert len(gdf) == expected_count, f"Expected {expected_count} neighborhoods but got {len(gdf)}"