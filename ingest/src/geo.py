import requests
import zipfile
import io
import geopandas as gpd
import os


print("working directory:", os.getcwd())

# 1. URL for STL neighborhood boundary shapefile
STL_NEIGHBORHOOD_URL = "https://static.stlouis-mo.gov/open-data/planning/neighborhoods/neighborhoods.zip"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "ingest", "raw_data", "stl_neighborhoods.geojson")

print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("OUTPUT_PATH:", OUTPUT_PATH)


# 2. Download and extract the shapefile
def download_and_extract_shapefile(url):
    response = requests.get(url)

    print("Content-Type:", response.headers.get("Content-Type"))

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("stl_neighborhoods")
    shapefile_path = "stl_neighborhoods/neighborhoods/neighborhoods.shp"
    return shapefile_path

# 3. Load the shapefile into a GeoDataFrame
def load_shapefile_to_gdf(shapefile_path):
    return gpd.read_file(shapefile_path)
   
# 4. Reproject the GeoDataFrame to WGS84 (EPSG:4326)
def reproject_gdf(gdf):
    return gdf.to_crs(epsg=4326)

# 5. Convert GeoDataFrame to GeoJSON
def gdf_to_geojson(gdf):
    return gdf.to_json()

# 6. Save GeoJSON to file
def save_geojson(geojson, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(geojson)

if __name__ == "__main__":      
    print("Downloading and extracting STL neighborhood shapefile...")
    shapefile_path = download_and_extract_shapefile(STL_NEIGHBORHOOD_URL)

    print("Loading shapefile into GeoDataFrame...") 
    gdf = load_shapefile_to_gdf(shapefile_path)

    print("Reprojecting to WGS84...")
    gdf = reproject_gdf(gdf)

    print("Converting GeoDataFrame to GeoJSON...") 
    geojson = gdf_to_geojson(gdf)

    save_geojson(geojson, OUTPUT_PATH)

    print(f"GeoJSON saved to {OUTPUT_PATH}")
    print("Done!")      