import requests
import json
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

load_dotenv()
# Fetching data | call Census ACS 5-year API for St. Louis tracts
def fetch_st_louis_census_data(api_key):
    path = "https://api.census.gov/data/2022/acs/acs5"

    # Pull variables: median household income (B19013), housing cost burden (B25070), commute time (B08303)
    variables = "NAME,B19013_001E,B25070_001E,B08303_001E"

    # Target: St. Louis City (FIPS 510) and St. Louis County (FIPS 189) in Missouri (FIPS 29)
    # Using '*' for all tracts within those counties
    locations = [
            {"state": "29", "county": "189"}, # St. Louis County
            {"state": "29", "county": "510"}  # St. Louis City
    ]

    all_result = []
    
    for loc in locations:
        prm = {
                "get": variables,
                "for": "tract:*",
                "in": f"state:{loc['state']} county:{loc['county']}",
                "key": api_key
        }
        
        response = requests.get(path, params=prm)

        if response.status_code == 200:
            all_result.extend(response.json())

        else:
            print(f"Error fetching data for county {loc['county']}: {response.status_code}")

    save_raw_data(all_result)
    return all_result

def save_raw_data(data):
    date_str = datetime.now().strftime("%Y-%m-%d")
    root_dir = Path(__file__).resolve().parent.parent
    folder = root_dir / "raw_data"
    os.makedirs(folder, exist_ok=True)

    filename = f"{folder}/stl_census_{date_str}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data sucessfully saved to {filename}")


api_key = os.getenv('CENSUS_API_KEY')
if api_key:
    fetch_st_louis_census_data(api_key)
else:
    print("API Key not found.")



