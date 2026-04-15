import os
import json
import pytest
from pathlib import Path

# Resolve the absolute path to the root of your project
# (Assuming this file is in /tests/ and needs to go up one level to root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "raw_data"

def test_geographic_coverage():
    # Print the path for debugging if it fails
    print(f"\nDEBUG: Looking for data in {DATA_DIR}")

    if not DATA_DIR.exists():
        pytest.fail(f"Directory NOT FOUND: {DATA_DIR}")

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    
    if not files:
        pytest.fail(f"No JSON files found in {DATA_DIR}")

    latest_file = DATA_DIR / sorted(files)[-1]
    
    with open(latest_file, "r") as f:
        data = json.load(f)
        header = data[0]
        
        try:
            county_idx = header.index("county")
        except ValueError:
            pytest.fail("API response is missing the 'county' column.")
            
        counties = {row[county_idx] for row in data[1:]}
        
        assert "189" in counties, f"County 189 missing. Found: {counties}"
        assert "510" in counties, f"County 510 missing. Found: {counties}"