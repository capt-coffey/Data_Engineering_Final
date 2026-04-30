# STL Neighborhood Intelligence Pipeline
### DATA 5035: Foundations of Data Engineering — Spring 2026

---

## Overview

The STL Neighborhood Intelligence Pipeline is an end-to-end data engineering project that integrates four public data sources to produce a neighborhood-level intelligence platform for the City of St. Louis.

St. Louis faces persistent challenges around urban disinvestment, crime concentration, property vacancy, and extreme weather vulnerability. These challenges are well-documented but historically analyzed in silos. This pipeline brings them together into a unified analytical layer that city planners, policy makers, community organizations, and researchers can use to understand neighborhood conditions holistically.

The pipeline answers questions like:
- Which St. Louis neighborhoods have the highest concentration of crime, vacancy, and economic distress simultaneously?
- How does property vacancy correlate with crime rates across neighborhoods?
- Which neighborhoods face compounded risk from both high crime and FEMA flood zones?
- What does a machine-learning-ready feature set for neighborhood property value prediction look like?

---

## Architecture

The pipeline follows a standard three-layer data engineering architecture:

```
Raw Sources
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  INGEST LAYER                                        │
│  Downloads, extracts, normalizes raw data            │
│  Writes Parquet to /Volumes/.../raw/                 │
│  ingest/gtfs.py · notebooks/ingest/                 │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  TRANSFORM LAYER                                     │
│  Cleans, joins, derives features                     │
│  Writes Parquet to /Volumes/.../transformed/         │
│  transform/geo.py · crime.py · parcels.py            │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  SERVING LAYER                                       │
│  Produces analytics and ML-ready outputs             │
│  Writes Parquet to /Volumes/.../serving/             │
│  serve/analytical.py                                 │
└─────────────────────────────────────────────────────┘
```

---

## Data Sources

| Source | Format | Coverage | Notebook |
|--------|--------|----------|----------|
| SLMPD City Crime | CSV (monthly) | Jan 2024 – present | `notebooks/ingest/nb_Crime_Ingestion.ipynb` |
| City of STL Parcels | Shapefile + DBF | Current snapshot | `notebooks/ingest/nb_Parcel_Ingestion.ipynb` |
| NOAA Climate Data | JSON REST API | Jan 2024 – present | `notebooks/ingest/nb_NOAA_Ingestion.ipynb` |
| Metro STL GTFS | ZIP (static feed) | Current schedule | `notebooks/ingest/nb_GTFS_Ingestion.ipynb` |
| Census ACS (Alondra) | JSON / CSV | 2023 ACS 5-year | `KAN19/` |

**Why these sources?**
- **Crime** — SLMPD publishes monthly NIBRS-formatted incident data covering all 79 residential neighborhoods
- **Parcels** — The city's land records DBF contains every parcel's vacancy status, assessment value, and neighborhood code
- **NOAA** — Lambert Airport is the official NWS station for STL metro; daily summaries capture extreme weather events
- **GTFS** — Metro STL's static feed provides stop locations and route coverage for transit accessibility analysis
- **Census ACS** — Median household income and housing cost burden at Census tract level, area-weighted to neighborhood boundaries

---

## Repository Structure

```
Data_Engineering_Final/
│
├── README.md
│
├── ingest/                        # Ingestion modules (Josh)
│   ├── __init__.py
│   └── gtfs.py                    # Metro STL GTFS ZIP extraction + MD5 checkpoint
│
├── transform/                     # Transformation modules (Josh)
│   ├── __init__.py
│   ├── geo.py                     # Neighborhood boundaries + Census tract overlay
│   ├── crime.py                   # Date parse, type classify, neighborhood agg
│   └── parcels.py                 # Vacancy normalize, neighborhood agg
│
├── serve/                         # Serving modules (Josh)
│   ├── __init__.py
│   └── analytical.py              # Summary table + livability ranking + ML features
│
├── notebooks/                     # Databricks notebooks — interactive runners
│   ├── ingest/
│   │   ├── nb_Crime_Ingestion.ipynb    # SLMPD monthly CSV download + normalize
│   │   ├── nb_Parcel_Ingestion.ipynb   # City shapefile + DBF download + join
│   │   ├── nb_GTFS_Ingestion.ipynb     # Metro STL GTFS ZIP extraction
│   │   └── nb_NOAA_Ingestion.ipynb     # NOAA CDO API JSON ingestion
│   ├── transform/
│   │   ├── nb_Geo_Transform.ipynb      # Build neighborhood GeoJSON + tract overlay
│   │   ├── nb_Crime_Transform.py       # Crime features + Census enrichment
│   │   └── nb_Parcel_Transform.py      # Parcel features + vacancy rate
│   └── serve/
│       ├── nb_Neighborhood_Summary.ipynb  # Livability ranking table
│       └── nb_Feature_Table.ipynb         # ML-ready feature table
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_ingest_gtfs.py        # 25 passing tests
│   ├── test_transform_crime.py    # 24 passing tests
│   └── test_transform_parcels.py  # 24 passing tests
│
└── KAN19/                         # Teammate Alondra's Census + geo work
    ├── readme.md
    ├── data/
    │   └── raw/
    │       ├── census/
    │       │   ├── acs_tract_stl.csv
    │       │   └── neighborhood_data.csv
    │       └── geo/
    │           ├── stl_neighborhoods.geojson
    │           ├── fema_flood_zones.geojson
    │           └── tract_neighborhood_overlay.geojson
    ├── ingest/
    │   └── geo.ipynb
    └── transform/
        └── census.ipynb
```

---

## Data Volume Paths (Databricks Unity Catalog)

```
workspace.default
├── raw/
│   ├── crime/city/          # Partitioned by year=/month=
│   ├── parcels/             # Single snapshot
│   ├── weather/             # Partitioned by year=/month=
│   ├── gtfs/                # Partitioned by ingest_date=
│   └── geo/                 # GeoJSON boundary files
├── transformed/
│   ├── crime/               # Neighborhood-month grain, Census enriched
│   └── parcels/             # Neighborhood grain
└── serving/
    ├── neighborhood_summary # 63 rows — all metrics joined
    ├── neighborhood_ranking # 63 rows — with livability scores
    └── ml_feature_table     # 63 rows — scaled features + train/test split
```

---

## Serving Layer Outputs

### 1. Neighborhood Summary (`serving/neighborhood_summary`)
One row per residential neighborhood. Joins crime, parcel, Census, and weather data into a single wide table for city planner and analyst use.

**Key columns:** `neighborhood`, `avg_monthly_incidents`, `vacancy_rate`, `median_income`, `housing_cost_burden`, `flood_zone_pct`, `avg_assessed_value`

### 2. Neighborhood Ranking (`serving/neighborhood_ranking`)
Same as summary plus a composite **livability score (0–100)** derived from four equally-weighted dimensions:

| Dimension | Metric | Direction |
|-----------|--------|-----------|
| Safety | avg_monthly_incidents | Lower = better |
| Vacancy | vacancy_rate | Lower = better |
| Income | median_income | Higher = better |
| Flood Risk | flood_zone_pct | Lower = better |

**Top 5 most livable:** Franz Park (86.2), Shaw (70.4), Compton Heights (65.0), Boulevard Heights (63.3), Kings Oak (63.2)

**Bottom 5:** The Ville (22.2), Wells Goodfellow (22.5), Downtown (24.2), Downtown West (26.0), Walnut Park East (27.4)

### 3. ML Feature Table (`serving/ml_feature_table`)
63 neighborhoods with raw and min-max scaled features for downstream ML modeling. Target variable: `avg_assessed_value`. 80/20 train/test split via deterministic neighborhood name hash (seed=42).

**Features:** crime rate, violent crime pct, vacancy rate, median income, housing cost burden, flood zone pct, avg high temp, avg precipitation

---

## Derived Features

| Feature | Formula | Source |
|---------|---------|--------|
| `violent_crime_pct` | violent_count / total_incidents | Crime transform |
| `vacancy_rate` | vacant_parcels / total_parcels | Parcel transform |
| `livability_score` | avg(safety, vacancy, income, flood scores) | Serving layer |
| `safety_score` | min-max normalized crime rate (inverted) | Serving layer |
| `tmax_f` / `tmin_f` | (°C × 9/5) + 32 | NOAA ingest |
| `area_weight` | intersection_area / tract_area | Geo transform |

---

## Running the Pipeline

### Prerequisites
- Databricks workspace with Unity Catalog enabled
- `workspace.default` catalog with `raw`, `transformed`, and `serving` volumes
- Python packages: `geopandas`, `simpledbf`, `requests`, `pandas`, `pytest`
- NOAA API token (free at https://www.ncdc.noaa.gov/cdo-web/token)

### Execution Order

```
1. notebooks/transform/nb_Geo_Transform.ipynb      # Build neighborhood boundaries
2. notebooks/ingest/nb_Crime_Ingestion.ipynb        # Ingest SLMPD crime data
3. notebooks/ingest/nb_Parcel_Ingestion.ipynb       # Ingest city parcel data
4. notebooks/ingest/nb_NOAA_Ingestion.ipynb         # Ingest weather data
5. notebooks/ingest/nb_GTFS_Ingestion.ipynb         # Ingest Metro transit data
6. notebooks/transform/nb_Crime_Transform.py        # Transform crime data
7. notebooks/transform/nb_Parcel_Transform.py       # Transform parcel data
8. notebooks/serve/nb_Neighborhood_Summary.ipynb    # Build serving tables
9. notebooks/serve/nb_Feature_Table.ipynb           # Build ML feature table
```

### Running Tests

```python
import subprocess, sys, os, shutil, tempfile

REPO_ROOT = "/Workspace/Users/jcoffey@wustl.edu/Data_Engineering_Final"
test_files = [
    "tests/test_ingest_gtfs.py",
    "tests/test_transform_crime.py",
    "tests/test_transform_parcels.py",
]

for test_file in test_files:
    tmpdir = tempfile.mkdtemp()
    tmp_test = os.path.join(tmpdir, os.path.basename(test_file))
    shutil.copy2(os.path.join(REPO_ROOT, test_file), tmp_test)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", tmp_test, "-v",
         "--tb=short", "--no-header"],
        capture_output=True, text=True, cwd=tmpdir,
        env={**os.environ, "PYTHONPATH": REPO_ROOT}
    )
    print(result.stdout)
    shutil.rmtree(tmpdir)
```

---

## Technical Design Decisions

**Why Databricks Serverless?**
Unity Catalog volumes provide a clean separation between raw, transformed, and serving data. Serverless compute eliminates cluster management overhead for an academic project.

**Why Parquet throughout?**
Parquet's columnar storage and partition pruning make time-range queries on crime data (year=/month= partitions) significantly faster than CSV or JSON.

**Why Pandas for CSV parsing in the ingest layer?**
SLMPD files have inconsistent quoting and mixed encodings across years. Pandas handles these edge cases more gracefully than Spark's CSV reader.

**Why min-max normalization for the livability score?**
Min-max scaling is transparent and interpretable — a score of 75 means a neighborhood is 75% of the way from worst to best on that dimension. Equal weighting is a deliberate starting point adjustable based on policy priorities.

**Why keep crime_rate_per_1000 null?**
The Census ACS data available includes median household income but not raw population counts. Rather than computing a misleading proxy metric, we built the field and documented it for when population data becomes available.

---

## Known Limitations

- `crime_rate_per_1000` is null pending Census total population data (`B01001_001E`)
- Weather data is city-wide (Lambert Airport) — no neighborhood-level weather variation
- GTFS data is ingested but not yet joined to the neighborhood serving layer
- STL County crime data source was inaccessible programmatically — city data only
- West End neighborhood excluded from serving layer due to bad Census income value
- `flood_zone_pct` appears to reflect FEMA survey coverage rather than flood risk — treat with caution until verified

---

## Future Roadmap

The current pipeline covers the core neighborhood intelligence use case but several high-value data sources were identified and scoped but not implemented within the project timeline.

### Data Sources — Planned but Not Included

**STL County Crime (NIBRS)**
County crime data is available on `data.stlouisco.com` via ArcGIS Hub for 2021–2025. The pipeline architecture supports it — a `ingest/county_crime.py` module was designed — but the ArcGIS Feature Service API returned authentication errors that were not resolved in time. Adding county data would extend coverage to the broader metro area beyond city limits.

**Census Total Population (`B01001_001E`)**
The Census ACS pull currently retrieves median household income and housing cost burden but not raw population counts. `crime_rate_per_1000` is built and ready in the crime transform — it just needs population data substituted for the income proxy. This is a one-line change once the field is added to the Census ingest module.

**Metro STL Real-Time GTFS**
The static GTFS feed (stops, routes, trips) is ingested but not joined to the neighborhood serving layer. The real-time GTFS protobuf feeds (`StlRealTimeTrips.pb`, `StlRealTimeVehicles.pb`) were identified but not implemented — these would enable transit frequency and reliability analysis per neighborhood.

**STL 311 Service Requests**
The city's 311 data (available at `stlouis-mo.gov/data`) captures resident-reported issues like illegal dumping, abandoned vehicles, and building violations. This is a strong proxy for neighborhood maintenance and city responsiveness — highly relevant to the vacancy and disinvestment story.

**Property Sale Transactions**
The city recorder's office publishes property deed transfers which would enable tracking of sales velocity, price trends, and investor activity by neighborhood. This would strengthen the ML feature table significantly for property value prediction.

### Technical Improvements — Planned but Not Implemented

**Automated Scheduling**
All four ingest notebooks were designed for scheduled Databricks Jobs (monthly for crime, weekly for GTFS, daily for NOAA, weekly for parcels). The checkpoint pattern in the GTFS ingest module demonstrates the approach. Full scheduling setup was deferred to post-presentation.

**Interactive Neighborhood Map**
The serving layer is designed to power a neighborhood intelligence map where users could click any address in St. Louis and see the neighborhood's livability score, crime breakdown, vacancy rate, and Census metrics. The `ml_feature_table` and `neighborhood_ranking` tables are the direct inputs to such a tool. Implementation would require a web front-end (e.g. Mapbox + Streamlit) connected to the serving layer.

**Weighted Livability Score**
The current livability score uses equal weighting across four dimensions. A production version would derive weights from community surveys or policy priorities — for example, weighting safety more heavily in a public safety application vs weighting income more heavily in an economic development context.

**STL County Geographic Coverage**
The current pipeline covers the 79 residential neighborhoods of St. Louis City only. Extending to St. Louis County's municipalities would require a different geographic boundary framework (municipalities rather than neighborhoods) but the same pipeline architecture would apply.


---

## Acknowledgments

- City of St. Louis GIS Department — neighborhood shapefiles and parcel data
- SLMPD — monthly crime incident data
- NOAA NCEI — Climate Data Online API
- Metro St. Louis — GTFS static feed
- US Census Bureau — American Community Survey 5-year estimates
- SLU openGIS project — neighborhood code reference data
