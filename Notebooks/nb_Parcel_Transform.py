# Databricks notebook source
# STL Neighborhood Intelligence — Parcel Data Transformation
# Reads raw parcel Parquet and produces a neighborhood-level
# feature table with vacancy rate and assessment value metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Imports & Logging
# MAGIC
# MAGIC Standard imports. Note that `pyspark.sql.functions` is aliased
# MAGIC as `F` throughout — this is the standard Spark convention.

# COMMAND ----------

import logging
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("stl_parcel_transform")

print("✓ Imports loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Configuration
# MAGIC
# MAGIC All paths in one place. `RAW_PATH` points to the ingested parcel
# MAGIC Parquet. `OUTPUT_PATH` is where the transformed neighborhood-level
# MAGIC feature table will be written.

# COMMAND ----------

# Input — raw parcel Parquet from parcel ingest notebook
RAW_PATH = "/Volumes/workspace/default/raw/parcels"

# Output — transformed neighborhood-level feature table
OUTPUT_PATH = "/Volumes/workspace/default/transformed/parcels"

print(f"✓ Raw input:  {RAW_PATH}")
print(f"✓ Output:     {OUTPUT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Neighborhood Lookup Table
# MAGIC
# MAGIC The parcel DBF stores `neighborhood` as a numeric code (1–88)
# MAGIC matching the NHD_NUM field in the city shapefile. This lookup
# MAGIC table maps those codes to official neighborhood names.
# MAGIC
# MAGIC Source: City of St. Louis neighborhoods shapefile, verified
# MAGIC against the downloaded shapefile on 2026-04-19.

# COMMAND ----------

# Authoritative NHD_NUM → NHD_NAME mapping
# Verified directly from City of St. Louis neighborhoods shapefile
NEIGHBORHOOD_LOOKUP: dict[int, str] = {
    0:  "Unknown",
    1:  "Carondelet",
    2:  "Patch",
    3:  "Holly Hills",
    4:  "Boulevard Heights",
    5:  "Bevo Mill",
    6:  "Princeton Heights",
    7:  "Southampton",
    8:  "St. Louis Hills",
    9:  "Lindenwood Park",
    10: "Ellendale",
    11: "Clifton Heights",
    12: "The Hill",
    13: "Southwest Garden",
    14: "North Hampton",
    15: "Tower Grove South",
    16: "Dutchtown",
    17: "Mount Pleasant",
    18: "Marine Villa",
    19: "Gravois Park",
    20: "Kosciusko",
    21: "Soulard",
    22: "Benton Park",
    23: "McKinley Heights",
    24: "Fox Park",
    25: "Tower Grove East",
    26: "Compton Heights",
    27: "Shaw",
    28: "Botanical Heights",
    29: "Tiffany",
    30: "Benton Park West",
    31: "The Gate District",
    32: "Lafayette Square",
    33: "Peabody Darst Webbe",
    34: "LaSalle Park",
    35: "Downtown",
    36: "Downtown West",
    37: "Midtown",
    38: "Central West End",
    39: "Forest Park South East",
    40: "Kings Oak",
    41: "Cheltenham",
    42: "Clayton-Tamm",
    43: "Franz Park",
    44: "Hi-Pointe",
    45: "Wydown Skinker",
    46: "Skinker DeBaliviere",
    47: "DeBaliviere Place",
    48: "West End",
    49: "Visitation Park",
    50: "Wells Goodfellow",
    51: "Academy",
    52: "Kingsway West",
    53: "Fountain Park",
    54: "Lewis Place",
    55: "Kingsway East",
    56: "Greater Ville",
    57: "The Ville",
    58: "Vandeventer",
    59: "Jeff Vanderlou",
    60: "St. Louis Place",
    61: "Carr Square",
    62: "Columbus Square",
    63: "Old North St. Louis",
    64: "Near North Riverfront",
    65: "Hyde Park",
    66: "College Hill",
    67: "Fairground Neighborhood",
    68: "O'Fallon",
    69: "Penrose",
    70: "Mark Twain I-70 Industrial",
    71: "Mark Twain",
    72: "Walnut Park East",
    73: "North Pointe",
    74: "Baden",
    75: "Riverview",
    76: "Walnut Park West",
    77: "Covenant Blu-Grand Center",
    78: "Hamilton Heights",
    79: "North Riverfront",
}

print(f"✓ Lookup table loaded: {len(NEIGHBORHOOD_LOOKUP)} codes")
print(f"\nSpot checks:")
print(f"  Code 16 → {NEIGHBORHOOD_LOOKUP[16]}")   # Dutchtown — highest parcel count
print(f"  Code 35 → {NEIGHBORHOOD_LOOKUP[35]}")   # Downtown
print(f"  Code 38 → {NEIGHBORHOOD_LOOKUP[38]}")   # Central West End

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Load Raw Parcel Data
# MAGIC
# MAGIC Reads the raw Parquet written by the parcel ingest notebook.
# MAGIC Prints a summary to confirm the data looks as expected before
# MAGIC applying any transformations.

# COMMAND ----------

df_raw = spark.read.parquet(RAW_PATH)

total_rows = df_raw.count()
print(f"✓ Loaded {total_rows:,} raw parcel records")
print(f"\nSchema:")
df_raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Map Neighborhood Codes to Names
# MAGIC
# MAGIC The parcel data stores neighborhood as a numeric string (e.g. "16").
# MAGIC We cast it to integer and use a Spark map literal to look up the
# MAGIC official neighborhood name.
# MAGIC
# MAGIC `F.create_map()` builds a column-level key-value map from the
# MAGIC Python lookup dict — this is more efficient than a UDF because
# MAGIC it runs natively in the Spark execution engine.

# COMMAND ----------

# Build a Spark map literal from the Python lookup dict
# F.create_map takes alternating key, value literal expressions
map_entries = []
for code, name in NEIGHBORHOOD_LOOKUP.items():
    map_entries.extend([F.lit(code), F.lit(name)])

neighborhood_map = F.create_map(*map_entries)

df_named = (
    df_raw
    # Cast neighborhood string to integer for map lookup
    .withColumn(
        "neighborhood_id",
        F.col("neighborhood").cast(IntegerType())
    )
    # Look up the name — returns null for codes not in the map
    .withColumn(
        "neighborhood_name",
        neighborhood_map[F.col("neighborhood_id")]
    )
    # Fill any unmatched codes with "Unknown"
    .withColumn(
        "neighborhood_name",
        F.coalesce(F.col("neighborhood_name"), F.lit("Unknown"))
    )
)

# Verify the mapping worked
unknown_count = df_named.filter(
    F.col("neighborhood_name") == "Unknown"
).count()
print(f"✓ Neighborhood names mapped")
print(f"  Matched:   {total_rows - unknown_count:,}")
print(f"  Unmatched: {unknown_count:,}")

# Show the distribution
display(
    df_named.groupBy("neighborhood_id", "neighborhood_name")
    .count()
    .orderBy("count", ascending=False)
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Normalize Vacancy Flag
# MAGIC
# MAGIC The raw parcel data stores vacancy as `"Y"` or `"N"` strings.
# MAGIC We convert this to a 1/0 integer flag so it can be directly
# MAGIC summed in the aggregation step — `SUM(is_vacant_flag)` gives
# MAGIC us the vacant parcel count without any extra logic.

# COMMAND ----------

df_vacancy = df_named.withColumn(
    "is_vacant_flag",
    # Use F.upper() to handle any case variations (y, Y, yes, YES)
    F.when(F.upper(F.col("is_vacant")) == "Y", 1)
     .otherwise(0)
)

# Verify distribution matches what we saw in raw data
print("✓ Vacancy flag normalized")
display(
    df_vacancy.groupBy("is_vacant", "is_vacant_flag")
    .count()
    .orderBy("is_vacant")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Aggregate to Neighborhood Grain
# MAGIC
# MAGIC Groups all parcels by neighborhood and computes:
# MAGIC - Total parcel count
# MAGIC - Vacant parcel count (sum of 1/0 flag)
# MAGIC - Average and total assessed/appraised values
# MAGIC
# MAGIC Note: $0 assessed values are filtered from the averages since
# MAGIC city-owned and tax-exempt parcels assessed at $0 would skew
# MAGIC the mean downward and misrepresent actual property values.

# COMMAND ----------

df_agg = (
    df_vacancy
    # Exclude parcels with unknown neighborhood code
    .filter(F.col("neighborhood_name") != "Unknown")
    .groupBy("neighborhood_id", "neighborhood_name")
    .agg(
        # Total parcel count
        F.count("parcel_id").alias("total_parcels"),

        # Vacant parcel count — direct sum of 1/0 flag
        F.sum("is_vacant_flag").alias("vacant_parcels"),

        # Average assessed value — exclude $0 parcels from mean
        F.round(
            F.avg(
                F.when(F.col("assessed_value") > 0,
                       F.col("assessed_value"))
            ), 2
        ).alias("avg_assessed_value"),

        # Average appraised value — exclude $0 parcels from mean
        F.round(
            F.avg(
                F.when(F.col("appraised_value") > 0,
                       F.col("appraised_value"))
            ), 2
        ).alias("avg_appraised_value"),

        # Total assessed value for tax base analysis
        F.round(
            F.sum("assessed_value"), 2
        ).alias("total_assessed_value"),
    )
    # Derive occupied parcel count
    .withColumn(
        "occupied_parcels",
        F.col("total_parcels") - F.col("vacant_parcels")
    )
    .orderBy("neighborhood_name")
)

print(f"✓ Aggregation complete: {df_agg.count()} neighborhoods")
display(df_agg.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Derive vacancy_rate Feature
# MAGIC
# MAGIC Computes the share of parcels that are vacant per neighborhood.
# MAGIC This is the primary derived feature from the parcel data.
# MAGIC
# MAGIC Formula: `vacancy_rate = vacant_parcels / total_parcels`
# MAGIC
# MAGIC Rounded to 4 decimal places (e.g. 0.1823 = 18.23% vacant).
# MAGIC Null-safe — if total_parcels is somehow 0 the result is null
# MAGIC rather than a division error.

# COMMAND ----------

df_features = df_agg.withColumn(
    "vacancy_rate",
    F.round(
        # Cast to double before division — integer division truncates
        F.col("vacant_parcels").cast(DoubleType()) /
        F.col("total_parcels").cast(DoubleType()),
        4
    )
)

print("✓ vacancy_rate derived")
print("\nVacancy rate summary:")
display(
    df_features.select(
        F.min("vacancy_rate").alias("min"),
        F.round(F.avg("vacancy_rate"), 4).alias("avg"),
        F.max("vacancy_rate").alias("max"),
        F.percentile_approx("vacancy_rate", 0.25).alias("p25"),
        F.percentile_approx("vacancy_rate", 0.75).alias("p75"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 — Write Output Parquet
# MAGIC
# MAGIC Writes the final neighborhood-level parcel feature table.
# MAGIC Unlike the crime transform, parcels are a single snapshot
# MAGIC so there's no year/month partitioning — one row per neighborhood.

# COMMAND ----------

# Ensure the transformed volume exists before writing
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.transformed")

df_features.write.mode("overwrite").parquet(OUTPUT_PATH)

output_count = spark.read.parquet(OUTPUT_PATH).count()
print(f"✓ Wrote {output_count:,} rows → {OUTPUT_PATH}")
df_features.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10 — Quality Checks
# MAGIC
# MAGIC Verify the output looks correct. Key things to check:
# MAGIC - Null rates on all derived features
# MAGIC - Top neighborhoods by vacancy rate (high vacancy = disinvestment)
# MAGIC - Top neighborhoods by assessed value (wealth distribution)
# MAGIC - Overall vacancy rate matches raw data (~18%)

# COMMAND ----------

result = spark.read.parquet(OUTPUT_PATH)

print("=== NULL RATES (%) ===")
display(result.select([
    F.round(
        F.count(F.when(F.col(c).isNull(), c)) /
        F.count(F.lit(1)) * 100, 2
    ).alias(c)
    for c in [
        "neighborhood_name", "total_parcels", "vacant_parcels",
        "vacancy_rate", "avg_assessed_value"
    ]
]))

# COMMAND ----------

print("=== TOP 15 NEIGHBORHOODS BY VACANCY RATE ===")
display(
    result
    .orderBy("vacancy_rate", ascending=False)
    .select(
        "neighborhood_name",
        "total_parcels",
        "vacant_parcels",
        "vacancy_rate"
    )
    .limit(15)
)

# COMMAND ----------

print("=== TOP 15 NEIGHBORHOODS BY AVG ASSESSED VALUE ===")
display(
    result
    .orderBy("avg_assessed_value", ascending=False)
    .select(
        "neighborhood_name",
        "total_parcels",
        "avg_assessed_value",
        "avg_appraised_value",
        "vacancy_rate"
    )
    .limit(15)
)

# COMMAND ----------

print("=== OVERALL VACANCY RATE ===")
display(
    result.select(
        F.sum("vacant_parcels").alias("total_vacant"),
        F.sum("total_parcels").alias("total_parcels"),
        F.round(
            F.sum("vacant_parcels").cast(DoubleType()) /
            F.sum("total_parcels").cast(DoubleType()) * 100, 2
        ).alias("overall_vacancy_pct")
    )
)
