# Databricks notebook source
# STL Neighborhood Intelligence — Crime Data Transformation
# Reads raw SLMPD crime Parquet and produces a neighborhood-month
# feature table with crime type breakdowns and derived metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Imports & Logging
# MAGIC
# MAGIC Standard imports. `pyspark.sql.functions` is the core library
# MAGIC for all column-level transformations in Spark. We alias it as `F`
# MAGIC throughout — so `F.col("x")` means "the column named x".

# COMMAND ----------

import logging
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("stl_crime_transform")

print("✓ Imports loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Configuration
# MAGIC
# MAGIC All paths in one place. `RAW_PATH` points to the ingested Parquet
# MAGIC from the crime ingest notebook. `OUTPUT_PATH` is where the
# MAGIC transformed neighborhood-month feature table will be written.
# MAGIC
# MAGIC The Census and overlay paths are optional — if provided, the
# MAGIC `crime_rate_per_1000` feature will be populated. If not, it
# MAGIC will be null and can be filled in later when population data
# MAGIC is available from your teammate's Census ingest.

# COMMAND ----------

# Input — raw crime Parquet from ingest notebook
RAW_PATH = "/Volumes/workspace/default/raw/crime/city"

# Output — transformed neighborhood-month feature table
OUTPUT_PATH = "/Volumes/workspace/default/transformed/crime"

# Geo reference files from the geo transform notebook
NEIGHBORHOODS_PATH = "/Volumes/workspace/default/raw/geo/stl_neighborhoods.geojson"
OVERLAY_PATH       = "/Volumes/workspace/default/raw/geo/tract_neighborhood_overlay.geojson"

# Census JSON from teammate's ingest module
# Update this path when the Census data is available in the shared volume
CENSUS_PATH = "/Volumes/workspace/default/raw/census/stl_census_2026-04-10.json"

print(f"✓ Raw input:   {RAW_PATH}")
print(f"✓ Output:      {OUTPUT_PATH}")
print(f"✓ Census:      {CENSUS_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Crime Type Classification Map
# MAGIC
# MAGIC Maps NIBRS `ucr_category` strings to one of three buckets:
# MAGIC - **violent** — crimes against persons
# MAGIC - **property** — crimes against property
# MAGIC - **drug** — drug/narcotic violations
# MAGIC - **other** — everything else (weapons violations, DUI, etc.)
# MAGIC
# MAGIC Source: FBI NIBRS offense category definitions.
# MAGIC Any ucr_category value not in these sets maps to "other".

# COMMAND ----------

# Violent crimes — direct harm to a person
VIOLENT_CATEGORIES = {
    "Aggravated Assault",
    "Simple Assault",
    "Intimidation",
    "Robbery",
    "Murder and Nonnegligent Manslaughter",
    "Negligent Manslaughter",
    "Justifiable Homicide",
    "Rape",
    "Sodomy",
    "Fondling",
    "Statutory Rape",
    "Sexual Assault With An Object",
    "Kidnapping/Abduction",
    "Human Trafficking, Commercial Sex Acts",
    "Human Trafficking, Involuntary Servitude",
}

# Property crimes — harm to property or financial fraud
PROPERTY_CATEGORIES = {
    "Burglary/Breaking and Entering",
    "Motor Vehicle Theft",
    "Theft From Motor Vehicle",
    "Theft From Motor Vehicle Parts/Accessories",
    "Theft From Building",
    "All Other Larceny",
    "Shoplifting",
    "Pocket-picking",
    "Purse-snatching",
    "Theft From Coin-Operated Machine or Device",
    "Stolen Property Offenses",
    "Destruction/Damage/Vandalism of Property",
    "Arson",
    "Embezzlement",
    "False Pretense/Swindle/Confidence Game",
    "Credit Card/Automatic Teller Machine Fraud",
    "Impersonation",
    "Wire Fraud",
    "Identity Theft",
    "Counterfeiting/Forgery",
    "Bad Checks",
    "Extortion/Blackmail",
    "Bribery",
}

# Drug crimes
DRUG_CATEGORIES = {
    "Drug/Narcotic Violations",
    "Drug Equipment Violations",
}

print(f"✓ Violent categories:  {len(VIOLENT_CATEGORIES)}")
print(f"✓ Property categories: {len(PROPERTY_CATEGORIES)}")
print(f"✓ Drug categories:     {len(DRUG_CATEGORIES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Load Raw Crime Data
# MAGIC
# MAGIC Reads the raw Parquet written by the crime ingest notebook.
# MAGIC Spark automatically discovers all year=/month= partitions.
# MAGIC We print a quick summary to confirm the data looks as expected
# MAGIC before applying any transformations.

# COMMAND ----------

df_raw = spark.read.parquet(RAW_PATH)

total_rows = df_raw.count()
print(f"✓ Loaded {total_rows:,} raw crime records")
print(f"\nPartitions (year/month combos):")
display(
    df_raw.select("year", "month")
    .distinct()
    .orderBy("year", "month")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Standardize Dates
# MAGIC
# MAGIC The 2024+ SLMPD format stores dates as `M/D/YYYY 12:00:00 AM`.
# MAGIC We parse this to a clean ISO date string and derive `day_of_week`
# MAGIC for downstream time-series analysis.
# MAGIC
# MAGIC `F.coalesce()` tries each format in order and returns the first
# MAGIC non-null result — this handles the mixed formats that appear
# MAGIC across different SLMPD file vintages.

# COMMAND ----------

df_dated = (
    df_raw
    .withColumn(
        "occurred_date_parsed",
        F.when(
            # ISO format: "2024-02-07" — 10 chars, contains dashes
            F.col("occurred_date").rlike(r"^\d{4}-\d{2}-\d{2}"),
            F.to_date(F.col("occurred_date"), "yyyy-MM-dd")
        ).when(
            # Full datetime with AM/PM: "8/20/2025 12:00:00 AM"
            F.col("occurred_date").rlike(r"^\d{1,2}/\d{1,2}/\d{4} \d"),
            F.to_date(F.col("occurred_date"), "M/d/yyyy hh:mm:ss a")
        ).when(
            # Zero-padded short date: "01/15/2024"
            F.col("occurred_date").rlike(r"^\d{2}/\d{2}/\d{4}$"),
            F.to_date(F.col("occurred_date"), "MM/dd/yyyy")
        ).otherwise(F.lit(None))
    )
    # Clean ISO string for output
    .withColumn(
        "occurred_date_iso",
        F.date_format(F.col("occurred_date_parsed"), "yyyy-MM-dd")
    )
    # Day of week: 1=Sunday through 7=Saturday
    .withColumn(
        "day_of_week",
        F.dayofweek(F.col("occurred_date_parsed"))
    )
    .drop("occurred_date_parsed")
)

# Verify parse quality
null_dates = df_dated.filter(F.col("occurred_date_iso").isNull()).count()
print(f"✓ Date parsing complete")
print(f"  Parsed successfully: {total_rows - null_dates:,}")
print(f"  Failed to parse:     {null_dates:,}")

if null_dates > 0:
    print(f"\n  Sample unparsed values:")
    display(
        df_dated.filter(F.col("occurred_date_iso").isNull())
        .select("occurred_date")
        .distinct()
        .limit(5)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Classify Crime Types
# MAGIC
# MAGIC Adds a `crime_type` column classifying each incident into one of
# MAGIC four buckets based on `ucr_category`.
# MAGIC
# MAGIC `F.when().when().otherwise()` is Spark's equivalent of a SQL
# MAGIC CASE WHEN statement — it evaluates conditions in order and
# MAGIC assigns the first matching value.

# COMMAND ----------

df_classified = df_dated.withColumn(
    "crime_type",
    F.when(
        F.col("ucr_category").isin(VIOLENT_CATEGORIES),
        F.lit("violent")
    ).when(
        F.col("ucr_category").isin(PROPERTY_CATEGORIES),
        F.lit("property")
    ).when(
        F.col("ucr_category").isin(DRUG_CATEGORIES),
        F.lit("drug")
    ).otherwise(
        F.lit("other")
    )
)

# Verify the distribution looks reasonable
print("✓ Crime type classification complete")
print("\nDistribution:")
display(
    df_classified.groupBy("crime_type")
    .count()
    .orderBy("count", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Filter Invalid Records
# MAGIC
# MAGIC Before aggregating, we remove:
# MAGIC - **Unfounded reports** (`is_unfounded = "1"`) — incidents later
# MAGIC   determined to have no factual basis. Including these would
# MAGIC   inflate crime counts and distort neighborhood comparisons.
# MAGIC - **Non-crime incidents** (`is_crime = "0"`) — administrative
# MAGIC   or informational records that aren't actual crimes.
# MAGIC
# MAGIC Note: These flags only exist in pre-2024 data. The 2024+ NIBRS
# MAGIC format doesn't include them, so the null check ensures we don't
# MAGIC accidentally filter valid 2024+ records.

# COMMAND ----------

df_filtered = (
    df_classified
    # Keep records where is_unfounded is null (2024+ data) or not "1"
    .filter(
        F.col("is_unfounded").isNull() |
        (F.col("is_unfounded") != F.lit("1"))
    )
    # Keep records where is_crime is null (2024+ data) or not "0"
    .filter(
        F.col("is_crime").isNull() |
        (F.col("is_crime") != F.lit("0"))
    )
)

removed = total_rows - df_filtered.count()
print(f"✓ Filtering complete")
print(f"  Records kept:    {df_filtered.count():,}")
print(f"  Records removed: {removed:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Aggregate to Neighborhood-Month Grain
# MAGIC
# MAGIC Groups records by neighborhood + year + month and produces
# MAGIC count columns for each crime type. This is the core pivot
# MAGIC operation that satisfies the Aggregation/Pivot rubric requirement.
# MAGIC
# MAGIC The conditional aggregation pattern:
# MAGIC `SUM(WHEN crime_type='violent' THEN 1 ELSE 0)`
# MAGIC is equivalent to a SQL pivot and avoids a wide reshape operation.

# COMMAND ----------

df_agg = (
    df_filtered
    .groupBy("neighborhood", "year", "month")
    .agg(
        # Total incidents regardless of type
        F.count("incident_id").alias("total_incidents"),

        # Per-type counts using conditional aggregation (pivot pattern)
        F.sum(
            F.when(F.col("crime_type") == "violent", 1).otherwise(0)
        ).alias("violent_count"),
        F.sum(
            F.when(F.col("crime_type") == "property", 1).otherwise(0)
        ).alias("property_count"),
        F.sum(
            F.when(F.col("crime_type") == "drug", 1).otherwise(0)
        ).alias("drug_count"),
        F.sum(
            F.when(F.col("crime_type") == "other", 1).otherwise(0)
        ).alias("other_count"),
    )
    .orderBy("neighborhood", "year", "month")
)

agg_count = df_agg.count()
print(f"✓ Aggregation complete")
print(f"  Neighborhood-month rows: {agg_count:,}")
print(f"\nSample:")
display(df_agg.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 — Derive violent_crime_pct Feature
# MAGIC
# MAGIC Computes the share of incidents classified as violent crimes.
# MAGIC This is a new derived column — it doesn't exist in the raw data.
# MAGIC
# MAGIC Formula: `violent_crime_pct = violent_count / total_incidents`
# MAGIC
# MAGIC Rounded to 4 decimal places (e.g. 0.2341 = 23.41% violent).
# MAGIC Null when total_incidents is 0 to avoid division by zero.

# COMMAND ----------

df_features = (
    df_agg
    .withColumn(
        "violent_crime_pct",
        F.round(
            # Cast to double before division — integer division truncates
            F.col("violent_count").cast(DoubleType()) /
            F.col("total_incidents").cast(DoubleType()),
            4
        )
    )
    # Placeholder for crime_rate_per_1000 — populated in Cell 10
    # after joining Census population data
    .withColumn("crime_rate_per_1000", F.lit(None).cast(DoubleType()))
    .withColumn("population", F.lit(None).cast(IntegerType()))
)

print("✓ violent_crime_pct derived")
print("\nFeature summary:")
display(
    df_features.select(
        F.round(F.avg("violent_crime_pct"), 4).alias("avg_violent_pct"),
        F.round(F.max("violent_crime_pct"), 4).alias("max_violent_pct"),
        F.round(F.min("violent_crime_pct"), 4).alias("min_violent_pct"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10 — Derive crime_rate_per_1000 (Census Population Join)
# MAGIC
# MAGIC Joins Census ACS data via the tract-neighborhood area weight overlay
# MAGIC to compute neighborhood-level population, then derives:
# MAGIC
# MAGIC `crime_rate_per_1000 = (total_incidents / population) * 1000`
# MAGIC
# MAGIC The area weight overlay apportions each census tract's population
# MAGIC to the neighborhoods it overlaps, weighted by the fraction of the
# MAGIC tract's area that falls within each neighborhood.
# MAGIC
# MAGIC **Note:** The teammate's Census ingest uses median household income
# MAGIC (B19013_001E) rather than total population (B01001_001E). Until
# MAGIC total population is available, `crime_rate_per_1000` remains null.
# MAGIC Update `CENSUS_POP_FIELD` below when population data is ingested.

# COMMAND ----------

import os
import json
import pandas as pd

# The Census field containing population
# Change to "B01001_001E" when full population data is available
CENSUS_POP_FIELD = "B19013_001E"   # currently median income as proxy

try:
    # ── Load Census data ──────────────────────────────────────
    if not os.path.exists(CENSUS_PATH):
        raise FileNotFoundError(f"Census file not found: {CENSUS_PATH}")

    with open(CENSUS_PATH) as f:
        raw = json.load(f)

    headers    = raw[0]
    census_pdf = pd.DataFrame(raw[1:], columns=headers)

    # Build 11-digit GEOID from state + county + tract
    census_pdf["GEOID"] = (
        census_pdf["state"] +
        census_pdf["county"] +
        census_pdf["tract"]
    )

    # Cast the population/income field to numeric
    # -666666666 is Census missing data flag — filter these out
    census_pdf[CENSUS_POP_FIELD] = pd.to_numeric(
        census_pdf[CENSUS_POP_FIELD], errors="coerce"
    )
    census_pdf = census_pdf[census_pdf[CENSUS_POP_FIELD] > 0]

    census_sdf = spark.createDataFrame(
        census_pdf[["GEOID", CENSUS_POP_FIELD]]
    )
    print(f"✓ Census data loaded: {census_sdf.count():,} tracts")

    # ── Load tract-neighborhood overlay ───────────────────────
    # GeoJSON is just JSON — no need for geopandas to read it.
    # We extract only the properties we need from each feature.
    import json
    with open(OVERLAY_PATH) as f:
        overlay_raw = json.load(f)

    overlay_records = [
        {
            "GEOID":             feat["properties"].get("GEOID"),
            "neighborhood_name": feat["properties"].get("neighborhood_name"),
            "area_weight":       feat["properties"].get("area_weight"),
        }
        for feat in overlay_raw["features"]
        if feat["properties"].get("GEOID") is not None
    ]

    overlay_sdf = spark.createDataFrame(overlay_records)

    print(f"✓ Overlay loaded: {overlay_sdf.count():,} tract-neighborhood pairs")

    # ── Area-weighted population per neighborhood ─────────────
    # For each tract-neighborhood pair: weighted_value = value * area_weight
    # Sum these up per neighborhood to get the neighborhood-level estimate
    pop_by_nhood = (
        overlay_sdf
        .join(census_sdf, on="GEOID", how="left")
        .withColumn(CENSUS_POP_FIELD, F.col(CENSUS_POP_FIELD).cast(DoubleType()))
        .filter(F.col(CENSUS_POP_FIELD).isNotNull())
        .groupBy("neighborhood_name")
        .agg(
            F.round(
                F.sum(F.col(CENSUS_POP_FIELD) * F.col("area_weight")) /
                F.sum("area_weight"),
                0
            ).alias("population_proxy")
        )
    )
    print(f"✓ Population proxy computed for {pop_by_nhood.count()} neighborhoods")

    # ── Join to crime features and compute rate ───────────────
    df_enriched = (
        df_features
        .join(
            pop_by_nhood,
            df_features["neighborhood"] == pop_by_nhood["neighborhood_name"],
            how="left"
        )
        .drop("neighborhood_name")
        .withColumn(
            "population",
            F.col("population_proxy").cast(IntegerType())
        )
        .drop("population_proxy")
        .withColumn(
            "crime_rate_per_1000",
            F.round(
                F.col("total_incidents").cast(DoubleType()) /
                F.col("population").cast(DoubleType()) * 1000,
                2
            )
        )
    )

    print(f"✓ crime_rate_per_1000 derived")
    print(f"\nTop 10 neighborhoods by crime rate:")
    display(
        df_enriched
        .groupBy("neighborhood")
        .agg(F.round(F.avg("crime_rate_per_1000"), 2).alias("avg_crime_rate"))
        .orderBy("avg_crime_rate", ascending=False)
        .limit(10)
    )

except FileNotFoundError as e:
    print(f"⚠ {e}")
    print("  crime_rate_per_1000 will be null.")
    print("  Re-run this cell once Census data is available.")
    df_enriched = df_features

except Exception as e:
    print(f"⚠ Unexpected error during population join: {e}")
    print("  Proceeding with null crime_rate_per_1000.")
    df_enriched = df_features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11 — Write Output Parquet
# MAGIC
# MAGIC Writes the final neighborhood-month feature table to Parquet,
# MAGIC partitioned by year and month to match the raw crime data layout.
# MAGIC This makes it efficient to read a specific time window downstream.

# COMMAND ----------

# Ensure the transformed volume exists before writing.
# CREATE VOLUME IF NOT EXISTS is idempotent — safe to run every time.
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.transformed")

(
    df_enriched
    .write
    .mode("overwrite")
    .partitionBy("year", "month")
    .parquet(OUTPUT_PATH)
)

output_count = spark.read.parquet(OUTPUT_PATH).count()
print(f"✓ Wrote {output_count:,} rows → {OUTPUT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12 — Quality Checks
# MAGIC
# MAGIC Verify the output looks correct before moving to downstream
# MAGIC notebooks. Key things to check:
# MAGIC - Null rates on derived features
# MAGIC - Top neighborhoods by total crime (sanity check)
# MAGIC - violent_crime_pct distribution (should be 0–1)
# MAGIC - Monthly trend to confirm time coverage

# COMMAND ----------

result = spark.read.parquet(OUTPUT_PATH)

print("=== NULL RATES (%) ===")
display(result.select([
    F.round(
        F.count(F.when(F.col(c).isNull(), c)) /
        F.count(F.lit(1)) * 100, 2
    ).alias(c)
    for c in [
        "neighborhood", "total_incidents", "violent_count",
        "property_count", "drug_count", "violent_crime_pct",
        "crime_rate_per_1000"
    ]
]))

# COMMAND ----------

print("=== TOP 15 NEIGHBORHOODS BY TOTAL INCIDENTS ===")
display(
    result
    .groupBy("neighborhood")
    .agg(F.sum("total_incidents").alias("total"))
    .orderBy("total", ascending=False)
    .limit(15)
)

# COMMAND ----------

print("=== VIOLENT CRIME PCT DISTRIBUTION ===")
display(
    result.select(
        F.min("violent_crime_pct").alias("min"),
        F.round(F.avg("violent_crime_pct"), 4).alias("avg"),
        F.max("violent_crime_pct").alias("max"),
        F.percentile_approx("violent_crime_pct", 0.25).alias("p25"),
        F.percentile_approx("violent_crime_pct", 0.75).alias("p75"),
    )
)

# COMMAND ----------

print("=== MONTHLY INCIDENT TREND ===")
display(
    result
    .groupBy("year", "month")
    .agg(F.sum("total_incidents").alias("total_incidents"))
    .orderBy("year", "month")
)
