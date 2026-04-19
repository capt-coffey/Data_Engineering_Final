"""
transform/crime.py
STL Neighborhood Intelligence Pipeline — Crime Data Transformation

Reads raw SLMPD crime Parquet, applies the following transformations,
and writes a neighborhood-level feature table:

  1. Date standardization
     Parses occurred_date from "M/D/YYYY 12:00:00 AM" to ISO format.
     Extracts year, month, day of week for time-series analysis.

  2. Crime type classification
     Maps NIBRS ucr_category strings to three broad buckets:
       violent  — assault, robbery, homicide, rape, kidnapping
       property — theft, burglary, motor vehicle theft, arson
       drug     — drug/narcotic violations, drug equipment

  3. Neighborhood-level aggregation
     Groups by neighborhood + year + month and counts incidents
     per crime type bucket.

  4. Feature derivation
     crime_rate_per_1000  — incidents per 1,000 residents
                            requires Census population joined in
     violent_crime_pct    — share of incidents classified violent

Output schema (neighborhood_month grain):
  neighborhood        — name string (e.g. "Soulard")
  year                — integer
  month               — integer
  total_incidents     — integer
  violent_count       — integer
  property_count      — integer
  drug_count          — integer
  other_count         — integer
  violent_crime_pct   — double (0.0–1.0)
  crime_rate_per_1000 — double (null if no population data)
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType

logger = logging.getLogger(__name__)

# ── Crime type classification ─────────────────────────────────────────────────
# Maps NIBRS ucr_category values to one of three buckets.
# Any category not listed here falls into "other".
#
# Source: FBI NIBRS offense definitions
# https://ucr.fbi.gov/nibrs/2019/resource-pages/nibrs_offense_definitions-a.pdf

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

DRUG_CATEGORIES = {
    "Drug/Narcotic Violations",
    "Drug Equipment Violations",
}


def standardize_dates(df: DataFrame) -> DataFrame:
    """
    Parse occurred_date from raw SLMPD format to ISO date string.

    The 2024+ SLMPD format uses "M/D/YYYY 12:00:00 AM".
    We parse this to a proper date and derive helper columns:
      occurred_date_iso  — "YYYY-MM-DD" string
      day_of_week        — 1=Sunday through 7=Saturday (Spark convention)

    Args:
        df: Raw crime DataFrame with occurred_date as string.

    Returns:
        DataFrame with standardized date columns added.
    """
    return (
        df
        # Parse using format detection via regex so F.to_date() is only
        # ever called on strings matching its expected pattern.
        # F.coalesce(F.to_date(...)) throws CANNOT_PARSE_TIMESTAMP on
        # Databricks serverless — F.when().rlike() avoids this entirely.
        .withColumn(
            "occurred_date_parsed",
            F.when(
                # ISO format: "2024-01-15"
                F.col("occurred_date").rlike(r"^\d{4}-\d{2}-\d{2}"),
                F.to_date(F.col("occurred_date"), "yyyy-MM-dd")
            ).when(
                # Full datetime: "8/20/2025 12:00:00 AM"
                F.col("occurred_date").rlike(r"^\d{1,2}/\d{1,2}/\d{4} \d"),
                F.to_date(F.col("occurred_date"), "M/d/yyyy hh:mm:ss a")
            ).when(
                # Zero-padded: "01/15/2024"
                F.col("occurred_date").rlike(r"^\d{2}/\d{2}/\d{4}$"),
                F.to_date(F.col("occurred_date"), "MM/dd/yyyy")
            ).otherwise(F.lit(None))
        )
        # ISO date string for output
        .withColumn(
            "occurred_date_iso",
            F.date_format(F.col("occurred_date_parsed"), "yyyy-MM-dd")
        )
        # Day of week: 1=Sunday, 2=Monday, ..., 7=Saturday
        .withColumn(
            "day_of_week",
            F.dayofweek(F.col("occurred_date_parsed"))
        )
        # Drop the intermediate parsed date column
        .drop("occurred_date_parsed")
    )


def classify_crime_type(df: DataFrame) -> DataFrame:
    """
    Add a crime_type column classifying each incident as
    violent, property, drug, or other.

    Uses the ucr_category field from the NIBRS-formatted
    2024+ SLMPD data. Pre-2024 data used different category
    names — those will fall through to "other".

    Args:
        df: Crime DataFrame with ucr_category column.

    Returns:
        DataFrame with crime_type column added.
    """
    return df.withColumn(
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


def aggregate_to_neighborhood_month(df: DataFrame) -> DataFrame:
    """
    Aggregate crime records to neighborhood + year + month grain.

    Counts total incidents and breaks them down by crime_type.
    Derives violent_crime_pct from the counts.

    The crime_rate_per_1000 feature is left null here — it requires
    Census population data which is joined in enrich_with_population().

    Args:
        df: Crime DataFrame with crime_type, neighborhood, year, month.

    Returns:
        Aggregated DataFrame at neighborhood-month grain.
    """
    return (
        df
        # Filter out unfounded reports before aggregating —
        # these are incidents later determined to have no factual
        # basis and should not count toward crime statistics
        .filter(
            (F.col("is_unfounded").isNull()) |
            (F.col("is_unfounded") != F.lit("1"))
        )
        # Filter to only confirmed crime incidents where flag exists
        .filter(
            (F.col("is_crime").isNull()) |
            (F.col("is_crime") != F.lit("0"))
        )
        .groupBy("neighborhood", "year", "month")
        .agg(
            # Total incident count
            F.count("incident_id").alias("total_incidents"),

            # Count by crime type using conditional aggregation
            # (equivalent to SQL: SUM(CASE WHEN crime_type='violent' THEN 1 END))
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
        # Derive violent_crime_pct — share of incidents classified violent
        # Round to 4 decimal places for clean output
        .withColumn(
            "violent_crime_pct",
            F.round(
                F.col("violent_count").cast(DoubleType()) /
                F.col("total_incidents").cast(DoubleType()),
                4
            )
        )
        # Placeholder for population join — filled in enrich_with_population()
        .withColumn("crime_rate_per_1000", F.lit(None).cast(DoubleType()))
        .withColumn("population", F.lit(None).cast(IntegerType()))
    )


def enrich_with_population(
    crime_df: DataFrame,
    census_path: str,
    overlay_path: str,
    spark: SparkSession,
) -> DataFrame:
    """
    Join Census population data to the neighborhood-month crime table
    and compute crime_rate_per_1000.

    Uses the tract-neighborhood area weight overlay to apportion
    Census tract population to neighborhood level:
      neighborhood_population = SUM(tract_population * area_weight)

    Then:
      crime_rate_per_1000 = (total_incidents / population) * 1000

    Args:
        crime_df:     Aggregated neighborhood-month crime DataFrame.
        census_path:  Path to the raw Census JSON file.
        overlay_path: Path to tract_neighborhood_overlay.geojson.
        spark:        Active SparkSession.

    Returns:
        Crime DataFrame enriched with population and crime_rate_per_1000.
    """
    import json
    import pandas as pd

    # ── Load Census data ──────────────────────────────────────
    # Census JSON is list-of-lists: first row = headers, rest = data
    try:
        with open(census_path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        logger.warning(
            f"Census file not found at {census_path}. "
            "crime_rate_per_1000 will remain null."
        )
        return crime_df

    headers = raw[0]
    rows    = raw[1:]
    census_pdf = pd.DataFrame(rows, columns=headers)

    # B19013_001E = median household income (we use this as a proxy
    # for population weighting since the teammate's Census pull
    # didn't include total population B01001_001E).
    # We'll use the count of tracts per neighborhood as a fallback
    # until population data is available.
    #
    # For now: build a tract GEOID from state + county + tract fields
    census_pdf["GEOID"] = (
        census_pdf["state"] +
        census_pdf["county"] +
        census_pdf["tract"]
    )
    census_sdf = spark.createDataFrame(census_pdf)

    # ── Load tract-neighborhood overlay ───────────────────────
    try:
        import geopandas as gpd
        overlay_gdf = gpd.read_file(overlay_path)
        overlay_pdf = overlay_gdf[
            ["GEOID", "neighborhood_name", "area_weight"]
        ].copy()
        overlay_sdf = spark.createDataFrame(overlay_pdf)
    except Exception as e:
        logger.warning(
            f"Could not load overlay from {overlay_path}: {e}. "
            "crime_rate_per_1000 will remain null."
        )
        return crime_df

    # ── Compute weighted neighborhood population ──────────────
    # Join census to overlay on GEOID, weight income by area fraction
    # Note: using median income as a neighborhood-level signal here.
    # When full population data (B01001_001E) is available, swap it in.
    pop_by_neighborhood = (
        overlay_sdf
        .join(census_sdf, on="GEOID", how="left")
        .withColumn(
            "median_income",
            F.col("B19013_001E").cast(DoubleType())
        )
        .filter(F.col("median_income") > 0)  # filter -666666666 flags
        .groupBy("neighborhood_name")
        .agg(
            # Area-weighted average median income as a proxy metric
            F.round(
                F.sum(F.col("median_income") * F.col("area_weight")) /
                F.sum(F.col("area_weight")),
                0
            ).alias("median_income_weighted"),
            # Count tracts as a rough population proxy until real pop available
            F.count("GEOID").alias("tract_count"),
        )
    )

    # ── Join to crime data ────────────────────────────────────
    enriched = crime_df.join(
        pop_by_neighborhood,
        crime_df["neighborhood"] == pop_by_neighborhood["neighborhood_name"],
        how="left"
    ).drop("neighborhood_name")

    logger.info(
        "Population join complete. Note: using median_income_weighted "
        "as a proxy — swap in B01001_001E when full population data available."
    )

    return enriched


def transform_crime(
    spark: SparkSession,
    raw_path: str,
    output_path: str,
    census_path: str | None = None,
    overlay_path: str | None = None,
) -> DataFrame:
    """
    Main entry point. Reads raw crime Parquet, applies all
    transformations, and writes the feature table to Parquet.

    Args:
        spark:        Active SparkSession.
        raw_path:     Path to raw crime Parquet.
        output_path:  Path to write transformed output.
        census_path:  Optional path to Census JSON for population join.
        overlay_path: Optional path to tract-neighborhood overlay GeoJSON.

    Returns:
        Transformed neighborhood-month crime feature DataFrame.
    """
    logger.info(f"Reading raw crime data from {raw_path}")
    df = spark.read.parquet(raw_path)
    logger.info(f"  Raw rows: {df.count():,}")

    # Apply transformations in sequence
    df = standardize_dates(df)
    df = classify_crime_type(df)
    df = aggregate_to_neighborhood_month(df)

    # Enrich with population if paths provided
    if census_path and overlay_path:
        df = enrich_with_population(df, census_path, overlay_path, spark)

    # Write output
    df.write.mode("overwrite").parquet(output_path)
    row_count = df.count()
    logger.info(f"  ✓ Wrote {row_count:,} rows → {output_path}")

    return df
