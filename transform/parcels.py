"""
transform/parcels.py
STL Neighborhood Intelligence Pipeline — Parcel Data Transformation

Reads raw parcel Parquet and produces a neighborhood-level feature
table with vacancy and assessment metrics.

Key challenge: the parcel data stores neighborhood as a numeric code
(1–88) matching the NHD_NUM field in the city shapefile. We map these
to human-readable names using the NEIGHBORHOOD_LOOKUP table from
transform/geo.py before aggregating.

Transformations applied:
  1. Neighborhood code → name mapping
  2. Vacancy flag normalization (Y/N → boolean-like)
  3. Neighborhood-level aggregation
  4. Feature derivation:
       vacancy_rate       — vacant parcels / total parcels
       avg_assessed_value — mean assessed value per neighborhood

Output schema (neighborhood grain — one row per neighborhood):
  neighborhood_id    — numeric code
  neighborhood_name  — official name (e.g. "Soulard")
  total_parcels      — integer
  vacant_parcels     — integer
  occupied_parcels   — integer
  vacancy_rate       — double (0.0–1.0)
  avg_assessed_value — double
  avg_appraised_value— double
  total_assessed_value — double
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

logger = logging.getLogger(__name__)


def add_neighborhood_names(
    df: DataFrame,
    lookup: dict[int, str],
) -> DataFrame:
    """
    Map numeric neighborhood codes to official neighborhood names.

    The parcel DBF stores neighborhood as a numeric code (e.g. 35)
    matching the NHD_NUM field in the city shapefile. This function
    adds a neighborhood_name column using the lookup table from
    transform/geo.py.

    Args:
        df:     Parcel DataFrame with numeric neighborhood column.
        lookup: Dict mapping int code → name string.
                Use NEIGHBORHOOD_LOOKUP from transform/geo.py.

    Returns:
        DataFrame with neighborhood_id and neighborhood_name added.
    """
    # Build a Spark map from the Python dict so we can use it
    # as a column operation rather than a slow Python UDF.
    # F.create_map takes alternating key, value expressions.
    map_entries = []
    for code, name in lookup.items():
        map_entries.extend([F.lit(code), F.lit(name)])

    neighborhood_map = F.create_map(*map_entries)

    return (
        df
        # Cast neighborhood to integer for map lookup —
        # stored as string in Parquet but values are numeric
        .withColumn(
            "neighborhood_id",
            F.col("neighborhood").cast(IntegerType())
        )
        # Look up the name — returns null if code not in map
        .withColumn(
            "neighborhood_name",
            neighborhood_map[F.col("neighborhood_id")]
        )
        # Fill nulls with "Unknown" for any unmatched codes
        .withColumn(
            "neighborhood_name",
            F.coalesce(F.col("neighborhood_name"), F.lit("Unknown"))
        )
    )


def normalize_vacancy(df: DataFrame) -> DataFrame:
    """
    Normalize the is_vacant field from Y/N strings to a clean
    boolean-like integer flag (1=vacant, 0=occupied).

    The raw parcel data stores vacancy as "Y" or "N" strings.
    We convert to 1/0 integers so they can be summed directly
    in the aggregation step.

    Args:
        df: Parcel DataFrame with is_vacant as string column.

    Returns:
        DataFrame with is_vacant_flag integer column added.
    """
    return df.withColumn(
        "is_vacant_flag",
        F.when(F.upper(F.col("is_vacant")) == "Y", 1)
         .otherwise(0)
    )


def aggregate_to_neighborhood(df: DataFrame) -> DataFrame:
    """
    Aggregate parcel records to neighborhood grain.

    Computes parcel counts, vacancy counts, and average assessment
    values per neighborhood. Derives vacancy_rate as the primary
    output feature.

    Args:
        df: Parcel DataFrame with neighborhood_name and
            is_vacant_flag columns.

    Returns:
        Aggregated DataFrame at neighborhood grain.
    """
    return (
        df
        # Exclude parcels with unknown neighborhood
        .filter(F.col("neighborhood_name") != "Unknown")
        .groupBy("neighborhood_id", "neighborhood_name")
        .agg(
            # Total parcel count
            F.count("parcel_id").alias("total_parcels"),

            # Vacant parcel count — sum of is_vacant_flag (1=vacant)
            F.sum("is_vacant_flag").alias("vacant_parcels"),

            # Assessment value metrics
            # Filter $0 values from averages — these are city-owned
            # or exempt parcels that would skew the mean downward
            F.round(
                F.avg(
                    F.when(F.col("assessed_value") > 0,
                           F.col("assessed_value"))
                ), 2
            ).alias("avg_assessed_value"),

            F.round(
                F.avg(
                    F.when(F.col("appraised_value") > 0,
                           F.col("appraised_value"))
                ), 2
            ).alias("avg_appraised_value"),

            # Total assessed value — useful for tax base analysis
            F.round(
                F.sum("assessed_value"), 2
            ).alias("total_assessed_value"),
        )
        # Derive occupied count
        .withColumn(
            "occupied_parcels",
            F.col("total_parcels") - F.col("vacant_parcels")
        )
        # Derive vacancy_rate — the primary feature
        # vacant parcels / total parcels, rounded to 4 decimal places
        .withColumn(
            "vacancy_rate",
            F.round(
                F.col("vacant_parcels").cast(DoubleType()) /
                F.col("total_parcels").cast(DoubleType()),
                4
            )
        )
        .orderBy("neighborhood_name")
    )


def transform_parcels(
    spark: SparkSession,
    raw_path: str,
    output_path: str,
    neighborhood_lookup: dict[int, str],
) -> DataFrame:
    """
    Main entry point. Reads raw parcel Parquet, maps neighborhood
    codes to names, aggregates to neighborhood grain, derives
    vacancy_rate and assessment features, writes output Parquet.

    Args:
        spark:               Active SparkSession.
        raw_path:            Path to raw parcel Parquet.
        output_path:         Path to write transformed output.
        neighborhood_lookup: Dict mapping int code → name string.
                             Use NEIGHBORHOOD_LOOKUP from transform/geo.py.

    Returns:
        Transformed neighborhood-level parcel feature DataFrame.
    """
    logger.info(f"Reading raw parcel data from {raw_path}")
    df = spark.read.parquet(raw_path)
    logger.info(f"  Raw rows: {df.count():,}")

    df = add_neighborhood_names(df, neighborhood_lookup)
    df = normalize_vacancy(df)
    df = aggregate_to_neighborhood(df)

    spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.transformed")
    df.write.mode("overwrite").parquet(output_path)

    row_count = df.count()
    logger.info(f"  ✓ Wrote {row_count:,} rows → {output_path}")

    return df
