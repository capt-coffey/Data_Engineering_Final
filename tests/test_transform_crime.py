"""
tests/test_transform_crime.py
STL Neighborhood Intelligence — Crime Transform Unit Tests

Tests the transform/crime.py functions using small synthetic
DataFrames so tests run without needing the full Parquet dataset
or a production Spark session.

Test coverage:
  - Date standardization across all three known formats
  - Crime type classification for violent/property/drug/other
  - Aggregation produces correct row counts and column structure
  - violent_crime_pct is computed correctly
  - Unfounded and non-crime records are filtered before aggregation
  - Edge cases: null dates, unknown UCR categories, zero incidents
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType
)

# Adjust import path for Databricks vs local
try:
    from transform.crime import (
        standardize_dates,
        classify_crime_type,
        aggregate_to_neighborhood_month,
        VIOLENT_CATEGORIES,
        PROPERTY_CATEGORIES,
        DRUG_CATEGORIES,
    )
except ImportError:
    import sys
    sys.path.insert(0, "/Workspace/Users/jcoffey@wustl.edu/Data_Engineering_Final")
    from transform.crime import (
        standardize_dates,
        classify_crime_type,
        aggregate_to_neighborhood_month,
        VIOLENT_CATEGORIES,
        PROPERTY_CATEGORIES,
        DRUG_CATEGORIES,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    """
    Return the existing Spark session managed by Databricks.
    On Databricks serverless, SparkSession is pre-initialized and
    cannot be configured with .master() — we just retrieve it.
    """
    return SparkSession.builder.getOrCreate()


@pytest.fixture
def sample_crime_df(spark):
    """
    Synthetic crime DataFrame covering multiple date formats,
    crime types, and edge cases for testing.
    """
    data = [
        # (incident_id, occurred_date, neighborhood, ucr_category,
        #  year, month, is_unfounded, is_crime)

        # ISO format dates (2024+ NIBRS format)
        ("INC001", "2024-01-15", "Soulard",   "Aggravated Assault",                  2024, 1, None, None),
        ("INC002", "2024-01-20", "Soulard",   "Burglary/Breaking and Entering",       2024, 1, None, None),
        ("INC003", "2024-01-25", "Soulard",   "Drug/Narcotic Violations",             2024, 1, None, None),
        ("INC004", "2024-01-28", "Soulard",   "All Other Offenses",                   2024, 1, None, None),

        # Datetime format (older SLMPD format)
        ("INC005", "8/20/2024 12:00:00 AM", "Downtown", "Robbery",                   2024, 8, None, None),
        ("INC006", "8/21/2024 12:00:00 AM", "Downtown", "Motor Vehicle Theft",        2024, 8, None, None),

        # Zero-padded format
        ("INC007", "01/15/2024",            "The Hill",  "Simple Assault",            2024, 1, None, None),

        # Unfounded record — should be filtered out before aggregation
        ("INC008", "2024-01-10", "Soulard",  "Aggravated Assault",                   2024, 1, "1",  None),

        # Non-crime record — should be filtered out
        ("INC009", "2024-01-12", "Soulard",  "All Other Offenses",                   2024, 1, None, "0"),

        # Null date — should survive with null occurred_date_iso
        ("INC010", None,         "Downtown", "Robbery",                               2024, 1, None, None),
    ]

    schema = StructType([
        StructField("incident_id",   StringType(),  True),
        StructField("occurred_date", StringType(),  True),
        StructField("neighborhood",  StringType(),  True),
        StructField("ucr_category",  StringType(),  True),
        StructField("year",          IntegerType(), True),
        StructField("month",         IntegerType(), True),
        StructField("is_unfounded",  StringType(),  True),
        StructField("is_crime",      StringType(),  True),
    ])

    return spark.createDataFrame(data, schema)


# ── Date standardization tests ────────────────────────────────────────────────

class TestStandardizeDates:

    def test_iso_format_parsed(self, spark, sample_crime_df):
        """ISO dates like '2024-01-15' should parse correctly."""
        result = standardize_dates(sample_crime_df)
        row = result.filter(
            F.col("incident_id") == "INC001"
        ).select("occurred_date_iso").first()
        assert row["occurred_date_iso"] == "2024-01-15"

    def test_datetime_format_parsed(self, spark, sample_crime_df):
        """Datetime format '8/20/2024 12:00:00 AM' should parse correctly."""
        result = standardize_dates(sample_crime_df)
        row = result.filter(
            F.col("incident_id") == "INC005"
        ).select("occurred_date_iso").first()
        assert row["occurred_date_iso"] == "2024-08-20"

    def test_zero_padded_format_parsed(self, spark, sample_crime_df):
        """Zero-padded format '01/15/2024' should parse correctly."""
        result = standardize_dates(sample_crime_df)
        row = result.filter(
            F.col("incident_id") == "INC007"
        ).select("occurred_date_iso").first()
        assert row["occurred_date_iso"] == "2024-01-15"

    def test_null_date_returns_null(self, spark, sample_crime_df):
        """Null occurred_date should produce null occurred_date_iso."""
        result = standardize_dates(sample_crime_df)
        row = result.filter(
            F.col("incident_id") == "INC010"
        ).select("occurred_date_iso").first()
        assert row["occurred_date_iso"] is None

    def test_day_of_week_added(self, spark, sample_crime_df):
        """day_of_week column should be added (1=Sunday through 7=Saturday)."""
        result = standardize_dates(sample_crime_df)
        assert "day_of_week" in result.columns

    def test_day_of_week_range(self, spark, sample_crime_df):
        """day_of_week values should be 1–7."""
        result = standardize_dates(sample_crime_df)
        invalid = result.filter(
            F.col("day_of_week").isNotNull() &
            ((F.col("day_of_week") < 1) | (F.col("day_of_week") > 7))
        ).count()
        assert invalid == 0

    def test_original_date_column_preserved(self, spark, sample_crime_df):
        """The raw occurred_date column should still exist after transform."""
        result = standardize_dates(sample_crime_df)
        assert "occurred_date" in result.columns

    def test_row_count_unchanged(self, spark, sample_crime_df):
        """Date standardization should not add or remove rows."""
        result = standardize_dates(sample_crime_df)
        assert result.count() == sample_crime_df.count()


# ── Crime type classification tests ──────────────────────────────────────────

class TestClassifyCrimeType:

    def test_violent_category_classified(self, spark, sample_crime_df):
        """Aggravated Assault should classify as violent."""
        result = classify_crime_type(sample_crime_df)
        row = result.filter(
            F.col("incident_id") == "INC001"
        ).select("crime_type").first()
        assert row["crime_type"] == "violent"

    def test_property_category_classified(self, spark, sample_crime_df):
        """Burglary should classify as property."""
        result = classify_crime_type(sample_crime_df)
        row = result.filter(
            F.col("incident_id") == "INC002"
        ).select("crime_type").first()
        assert row["crime_type"] == "property"

    def test_drug_category_classified(self, spark, sample_crime_df):
        """Drug/Narcotic Violations should classify as drug."""
        result = classify_crime_type(sample_crime_df)
        row = result.filter(
            F.col("incident_id") == "INC003"
        ).select("crime_type").first()
        assert row["crime_type"] == "drug"

    def test_unknown_category_classified_as_other(self, spark, sample_crime_df):
        """Unknown UCR categories should classify as other."""
        result = classify_crime_type(sample_crime_df)
        row = result.filter(
            F.col("incident_id") == "INC004"
        ).select("crime_type").first()
        assert row["crime_type"] == "other"

    def test_robbery_classified_as_violent(self, spark, sample_crime_df):
        """Robbery should classify as violent."""
        result = classify_crime_type(sample_crime_df)
        row = result.filter(
            F.col("incident_id") == "INC005"
        ).select("crime_type").first()
        assert row["crime_type"] == "violent"

    def test_motor_vehicle_theft_classified_as_property(self, spark, sample_crime_df):
        """Motor Vehicle Theft should classify as property."""
        result = classify_crime_type(sample_crime_df)
        row = result.filter(
            F.col("incident_id") == "INC006"
        ).select("crime_type").first()
        assert row["crime_type"] == "property"

    def test_crime_type_column_added(self, spark, sample_crime_df):
        """crime_type column should be added to the DataFrame."""
        result = classify_crime_type(sample_crime_df)
        assert "crime_type" in result.columns

    def test_no_null_crime_types(self, spark, sample_crime_df):
        """Every row should have a non-null crime_type."""
        result = classify_crime_type(sample_crime_df)
        null_count = result.filter(F.col("crime_type").isNull()).count()
        assert null_count == 0

    def test_only_valid_crime_types(self, spark, sample_crime_df):
        """crime_type values should only be violent/property/drug/other."""
        result = classify_crime_type(sample_crime_df)
        valid = {"violent", "property", "drug", "other"}
        values = {
            r["crime_type"]
            for r in result.select("crime_type").distinct().collect()
        }
        assert values.issubset(valid)

    def test_all_violent_categories_covered(self, spark):
        """Every category in VIOLENT_CATEGORIES should map to violent."""
        data = [(cat,) for cat in VIOLENT_CATEGORIES]
        df   = spark.createDataFrame(data, ["ucr_category"])
        result = classify_crime_type(df)
        non_violent = result.filter(
            F.col("crime_type") != "violent"
        ).count()
        assert non_violent == 0

    def test_all_drug_categories_covered(self, spark):
        """Every category in DRUG_CATEGORIES should map to drug."""
        data = [(cat,) for cat in DRUG_CATEGORIES]
        df   = spark.createDataFrame(data, ["ucr_category"])
        result = classify_crime_type(df)
        non_drug = result.filter(
            F.col("crime_type") != "drug"
        ).count()
        assert non_drug == 0


# ── Aggregation tests ─────────────────────────────────────────────────────────

class TestAggregateToNeighborhoodMonth:

    @pytest.fixture
    def classified_df(self, spark, sample_crime_df):
        """Classified DataFrame ready for aggregation."""
        return classify_crime_type(sample_crime_df)

    def test_unfounded_records_filtered(self, spark, classified_df):
        """Records with is_unfounded='1' should not appear in aggregation."""
        result = aggregate_to_neighborhood_month(classified_df)
        # INC008 was unfounded — Soulard Jan 2024 should have 3 incidents
        # (INC001=violent, INC002=property, INC003=drug) not 4
        soulard_jan = result.filter(
            (F.col("neighborhood") == "Soulard") &
            (F.col("year") == 2024) &
            (F.col("month") == 1)
        ).select("total_incidents").first()
        # INC004 (other) + INC001 (violent) + INC002 (property) + INC003 (drug)
        # minus INC008 (unfounded) minus INC009 (is_crime=0) = 4 records
        assert soulard_jan["total_incidents"] == 4

    def test_non_crime_records_filtered(self, spark, classified_df):
        """Records with is_crime='0' should not appear in aggregation."""
        result = aggregate_to_neighborhood_month(classified_df)
        # INC009 had is_crime='0' — verify it's excluded
        # Soulard Jan 2024: INC001, INC002, INC003, INC004 = 4 (INC008 unfounded, INC009 non-crime)
        soulard_jan = result.filter(
            (F.col("neighborhood") == "Soulard") &
            (F.col("year") == 2024) &
            (F.col("month") == 1)
        ).select("total_incidents").first()
        assert soulard_jan["total_incidents"] == 4

    def test_output_columns_present(self, spark, classified_df):
        """All expected output columns should be present."""
        result = aggregate_to_neighborhood_month(classified_df)
        expected = [
            "neighborhood", "year", "month",
            "total_incidents", "violent_count", "property_count",
            "drug_count", "other_count", "violent_crime_pct",
            "crime_rate_per_1000", "population",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_type_counts_sum_to_total(self, spark, classified_df):
        """violent + property + drug + other should equal total_incidents."""
        result = aggregate_to_neighborhood_month(classified_df)
        mismatch = result.filter(
            (F.col("violent_count") +
             F.col("property_count") +
             F.col("drug_count") +
             F.col("other_count")) != F.col("total_incidents")
        ).count()
        assert mismatch == 0

    def test_violent_crime_pct_range(self, spark, classified_df):
        """violent_crime_pct should be between 0 and 1 inclusive."""
        result = aggregate_to_neighborhood_month(classified_df)
        out_of_range = result.filter(
            (F.col("violent_crime_pct") < 0) |
            (F.col("violent_crime_pct") > 1)
        ).count()
        assert out_of_range == 0

    def test_violent_crime_pct_calculation(self, spark, classified_df):
        """violent_crime_pct should equal violent_count / total_incidents."""
        result = aggregate_to_neighborhood_month(classified_df)
        # For Soulard Jan 2024: 1 violent out of 4 total = 0.25
        soulard_jan = result.filter(
            (F.col("neighborhood") == "Soulard") &
            (F.col("year") == 2024) &
            (F.col("month") == 1)
        ).select("violent_crime_pct").first()
        assert abs(soulard_jan["violent_crime_pct"] - 0.25) < 0.001

    def test_crime_rate_per_1000_starts_null(self, spark, classified_df):
        """crime_rate_per_1000 should be null before population join."""
        result = aggregate_to_neighborhood_month(classified_df)
        non_null = result.filter(
            F.col("crime_rate_per_1000").isNotNull()
        ).count()
        assert non_null == 0

    def test_grain_is_neighborhood_year_month(self, spark, classified_df):
        """Output should have one row per neighborhood-year-month combination."""
        result = aggregate_to_neighborhood_month(classified_df)
        total   = result.count()
        distinct = result.select(
            "neighborhood", "year", "month"
        ).distinct().count()
        assert total == distinct


# ── Category set integrity tests ──────────────────────────────────────────────

class TestCategorySetIntegrity:

    def test_no_overlap_between_violent_and_property(self):
        """A category should not appear in both violent and property sets."""
        overlap = VIOLENT_CATEGORIES & PROPERTY_CATEGORIES
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_no_overlap_between_violent_and_drug(self):
        """A category should not appear in both violent and drug sets."""
        overlap = VIOLENT_CATEGORIES & DRUG_CATEGORIES
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_no_overlap_between_property_and_drug(self):
        """A category should not appear in both property and drug sets."""
        overlap = PROPERTY_CATEGORIES & DRUG_CATEGORIES
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_all_categories_are_strings(self):
        """All category values should be non-empty strings."""
        all_cats = VIOLENT_CATEGORIES | PROPERTY_CATEGORIES | DRUG_CATEGORIES
        for cat in all_cats:
            assert isinstance(cat, str) and len(cat) > 0
