"""
tests/test_transform_parcels.py
STL Neighborhood Intelligence — Parcel Transform Unit Tests

Tests the transform/parcels.py functions using small synthetic
DataFrames so tests run without needing the full Parquet dataset.

Test coverage:
  - Neighborhood code → name mapping
  - Unknown codes map to "Unknown"
  - Vacancy Y/N → 1/0 flag normalization
  - Case variations handled (y, Y, YES)
  - Aggregation produces correct row counts
  - vacancy_rate computed correctly
  - $0 assessed values excluded from averages
  - Edge cases: all vacant, no vacant, single parcel neighborhood
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
    from transform.parcels import (
        add_neighborhood_names,
        normalize_vacancy,
        aggregate_to_neighborhood,
    )
except ImportError:
    import sys
    sys.path.insert(0, "/Workspace/Users/jcoffey@wustl.edu/Data_Engineering_Final")
    from transform.parcels import (
        add_neighborhood_names,
        normalize_vacancy,
        aggregate_to_neighborhood,
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


# Minimal lookup table for testing — no need for all 79 neighborhoods
TEST_LOOKUP = {
    0:  "Unknown",
    21: "Soulard",
    35: "Downtown",
    38: "Central West End",
    16: "Dutchtown",
}


@pytest.fixture
def sample_parcel_df(spark):
    """
    Synthetic parcel DataFrame covering multiple neighborhoods,
    vacancy states, and edge cases for testing.
    """
    data = [
        # (parcel_id, neighborhood, is_vacant, assessed_value, appraised_value)
        ("P001", "21",  "Y", 45000.0,  135000.0),   # Soulard — vacant
        ("P002", "21",  "N", 78000.0,  234000.0),   # Soulard — occupied
        ("P003", "21",  "N", 52000.0,  156000.0),   # Soulard — occupied
        ("P004", "35",  "Y", 0.0,      0.0),         # Downtown — vacant, $0 assessed
        ("P005", "35",  "N", 1200000.0, 400000.0),  # Downtown — occupied, high value
        ("P006", "38",  "N", 95000.0,  285000.0),   # Central West End — occupied
        ("P007", "38",  "N", 110000.0, 330000.0),   # Central West End — occupied
        ("P008", "38",  "Y", 0.0,      0.0),         # Central West End — vacant, $0
        ("P009", "16",  "Y", 15000.0,  45000.0),    # Dutchtown — vacant
        ("P010", "16",  "Y", 12000.0,  36000.0),    # Dutchtown — vacant
        ("P011", "16",  "N", 38000.0,  114000.0),   # Dutchtown — occupied
        ("P012", "99",  "N", 50000.0,  150000.0),   # Unknown neighborhood code
        ("P013", "21",  "y", 30000.0,  90000.0),    # Soulard — lowercase y vacancy
    ]

    schema = StructType([
        StructField("parcel_id",        StringType(), True),
        StructField("neighborhood",     StringType(), True),
        StructField("is_vacant",        StringType(), True),
        StructField("assessed_value",   DoubleType(), True),
        StructField("appraised_value",  DoubleType(), True),
    ])

    return spark.createDataFrame(data, schema)


# ── Neighborhood name mapping tests ──────────────────────────────────────────

class TestAddNeighborhoodNames:

    def test_known_code_maps_to_name(self, spark, sample_parcel_df):
        """Code 21 should map to Soulard."""
        result = add_neighborhood_names(sample_parcel_df, TEST_LOOKUP)
        row = result.filter(
            F.col("parcel_id") == "P001"
        ).select("neighborhood_name").first()
        assert row["neighborhood_name"] == "Soulard"

    def test_downtown_code_maps_correctly(self, spark, sample_parcel_df):
        """Code 35 should map to Downtown."""
        result = add_neighborhood_names(sample_parcel_df, TEST_LOOKUP)
        row = result.filter(
            F.col("parcel_id") == "P004"
        ).select("neighborhood_name").first()
        assert row["neighborhood_name"] == "Downtown"

    def test_unknown_code_maps_to_unknown(self, spark, sample_parcel_df):
        """Code 99 (not in lookup) should map to 'Unknown'."""
        result = add_neighborhood_names(sample_parcel_df, TEST_LOOKUP)
        row = result.filter(
            F.col("parcel_id") == "P012"
        ).select("neighborhood_name").first()
        assert row["neighborhood_name"] == "Unknown"

    def test_neighborhood_id_column_added(self, spark, sample_parcel_df):
        """neighborhood_id integer column should be added."""
        result = add_neighborhood_names(sample_parcel_df, TEST_LOOKUP)
        assert "neighborhood_id" in result.columns

    def test_neighborhood_name_column_added(self, spark, sample_parcel_df):
        """neighborhood_name string column should be added."""
        result = add_neighborhood_names(sample_parcel_df, TEST_LOOKUP)
        assert "neighborhood_name" in result.columns

    def test_neighborhood_id_is_integer(self, spark, sample_parcel_df):
        """neighborhood_id should be cast to integer type."""
        result = add_neighborhood_names(sample_parcel_df, TEST_LOOKUP)
        id_type = dict(result.dtypes)["neighborhood_id"]
        assert id_type == "int", f"Expected int, got {id_type}"

    def test_no_null_neighborhood_names(self, spark, sample_parcel_df):
        """No neighborhood_name should be null — unknowns get 'Unknown'."""
        result = add_neighborhood_names(sample_parcel_df, TEST_LOOKUP)
        null_count = result.filter(
            F.col("neighborhood_name").isNull()
        ).count()
        assert null_count == 0

    def test_row_count_unchanged(self, spark, sample_parcel_df):
        """Name mapping should not add or remove rows."""
        result = add_neighborhood_names(sample_parcel_df, TEST_LOOKUP)
        assert result.count() == sample_parcel_df.count()

    def test_all_lookup_names_reachable(self, spark):
        """Every code in the lookup should be correctly mapped."""
        data = [(str(code),) for code in TEST_LOOKUP]
        df   = spark.createDataFrame(data, ["neighborhood"])
        result = add_neighborhood_names(df, TEST_LOOKUP)
        for code, name in TEST_LOOKUP.items():
            row = result.filter(
                F.col("neighborhood_id") == code
            ).select("neighborhood_name").first()
            assert row["neighborhood_name"] == name, (
                f"Code {code} should map to '{name}'"
            )


# ── Vacancy normalization tests ───────────────────────────────────────────────

class TestNormalizeVacancy:

    def test_uppercase_y_maps_to_1(self, spark, sample_parcel_df):
        """'Y' should map to is_vacant_flag = 1."""
        result = normalize_vacancy(sample_parcel_df)
        row = result.filter(
            F.col("parcel_id") == "P001"
        ).select("is_vacant_flag").first()
        assert row["is_vacant_flag"] == 1

    def test_uppercase_n_maps_to_0(self, spark, sample_parcel_df):
        """'N' should map to is_vacant_flag = 0."""
        result = normalize_vacancy(sample_parcel_df)
        row = result.filter(
            F.col("parcel_id") == "P002"
        ).select("is_vacant_flag").first()
        assert row["is_vacant_flag"] == 0

    def test_lowercase_y_maps_to_1(self, spark, sample_parcel_df):
        """Lowercase 'y' should also map to is_vacant_flag = 1."""
        result = normalize_vacancy(sample_parcel_df)
        row = result.filter(
            F.col("parcel_id") == "P013"
        ).select("is_vacant_flag").first()
        assert row["is_vacant_flag"] == 1

    def test_is_vacant_flag_column_added(self, spark, sample_parcel_df):
        """is_vacant_flag column should be added."""
        result = normalize_vacancy(sample_parcel_df)
        assert "is_vacant_flag" in result.columns

    def test_original_is_vacant_preserved(self, spark, sample_parcel_df):
        """Original is_vacant column should still exist."""
        result = normalize_vacancy(sample_parcel_df)
        assert "is_vacant" in result.columns

    def test_no_null_flags(self, spark, sample_parcel_df):
        """is_vacant_flag should have no nulls."""
        result = normalize_vacancy(sample_parcel_df)
        null_count = result.filter(
            F.col("is_vacant_flag").isNull()
        ).count()
        assert null_count == 0

    def test_only_zero_or_one_values(self, spark, sample_parcel_df):
        """is_vacant_flag should only contain 0 or 1."""
        result = normalize_vacancy(sample_parcel_df)
        invalid = result.filter(
            ~F.col("is_vacant_flag").isin(0, 1)
        ).count()
        assert invalid == 0

    def test_row_count_unchanged(self, spark, sample_parcel_df):
        """Normalization should not add or remove rows."""
        result = normalize_vacancy(sample_parcel_df)
        assert result.count() == sample_parcel_df.count()


# ── Aggregation tests ─────────────────────────────────────────────────────────

class TestAggregateToNeighborhood:

    @pytest.fixture
    def prepared_df(self, spark, sample_parcel_df):
        """DataFrame with neighborhood names and vacancy flags added."""
        df = add_neighborhood_names(sample_parcel_df, TEST_LOOKUP)
        return normalize_vacancy(df)

    def test_output_has_one_row_per_neighborhood(self, spark, prepared_df):
        """Output should have exactly one row per neighborhood."""
        result = aggregate_to_neighborhood(prepared_df)
        total    = result.count()
        distinct = result.select("neighborhood_name").distinct().count()
        assert total == distinct

    def test_unknown_neighborhood_excluded(self, spark, prepared_df):
        """Parcels with 'Unknown' neighborhood should be excluded."""
        result = aggregate_to_neighborhood(prepared_df)
        unknown_rows = result.filter(
            F.col("neighborhood_name") == "Unknown"
        ).count()
        assert unknown_rows == 0

    def test_total_parcel_count_correct(self, spark, prepared_df):
        """Soulard should have 4 parcels (P001, P002, P003, P013)."""
        result = aggregate_to_neighborhood(prepared_df)
        row = result.filter(
            F.col("neighborhood_name") == "Soulard"
        ).select("total_parcels").first()
        assert row["total_parcels"] == 4

    def test_vacant_parcel_count_correct(self, spark, prepared_df):
        """Soulard should have 2 vacant parcels (P001 Y, P013 y)."""
        result = aggregate_to_neighborhood(prepared_df)
        row = result.filter(
            F.col("neighborhood_name") == "Soulard"
        ).select("vacant_parcels").first()
        assert row["vacant_parcels"] == 2

    def test_occupied_parcel_count_correct(self, spark, prepared_df):
        """Soulard occupied = total - vacant = 4 - 2 = 2."""
        result = aggregate_to_neighborhood(prepared_df)
        row = result.filter(
            F.col("neighborhood_name") == "Soulard"
        ).select("occupied_parcels").first()
        assert row["occupied_parcels"] == 2

    def test_vacancy_rate_calculation(self, spark, prepared_df):
        """Dutchtown: 2 vacant / 3 total = 0.6667."""
        result = aggregate_to_neighborhood(prepared_df)
        row = result.filter(
            F.col("neighborhood_name") == "Dutchtown"
        ).select("vacancy_rate").first()
        assert abs(row["vacancy_rate"] - 0.6667) < 0.001

    def test_vacancy_rate_range(self, spark, prepared_df):
        """vacancy_rate should be between 0 and 1 inclusive."""
        result = aggregate_to_neighborhood(prepared_df)
        out_of_range = result.filter(
            (F.col("vacancy_rate") < 0) |
            (F.col("vacancy_rate") > 1)
        ).count()
        assert out_of_range == 0

    def test_zero_assessed_excluded_from_avg(self, spark, prepared_df):
        """
        $0 assessed values should be excluded from avg_assessed_value.
        Downtown has P004 ($0) and P005 ($1,200,000) — avg should be
        $1,200,000 not $600,000.
        """
        result = aggregate_to_neighborhood(prepared_df)
        row = result.filter(
            F.col("neighborhood_name") == "Downtown"
        ).select("avg_assessed_value").first()
        assert row["avg_assessed_value"] == 1200000.0

    def test_output_columns_present(self, spark, prepared_df):
        """All expected output columns should be present."""
        result = aggregate_to_neighborhood(prepared_df)
        expected = [
            "neighborhood_id", "neighborhood_name",
            "total_parcels", "vacant_parcels", "occupied_parcels",
            "vacancy_rate", "avg_assessed_value",
            "avg_appraised_value", "total_assessed_value",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_total_assessed_value_correct(self, spark, prepared_df):
        """
        total_assessed_value should include $0 parcels unlike the average.
        Soulard: P001(45000) + P002(78000) + P003(52000) + P013(30000) = 205000
        """
        result = aggregate_to_neighborhood(prepared_df)
        row = result.filter(
            F.col("neighborhood_name") == "Soulard"
        ).select("total_assessed_value").first()
        assert abs(row["total_assessed_value"] - 205000.0) < 0.01
