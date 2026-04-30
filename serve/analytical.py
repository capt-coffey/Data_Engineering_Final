# Databricks notebook source
# MAGIC %md
# MAGIC ## Cell 1 — Imports & Logging
# MAGIC This cell loads everything the notebook needs. Nothing new here compared to our other notebooks — pyspark.sql.functions as F is the core Spark column operations library, and logging gives us timestamped output in the cell panel.

# COMMAND ----------

import logging
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("stl_analytical_serve")

print("✓ Imports loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Configuration
# MAGIC All paths in one place. The key design decision here is the three-layer separation: raw/ is what we ingested, transformed/ is what we cleaned and derived features from, and serving/ is what we're building now — optimized for consumption. Anyone running this notebook only needs to edit this cell if their paths differ.

# COMMAND ----------

# ── Input paths ───────────────────────────────────────────────
# These point to the outputs of the transform notebooks.
# Crime is at neighborhood-month grain (2,319 rows).
# Parcels is already at neighborhood grain (79 rows).
# Weather is at daily grain (1,627 rows) — city-wide, one station.
CRIME_PATH   = "/Volumes/workspace/default/transformed/crime"
PARCELS_PATH = "/Volumes/workspace/default/transformed/parcels"
WEATHER_PATH = "/Volumes/workspace/default/raw/weather"

# ── Output paths ──────────────────────────────────────────────
# Serving tables land in a dedicated /serving/ volume.
# Keeping raw, transformed, and serving data in separate volumes
# makes it clear what stage of the pipeline each dataset is at.
SERVING_BASE = "/Volumes/workspace/default/serving"

# Purpose 1: one row per neighborhood, all metrics combined.
# Used for neighborhood comparison and dashboard views.
SUMMARY_PATH = f"{SERVING_BASE}/neighborhood_summary"

# Purpose 2: same data plus livability scores, sorted by rank.
# Used for ranking analysis and presentation.
RANKING_PATH = f"{SERVING_BASE}/neighborhood_ranking"

print("✓ Configuration set")
print(f"  Crime input:    {CRIME_PATH}")
print(f"  Parcels input:  {PARCELS_PATH}")
print(f"  Weather input:  {WEATHER_PATH}")
print(f"  Summary output: {SUMMARY_PATH}")
print(f"  Ranking output: {RANKING_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Load Transformed Tables
# MAGIC Reads all three input tables and prints a quick summary to confirm they loaded correctly before we do any work. If any of these fail it means the transform notebooks haven't been run yet or the paths are wrong — better to catch that here than have a cryptic error halfway through the notebook.

# COMMAND ----------

# Read all three transformed tables from Parquet.
# Spark automatically discovers partitions (year=/month=) for crime
# and weather — no need to specify partition columns explicitly.
crime_df   = spark.read.parquet(CRIME_PATH)
parcels_df = spark.read.parquet(PARCELS_PATH)
weather_df = spark.read.parquet(WEATHER_PATH)

# Print row counts and schemas to confirm everything loaded correctly.
# Expected:
#   Crime:   2,319 rows — neighborhood × month combinations
#   Parcels: 79 rows   — one per residential neighborhood
#   Weather: 1,627 rows — one per day across both stations
print(f"✓ Crime:   {crime_df.count():,} rows")
print(f"  Columns: {list(crime_df.columns)}")
print()
print(f"✓ Parcels: {parcels_df.count():,} rows")
print(f"  Columns: {list(parcels_df.columns)}")
print()
print(f"✓ Weather: {weather_df.count():,} rows")
print(f"  Columns: {list(weather_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Aggregate Crime to Neighborhood Grain
# MAGIC Crime data is currently at neighborhood-month grain — one row per neighborhood per month. To join it with parcels (which is already at neighborhood grain), we need to roll it up to one row per neighborhood. We compute both totals (for overall magnitude) and monthly averages (for fair comparison between neighborhoods that have different amounts of data).
# MAGIC We also filter out two things here:
# MAGIC
# MAGIC - **Null neighborhoods** — incidents with no location recorded
# MAGIC - **Null median_income** — parks and cemeteries that had no Census match. These aren't residential neighborhoods and would distort comparisons.

# COMMAND ----------

crime_agg = (
    crime_df
    # Filter out Census missing data flag (-666666666) which appears
    # as an extremely large negative number in median_income
    .filter(F.col("median_income") > 0)
    
    # Remove records with no neighborhood — ungeocoded incidents
    .filter(F.col("neighborhood").isNotNull())
    # Remove parks/cemeteries — these had no Census match so
    # median_income is null. They aren't residential neighborhoods
    # and shouldn't appear in livability comparisons.
    .filter(F.col("median_income").isNotNull())

    # Group by neighborhood plus its Census attributes.
    # We include median_income, housing_cost_burden, and flood_zone_pct
    # in the GROUP BY so they carry through to the output — they're
    # constant per neighborhood so grouping on them is safe.
    .groupBy(
        "neighborhood",
        "median_income",
        "housing_cost_burden",
        "flood_zone_pct"
    )
    .agg(
        # ── Totals across the full date range ─────────────────
        # Useful for understanding overall crime volume per neighborhood
        F.sum("total_incidents").alias("total_incidents"),
        F.sum("violent_count").alias("total_violent"),
        F.sum("property_count").alias("total_property"),
        F.sum("drug_count").alias("total_drug"),

        # ── Monthly averages ──────────────────────────────────
        # More meaningful for comparison than raw totals because
        # some neighborhoods have fewer months of data. A neighborhood
        # with 10 months of data would look artificially lower than
        # one with 25 months if we only compared totals.
        F.round(F.avg("total_incidents"), 1).alias("avg_monthly_incidents"),
        F.round(F.avg("violent_crime_pct"), 4).alias("avg_violent_pct"),

        # Track how many months of data exist for this neighborhood
        # — useful context when interpreting the averages
        F.count("month").alias("months_of_data"),
    )
)

print(f"✓ Crime aggregated: {crime_agg.count()} neighborhoods")
print(f"\nTop 10 by avg monthly incidents:")
display(
    crime_agg
    .orderBy("avg_monthly_incidents", ascending=False)
    .select("neighborhood", "avg_monthly_incidents", "total_incidents", "months_of_data")
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Aggregate Weather to City-Wide Summary
# MAGIC Weather data comes from Lambert Airport — a single station covering the whole city. This means we can't assign different weather values to different neighborhoods, so instead we compute city-wide averages across the full date range and broadcast them as context columns on every neighborhood row.

# COMMAND ----------

# Aggregate weather to a single city-wide summary row.
# We pull key metrics that are most relevant to the STL extreme
# weather context — temperature range, precipitation, and snowfall.
weather_agg = weather_df.agg(

    # Average daily high and low temps across the full date range
    # Gives context for seasonal patterns in crime and vacancy
    F.round(F.avg("tmax_f"), 1).alias("avg_high_temp_f"),
    F.round(F.avg("tmin_f"), 1).alias("avg_low_temp_f"),

    # Average daily precipitation in millimeters
    F.round(F.avg("prcp"), 2).alias("avg_daily_precip_mm"),

    # Total snowfall across the full period — relevant for STL
    # which has had significant ice/snow events in 2024-2026
    F.round(F.sum("snow"), 0).alias("total_snowfall_mm"),

    # Single largest rain event — proxy for flash flood risk
    # STL has experienced major flooding in recent years
    F.round(F.max("prcp"), 2).alias("max_single_day_precip_mm"),
)

# Collect to a single Python row so we can use the values
# as literals when adding them to the neighborhood summary
weather_row = weather_agg.first()

print("✓ Weather summarized (city-wide averages):")
display(weather_agg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Join Crime + Parcels + Weather
# MAGIC This is where the three datasets come together into one wide table. Two things worth understanding about this join:
# MAGIC
# MAGIC - Crime + Parcels is a true join on neighborhood name — both datasets use the official NHD_NAME values from the city shapefile so they align cleanly
# MAGIC - Weather isn't joined in the traditional sense — since it's city-wide we add it as literal constant columns using F.lit(), meaning every neighborhood row gets the same weather values. Think of it as stamping context onto the table rather than joining it.

# COMMAND ----------

# ── Step 1: Join crime aggregation to parcel data ─────────────
# Crime uses "neighborhood" and parcels uses "neighborhood_name"
# — both contain official NHD_NAME values from the city shapefile
# so the join key aligns correctly.
# Left join keeps all crime neighborhoods even if parcel data
# somehow doesn't have a match (defensive coding).
summary = (
    crime_agg
    .join(
        parcels_df,
        crime_agg["neighborhood"] == parcels_df["neighborhood_name"],
        how="left"
    )
    # Drop redundant columns that came from the parcels table —
    # we already have "neighborhood" from crime, don't need
    # "neighborhood_name" or the numeric "neighborhood_id" as well
    .drop("neighborhood_name", "neighborhood_id")
)

# ── Step 2: Fix bad Census values ────────────────────────────
# The Census API uses -666666666 as a missing data flag for tracts
# where data wasn't collected. After area-weighted aggregation this
# can produce extremely large negative median_income values.
# We set any negative income to null so it doesn't corrupt the
# min-max normalization in Cell 7.
summary = summary.withColumn(
    "median_income",
    F.when(F.col("median_income") < 0, None)
     .otherwise(F.col("median_income"))
)

# ── Step 3: Add city-wide weather as broadcast columns ────────
# F.lit() creates a constant column — every row gets the same value.
# This is the correct pattern when one dataset has no geographic
# granularity to join on (city-wide weather station).
summary = (
    summary
    .withColumn("avg_high_temp_f",
                F.lit(weather_row["avg_high_temp_f"]))
    .withColumn("avg_low_temp_f",
                F.lit(weather_row["avg_low_temp_f"]))
    .withColumn("avg_daily_precip_mm",
                F.lit(weather_row["avg_daily_precip_mm"]))
    # Cast to float explicitly — Spark can be fussy about
    # integer vs double when using F.lit() with large numbers
    .withColumn("total_snowfall_mm",
                F.lit(float(weather_row["total_snowfall_mm"])))
    .withColumn("max_single_day_precip_mm",
                F.lit(weather_row["max_single_day_precip_mm"]))
)

# ── Step 4: Select final column order ─────────────────────────
# Organize columns into logical groups: identity → crime →
# parcels → Census → weather. This makes the table readable
# and predictable for downstream consumers.
summary = summary.select(
    # Neighborhood identity
    "neighborhood",

    # Crime metrics
    "total_incidents",
    "total_violent",
    "total_property",
    "total_drug",
    "avg_monthly_incidents",
    "avg_violent_pct",
    "months_of_data",

    # Parcel metrics
    "total_parcels",
    "vacant_parcels",
    "occupied_parcels",
    "vacancy_rate",
    "avg_assessed_value",
    "avg_appraised_value",
    "total_assessed_value",

    # Census metrics (joined in crime transform via teammate's data)
    "median_income",
    "housing_cost_burden",
    "flood_zone_pct",

    # City-wide weather context
    "avg_high_temp_f",
    "avg_low_temp_f",
    "avg_daily_precip_mm",
    "total_snowfall_mm",
    "max_single_day_precip_mm",
).orderBy("neighborhood")

print(f"✓ Summary built: {summary.count()} neighborhoods")
print(f"  Total columns: {len(summary.columns)}")

# Verify the bad Census value is fixed
west_end = summary.filter(F.col("neighborhood") == "West End")
if west_end.count() > 0:
    income = west_end.select("median_income").first()["median_income"]
    print(f"\n  West End median_income check: {income}")
    if income is None or income > 0:
        print("  ✓ Bad Census value resolved")
    else:
        print("  ⚠ Bad Census value still present")

print(f"\nSample rows:")
display(summary.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Derive Livability Score
# MAGIC This is the most analytically interesting cell in the whole pipeline —  We're deriving a composite score that ranks every neighborhood on a 0–100 scale by combining four dimensions into one number.
# MAGIC
# MAGIC The technique is min-max normalization: for each metric, we find the worst value across all neighborhoods (score = 0) and the best (score = 100), then scale every neighborhood linearly between them. For metrics where lower is better (crime, vacancy, flood risk) we invert the scale.
# MAGIC
# MAGIC **Why crime isn't weighted more heavily than flood risk?** short answer - is that equal weighting is a transparent starting point and the weights could be adjusted based on community input or policy priorities.

# COMMAND ----------

# ── Step 1: Compute min/max across all neighborhoods ──────────
# We need these to normalize each metric to a 0–100 scale.
# Collecting as a single row so we can use the values as
# literals in the column expressions below.
stats = summary.agg(
    F.min("avg_monthly_incidents").alias("min_crime"),
    F.max("avg_monthly_incidents").alias("max_crime"),
    F.min("vacancy_rate").alias("min_vacancy"),
    F.max("vacancy_rate").alias("max_vacancy"),
    F.min("median_income").alias("min_income"),
    F.max("median_income").alias("max_income"),
    F.min("flood_zone_pct").alias("min_flood"),
    F.max("flood_zone_pct").alias("max_flood"),
).first()

# ── Step 2: Min-max normalization helper ─────────────────────
# Formula: normalized = (value - min) / (max - min)
# Result is 0.0 to 1.0, then multiplied by 100 for readability.
#
# invert=True  → lower raw value = higher score (crime, vacancy, flood)
# invert=False → higher raw value = higher score (income)
#
# Edge case: if min == max (all neighborhoods identical on this metric)
# return 50.0 so no neighborhood is unfairly penalized.
def norm(col_name, min_val, max_val, invert=False):
    if max_val == min_val:
        # All neighborhoods are identical on this dimension —
        # give everyone a neutral score rather than 0
        return F.lit(50.0)
    normalized = (
        (F.col(col_name) - F.lit(float(min_val))) /
        F.lit(float(max_val - min_val))
    )
    # Invert for "lower is better" metrics so the worst
    # neighborhood gets 0 and the best gets 100
    scaled = (F.lit(1.0) - normalized) if invert else normalized
    return F.round(scaled * 100, 1)

# ── Step 3: Compute dimension scores ─────────────────────────
ranking = (
    summary

    # Safety score: fewer monthly incidents = higher score
    # The neighborhood with the lowest crime gets 100,
    # the highest gets 0
    .withColumn("safety_score",
        norm("avg_monthly_incidents",
             stats["min_crime"], stats["max_crime"],
             invert=True))

    # Vacancy score: fewer vacant parcels = higher score
    # High vacancy signals disinvestment and abandonment
    .withColumn("vacancy_score",
        norm("vacancy_rate",
             stats["min_vacancy"], stats["max_vacancy"],
             invert=True))

    # Income score: higher median income = higher score
    # Reflects economic opportunity and neighborhood investment
    .withColumn("income_score",
        norm("median_income",
             stats["min_income"], stats["max_income"],
             invert=False))

    # Flood score: less flood zone coverage = higher score
    # Relevant for STL given recent extreme weather events
    .withColumn("flood_score",
        norm("flood_zone_pct",
             stats["min_flood"], stats["max_flood"],
             invert=True))

    # ── Step 4: Composite livability score ───────────────────
    # Equal-weighted average of all four dimension scores.
    # Range: 0 (worst on all dimensions) to 100 (best on all).
    # Equal weighting is a transparent starting point —
    # weights could be adjusted based on policy priorities.
    .withColumn("livability_score",
        F.round(
            (F.col("safety_score") +
             F.col("vacancy_score") +
             F.col("income_score") +
             F.col("flood_score")) / 4, 1
        )
    )
    .orderBy("livability_score", ascending=False)
)

print("✓ Livability scores computed")
print(f"\nTop 15 most livable neighborhoods:")
display(
    ranking.select(
        "neighborhood",
        "livability_score",
        "safety_score",
        "vacancy_score",
        "income_score",
        "flood_score"
    ).limit(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Bottom 15 Neighborhoods
# MAGIC The bottom of the ranking tells a more compelling story than the top — these are the neighborhoods where multiple dimensions of disadvantage converge simultaneously. High crime AND high vacancy AND low income AND high flood risk in the same places is exactly the kind of insight a city planner or policy maker might want from this platform.

# COMMAND ----------

# The bottom neighborhoods are often the most analytically interesting
# because they show where multiple challenges overlap — high crime,
# high vacancy, low income, and flood risk converging in the same
# places. This is the core insight the platform is designed to surface.
#
# For the presentation: these neighborhoods represent areas where
# city resources (code enforcement, investment, infrastructure)
# could have the highest impact. The platform doesn't just rank
# neighborhoods — it shows WHY they rank where they do through
# the four dimension scores.

print("Bottom 15 neighborhoods by livability score:")
display(
    ranking.select(
        # Identity and composite score
        "neighborhood",
        "livability_score",

        # Dimension scores — shows which factors are driving
        # the low ranking for each neighborhood
        "safety_score",
        "vacancy_score",
        "income_score",
        "flood_score",

        # Raw metrics alongside scores so the numbers are
        # interpretable — a score of 12 means more when you
        # can see it corresponds to 340 monthly incidents
        "avg_monthly_incidents",
        "vacancy_rate",
        "median_income",
        "flood_zone_pct"
    )
    .orderBy("livability_score")
    .limit(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 — Write Serving Tables
# MAGIC Writes both tables to Parquet in the serving volume. One important detail: we write `summary` and `ranking` as separate tables rather than just the ranking, because they serve different purposes. The summary is the clean canonical output for any consumer who just wants the data. The ranking adds the scoring layer on top for anyone who wants the opinionated ranking view.

# COMMAND ----------

# Ensure the serving volume exists before writing.
# CREATE VOLUME IF NOT EXISTS is idempotent — safe to run every time.
# This is the same pattern we used in the transform notebooks.
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.serving")

# ── Write neighborhood summary ────────────────────────────────
# The canonical wide table — all metrics, one row per neighborhood,
# no scoring applied. Clean input for any downstream consumer
# that wants the data without the opinionated ranking layer.
summary.write.mode("overwrite").parquet(SUMMARY_PATH)
summary_count = spark.read.parquet(SUMMARY_PATH).count()
print(f"✓ neighborhood_summary: {summary_count} rows → {SUMMARY_PATH}")

# ── Write neighborhood ranking ────────────────────────────────
# Same data as summary plus four dimension scores and the
# composite livability_score. Sorted by livability_score descending
# so the best neighborhoods appear first when read back.
ranking.write.mode("overwrite").parquet(RANKING_PATH)
ranking_count = spark.read.parquet(RANKING_PATH).count()
print(f"✓ neighborhood_ranking: {ranking_count} rows → {RANKING_PATH}")

# Both tables should have the same row count — ranking is just
# summary with additional score columns, no rows added or removed
if summary_count == ranking_count:
    print(f"\n✓ Row counts match: {summary_count} neighborhoods")
else:
    print(f"\n⚠ Row count mismatch — summary={summary_count}, ranking={ranking_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10 — Quality Checks
# MAGIC Three checks that confirm the serving tables are correct. The most important one is the full ranking table at the bottom. If the neighborhoods at the top and bottom make intuitive sense for St. Louis, the pipeline is working correctly.

# COMMAND ----------

# Read back from Parquet to verify the write succeeded cleanly
result = spark.read.parquet(RANKING_PATH)

# ── Check 1: Null rates ───────────────────────────────────────
# Key fields should all be 0% null. If livability_score has nulls
# it means a neighborhood was missing one of the four input metrics
# which would cause the averaging to produce null.
print("=== NULL RATES (%) ===")
display(result.select([
    F.round(
        F.count(F.when(F.col(c).isNull(), c)) /
        F.count(F.lit(1)) * 100, 2
    ).alias(c)
    for c in [
        "neighborhood",
        "avg_monthly_incidents",
        "vacancy_rate",
        "median_income",
        "flood_zone_pct",
        "livability_score",
        "safety_score",
        "vacancy_score",
        "income_score",
        "flood_score",
    ]
]))

# COMMAND ----------

# ── Check 2: Score distribution ───────────────────────────────
# Scores should be spread across the 0–100 range.
# If min and max are very close together the normalization didn't
# work correctly. A healthy spread means the metric is actually
# differentiating between neighborhoods.
print("=== LIVABILITY SCORE DISTRIBUTION ===")
display(
    result.select(
        F.min("livability_score").alias("min"),
        F.round(F.avg("livability_score"), 1).alias("avg"),
        F.max("livability_score").alias("max"),
        F.percentile_approx("livability_score", 0.25).alias("p25"),
        F.percentile_approx("livability_score", 0.75).alias("p75"),
    )
)

# COMMAND ----------

# ── Check 3: Full ranking table ───────────────────────────────
# This is your presentation output. Review it and confirm the
# neighborhoods at the top and bottom make intuitive sense
# for St. Louis. Expected top neighborhoods: Hill, Soulard,
# Central West End, Clayton-Tamm. Expected bottom: high-vacancy
# north side neighborhoods like St. Louis Place, Jeff Vanderlou.
print("=== FULL NEIGHBORHOOD RANKING ===")
display(
    result.select(
        "neighborhood",
        "livability_score",
        "safety_score",
        "vacancy_score",
        "income_score",
        "flood_score",
        "avg_monthly_incidents",
        "vacancy_rate",
        "median_income",
        "flood_zone_pct",
    )
    .orderBy("livability_score", ascending=False)
)
