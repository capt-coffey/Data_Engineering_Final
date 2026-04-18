"""
ingest/gtfs.py
STL-5 — Metro St. Louis GTFS Static Ingestion Module

Downloads the Metro STL GTFS static ZIP from the official
developer portal, extracts stops.txt, routes.txt, and trips.txt,
normalizes schemas, and writes raw output to Parquet.

Designed to run on a weekly Databricks Job schedule. Uses MD5
hash comparison to skip reprocessing when the ZIP hasn't changed
since the last run — Metro typically updates ~5 times per year.

Data source: https://www.metrostlouis.org/developer-resources/
GTFS spec:   https://developers.google.com/transit/gtfs/reference
"""

import io
import os
import hashlib
import logging
import zipfile
import requests
import pandas as pd

from datetime import datetime, timezone
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

GTFS_ZIP_URL = "https://www.metrostlouis.org/Transit/google_transit.zip"

# Files we extract from the ZIP — other GTFS files exist but
# are not needed for the neighborhood intelligence model
GTFS_FILES = ["stops.txt", "routes.txt", "trips.txt"]

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Expected columns per file — required columns must be present
# or ingestion fails; optional columns are null-filled if absent
GTFS_SCHEMAS = {
    "stops.txt": {
        "required": ["stop_id", "stop_name", "stop_lat", "stop_lon"],
        "optional": [
            "stop_code", "stop_desc", "zone_id", "stop_url",
            "location_type", "parent_station", "wheelchair_boarding",
        ],
    },
    "routes.txt": {
        "required": [
            "route_id", "route_short_name",
            "route_long_name", "route_type",
        ],
        "optional": [
            "agency_id", "route_desc", "route_url",
            "route_color", "route_text_color",
        ],
    },
    "trips.txt": {
        "required": ["route_id", "service_id", "trip_id"],
        "optional": [
            "trip_headsign", "trip_short_name", "direction_id",
            "block_id", "shape_id", "wheelchair_accessible",
            "bikes_allowed",
        ],
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_md5(data: bytes) -> str:
    """
    Compute the MD5 hex digest of raw bytes.
    Used to fingerprint the GTFS ZIP for change detection.
    """
    return hashlib.md5(data).hexdigest()


def download_gtfs_zip(url: str = GTFS_ZIP_URL, timeout: int = 90) -> bytes | None:
    """
    Download the Metro STL GTFS static ZIP.

    Retries up to 3 times on transient failures.

    Args:
        url:     Full URL to the GTFS ZIP file.
        timeout: Per-request timeout in seconds.

    Returns:
        Raw ZIP bytes, or None if all attempts fail.
    """
    for attempt in range(1, 4):
        try:
            logger.info(f"GET {url} (attempt {attempt}/3)")
            resp = requests.get(
                url,
                headers=BROWSER_HEADERS,
                timeout=timeout
            )

            if resp.status_code == 200:
                logger.info(
                    f"Downloaded {len(resp.content) / 1e6:.2f} MB"
                )
                return resp.content

            if resp.status_code in (403, 404):
                logger.error(f"HTTP {resp.status_code} — not retrying")
                return None

            logger.warning(f"HTTP {resp.status_code} on attempt {attempt}/3")

        except requests.Timeout:
            logger.warning(f"Timeout on attempt {attempt}/3")
        except requests.RequestException as e:
            logger.warning(f"Request error on attempt {attempt}/3: {e}")

    logger.error("All download attempts failed")
    return None


def parse_gtfs_file(zip_bytes: bytes, filename: str) -> pd.DataFrame | None:
    """
    Extract and parse one file from the GTFS ZIP.

    Validates required columns, null-fills missing optional columns,
    drops unknown columns, and attaches provenance metadata.

    Args:
        zip_bytes: Raw bytes of the downloaded GTFS ZIP.
        filename:  GTFS filename to extract e.g. "stops.txt".

    Returns:
        Normalized Pandas DataFrame, or None on failure.
    """
    schema = GTFS_SCHEMAS.get(filename)
    if schema is None:
        logger.error(f"No schema defined for {filename}")
        return None

    all_cols      = schema["required"] + schema["optional"]
    required_cols = schema["required"]

    # Open ZIP from bytes — no temp file needed
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            if filename not in zf.namelist():
                logger.error(
                    f"{filename} not in ZIP. "
                    f"Available: {zf.namelist()}"
                )
                return None

            # utf-8-sig handles optional BOM in GTFS files
            with zf.open(filename) as f:
                pdf = pd.read_csv(
                    f,
                    dtype=str,
                    encoding="utf-8-sig",
                    low_memory=False
                )

    except zipfile.BadZipFile as e:
        logger.error(f"Bad ZIP: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return None

    # Strip whitespace from column names
    pdf.columns = [c.strip() for c in pdf.columns]

    # Validate required columns
    missing = [c for c in required_cols if c not in pdf.columns]
    if missing:
        logger.error(f"{filename} missing required columns: {missing}")
        return None

    # Add missing optional columns as null
    for col in schema["optional"]:
        if col not in pdf.columns:
            pdf[col] = None

    # Keep only schema columns
    pdf = pdf[[c for c in all_cols if c in pdf.columns]]

    # Attach provenance
    pdf["_source"]      = f"metro_stl_gtfs/{filename}"
    pdf["_ingested_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(f"Parsed {filename}: {len(pdf):,} rows")
    return pdf


def ingest_gtfs(
    spark: SparkSession,
    output_path: str,
    force: bool = False,
    checkpoint_path: str | None = None,
) -> dict[str, int]:
    """
    Main entry point. Downloads Metro STL GTFS ZIP, checks for
    changes via MD5 hash, and writes stops/routes/trips to Parquet.

    Args:
        spark:            Active SparkSession.
        output_path:      Base path for Parquet output.
        force:            Skip hash check and always reprocess.
        checkpoint_path:  Path to store the last ZIP hash.
                          Defaults to output_path/_checkpoint/last_zip_hash.txt

    Returns:
        Dict mapping table name → row count written.
        Empty dict if skipped due to no change detected.
    """
    if checkpoint_path is None:
        checkpoint_path = f"{output_path}/_checkpoint/last_zip_hash.txt"

    ingest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: dict[str, int] = {}

    # Download ZIP
    zip_bytes = download_gtfs_zip()
    if zip_bytes is None:
        raise RuntimeError("GTFS ZIP download failed")

    # Hash check — skip if unchanged
    current_hash = compute_md5(zip_bytes)
    try:
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
        previous_hash = dbutils.fs.head(checkpoint_path, 32).strip()
    except Exception:
        previous_hash = None

    if not force and current_hash == previous_hash:
        logger.info(f"ZIP unchanged (hash: {current_hash}) — skipping")
        return {}

    # Extract and write each file
    for filename in GTFS_FILES:
        pdf = parse_gtfs_file(zip_bytes, filename)
        if pdf is None:
            logger.warning(f"Skipping {filename} — parse failed")
            continue

        table_name  = filename.replace(".txt", "")
        output_full = f"{output_path}/{table_name}/ingest_date={ingest_date}"

        sdf = spark.createDataFrame(pdf)
        sdf.write.mode("overwrite").parquet(output_full)

        count               = sdf.count()
        results[table_name] = count
        logger.info(f"Wrote {count:,} rows → {output_full}")

    # Write checkpoint only if all files succeeded
    if len(results) == len(GTFS_FILES):
        try:
            dbutils.fs.mkdirs("/".join(checkpoint_path.split("/")[:-1]))
            dbutils.fs.put(checkpoint_path, current_hash, overwrite=True)
        except Exception as e:
            logger.warning(f"Could not write checkpoint: {e}")

    return results
