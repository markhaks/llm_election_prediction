"""

06_merge_city_with_nchs_categories.py

Purpose:
--------
Merges the city table with the official NCHS county classification.

This script keeps the original 6 NCHS categories.
No reduction to 2 or 3 groups is applied yet.

Input:
------
data/external/us_cities.csv
data/external/data-table_nch_categories.csv

Output:
-------
data/processed/06_urban_rural/cities_with_nchs_categories.csv
"""

from pathlib import Path
import pandas as pd


CITY_FILE = Path("data/external/uscities.csv")
NCHS_FILE = Path("data/external/data-table_nch_categories.csv")

OUTPUT_DIR = Path("data/processed/06_urban_rural")
OUTPUT_FILE = OUTPUT_DIR / "cities_with_nchs_categories.csv"


def extract_nchs_code(value):
    """
    Extracts the numeric code from values like:
    '6 - Noncore'
    '5 - Micropolitan'
    '1 - Large central metro'
    """
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()
    code = value.split("-")[0].strip()

    try:
        return int(code)
    except ValueError:
        return pd.NA


def extract_nchs_name(value):
    """
    Extracts the category name from values like:
    '6 - Noncore' -> 'Noncore'
    """
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if "-" not in value:
        return value

    return value.split("-", 1)[1].strip()


def load_city_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"City file not found: {path}")

    df = pd.read_csv(path)

    required_columns = [
        "city",
        "state_id",
        "county_fips",
        "county_name",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"City file is missing columns: {missing}")

    df = df.copy()

    df["county_fips"] = (
        pd.to_numeric(df["county_fips"], errors="coerce")
        .astype("Int64")
    )

    df["state_id"] = df["state_id"].astype(str).str.strip().str.upper()

    return df


def load_nchs_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"NCHS file not found: {path}")

    df = pd.read_csv(path)

    required_columns = [
        "Location",
        "State",
        "County_name",
        "2023 Code",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"NCHS file is missing columns: {missing}")

    df = df[required_columns].copy()

    df = df.rename(
        columns={
            "Location": "county_fips",
            "State": "state_id",
            "County_name": "nchs_county_name",
            "2023 Code": "nchs_2023_category",
        }
    )

    df["county_fips"] = (
        pd.to_numeric(df["county_fips"], errors="coerce")
        .astype("Int64")
    )

    df["state_id"] = df["state_id"].astype(str).str.strip().str.upper()

    df["nchs_code"] = df["nchs_2023_category"].apply(extract_nchs_code)
    df["nchs_category_name"] = df["nchs_2023_category"].apply(extract_nchs_name)

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading city data...")
    city_df = load_city_data(CITY_FILE)

    print("Loading NCHS data...")
    nchs_df = load_nchs_data(NCHS_FILE)

    print("Merging city data with NCHS categories...")

    merged_df = city_df.merge(
        nchs_df[
            [
                "county_fips",
                "state_id",
                "nchs_county_name",
                "nchs_2023_category",
                "nchs_code",
                "nchs_category_name",
            ]
        ],
        on=["county_fips", "state_id"],
        how="left",
    )

    missing_matches = merged_df["nchs_code"].isna().sum()

    print(f"Total city rows: {len(merged_df):,}")
    print(f"Rows without NCHS match: {missing_matches:,}")

    category_counts = (
        merged_df["nchs_2023_category"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nNCHS category distribution:")
    print(category_counts)

    merged_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nDone. Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()