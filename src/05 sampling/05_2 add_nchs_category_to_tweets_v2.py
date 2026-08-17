"""
Mit neuen Daten aus processed\04_extract_location\Step 2_fast_strict

07_add_nchs_category_to_tweets.py

Purpose:
--------
Adds the NCHS urban-rural category to every tweet CSV file.
Additionally, all unmatched rows are saved in a separate review file.

Input:
------
Tweet CSV files:
data/processed/04_extract_location/step 5_all_detected_locations/

City-NCHS table:
data/external/cities_with_nchs_categories.csv

Output:
-------
data/processed/06_urban_rural/tweets_with_nchs_category/

New columns:
------------
nchs_code
nchs_category_name
nchs_2023_category

Additional files:
-----------------
nchs_merge_report.csv
unmatched_nchs_locations.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\04_extract_location\Step 2_fast_strict\city_state_matched"
)

CITY_NCHS_FILE = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\external\cities_with_nchs_categories.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_2c new data"
)

REPORT_FILE = OUTPUT_DIR / "nchs_merge_report.csv"
UNMATCHED_FILE = OUTPUT_DIR / "unmatched_nchs_locations.csv"


TWEET_CITY_COLUMN = "user_city"
TWEET_STATE_COLUMN = "user_state"

CITY_TABLE_CITY_COLUMN = "city"
CITY_TABLE_STATE_COLUMN = "state_id"


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def normalize_city(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace(".", "")
    )


def normalize_state(value):
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def find_csv_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    return files


def load_city_nchs_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"City-NCHS file not found: {path}")

    df = pd.read_csv(path, dtype=str)

    required_columns = [
        CITY_TABLE_CITY_COLUMN,
        CITY_TABLE_STATE_COLUMN,
        "nchs_code",
        "nchs_category_name",
        "nchs_2023_category",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"City-NCHS file is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()

    df["city_key"] = df[CITY_TABLE_CITY_COLUMN].apply(normalize_city)
    df["state_key"] = df[CITY_TABLE_STATE_COLUMN].apply(normalize_state)

    keep_columns = [
        "city_key",
        "state_key",
        "nchs_code",
        "nchs_category_name",
        "nchs_2023_category",
    ]

    df = df[keep_columns].drop_duplicates(
        subset=["city_key", "state_key"],
        keep="first"
    )

    return df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading City-NCHS mapping table...")
    mapping_df = load_city_nchs_mapping(CITY_NCHS_FILE)

    print(f"Mapping rows: {len(mapping_df):,}")

    files = find_csv_files(INPUT_DIR)

    print(f"Found CSV files: {len(files):,}")

    report_rows = []
    unmatched_rows = []

    for file_path in tqdm(files, desc="Adding NCHS categories"):
        try:
            df = pd.read_csv(file_path, dtype=str)

            required_columns = [
                TWEET_CITY_COLUMN,
                TWEET_STATE_COLUMN,
            ]

            missing = [col for col in required_columns if col not in df.columns]

            if missing:
                report_rows.append(
                    {
                        "file": str(file_path),
                        "status": "missing_columns",
                        "rows_total": len(df),
                        "rows_matched": 0,
                        "rows_unmatched": len(df),
                        "missing_columns": ", ".join(missing),
                        "error": "",
                    }
                )
                continue

            df["city_key"] = df[TWEET_CITY_COLUMN].apply(normalize_city)
            df["state_key"] = df[TWEET_STATE_COLUMN].apply(normalize_state)

            merged_df = df.merge(
                mapping_df,
                on=["city_key", "state_key"],
                how="left",
            )

            rows_total = len(merged_df)
            rows_matched = merged_df["nchs_code"].notna().sum()
            rows_unmatched = merged_df["nchs_code"].isna().sum()

            unmatched_df = merged_df[merged_df["nchs_code"].isna()].copy()

            if not unmatched_df.empty:
                unmatched_df["source_file"] = file_path.name
                unmatched_rows.append(unmatched_df)

            merged_df = merged_df.drop(
                columns=["city_key", "state_key"],
                errors="ignore"
            )

            output_file = OUTPUT_DIR / file_path.name

            merged_df.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            report_rows.append(
                {
                    "file": str(file_path),
                    "status": "processed",
                    "rows_total": rows_total,
                    "rows_matched": rows_matched,
                    "rows_unmatched": rows_unmatched,
                    "missing_columns": "",
                    "error": "",
                }
            )

        except Exception as e:
            report_rows.append(
                {
                    "file": str(file_path),
                    "status": "error",
                    "rows_total": 0,
                    "rows_matched": 0,
                    "rows_unmatched": 0,
                    "missing_columns": "",
                    "error": str(e),
                }
            )

    report_df = pd.DataFrame(report_rows)

    report_df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    if unmatched_rows:
        all_unmatched_df = pd.concat(
            unmatched_rows,
            ignore_index=True
        )

        all_unmatched_df = all_unmatched_df.drop(
            columns=["city_key", "state_key"],
            errors="ignore"
        )

        all_unmatched_df.to_csv(
            UNMATCHED_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\nUnmatched rows written to: {UNMATCHED_FILE}")
    else:
        print("\nNo unmatched rows found.")

    print("\nDone.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Report file: {REPORT_FILE}")

    total_rows = report_df["rows_total"].sum()
    total_matched = report_df["rows_matched"].sum()
    total_unmatched = report_df["rows_unmatched"].sum()

    print("\nSummary:")
    print(f"Total rows: {total_rows:,}")
    print(f"Matched rows: {total_matched:,}")
    print(f"Unmatched rows: {total_unmatched:,}")

    if total_rows > 0:
        print(f"Match rate: {total_matched / total_rows:.2%}")


if __name__ == "__main__":
    main()