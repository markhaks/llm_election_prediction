"""
03b_filter_irrelevant_locations.py

Purpose:
--------
Reduces dataset size before detailed geolocation processing.

This script removes rows where:
1. user_location is empty
2. user_location contains very generic non-usable locations
3. user_location appears in the manual review list with decision == "raus"

Input:
------
data/processed/03_global_deduplicated/

Manual review file:
-------------------
data/external/manuel_list_most_common_location.CSV

Required columns in manual file:
--------------------------------
- user_location_original
- decision

Output:
-------
data/processed/03b_filtered_locations/

Report:
-------
data/processed/03b_filtered_locations/location_filter_report.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/03_global_deduplicated")

OUTPUT_DIR = Path(
    "data/processed/04_extract_location"
)

MANUAL_FILTER_FILE = Path(
    "data/external/manuel_list_most_common_location.CSV"
)

REPORT_FILE = OUTPUT_DIR / "location_filter_report.csv"


GENERIC_LOCATIONS = {
    "united states",
    "usa",
    "earth",
    "canada",
    "united states of america",
    "america",
    "united kingdom",
    "australia",
    "planet earth",
    "london, england",
}


REQUIRED_MANUAL_COLUMNS = [
    "user_location_original",
    "decision",
]


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_global_deduplicated.csv"))

    if not files:
        raise FileNotFoundError(
            f"No globally deduplicated files found in: {input_dir}"
        )

    return files


def normalize_location(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def load_manual_filter_list(file_path: Path) -> set[str]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Manual filter file not found: {file_path}"
        )

    manual_df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        sep=";",
    )

    missing_columns = [
        col for col in REQUIRED_MANUAL_COLUMNS
        if col not in manual_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Manual filter file missing columns: {missing_columns}"
        )

    manual_df["location_key"] = manual_df[
        "user_location_original"
    ].apply(normalize_location)

    manual_df["decision"] = manual_df[
        "decision"
    ].apply(normalize_location)

    remove_locations = manual_df[
        manual_df["decision"] == "raus"
    ]["location_key"]

    return set(remove_locations)


def process_file(
    file_path: Path,
    manual_remove_locations: set[str],
) -> dict:

    result = {
        "source_file": str(file_path),
        "output_file": None,
        "status": None,
        "rows_before": 0,
        "rows_removed_empty": 0,
        "rows_removed_generic": 0,
        "rows_removed_manual": 0,
        "rows_after": 0,
        "error_type": None,
        "error_message": None,
    }

    try:
        df = pd.read_csv(file_path, low_memory=False)

        if "user_location" not in df.columns:
            result["status"] = "missing_user_location"
            result["error_message"] = "Column user_location not found"
            return result

        result["rows_before"] = len(df)

        df["location_key"] = df["user_location"].apply(
            normalize_location
        )

        # -----------------------------------------
        # 1. Remove empty locations
        # -----------------------------------------

        empty_mask = (
            df["location_key"].isna()
            | (df["location_key"] == "")
        )

        result["rows_removed_empty"] = int(empty_mask.sum())

        df = df[~empty_mask].copy()

        # -----------------------------------------
        # 2. Remove generic locations
        # -----------------------------------------

        generic_mask = df["location_key"].isin(
            GENERIC_LOCATIONS
        )

        result["rows_removed_generic"] = int(generic_mask.sum())

        df = df[~generic_mask].copy()

        # -----------------------------------------
        # 3. Remove manual "raus" locations
        # -----------------------------------------

        manual_mask = df["location_key"].isin(
            manual_remove_locations
        )

        result["rows_removed_manual"] = int(manual_mask.sum())

        df = df[~manual_mask].copy()

        # cleanup helper column
        df = df.drop(columns=["location_key"])

        output_file = OUTPUT_DIR / file_path.name.replace(
            "_global_deduplicated.csv",
            "_location_filtered.csv",
        )

        df.to_csv(
            output_file,
            index=False,
            encoding="utf-8",
        )

        result["output_file"] = str(output_file)
        result["rows_after"] = len(df)
        result["status"] = "success"

        return result

    except Exception as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)

        return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_input_files(INPUT_DIR)

    manual_remove_locations = load_manual_filter_list(
        MANUAL_FILTER_FILE
    )

    print(f"Files found: {len(files)}")
    print(f"Manual remove locations loaded: {len(manual_remove_locations):,}")

    results = []

    for file_path in tqdm(files, desc="Filtering locations"):
        result = process_file(
            file_path=file_path,
            manual_remove_locations=manual_remove_locations,
        )

        results.append(result)

    report_df = pd.DataFrame(results)

    report_df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8",
    )

    successful = report_df[
        report_df["status"] == "success"
    ]

    total_before = successful["rows_before"].sum()
    total_empty = successful["rows_removed_empty"].sum()
    total_generic = successful["rows_removed_generic"].sum()
    total_manual = successful["rows_removed_manual"].sum()
    total_after = successful["rows_after"].sum()

    print("\nLocation filtering finished.")
    print(f"Rows before: {total_before:,}")
    print(f"Removed empty locations: {total_empty:,}")
    print(f"Removed generic locations: {total_generic:,}")
    print(f"Removed manual locations: {total_manual:,}")
    print(f"Rows after filtering: {total_after:,}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()