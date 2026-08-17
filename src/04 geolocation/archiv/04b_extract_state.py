"""
04b_apply_manual_location_review.py

Purpose:
--------
Uses a manually reviewed location list to detect additional user states.

Workflow:
---------
1. Read rows from:
   data/processed/04_extract_location/state_not_detected/

2. Compare user_location with the manual review file.

3. If a location matches the manual review file and manual_state is available:
   - write matched rows to:
     data/processed/04_extract_location/04b state_detected/

4. If a location matches the manual review file but no manual_state is available:
   - remove the row from state_not_detected
   - do not write it to state_detected

5. If a location does not match the manual review file:
   - keep the row in state_not_detected

Safety:
-------
Before overwriting a state_not_detected file, a .bak backup is created.

Output:
-------
- manually detected CSV files
- updated state_not_detected CSV files
- manual_location_matching_report.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


NOT_DETECTED_DIR = Path("data/processed/04_extract_location/state_not_detected")
OUTPUT_DETECTED_DIR = Path("data/processed/04_extract_location/04b state_detected")

MANUAL_REVIEW_FILE = Path(
    "data/external/location_review_not_detected_manual.csv"
)

REPORT_FILE = Path(
    "data/processed/04_extract_location/manual_location_matching_report.csv"
)

REQUIRED_MANUAL_COLUMNS = [
    "user_location_original",
    "manual_city",
    "manual_state",
]


def find_not_detected_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_state_not_detected.csv"))

    if not files:
        raise FileNotFoundError(f"No state-not-detected CSV files found in: {input_dir}")

    return files


def normalize_location(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()



def load_manual_mapping(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Manual review file not found: {file_path}")

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
            f"Manual review file missing columns: {missing_columns}. "
            f"Found columns: {list(manual_df.columns)}"
        )

    manual_df["location_key"] = manual_df["user_location_original"].apply(
        normalize_location
    )

    manual_df["manual_city"] = manual_df["manual_city"].apply(normalize_location)
    manual_df["manual_state"] = manual_df["manual_state"].apply(normalize_location)

    manual_df = manual_df.drop_duplicates(
        subset=["location_key"],
        keep="first",
    )

    return manual_df[
        [
            "location_key",
            "manual_city",
            "manual_state",
        ]
    ]

def safe_overwrite_csv(df: pd.DataFrame, target_file: Path) -> None:
    backup_file = target_file.with_suffix(target_file.suffix + ".bak")
    temp_file = target_file.with_suffix(target_file.suffix + ".tmp")

    if not backup_file.exists():
        target_file.replace(backup_file)
    else:
        target_file.unlink()

    df.to_csv(temp_file, index=False, encoding="utf-8")
    temp_file.replace(target_file)


def process_file(file_path: Path, manual_mapping_df: pd.DataFrame) -> dict:
    result = {
        "source_file": str(file_path),
        "detected_output_file": None,
        "status": None,
        "rows_before": 0,
        "rows_manual_detected": 0,
        "rows_removed_manually": 0,
        "rows_remaining_not_detected": 0,
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

        df["location_key"] = df["user_location"].apply(normalize_location)

        merged_df = df.merge(
            manual_mapping_df,
            on="location_key",
            how="left",
            indicator=True,
        )

        exists_in_manual_review = merged_df["_merge"] == "both"

        has_valid_manual_state = (
            merged_df["manual_state"].notna()
            & (merged_df["manual_state"].astype(str).str.strip() != "")
        )

        matched_mask = exists_in_manual_review & has_valid_manual_state

        remove_mask = exists_in_manual_review & ~has_valid_manual_state

        detected_df = merged_df[matched_mask].copy()
        removed_df = merged_df[remove_mask].copy()
        remaining_df = merged_df[~(matched_mask | remove_mask)].copy()

        detected_df["user_state"] = detected_df["manual_state"]
        detected_df["user_city"] = detected_df["manual_city"]
        detected_df["user_state_match_method"] = "manual_location_review"

        cleanup_columns = [
            "location_key",
            "manual_city",
            "manual_state",
            "_merge",
        ]

        detected_df = detected_df.drop(
            columns=cleanup_columns,
            errors="ignore",
        )

        remaining_df = remaining_df.drop(
            columns=cleanup_columns,
            errors="ignore",
        )

        detected_output_file = OUTPUT_DETECTED_DIR / file_path.name.replace(
            "_state_not_detected.csv",
            "_manual_state_detected.csv",
        )

        if len(detected_df) > 0:
            detected_df.to_csv(
                detected_output_file,
                index=False,
                encoding="utf-8",
            )

        safe_overwrite_csv(remaining_df, file_path)

        result["detected_output_file"] = (
            str(detected_output_file) if len(detected_df) > 0 else None
        )
        result["rows_manual_detected"] = len(detected_df)
        result["rows_removed_manually"] = len(removed_df)
        result["rows_remaining_not_detected"] = len(remaining_df)
        result["status"] = "success"

        return result

    except Exception as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        return result


def main() -> None:
    OUTPUT_DETECTED_DIR.mkdir(parents=True, exist_ok=True)

    manual_mapping_df = load_manual_mapping(MANUAL_REVIEW_FILE)
    files = find_not_detected_files(NOT_DETECTED_DIR)

    print(f"Manual review entries loaded: {len(manual_mapping_df):,}")
    print(f"Not-detected files found: {len(files)}")
    print(f"Detected output folder: {OUTPUT_DETECTED_DIR}")

    results = []

    for file_path in tqdm(files, desc="Applying manual location mappings"):
        result = process_file(file_path, manual_mapping_df)
        results.append(result)

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8")

    successful = report_df[report_df["status"] == "success"]

    total_before = successful["rows_before"].sum()
    total_detected = successful["rows_manual_detected"].sum()
    total_removed = successful["rows_removed_manually"].sum()
    total_remaining = successful["rows_remaining_not_detected"].sum()

    print("\nManual location matching finished.")
    print(f"Rows before: {total_before:,}")
    print(f"Rows manually detected: {total_detected:,}")
    print(f"Rows removed manually: {total_removed:,}")
    print(f"Rows remaining in not_detected: {total_remaining:,}")
    print(f"Control sum: {total_detected + total_removed + total_remaining:,}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()