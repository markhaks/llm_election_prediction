"""
04_2_review_not_detected_locations.py

Purpose:
--------
Creates a frequency review of user_location values where no US state was detected.

This script helps identify common unresolved location patterns, such as:
- NYC
- Bay Area
- Chicago
- Las Vegas
- Earth
- USA

Input:
------
data/processed/04_extract_location/state_not_detected/

Output:
-------
data/processed/04_extract_location/location_review_not_detected.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/04_extract_location/state_not_detected")
OUTPUT_FILE = Path("data/processed/04_extract_location/location_review_not_detected.csv")

REQUIRED_COLUMNS = [
    "user_location",
]


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_state_not_detected.csv"))

    if not files:
        raise FileNotFoundError(f"No state-not-detected CSV files found in: {input_dir}")

    return files


def normalize_location(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def main() -> None:
    files = find_input_files(INPUT_DIR)

    print(f"Found {len(files)} state-not-detected files.")
    print("Creating not-detected location review...")

    summaries = []

    for file_path in tqdm(files, desc="Reading not-detected files"):
        try:
            df = pd.read_csv(
                file_path,
                usecols=lambda col: col in REQUIRED_COLUMNS,
                low_memory=False,
            )

            missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

            if missing_columns:
                print(f"Skipping {file_path}. Missing columns: {missing_columns}")
                continue

            df["user_location_original"] = df["user_location"].apply(normalize_location)

            grouped = (
                df.groupby("user_location_original", dropna=False)
                .size()
                .reset_index(name="count")
            )

            summaries.append(grouped)

        except Exception as error:
            print(f"Error reading {file_path}: {error}")
            continue

    if not summaries:
        raise ValueError("No valid files were processed.")

    review_df = pd.concat(summaries, ignore_index=True)

    final_review_df = (
        review_df.groupby("user_location_original", dropna=False)["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
    )

    final_review_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\nNot-detected location review created.")
    print(f"Unique not-detected location variants: {len(final_review_df):,}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()