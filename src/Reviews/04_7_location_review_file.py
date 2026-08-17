"""
create_location_review_file.py

Purpose:
--------
Creates a review CSV with all unique user_location variants and what was detected from them.

Input:
------
Folder with matched CSV files

Output:
-------
location_review_unique_variants.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\04_extract_location\Step 2_fast_strict\city_state_not_matched"
)

OUTPUT_FILE = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\04_extract_location\location_review_unique_variants_not_matched.csv"
)

LOCATION_COLUMN = "user_location"


COLUMNS_TO_KEEP = [
    "user_location",
    "user_city",
    "user_state",
    "user_state_name",
    "user_county_fips",
    "user_county_name",
    "user_city_population",
    "user_state_match_method",
]


def find_csv_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    return files


def normalize_location(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def main():
    files = find_csv_files(INPUT_DIR)

    print(f"CSV files found: {len(files):,}")

    all_rows = []

    for file_path in tqdm(files, desc="Reading files"):
        df = pd.read_csv(file_path, dtype=str, low_memory=False)

        if LOCATION_COLUMN not in df.columns:
            print(f"Skipping {file_path.name}: missing {LOCATION_COLUMN}")
            continue

        existing_columns = [col for col in COLUMNS_TO_KEEP if col in df.columns]

        temp = df[existing_columns].copy()
        temp["user_location_clean"] = temp[LOCATION_COLUMN].apply(normalize_location)

        all_rows.append(temp)

    if not all_rows:
        raise RuntimeError("No valid rows found.")

    all_df = pd.concat(all_rows, ignore_index=True)

    # Count how often each original location variant appears
    grouped = (
        all_df
        .groupby(
            [
                "user_location_clean",
                "user_city",
                "user_state",
                "user_state_name",
                "user_county_fips",
                "user_county_name",
                "user_city_population",
                "user_state_match_method",
            ],
            dropna=False,
            as_index=False
        )
        .size()
        .rename(columns={"size": "occurrences"})
    )

    grouped = grouped.sort_values(
        by=["user_state", "user_city", "occurrences"],
        ascending=[True, True, False]
    )

    grouped.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Unique location variants: {len(grouped):,}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()