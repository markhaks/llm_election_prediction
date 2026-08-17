"""
10_extract_specific_state_nchs_combinations.py

Purpose:
--------
Extracts specific state + NCHS category combinations
from all tweet CSV files into a separate review file.

The original CSV files are NOT modified.

Input:
------
C:/Users/Mark/Documents/Uni/Bachelorarbeit/
llm_election_prediction/data/processed/
05_sampling/05_2b tweets_without_puerto_rico

Output:
-------
matched_state_nchs_review.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_2b tweets_without_puerto_rico"
)

OUTPUT_FILE = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\matched_state_nchs_review.csv"
)

STATE_COLUMN = "user_state"
NCHS_COLUMN = "nchs_code"


# --------------------------------------------------
# TARGET COMBINATIONS
# --------------------------------------------------

TARGET_COMBINATIONS = {
    ("MO", 1),
    ("MO", 6),

    ("AK", 4),

    ("AR", 3),

    ("DE", 4),
    ("DE", 5),

    ("ID", 4),
    ("ID", 5),

    ("IL", 1),
    ("IL", 2),
    ("IL", 5),

    ("IN", 5),

    ("KS", 2),
    ("KS", 3),
    ("KS", 6),

    ("KY", 1),
    ("KY", 4),

    ("LA", 3),
    ("LA", 4),

    ("ME", 3),
    ("ME", 6),

    ("MN", 1),
    ("MN", 6),

    ("MS", 3),
    ("MS", 6),

    ("ND", 3),
    ("ND", 6),

    ("NE", 3),
    ("NE", 6),

    ("NM", 3),
    ("NM", 6),

    ("OH", 5),

    ("OK", 1),
    ("OK", 5),

    ("PA", 1),
    ("PA", 5),

    ("SC", 3),
    ("SC", 5),

    ("TX", 6),

    ("UT", 1),
    ("UT", 3),

    ("WY", 4),
    ("WY", 5),
}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def normalize_state(value):
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def find_csv_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    return files


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    files = find_csv_files(INPUT_DIR)

    print(f"Found CSV files: {len(files):,}")

    matched_rows = []

    for file_path in tqdm(files, desc="Checking files"):
        try:
            df = pd.read_csv(file_path, dtype=str)

            required_columns = [
                STATE_COLUMN,
                NCHS_COLUMN,
            ]

            missing = [col for col in required_columns if col not in df.columns]

            if missing:
                print(
                    f"Skipping {file_path.name} "
                    f"(missing columns: {missing})"
                )
                continue

            df[STATE_COLUMN] = (
                df[STATE_COLUMN]
                .apply(normalize_state)
            )

            df[NCHS_COLUMN] = pd.to_numeric(
                df[NCHS_COLUMN],
                errors="coerce"
            ).astype("Int64")

            mask = df.apply(
                lambda row: (
                    row[STATE_COLUMN],
                    row[NCHS_COLUMN]
                ) in TARGET_COMBINATIONS,
                axis=1
            )

            matched_df = df[mask].copy()

            if not matched_df.empty:
                matched_df["source_file"] = file_path.name
                matched_rows.append(matched_df)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    if not matched_rows:
        print("No matching rows found.")
        return

    result_df = pd.concat(
        matched_rows,
        ignore_index=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Matched rows: {len(result_df):,}")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()