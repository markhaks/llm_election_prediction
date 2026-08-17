"""
11_split_review_file_by_state.py

Purpose:
--------
Splits one CSV file into separate CSV files by state.

Input:
------
matched_state_nchs_review.csv

Output:
-------
One CSV per state, for example:
AK.csv
AR.csv
DE.csv
TX.csv
"""

from pathlib import Path
import pandas as pd


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_FILE = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\matched_state_nchs_review.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\matched_state_nchs_review_by_state"
)

STATE_COLUMN = "user_state"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, dtype=str)

    if STATE_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{STATE_COLUMN}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df[STATE_COLUMN] = df[STATE_COLUMN].astype(str).str.strip().str.upper()

    for state, state_df in df.groupby(STATE_COLUMN):
        if not state:
            continue

        output_file = OUTPUT_DIR / f"{state}.csv"

        state_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"{state}: {len(state_df):,} rows written")

    print("\nDone.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()