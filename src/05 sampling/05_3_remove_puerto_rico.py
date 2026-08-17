"""
08_remove_puerto_rico_tweets.py

Purpose:
--------
Removes tweets from Puerto Rico (PR) from all CSV files.

Input:
------
data/processed/05_sampling/tweets_with_nchs_category/

Output:
-------
data/processed/05_sampling/tweets_without_puerto_rico/

Removed:
--------
- user_state == "PR"
- user_state == "Puerto Rico"

Additionally:
-------------
Creates a processing report with:
- total rows
- removed rows
- remaining rows
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_2 tweets_with_nchs_category"
)

OUTPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_2b tweets_without_puerto_rico"
)

REPORT_FILE = OUTPUT_DIR / "puerto_rico_removal_report.csv"


# Column containing the state
STATE_COLUMN = "user_state"


# Puerto Rico identifiers
PUERTO_RICO_VALUES = {
    "PR",
    "PUERTO RICO",
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_csv_files(INPUT_DIR)

    print(f"Found CSV files: {len(files):,}")

    report_rows = []

    total_rows_all = 0
    total_removed_all = 0
    total_remaining_all = 0

    for file_path in tqdm(files, desc="Removing Puerto Rico tweets"):
        try:
            df = pd.read_csv(file_path, dtype=str)

            if STATE_COLUMN not in df.columns:
                report_rows.append(
                    {
                        "file": file_path.name,
                        "status": "missing_state_column",
                        "rows_total": len(df),
                        "rows_removed": 0,
                        "rows_remaining": len(df),
                    }
                )
                continue

            rows_total = len(df)

            state_normalized = df[STATE_COLUMN].apply(normalize_state)

            remove_mask = state_normalized.isin(PUERTO_RICO_VALUES)

            rows_removed = remove_mask.sum()

            cleaned_df = df[~remove_mask].copy()

            rows_remaining = len(cleaned_df)

            output_file = OUTPUT_DIR / file_path.name

            cleaned_df.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            report_rows.append(
                {
                    "file": file_path.name,
                    "status": "processed",
                    "rows_total": rows_total,
                    "rows_removed": rows_removed,
                    "rows_remaining": rows_remaining,
                }
            )

            total_rows_all += rows_total
            total_removed_all += rows_removed
            total_remaining_all += rows_remaining

        except Exception as e:
            report_rows.append(
                {
                    "file": file_path.name,
                    "status": "error",
                    "rows_total": 0,
                    "rows_removed": 0,
                    "rows_remaining": 0,
                    "error": str(e),
                }
            )

    report_df = pd.DataFrame(report_rows)

    report_df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Report file: {REPORT_FILE}")

    print("\nSummary:")
    print(f"Total rows: {total_rows_all:,}")
    print(f"Puerto Rico rows removed: {total_removed_all:,}")
    print(f"Remaining rows: {total_remaining_all:,}")

    if total_rows_all > 0:
        removal_rate = total_removed_all / total_rows_all
        print(f"Removal rate: {removal_rate:.2%}")


if __name__ == "__main__":
    main()