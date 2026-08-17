"""
03_count_duplicates.py

Purpose:
--------
Counts duplicate records across all reduced CSV files before removing anything.

This script checks:
1. pure text duplicates: same tweet_text
2. pure ID duplicates: same tweet_id
3. text + user duplicates: same tweet_text and same user_id

Important:
----------
This script does NOT delete data.
It only creates a duplicate analysis report.

Input:
------
data/processed/02_reduced_columns/

Output:
-------
data/processed/03_duplicate_analysis/duplicate_summary.csv
data/processed/03_duplicate_analysis/duplicate_file_report.csv
"""

from pathlib import Path
import hashlib
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/02_reduced_columns")
OUTPUT_DIR = Path("data/processed/03_duplicate_analysis")

SUMMARY_FILE = OUTPUT_DIR / "duplicate_summary.csv"
FILE_REPORT_FILE = OUTPUT_DIR / "duplicate_file_report.csv"


REQUIRED_COLUMNS = [
    "tweet_id",
    "tweet_text",
    "user_id",
]


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_reduced.csv"))

    if not files:
        raise FileNotFoundError(f"No reduced CSV files found in: {input_dir}")

    return files


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def make_hash(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def count_duplicates() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_input_files(INPUT_DIR)

    print(f"Found {len(files)} reduced files.")
    print("Counting duplicates across all files...")

    text_hashes = []
    tweet_ids = []
    text_user_hashes = []

    file_results = []

    for file_path in tqdm(files, desc="Reading files"):
        result = {
            "source_file": str(file_path),
            "rows_read": 0,
            "status": None,
            "missing_columns": None,
            "error_type": None,
            "error_message": None,
        }

        try:
            df = pd.read_csv(
                file_path,
                usecols=lambda col: col in REQUIRED_COLUMNS,
                low_memory=False,
            )

            missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

            if missing_columns:
                result["status"] = "missing_columns"
                result["missing_columns"] = ", ".join(missing_columns)
                file_results.append(result)
                continue

            result["rows_read"] = len(df)

            text_series = df["tweet_text"].apply(normalize_text)
            user_series = df["user_id"].apply(normalize_id)
            id_series = df["tweet_id"].apply(normalize_id)

            text_hashes.extend(text_series.apply(make_hash).tolist())
            tweet_ids.extend(id_series.tolist())

            combined_text_user = text_series + "||" + user_series
            text_user_hashes.extend(combined_text_user.apply(make_hash).tolist())

            result["status"] = "success"
            file_results.append(result)

        except Exception as error:
            result["status"] = "error"
            result["error_type"] = type(error).__name__
            result["error_message"] = str(error)
            file_results.append(result)

    file_report_df = pd.DataFrame(file_results)
    file_report_df.to_csv(FILE_REPORT_FILE, index=False, encoding="utf-8")

    total_rows = len(text_hashes)

    text_unique = len(set(text_hashes))
    id_unique = len(set(tweet_ids))
    text_user_unique = len(set(text_user_hashes))

    summary = [
        {
            "duplicate_type": "tweet_text",
            "total_rows": total_rows,
            "unique_values": text_unique,
            "duplicate_rows": total_rows - text_unique,
            "duplicate_rate_percent": round((total_rows - text_unique) / total_rows * 100, 4)
            if total_rows > 0 else 0,
        },
        {
            "duplicate_type": "tweet_id",
            "total_rows": total_rows,
            "unique_values": id_unique,
            "duplicate_rows": total_rows - id_unique,
            "duplicate_rate_percent": round((total_rows - id_unique) / total_rows * 100, 4)
            if total_rows > 0 else 0,
        },
        {
            "duplicate_type": "tweet_text_and_user_id",
            "total_rows": total_rows,
            "unique_values": text_user_unique,
            "duplicate_rows": total_rows - text_user_unique,
            "duplicate_rate_percent": round((total_rows - text_user_unique) / total_rows * 100, 4)
            if total_rows > 0 else 0,
        },
    ]

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")

    print("\nDuplicate analysis finished.")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {SUMMARY_FILE}")
    print(f"File report saved to: {FILE_REPORT_FILE}")


if __name__ == "__main__":
    count_duplicates()