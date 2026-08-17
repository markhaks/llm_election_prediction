"""
03_global_deduplication.py

Purpose:
--------
Globally removes duplicate tweets across ALL processed CSV files.

Removed duplicates:
-------------------
1. duplicate tweet_id
2. duplicate combination of:
   - tweet_text
   - user_id

Important:
----------
Identical tweet texts from DIFFERENT users are kept,
because they may represent political agreement,
retweets or organic information spread.

This script performs GLOBAL deduplication across
the entire dataset, not just within single files.

Outputs:
--------
- globally deduplicated CSV files
- deduplication report CSV
"""

from pathlib import Path
import hashlib
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/02_reduced_columns")
OUTPUT_DIR = Path("data/processed/03_global_deduplicated")

REPORT_FILE = OUTPUT_DIR / "global_deduplication_report.csv"


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


def process_file(
    file_path: Path,
    output_dir: Path,
    seen_tweet_ids: set,
    seen_text_user: set,
) -> dict:

    result = {
        "source_file": str(file_path),
        "output_file": None,
        "status": None,
        "rows_before": 0,
        "rows_after": 0,
        "tweet_id_duplicates_removed": 0,
        "text_user_duplicates_removed": 0,
        "total_removed": 0,
        "error_type": None,
        "error_message": None,
    }

    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
        )

        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            result["status"] = "missing_columns"
            result["error_message"] = ", ".join(missing_columns)
            return result

        result["rows_before"] = len(df)

        keep_rows = []

        tweet_id_removed = 0
        text_user_removed = 0

        for _, row in df.iterrows():

            tweet_id = normalize_id(row["tweet_id"])

            tweet_text = normalize_text(row["tweet_text"])

            user_id = normalize_id(row["user_id"])

            text_user_key = make_hash(
                tweet_text + "||" + user_id
            )

            # -----------------------------
            # RULE 1:
            # remove duplicate tweet_id
            # -----------------------------

            if tweet_id in seen_tweet_ids:
                tweet_id_removed += 1
                continue

            # -----------------------------
            # RULE 2:
            # remove duplicate
            # tweet_text + user_id
            # -----------------------------

            if text_user_key in seen_text_user:
                text_user_removed += 1
                continue

            seen_tweet_ids.add(tweet_id)
            seen_text_user.add(text_user_key)

            keep_rows.append(row)

        deduplicated_df = pd.DataFrame(keep_rows)

        output_file = output_dir / file_path.name.replace(
            "_reduced.csv",
            "_global_deduplicated.csv",
        )

        deduplicated_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8",
        )

        result["output_file"] = str(output_file)

        result["rows_after"] = len(deduplicated_df)

        result["tweet_id_duplicates_removed"] = (
            tweet_id_removed
        )

        result["text_user_duplicates_removed"] = (
            text_user_removed
        )

        result["total_removed"] = (
            result["rows_before"] - result["rows_after"]
        )

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

    print(f"Found {len(files)} reduced files.")
    print("Starting GLOBAL deduplication...")

    seen_tweet_ids = set()
    seen_text_user = set()

    results = []

    for file_path in tqdm(files, desc="Processing files"):

        result = process_file(
            file_path=file_path,
            output_dir=OUTPUT_DIR,
            seen_tweet_ids=seen_tweet_ids,
            seen_text_user=seen_text_user,
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

    total_after = successful["rows_after"].sum()

    total_tweet_id_removed = successful[
        "tweet_id_duplicates_removed"
    ].sum()

    total_text_user_removed = successful[
        "text_user_duplicates_removed"
    ].sum()

    total_removed = successful[
        "total_removed"
    ].sum()

    print("\nGLOBAL deduplication finished.")

    print(f"\nRows before: {total_before:,}")
    print(f"Rows after: {total_after:,}")

    print(
        f"\nDuplicate tweet_id removed: "
        f"{total_tweet_id_removed:,}"
    )

    print(
        f"Duplicate tweet_text + user_id removed: "
        f"{total_text_user_removed:,}"
    )

    print(f"\nTotal removed rows: {total_removed:,}")

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()