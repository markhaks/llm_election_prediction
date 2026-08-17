"""
03_remove_duplicates.py

Purpose:
--------
Removes unwanted duplicate tweets from the processed dataset.

Removed duplicates:
-------------------
1. duplicate tweet_id
2. duplicate combination of:
   - tweet_text
   - user_id

Important:
----------
Identical tweet texts from DIFFERENT users are kept,
because they may represent political agreement, reposting
or organic information spread.

Outputs:
--------
- deduplicated CSV files
- processing report CSV
"""

from pathlib import Path
import hashlib
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/02_reduced_columns")
OUTPUT_DIR = Path("data/processed/03_deduplicated")

REPORT_FILE = OUTPUT_DIR / "deduplication_report.csv"


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


def deduplicate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {
        "rows_before": len(df),
        "tweet_id_duplicates_removed": 0,
        "text_user_duplicates_removed": 0,
        "rows_after": 0,
    }

    # -----------------------------
    # STEP 1: remove duplicate tweet_id
    # -----------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["tweet_id"],
        keep="first",
    ).copy()

    after = len(df)

    stats["tweet_id_duplicates_removed"] = before - after

    # -----------------------------
    # STEP 2: remove duplicate
    # tweet_text + user_id
    # -----------------------------

    df["normalized_text"] = df["tweet_text"].apply(normalize_text)
    df["normalized_user_id"] = df["user_id"].apply(normalize_id)

    df["text_user_key"] = (
        df["normalized_text"] + "||" + df["normalized_user_id"]
    )

    before = len(df)

    df = df.drop_duplicates(
        subset=["text_user_key"],
        keep="first",
    ).copy()

    after = len(df)

    stats["text_user_duplicates_removed"] = before - after

    # cleanup helper columns
    df = df.drop(
        columns=[
            "normalized_text",
            "normalized_user_id",
            "text_user_key",
        ]
    )

    stats["rows_after"] = len(df)

    return df, stats


def process_file(file_path: Path, output_dir: Path) -> dict:
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

        deduplicated_df, stats = deduplicate_dataframe(df)

        output_file = output_dir / file_path.name.replace(
            "_reduced.csv",
            "_deduplicated.csv",
        )

        deduplicated_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8",
        )

        result["output_file"] = str(output_file)

        result["rows_before"] = stats["rows_before"]
        result["rows_after"] = stats["rows_after"]

        result["tweet_id_duplicates_removed"] = (
            stats["tweet_id_duplicates_removed"]
        )

        result["text_user_duplicates_removed"] = (
            stats["text_user_duplicates_removed"]
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
    print(f"Writing deduplicated files to: {OUTPUT_DIR}")

    results = []

    for file_path in tqdm(files, desc="Removing duplicates"):
        result = process_file(file_path, OUTPUT_DIR)
        results.append(result)

    report_df = pd.DataFrame(results)

    report_df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8",
    )

    successful = report_df[report_df["status"] == "success"]

    total_before = successful["rows_before"].sum()
    total_after = successful["rows_after"].sum()

    total_tweet_id_removed = successful[
        "tweet_id_duplicates_removed"
    ].sum()

    total_text_user_removed = successful[
        "text_user_duplicates_removed"
    ].sum()

    total_removed = successful["total_removed"].sum()

    print("\nFinished deduplication.")
    print(f"Rows before: {total_before:,}")
    print(f"Rows after: {total_after:,}")

    print(f"\nRemoved duplicate tweet_id rows: {total_tweet_id_removed:,}")

    print(
        f"Removed duplicate tweet_text + user_id rows: "
        f"{total_text_user_removed:,}"
    )

    print(f"\nTotal removed rows: {total_removed:,}")

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()