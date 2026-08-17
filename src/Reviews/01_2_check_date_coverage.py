"""
03_check_date_coverage.py

Purpose:
--------
Analyzes date coverage of processed tweet CSV files.

The script checks per file:
- number of rows
- min tweet date
- max tweet date
- number of distinct days
- tweets per day

Input:
------
data/processed/03_global_deduplicated/

Output:
-------
data/processed/analysis_date_coverage/file_date_coverage.csv
data/processed/analysis_date_coverage/tweets_per_day.csv
data/processed/analysis_date_coverage/date_coverage_summary.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/03_global_deduplicated")
OUTPUT_DIR = Path("data/processed/analysis_date_coverage")

FILE_REPORT = OUTPUT_DIR / "file_date_coverage.csv"
DAY_REPORT = OUTPUT_DIR / "tweets_per_day.csv"
SUMMARY_REPORT = OUTPUT_DIR / "date_coverage_summary.csv"


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_global_deduplicated.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    return files


def get_date_column(df: pd.DataFrame) -> str | None:
    if "tweet_date" in df.columns:
        return "tweet_date"
    if "date" in df.columns:
        return "date"
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_input_files(INPUT_DIR)

    file_results = []
    daily_results = []

    print(f"Found {len(files)} files.")

    for file_path in tqdm(files, desc="Analyzing date coverage"):
        result = {
            "source_file": str(file_path),
            "status": None,
            "rows_total": 0,
            "rows_with_valid_date": 0,
            "min_date": None,
            "max_date": None,
            "distinct_days": 0,
            "error_type": None,
            "error_message": None,
        }

        try:
            df = pd.read_csv(file_path, low_memory=False)

            result["rows_total"] = len(df)

            date_column = get_date_column(df)

            if date_column is None:
                result["status"] = "missing_date_column"
                result["error_message"] = "No tweet_date or date column found"
                file_results.append(result)
                continue

            df["tweet_day"] = pd.to_datetime(
                df[date_column],
                errors="coerce",
            ).dt.date

            valid_df = df[df["tweet_day"].notna()].copy()

            result["rows_with_valid_date"] = len(valid_df)

            if len(valid_df) == 0:
                result["status"] = "no_valid_dates"
                file_results.append(result)
                continue

            result["min_date"] = valid_df["tweet_day"].min()
            result["max_date"] = valid_df["tweet_day"].max()
            result["distinct_days"] = valid_df["tweet_day"].nunique()
            result["status"] = "success"

            daily_counts = (
                valid_df.groupby("tweet_day")
                .size()
                .reset_index(name="tweet_count")
            )

            daily_counts["source_file"] = str(file_path)

            daily_results.append(daily_counts)
            file_results.append(result)

        except Exception as error:
            result["status"] = "error"
            result["error_type"] = type(error).__name__
            result["error_message"] = str(error)
            file_results.append(result)

    file_report_df = pd.DataFrame(file_results)
    file_report_df.to_csv(FILE_REPORT, index=False, encoding="utf-8")

    if daily_results:
        all_daily_df = pd.concat(daily_results, ignore_index=True)

        tweets_per_day_df = (
            all_daily_df.groupby("tweet_day")["tweet_count"]
            .sum()
            .reset_index()
            .sort_values("tweet_day")
        )

        tweets_per_day_df.to_csv(DAY_REPORT, index=False, encoding="utf-8")
    else:
        tweets_per_day_df = pd.DataFrame(columns=["tweet_day", "tweet_count"])
        tweets_per_day_df.to_csv(DAY_REPORT, index=False, encoding="utf-8")

    successful = file_report_df[file_report_df["status"] == "success"]

    summary = {
        "total_files": len(file_report_df),
        "successful_files": len(successful),
        "problematic_files": len(file_report_df) - len(successful),
        "total_rows": file_report_df["rows_total"].sum(),
        "total_rows_with_valid_date": file_report_df["rows_with_valid_date"].sum(),
        "global_min_date": successful["min_date"].min() if len(successful) > 0 else None,
        "global_max_date": successful["max_date"].max() if len(successful) > 0 else None,
        "number_of_days_with_tweets": len(tweets_per_day_df),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_REPORT, index=False, encoding="utf-8")

    print("\nDate coverage analysis finished.")
    print(f"Total files: {summary['total_files']:,}")
    print(f"Successful files: {summary['successful_files']:,}")
    print(f"Problematic files: {summary['problematic_files']:,}")
    print(f"Total rows: {summary['total_rows']:,}")
    print(f"Rows with valid date: {summary['total_rows_with_valid_date']:,}")
    print(f"Global min date: {summary['global_min_date']}")
    print(f"Global max date: {summary['global_max_date']}")
    print(f"Days with tweets: {summary['number_of_days_with_tweets']:,}")
    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()