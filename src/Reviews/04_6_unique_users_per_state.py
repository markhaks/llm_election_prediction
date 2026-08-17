"""
04_unique_users_weeks_before_election_by_state.py

Purpose:
--------
Counts unique users per US state in cumulative weekly windows before the
2024 US Election.

Example:
--------
1 week before election  = unique users from 2024-10-29 to 2024-11-04
2 weeks before election = unique users from 2024-10-22 to 2024-11-04
3 weeks before election = unique users from 2024-10-15 to 2024-11-04

Input:
------
data/processed/04_extract_location/all_detected_locations/

Output:
-------
data/processed/analysis_election_windows/unique_users_weeks_before_election_by_state.csv
data/processed/analysis_election_windows/unique_users_weeks_before_election_total.csv
data/processed/analysis_election_windows/file_processing_report.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/04_extract_location/step 5_all_detected_locations")
OUTPUT_DIR = Path("data/processed/analysis_election_windows")

BY_STATE_FILE = OUTPUT_DIR / "unique_users_weeks_before_election_by_state.csv"
TOTAL_FILE = OUTPUT_DIR / "unique_users_weeks_before_election_total.csv"
FILE_REPORT = OUTPUT_DIR / "file_processing_report.csv"

ELECTION_DATE = pd.Timestamp("2024-11-05")
MAX_WEEKS_BEFORE = 12

REQUIRED_COLUMNS = [
    "user_id",
    "user_state",
]


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    return files


def get_date_column(df: pd.DataFrame) -> str | None:
    if "tweet_date" in df.columns:
        return "tweet_date"
    if "date" in df.columns:
        return "date"
    return None


def normalize_state(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def normalize_user_id(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_input_files(INPUT_DIR)

    print(f"Input files found: {len(files)}")
    print(f"Election date: {ELECTION_DATE.date()}")
    print(f"Max weeks before election: {MAX_WEEKS_BEFORE}")

    users_by_window_state = {}
    users_by_window_total = {}
    file_results = []

    for weeks_before in range(1, MAX_WEEKS_BEFORE + 1):
        users_by_window_total[weeks_before] = set()

    for file_path in tqdm(files, desc="Counting unique users"):
        result = {
            "source_file": str(file_path),
            "status": None,
            "rows_read": 0,
            "rows_used": 0,
            "error_type": None,
            "error_message": None,
        }

        try:
            df = pd.read_csv(file_path, low_memory=False)

            date_column = get_date_column(df)

            missing_columns = [
                col for col in REQUIRED_COLUMNS
                if col not in df.columns
            ]

            if date_column is None:
                result["status"] = "missing_date_column"
                result["error_message"] = "No tweet_date/date column found"
                file_results.append(result)
                continue

            if missing_columns:
                result["status"] = "missing_required_columns"
                result["error_message"] = ", ".join(missing_columns)
                file_results.append(result)
                continue

            result["rows_read"] = len(df)

            df["tweet_day"] = pd.to_datetime(
                df[date_column],
                errors="coerce",
            ).dt.normalize()

            df["user_id_norm"] = df["user_id"].apply(normalize_user_id)
            df["user_state_norm"] = df["user_state"].apply(normalize_state)

            df = df[
                df["tweet_day"].notna()
                & (df["user_id_norm"] != "")
                & (df["user_state_norm"] != "")
                & (df["tweet_day"] < ELECTION_DATE)
            ].copy()

            result["rows_used"] = len(df)

            for weeks_before in range(1, MAX_WEEKS_BEFORE + 1):
                window_start = ELECTION_DATE - pd.Timedelta(days=7 * weeks_before)
                window_end = ELECTION_DATE - pd.Timedelta(days=1)

                window_df = df[
                    (df["tweet_day"] >= window_start)
                    & (df["tweet_day"] <= window_end)
                ]

                if window_df.empty:
                    continue

                users_by_window_total[weeks_before].update(
                    window_df["user_id_norm"].tolist()
                )

                for state, group in window_df.groupby("user_state_norm"):
                    key = (weeks_before, state)

                    if key not in users_by_window_state:
                        users_by_window_state[key] = set()

                    users_by_window_state[key].update(
                        group["user_id_norm"].tolist()
                    )

            result["status"] = "success"
            file_results.append(result)

        except Exception as error:
            result["status"] = "error"
            result["error_type"] = type(error).__name__
            result["error_message"] = str(error)
            file_results.append(result)

    by_state_rows = []

    for (weeks_before, state), users in users_by_window_state.items():
        window_start = ELECTION_DATE - pd.Timedelta(days=7 * weeks_before)
        window_end = ELECTION_DATE - pd.Timedelta(days=1)

        by_state_rows.append(
            {
                "weeks_before_election": weeks_before,
                "window_start": window_start.date(),
                "window_end": window_end.date(),
                "user_state": state,
                "unique_users": len(users),
            }
        )

    by_state_df = pd.DataFrame(by_state_rows).sort_values(
        ["weeks_before_election", "user_state"]
    )

    total_rows = []

    for weeks_before, users in users_by_window_total.items():
        window_start = ELECTION_DATE - pd.Timedelta(days=7 * weeks_before)
        window_end = ELECTION_DATE - pd.Timedelta(days=1)

        total_rows.append(
            {
                "weeks_before_election": weeks_before,
                "window_start": window_start.date(),
                "window_end": window_end.date(),
                "unique_users": len(users),
            }
        )

    total_df = pd.DataFrame(total_rows).sort_values("weeks_before_election")
    file_report_df = pd.DataFrame(file_results)

    by_state_df.to_csv(BY_STATE_FILE, index=False, encoding="utf-8")
    total_df.to_csv(TOTAL_FILE, index=False, encoding="utf-8")
    file_report_df.to_csv(FILE_REPORT, index=False, encoding="utf-8")

    print("\nFinished.")
    print(f"By-state report saved to: {BY_STATE_FILE}")
    print(f"Total report saved to: {TOTAL_FILE}")
    print(f"File report saved to: {FILE_REPORT}")


if __name__ == "__main__":
    main()
