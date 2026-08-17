"""
02_count_tweets_users_per_day_state.py

Purpose:
--------
Counts tweets and unique users by day and by detected US state.

Input folders:
--------------
1. data/processed/04_extract_location/step 3/step_3_city_state_pair_detected
2. data/processed/04_extract_location/Step 2/city_matched

Output:
-------
data/processed/analysis_state/tweet_user_state_summary.csv
data/processed/analysis_state/tweets_per_day_state.csv
data/processed/analysis_state/users_per_day_state.csv
data/processed/analysis_state/tweets_per_state.csv
data/processed/analysis_state/users_per_state.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


INPUT_DIRS = [
    Path("data/processed/04_extract_location/step 3/step_3_city_state_pair_detected"),
    Path("data/processed/04_extract_location/Step 2/city_matched"),
]

OUTPUT_DIR = Path("data/processed/analysis_state")

SUMMARY_FILE = OUTPUT_DIR / "tweet_user_state_summary.csv"
TWEETS_PER_DAY_STATE_FILE = OUTPUT_DIR / "tweets_per_day_state.csv"
USERS_PER_DAY_STATE_FILE = OUTPUT_DIR / "users_per_day_state.csv"
TWEETS_PER_STATE_FILE = OUTPUT_DIR / "tweets_per_state.csv"
USERS_PER_STATE_FILE = OUTPUT_DIR / "users_per_state.csv"


def find_input_files(input_dirs: list[Path]) -> list[Path]:
    files = []

    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"Warning: input folder does not exist: {input_dir}")
            continue

        files.extend(sorted(input_dir.glob("*.csv")))

    if not files:
        raise FileNotFoundError("No CSV files found in the input folders.")

    return files


def get_date_column(df: pd.DataFrame) -> str | None:
    if "tweet_date" in df.columns:
        return "tweet_date"
    if "date" in df.columns:
        return "date"
    return None


def get_state_column(df: pd.DataFrame) -> str | None:
    if "user_state" in df.columns:
        return "user_state"
    if "state" in df.columns:
        return "state"
    return None


def normalize_state(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_input_files(INPUT_DIRS)

    print(f"Found {len(files)} CSV files.")

    total_tweets = 0
    unique_users_total = set()

    daily_state_tweet_counts = []
    state_tweet_counts = []

    daily_state_user_sets = {}
    state_user_sets = {}

    file_results = []

    for file_path in tqdm(files, desc="Counting by day and state"):
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
            state_column = get_state_column(df)

            if date_column is None:
                result["status"] = "missing_date_column"
                result["error_message"] = "No tweet_date/date column found"
                file_results.append(result)
                continue

            if "user_id" not in df.columns:
                result["status"] = "missing_user_id"
                result["error_message"] = "No user_id column found"
                file_results.append(result)
                continue

            if state_column is None:
                result["status"] = "missing_state_column"
                result["error_message"] = "No user_state/state column found"
                file_results.append(result)
                continue

            result["rows_read"] = len(df)

            df["tweet_day"] = pd.to_datetime(
                df[date_column],
                errors="coerce",
            ).dt.date

            df["state_norm"] = df[state_column].apply(normalize_state)

            df = df[
                df["tweet_day"].notna()
                & (df["state_norm"] != "")
            ].copy()

            result["rows_used"] = len(df)

            total_tweets += len(df)

            user_ids = df["user_id"].dropna().astype(str)
            unique_users_total.update(user_ids)

            daily_counts = (
                df.groupby(["tweet_day", "state_norm"])
                .size()
                .reset_index(name="tweet_count")
            )
            daily_state_tweet_counts.append(daily_counts)

            state_counts = (
                df.groupby("state_norm")
                .size()
                .reset_index(name="tweet_count")
            )
            state_tweet_counts.append(state_counts)

            for (day, state), group in df.groupby(["tweet_day", "state_norm"]):
                key = (day, state)

                if key not in daily_state_user_sets:
                    daily_state_user_sets[key] = set()

                daily_state_user_sets[key].update(
                    group["user_id"].dropna().astype(str)
                )

            for state, group in df.groupby("state_norm"):
                if state not in state_user_sets:
                    state_user_sets[state] = set()

                state_user_sets[state].update(
                    group["user_id"].dropna().astype(str)
                )

            result["status"] = "success"
            file_results.append(result)

        except Exception as error:
            result["status"] = "error"
            result["error_type"] = type(error).__name__
            result["error_message"] = str(error)
            file_results.append(result)

    tweets_per_day_state = (
        pd.concat(daily_state_tweet_counts, ignore_index=True)
        .groupby(["tweet_day", "state_norm"])["tweet_count"]
        .sum()
        .reset_index()
        .rename(columns={"state_norm": "user_state"})
        .sort_values(["tweet_day", "user_state"])
    )

    tweets_per_state = (
        pd.concat(state_tweet_counts, ignore_index=True)
        .groupby("state_norm")["tweet_count"]
        .sum()
        .reset_index()
        .rename(columns={"state_norm": "user_state"})
        .sort_values("tweet_count", ascending=False)
    )

    users_per_day_state = pd.DataFrame(
        [
            {
                "tweet_day": day,
                "user_state": state,
                "unique_users": len(users),
            }
            for (day, state), users in daily_state_user_sets.items()
        ]
    ).sort_values(["tweet_day", "user_state"])

    users_per_state = pd.DataFrame(
        [
            {
                "user_state": state,
                "unique_users": len(users),
            }
            for state, users in state_user_sets.items()
        ]
    ).sort_values("unique_users", ascending=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "total_tweets_with_state",
                "value": total_tweets,
            },
            {
                "metric": "unique_users_total_with_state",
                "value": len(unique_users_total),
            },
            {
                "metric": "number_of_states",
                "value": tweets_per_state["user_state"].nunique(),
            },
        ]
    )

    file_report = pd.DataFrame(file_results)

    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")
    tweets_per_day_state.to_csv(TWEETS_PER_DAY_STATE_FILE, index=False, encoding="utf-8")
    users_per_day_state.to_csv(USERS_PER_DAY_STATE_FILE, index=False, encoding="utf-8")
    tweets_per_state.to_csv(TWEETS_PER_STATE_FILE, index=False, encoding="utf-8")
    users_per_state.to_csv(USERS_PER_STATE_FILE, index=False, encoding="utf-8")
    file_report.to_csv(OUTPUT_DIR / "file_processing_report.csv", index=False, encoding="utf-8")

    print("\nFinished.")
    print(f"Total tweets with state: {total_tweets:,}")
    print(f"Unique users with state: {len(unique_users_total):,}")
    print(f"Reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()