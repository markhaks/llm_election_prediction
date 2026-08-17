"""
01_count_tweets_per_day.py

Purpose:
--------
Counts tweets per day and unique users in the globally deduplicated dataset.

Input:
------
data/processed/03_global_deduplicated/

Output:
-------
data/processed/analysis/tweet_user_summary.csv
data/processed/analysis/tweets_per_day.csv
data/processed/analysis/users_per_day.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/03_global_deduplicated")
OUTPUT_DIR = Path("data/processed/analysis")

SUMMARY_FILE = OUTPUT_DIR / "tweet_user_summary.csv"
TWEETS_PER_DAY_FILE = OUTPUT_DIR / "tweets_per_day.csv"
USERS_PER_DAY_FILE = OUTPUT_DIR / "users_per_day.csv"


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_global_deduplicated.csv"))

    if not files:
        raise FileNotFoundError(f"No global deduplicated CSV files found in: {input_dir}")

    return files


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_input_files(INPUT_DIR)

    print(f"Found {len(files)} files.")

    total_tweets = 0
    unique_users = set()

    daily_tweet_counts = []
    daily_user_sets = {}

    for file_path in tqdm(files, desc="Counting tweets and users"):
        try:
            df = pd.read_csv(
                file_path,
                usecols=lambda col: col in ["tweet_date", "date", "user_id"],
                low_memory=False,
            )

            date_column = "tweet_date" if "tweet_date" in df.columns else "date"

            if date_column not in df.columns:
                print(f"Skipping {file_path}: no tweet_date/date column found")
                continue

            if "user_id" not in df.columns:
                print(f"Skipping {file_path}: no user_id column found")
                continue

            df["tweet_day"] = pd.to_datetime(
                df[date_column],
                errors="coerce",
            ).dt.date

            df = df[df["tweet_day"].notna()].copy()

            total_tweets += len(df)

            user_ids = df["user_id"].dropna().astype(str)
            unique_users.update(user_ids)

            daily_counts = (
                df.groupby("tweet_day")
                .size()
                .reset_index(name="tweet_count")
            )

            daily_tweet_counts.append(daily_counts)

            for day, group in df.groupby("tweet_day"):
                if day not in daily_user_sets:
                    daily_user_sets[day] = set()

                daily_user_sets[day].update(
                    group["user_id"].dropna().astype(str)
                )

        except Exception as error:
            print(f"Error processing {file_path}: {error}")

    tweets_per_day = (
        pd.concat(daily_tweet_counts, ignore_index=True)
        .groupby("tweet_day")["tweet_count"]
        .sum()
        .reset_index()
        .sort_values("tweet_day")
    )

    users_per_day = pd.DataFrame(
        [
            {
                "tweet_day": day,
                "unique_users": len(users),
            }
            for day, users in daily_user_sets.items()
        ]
    ).sort_values("tweet_day")

    summary = pd.DataFrame(
        [
            {
                "metric": "total_tweets",
                "value": total_tweets,
            },
            {
                "metric": "unique_users_total",
                "value": len(unique_users),
            },
        ]
    )

    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")
    tweets_per_day.to_csv(TWEETS_PER_DAY_FILE, index=False, encoding="utf-8")
    users_per_day.to_csv(USERS_PER_DAY_FILE, index=False, encoding="utf-8")

    print("\nFinished.")
    print(f"Total tweets: {total_tweets:,}")
    print(f"Unique users total: {len(unique_users):,}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Tweets per day saved to: {TWEETS_PER_DAY_FILE}")
    print(f"Users per day saved to: {USERS_PER_DAY_FILE}")


if __name__ == "__main__":
    main()