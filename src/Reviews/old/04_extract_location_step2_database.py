"""
05_check_geolocation_coverage.py

Purpose:
--------
Checks the current geolocation data situation after filtering and city matching.

It reports:
- total rows after location filtering
- rows matched by city
- rows not matched
- match rate
- tweets per state
- tweets per city
- top unresolved locations
- tweets per state per week before the 2024 US election

Election date:
--------------
2024-11-05
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


MATCHED_DIR = Path("data/processed/04_extract_location/Step 2/city_matched")
NOT_MATCHED_DIR = Path("data/processed/04_extract_location/Step 2/city_not_matched")

OUTPUT_DIR = Path("data/processed/05_geolocation_quality")

SUMMARY_FILE = OUTPUT_DIR / "geolocation_summary.csv"
STATE_COUNTS_FILE = OUTPUT_DIR / "tweets_per_state.csv"
CITY_COUNTS_FILE = OUTPUT_DIR / "tweets_per_city.csv"
UNRESOLVED_LOCATIONS_FILE = OUTPUT_DIR / "top_unresolved_locations.csv"
STATE_WEEK_COUNTS_FILE = OUTPUT_DIR / "tweets_per_state_week_before_election.csv"

ELECTION_DATE = pd.Timestamp("2024-11-05")


def find_csv_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        print(f"No CSV files found in: {input_dir}")
        return []

    return files


def count_rows(files: list[Path]) -> int:
    total_rows = 0

    for file_path in tqdm(files, desc="Counting rows"):
        df = pd.read_csv(file_path, usecols=lambda col: True, low_memory=False)
        total_rows += len(df)

    return total_rows


def aggregate_counts(files: list[Path], group_columns: list[str]) -> pd.DataFrame:
    summaries = []

    for file_path in tqdm(files, desc=f"Aggregating {group_columns}"):
        df = pd.read_csv(
            file_path,
            usecols=lambda col: col in group_columns,
            low_memory=False,
        )

        missing_columns = [col for col in group_columns if col not in df.columns]

        if missing_columns:
            print(f"Skipping {file_path}. Missing columns: {missing_columns}")
            continue

        grouped = (
            df.groupby(group_columns, dropna=False)
            .size()
            .reset_index(name="count")
        )

        summaries.append(grouped)

    if not summaries:
        return pd.DataFrame(columns=group_columns + ["count"])

    combined = pd.concat(summaries, ignore_index=True)

    final = (
        combined.groupby(group_columns, dropna=False)["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
    )

    return final


def aggregate_state_week_counts(files: list[Path]) -> pd.DataFrame:
    summaries = []

    required_columns = ["date", "user_state", "user_state_name"]

    for file_path in tqdm(files, desc="Aggregating state/week counts"):
        df = pd.read_csv(
            file_path,
            usecols=lambda col: col in required_columns,
            low_memory=False,
        )

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Skipping {file_path}. Missing columns: {missing_columns}")
            continue

        df["tweet_date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["tweet_date"].notna()].copy()

        df["days_before_election"] = (
            ELECTION_DATE - df["tweet_date"].dt.normalize()
        ).dt.days

        df = df[df["days_before_election"] >= 0].copy()

        df["week_before_election"] = (df["days_before_election"] // 7) + 1

        grouped = (
            df.groupby(
                ["user_state", "user_state_name", "week_before_election"],
                dropna=False,
            )
            .size()
            .reset_index(name="tweet_count")
        )

        summaries.append(grouped)

    if not summaries:
        return pd.DataFrame(
            columns=[
                "user_state",
                "user_state_name",
                "week_before_election",
                "tweet_count",
            ]
        )

    combined = pd.concat(summaries, ignore_index=True)

    final = (
        combined.groupby(
            ["user_state", "user_state_name", "week_before_election"],
            dropna=False,
        )["tweet_count"]
        .sum()
        .reset_index()
        .sort_values(["week_before_election", "user_state"])
    )

    return final


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matched_files = find_csv_files(MATCHED_DIR)
    not_matched_files = find_csv_files(NOT_MATCHED_DIR)

    print(f"Matched files: {len(matched_files)}")
    print(f"Not matched files: {len(not_matched_files)}")

    matched_rows = count_rows(matched_files)
    not_matched_rows = count_rows(not_matched_files)

    total_rows = matched_rows + not_matched_rows
    match_rate = matched_rows / total_rows * 100 if total_rows > 0 else 0

    summary_df = pd.DataFrame(
        [
            {"metric": "matched_rows", "value": matched_rows},
            {"metric": "not_matched_rows", "value": not_matched_rows},
            {"metric": "total_rows_after_filtering", "value": total_rows},
            {"metric": "match_rate_percent", "value": round(match_rate, 4)},
            {"metric": "election_date", "value": str(ELECTION_DATE.date())},
        ]
    )

    summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")

    state_counts = aggregate_counts(
        matched_files,
        ["user_state", "user_state_name"],
    )
    state_counts.to_csv(STATE_COUNTS_FILE, index=False, encoding="utf-8")

    city_counts = aggregate_counts(
        matched_files,
        [
            "user_city",
            "user_state",
            "user_state_name",
            "user_city_population",
        ],
    )
    city_counts.to_csv(CITY_COUNTS_FILE, index=False, encoding="utf-8")

    unresolved_counts = aggregate_counts(
        not_matched_files,
        ["user_location"],
    )
    unresolved_counts.to_csv(
        UNRESOLVED_LOCATIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    state_week_counts = aggregate_state_week_counts(matched_files)
    state_week_counts.to_csv(
        STATE_WEEK_COUNTS_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\nGeolocation quality check finished.")
    print(f"Matched rows: {matched_rows:,}")
    print(f"Not matched rows: {not_matched_rows:,}")
    print(f"Total rows after filtering: {total_rows:,}")
    print(f"Match rate: {match_rate:.2f}%")
    print(f"Election date: {ELECTION_DATE.date()}")
    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()