"""
01_sample_200_tweets_for_manual_labeling.py

Purpose:
--------
Randomly samples 200 tweets from the last 12 weeks before the 2024 US election.

Election Day:
-------------
2024-11-05

Included time window:
---------------------
2024-08-13 <= tweet_date < 2024-11-05
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_2c new data"
)

OUTPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\07_llm_classification"
)

OUTPUT_FILE = OUTPUT_DIR / "manual_labeling_sample_200_last_12_weeks.csv"
REPORT_FILE = OUTPUT_DIR / "manual_labeling_sample_200_last_12_weeks_report.csv"


SAMPLE_SIZE = 500
RANDOM_SEED = 42

ELECTION_DATE = pd.Timestamp("2024-11-05")
START_DATE = ELECTION_DATE - pd.Timedelta(weeks=12)


TEXT_COLUMNS_PRIORITY = [
    "rawContent",
    "text",
    "tweet_text",
]

ID_COLUMNS_PRIORITY = [
    "id",
    "tweet_id",
]

DATE_COLUMNS_PRIORITY = [
    "date",
    "tweet_date",
]


COLUMNS_TO_KEEP_IF_AVAILABLE = [
    "id",
    "tweet_id",
    "date",
    "tweet_date",
    "text",
    "rawContent",
    "lang",
    "user_id",
    "user_name",
    "username",
    "user_location",
    "user_city",
    "user_state",
    "nchs_code",
    "nchs_category_name",
    "likeCount",
    "retweetCount",
    "replyCount",
    "quoteCount",
]


def find_csv_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    return files


def first_existing_column(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def load_file_for_sampling(file_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(file_path, dtype=str, low_memory=False)

    text_col = first_existing_column(df.columns, TEXT_COLUMNS_PRIORITY)
    id_col = first_existing_column(df.columns, ID_COLUMNS_PRIORITY)
    date_col = first_existing_column(df.columns, DATE_COLUMNS_PRIORITY)

    report = {
        "file": file_path.name,
        "status": "loaded",
        "rows_original": len(df),
        "rows_usable_after_text_filter": 0,
        "rows_usable_after_date_filter": 0,
        "text_column_used": text_col,
        "id_column_used": id_col,
        "date_column_used": date_col,
        "error": "",
    }

    if text_col is None:
        report["status"] = "missing_text_column"
        report["error"] = f"No text column found. Available columns: {df.columns.tolist()}"
        return pd.DataFrame(), report

    if date_col is None:
        report["status"] = "missing_date_column"
        report["error"] = f"No date column found. Available columns: {df.columns.tolist()}"
        return pd.DataFrame(), report

    keep_cols = [
        col for col in COLUMNS_TO_KEEP_IF_AVAILABLE
        if col in df.columns
    ]

    sample_df = df[keep_cols].copy()

    sample_df["tweet_text_for_labeling"] = df[text_col].astype(str).str.strip()
    sample_df["source_file"] = file_path.name

    if id_col:
        sample_df["tweet_id_for_labeling"] = df[id_col].astype(str).str.strip()
    else:
        sample_df["tweet_id_for_labeling"] = ""

    sample_df["tweet_date_for_labeling"] = df[date_col].astype(str).str.strip()

    sample_df = sample_df[
        sample_df["tweet_text_for_labeling"].notna()
        & (sample_df["tweet_text_for_labeling"].str.strip() != "")
        & (sample_df["tweet_text_for_labeling"].str.lower() != "nan")
    ].copy()

    report["rows_usable_after_text_filter"] = len(sample_df)

    sample_df["tweet_date_parsed"] = pd.to_datetime(
        sample_df["tweet_date_for_labeling"],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)

    sample_df = sample_df[
        (sample_df["tweet_date_parsed"] >= START_DATE)
        & (sample_df["tweet_date_parsed"] < ELECTION_DATE)
    ].copy()

    report["rows_usable_after_date_filter"] = len(sample_df)

    return sample_df, report


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_csv_files(INPUT_DIR)

    print(f"CSV files found: {len(files):,}")
    print(f"Sampling window: {START_DATE.date()} to {ELECTION_DATE.date()} exclusive")

    all_rows = []
    report_rows = []

    for file_path in tqdm(files, desc="Loading CSV files"):
        try:
            df, report = load_file_for_sampling(file_path)
            report_rows.append(report)

            if not df.empty:
                all_rows.append(df)

        except Exception as e:
            report_rows.append(
                {
                    "file": file_path.name,
                    "status": "error",
                    "rows_original": 0,
                    "rows_usable_after_text_filter": 0,
                    "rows_usable_after_date_filter": 0,
                    "text_column_used": "",
                    "id_column_used": "",
                    "date_column_used": "",
                    "error": str(e),
                }
            )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")

    if not all_rows:
        raise RuntimeError("No usable tweets found in the last 12 weeks before election.")

    all_df = pd.concat(all_rows, ignore_index=True)

    print(f"Usable tweets in date window: {len(all_df):,}")

    if len(all_df) < SAMPLE_SIZE:
        sample_df = all_df.copy()
    else:
        sample_df = all_df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_SEED
        ).copy()

    sample_df = sample_df.reset_index(drop=True)
    sample_df.insert(0, "manual_review_id", range(1, len(sample_df) + 1))

    sample_df["manual_label"] = ""
    sample_df["manual_confidence"] = ""
    sample_df["manual_notes"] = ""

    sample_df["llm_label"] = ""
    sample_df["llm_confidence"] = ""
    sample_df["llm_raw_response"] = ""

    sample_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Sample size: {len(sample_df):,}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Report file: {REPORT_FILE}")


if __name__ == "__main__":
    main()