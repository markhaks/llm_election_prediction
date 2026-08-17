"""
02_reduce_columns.py

Purpose:
--------
Reduces the raw tweet dataset to the columns required for the thesis project.

This script:
- reads all compressed raw CSV files (.gz)
- keeps only relevant tweet-level information
- extracts relevant user-level information from the "user" column
- writes smaller processed CSV files for later geolocation, LLM classification and SQL import
- logs unreadable or problematic files to a CSV report

Kept tweet columns:
- id: unique tweet identifier
- text: tweet text for LLM classification
- rawContent: original tweet content, used as fallback if text is missing
- date: tweet creation date for time-based analysis
- lang: language filter, mainly English tweets
- likeCount, retweetCount, replyCount, quoteCount: optional engagement signals
- hashtags: optional political/context signal
- user: source column for extracting user metadata

Extracted user columns:
- user_id
- username
- user_location
- user_created
- followers_count
- friends_count
- statuses_count
- verified
- blue

Removed columns:
- url, media, links, mentionedUsers, quotedTweet, conversation IDs,
  reply target IDs and other metadata not directly needed for classification,
  state-level aggregation or prediction.
"""

from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm


RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/02_reduced_columns")
LOG_FILE = OUTPUT_DIR / "processing_report.csv"

TWEET_COLUMNS = [
    "id",
    "text",
    "rawContent",
    "date",
    "lang",
    "likeCount",
    "retweetCount",
    "replyCount",
    "quoteCount",
    "hashtags",
    "user",
]


def find_raw_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("*.gz"))

    if not files:
        raise FileNotFoundError(f"No .gz files found in: {input_dir}")

    return files


def safe_user_text(user_text) -> str:
    if pd.isna(user_text):
        return ""
    return str(user_text)


def extract_string_field(user_text, field_name: str) -> str | None:
    user_text = safe_user_text(user_text)
    pattern = rf"'{field_name}':\s*'([^']*)'"
    match = re.search(pattern, user_text)
    return match.group(1) if match else None


def extract_number_field(user_text, field_name: str) -> int | None:
    user_text = safe_user_text(user_text)
    pattern = rf"'{field_name}':\s*(\d+)"
    match = re.search(pattern, user_text)
    return int(match.group(1)) if match else None


def extract_bool_field(user_text, field_name: str) -> bool | None:
    user_text = safe_user_text(user_text)
    pattern = rf"'{field_name}':\s*(True|False)"
    match = re.search(pattern, user_text)

    if not match:
        return None

    return match.group(1) == "True"


def extract_user_created(user_text) -> str | None:
    user_text = safe_user_text(user_text)

    pattern = r"'created':\s*datetime\.datetime\(([^)]*)\)"
    match = re.search(pattern, user_text)

    if not match:
        return None

    values = match.group(1).split(",")

    try:
        year = int(values[0].strip())
        month = int(values[1].strip())
        day = int(values[2].strip())
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (IndexError, ValueError):
        return None


def extract_user_fields(df: pd.DataFrame) -> pd.DataFrame:
    user_text = df["user"]

    df["user_id"] = user_text.apply(lambda x: extract_number_field(x, "id"))
    df["username"] = user_text.apply(lambda x: extract_string_field(x, "username"))
    df["user_location"] = user_text.apply(lambda x: extract_string_field(x, "location"))
    df["user_created"] = user_text.apply(extract_user_created)

    df["followers_count"] = user_text.apply(lambda x: extract_number_field(x, "followersCount"))
    df["friends_count"] = user_text.apply(lambda x: extract_number_field(x, "friendsCount"))
    df["statuses_count"] = user_text.apply(lambda x: extract_number_field(x, "statusesCount"))

    df["verified"] = user_text.apply(lambda x: extract_bool_field(x, "verified"))
    df["blue"] = user_text.apply(lambda x: extract_bool_field(x, "blue"))

    return df


def build_output_file_path(file_path: Path, output_dir: Path) -> Path:
    relative_path = file_path.relative_to(RAW_DATA_DIR)
    safe_name = "__".join(relative_path.parts)

    if safe_name.endswith(".gz"):
        safe_name = safe_name[:-3]

    output_name = f"{safe_name}_reduced.csv"
    return output_dir / output_name


def reduce_file(file_path: Path, output_dir: Path) -> dict:
    result = {
        "source_file": str(file_path),
        "source_folder": str(file_path.parent),
        "output_file": None,
        "status": None,
        "rows_read": 0,
        "rows_written": 0,
        "missing_columns": None,
        "error_type": None,
        "error_message": None,
    }

    try:
        df = pd.read_csv(
            file_path,
            usecols=lambda col: col in TWEET_COLUMNS,
            low_memory=False,
        )
    except Exception as error:
        result["status"] = "read_error"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        return result

    result["rows_read"] = len(df)

    missing_columns = [col for col in TWEET_COLUMNS if col not in df.columns]

    if missing_columns:
        result["status"] = "missing_columns"
        result["missing_columns"] = ", ".join(missing_columns)
        return result

    try:
        df = extract_user_fields(df)

        reduced_df = df[
            [
                "id",
                "text",
                "rawContent",
                "date",
                "lang",
                "likeCount",
                "retweetCount",
                "replyCount",
                "quoteCount",
                "hashtags",
                "user_id",
                "username",
                "user_location",
                "user_created",
                "followers_count",
                "friends_count",
                "statuses_count",
                "verified",
                "blue",
            ]
        ].copy()

        reduced_df = reduced_df.rename(
            columns={
                "id": "tweet_id",
                "likeCount": "like_count",
                "retweetCount": "retweet_count",
                "replyCount": "reply_count",
                "quoteCount": "quote_count",
            }
        )

        reduced_df["tweet_text"] = reduced_df["text"].fillna(reduced_df["rawContent"])
        reduced_df = reduced_df.drop(columns=["text", "rawContent"])

        output_file = build_output_file_path(file_path, output_dir)
        reduced_df.to_csv(output_file, index=False, encoding="utf-8")

        result["output_file"] = str(output_file)
        result["status"] = "success"
        result["rows_written"] = len(reduced_df)
        return result

    except Exception as error:
        result["status"] = "processing_error"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_raw_files(RAW_DATA_DIR)

    print(f"Found {len(files)} raw files.")
    print(f"Writing reduced files to: {OUTPUT_DIR}")
    print(f"Writing processing report to: {LOG_FILE}")

    results = []

    for file_path in tqdm(files, desc="Reducing files"):
        result = reduce_file(file_path, OUTPUT_DIR)
        results.append(result)

    report_df = pd.DataFrame(results)
    report_df.to_csv(LOG_FILE, index=False, encoding="utf-8")

    success_count = (report_df["status"] == "success").sum()
    error_count = len(report_df) - success_count

    print("Finished reducing raw files.")
    print(f"Successful files: {success_count}")
    print(f"Problematic files: {error_count}")
    print(f"Report saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()