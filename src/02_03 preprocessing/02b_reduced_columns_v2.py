"""
02b_process_missing_retweet_columns.py

Purpose:
--------
Reprocesses files where retweetedTweetID and retweetedUserID are missing.

This script:
- loads problematic raw files
- creates missing retweet columns
- fills missing values with "NOT_AVAILABLE"
- runs the normal reduction pipeline
- saves corrected reduced files
- creates a processing report
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re


OUTPUT_DIR = Path("data/processed/02_reduced_columns")
REPORT_FILE = OUTPUT_DIR / "missing_retweet_column_report.csv"


FILES_TO_REPROCESS = [
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_1.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_10.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_11.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_12.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_13.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_14.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_15.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_16.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_17.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_18.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_19.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_2.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_20.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_3.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_4.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_5.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_6.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_7.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_8.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_45/octobergap_chunk_9.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_21.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_22.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_23.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_24.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_25.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_26.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_27.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_28.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_29.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_30.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_31.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_32.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_33.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_34.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_35.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_36.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_37.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_38.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_39.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_46/octobergap_chunk_40.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_47/octobergap_chunk_41.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_47/octobergap_chunk_42.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_47/octobergap_chunk_43.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_47/octobergap_chunk_44.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_47/octobergap_chunk_45.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_47/octobergap_chunk_46.csv.gz"),
    Path(r"data/raw/usc-x-24-us-election/part_47/octobergap_chunk_47.csv.gz"),
]


def safe_user_text(user_text) -> str:
    if pd.isna(user_text):
        return ""
    return str(user_text)


def extract_string_field(user_text, field_name: str):
    user_text = safe_user_text(user_text)
    pattern = rf"'{field_name}':\s*'([^']*)'"
    match = re.search(pattern, user_text)
    return match.group(1) if match else None


def extract_number_field(user_text, field_name: str):
    user_text = safe_user_text(user_text)
    pattern = rf"'{field_name}':\s*(\d+)"
    match = re.search(pattern, user_text)
    return int(match.group(1)) if match else None


def extract_bool_field(user_text, field_name: str):
    user_text = safe_user_text(user_text)
    pattern = rf"'{field_name}':\s*(True|False)"
    match = re.search(pattern, user_text)

    if not match:
        return None

    return match.group(1) == "True"


def extract_user_created(user_text):
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
    except:
        return None


def process_file(file_path: Path) -> dict:

    result = {
        "source_file": str(file_path),
        "status": None,
        "rows_written": 0,
        "error_type": None,
        "error_message": None,
    }

    try:
        df = pd.read_csv(file_path, low_memory=False)

        if "retweetedTweetID" not in df.columns:
            df["retweetedTweetID"] = "NOT_AVAILABLE"

        if "retweetedUserID" not in df.columns:
            df["retweetedUserID"] = "NOT_AVAILABLE"

        user_text = df["user"]

        df["user_id"] = user_text.apply(lambda x: extract_number_field(x, "id"))
        df["user_username"] = user_text.apply(lambda x: extract_string_field(x, "username"))
        df["user_location"] = user_text.apply(lambda x: extract_string_field(x, "location"))
        df["user_created"] = user_text.apply(extract_user_created)

        df["user_followers_count"] = user_text.apply(lambda x: extract_number_field(x, "followersCount"))
        df["user_friends_count"] = user_text.apply(lambda x: extract_number_field(x, "friendsCount"))
        df["user_statuses_count"] = user_text.apply(lambda x: extract_number_field(x, "statusesCount"))

        df["user_verified"] = user_text.apply(lambda x: extract_bool_field(x, "verified"))
        df["user_blue"] = user_text.apply(lambda x: extract_bool_field(x, "blue"))

        reduced_df = pd.DataFrame({
            "tweet_id": df["id"],
            "tweet_text": df["text"].fillna(df["rawContent"]),
            "tweet_date": df["date"],
            "tweet_lang": df["lang"],
            "tweet_like_count": df["likeCount"],
            "tweet_retweet_count": df["retweetCount"],
            "tweet_reply_count": df["replyCount"],
            "tweet_quote_count": df["quoteCount"],
            "tweet_hashtags": df["hashtags"],
            "tweet_is_retweet": df["retweetedTweet"],
            "tweet_retweeted_tweet_id": df["retweetedTweetID"],
            "tweet_retweeted_user_id": df["retweetedUserID"],
            "user_id": df["user_id"],
            "user_username": df["user_username"],
            "user_location": df["user_location"],
            "user_created": df["user_created"],
            "user_followers_count": df["user_followers_count"],
            "user_friends_count": df["user_friends_count"],
            "user_statuses_count": df["user_statuses_count"],
            "user_verified": df["user_verified"],
            "user_blue": df["user_blue"],
        })

        output_file = OUTPUT_DIR / f"{file_path.stem}_reduced.csv"

        reduced_df.to_csv(output_file, index=False, encoding="utf-8")

        result["status"] = "success"
        result["rows_written"] = len(reduced_df)

        return result

    except Exception as error:

        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)

        return result


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for file_path in tqdm(FILES_TO_REPROCESS, desc="Reprocessing files"):

        result = process_file(file_path)
        results.append(result)

    report_df = pd.DataFrame(results)

    report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8")

    print("Finished reprocessing missing retweet columns.")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()