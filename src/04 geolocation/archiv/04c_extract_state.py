"""
04c_extract_city_locations.py

Purpose:
--------
Detects US states from user_location by matching clear city names efficiently.

Optimization:
-------------
Instead of checking every tweet against every city, this script:
1. builds a unique city lookup from uscities.csv
2. compiles one regex pattern for all unique city names
3. extracts city matches from unique user_location variants only
4. applies the result back to all rows via merge

Input:
------
data/processed/04_extract_location/state_not_detected/

Output:
-------
data/processed/04_extract_location/04c city_state_detected/
data/processed/04_extract_location/city_state_matching_report.csv
data/processed/04_extract_location/city_location_mapping_review.csv
"""

from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm


NOT_DETECTED_DIR = Path("data/processed/04_extract_location/state_not_detected")
CITY_FILE = Path("data/external/uscities.csv")

OUTPUT_DETECTED_DIR = Path(
    "data/processed/04_extract_location/04c city_state_detected"
)

REPORT_FILE = Path(
    "data/processed/04_extract_location/city_state_matching_report.csv"
)

CITY_MAPPING_REVIEW_FILE = Path(
    "data/processed/04_extract_location/city_location_mapping_review.csv"
)


def find_not_detected_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_state_not_detected.csv"))

    if not files:
        raise FileNotFoundError(f"No state-not-detected CSV files found in: {input_dir}")

    return files


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def load_unique_city_mapping(city_file: Path) -> pd.DataFrame:
    if not city_file.exists():
        raise FileNotFoundError(f"City file not found: {city_file}")

    cities_df = pd.read_csv(city_file, low_memory=False)

    required_columns = ["city", "city_ascii", "state_id", "state_name"]

    missing_columns = [
        col for col in required_columns
        if col not in cities_df.columns
    ]

    if missing_columns:
        raise ValueError(f"City file missing columns: {missing_columns}")

    city_variants = []

    for _, row in cities_df.iterrows():
        for city_col in ["city", "city_ascii"]:
            city_name = normalize_text(row[city_col])

            if city_name:
                city_variants.append(
                    {
                        "city_key": city_name,
                        "matched_city": row["city"],
                        "matched_state": row["state_id"],
                        "matched_state_name": row["state_name"],
                    }
                )

    city_map_df = pd.DataFrame(city_variants).drop_duplicates()

    state_counts = (
        city_map_df.groupby("city_key")["matched_state"]
        .nunique()
        .reset_index(name="state_count")
    )

    city_map_df = city_map_df.merge(
        state_counts,
        on="city_key",
        how="left",
    )

    unique_city_map_df = city_map_df[
        city_map_df["state_count"] == 1
    ].copy()

    unique_city_map_df = unique_city_map_df.drop_duplicates(
        subset=["city_key"],
        keep="first",
    )

    # Very short city names are risky, e.g. "us", "go", "me"
    unique_city_map_df = unique_city_map_df[
        unique_city_map_df["city_key"].str.len() >= 4
    ].copy()

    return unique_city_map_df[
        [
            "city_key",
            "matched_city",
            "matched_state",
            "matched_state_name",
        ]
    ]


def build_city_regex(city_mapping_df: pd.DataFrame) -> re.Pattern:
    city_names = city_mapping_df["city_key"].dropna().unique().tolist()

    # Longest first avoids partial matching problems
    city_names = sorted(city_names, key=len, reverse=True)

    escaped_city_names = [re.escape(city) for city in city_names]

    pattern = r"\b(" + "|".join(escaped_city_names) + r")\b"

    return re.compile(pattern, flags=re.IGNORECASE)


def collect_unique_locations(files: list[Path]) -> pd.DataFrame:
    summaries = []

    for file_path in tqdm(files, desc="Collecting unique locations"):
        df = pd.read_csv(
            file_path,
            usecols=["user_location"],
            low_memory=False,
        )

        df["location_key"] = df["user_location"].apply(normalize_text)

        grouped = (
            df.groupby("location_key", dropna=False)
            .size()
            .reset_index(name="count")
        )

        summaries.append(grouped)

    all_locations_df = pd.concat(summaries, ignore_index=True)

    unique_locations_df = (
        all_locations_df.groupby("location_key", dropna=False)["count"]
        .sum()
        .reset_index()
    )

    unique_locations_df = unique_locations_df[
        unique_locations_df["location_key"] != ""
    ].copy()

    return unique_locations_df


def match_unique_locations(
    unique_locations_df: pd.DataFrame,
    city_mapping_df: pd.DataFrame,
    city_regex: re.Pattern,
) -> pd.DataFrame:

    city_lookup = city_mapping_df.set_index("city_key").to_dict(orient="index")

    matched_rows = []

    for _, row in tqdm(
        unique_locations_df.iterrows(),
        total=len(unique_locations_df),
        desc="Matching unique locations",
    ):
        location_key = row["location_key"]

        match = city_regex.search(location_key)

        if not match:
            continue

        matched_city_key = match.group(1).lower()

        if matched_city_key not in city_lookup:
            continue

        city_info = city_lookup[matched_city_key]

        matched_rows.append(
            {
                "location_key": location_key,
                "matched_city_key": matched_city_key,
                "user_city": city_info["matched_city"],
                "user_state": city_info["matched_state"],
                "user_state_match_method": "unique_city_match",
                "count": row["count"],
            }
        )

    if not matched_rows:
        return pd.DataFrame(
            columns=[
                "location_key",
                "matched_city_key",
                "user_city",
                "user_state",
                "user_state_match_method",
                "count",
            ]
        )

    return pd.DataFrame(matched_rows)


def safe_overwrite_csv(df: pd.DataFrame, target_file: Path) -> None:
    backup_file = target_file.with_suffix(target_file.suffix + ".bak")
    temp_file = target_file.with_suffix(target_file.suffix + ".tmp")

    if not backup_file.exists():
        target_file.replace(backup_file)
    else:
        target_file.unlink()

    df.to_csv(temp_file, index=False, encoding="utf-8")
    temp_file.replace(target_file)


def process_file(file_path: Path, location_mapping_df: pd.DataFrame) -> dict:
    result = {
        "source_file": str(file_path),
        "detected_output_file": None,
        "status": None,
        "rows_before": 0,
        "rows_city_detected": 0,
        "rows_remaining_not_detected": 0,
        "error_type": None,
        "error_message": None,
    }

    try:
        df = pd.read_csv(file_path, low_memory=False)

        if "user_location" not in df.columns:
            result["status"] = "missing_user_location"
            result["error_message"] = "Column user_location not found"
            return result

        result["rows_before"] = len(df)

        df["location_key"] = df["user_location"].apply(normalize_text)

        merged_df = df.merge(
            location_mapping_df[
                [
                    "location_key",
                    "user_city",
                    "user_state",
                    "user_state_match_method",
                ]
            ],
            on="location_key",
            how="left",
        )

        detected_df = merged_df[merged_df["user_state"].notna()].copy()
        remaining_df = merged_df[merged_df["user_state"].isna()].copy()

        cleanup_columns = ["location_key"]

        detected_df = detected_df.drop(columns=cleanup_columns, errors="ignore")
        remaining_df = remaining_df.drop(columns=cleanup_columns, errors="ignore")

        detected_output_file = OUTPUT_DETECTED_DIR / file_path.name.replace(
            "_state_not_detected.csv",
            "_city_state_detected.csv",
        )

        if len(detected_df) > 0:
            detected_df.to_csv(
                detected_output_file,
                index=False,
                encoding="utf-8",
            )

        safe_overwrite_csv(remaining_df, file_path)

        result["detected_output_file"] = (
            str(detected_output_file) if len(detected_df) > 0 else None
        )
        result["rows_city_detected"] = len(detected_df)
        result["rows_remaining_not_detected"] = len(remaining_df)
        result["status"] = "success"

        return result

    except Exception as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        return result


def main() -> None:
    OUTPUT_DETECTED_DIR.mkdir(parents=True, exist_ok=True)

    files = find_not_detected_files(NOT_DETECTED_DIR)

    print(f"Not-detected files found: {len(files)}")

    city_mapping_df = load_unique_city_mapping(CITY_FILE)
    print(f"Unique city mappings loaded: {len(city_mapping_df):,}")

    city_regex = build_city_regex(city_mapping_df)
    print("City regex compiled.")

    unique_locations_df = collect_unique_locations(files)
    print(f"Unique unresolved locations collected: {len(unique_locations_df):,}")

    location_mapping_df = match_unique_locations(
        unique_locations_df=unique_locations_df,
        city_mapping_df=city_mapping_df,
        city_regex=city_regex,
    )

    location_mapping_df.to_csv(
        CITY_MAPPING_REVIEW_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"Location mappings found: {len(location_mapping_df):,}")
    print(f"Mapping review saved to: {CITY_MAPPING_REVIEW_FILE}")

    results = []

    for file_path in tqdm(files, desc="Applying city-state mapping"):
        result = process_file(file_path, location_mapping_df)
        results.append(result)

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8")

    successful = report_df[report_df["status"] == "success"]

    total_before = successful["rows_before"].sum()
    total_detected = successful["rows_city_detected"].sum()
    total_remaining = successful["rows_remaining_not_detected"].sum()

    print("\nCity-state matching finished.")
    print(f"Rows before: {total_before:,}")
    print(f"Rows detected by city matching: {total_detected:,}")
    print(f"Rows remaining in not_detected: {total_remaining:,}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()