"""
04d_extract_city_state_pairs_fast.py

Purpose:
--------
Fast extraction of reliable city-state pairs from user_location.

Accepted examples:
------------------
- Sacramento, California
- Sacramento, CA
- CA, Sacramento
- California, Sacramento

Rejected:
---------
- Sacramento
- California
- locations with no comma
- locations with multiple possible matches
- locations where city and state do not belong together

Input:
------
data/processed/04_extract_location/Step 2/city_not_matched/

Reference:
----------
data/external/uscities.csv

Output:
-------
data/processed/04_extract_location/step 3/step_3_city_state_pair_detected/
data/processed/04_extract_location/step 3/step_3_city_state_pair_not_detected/

Report:
-------
data/processed/04_extract_location/step 3/city_state_pair_report.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/04_extract_location/Step 2/city_not_matched")
CITY_FILE = Path("data/external/uscities.csv")

OUTPUT_DIR = Path("data/processed/04_extract_location/step 3")
MATCHED_DIR = OUTPUT_DIR / "step_3_city_state_pair_detected"
NOT_MATCHED_DIR = OUTPUT_DIR / "step_3_city_state_pair_not_detected"

REPORT_FILE = OUTPUT_DIR / "city_state_pair_report.csv"


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_city_not_matched.csv"))

    if not files:
        raise FileNotFoundError(
            f"No city-not-matched files found in: {input_dir}"
        )

    return files


def load_city_state_lookup(city_file: Path) -> dict:
    cities_df = pd.read_csv(city_file, low_memory=False)

    required_columns = [
        "city",
        "city_ascii",
        "state_id",
        "state_name",
        "population",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in cities_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"uscities.csv missing columns: {missing_columns}"
        )

    lookup = {}

    for _, row in cities_df.iterrows():
        state_abbr = normalize_text(row["state_id"])
        state_name = normalize_text(row["state_name"])

        city_variants = {
            normalize_text(row["city"]),
            normalize_text(row["city_ascii"]),
        }

        state_variants = {
            state_abbr,
            state_name,
        }

        for city_key in city_variants:
            if len(city_key) < 4:
                continue

            for state_key in state_variants:
                if not state_key:
                    continue

                pair_key = (city_key, state_key)

                lookup[pair_key] = {
                    "user_city": row["city"],
                    "user_state": row["state_id"],
                    "user_state_name": row["state_name"],
                    "user_city_population": row["population"],
                    "user_state_match_method": "city_state_pair_match",
                }

    return lookup


def split_location_parts(location) -> list[str]:
    location_key = normalize_text(location)

    if not location_key:
        return []

    if "," not in location_key:
        return []

    parts = [
        part.strip()
        for part in location_key.split(",")
        if part.strip()
    ]

    return parts


def extract_city_state_pair(location, lookup: dict):
    parts = split_location_parts(location)

    if len(parts) < 2:
        return None, None, None, None, None

    matches = []

    # Only compare comma-separated parts.
    # This catches:
    # Sacramento, CA
    # CA, Sacramento
    # Sacramento, California
    # California, Sacramento
    for i, left in enumerate(parts):
        for j, right in enumerate(parts):
            if i == j:
                continue

            # city, state
            key_1 = (left, right)

            if key_1 in lookup:
                matches.append(lookup[key_1])

    # remove exact duplicate matches
    unique_matches = []
    seen = set()

    for match in matches:
        match_key = (
            match["user_city"],
            match["user_state"],
            match["user_state_name"],
        )

        if match_key not in seen:
            seen.add(match_key)
            unique_matches.append(match)

    # Only accept exactly one unambiguous match
    if len(unique_matches) != 1:
        return None, None, None, None, None

    match = unique_matches[0]

    return (
        match["user_city"],
        match["user_state"],
        match["user_state_name"],
        match["user_city_population"],
        match["user_state_match_method"],
    )


def process_file(file_path: Path, lookup: dict) -> dict:
    result = {
        "source_file": str(file_path),
        "status": None,
        "rows_before": 0,
        "rows_matched": 0,
        "rows_not_matched": 0,
        "matched_output_file": None,
        "not_matched_output_file": None,
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

        extracted = df["user_location"].apply(
            lambda x: extract_city_state_pair(x, lookup)
        )

        df["user_city"] = extracted.apply(lambda x: x[0])
        df["user_state"] = extracted.apply(lambda x: x[1])
        df["user_state_name"] = extracted.apply(lambda x: x[2])
        df["user_city_population"] = extracted.apply(lambda x: x[3])
        df["user_state_match_method"] = extracted.apply(lambda x: x[4])

        matched_df = df[df["user_state"].notna()].copy()
        not_matched_df = df[df["user_state"].isna()].copy()

        matched_output_file = MATCHED_DIR / file_path.name.replace(
            "_city_not_matched.csv",
            "_city_state_pair_detected.csv",
        )

        not_matched_output_file = NOT_MATCHED_DIR / file_path.name.replace(
            "_city_not_matched.csv",
            "_city_state_pair_not_detected.csv",
        )

        matched_df.to_csv(
            matched_output_file,
            index=False,
            encoding="utf-8",
        )

        not_matched_df.to_csv(
            not_matched_output_file,
            index=False,
            encoding="utf-8",
        )

        result["rows_matched"] = len(matched_df)
        result["rows_not_matched"] = len(not_matched_df)
        result["matched_output_file"] = str(matched_output_file)
        result["not_matched_output_file"] = str(not_matched_output_file)
        result["status"] = "success"

        return result

    except Exception as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        return result


def main() -> None:
    MATCHED_DIR.mkdir(parents=True, exist_ok=True)
    NOT_MATCHED_DIR.mkdir(parents=True, exist_ok=True)

    files = find_input_files(INPUT_DIR)

    print(f"Input files found: {len(files)}")

    lookup = load_city_state_lookup(CITY_FILE)

    print(f"City-state lookup pairs loaded: {len(lookup):,}")

    results = []

    for file_path in tqdm(files, desc="Extracting city-state pairs"):
        result = process_file(
            file_path=file_path,
            lookup=lookup,
        )

        results.append(result)

    report_df = pd.DataFrame(results)

    report_df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8",
    )

    successful = report_df[
        report_df["status"] == "success"
    ]

    total_before = successful["rows_before"].sum()
    total_matched = successful["rows_matched"].sum()
    total_not_matched = successful["rows_not_matched"].sum()

    match_rate = (
        total_matched / total_before * 100
        if total_before > 0
        else 0
    )

    print("\nCity-state pair extraction finished.")
    print(f"Rows before: {total_before:,}")
    print(f"Rows matched: {total_matched:,}")
    print(f"Rows not matched: {total_not_matched:,}")
    print(f"Match rate: {match_rate:.2f}%")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()