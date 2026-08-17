"""
04_city_match_unique.py

Purpose:
--------
Matches remaining user_location values against unique US city names.

A row is accepted only if:
1. exactly one unique city name is found in user_location
2. this city exists in only one US state in uscities.csv

If no city or multiple city names are found, the row is not matched.

Input:
------
data/processed/03b_filtered_locations/
data/external/uscities.csv

Output:
-------
data/processed/04_city_matched/
data/processed/04_city_matched/city_matching_report.csv
"""

from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/04_extract_location/Step 1")
CITY_FILE = Path("data/external/uscities.csv")

OUTPUT_DIR = Path("data/processed/extract_location/Step 2")
MATCHED_DIR = OUTPUT_DIR / "city_matched"
NOT_MATCHED_DIR = OUTPUT_DIR / "city_not_matched"

REPORT_FILE = OUTPUT_DIR / "city_matching_report.csv"


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_location_filtered.csv"))

    if not files:
        raise FileNotFoundError(f"No filtered CSV files found in: {input_dir}")

    return files


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def load_unique_city_mapping(city_file: Path) -> pd.DataFrame:
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
        raise ValueError(f"uscities.csv missing columns: {missing_columns}")

    city_variants = []

    for _, row in cities_df.iterrows():
        for city_col in ["city", "city_ascii"]:
            city_key = normalize_text(row[city_col])

            if city_key:
                city_variants.append(
                    {
                        "city_key": city_key,
                        "matched_city": row["city"],
                        "matched_state": row["state_id"],
                        "matched_state_name": row["state_name"],
                        "matched_city_population": row["population"],
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

    # Avoid risky very short city names
    unique_city_map_df = unique_city_map_df[
        unique_city_map_df["city_key"].str.len() >= 4
    ].copy()

    return unique_city_map_df[
        [
            "city_key",
            "matched_city",
            "matched_state",
            "matched_state_name",
            "matched_city_population",
        ]
    ]


def build_city_regex(city_mapping_df: pd.DataFrame) -> re.Pattern:
    city_names = city_mapping_df["city_key"].dropna().unique().tolist()

    city_names = sorted(city_names, key=len, reverse=True)
    escaped_city_names = [re.escape(city) for city in city_names]

    pattern = r"\b(" + "|".join(escaped_city_names) + r")\b"

    return re.compile(pattern, flags=re.IGNORECASE)


def extract_single_city_match(
    location,
    city_regex: re.Pattern,
    city_lookup: dict,
) -> tuple[str | None, str | None, str | None, float | None, str | None]:

    location_key = normalize_text(location)

    if not location_key:
        return None, None, None, None, None

    matches = city_regex.findall(location_key)

    unique_matches = sorted(set(match.lower() for match in matches))

    if len(unique_matches) != 1:
        return None, None, None, None, None

    city_key = unique_matches[0]

    if city_key not in city_lookup:
        return None, None, None, None, None

    city_info = city_lookup[city_key]

    return (
        city_info["matched_city"],
        city_info["matched_state"],
        city_info["matched_state_name"],
        city_info["matched_city_population"],
        "unique_city_match",
    )


def process_file(
    file_path: Path,
    city_regex: re.Pattern,
    city_lookup: dict,
) -> dict:

    result = {
        "source_file": str(file_path),
        "matched_output_file": None,
        "not_matched_output_file": None,
        "status": None,
        "rows_before": 0,
        "rows_city_matched": 0,
        "rows_not_matched": 0,
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
            lambda x: extract_single_city_match(
                x,
                city_regex,
                city_lookup,
            )
        )

        df["user_city"] = extracted.apply(lambda x: x[0])
        df["user_state"] = extracted.apply(lambda x: x[1])
        df["user_state_name"] = extracted.apply(lambda x: x[2])
        df["user_city_population"] = extracted.apply(lambda x: x[3])
        df["user_state_match_method"] = extracted.apply(lambda x: x[4])

        matched_df = df[df["user_state"].notna()].copy()
        not_matched_df = df[df["user_state"].isna()].copy()

        matched_output_file = MATCHED_DIR / file_path.name.replace(
            "_location_filtered.csv",
            "_city_matched.csv",
        )

        not_matched_output_file = NOT_MATCHED_DIR / file_path.name.replace(
            "_location_filtered.csv",
            "_city_not_matched.csv",
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

        result["matched_output_file"] = str(matched_output_file)
        result["not_matched_output_file"] = str(not_matched_output_file)
        result["rows_city_matched"] = len(matched_df)
        result["rows_not_matched"] = len(not_matched_df)
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

    city_mapping_df = load_unique_city_mapping(CITY_FILE)
    city_regex = build_city_regex(city_mapping_df)

    city_lookup = city_mapping_df.set_index("city_key").to_dict(orient="index")

    print(f"Input files found: {len(files)}")
    print(f"Unique city mappings loaded: {len(city_mapping_df):,}")
    print(f"Matched output folder: {MATCHED_DIR}")
    print(f"Not matched output folder: {NOT_MATCHED_DIR}")

    results = []

    for file_path in tqdm(files, desc="Matching city locations"):
        result = process_file(
            file_path=file_path,
            city_regex=city_regex,
            city_lookup=city_lookup,
        )
        results.append(result)

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8")

    successful = report_df[report_df["status"] == "success"]

    total_before = successful["rows_before"].sum()
    total_matched = successful["rows_city_matched"].sum()
    total_not_matched = successful["rows_not_matched"].sum()

    match_rate = total_matched / total_before * 100 if total_before > 0 else 0

    print("\nCity matching finished.")
    print(f"Rows before: {total_before:,}")
    print(f"Rows city matched: {total_matched:,}")
    print(f"Rows not matched: {total_not_matched:,}")
    print(f"Match rate: {match_rate:.2f}%")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()