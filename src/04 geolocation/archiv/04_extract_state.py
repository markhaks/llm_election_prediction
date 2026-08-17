"""
04_extract_location.py

Purpose:
--------
Extracts clear US state information from user_location.

This first rule-based version only detects unambiguous locations:
- full state names, e.g. "Texas"
- state abbreviations, e.g. "TX"

Outputs:
--------
Creates two folders:

data/processed/04_extract_location/state_detected/
- rows where a state was clearly detected

data/processed/04_extract_location/state_not_detected/
- rows where no clear state was detected

Important:
----------
This script does not use fuzzy matching or city matching yet.
Only clear and reproducible state matches are used.
"""

from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path("data/processed/03_global_deduplicated")
OUTPUT_DIR = Path("data/processed/04_extract_location")

DETECTED_DIR = OUTPUT_DIR / "state_detected"
NOT_DETECTED_DIR = OUTPUT_DIR / "state_not_detected"

REPORT_FILE = OUTPUT_DIR / "location_extraction_report.csv"


US_STATES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}

STATE_NAME_TO_ABBR = {
    name.lower(): abbr for abbr, name in US_STATES.items()
}


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_global_deduplicated.csv"))

    if not files:
        raise FileNotFoundError(f"No deduplicated CSV files found in: {input_dir}")

    return files


def normalize_location(location) -> str:
    if pd.isna(location):
        return ""

    return str(location).strip()


def extract_state(location) -> tuple[str | None, str | None]:
    """
    Returns:
    - state abbreviation, e.g. TX
    - match method, e.g. state_name or state_abbreviation
    """
    location = normalize_location(location)

    if not location:
        return None, None

    location_lower = location.lower()

    # 1. Match full state names, e.g. "Texas"
    for state_name, state_abbr in STATE_NAME_TO_ABBR.items():
        pattern = rf"\b{re.escape(state_name)}\b"

        if re.search(pattern, location_lower):
            return state_abbr, "state_name"

    # 2. Match state abbreviations, e.g. "TX"
    # Split by common separators to avoid matching random letters inside words.
    tokens = re.split(r"[\s,.;:/|()\[\]{}\-]+", location.upper())

    for token in tokens:
        if token in US_STATES:
            return token, "state_abbreviation"

    return None, None


def process_file(file_path: Path) -> dict:
    result = {
        "source_file": str(file_path),
        "status": None,
        "rows_total": 0,
        "rows_state_detected": 0,
        "rows_state_not_detected": 0,
        "detected_output_file": None,
        "not_detected_output_file": None,
        "error_type": None,
        "error_message": None,
    }

    try:
        df = pd.read_csv(file_path, low_memory=False)

        if "user_location" not in df.columns:
            result["status"] = "missing_user_location"
            result["error_message"] = "Column user_location not found"
            return result

        result["rows_total"] = len(df)

        extracted = df["user_location"].apply(extract_state)

        df["user_state"] = extracted.apply(lambda x: x[0])
        df["user_state_match_method"] = extracted.apply(lambda x: x[1])

        detected_df = df[df["user_state"].notna()].copy()
        not_detected_df = df[df["user_state"].isna()].copy()

        detected_output_file = DETECTED_DIR / file_path.name.replace(
            "_global_deduplicated.csv",
            "_state_detected.csv",
        )

        not_detected_output_file = NOT_DETECTED_DIR / file_path.name.replace(
            "_global_deduplicated.csv",
            "_state_not_detected.csv",
        )

        detected_df.to_csv(detected_output_file, index=False, encoding="utf-8")
        not_detected_df.to_csv(not_detected_output_file, index=False, encoding="utf-8")

        result["rows_state_detected"] = len(detected_df)
        result["rows_state_not_detected"] = len(not_detected_df)
        result["detected_output_file"] = str(detected_output_file)
        result["not_detected_output_file"] = str(not_detected_output_file)
        result["status"] = "success"

        return result

    except Exception as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        return result


def main() -> None:
    DETECTED_DIR.mkdir(parents=True, exist_ok=True)
    NOT_DETECTED_DIR.mkdir(parents=True, exist_ok=True)

    files = find_input_files(INPUT_DIR)

    print(f"Found {len(files)} globally deduplicated files.")
    print(f"State detected files will be saved to: {DETECTED_DIR}")
    print(f"State not detected files will be saved to: {NOT_DETECTED_DIR}")

    results = []

    for file_path in tqdm(files, desc="Extracting states"):
        result = process_file(file_path)
        results.append(result)

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8")

    successful = report_df[report_df["status"] == "success"]

    total_rows = successful["rows_total"].sum()
    total_detected = successful["rows_state_detected"].sum()
    total_not_detected = successful["rows_state_not_detected"].sum()

    detection_rate = (
        total_detected / total_rows * 100
        if total_rows > 0
        else 0
    )

    print("\nLocation extraction finished.")
    print(f"Total rows: {total_rows:,}")
    print(f"State detected: {total_detected:,}")
    print(f"State not detected: {total_not_detected:,}")
    print(f"Detection rate: {detection_rate:.2f}%")
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()