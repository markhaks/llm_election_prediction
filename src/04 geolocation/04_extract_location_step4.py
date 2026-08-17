"""
04d_extract_state_only.py

Purpose:
--------
Extracts US states from user_location where no city-state pair was detected.

This script only accepts conservative and clear state matches:
- full state names, e.g. "Texas", "California"
- clear state abbreviations, e.g. "TX", "CA"
- ambiguous abbreviations like LA, DE, IN, OR, ME, HI are only accepted
  with clear US context

Input:
------
data/processed/04_extract_location/step 3/step_3_city_state_pair_not_detected/

Output:
-------
data/processed/04_extract_location/step 4/step_4_state_only_detected/
data/processed/04_extract_location/step 4/step_4_state_only_not_detected/
data/processed/04_extract_location/step 4/state_only_extraction_report.csv
"""

from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm


INPUT_DIR = Path(
    "data/processed/04_extract_location/step 3/step_3_city_state_pair_not_detected"
)

OUTPUT_DIR = Path("data/processed/04_extract_location/step 4")
DETECTED_DIR = OUTPUT_DIR / "step_4_state_only_detected"
NOT_DETECTED_DIR = OUTPUT_DIR / "step_4_state_only_not_detected"
REPORT_FILE = OUTPUT_DIR / "state_only_extraction_report.csv"


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

AMBIGUOUS_STATE_ABBR = {"LA", "DE", "IN", "OR", "ME", "HI"}


def find_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    return files


def normalize_location(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def has_us_context(location: str) -> bool:
    pattern = r"\b(usa|u\.s\.a\.|us|u\.s\.|united states|america)\b"
    return re.search(pattern, location, flags=re.IGNORECASE) is not None


def extract_state_from_name(location: str) -> tuple[str | None, str | None]:
    location_lower = location.lower()
    matches = []

    for state_name, state_abbr in STATE_NAME_TO_ABBR.items():
        pattern = rf"\b{re.escape(state_name)}\b"

        if re.search(pattern, location_lower):
            matches.append((state_abbr, "state_name"))

    unique_states = sorted(set(state for state, _ in matches))

    if len(unique_states) == 1:
        return unique_states[0], "state_name"

    return None, None


def extract_state_from_abbreviation(location: str) -> tuple[str | None, str | None]:
    if not location:
        return None, None

    candidates = []

    location_original = location.strip()

    # Exact abbreviation only, e.g. "TX"
    if location_original in US_STATES:
        candidates.append((location_original, "state_abbreviation_exact"))

    # Abbreviation after comma, e.g. "Austin, TX"
    comma_pattern = r",\s*([A-Z]{2})(?:\s|,|$)"
    for state_abbr in re.findall(comma_pattern, location_original):
        if state_abbr in US_STATES:
            if state_abbr in AMBIGUOUS_STATE_ABBR and not has_us_context(location_original):
                continue

            candidates.append((state_abbr, "state_abbreviation_after_comma"))

    # Abbreviation at the end, e.g. "Austin TX"
    end_pattern = r"\b([A-Z]{2})$"
    match = re.search(end_pattern, location_original)

    if match:
        state_abbr = match.group(1)

        if state_abbr in US_STATES:
            if state_abbr not in AMBIGUOUS_STATE_ABBR or has_us_context(location_original):
                candidates.append((state_abbr, "state_abbreviation_at_end"))

    unique_states = sorted(set(state for state, _ in candidates))

    if len(unique_states) == 1:
        return unique_states[0], candidates[0][1]

    return None, None


def extract_state(location) -> tuple[str | None, str | None]:
    location = normalize_location(location)

    if not location:
        return None, None

    location_lower = location.lower()

    # -----------------------------------------
    # 1. Exact full state name only
    # Examples:
    # "California"
    # "Texas"
    # -----------------------------------------

    if location_lower in STATE_NAME_TO_ABBR:
        return (
            STATE_NAME_TO_ABBR[location_lower],
            "exact_state_name_only",
        )

    # -----------------------------------------
    # 2. Exact full state name + USA
    # Examples:
    # "California, USA"
    # "Texas, usa"
    # -----------------------------------------

    for state_name, state_abbr in STATE_NAME_TO_ABBR.items():

        allowed_patterns = [
            f"{state_name}, usa",
            f"{state_name},us",
            f"{state_name}, u.s.a.",
            f"{state_name}, united states",
        ]

        if location_lower in allowed_patterns:
            return (
                state_abbr,
                "exact_state_name_with_usa",
            )

    # -----------------------------------------
    # 3. Exact state abbreviation only
    # Examples:
    # "CA"
    # "TX"
    # -----------------------------------------

    if location.upper() in US_STATES:
        return (
            location.upper(),
            "exact_state_abbreviation_only",
        )

    # -----------------------------------------
    # 4. Exact state abbreviation + USA
    # Examples:
    # "CA, USA"
    # "TX, usa"
    # -----------------------------------------

    location_upper = location.upper()

    for state_abbr in US_STATES:

        allowed_patterns = [
            f"{state_abbr}, USA",
            f"{state_abbr},US",
            f"{state_abbr}, U.S.A.",
            f"{state_abbr}, UNITED STATES",
        ]

        if location_upper in allowed_patterns:
            return (
                state_abbr,
                "exact_state_abbreviation_with_usa",
            )

    return None, None


def process_file(file_path: Path) -> dict:
    result = {
        "source_file": str(file_path),
        "detected_output_file": None,
        "not_detected_output_file": None,
        "status": None,
        "rows_before": 0,
        "rows_state_detected": 0,
        "rows_state_not_detected": 0,
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

        extracted = df["user_location"].apply(extract_state)

        df["user_state"] = extracted.apply(lambda x: x[0])
        df["user_state_match_method"] = extracted.apply(lambda x: x[1])

        detected_df = df[df["user_state"].notna()].copy()
        not_detected_df = df[df["user_state"].isna()].copy()

        detected_output_file = DETECTED_DIR / file_path.name.replace(
            ".csv",
            "_state_only_detected.csv",
        )

        not_detected_output_file = NOT_DETECTED_DIR / file_path.name.replace(
            ".csv",
            "_state_only_not_detected.csv",
        )

        detected_df.to_csv(detected_output_file, index=False, encoding="utf-8")
        not_detected_df.to_csv(not_detected_output_file, index=False, encoding="utf-8")

        result["detected_output_file"] = str(detected_output_file)
        result["not_detected_output_file"] = str(not_detected_output_file)
        result["rows_state_detected"] = len(detected_df)
        result["rows_state_not_detected"] = len(not_detected_df)
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

    print(f"Input files found: {len(files)}")
    print(f"Detected output folder: {DETECTED_DIR}")
    print(f"Not detected output folder: {NOT_DETECTED_DIR}")

    results = []

    for file_path in tqdm(files, desc="Extracting state-only locations"):
        result = process_file(file_path)
        results.append(result)

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8")

    successful = report_df[report_df["status"] == "success"]

    total_before = successful["rows_before"].sum()
    total_detected = successful["rows_state_detected"].sum()
    total_not_detected = successful["rows_state_not_detected"].sum()

    detection_rate = total_detected / total_before * 100 if total_before > 0 else 0

    print("\nState-only extraction finished.")
    print(f"Rows before: {total_before:,}")
    print(f"Rows state detected: {total_detected:,}")
    print(f"Rows still not detected: {total_not_detected:,}")
    print(f"Detection rate: {detection_rate:.2f}%")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()