"""
05_collect_all_detected_locations.py

Purpose:
--------
Collects all successfully detected location CSV files into one folder.

Source folders:
---------------
1. data/processed/04_extract_location/step 3/step_3_city_state_pair_detected
2. data/processed/04_extract_location/Step 2/city_matched

Output:
-------
data/processed/04_extract_location/all_detected_locations/

Important:
----------
- Files are copied, not moved
- Original files remain unchanged
- Duplicate filenames are handled automatically
"""

from pathlib import Path
import shutil
import pandas as pd


INPUT_DIRS = [
    Path(
        "data/processed/04_extract_location/step 3/step_3_city_state_pair_detected"
    ),
    Path(
        "data/processed/04_extract_location/Step 2/city_matched"
    ),
]

OUTPUT_DIR = Path(
    "data/processed/04_extract_location/all_detected_locations"
)

REPORT_FILE = OUTPUT_DIR / "collect_detected_locations_report.csv"


def get_all_csv_files(input_dirs: list[Path]) -> list[Path]:
    files = []

    for input_dir in input_dirs:

        if not input_dir.exists():
            print(f"WARNING: folder not found: {input_dir}")
            continue

        csv_files = sorted(input_dir.glob("*.csv"))

        files.extend(csv_files)

    if not files:
        raise FileNotFoundError("No CSV files found.")

    return files


def create_unique_filename(target_dir: Path, filename: str) -> str:
    target_path = target_dir / filename

    if not target_path.exists():
        return filename

    stem = target_path.stem
    suffix = target_path.suffix

    counter = 1

    while True:
        new_filename = f"{stem}__duplicate_{counter}{suffix}"

        if not (target_dir / new_filename).exists():
            return new_filename

        counter += 1


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = get_all_csv_files(INPUT_DIRS)

    print(f"CSV files found: {len(files)}")
    print(f"Output folder: {OUTPUT_DIR}")

    results = []

    copied_files = 0
    failed_files = 0

    for file_path in files:

        result = {
            "source_file": str(file_path),
            "target_file": None,
            "status": None,
            "error_type": None,
            "error_message": None,
        }

        try:
            unique_filename = create_unique_filename(
                OUTPUT_DIR,
                file_path.name,
            )

            target_file = OUTPUT_DIR / unique_filename

            shutil.copy2(file_path, target_file)

            result["target_file"] = str(target_file)
            result["status"] = "success"

            copied_files += 1

        except Exception as error:

            result["status"] = "error"
            result["error_type"] = type(error).__name__
            result["error_message"] = str(error)

            failed_files += 1

        results.append(result)

    report_df = pd.DataFrame(results)

    report_df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\nFinished.")
    print(f"Copied files: {copied_files}")
    print(f"Failed files: {failed_files}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()