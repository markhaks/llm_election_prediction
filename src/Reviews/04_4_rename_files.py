"""
fix_step3_filenames.py

Purpose:
--------
Renames all files in:

data/processed/04_extract_location/step 3/step_3_city_state_pair_not_detected

to a clean and consistent naming format.

Example:
--------
OLD:
usc-x-24-us-election__part_1__may_july_chunk_1.csv_city_state_pair_not_detected

NEW:
usc-x-24-us-election__part_1__may_july_chunk_1_city_state_pair_not_detected.csv
"""

from pathlib import Path


TARGET_DIR = Path(
    "data/processed/04_extract_location/step 3/step_3_city_state_pair_not_detected"
)


WRONG_PART = ".csv_city_state_pair_not_detected"
CORRECT_PART = "_city_state_pair_not_detected.csv"


def main() -> None:
    if not TARGET_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {TARGET_DIR}")

    files = sorted(TARGET_DIR.iterdir())

    if not files:
        raise FileNotFoundError(f"No files found in: {TARGET_DIR}")

    renamed_count = 0
    skipped_count = 0

    print(f"Files found: {len(files)}\n")

    for file_path in files:

        if not file_path.is_file():
            continue

        old_name = file_path.name

        # only rename files with wrong pattern
        if WRONG_PART not in old_name:
            skipped_count += 1
            continue

        new_name = old_name.replace(
            WRONG_PART,
            CORRECT_PART,
        )

        new_path = file_path.with_name(new_name)

        # safety check
        if new_path.exists():
            print(f"SKIPPED (target exists): {new_name}")
            skipped_count += 1
            continue

        file_path.rename(new_path)

        renamed_count += 1

        print(f"RENAMED")
        print(f"OLD: {old_name}")
        print(f"NEW: {new_name}\n")

    print("Finished.")
    print(f"Renamed files: {renamed_count}")
    print(f"Skipped files: {skipped_count}")


if __name__ == "__main__":
    main()