"""Extract selected HRS variables and build the combined events dataset.

This script is the standalone Stage 1 equivalent of cell 0 in
``database_events_epig_oct.ipynb``. Variable selection is driven by the
``All_data`` sheet in ``data_description.xlsx`` and the dataset paths defined
in ``config.py``.
"""

import os
import sys
from pathlib import Path

import pandas as pd


# Make the repository-root config.py importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    COMBINED_EVENTS_FILE,
    DATASET_MAP,
    VARIABLE_DESCRIPTION_FILE,
    validate_input_files,
)


def main():
    """Extract configured variables, outer-merge them, and save the result."""
    validate_input_files()
    # Load the variable description file used to select raw HRS variables.
    desc_df = pd.read_excel(
        VARIABLE_DESCRIPTION_FILE,
        sheet_name="All_data",
    )

    # Merge all selected variables into a single participant-level dataframe.
    master_df = None

    for col_name, file_list in DATASET_MAP.items():
        print(f"Processing column: {col_name}")

        raw_vars = desc_df[col_name].dropna().astype(str).tolist()
        split_vars = [
            v.strip().upper()
            for item in raw_vars
            for v in item.split(",")
            if v.strip() and v.strip() != "-"
        ]
        var_list = list(set(split_vars))  # Remove duplicates

        for file_path in file_list:
            print(f"Reading: {file_path}")

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            try:
                # Load .dta and standardize column names to uppercase.
                df = pd.read_stata(file_path)
                df.columns = [col.upper() for col in df.columns]

                # Identify ID variables and requested variables.
                id_vars = [v for v in ["HHID", "PN"] if v in df.columns]
                vars_in_file = [v for v in var_list if v in df.columns]
                selected_vars = id_vars + vars_in_file

                if not id_vars:
                    print(f"⚠️ No HHID/PN in {file_path}, skipping.")
                    continue

                df_selected = df[selected_vars].copy()

                # Preserve the original outer-merge behaviour.
                if master_df is None:
                    master_df = df_selected
                else:
                    master_df = pd.merge(
                        master_df,
                        df_selected,
                        on=id_vars,
                        how="outer",
                    )

            except Exception as exc:
                print(f"Error reading {file_path}: {exc}")

    # Combine PC variables from PGI_A, PGI_E, and PGI_H, where names overlap.
    pc_vars = [
        "PC1_5A",
        "PC1_5B",
        "PC1_5C",
        "PC1_5D",
        "PC1_5E",
        "PC6_10A",
        "PC6_10B",
        "PC6_10C",
        "PC6_10D",
        "PC6_10E",
    ]

    for var in pc_vars:
        # Combine duplicate columns by taking the first non-missing value.
        cols_to_combine = [
            col for col in master_df.columns if col.startswith(var)
        ]
        if len(cols_to_combine) > 1:
            master_df[var] = master_df[cols_to_combine].bfill(axis=1).iloc[:, 0]
            master_df.drop(
                columns=[col for col in cols_to_combine if col != var],
                inplace=True,
            )

    # Save the final combined dataset.
    if master_df is not None:
        master_df.to_csv(COMBINED_EVENTS_FILE, index=False)
        print(f"Final mega dataset saved as '{COMBINED_EVENTS_FILE.name}'")
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()
