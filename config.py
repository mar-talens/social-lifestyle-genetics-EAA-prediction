"""
Configuration file for the database-construction workflow.

Before running the scripts, users must update HRS_DATA_DIR so that it points
to the folder containing their authorised HRS datasets.

The HRS datasets are not distributed with this repository.
"""

from pathlib import Path


# =============================================================================
# 1. PATHS USERS NEED TO CHANGE
# =============================================================================

# Change this path to the folder containing the downloaded HRS datasets.

HRS_DATA_DIR = Path("/path/to/your/HRS_data")


# =============================================================================
# 2. REPOSITORY PATHS
# =============================================================================

# Folder containing this config.py file
DATABASE_CONSTRUCTION_DIR = Path(__file__).resolve().parent

# Root folder of the GitHub repository
PROJECT_DIR = DATABASE_CONSTRUCTION_DIR.parent

# Excel file containing the variable descriptions and extraction information
VARIABLE_DESCRIPTION_FILE = (
    DATABASE_CONSTRUCTION_DIR / "data_description.xlsx"
)

# Folder where intermediate and final datasets will be saved
OUTPUT_DIR = DATABASE_CONSTRUCTION_DIR / "output"

# Create the output folder automatically if it does not already exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3. HRS INPUT DATASETS USED TO BUILD THE PREDICTOR DATABASE
# =============================================================================

DATASET_MAP = {
    "2006_K": [
        HRS_DATA_DIR / "H06B_R.dta",
        HRS_DATA_DIR / "H06C_R.dta",
        HRS_DATA_DIR / "H06LB_R.dta",
    ],

    "2008_L": [
        HRS_DATA_DIR / "H08B_R.dta",
        HRS_DATA_DIR / "H08C_R.dta",
        HRS_DATA_DIR / "H08LB_R.dta",
    ],

    "2010_M": [
        HRS_DATA_DIR / "H10B_R.dta",
        HRS_DATA_DIR / "H10C_R.dta",
        HRS_DATA_DIR / "H10LB_R.dta",
    ],

    "2012_N": [
        HRS_DATA_DIR / "H12B_R.dta",
        HRS_DATA_DIR / "H12C_R.dta",
        HRS_DATA_DIR / "H12LB_R.dta",
    ],

    "2014_O": [
        HRS_DATA_DIR / "H14B_R.dta",
        HRS_DATA_DIR / "H14C_R.dta",
        HRS_DATA_DIR / "H14LB_R.dta",
    ],

    "2016_P": [
        HRS_DATA_DIR / "H16B_R.dta",
        HRS_DATA_DIR / "H16C_R.dta",
        HRS_DATA_DIR / "H16LB_R.dta",
    ],

    "AGGADVTR0619A": [
        HRS_DATA_DIR / "aggadvtr0619a_r.dta",
    ],

    "AGGCHLDFH2016A": [
        HRS_DATA_DIR / "AGGCHLDFH2016A_R.dta",
    ],

    "TRCK": [
        HRS_DATA_DIR / "trk2022tr_r.dta",
    ],

    "LMS15_19": [
        HRS_DATA_DIR / "LHMS1519A_R.dta",
    ],

    "RAND": [
        HRS_DATA_DIR / "randhrs1992_2016v2.dta",
    ],

    "PGI_A": [
        HRS_DATA_DIR / "PGENSCOREA_R.dta",
    ],

    "PGI_E": [
        HRS_DATA_DIR / "PGENSCOREE_R.dta",
    ],

    "PGI_H": [
        HRS_DATA_DIR / "PGENSCOREH_R.dta",
    ],
}


# =============================================================================
# 4. DATASETS USED TO CONSTRUCT THE EPIGENETIC OUTCOMES
# =============================================================================

EPIGENETIC_CLOCKS_FILE = (
    HRS_DATA_DIR / "EPICLOCKA_R.dta"
)

TRACKER_FILE = (
    HRS_DATA_DIR / "trk2022tr_r.dta"
)

# =============================================================================
# 5. OUTPUT FILES
# =============================================================================

COMBINED_EVENTS_FILE = (
    OUTPUT_DIR / "combined_events.csv"
)

CLEAN_EVENTS_FILE = (
    OUTPUT_DIR / "events_df.csv"
)

EPIGENETIC_AGE_FILE = (
    OUTPUT_DIR / "epigenetic_age.csv"
)

EPIGENETIC_AGE_EVENTS_PRE_FILE = (
    OUTPUT_DIR / "epigenetic_age_events_pre.csv"
)

EPIGENETIC_AGE_EVENTS_FILE = (
    OUTPUT_DIR / "epigenetic_age_events.csv"
)

FINAL_ANALYSIS_FILE = (
    OUTPUT_DIR / "analysis_dataset.csv"
)


# =============================================================================
# 6. CHECK THAT REQUIRED FILES EXIST
# =============================================================================

def get_all_required_files():
    """
    Return a list containing every input file required by the workflow.
    """

    predictor_files = [
        file_path
        for dataset_files in DATASET_MAP.values()
        for file_path in dataset_files
    ]

    outcome_files = [
        EPIGENETIC_CLOCKS_FILE,
        TRACKER_FILE,
        VARIABLE_DESCRIPTION_FILE,
    ]

    return predictor_files + outcome_files


def validate_input_files():
    """
    Check whether all required input files can be found.

    Raises
    ------
    FileNotFoundError
        If one or more required files cannot be found.
    """

    missing_files = [
        file_path
        for file_path in get_all_required_files()
        if not file_path.exists()
    ]

    if missing_files:
        formatted_missing_files = "\n".join(
            f"  - {file_path}" for file_path in missing_files
        )

        raise FileNotFoundError(
            "\nThe following required files could not be found:\n"
            f"{formatted_missing_files}\n\n"
            "Update HRS_DATA_DIR and, if necessary, the filenames in config.py."
        )

    print("All required input files were found.")
