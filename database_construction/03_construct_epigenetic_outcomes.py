"""Construct epigenetic age outcomes and merge them with predictor data.

This script preserves the analytical order of cell 2 in
``database_events_epig_oct.ipynb``. It uses only the HRS Epigenetic Clocks
and Tracker inputs for outcome construction.
"""

#########################################################################################################
####### CREATING THE EPIGENETIC AGE ACCELERATION VARIABLES #######
####### input: see load data section
####### output: 
####### - epigenetic_age.csv: EEAA values with core covariates
####### - epigenetic_age_events.csv: EEAA merged with life event variables
####### last update: 24/05/2025
#########################################################################################################

import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    CLEAN_EVENTS_FILE,
    EPIGENETIC_AGE_EVENTS_PRE_FILE,
    EPIGENETIC_AGE_FILE,
    EPIGENETIC_CLOCKS_FILE,
    TRACKER_FILE,
)



def main():
    ########################################
    # 1. Load Data
    ########################################

    epigenetic_clocks = pd.read_stata(EPIGENETIC_CLOCKS_FILE)
    track = pd.read_stata(TRACKER_FILE)

    events_df = pd.read_csv(CLEAN_EVENTS_FILE)
    # Standardize column names
    for df in [epigenetic_clocks, track, events_df]:
        df.columns = df.columns.str.upper()

    ########################################
    # 2. Select and Merge Relevant Variables
    ########################################

    track_vars = ["HHID", "PN", "BIRTHMO", "BIRTHYR", "GENDER"]
    clock_vars = ["HHID", "PN", "LEVINE_DNAMAGE", "DNAMGRIMAGE", "MPOA", 'HORVATH_DNAMAGE', 'HANNUM_DNAMAGE']

    track_sub = track[track_vars]
    clocks_sub = epigenetic_clocks[clock_vars]

    # Merge
    df = track_sub.merge(clocks_sub, on=["HHID", "PN"], how="inner")

    # Compute chronological age (midpoint of birth month = 15th assumed)
    df["CHRON_AGE"] = 2016 - df["BIRTHYR"] + (1 - df["BIRTHMO"]) / 12

    ########################################
    # 3. Define Residualization Function
    ########################################

    def compute_residuals(df, outcome, covariates):
        X = df[covariates].copy()
        X = sm.add_constant(X)
        y = df[outcome]
        model = sm.OLS(y, X, missing='drop').fit()
        return model.resid

    ########################################
    # 4. Compute Epigenetic Age Acceleration (EEAA)
    ########################################

    covariates = ["CHRON_AGE"] 

    df["EAA_LEVINE"] = compute_residuals(df, "LEVINE_DNAMAGE", covariates)
    df["EAA_GRIMAGE"] = compute_residuals(df, "DNAMGRIMAGE", covariates)
    df["EAA_HORVATH"] = compute_residuals(df, "HORVATH_DNAMAGE", covariates)
    df["EAA_HANNUM"] = compute_residuals(df, "HANNUM_DNAMAGE", covariates)
    #MPOA does not need to be regressed
    df["EAA_DUNEDINMPOA"] = df["MPOA"]

    epi_clocks = ['EAA_LEVINE', 'EAA_GRIMAGE', 'EAA_HORVATH', 'EAA_HANNUM', 'EAA_DUNEDINMPOA']
    for col in epi_clocks:
        if col in df.columns:
            median = df[col].median()
            q1 = df[col].quantile(1/3)
            q2 = df[col].quantile(2/3)
            df[col + '_TERTILE'] = np.where(df[col].isna(), np.nan,
                np.where(df[col] <= q1, 0,
                np.where(df[col] <= q2, 1, 2)))
            df[col + '_BINARY'] = np.where(df[col].isna(), np.nan, np.where(df[col] > median, 1, 0))

    ########################################
    # 5. Export EEAA Dataset
    ########################################

    OUTPUT_PATH = EPIGENETIC_AGE_FILE
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Epigenetic age file saved to: {OUTPUT_PATH}")

    ########################################
    # 6. Merge EEAA with Life Events
    ########################################
    eaa_vars = ['HHID', 'PN', 'EAA_LEVINE', 'EAA_GRIMAGE', 'EAA_DUNEDINMPOA', 'EAA_HORVATH', 'EAA_HANNUM', 'EAA_LEVINE_BINARY', 'EAA_GRIMAGE_BINARY', 'EAA_DUNEDINMPOA_BINARY', 'EAA_HORVATH_BINARY', 'EAA_HANNUM_BINARY', 'EAA_LEVINE_TERTILE', 'EAA_GRIMAGE_TERTILE', 'EAA_DUNEDINMPOA_TERTILE', 'EAA_HORVATH_TERTILE', 'EAA_HANNUM_TERTILE', 'LEVINE_DNAMAGE', 'DNAMGRIMAGE', 'MPOA', 'HORVATH_DNAMAGE', 'HANNUM_DNAMAGE', 'CHRON_AGE']

    epigenetic_eeaa = df[eaa_vars].dropna(subset=eaa_vars[2:]).apply(pd.to_numeric, errors='coerce')

    # Merge
    events_df[['HHID', 'PN']] = events_df[['HHID', 'PN']].astype(float)
    epigenetic_eeaa[['HHID', 'PN']] = epigenetic_eeaa[['HHID', 'PN']].astype(float)
    all_merged = pd.merge(epigenetic_eeaa, events_df, on=["HHID", "PN"], how="left")


    output_path = EPIGENETIC_AGE_EVENTS_PRE_FILE
    all_merged.to_csv(output_path, index=False)
    print(f"Epigenetic age + events file saved to: {output_path}")


if __name__ == "__main__":
    main()
