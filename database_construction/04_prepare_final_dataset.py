"""Prepare the final epigenetic-age events dataset.

This script preserves the final dataset-preparation workflow in cell 3 of
``database_events_epig_oct.ipynb``. Transformations and their order are kept
unchanged; only path handling and standalone execution are added.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import EPIGENETIC_AGE_EVENTS_PRE_FILE, EPIGENETIC_AGE_EVENTS_FILE


def main():
    #### OHE and creating missing values flags ########
    df = pd.read_csv(EPIGENETIC_AGE_EVENTS_PRE_FILE)

    #####################################
    ### ETHNICITY FLAGS
        # Conditions for non-hispanic WHITE
    conditions_WHITE = (
        ((df['RACE'] == 1) &
        (df['HISPANIC'] == 5)) |
        (df['ANCESTRY_FLAG'] == 2)
    )
        # Conditions for Hispanic
    conditions_HISPANIC = (
        (df['HISPANIC'].isin([1, 2, 3])) |
        (df['ANCESTRY_FLAG'] == 3)
    )
        # Conditions for African American
    conditions_AFRICAN_AMERICAN = (
        (df['RACE'] == 2) |
        (df['ANCESTRY_FLAG'] == 1))
        # Conditions Other/Not obtained
    conditions_OTHER = (
        (df['RACE'].isin([0, 7])) &
        (df['ANCESTRY_FLAG'].isna()))

    df['WHITE_BINARY'] = np.where(conditions_WHITE, 1, 0)
    df['HISPANIC_BINARY'] = np.where(conditions_HISPANIC, 1, 0)
    df['AFRICAN_AMERICAN_BINARY'] = np.where(conditions_AFRICAN_AMERICAN, 1, 0)
    df['NO_ETHNICITY_BINARY'] = np.where(conditions_OTHER, 1, 0)


    ######################################
    ### MARITAL STATUS OHE
    df = pd.get_dummies(df, columns=['MARITAL_STATUS_RAND'], prefix='MARITAL_STATUS', dtype='int8', dummy_na=True)
    df['MARITAL_STATUS_MARRIED'] = df['MARITAL_STATUS_1.0']
    df['MARITAL_STATUS_PARTNERED'] = df['MARITAL_STATUS_2.0']
    df['MARITAL_STATUS_DIVORCED'] = df['MARITAL_STATUS_3.0']
    df['MARITAL_STATUS_WIDOWED'] = df['MARITAL_STATUS_4.0']
    df['MARITAL_STATUS_NEVER_MARRIED'] = df['MARITAL_STATUS_5.0']

    df = df.drop(columns=['MARITAL_STATUS_1.0', 'MARITAL_STATUS_nan', 'MARITAL_STATUS_2.0', 'MARITAL_STATUS_3.0', 'MARITAL_STATUS_4.0', 'MARITAL_STATUS_5.0']) #dropping also the NAN because there were no missing values


    #########################################
    ### JOB AND EDUCATION OHE
    df = pd.get_dummies(df, columns=["LONGEST_INDUSTRY","LONGEST_OCCUPATION"], prefix=['IND','OCC'], dtype="int8")
    df = pd.get_dummies(df, columns=["FJOB_CAT"], prefix="FJOB", dtype="int8", dummy_na=True)

    #######################################
    ### LHMS
    df["NO_LHMS_SUPP"] = df["LHMSWIND"].isin([3, 5, 6, 7, 8]).astype(int)
    df = df.drop(columns = ["LHMSWIND"])

    ## house type one-hot encoding
    df = pd.get_dummies(df, columns=['LH16'], prefix='HOUSE_FIRSTJOB', dtype='int8', dummy_na=True)
    df = pd.get_dummies(df, columns=['LH20'], prefix='HOUSE_AT40', dtype='int8', dummy_na=True)
    df = pd.get_dummies(df, columns=['LH9'], prefix='HOUSE_AT10', dtype='int8', dummy_na=True)



    #############################
    #### OTHERS
    ### changes of the 13/10/2025
    df[['DRUG', 'LH33', 'USBORN']] = df[['DRUG', 'LH33', 'USBORN']].replace(5, 0)
    df['GENDER'] = df['GENDER'].replace(2, 0)
    df['FAMFIN'] = df['FAMFIN'].replace({3:2, 6:2, 5:3})

    # Create NOTAPPLY flags and set "6" (does not apply) to NA.
    # Using .replace with a list key raises "unhashable type: 'list'", so use isin / np.where / mask instead.
    df['RELWMO_NOTAPPLY'] = np.where(df['RELWMO'] == 6, 1,
                                     np.where(df['RELWMO'].isin([1, 2, 3, 4, 5]), 0, pd.NA))
    df['RELWMO'] = df['RELWMO'].mask(df['RELWMO'] == 6, pd.NA)

    df['RELWFA_NOTAPPLY'] = np.where(df['RELWFA'] == 6, 1,
                                     np.where(df['RELWFA'].isin([1, 2, 3, 4, 5]), 0, pd.NA))
    df['RELWFA'] = df['RELWFA'].mask(df['RELWFA'] == 6, pd.NA)
    df['BINGE_LIFECOURSE'] = df['BINGE_LIFECOURSE'].isin([1,2]).replace({True:1, False:0, pd.NA:pd.NA})

    df = pd.get_dummies(df, columns=['BIRTH_PLACE_RAND'], prefix='BIRTH_PLACE', dtype='int8', dummy_na=True)


    df['DAD_ALIVE_RAND'] = pd.to_numeric(df['DAD_ALIVE_RAND'], errors='coerce')
    df['MOTHER_ALIVE_RAND'] = pd.to_numeric(df['MOTHER_ALIVE_RAND'], errors='coerce')

    df[['PHYS_ACTIVITY_VIGOROUS', 'PHYS_ACTIVITY_MODERATE', 'PHYS_ACTIVITY_MILD']] = df[['PHYS_ACTIVITY_VIGOROUS', 'PHYS_ACTIVITY_MODERATE', 'PHYS_ACTIVITY_MILD']].replace({1:4, 2:3, 3:2, 4:1})


    df = pd.get_dummies(df, columns=['RELIGION_RAND'], prefix='RELIGION', dtype='int8', dummy_na=False)


    ## SMOKING STATUS: 
    df['FORMER_SMOKER'] = np.where((df['CURRENTLY_SMOKING_RAND'] == 0) & (df['EVER_SMOKED_RAND'] == 1), 1, 0)


    # Save
    output_path = EPIGENETIC_AGE_EVENTS_FILE
    df.to_csv(output_path, index=False)
    print(f"Epigenetic age + events file saved to: {output_path}")

if __name__ == "__main__":
    main()
