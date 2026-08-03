"""Generate the aggregate descriptive-characteristics table."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("output/epigenetic_age_events.csv")
OUTPUT_FILE = Path("output/tables/Table1_descriptive_characteristics.csv")

CAT_VARS = [
    "GENDER",
    "WHITE_BINARY",
    "HISPANIC_BINARY",
    "AFRICAN_AMERICAN_BINARY",
    "MARITAL_STATUS_MARRIED",
    "MARITAL_STATUS_WIDOWED",
    "MARITAL_STATUS_DIVORCED",
    "RAEDUC_CAT",
    "HH_INCOME_TERTILE",
    "BMI_UNDERWEIGHT",
    "BMI_NORMAL_WEIGHT",
    "BMI_OVERWEIGHT",
    "BMI_OBESITY",
    "SMOKING_3CAT",
    "ALCOHOL_SAFE",
    "PHYS_ACTIVITY_SEDENTARY",
]

EAA_COLS = [
    "EAA_DUNEDINMPOA",
    "EAA_GRIMAGE",
    "EAA_LEVINE",
    "EAA_HANNUM",
    "EAA_HORVATH",
]


def construct_table1_variables(df):
    """Reproduce the derived variables from notebook cell 3 exactly."""
    df["BMI_UNDERWEIGHT"] = (df["BMI"] < 18.5).astype(int)
    df["BMI_NORMAL_WEIGHT"] = (
        (df["BMI"] >= 18.5) & (df["BMI"] <= 24.9)
    ).astype(int)
    df["BMI_OVERWEIGHT"] = (
        (df["BMI"] >= 25.0) & (df["BMI"] <= 29.9)
    ).astype(int)
    df["BMI_OBESITY"] = (df["BMI"] >= 30.0).astype(int)

    df["PHYS_ACTIVITY_SEDENTARY"] = (
        (df["PHYS_ACTIVITY_MILD"].isin([1, 2, 3]))
        & (df["PHYS_ACTIVITY_MODERATE"] == 1)
        & (df["PHYS_ACTIVITY_VIGOROUS"] == 1)
    ).astype(int)

    df["ALCOHOL_SAFE"] = np.where(
        ((df["GENDER"] == 0) & (df["DRINKS_PER_WEEK"] <= 7))
        | ((df["GENDER"] == 1) & (df["DRINKS_PER_WEEK"] <= 14)),
        1,
        0,
    )

    df["HH_INCOME_TERTILE"] = pd.qcut(
        df["HH_MEAN_INCOME"],
        q=3,
        labels=[1, 2, 3],
    )
    df["RAEDUC_CAT"] = df["RAEDUC"].replace({2: 1, 3: 2, 4: 3, 5: 3})

    df["SMOKING_3CAT"] = np.nan

    df.loc[
        (df["EVER_SMOKED_RAND"] == 1)
        & (df["CURRENTLY_SMOKING_RAND"] == 1),
        "SMOKING_3CAT",
    ] = 1

    df.loc[
        (df["EVER_SMOKED_RAND"] == 1)
        & (df["CURRENTLY_SMOKING_RAND"] == 0),
        "SMOKING_3CAT",
    ] = 2

    df.loc[
        df["EVER_SMOKED_RAND"] == 0,
        "SMOKING_3CAT",
    ] = 3

    df["SMOKING_3CAT"] = df["SMOKING_3CAT"].astype("Int64")
    return df


def summarize_categorical(var, df):
    """Reproduce cell 4's categorical summary and column ordering."""
    total_n = len(df)

    agg_dict = {"N": (var, "size")}
    for eaa in EAA_COLS:
        agg_dict[f"{eaa}_MEAN"] = (eaa, "mean")
        agg_dict[f"{eaa}_SD"] = (eaa, "std")

    tab = df.groupby(var, dropna=False).agg(**agg_dict)
    tab["%"] = 100 * tab["N"] / total_n

    mean_cols = [column for column in tab.columns if column.endswith("_MEAN")]
    sd_cols = [column for column in tab.columns if column.endswith("_SD")]
    return tab[["N", "%"] + mean_cols + sd_cols]


def make_age_row(df, columns):
    """Create the requested overall chronological-age descriptive row."""
    row = {column: np.nan for column in columns}
    row.update(
        {
            "Variable": "CHRON_AGE",
            "Category": np.nan,
            "N": np.nan,
            "%": np.nan,
            "CHRON_AGE_MEAN": df["CHRON_AGE"].mean(),
            "CHRON_AGE_SD": df["CHRON_AGE"].std(),
        }
    )
    return pd.DataFrame([row], columns=columns)


def main():
    df = pd.read_csv(INPUT_FILE)
    df = construct_table1_variables(df)

    tables = []
    for variable in CAT_VARS:
        table = summarize_categorical(variable, df)
        table.index = pd.MultiIndex.from_product(
            [[variable], table.index],
            names=["Variable", "Category"],
        )
        tables.append(table)

    table1_cats = pd.concat(tables)
    table1 = pd.concat([table1_cats])
    output = table1.round(2).reset_index()

    output.insert(4, "CHRON_AGE_MEAN", np.nan)
    output.insert(5, "CHRON_AGE_SD", np.nan)
    age_row = make_age_row(df, output.columns)
    output = pd.concat([age_row, output], ignore_index=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
