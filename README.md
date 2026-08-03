# Integrating social, lifestyle, and genetic profiles to predict epigenetic age acceleration in older adults

## Overview

This repository contains the code used to construct the analysis database,
run the machine-learning analyses, and generate the figures and supplementary
tables accompanying the manuscript.

The restricted Health and Retirement Study (HRS) datasets are not distributed
with this repository. Reproduction requires authorized access to the relevant
HRS files.

## Repository structure

- `database_construction/` — the four ordered stages that extract HRS
  variables, construct predictors, construct epigenetic outcomes, and prepare
  the final analysis dataset.
- `main_analyses/` — the canonical Random Forest/XGBoost and LASSO analysis
  scripts.
- `figures_tables/` — standalone scripts for the main figures, supplementary
  figures, descriptive Table 1, and supplementary feature-importance tables.
- `paper/` — manuscript and supplementary materials used to document the
  published workflow.
- `output/` — generated intermediate datasets, model outputs, figures, and
  tables. Participant-level and restricted outputs are intended to remain
  local and ignored by version control.
- `docs/` — supplementary documentation, including the supplementary-table
  data and provenance notes.

## Data access

Users must independently obtain authorized HRS datasets through the HRS
website and comply with all HRS data-use requirements. The repository does not
contain restricted HRS source files or participant-level analysis data.

Participant-level datasets, predictions, SHAP records, and intermediate
participant-level outputs are intentionally excluded from public commits.

## Installation

The repository uses both Python and R workflows.

Python environments should provide the dependencies required by the database,
machine-learning, and Python figure/table scripts. R environments should
provide the packages required by the R figure and table scripts.

Dependency placeholders:

- `requirements.txt`
- `environment.yml`

No package versions are prescribed in this README.

## Configuration

Before running any workflow, edit the repository-root `config.py` and set
`HRS_DATA_DIR` to the local directory containing the authorized HRS datasets.

The configuration also defines repository-relative input and output locations.
Do not commit restricted source data or participant-level generated files.

## Database construction workflow

Run the database-construction stages in this order:

```text
01_extract_hrs_variables.py
        ↓
02_construct_predictors.py
        ↓
03_construct_epigenetic_outcomes.py
        ↓
04_prepare_final_dataset.py
```

The stages produce the following files in sequence:

1. `combined_events.csv` — extracted and merged HRS event variables.
2. `events_df.csv` — constructed predictor dataset.
3. `epigenetic_age.csv` — constructed epigenetic-age outcomes.
4. `epigenetic_age_events_pre.csv` — outcome data merged with predictors before final preparation.
5. `epigenetic_age_events.csv` — final analysis dataset.

The required HRS source files are private and must be available locally before
the corresponding stage is run.

## Machine-learning analyses

The canonical analysis scripts are:

- `main_analyses/RF_XGB.py`
- `main_analyses/LASSO.py`

Both workflows use the same five-fold outer cross-validation splits based on
the final dataset row order, with shuffling and random seed 42.

The Random Forest/XGBoost workflow additionally produces or uses:

- SHAP importance;
- permutation importance;
- domain-only analyses;
- ablation analyses;
- learning curves;
- stratified analyses.

Generated model outputs are written under `output/analyses/` and may include
private participant-level files that must remain local.

## Figures

Run all figure and table scripts from the repository root because they use repository-relative paths.

### Main figures

- `figures_tables/Fig1_modelperformance.R` — model-performance outputs from
  the RF, XGBoost, and LASSO analyses.
- `figures_tables/Fig2_featureimportance.R` — aggregate XGBoost SHAP,
  Random Forest permutation, and LASSO importance outputs plus metadata.
- `figures_tables/Fig3_beeswarm_plot.R` — local XGBoost SHAP files, the final
  analysis dataset, and variable metadata.
- `figures_tables/Fig4_panel_domain_abl_perm.R` — aggregate XGBoost baseline,
  domain-only, ablation, and KNN permutation performance outputs.

### Supplementary figures

- `figures_tables/SuppFig1_clock_correlations.py` — the final analysis dataset.
- `figures_tables/SuppFig2_learning_curves.R` — learning-curve and model-
  performance outputs for LASSO, Random Forest, and XGBoost.
- `figures_tables/SuppFig3_SHAP_dependence.R` — local SHAP files, the final
  analysis dataset, and `variables_dictionary.csv`.
- `figures_tables/SuppFig4_6_stratified_feature_importance.R` — stratified
  XGBoost SHAP files for gender, ethnicity, and smoking.
- `figures_tables/SuppFig7_9_stratified_domains.R` — aggregate stratified
  XGBoost domain-only, ablation, and KNN permutation outputs.

The figure scripts write generated images under `output/figures/`. Private
SHAP and participant-level inputs are expected to exist locally when required.

## Tables

- `figures_tables/Table1_descriptive_characteristics.py` — aggregate
  descriptive characteristics from the final analysis dataset.
- `figures_tables/SuppTables_feature_importance.R` — aggregate feature-
  importance tables corresponding to Data S2, S3, S4, S8, S9, S13, and S14.

These scripts reproduce the analytical values, rankings, normalization, joins,
and rounding used for the tables. The accepted supplementary workbook was
assembled separately for publication; merged headers, publication titles,
multi-row layouts, and other workbook formatting are intentionally not
reproduced.

Generated aggregate tables are written under `output/tables/`.

## Data availability

Public repository contents include:

- source scripts;
- metadata files;
- aggregate publication tables and final figures when provided.

Private or restricted contents include:

- HRS source datasets;
- participant-level SHAP files;
- participant-level predictions;
- participant-level intermediate datasets.

## Citation

Paper citation: *to be added*.

## License

License: *to be added*.
