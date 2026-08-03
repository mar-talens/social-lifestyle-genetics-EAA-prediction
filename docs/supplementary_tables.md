# Supplementary tables

`figures_tables/SuppTables_feature_importance.R` is the canonical generator
for the aggregate supplementary feature-importance tables. It reproduces the
published analytical values, rankings, normalization, joins, and rounding.

The generated workbooks are written to `output/tables/`:

| Output | Paper table | Main inputs |
|---|---|---|
| `Data_S2_LASSO_feature_importance.xlsx` | Data S2 | `output/analyses/LASSO/lasso_coefficients.csv`, `additional_file_1.xlsx` |
| `Data_S3_RF_regression_feature_importance.xlsx` | Data S3 | `output/analyses/RF/Regressor/permutation_feature_standard.csv`, `additional_file_1.xlsx` |
| `Data_S4_XGB_regression_feature_importance.xlsx` | Data S4 | XGBoost regression SHAP and permutation files under `output/analyses/XGB/Regressor`, `additional_file_1.xlsx` |
| `Data_S8_stratified_LASSO_feature_importance.xlsx` | Data S8 | `output/analyses/LASSO/stratified_coefficients.csv`, `additional_file_1.xlsx` |
| `Data_S9_stratified_XGB_SHAP_feature_importance.xlsx` | Data S9 | Six stratified XGBoost SHAP files under `output/analyses/XGB/Regressor`, `additional_file_1.xlsx` |
| `Data_S13_RF_classifier_feature_importance.xlsx` | Data S13 | `output/analyses/RF/Classifier/permutation_feature_standard.csv`, `additional_file_1.xlsx` |
| `Data_S14_XGB_classifier_feature_importance.xlsx` | Data S14 | XGBoost classifier SHAP and permutation files under `output/analyses/XGB/Classifier`, `additional_file_1.xlsx` |

The long-format SHAP files, including the six stratified SHAP files, are
participant-level generated inputs. They are private local files and must not
be committed to the public repository. The LASSO coefficient and permutation
files are generated analysis inputs and should likewise be handled according
to the repository's data-release policy.

The generated workbooks contain aggregate analytical tables and are suitable
for publication use. They reproduce the scientific values and calculations,
but not the exact formatting of the accepted supplementary workbook. In
particular, the accepted workbook was assembled separately and includes
publication titles, merged cells, multi-row headers, and other layout
cosmetics that this reproducible generator intentionally does not recreate.
