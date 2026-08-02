library(dplyr)
library(readr)
library(readxl)
library(writexl)
strat <- read.csv("/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_stratified_EAA_DUNEDINMPOA_GENDER.csv")
stratperm <- read.csv("/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_stratified_EAA_DUNEDINMPOA_GENDER.csv")

## 1) Paths
grim_path    <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_EAA_GRIMAGE.csv"
dune_path    <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_EAA_DUNEDINMPOA.csv"
perm_path    <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/permutation_feature_standard.csv"
name_map_path  <- "/Users/martalens/Desktop/ML_EAA/reports/paper_word/additional_file_1.xlsx"

## 2) Read SHAP & permutation data
GRIMAGE_XGBR_SHAP     <- read_csv(grim_path, show_col_types = FALSE)
DUNEDINMPOA_XGBR_SHAP <- read_csv(dune_path, show_col_types = FALSE)
XGBR_PERM             <- read_csv(perm_path, show_col_types = FALSE)

## 3) Read dictionary (XLSX)
# columns: "VARIABLE NAME", "DESCRITPION"
name_map_raw <- read_excel(name_map_path)

name_map <- name_map_raw %>%
    transmute(
        variable  = `VARIABLE NAME`,
        var_label = DESCRIPTION
    )

## 4) SHAP importance + % of total (within each clock)

# GrimAge
grim_shap_imp <- GRIMAGE_XGBR_SHAP %>%
    group_by(variable) %>%
    summarise(
        mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
        .groups = "drop"
    ) %>%
    left_join(name_map, by = "variable") %>%
    arrange(desc(mean_abs_shap)) %>%
    mutate(
        rank = row_number(),
        SHAPGrimage_share_percent = mean_abs_shap / sum(mean_abs_shap, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        SHAPGrimage_variable          = var_label,
        SHAPGrimage_mean_abs_SHAP     = mean_abs_shap,
        SHAPGrimage_share_percent
    )

# DunedinPoAm
dun_shap_imp <- DUNEDINMPOA_XGBR_SHAP %>%
    group_by(variable) %>%
    summarise(
        mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
        .groups = "drop"
    ) %>%
    left_join(name_map, by = "variable") %>%
    arrange(desc(mean_abs_shap)) %>%
    mutate(
        rank = row_number(),
        SHAPDunedinPoAm_share_percent = mean_abs_shap / sum(mean_abs_shap, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        SHAPDunedinPoAm_variable          = var_label,
        SHAPDunedinPoAm_mean_abs_SHAP     = mean_abs_shap,
        SHAPDunedinPoAm_share_percent
    )

## 5) Permutation importance + % of total (within each clock)

perm_imp <- XGBR_PERM %>%
    group_by(clock, feature) %>%
    summarise(
        mean_perm = mean(mean_drop, na.rm = TRUE),
        .groups   = "drop"
    ) %>%
    rename(variable = feature) %>%
    left_join(name_map, by = "variable")

# GrimAge
grim_perm_imp <- perm_imp %>%
    filter(clock == "EAA_GRIMAGE") %>%
    arrange(desc(mean_perm)) %>%
    mutate(
        rank = row_number(),
        PermGrimage_share_percent = mean_perm / sum(mean_perm, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        PermGrimage_variable      = var_label,
        PermGrimage_mean_perm     = mean_perm,
        PermGrimage_share_percent
    )

# DunedinPoAm
dun_perm_imp <- perm_imp %>%
    filter(clock == "EAA_DUNEDINMPOA") %>%
    arrange(desc(mean_perm)) %>%
    mutate(
        rank = row_number(),
        PermDunedinPoAm_share_percent = mean_perm / sum(mean_perm, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        PermDunedinPoAm_variable      = var_label,
        PermDunedinPoAm_mean_perm     = mean_perm,
        PermDunedinPoAm_share_percent
    )

## 6) Combine everything by rank (all variables, ordered)

summary_tbl <- grim_shap_imp %>%
    full_join(dun_shap_imp,  by = "rank") %>%
    full_join(grim_perm_imp, by = "rank") %>%
    full_join(dun_perm_imp,  by = "rank") %>%
    arrange(rank)

## (Optional) Round percentages & importances for nicer display
summary_tbl_print <- summary_tbl %>%
    mutate(
        across(ends_with("_percent"), ~round(.x, 2)),
        across(contains("mean_abs_SHAP"), ~round(.x, 4)),
        across(contains("mean_perm"), ~round(.x, 4))
    )

## 7) Save to CSV
out_path_xlsx <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/tables/feature_importance_all_xgbregr.xlsx"

write_xlsx(summary_tbl_print, out_path_xlsx)








### XGB Binary
library(dplyr)
library(readr)
library(readxl)
library(writexl)

## 1) Paths
grim_path    <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Classifier/shap_long_EAA_GRIMAGE_BINARY.csv"
dune_path    <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Classifier/shap_long_EAA_DUNEDINMPOA_BINARY.csv"
perm_path    <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Classifier/permutation_feature_standard.csv"
name_map_path  <- "/Users/martalens/Desktop/ML_EAA/reports/paper_word/additional_file_1.xlsx"

## 2) Read SHAP & permutation data
GRIMAGE_XGBR_SHAP     <- read_csv(grim_path, show_col_types = FALSE)
DUNEDINMPOA_XGBR_SHAP <- read_csv(dune_path, show_col_types = FALSE)
XGBR_PERM             <- read_csv(perm_path, show_col_types = FALSE)

## 3) Read dictionary (XLSX)
# columns: "VARIABLE NAME", "DESCRITPION"
name_map_raw <- read_excel(name_map_path)

name_map <- name_map_raw %>%
    transmute(
        variable  = `VARIABLE NAME`,
        var_label = DESCRIPTION
    )

## 4) SHAP importance + % of total (within each clock)

# GrimAge
grim_shap_imp <- GRIMAGE_XGBR_SHAP %>%
    group_by(variable) %>%
    summarise(
        mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
        .groups = "drop"
    ) %>%
    left_join(name_map, by = "variable") %>%
    arrange(desc(mean_abs_shap)) %>%
    mutate(
        rank = row_number(),
        SHAPGrimage_share_percent = mean_abs_shap / sum(mean_abs_shap, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        SHAPGrimage_variable          = var_label,
        SHAPGrimage_mean_abs_SHAP     = mean_abs_shap,
        SHAPGrimage_share_percent
    )

# DunedinPoAm
dun_shap_imp <- DUNEDINMPOA_XGBR_SHAP %>%
    group_by(variable) %>%
    summarise(
        mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
        .groups = "drop"
    ) %>%
    left_join(name_map, by = "variable") %>%
    arrange(desc(mean_abs_shap)) %>%
    mutate(
        rank = row_number(),
        SHAPDunedinPoAm_share_percent = mean_abs_shap / sum(mean_abs_shap, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        SHAPDunedinPoAm_variable          = var_label,
        SHAPDunedinPoAm_mean_abs_SHAP     = mean_abs_shap,
        SHAPDunedinPoAm_share_percent
    )

## 5) Permutation importance + % of total (within each clock)

perm_imp <- XGBR_PERM %>%
    group_by(clock, feature) %>%
    summarise(
        mean_perm = mean(mean_drop, na.rm = TRUE),
        .groups   = "drop"
    ) %>%
    rename(variable = feature) %>%
    left_join(name_map, by = "variable")

# GrimAge
grim_perm_imp <- perm_imp %>%
    filter(clock == "EAA_GRIMAGE_BINARY") %>%
    arrange(desc(mean_perm)) %>%
    mutate(
        rank = row_number(),
        PermGrimage_share_percent = mean_perm / sum(mean_perm, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        PermGrimage_variable      = var_label,
        PermGrimage_mean_perm     = mean_perm,
        PermGrimage_share_percent
    )

# DunedinPoAm
dun_perm_imp <- perm_imp %>%
    filter(clock == "EAA_DUNEDINMPOA_BINARY") %>%
    arrange(desc(mean_perm)) %>%
    mutate(
        rank = row_number(),
        PermDunedinPoAm_share_percent = mean_perm / sum(mean_perm, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        PermDunedinPoAm_variable      = var_label,
        PermDunedinPoAm_mean_perm     = mean_perm,
        PermDunedinPoAm_share_percent
    )

## 6) Combine everything by rank (all variables, ordered)

summary_tbl <- grim_shap_imp %>%
    full_join(dun_shap_imp,  by = "rank") %>%
    full_join(grim_perm_imp, by = "rank") %>%
    full_join(dun_perm_imp,  by = "rank") %>%
    arrange(rank)

## (Optional) Round percentages & importances for nicer display
summary_tbl_print <- summary_tbl %>%
    mutate(
        across(ends_with("_percent"), ~round(.x, 2)),
        across(contains("mean_abs_SHAP"), ~round(.x, 4)),
        across(contains("mean_perm"), ~round(.x, 4))
    )

## 7) Save to CSV
out_path_xlsx <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/tables/feature_importance_all_xgbclass.xlsx"

write_xlsx(summary_tbl_print, out_path_xlsx)








### RF Binary
library(dplyr)
library(readr)
library(readxl)
library(writexl)

## 1) Paths
perm_path    <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/RF/Classifier/permutation_feature_standard.csv"
name_map_path  <- "/Users/martalens/Desktop/ML_EAA/data/metadata/variables_dictionary.csv"

## 2) Read SHAP & permutation data
XGBR_PERM             <- read_csv(perm_path, show_col_types = FALSE)

## 3) Read dictionary (XLSX)
# columns: "VARIABLE NAME", "DESCRITPION"
name_map_raw <- read_excel(name_map_path)

name_map <- name_map_raw %>%
    transmute(
        variable  = `VARIABLE NAME`,
        var_label = DESCRIPTION
    )

## 5) Permutation importance + % of total (within each clock)

perm_imp <- XGBR_PERM %>%
    group_by(clock, feature) %>%
    summarise(
        mean_perm = mean(mean_drop, na.rm = TRUE),
        .groups   = "drop"
    ) %>%
    rename(variable = feature) %>%
    left_join(name_map, by = "variable")

# GrimAge
grim_perm_imp <- perm_imp %>%
    filter(clock == "EAA_GRIMAGE_BINARY") %>%
    arrange(desc(mean_perm)) %>%
    mutate(
        rank = row_number(),
        PermGrimage_share_percent = mean_perm / sum(mean_perm, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        PermGrimage_variable      = var_label,
        PermGrimage_mean_perm     = mean_perm,
        PermGrimage_share_percent
    )

# DunedinPoAm
dun_perm_imp <- perm_imp %>%
    filter(clock == "EAA_DUNEDINMPOA_BINARY") %>%
    arrange(desc(mean_perm)) %>%
    mutate(
        rank = row_number(),
        PermDunedinPoAm_share_percent = mean_perm / sum(mean_perm, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        PermDunedinPoAm_variable      = var_label,
        PermDunedinPoAm_mean_perm     = mean_perm,
        PermDunedinPoAm_share_percent
    )

## 6) Combine everything by rank (all variables, ordered)

summary_tbl <- grim_perm_imp %>%
    full_join(dun_perm_imp,  by = "rank") %>%
    arrange(rank)

## (Optional) Round percentages & importances for nicer display
summary_tbl_print <- summary_tbl %>%
    mutate(
        across(ends_with("_percent"), ~round(.x, 2)),
        #across(contains("mean_abs_SHAP"), ~round(.x, 4)),
        across(contains("mean_perm"), ~round(.x, 4))
    )

## 7) Save to CSV
out_path_xlsx <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/tables/feature_importance_all_rfclass.xlsx"

write_xlsx(summary_tbl_print, out_path_xlsx)









### RF Regressor
library(dplyr)
library(readr)
library(readxl)
library(writexl)

## 1) Paths
perm_path    <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/RF/Regressor/permutation_feature_standard.csv"
name_map_path  <- "/Users/martalens/Desktop/ML_EAA/reports/paper_word/additional_file_1.xlsx"

## 2) Read SHAP & permutation data
XGBR_PERM             <- read_csv(perm_path, show_col_types = FALSE)

## 3) Read dictionary (XLSX)
# columns: "VARIABLE NAME", "DESCRITPION"
name_map_raw <- read_excel(name_map_path)

name_map <- name_map_raw %>%
    transmute(
        variable  = `VARIABLE NAME`,
        var_label = DESCRIPTION
    )

## 5) Permutation importance + % of total (within each clock)

perm_imp <- XGBR_PERM %>%
    group_by(clock, feature) %>%
    summarise(
        mean_perm = mean(mean_drop, na.rm = TRUE),
        .groups   = "drop"
    ) %>%
    rename(variable = feature) %>%
    left_join(name_map, by = "variable")

# GrimAge
grim_perm_imp <- perm_imp %>%
    filter(clock == "EAA_GRIMAGE") %>%
    arrange(desc(mean_perm)) %>%
    mutate(
        rank = row_number(),
        PermGrimage_share_percent = mean_perm / sum(mean_perm, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        PermGrimage_variable      = var_label,
        PermGrimage_mean_perm     = mean_perm,
        PermGrimage_share_percent
    )

# DunedinPoAm
dun_perm_imp <- perm_imp %>%
    filter(clock == "EAA_DUNEDINMPOA") %>%
    arrange(desc(mean_perm)) %>%
    mutate(
        rank = row_number(),
        PermDunedinPoAm_share_percent = mean_perm / sum(mean_perm, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        PermDunedinPoAm_variable      = var_label,
        PermDunedinPoAm_mean_perm     = mean_perm,
        PermDunedinPoAm_share_percent
    )

## 6) Combine everything by rank (all variables, ordered)

summary_tbl <- grim_perm_imp %>%
    full_join(dun_perm_imp,  by = "rank") %>%
    arrange(rank)

## (Optional) Round percentages & importances for nicer display
summary_tbl_print <- summary_tbl %>%
    mutate(
        across(ends_with("_percent"), ~round(.x, 2)),
        #across(contains("mean_abs_SHAP"), ~round(.x, 4)),
        across(contains("mean_perm"), ~round(.x, 4))
    )

## 7) Save to CSV
out_path_xlsx <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/tables/feature_importance_all_rfregr.xlsx"

write_xlsx(summary_tbl_print, out_path_xlsx)




#### STRATIFIED
library(dplyr)
library(readr)
library(readxl)
library(writexl)

make_stratified_shap_table <- function(shap_path, dict_path, out_path_xlsx) {
    # 1) Read SHAP long (stratified) file
    shap_df <- read_csv(shap_path, show_col_types = FALSE)
    
    # 2) Read dictionary
    # columns: "VARIABLE NAME", "DESCRIPTION"
    name_map_raw <- read_excel(dict_path)
    
    name_map <- name_map_raw %>%
        transmute(
            variable  = `VARIABLE NAME`,
            var_label = DESCRIPTION
        )
    
    # 3) Compute mean |SHAP| and percentage within each clock × stratum
    summary_tbl <- shap_df %>%
        group_by(clock, stratum, variable) %>%
        summarise(
            mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
            .groups = "drop"
        ) %>%
        left_join(name_map, by = "variable") %>%
        group_by(clock, stratum) %>%
        arrange(desc(mean_abs_shap), .by_group = TRUE) %>%
        mutate(
            rank = row_number(),
            share_percent = mean_abs_shap / sum(mean_abs_shap, na.rm = TRUE) * 100
        ) %>%
        ungroup() %>%
        select(
            clock,
            stratify_col = stratum,   # rename for clarity if you like
            rank,
            variable_label = var_label,
            mean_abs_shap,
            share_percent
        ) %>%
        mutate(
            mean_abs_shap = round(mean_abs_shap, 4),
            share_percent = round(share_percent, 2)
        )
    
    # 4) Save to xlsx
    write_xlsx(summary_tbl, out_path_xlsx)
}

dict_path <- "/Users/martalens/Desktop/ML_EAA/reports/paper_word/additional_file_1.xlsx"

make_stratified_shap_table(
    shap_path     = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_stratified_EAA_DUNEDINMPOA_GENDER.csv",
    dict_path     = dict_path,
    out_path_xlsx = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/tables/stratified_SHAP_EAA_DUNEDINMPOA_GENDER.xlsx"
)

make_stratified_shap_table(
    shap_path     = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_stratified_EAA_GRIMAGE_GENDER.csv",
    dict_path     = dict_path,
    out_path_xlsx = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/tables/stratified_SHAP_EAA_GRIMAGE_GENDER.xlsx"
)


make_stratified_shap_table(
    shap_path     = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_stratified_EAA_DUNEDINMPOA_WHITE_BINARY.csv",
    dict_path     = dict_path,
    out_path_xlsx = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/tables/stratified_SHAP_EAA_DUNEDINMPOA_WHITE_BINARY.xlsx"
)

make_stratified_shap_table(
    shap_path     = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_stratified_EAA_GRIMAGE_WHITE_BINARY.csv",
    dict_path     = dict_path,
    out_path_xlsx = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/tables/stratified_SHAP_EAA_GRIMAGE_WHITE_BINARY.xlsx"
)



make_stratified_shap_table(
    shap_path     = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_stratified_EAA_GRIMAGE_EVER_SMOKED_RAND.csv",
    dict_path     = dict_path,
    out_path_xlsx = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/tables/stratified_SHAP_EAA_GRIMAGE_EVER_SMOKED_RAND.xlsx"
)


make_stratified_shap_table(
    shap_path     = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_stratified_EAA_DUNEDINMPOA_EVER_SMOKED_RAND.csv",
    dict_path     = dict_path,
    out_path_xlsx = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/tables/stratified_SHAP_EAA_DUNEDINMPOAE_EVER_SMOKED_RAND.xlsx"
)



























### LASSO Regressor
library(dplyr)
library(readr)
library(readxl)
library(writexl)

## 1) Paths
lasso_path    <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/LASSO/lasso_coefficients.csv"
name_map_path <- "/Users/martalens/Desktop/ML_EAA/reports/paper_word/npj_aging/additional_file_1.xlsx"

## 2) Read LASSO coefficients
LASSO_COEF <- read_csv(lasso_path, show_col_types = FALSE)

## 3) Read dictionary
name_map_raw <- read_xlsx(name_map_path)

name_map <- name_map_raw %>%
    transmute(
        variable  = `VARIABLE NAME`,
        var_label = DESCRIPTION
    )

## 4) Compute LASSO importance
lasso_imp <- LASSO_COEF %>%
    mutate(
        importance = abs(mean_coef) * selection_freq
    ) %>%
    left_join(name_map, by = "variable") %>%
    mutate(var_label = coalesce(var_label, variable))

## 5) Split per clock (same structure as others)

# GrimAge
grim_lasso_imp <- lasso_imp %>%
    filter(clock == "EAA_GRIMAGE") %>%
    arrange(desc(importance)) %>%
    mutate(
        rank = row_number(),
        LASSOGrimage_share_percent = importance / sum(importance, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        LASSOGrimage_variable        = var_label,
        LASSOGrimage_mean_coef       = mean_coef,
        LASSOGrimage_sd_coef         = sd_coef,
        LASSOGrimage_selection_freq  = selection_freq,
        LASSOGrimage_importance      = importance,
        LASSOGrimage_share_percent
    )

# DunedinPoAm
dun_lasso_imp <- lasso_imp %>%
    filter(clock == "EAA_DUNEDINMPOA") %>%
    arrange(desc(importance)) %>%
    mutate(
        rank = row_number(),
        LASSODunedinPoAm_share_percent = importance / sum(importance, na.rm = TRUE) * 100
    ) %>%
    select(
        rank,
        LASSODunedinPoAm_variable        = var_label,
        LASSODunedinPoAm_mean_coef       = mean_coef,
        LASSODunedinPoAm_sd_coef         = sd_coef,
        LASSODunedinPoAm_selection_freq  = selection_freq,
        LASSODunedinPoAm_importance      = importance,
        LASSODunedinPoAm_share_percent
    )

## 6) Combine (same logic as before)

summary_tbl <- grim_lasso_imp %>%
    full_join(dun_lasso_imp, by = "rank") %>%
    arrange(rank)

## 7) Round for display
summary_tbl_print <- summary_tbl %>%
    mutate(
        across(ends_with("_percent"), ~round(.x, 2)),
        across(contains("importance"), ~round(.x, 4))
    )

## 8) Save
out_path_xlsx <- "/Users/martalens/Desktop/ML_EAA/output/npj_aging_output/supp_tables/feature_importance_all_lasso.xlsx"

write_xlsx(summary_tbl_print, out_path_xlsx)






## LASSO Stratified
### LASSO STRATIFIED
library(dplyr)
library(readr)
library(readxl)
library(writexl)

make_stratified_lasso_table <- function(lasso_path, dict_path, out_path_xlsx) {
    
    # 1) Read LASSO stratified coefficients
    lasso_df <- read_csv(lasso_path, show_col_types = FALSE)
    
    # 2) Read dictionary
    name_map_raw <- read_excel(dict_path)
    
    name_map <- name_map_raw %>%
        transmute(
            variable  = `VARIABLE NAME`,
            var_label = DESCRIPTION
        )
    
    # 3) Compute importance (NO summarising!)
    summary_tbl <- lasso_df %>%
        mutate(
            importance = abs(mean_coef) * selection_freq
        ) %>%
        left_join(name_map, by = "variable") %>%
        mutate(var_label = coalesce(var_label, variable)) %>%
        
        # rank WITHIN each clock × stratum
        group_by(clock, stratum) %>%
        arrange(desc(importance), .by_group = TRUE) %>%
        mutate(
            rank = row_number(),
            share_percent = importance / sum(importance, na.rm = TRUE) * 100
        ) %>%
        ungroup() %>%
        
        # select final structure
        select(
            clock,
            stratify_col = stratum,
            rank,
            variable_label = var_label,
            mean_coef,
            sd_coef,
            selection_freq,
            importance,
            share_percent
        ) %>%
        
        # rounding (publication-ready)
        mutate(
            mean_coef      = round(mean_coef, 4),
            sd_coef        = round(sd_coef, 4),
            selection_freq = round(selection_freq, 3),
            importance     = round(importance, 4),
            share_percent  = round(share_percent, 2)
        )
    
    # 4) Save
    write_xlsx(summary_tbl, out_path_xlsx)
}


lasso_strat_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/LASSO/stratified_coefficients.csv"

dict_path <- "/Users/martalens/Desktop/ML_EAA/reports/paper_word/npj_aging/additional_file_1.xlsx"

make_stratified_lasso_table(
    lasso_path     = lasso_strat_path,
    dict_path      = dict_path,
    out_path_xlsx  = "/Users/martalens/Desktop/ML_EAA/output/npj_aging_output/supp_tables/stratified_LASSO.xlsx"
)



