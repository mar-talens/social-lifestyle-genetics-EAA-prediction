# Publication supplementary feature-importance tables.
#
# This standalone script consolidates the published sections of
# feature_importance_tables.R. The long SHAP inputs are generated analysis
# outputs and are expected to remain local/private.
# The script reproduces the published aggregate calculations. The accepted
# supplementary workbook was assembled separately for publication; exact
# workbook formatting (merged headers, publication layout, and related
# cosmetics) is intentionally not reproduced. Analytical values, rankings,
# normalization, and rounding are preserved.

library(dplyr)
library(readr)
library(readxl)
library(writexl)

OUTPUT_DIR <- "output/tables"
METADATA_FILE <- "additional_file_1.xlsx"

XGB_REG_DIR <- "output/analyses/XGB/Regressor"
XGB_CLASS_DIR <- "output/analyses/XGB/Classifier"
RF_REG_DIR <- "output/analyses/RF/Regressor"
RF_CLASS_DIR <- "output/analyses/RF/Classifier"
LASSO_DIR <- "output/analyses/LASSO"

read_name_map <- function() {
  read_excel(METADATA_FILE) |>
    transmute(
      variable = `VARIABLE NAME`,
      var_label = DESCRIPTION
    )
}

round_feature_table <- function(table) {
  table |>
    mutate(
      across(ends_with("_percent"), ~ round(.x, 2)),
      across(contains("mean_abs_SHAP"), ~ round(.x, 4)),
      across(contains("mean_perm"), ~ round(.x, 4))
    )
}

make_shap_importance <- function(path, clock_label, name_map) {
  read_csv(path, show_col_types = FALSE) |>
    group_by(variable) |>
    summarise(
      mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
      .groups = "drop"
    ) |>
    left_join(name_map, by = "variable") |>
    arrange(desc(mean_abs_shap)) |>
    mutate(
      rank = row_number(),
      share_percent = mean_abs_shap / sum(mean_abs_shap, na.rm = TRUE) * 100
    ) |>
    select(rank, variable_label = var_label, mean_abs_shap, share_percent) |>
    rename(
      !!paste0("SHAP", clock_label, "_variable") := variable_label,
      !!paste0("SHAP", clock_label, "_mean_abs_SHAP") := mean_abs_shap,
      !!paste0("SHAP", clock_label, "_share_percent") := share_percent
    )
}

make_permutation_importance <- function(path, clock_values, clock_label,
                                         name_map) {
  read_csv(path, show_col_types = FALSE) |>
    group_by(clock, feature) |>
    summarise(
      mean_perm = mean(mean_drop, na.rm = TRUE),
      .groups = "drop"
    ) |>
    rename(variable = feature) |>
    filter(clock %in% clock_values) |>
    left_join(name_map, by = "variable") |>
    arrange(desc(mean_perm)) |>
    mutate(
      rank = row_number(),
      share_percent = mean_perm / sum(mean_perm, na.rm = TRUE) * 100
    ) |>
    select(rank, variable_label = var_label, mean_perm, share_percent) |>
    rename(
      !!paste0("Perm", clock_label, "_variable") := variable_label,
      !!paste0("Perm", clock_label, "_mean_perm") := mean_perm,
      !!paste0("Perm", clock_label, "_share_percent") := share_percent
    )
}

write_xgb_table <- function(shap_grim, shap_dune, permutation, output_file,
                            permutation_clocks = c("EAA_GRIMAGE", "EAA_DUNEDINMPOA")) {
  name_map <- read_name_map()
  grim_shap <- make_shap_importance(shap_grim, "Grimage", name_map)
  dune_shap <- make_shap_importance(shap_dune, "DunedinPoAm", name_map)
  grim_perm <- make_permutation_importance(
    permutation, permutation_clocks[1], "Grimage", name_map
  )
  dune_perm <- make_permutation_importance(
    permutation, permutation_clocks[2], "DunedinPoAm", name_map
  )

  summary_tbl <- grim_shap |>
    full_join(dune_shap, by = "rank") |>
    full_join(grim_perm, by = "rank") |>
    full_join(dune_perm, by = "rank") |>
    arrange(rank) |>
    round_feature_table()

  write_xlsx(summary_tbl, output_file)
}

write_rf_table <- function(permutation, output_file,
                           clock_values = c("EAA_GRIMAGE", "EAA_DUNEDINMPOA")) {
  name_map <- read_name_map()
  grim_perm <- make_permutation_importance(
    permutation, clock_values[1], "Grimage", name_map
  )
  dune_perm <- make_permutation_importance(
    permutation, clock_values[2], "DunedinPoAm", name_map
  )

  summary_tbl <- grim_perm |>
    full_join(dune_perm, by = "rank") |>
    arrange(rank) |>
    round_feature_table()

  write_xlsx(summary_tbl, output_file)
}

write_lasso_table <- function(path, output_file) {
  name_map <- read_name_map()
  lasso_imp <- read_csv(path, show_col_types = FALSE) |>
    mutate(importance = abs(mean_coef) * selection_freq) |>
    left_join(name_map, by = "variable") |>
    mutate(var_label = coalesce(var_label, variable))

  grim <- lasso_imp |>
    filter(clock == "EAA_GRIMAGE") |>
    arrange(desc(importance)) |>
    mutate(
      rank = row_number(),
      share_percent = importance / sum(importance, na.rm = TRUE) * 100
    ) |>
    select(
      rank,
      LASSOGrimage_variable = var_label,
      LASSOGrimage_mean_coef = mean_coef,
      LASSOGrimage_sd_coef = sd_coef,
      LASSOGrimage_selection_freq = selection_freq,
      LASSOGrimage_importance = importance,
      LASSOGrimage_share_percent = share_percent
    )

  dune <- lasso_imp |>
    filter(clock == "EAA_DUNEDINMPOA") |>
    arrange(desc(importance)) |>
    mutate(
      rank = row_number(),
      share_percent = importance / sum(importance, na.rm = TRUE) * 100
    ) |>
    select(
      rank,
      LASSODunedinPoAm_variable = var_label,
      LASSODunedinPoAm_mean_coef = mean_coef,
      LASSODunedinPoAm_sd_coef = sd_coef,
      LASSODunedinPoAm_selection_freq = selection_freq,
      LASSODunedinPoAm_importance = importance,
      LASSODunedinPoAm_share_percent = share_percent
    )

  summary_tbl <- grim |>
    full_join(dune, by = "rank") |>
    arrange(rank) |>
    mutate(
      across(ends_with("_percent"), ~ round(.x, 2)),
      across(contains("importance"), ~ round(.x, 4))
    )

  write_xlsx(summary_tbl, output_file)
}

make_stratified_shap <- function(path, clock_label, name_map) {
  read_csv(path, show_col_types = FALSE) |>
    group_by(clock, stratum, variable) |>
    summarise(
      mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
      .groups = "drop"
    ) |>
    left_join(name_map, by = "variable") |>
    group_by(clock, stratum) |>
    arrange(desc(mean_abs_shap), .by_group = TRUE) |>
    mutate(
      rank = row_number(),
      share_percent = mean_abs_shap / sum(mean_abs_shap, na.rm = TRUE) * 100,
      clock_label = clock_label
    ) |>
    ungroup() |>
    select(clock_label, stratify_col = stratum, rank,
           variable_label = var_label, mean_abs_shap, share_percent) |>
    mutate(
      mean_abs_shap = round(mean_abs_shap, 4),
      share_percent = round(share_percent, 2)
    )
}

stratified_block <- function(table, clock, stratum, include_rank = FALSE) {
  block <- table |>
    filter(clock_label == clock, stratify_col == stratum) |>
    arrange(rank)

  if (include_rank) {
    return(block |>
      select(rank, variable_label, mean_abs_shap, share_percent))
  }

  block |>
    select(variable_label, mean_abs_shap, share_percent)
}

write_stratified_xgb_table <- function(output_file) {
  name_map <- read_name_map()
  specs <- list(
    list("GrimAge", "GENDER_0.0", "Females", "shap_long_stratified_EAA_GRIMAGE_GENDER.csv"),
    list("GrimAge", "GENDER_1.0", "Males", "shap_long_stratified_EAA_GRIMAGE_GENDER.csv"),
    list("DunedinPoAm", "GENDER_0.0", "Females", "shap_long_stratified_EAA_DUNEDINMPOA_GENDER.csv"),
    list("DunedinPoAm", "GENDER_1.0", "Males", "shap_long_stratified_EAA_DUNEDINMPOA_GENDER.csv"),
    list("GrimAge", "WHITE_BINARY_1", "Non-Hispanic White", "shap_long_stratified_EAA_GRIMAGE_WHITE_BINARY.csv"),
    list("GrimAge", "WHITE_BINARY_0", "Other ethnicities", "shap_long_stratified_EAA_GRIMAGE_WHITE_BINARY.csv"),
    list("DunedinPoAm", "WHITE_BINARY_1", "Non-Hispanic White", "shap_long_stratified_EAA_DUNEDINMPOA_WHITE_BINARY.csv"),
    list("DunedinPoAm", "WHITE_BINARY_0", "Other ethnicities", "shap_long_stratified_EAA_DUNEDINMPOA_WHITE_BINARY.csv"),
    list("GrimAge", "EVER_SMOKED_RAND_0.0", "Never Smoked", "shap_long_stratified_EAA_GRIMAGE_EVER_SMOKED_RAND.csv"),
    list("GrimAge", "EVER_SMOKED_RAND_1.0", "Ever-Smoker", "shap_long_stratified_EAA_GRIMAGE_EVER_SMOKED_RAND.csv"),
    list("DunedinPoAm", "EVER_SMOKED_RAND_0.0", "Never Smoked", "shap_long_stratified_EAA_DUNEDINMPOA_EVER_SMOKED_RAND.csv"),
    list("DunedinPoAm", "EVER_SMOKED_RAND_1.0", "Ever-Smoker", "shap_long_stratified_EAA_DUNEDINMPOA_EVER_SMOKED_RAND.csv")
  )

  tables <- lapply(specs, function(spec) {
    make_stratified_shap(file.path(XGB_REG_DIR, spec[[4]]), spec[[1]], name_map)
  })

  blocks <- Map(function(table, spec, index) {
    block <- stratified_block(table, spec[[1]], spec[[2]], include_rank = index == 1)
    names(block) <- if (index == 1) {
      c("Rank", "Variable", "|Mean| SHAP", "SHAP %")
    } else {
      c(paste0("Variable_", index), paste0("|Mean| SHAP_", index), paste0("SHAP %_", index))
    }
    block
  }, tables, specs, seq_along(specs))

  # The published Data S9 layout has one rank column followed by paired
  # variable/value/percentage blocks. Blocks are aligned by rank position;
  # absent trailing ranks remain blank rather than being joined by labels.
  n_rows <- max(vapply(blocks, nrow, integer(1)))
  blocks <- lapply(blocks, function(block) {
    block[seq_len(n_rows), , drop = FALSE]
  })
  output <- bind_cols(blocks)
  write_xlsx(output, output_file)
}

write_stratified_lasso_table <- function(path, output_file) {
  name_map <- read_name_map()
  summary_tbl <- read_csv(path, show_col_types = FALSE) |>
    mutate(importance = abs(mean_coef) * selection_freq) |>
    left_join(name_map, by = "variable") |>
    mutate(var_label = coalesce(var_label, variable)) |>
    group_by(clock, stratum) |>
    arrange(desc(importance), .by_group = TRUE) |>
    mutate(
      rank = row_number(),
      share_percent = importance / sum(importance, na.rm = TRUE) * 100
    ) |>
    ungroup() |>
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
    ) |>
    mutate(
      mean_coef = round(mean_coef, 4),
      sd_coef = round(sd_coef, 4),
      selection_freq = round(selection_freq, 3),
      importance = round(importance, 4),
      share_percent = round(share_percent, 2)
    )

  write_xlsx(summary_tbl, output_file)
}

main <- function() {
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

  write_lasso_table(
    file.path(LASSO_DIR, "lasso_coefficients.csv"),
    file.path(OUTPUT_DIR, "Data_S2_LASSO_feature_importance.xlsx")
  )

  write_rf_table(
    file.path(RF_REG_DIR, "permutation_feature_standard.csv"),
    file.path(OUTPUT_DIR, "Data_S3_RF_regression_feature_importance.xlsx")
  )

  write_xgb_table(
    file.path(XGB_REG_DIR, "shap_long_EAA_GRIMAGE.csv"),
    file.path(XGB_REG_DIR, "shap_long_EAA_DUNEDINMPOA.csv"),
    file.path(XGB_REG_DIR, "permutation_feature_standard.csv"),
    file.path(OUTPUT_DIR, "Data_S4_XGB_regression_feature_importance.xlsx")
  )

  write_stratified_lasso_table(
    file.path(LASSO_DIR, "stratified_coefficients.csv"),
    file.path(OUTPUT_DIR, "Data_S8_stratified_LASSO_feature_importance.xlsx")
  )

  write_stratified_xgb_table(
    file.path(OUTPUT_DIR, "Data_S9_stratified_XGB_SHAP_feature_importance.xlsx")
  )

  write_rf_table(
    file.path(RF_CLASS_DIR, "permutation_feature_standard.csv"),
    file.path(OUTPUT_DIR, "Data_S13_RF_classifier_feature_importance.xlsx"),
    clock_values = c("EAA_GRIMAGE_BINARY", "EAA_DUNEDINMPOA_BINARY")
  )

  write_xgb_table(
    file.path(XGB_CLASS_DIR, "shap_long_EAA_GRIMAGE_BINARY.csv"),
    file.path(XGB_CLASS_DIR, "shap_long_EAA_DUNEDINMPOA_BINARY.csv"),
    file.path(XGB_CLASS_DIR, "permutation_feature_standard.csv"),
    file.path(OUTPUT_DIR, "Data_S14_XGB_classifier_feature_importance.xlsx"),
    permutation_clocks = c("EAA_GRIMAGE_BINARY", "EAA_DUNEDINMPOA_BINARY")
  )
}

if (sys.nframe() == 0) {
  main()
}
