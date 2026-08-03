# Supplementary Figures S4--S6: stratified XGBoost feature importance
#
# This script consolidates the three repeated workflows in
# graph_featimp_SHAP_stratified.R.  SHAP files are generated locally by the
# XGBoost analysis and are intentionally not part of the public repository.

library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(scales)
library(patchwork)

SHAP_DIR <- "output/analyses/XGB/Regressor"
LABEL_FILE <- "variables_dictionary.csv"
FIGURE_DIR <- "output/figures"

DOMAIN_COLOURS <- c(
  GrimAge = "#BF7C3F",
  DunedinPoAm = "#3B74A8"
)

# Canonicalise the stratum labels emitted by the analysis code while retaining
# the exact stratum-specific filtering used by the source plotting workflow.
canonicalize_stratum <- function(x, stratification) {
  x <- as.character(x)
  if (stratification == "GENDER") {
    return(case_when(
      x %in% c("GENDER_0.0", "GENDER_0", "GENDER0", "0", "0.0") ~ "GENDER_0.0",
      x %in% c("GENDER_1.0", "GENDER_1", "GENDER1", "1", "1.0") ~ "GENDER_1.0",
      TRUE ~ x
    ))
  }
  if (stratification == "WHITE_BINARY") {
    return(case_when(
      str_detect(x, "^WHITE_BINARY_0") ~ "WHITE_BINARY_0",
      str_detect(x, "^WHITE_BINARY_1") ~ "WHITE_BINARY_1",
      TRUE ~ x
    ))
  }
  if (stratification == "EVER_SMOKED_RAND") {
    return(case_when(
      str_detect(x, "^EVER_SMOKED_RAND_0") | x %in% c("0", "0.0") ~ "EVER_SMOKED_RAND_0.0",
      str_detect(x, "^EVER_SMOKED_RAND_1") | x %in% c("1", "1.0") ~ "EVER_SMOKED_RAND_1.0",
      TRUE ~ x
    ))
  }
  x
}

shap_to_importance <- function(path, clock_name, stratification) {
  read_csv(path, show_col_types = FALSE) |>
    mutate(
      stratum_raw = as.character(stratum),
      stratum_clean = canonicalize_stratum(stratum_raw, stratification)
    ) |>
    group_by(variable, stratum_clean) |>
    summarise(
      mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(clock = clock_name)
}

plot_for_stratum <- function(importance, labels, stratum_code, heading, subtitle) {
  imp_s <- importance |>
    filter(stratum_clean == stratum_code)

  # The source selects the top 15 predictors separately for each clock, then
  # retains their union while keeping both clocks' values for every predictor.
  top15 <- imp_s |>
    group_by(clock) |>
    slice_max(order_by = mean_abs_shap, n = 15, with_ties = FALSE) |>
    ungroup()
  keep_vars <- unique(top15$variable)

  plot_data <- imp_s |>
    group_by(clock) |>
    mutate(importance_prop = mean_abs_shap / sum(mean_abs_shap, na.rm = TRUE)) |>
    ungroup() |>
    filter(variable %in% keep_vars) |>
    left_join(top15 |>
      transmute(variable, clock, in_top15 = TRUE), by = c("variable", "clock")) |>
    mutate(
      in_top15 = replace_na(in_top15, FALSE),
      alpha_value = if_else(in_top15, 1, 0.25)
    ) |>
    left_join(labels, by = c("variable" = "raw_var")) |>
    mutate(label = str_wrap(coalesce(description, variable), width = 26))

  # Order by GrimAge importance, falling back to DunedinPoAm when GrimAge is
  # absent, exactly as in the authoritative source.
  ordering <- plot_data |>
    select(variable, clock, importance_prop) |>
    pivot_wider(names_from = clock, values_from = importance_prop) |>
    mutate(
      GrimAge = coalesce(GrimAge, 0),
      DunedinPoAm = coalesce(DunedinPoAm, 0),
      order_value = if_else(GrimAge > 0, GrimAge, DunedinPoAm)
    ) |>
    arrange(order_value) |>
    select(variable)

  label_levels <- labels |>
    filter(raw_var %in% ordering$variable) |>
    mutate(label = str_wrap(coalesce(description, raw_var), width = 26)) |>
    arrange(match(raw_var, ordering$variable)) |>
    pull(label)

  plot_data <- plot_data |>
    mutate(label = factor(label, levels = label_levels))

  ggplot(
    plot_data,
    aes(
      x = importance_prop,
      y = label,
      fill = clock,
      alpha = alpha_value,
      group = clock
    )
  ) +
    geom_col(position = position_dodge(width = 0.6), width = 0.35) +
    scale_alpha_identity() +
    scale_fill_manual(values = DOMAIN_COLOURS, name = NULL) +
    scale_x_continuous(
      labels = percent_format(accuracy = 1),
      breaks = breaks_pretty(n = 5),
      expand = expansion(mult = c(0, 0.05))
    ) +
    labs(
      title = heading,
      subtitle = subtitle,
      x = "Mean absolute SHAP value (% within clock and stratum)",
      y = NULL
    ) +
    theme_minimal(base_size = 18) +
    theme(
      legend.position = "top",
      legend.direction = "horizontal",
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(size = 13),
      axis.text.y = element_text(size = 12),
      panel.grid.major.y = element_blank(),
      plot.margin = margin(10, 10, 10, 25)
    )
}

make_stratified_figure <- function(stratification, clock_files, strata, headings,
                                   subtitle, output_file) {
  labels <- read_csv(LABEL_FILE, show_col_types = FALSE) |>
    select(raw_var, description)

  importance <- bind_rows(
    shap_to_importance(clock_files$GrimAge, "GrimAge", stratification),
    shap_to_importance(clock_files$DunedinPoAm, "DunedinPoAm", stratification)
  )

  plots <- Map(
    f = function(code, title) {
      plot_for_stratum(importance, labels, code, title, subtitle)
    },
    code = strata,
    title = headings
  )

  panel <- wrap_plots(plots, ncol = 2) +
    plot_layout(guides = "collect") &
    theme(legend.position = "top")

  ggsave(
    filename = output_file,
    plot = panel,
    width = 20,
    height = 14,
    dpi = 300
  )
}

main <- function() {
  dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

  make_stratified_figure(
    stratification = "GENDER",
    clock_files = list(
      GrimAge = file.path(SHAP_DIR, "shap_long_stratified_EAA_GRIMAGE_GENDER.csv"),
      DunedinPoAm = file.path(SHAP_DIR, "shap_long_stratified_EAA_DUNEDINMPOA_GENDER.csv")
    ),
    strata = c("GENDER_0.0", "GENDER_1.0"),
    headings = c("Females", "Males"),
    subtitle = "Top 15 predictors per clock, ordered by GrimAge importance",
    output_file = file.path(FIGURE_DIR, "SuppFig4_stratified_gender_feature_importance.png")
  )

  make_stratified_figure(
    stratification = "WHITE_BINARY",
    clock_files = list(
      GrimAge = file.path(SHAP_DIR, "shap_long_stratified_EAA_GRIMAGE_WHITE_BINARY.csv"),
      DunedinPoAm = file.path(SHAP_DIR, "shap_long_stratified_EAA_DUNEDINMPOA_WHITE_BINARY.csv")
    ),
    strata = c("WHITE_BINARY_1", "WHITE_BINARY_0"),
    headings = c("Non-Hispanic White", "Other ethnicities"),
    subtitle = "Top 15 predictors per clock, ordered by GrimAge importance",
    output_file = file.path(FIGURE_DIR, "SuppFig5_stratified_ethnicity_feature_importance.png")
  )

  make_stratified_figure(
    stratification = "EVER_SMOKED_RAND",
    clock_files = list(
      GrimAge = file.path(SHAP_DIR, "shap_long_stratified_EAA_GRIMAGE_EVER_SMOKED_RAND.csv"),
      DunedinPoAm = file.path(SHAP_DIR, "shap_long_stratified_EAA_DUNEDINMPOA_EVER_SMOKED_RAND.csv")
    ),
    strata = c("EVER_SMOKED_RAND_1.0", "EVER_SMOKED_RAND_0.0"),
    headings = c("Ever smoker", "Never smoked"),
    subtitle = "Top 15 predictors per clock, ordered by GrimAge importance",
    output_file = file.path(FIGURE_DIR, "SuppFig6_stratified_smoking_feature_importance.png")
  )
}

if (sys.nframe() == 0) {
  main()
}
