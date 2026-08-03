# Supplementary Figure S2 — Learning curves, 5 clocks × 3 algorithms
library(readr)
library(dplyr)
library(ggplot2)
library(purrr)

dir_xgb <- "output/analyses/XGB/Regressor"
dir_rf <- "output/analyses/RF/Regressor"
dir_lasso <- "output/analyses/LASSO"
out_dir <- "output/figures"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

clocks <- c(
    "EAA_DUNEDINMPOA", "EAA_GRIMAGE", "EAA_LEVINE",
    "EAA_HORVATH", "EAA_HANNUM"
)

clock_labels <- c(
    EAA_DUNEDINMPOA = "DunedinPoAm",
    EAA_GRIMAGE = "GrimAge",
    EAA_LEVINE = "PhenoAge",
    EAA_HORVATH = "Horvath",
    EAA_HANNUM = "Hannum"
)

read_lc_file_lasso <- function(dir_alg, clock_name, algo_label) {
    fn <- file.path(dir_alg, "learning_curve.csv")
    if (!file.exists(fn)) return(NULL)

    read_csv(fn, show_col_types = FALSE) |>
        filter(clock == clock_name) |>
        mutate(
            Clock = recode(clock, !!!clock_labels),
            Algorithm = algo_label
        ) |>
        transmute(
            Clock, Algorithm,
            train_n = round(train_n_mean),
            train_mean = train_mean,
            train_sd = train_sd,
            val_mean = val_mean,
            val_sd = val_sd
        )
}

read_lc_file <- function(dir_alg, clock, algo_label) {
    fn <- file.path(dir_alg, paste0("learning_curve_", clock, ".csv"))
    if (!file.exists(fn)) return(NULL)

    read_csv(fn, show_col_types = FALSE) |>
        mutate(
            clock = clock,
            Clock = recode(clock, !!!clock_labels),
            Algorithm = algo_label
        ) |>
        transmute(
            Clock, Algorithm,
            train_n = train_n_mean,
            train_mean = train_score_mean,
            train_sd = train_score_sd,
            val_mean = val_score_mean,
            val_sd = val_score_sd
        )
}

lc_xgb <- map_dfr(clocks, ~read_lc_file(dir_xgb, .x, "XGBoost"))
lc_rf <- map_dfr(clocks, ~read_lc_file(dir_rf, .x, "Random Forest"))
lc_lasso <- map_dfr(clocks, ~read_lc_file_lasso(dir_lasso, .x, "LASSO"))

lc <- bind_rows(lc_lasso, lc_rf, lc_xgb)
stopifnot(nrow(lc) > 0)

lc$Clock <- factor(
    lc$Clock,
    levels = c("DunedinPoAm", "GrimAge", "PhenoAge", "Horvath", "Hannum")
)
lc$Algorithm <- factor(
    lc$Algorithm,
    levels = c("LASSO", "Random Forest", "XGBoost")
)

lc_val <- lc |>
    transmute(
        Clock, Algorithm, train_n,
        mean = val_mean, sd = val_sd,
        Metric = "Validation"
    )

lc_train <- lc |>
    transmute(
        Clock, Algorithm, train_n,
        mean = train_mean, sd = train_sd,
        Metric = "Training"
    )

lc_long <- bind_rows(lc_val, lc_train)

read_ref <- function(dir_alg, algo_label) {
    fp <- file.path(dir_alg, "model_performance.csv")
    if (!file.exists(fp)) return(tibble())

    df <- read_csv(fp, show_col_types = FALSE)
    if ("model" %in% colnames(df) && "stratum" %in% colnames(df)) {
        df <- df |>
            filter(model == "all_vars", stratum == "ALL")
    }

    df |>
        filter(clock %in% clocks) |>
        distinct(clock, .keep_all = TRUE) |>
        mutate(
            Clock = recode(clock, !!!clock_labels),
            Algorithm = algo_label
        ) |>
        transmute(Clock, Algorithm, ref_r2 = mean_r2)
}

ref <- bind_rows(
    read_ref(dir_lasso, "LASSO"),
    read_ref(dir_rf, "Random Forest"),
    read_ref(dir_xgb, "XGBoost")
) |>
    mutate(Metric = "Outer-test (full model)")

p <- ggplot() +
    geom_ribbon(
        data = dplyr::filter(lc_long, Metric == "Validation"),
        aes(
            x = train_n, ymin = mean - sd, ymax = mean + sd,
            fill = Metric
        ),
        alpha = 0.20, colour = NA
    ) +
    geom_line(
        data = dplyr::filter(lc_long, Metric %in% c("Training", "Validation")),
        aes(
            x = train_n, y = mean,
            colour = Metric, linetype = Metric
        ),
        linewidth = 0.9
    ) +
    geom_hline(
        data = ref,
        aes(
            yintercept = ref_r2,
            colour = Metric, linetype = Metric
        ),
        linewidth = 0.9
    ) +
    facet_grid(Clock ~ Algorithm, scales = "fixed") +
    scale_colour_manual(
        name = NULL,
        values = c(
            "Validation" = "#1b9e77",
            "Training" = "#d95f02",
            "Outer-test (full model)" = "#7570b3"
        )
    ) +
    scale_fill_manual(
        values = c("Validation" = "#1b9e77"),
        guide = "none"
    ) +
    scale_linetype_manual(
        name = NULL,
        values = c(
            "Validation" = "solid",
            "Training" = "dashed",
            "Outer-test (full model)" = "dotted"
        )
    ) +
    labs(
        x = "Training set size (per outer fold)",
        y = expression(R^2),
        title = "Learning curves by algorithm and epigenetic clock"
    ) +
    theme_bw(base_size = 11) +
    theme(
        panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold"),
        legend.position = "top",
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill = NA),
        panel.spacing = unit(0.7, "lines")
    )

out_png <- file.path(out_dir, "SuppFig2_learning_curves.png")
ggsave(out_png, p, width = 8, height = 11, dpi = 300)
