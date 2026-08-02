library(readr)
library(dplyr)
library(ggplot2)

rf_perf_path <- "output/analyses/RF/Regressor/model_performance.csv"
xgb_perf_path <- "output/analyses/XGB/Regressor/model_performance.csv"
lasso_perf_path <- "output/analyses/LASSO/model_performance.csv"
figure_dir <- "output/figures"
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

lasso <- read_csv(lasso_perf_path, show_col_types = FALSE) |>
    mutate(
        Algorithm = "LASSO",
        model = "all_vars",
        stratum = "ALL"
    )

rf <- read_csv(rf_perf_path, show_col_types = FALSE) |>
    mutate(Algorithm = "Random Forest")
xgb <- read_csv(xgb_perf_path, show_col_types = FALSE) |>
    mutate(Algorithm = "XGBoost")
perf <- bind_rows(rf, xgb, lasso)

perf_clean <- perf |>
    filter(
        model == "all_vars",
        stratum == "ALL",
        !is.na(mean_r2)
    )

target_clocks <- c("EAA_HORVATH", "EAA_HANNUM", "EAA_LEVINE", "EAA_GRIMAGE", "EAA_DUNEDINMPOA")
perf_clean <- perf_clean |> filter(clock %in% target_clocks)

order_df <- perf_clean |>
    filter(Algorithm == "XGBoost") |>
    select(clock, mean_r2) |>
    distinct() |>
    arrange(desc(mean_r2))

clock_levels <- order_df$clock
clock_levels <- c(clock_levels, setdiff(target_clocks, clock_levels))

perf_clean <- perf_clean |>
    mutate(clock = factor(clock, levels = clock_levels),
           clock_lab = recode(as.character(clock),
                              "EAA_HORVATH"     = "Horvath",
                              "EAA_HANNUM"      = "Hannum",
                              "EAA_LEVINE"      = "PhenoAge",
                              "EAA_GRIMAGE"     = "GrimAge",
                              "EAA_DUNEDINMPOA" = "DunedinPoAm"))

pd <- position_dodge(width = 0.65)

p <- ggplot(perf_clean,
            aes(x = clock_lab, y = mean_r2, fill = Algorithm)) +
    geom_col(position = pd, width = 0.55, colour = "black", linewidth = 0) +
    geom_errorbar(aes(ymin = mean_r2 - sd_r2, ymax = mean_r2 + sd_r2),
                  position = pd, width = 0.25, linewidth = 0.4, colour = "black") +
    scale_fill_manual(
        values = c("XGBoost" = "#595959", "Random Forest" = "#C9C9C9", "LASSO" = "#A9A9A9"),
        breaks = c("XGBoost", "Random Forest", "LASSO"),
        name   = NULL
    ) +
    scale_y_continuous(
        breaks = seq(0, 0.4, 0.1),
        expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
        x = NULL,
        y = expression(R^2~"(mean ± SD across folds)")
    ) +
    theme_minimal(base_size = 12) +
    theme(
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        axis.text = element_text(color = "black", size = 11),
        axis.title.y = element_text(size = 12, margin = margin(r = 8)),
        legend.position = "top",
        legend.justification = "center",
        legend.direction = "horizontal",
        legend.text = element_text(size = 11),
        plot.margin = margin(5, 5, 5, 5),
        panel.background = element_rect(fill = "white", colour = NA),
        plot.background  = element_rect(fill = "white", colour = NA)
    )

ggsave(
    "output/figures/Fig1_modelperformance.png",
    p, width = 7, height = 4.5, dpi = 600
)
