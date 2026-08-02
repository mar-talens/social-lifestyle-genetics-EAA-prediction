# ===============================
# Figura: R² (±DE) por reloj y algoritmo (Regresión, ALL-vars)
# ===============================

library(readr)
library(dplyr)
library(stringr)
library(forcats)
library(ggplot2)

# ---- EDITA ESTAS RUTAS (RF y XGB) ----
rf_perf_path  <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/RF/Regressor/model_performance.csv"
xgb_perf_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/model_performance.csv"
lasso_perf_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/LASSO1/LASSO/model_performance.csv"

lasso <- read_csv(lasso_perf_path, show_col_types = FALSE) |>
    mutate(
        Algorithm = "LASSO",
        model = "all_vars",   # force compatibility with your filter
        stratum = "ALL"       # same here
    )


rf  <- read_csv(rf_perf_path,  show_col_types = FALSE) |> mutate(Algorithm = "Random Forest")
xgb <- read_csv(xgb_perf_path, show_col_types = FALSE) |> mutate(Algorithm = "XGBoost")
lasso <- read_csv(lasso_perf_path, show_col_types = FALSE) |> mutate(Algorithm = "LASSO")
lasso <- read_csv(lasso_perf_path, show_col_types = FALSE) |>
    mutate(
        Algorithm = "LASSO",
        model = "all_vars",
        stratum = "ALL"
    )
perf <- bind_rows(rf, xgb, lasso)

# ---- Filtra: solo modelos ALL-vars, stratum ALL, tareas de regresión (tendrán r2) ----
perf_clean <- perf |>
    filter(
        model == "all_vars",
        stratum == "ALL",
        !is.na(mean_r2)
    )

# (Opcional) limita a los 5 relojes de interés por si hay otros en el archivo
target_clocks <- c("EAA_HORVATH", "EAA_HANNUM", "EAA_LEVINE", "EAA_GRIMAGE", "EAA_DUNEDINMPOA")
perf_clean <- perf_clean |> filter(clock %in% target_clocks)

# ---- Orden de los relojes: por R² medio de XGB (desc) para una lectura clara ----
order_df <- perf_clean |>
    filter(Algorithm == "XGBoost") |>
    select(clock, mean_r2) |>
    distinct() |>
    arrange(desc(mean_r2))

clock_levels <- order_df$clock
# Si faltara alguno en XGB, añádelo al final manteniendo su orden original:
clock_levels <- c(clock_levels, setdiff(target_clocks, clock_levels))

perf_clean <- perf_clean |>
    mutate(clock = factor(clock, levels = clock_levels),
           # Etiquetas más amigables si quieres:
           clock_lab = recode(as.character(clock),
                              "EAA_HORVATH"     = "Horvath",
                              "EAA_HANNUM"      = "Hannum",
                              "EAA_LEVINE"      = "PhenoAge",
                              "EAA_GRIMAGE"     = "GrimAge",
                              "EAA_DUNEDINMPOA" = "DunedinPoAm"))

# ---- Gráfico: barras agrupadas + barras de error (DE entre folds)
# ---- Gráfico final estilo "BMC Medicine" ----

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
        # --- General layout ---
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        axis.text = element_text(color = "black", size = 11),
        axis.title.y = element_text(size = 12, margin = margin(r = 8)),
        legend.position = "top",
        legend.justification = "center",
        legend.direction = "horizontal",
        legend.text = element_text(size = 11),
        plot.margin = margin(5, 5, 5, 5),
        # --- Remove panel background for publication look ---
        panel.background = element_rect(fill = "white", colour = NA),
        plot.background  = element_rect(fill = "white", colour = NA)
    )

# Guarda versión definitiva
ggsave(
    "/Users/martalens/Desktop/ML_EAA/Paper_graphs/F1_modelperformance.png",
    p, width = 7, height = 4.5, dpi = 600
)