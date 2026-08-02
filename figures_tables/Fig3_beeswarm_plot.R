library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggbeeswarm)
library(viridis)
library(patchwork)

shap_grim_path <- "output/analyses/XGB/Regressor/shap_long_EAA_GRIMAGE.csv"
shap_dune_path <- "output/analyses/XGB/Regressor/shap_long_EAA_DUNEDINMPOA.csv"
dict_path <- "variables_dictionary.csv"
orig_df_path <- "output/epigenetic_age_events.csv"
out_png_panel <- "output/figures/Fig3.png"
TOP_N <- 15

dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)

make_beeswarm <- function(shap_long_path, target_title, top_n = 15) {
    shap_long <- read_csv(shap_long_path, show_col_types = FALSE)
    orig_full <- read_csv(orig_df_path, show_col_types = FALSE)
    dict <- read_csv(dict_path, show_col_types = FALSE) |>
        select(raw_var, description)

    top_feats <- shap_long |>
        group_by(variable) |>
        summarise(mean_abs = mean(abs(shap_value), na.rm = TRUE), .groups = "drop") |>
        arrange(desc(mean_abs)) |>
        slice_head(n = top_n)

    value_long <- orig_full |>
        select(HHID, PN, all_of(top_feats$variable)) |>
        pivot_longer(
            cols = -c(HHID, PN),
            names_to = "variable",
            values_to = "feature_value"
        ) |>
        mutate(
            feature_value_num = case_when(
                is.numeric(feature_value) ~ as.numeric(feature_value),
                is.logical(feature_value) ~ as.numeric(feature_value),
                TRUE ~ as.numeric(as.factor(feature_value)) - 1
            )
        ) |>
        group_by(variable) |>
        mutate(
            nuniq = n_distinct(feature_value_num[!is.na(feature_value_num)]),
            is_binary = nuniq == 2,
            mean_v = mean(feature_value_num, na.rm = TRUE),
            sd_v = sd(feature_value_num, na.rm = TRUE),
            z_raw = ifelse(sd_v > 0, (feature_value_num - mean_v) / sd_v, 0),
            z_clamp = pmin(pmax(z_raw, -3), 3),
            cont_scaled = (z_clamp + 3) / 6,
            feature_value_scaled = ifelse(is_binary, feature_value_num, cont_scaled)
        ) |>
        ungroup() |>
        select(HHID, PN, variable, feature_value, feature_value_scaled)

    plot_df <- shap_long |>
        semi_join(top_feats, by = "variable") |>
        left_join(value_long, by = c("HHID", "PN", "variable")) |>
        left_join(dict, by = c("variable" = "raw_var")) |>
        mutate(
            label = if_else(is.na(description), variable, description),
            label = str_wrap(label, 28)
        ) |>
        group_by(label) |>
        mutate(mean_abs = mean(abs(shap_value), na.rm = TRUE)) |>
        ungroup() |>
        mutate(label = reorder(label, mean_abs))

    ggplot(plot_df, aes(x = shap_value, y = label, colour = feature_value_scaled)) +
        geom_quasirandom(alpha = 0.55, size = 1.2, bandwidth = 0.2) +
        geom_vline(xintercept = 0, linetype = "dashed") +
        scale_color_viridis_c(
            option = "magma",
            direction = -1,
            begin = 0.15,
            end = 0.90,
            na.value = "grey70",
            name = "Feature value\n(high → low)"
        ) +
        labs(
            title = target_title,
            x = "SHAP value",
            y = NULL
        ) +
        theme_minimal(base_size = 18) +
        theme(
            plot.title = element_text(hjust = 0.5, face = "bold", size = 18,
                                      margin = margin(b = 8)),
            axis.text.y = element_text(size = 16, lineheight = 0.95),
            axis.text.x = element_text(size = 14),
            axis.title.x = element_text(size = 16, face = "bold",
                                        margin = margin(t = 8)),
            panel.grid.minor = element_blank(),
            legend.title = element_text(size = 16, face = "bold"),
            legend.text = element_text(size = 14)
        )
}

p_grim <- make_beeswarm(shap_grim_path, "GrimAge", TOP_N) +
    scale_x_continuous(breaks = seq(-2, 6, by = 1))

p_dune <- make_beeswarm(shap_dune_path, "DunedinPoAm", TOP_N) +
    scale_x_continuous(breaks = seq(-0.025, 0.075, by = 0.01))

panel <- (p_grim / p_dune) +
    plot_layout(ncol = 1, guides = "collect") +
    plot_annotation(tag_levels = "A") &
    theme(
        legend.position = "right",
        plot.tag = element_text(size = 22, face = "bold")
    )

ggsave(
    filename = out_png_panel,
    plot = panel,
    width = 12,
    height = 18,
    dpi = 600
)
