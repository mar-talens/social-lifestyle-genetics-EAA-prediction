# install.packages("ggbeeswarm") # if you don't have it
library(readr)
library(dplyr)
library(ggplot2)
library(ggbeeswarm)
library(stringr)
library(tidyr)



##### GRIMAGE BINARY!!!
# ---- inputs (edit these paths) ----
shap_long_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Classifier/shap_long_EAA_GRIMAGE_BINARY.csv"  
dict_path      <- "/Users/martalens/Desktop/ML_EAA/data/metadata/variables_dictionary.csv"
orig_df_path   <- "/Users/martalens/Desktop/ML_EAA/output/dataframes/epigenetic_age_events.csv"  
out_png        <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/plots/supplementary/shap_beeswarm_grimage_bin_top10.png"
TOP_N          <- 15
target_label    <- "GrimAge (Binary)"
x_label        <- NULL

# -------- read --------
shap_long <- read_csv(shap_long_path, show_col_types = FALSE)
orig_full <- read_csv(orig_df_path,   show_col_types = FALSE)   # includes HHID, PN, predictors
dict <- read_csv(dict_path, show_col_types = FALSE) %>%
    select(raw_var, description)


# -------- pick top N by mean |SHAP| --------
top_feats <- shap_long %>%
    group_by(variable) %>%
    summarise(mean_abs = mean(abs(shap_value), na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(mean_abs)) %>%
    slice_head(n = TOP_N)

# -------- pull ONLY those feature values from original data --------
value_long <- orig_full %>%
    select(HHID, PN, all_of(top_feats$variable)) %>%
    pivot_longer(
        cols = -c(HHID, PN),
        names_to = "variable",
        values_to = "feature_value"
    )

value_long <- value_long %>%
    mutate(
        feature_value_num = case_when(
            is.numeric(feature_value) ~ as.numeric(feature_value),
            is.logical(feature_value) ~ as.numeric(feature_value),     # FALSE=0, TRUE=1
            TRUE ~ as.numeric(as.factor(feature_value)) - 1            # e.g., "No/Yes" -> 0/1
        )
    ) %>%
    group_by(variable) %>%
    mutate(
        nuniq     = n_distinct(feature_value_num[!is.na(feature_value_num)]),
        is_binary = nuniq == 2,
        
        mean_v = mean(feature_value_num, na.rm = TRUE),
        sd_v   = sd(feature_value_num,   na.rm = TRUE),
        
        # Continuous: z-score, clamp, rescale to [0,1]
        z_raw   = ifelse(sd_v > 0, (feature_value_num - mean_v) / sd_v, 0),
        z_clamp = pmin(pmax(z_raw, -3), 3),
        cont_scaled = (z_clamp + 3) / 6,
        
        # Final color value: binaries stay 0/1; continuous use cont_scaled
        feature_value_scaled = ifelse(is_binary, feature_value_num, cont_scaled)
    ) %>%
    ungroup() %>%
    select(HHID, PN, variable, feature_value, feature_value_scaled)


# -------- assemble plotting df --------
plot_df <- shap_long %>%
    semi_join(top_feats, by = "variable") %>%
    left_join(value_long, by = c("HHID","PN","variable")) %>%
    left_join(dict, by = c("variable" = "raw_var")) %>%
    mutate(label = if_else(is.na(description), variable, description),
           label = str_wrap(label, 28)) %>%
    group_by(label) %>%
    mutate(mean_abs = mean(abs(shap_value), na.rm = TRUE)) %>%
    ungroup() %>%
    mutate(label = reorder(label, mean_abs))

# -------- beeswarm colored by predictor value --------
p <- ggplot(plot_df, aes(x = shap_value, y = label, color = feature_value_scaled)) +
    geom_quasirandom(alpha = 0.55, size = 0.9, bandwidth = 0.2) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    scale_color_viridis_c(
        option = "magma",
        direction = -1,     # invert: dark = higher values
        begin = 0.15,       # trim off the very light yellows
        end   = 0.90,       # trim off the deepest blacks
        na.value = "grey70",
        name = "Feature value\n(high → low)"   # legend matches the inversion
    ) +
    labs(
        title = NULL,
        subtitle = NULL,
        x = "SHAP values", y = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5))

print(p)
ggsave(out_png, p, width = 9, height = 8, dpi = 300)








##### GRIMAGE REGRESSOR!!!
# ---- inputs (edit these paths) ----
shap_long_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_EAA_GRIMAGE.csv"  
dict_path      <- "/Users/martalens/Desktop/ML_EAA/data/metadata/variables_dictionary.csv"
orig_df_path   <- "/Users/martalens/Desktop/ML_EAA/output/dataframes/epigenetic_age_events.csv"  
out_png        <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/plots/shap_beeswarm_grimage_regr_top10.png"
TOP_N          <- 15
target_label    <- "GrimAge (Regressor)"
x_label        <- NULL

# -------- read --------
shap_long <- read_csv(shap_long_path, show_col_types = FALSE)
orig_full <- read_csv(orig_df_path,   show_col_types = FALSE)   # includes HHID, PN, predictors
dict <- read_csv(dict_path, show_col_types = FALSE) %>%
    select(raw_var, description)


# -------- pick top N by mean |SHAP| --------
top_feats <- shap_long %>%
    group_by(variable) %>%
    summarise(mean_abs = mean(abs(shap_value), na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(mean_abs)) %>%
    slice_head(n = TOP_N)

# -------- pull ONLY those feature values from original data --------
value_long <- orig_full %>%
    select(HHID, PN, all_of(top_feats$variable)) %>%
    pivot_longer(
        cols = -c(HHID, PN),
        names_to = "variable",
        values_to = "feature_value"
    )

value_long <- value_long %>%
    mutate(
        feature_value_num = case_when(
            is.numeric(feature_value) ~ as.numeric(feature_value),
            is.logical(feature_value) ~ as.numeric(feature_value),     # FALSE=0, TRUE=1
            TRUE ~ as.numeric(as.factor(feature_value)) - 1            # e.g., "No/Yes" -> 0/1
        )
    ) %>%
    group_by(variable) %>%
    mutate(
        nuniq     = n_distinct(feature_value_num[!is.na(feature_value_num)]),
        is_binary = nuniq == 2,
        
        mean_v = mean(feature_value_num, na.rm = TRUE),
        sd_v   = sd(feature_value_num,   na.rm = TRUE),
        
        # Continuous: z-score, clamp, rescale to [0,1]
        z_raw   = ifelse(sd_v > 0, (feature_value_num - mean_v) / sd_v, 0),
        z_clamp = pmin(pmax(z_raw, -3), 3),
        cont_scaled = (z_clamp + 3) / 6,
        
        # Final color value: binaries stay 0/1; continuous use cont_scaled
        feature_value_scaled = ifelse(is_binary, feature_value_num, cont_scaled)
    ) %>%
    ungroup() %>%
    select(HHID, PN, variable, feature_value, feature_value_scaled)


# -------- assemble plotting df --------
plot_df <- shap_long %>%
    semi_join(top_feats, by = "variable") %>%
    left_join(value_long, by = c("HHID","PN","variable")) %>%
    left_join(dict, by = c("variable" = "raw_var")) %>%
    mutate(label = if_else(is.na(description), variable, description),
           label = str_wrap(label, 28)) %>%
    group_by(label) %>%
    mutate(mean_abs = mean(abs(shap_value), na.rm = TRUE)) %>%
    ungroup() %>%
    mutate(label = reorder(label, mean_abs))

# -------- beeswarm colored by predictor value --------
p <- ggplot(plot_df, aes(x = shap_value, y = label, color = feature_value_scaled)) +
    geom_quasirandom(alpha = 0.55, size = 0.9, bandwidth = 0.2) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    scale_color_viridis_c(
        option = "magma",
        direction = -1,     # invert: dark = higher values
        begin = 0.15,       # trim off the very light yellows
        end   = 0.90,       # trim off the deepest blacks
        na.value = "grey70",
        name = "Feature value\n(high → low)"   # legend matches the inversion
    ) +
    labs(
        title = NULL,
        subtitle = NULL,
        x = "SHAP values", y = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5))

print(p)
ggsave(out_png, p, width = 9, height = 8, dpi = 300)









##### DUNEDIN BINARY!!!
# ---- inputs (edit these paths) ----
shap_long_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Classifier/shap_long_EAA_DUNEDINMPOA_BINARY.csv"  
dict_path      <- "/Users/martalens/Desktop/ML_EAA/data/metadata/variables_dictionary.csv"
orig_df_path   <- "/Users/martalens/Desktop/ML_EAA/output/dataframes/epigenetic_age_events.csv"  
out_png        <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/plots/supplementary/shap_beeswarm_dunedin_bin_top10.png"
TOP_N          <- 15
target_label    <- "Dunedin (Binary)"
x_label        <- NULL

# -------- read --------
shap_long <- read_csv(shap_long_path, show_col_types = FALSE)
orig_full <- read_csv(orig_df_path,   show_col_types = FALSE)   # includes HHID, PN, predictors
dict <- read_csv(dict_path, show_col_types = FALSE) %>%
    select(raw_var, description)





# -------- pick top N by mean |SHAP| --------
top_feats <- shap_long %>%
    group_by(variable) %>%
    summarise(mean_abs = mean(abs(shap_value), na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(mean_abs)) %>%
    slice_head(n = TOP_N)

# -------- pull ONLY those feature values from original data --------
value_long <- orig_full %>%
    select(HHID, PN, all_of(top_feats$variable)) %>%
    pivot_longer(
        cols = -c(HHID, PN),
        names_to = "variable",
        values_to = "feature_value"
    )

value_long <- value_long %>%
    mutate(
        feature_value_num = case_when(
            is.numeric(feature_value) ~ as.numeric(feature_value),
            is.logical(feature_value) ~ as.numeric(feature_value),     # FALSE=0, TRUE=1
            TRUE ~ as.numeric(as.factor(feature_value)) - 1            # e.g., "No/Yes" -> 0/1
        )
    ) %>%
    group_by(variable) %>%
    mutate(
        nuniq     = n_distinct(feature_value_num[!is.na(feature_value_num)]),
        is_binary = nuniq == 2,
        
        mean_v = mean(feature_value_num, na.rm = TRUE),
        sd_v   = sd(feature_value_num,   na.rm = TRUE),
        
        # Continuous: z-score, clamp, rescale to [0,1]
        z_raw   = ifelse(sd_v > 0, (feature_value_num - mean_v) / sd_v, 0),
        z_clamp = pmin(pmax(z_raw, -3), 3),
        cont_scaled = (z_clamp + 3) / 6,
        
        # Final color value: binaries stay 0/1; continuous use cont_scaled
        feature_value_scaled = ifelse(is_binary, feature_value_num, cont_scaled)
    ) %>%
    ungroup() %>%
    select(HHID, PN, variable, feature_value, feature_value_scaled)


# -------- assemble plotting df --------
plot_df <- shap_long %>%
    semi_join(top_feats, by = "variable") %>%
    left_join(value_long, by = c("HHID","PN","variable")) %>%
    left_join(dict, by = c("variable" = "raw_var")) %>%
    mutate(label = if_else(is.na(description), variable, description),
           label = str_wrap(label, 28)) %>%
    group_by(label) %>%
    mutate(mean_abs = mean(abs(shap_value), na.rm = TRUE)) %>%
    ungroup() %>%
    mutate(label = reorder(label, mean_abs))

# -------- beeswarm colored by predictor value --------
p <- ggplot(plot_df, aes(x = shap_value, y = label, color = feature_value_scaled)) +
    geom_quasirandom(alpha = 0.55, size = 0.9, bandwidth = 0.2) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    scale_color_viridis_c(
        option = "magma",
        direction = -1,     # invert: dark = higher values
        begin = 0.15,       # trim off the very light yellows
        end   = 0.90,       # trim off the deepest blacks
        na.value = "grey70",
        name = "Feature value\n(high → low)"   # legend matches the inversion
    ) +
    labs(
        title = NULL,
        subtitle = NULL,
        x = "SHAP values", y = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5))

print(p)
ggsave(out_png, p, width = 9, height = 8, dpi = 300)






##### DUNEDIN REGRESSOR!!!
# ---- inputs (edit these paths) ----
shap_long_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_EAA_DUNEDINMPOA.csv"  
dict_path      <- "/Users/martalens/Desktop/ML_EAA/data/metadata/variables_dictionary.csv"
orig_df_path   <- "/Users/martalens/Desktop/ML_EAA/output/dataframes/epigenetic_age_events.csv"  
out_png        <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/plots/shap_beeswarm_dunedin_regr_top10.png"
TOP_N          <- 15
target_label    <- "Dunedin (Regressor)"
x_label        <- NULL

# -------- read --------
shap_long <- read_csv(shap_long_path, show_col_types = FALSE)
orig_full <- read_csv(orig_df_path,   show_col_types = FALSE)   # includes HHID, PN, predictors
dict <- read_csv(dict_path, show_col_types = FALSE) %>%
    select(raw_var, description)


# -------- pick top N by mean |SHAP| --------
top_feats <- shap_long %>%
    group_by(variable) %>%
    summarise(mean_abs = mean(abs(shap_value), na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(mean_abs)) %>%
    slice_head(n = TOP_N)

# -------- pull ONLY those feature values from original data --------
value_long <- orig_full %>%
    select(HHID, PN, all_of(top_feats$variable)) %>%
    pivot_longer(
        cols = -c(HHID, PN),
        names_to = "variable",
        values_to = "feature_value"
    )

value_long <- value_long %>%
    mutate(
        feature_value_num = case_when(
            is.numeric(feature_value) ~ as.numeric(feature_value),
            is.logical(feature_value) ~ as.numeric(feature_value),     # FALSE=0, TRUE=1
            TRUE ~ as.numeric(as.factor(feature_value)) - 1            # e.g., "No/Yes" -> 0/1
        )
    ) %>%
    group_by(variable) %>%
    mutate(
        nuniq     = n_distinct(feature_value_num[!is.na(feature_value_num)]),
        is_binary = nuniq == 2,
        
        mean_v = mean(feature_value_num, na.rm = TRUE),
        sd_v   = sd(feature_value_num,   na.rm = TRUE),
        
        # Continuous: z-score, clamp, rescale to [0,1]
        z_raw   = ifelse(sd_v > 0, (feature_value_num - mean_v) / sd_v, 0),
        z_clamp = pmin(pmax(z_raw, -3), 3),
        cont_scaled = (z_clamp + 3) / 6,
        
        # Final color value: binaries stay 0/1; continuous use cont_scaled
        feature_value_scaled = ifelse(is_binary, feature_value_num, cont_scaled)
    ) %>%
    ungroup() %>%
    select(HHID, PN, variable, feature_value, feature_value_scaled)


# -------- assemble plotting df --------
plot_df <- shap_long %>%
    semi_join(top_feats, by = "variable") %>%
    left_join(value_long, by = c("HHID","PN","variable")) %>%
    left_join(dict, by = c("variable" = "raw_var")) %>%
    mutate(label = if_else(is.na(description), variable, description),
           label = str_wrap(label, 28)) %>%
    group_by(label) %>%
    mutate(mean_abs = mean(abs(shap_value), na.rm = TRUE)) %>%
    ungroup() %>%
    mutate(label = reorder(label, mean_abs))

# -------- beeswarm colored by predictor value --------
p <- ggplot(plot_df, aes(x = shap_value, y = label, color = feature_value_scaled)) +
    geom_quasirandom(alpha = 0.55, size = 0.9, bandwidth = 0.2) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    scale_color_viridis_c(
        option = "magma",
        direction = -1,     # invert: dark = higher values
        begin = 0.15,       # trim off the very light yellows
        end   = 0.90,       # trim off the deepest blacks
        na.value = "grey70",
        name = "Feature value\n(high → low)"   # legend matches the inversion
    ) +
    labs(
        title = NULL,
        subtitle = NULL,
        x = "SHAP values", y = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5))

print(p)
ggsave(out_png, p, width = 9, height = 8, dpi = 300)











######## PANEL

# ============================
# XGB REGRESSOR — SHAP beeswarm panel (GrimAge + DunedinPoAm)
# ============================

# ---- packages ----
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggbeeswarm)
library(viridis)
library(patchwork)

# ---- inputs ----
shap_grim_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_EAA_GRIMAGE.csv"
shap_dune_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_EAA_DUNEDINMPOA.csv"
dict_path      <- "/Users/martalens/Desktop/ML_EAA/data/metadata/variables_dictionary.csv"
orig_df_path   <- "/Users/martalens/Desktop/ML_EAA/output/dataframes/epigenetic_age_events.csv"

out_png_panel  <- "/Users/martalens/Desktop/ML_EAA/Paper_graphs/Fig3.png"
TOP_N          <- 15

# ---------- helper: make one beeswarm plot ----------
make_beeswarm <- function(shap_long_path, target_title, top_n = 15) {
    
    # --- read ---
    shap_long <- read_csv(shap_long_path, show_col_types = FALSE)
    orig_full <- read_csv(orig_df_path,   show_col_types = FALSE)
    dict      <- read_csv(dict_path,      show_col_types = FALSE) |>
        select(raw_var, description)
    
    # --- pick top N by mean |SHAP| ---
    top_feats <- shap_long |>
        group_by(variable) |>
        summarise(mean_abs = mean(abs(shap_value), na.rm = TRUE), .groups = "drop") |>
        arrange(desc(mean_abs)) |>
        slice_head(n = top_n)
    
    # --- pull feature values from original data ---
    value_long <- orig_full |>
        select(HHID, PN, all_of(top_feats$variable)) |>
        pivot_longer(
            cols = -c(HHID, PN),
            names_to = "variable",
            values_to = "feature_value"
        )
    
    value_long <- value_long |>
        mutate(
            feature_value_num = case_when(
                is.numeric(feature_value) ~ as.numeric(feature_value),
                is.logical(feature_value) ~ as.numeric(feature_value),
                TRUE ~ as.numeric(as.factor(feature_value)) - 1
            )
        ) |>
        group_by(variable) |>
        mutate(
            nuniq     = n_distinct(feature_value_num[!is.na(feature_value_num)]),
            is_binary = nuniq == 2,
            mean_v    = mean(feature_value_num, na.rm = TRUE),
            sd_v      = sd(feature_value_num,   na.rm = TRUE),
            
            # continuous: z-score, clamp to [-3, 3], rescale to [0,1]
            z_raw        = ifelse(sd_v > 0, (feature_value_num - mean_v) / sd_v, 0),
            z_clamp      = pmin(pmax(z_raw, -3), 3),
            cont_scaled  = (z_clamp + 3) / 6,
            
            feature_value_scaled = ifelse(is_binary, feature_value_num, cont_scaled)
        ) |>
        ungroup() |>
        select(HHID, PN, variable, feature_value, feature_value_scaled)
    
    # --- assemble plotting df ---
    plot_df <- shap_long |>
        semi_join(top_feats, by = "variable") |>
        left_join(value_long, by = c("HHID", "PN", "variable")) |>
        left_join(dict, by = c("variable" = "raw_var")) |>
        mutate(label = if_else(is.na(description), variable, description),
               label = str_wrap(label, 28)) |>
        group_by(label) |>
        mutate(mean_abs = mean(abs(shap_value), na.rm = TRUE)) |>
        ungroup() |>
        mutate(label = reorder(label, mean_abs))
    
    # --- beeswarm plot ---
    p <- ggplot(plot_df, aes(x = shap_value, y = label, colour = feature_value_scaled)) +
        geom_quasirandom(alpha = 0.55, size = 1.2, bandwidth = 0.2) +
        geom_vline(xintercept = 0, linetype = "dashed") +
        scale_color_viridis_c(
            option = "magma",
            direction = -1,
            begin = 0.15,
            end   = 0.90,
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
            plot.title   = element_text(hjust = 0.5, face = "bold", size = 18,
                                        margin = margin(b = 8)),
            axis.text.y  = element_text(size = 16, lineheight = 0.95),
            axis.text.x  = element_text(size = 14),
            axis.title.x = element_text(size = 16, face = "bold", margin = margin(t = 8)),
            panel.grid.minor = element_blank(),
            legend.title = element_text(size = 16, face = "bold"),
            legend.text  = element_text(size = 14)
        )
    
    return(p)
}

# ---------- build both plots ----------
p_grim <- make_beeswarm(shap_grim_path,  "GrimAge",   TOP_N)
p_dune <- make_beeswarm(shap_dune_path,  "DunedinPoAm", TOP_N)


# ---------- build both plots ----------
p_grim <- make_beeswarm(shap_grim_path, "GrimAge", TOP_N) +
    scale_x_continuous(
        breaks = seq(-2, 6, by = 1),          # -2, -1, 0, 1, 2
        #minor_breaks = seq(-2, 2, by = 0.5)   # optional extra grid lines
    )

p_dune <- make_beeswarm(shap_dune_path, "DunedinPoAm", TOP_N) +
    scale_x_continuous(
        breaks = seq(-0.025, 0.075, by = 0.01)  # 0.01, 0.02, ...
        # minor_breaks = NULL  # or add tighter minor breaks if you want
    )

# ---------- combine into panel (same legend, stacked) ----------
panel <- (p_grim / p_dune) +
    plot_layout(ncol = 1, guides = "collect") +
    plot_annotation(tag_levels = "A") &
    theme(
        legend.position = "right",
        plot.tag = element_text(size = 22, face = "bold")
    )
# ---------- save panel ----------
ggsave(
    filename = out_png_panel,
    plot     = panel,
    width    = 12,
    height   = 18,
    dpi      = 600
)









######## PANEL

# ============================
# XGB REGRESSOR — SHAP beeswarm panel (GrimAge + DunedinPoAm)
# ============================

# ---- packages ----
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggbeeswarm)
library(viridis)
library(patchwork)

# ---- inputs ----
shap_grim_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Classifier/shap_long_EAA_GRIMAGE_BINARY.csv"
shap_dune_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Classifier/shap_long_EAA_DUNEDINMPOA_BINARY.csv"
dict_path      <- "/Users/martalens/Desktop/ML_EAA/data/metadata/variables_dictionary.csv"
orig_df_path   <- "/Users/martalens/Desktop/ML_EAA/output/dataframes/epigenetic_age_events.csv"

out_png_panel  <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/plots/supplementary/shap_beeswarm_GRIM_DUNE_panel_CLASSIFIER.png"
TOP_N          <- 15

# ---------- helper: make one beeswarm plot ----------
make_beeswarm <- function(shap_long_path, target_title, top_n = 15) {
    
    # --- read ---
    shap_long <- read_csv(shap_long_path, show_col_types = FALSE)
    orig_full <- read_csv(orig_df_path,   show_col_types = FALSE)
    dict      <- read_csv(dict_path,      show_col_types = FALSE) |>
        select(raw_var, description)
    
    # --- pick top N by mean |SHAP| ---
    top_feats <- shap_long |>
        group_by(variable) |>
        summarise(mean_abs = mean(abs(shap_value), na.rm = TRUE), .groups = "drop") |>
        arrange(desc(mean_abs)) |>
        slice_head(n = top_n)
    
    # --- pull feature values from original data ---
    value_long <- orig_full |>
        select(HHID, PN, all_of(top_feats$variable)) |>
        pivot_longer(
            cols = -c(HHID, PN),
            names_to = "variable",
            values_to = "feature_value"
        )
    
    value_long <- value_long |>
        mutate(
            feature_value_num = case_when(
                is.numeric(feature_value) ~ as.numeric(feature_value),
                is.logical(feature_value) ~ as.numeric(feature_value),
                TRUE ~ as.numeric(as.factor(feature_value)) - 1
            )
        ) |>
        group_by(variable) |>
        mutate(
            nuniq     = n_distinct(feature_value_num[!is.na(feature_value_num)]),
            is_binary = nuniq == 2,
            mean_v    = mean(feature_value_num, na.rm = TRUE),
            sd_v      = sd(feature_value_num,   na.rm = TRUE),
            
            # continuous: z-score, clamp to [-3, 3], rescale to [0,1]
            z_raw        = ifelse(sd_v > 0, (feature_value_num - mean_v) / sd_v, 0),
            z_clamp      = pmin(pmax(z_raw, -3), 3),
            cont_scaled  = (z_clamp + 3) / 6,
            
            feature_value_scaled = ifelse(is_binary, feature_value_num, cont_scaled)
        ) |>
        ungroup() |>
        select(HHID, PN, variable, feature_value, feature_value_scaled)
    
    # --- assemble plotting df ---
    plot_df <- shap_long |>
        semi_join(top_feats, by = "variable") |>
        left_join(value_long, by = c("HHID", "PN", "variable")) |>
        left_join(dict, by = c("variable" = "raw_var")) |>
        mutate(label = if_else(is.na(description), variable, description),
               label = str_wrap(label, 28)) |>
        group_by(label) |>
        mutate(mean_abs = mean(abs(shap_value), na.rm = TRUE)) |>
        ungroup() |>
        mutate(label = reorder(label, mean_abs))
    
    # --- beeswarm plot ---
    p <- ggplot(plot_df, aes(x = shap_value, y = label, colour = feature_value_scaled)) +
        geom_quasirandom(alpha = 0.55, size = 1.2, bandwidth = 0.2) +
        geom_vline(xintercept = 0, linetype = "dashed") +
        scale_color_viridis_c(
            option = "magma",
            direction = -1,
            begin = 0.15,
            end   = 0.90,
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
            plot.title   = element_text(hjust = 0.5, face = "bold", size = 18,
                                        margin = margin(b = 8)),
            axis.text.y  = element_text(size = 16, lineheight = 0.95),
            axis.text.x  = element_text(size = 14),
            axis.title.x = element_text(size = 16, face = "bold", margin = margin(t = 8)),
            panel.grid.minor = element_blank(),
            legend.title = element_text(size = 16, face = "bold"),
            legend.text  = element_text(size = 14)
        )
    
    return(p)
}

# ---------- build both plots ----------
p_grim <- make_beeswarm(shap_grim_path,  "GrimAge",   TOP_N)
p_dune <- make_beeswarm(shap_dune_path,  "DunedinPoAm", TOP_N)

# ---------- combine into panel (same legend, stacked) ----------
panel <- p_grim / p_dune +
    plot_layout(ncol = 1, guides = "collect") &
    theme(legend.position = "right")

# ---------- save panel ----------
ggsave(
    filename = out_png_panel,
    plot     = panel,
    width    = 12,
    height   = 18,
    dpi      = 300
)