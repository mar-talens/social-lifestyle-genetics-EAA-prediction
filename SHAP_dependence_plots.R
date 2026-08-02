

# ===============================
# SHAP DEPENDENCE PLOTS (R) 
# VERTICAL PANEL
#   - GrimAge + DunedinPoAm (XGB Regressor)
#   - ONE PANEL: BMI, Income, PGI + Physical Activity
# ===============================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(rlang)
library(stringr)

# ---- paths (edit if needed) ----
orig_df_path   <- "/Users/martalens/Desktop/ML_EAA/output/dataframes/epigenetic_age_events.csv"
shap_grim_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_EAA_GRIMAGE.csv"
shap_dune_path <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_EAA_DUNEDINMPOA.csv"
out_png        <- "/Users/martalens/Desktop/ML_EAA/output/npj_aging_output/plots/shap_dependence_ALL.png"
dict_path      <- "/Users/martalens/Desktop/ML_EAA/data/metadata/variables_dictionary.csv"

# ---- FEATURES: all 6 in one go ----
features <- c(
    "HH_MEAN_INCOME",
    "PHYS_ACTIVITY_VIGOROUS",
    "PHYS_ACTIVITY_MODERATE",
    "PHYS_ACTIVITY_MILD", 
    "BMI",
    "PGI_SCZ_PGC14"
)

# Map each feature to the variable used for colouring
color_by <- c(
    BMI                    = "GENDER",
    HH_MEAN_INCOME         = "GENDER",
    PGI_SCZ_PGC14          = "GENDER",
    PHYS_ACTIVITY_VIGOROUS = "GENDER",
    PHYS_ACTIVITY_MODERATE = "GENDER",
    PHYS_ACTIVITY_MILD     = "GENDER"
)

# Feature-specific x-axis transforms


x_transform <- list(
    HH_MEAN_INCOME = function(x) log(x)   # log income (dropping zeros first)
)

# ===============================
# Labels from dictionary
# ===============================

dict <- readr::read_csv(dict_path, show_col_types = FALSE) |>
    dplyr::select(raw_var, description)

lab_map <- dict |>
    dplyr::mutate(
        description = ifelse(is.na(description) | description == "", raw_var, description)
    ) |>
    tibble::deframe()   # named chr vector: raw_var -> description

wrap_lab <- function(s, width = 22) stringr::str_wrap(s, width)

# ===============================
# Helpers
# ===============================

read_shap_long <- function(path) {
    df <- read_csv(path, show_col_types = FALSE)
    stopifnot(all(c("HHID","PN","variable","shap_value") %in% names(df)))
    df
}

make_dep_df <- function(shap_long, Xvals, features, color_vars = character()) {
    keep_cols <- unique(c("HHID","PN", features, color_vars))
    shap_long %>%
        filter(variable %in% features) %>%
        mutate(variable = as.character(variable)) %>%
        left_join(Xvals %>% select(any_of(keep_cols)), by = c("HHID","PN")) %>%
        pivot_longer(cols = all_of(features), names_to = "feature", values_to = "x") %>%
        filter(feature == variable) %>%
        select(HHID, PN, feature, x, shap_value, any_of(color_vars))
}

col_is_available <- function(df, colname) {
    !is.na(colname) && nzchar(colname) && (colname %in% names(df))
}

shap_ylabel <- function(clock) {
    if (grepl("DUNEDIN", toupper(clock))) {
        "SHAP (y/y, DunedinPoAm)"
    } else {
        "SHAP (years, GrimAge EAA)"
    }
}

get_palette <- function(color_col) {
    if (identical(color_col, "GENDER")) {
        list(values = c("Female" = "yellow3", "Male" = "darkblue"), name = "Gender")
    } #else if (identical(color_col, "EVER_SMOKED_RAND")) {
      #  list(values = c("Never" = "seagreen2", "Ever" = "brown"), name = "Smoking history")
    #} else if (identical(color_col, "WHITE_BINARY")) {
    #    list(values = c("Non-Hispanic White" = "coral2", "Other" = "darkblue"), name = "Ethnicity")
    #} else {
    #    list(values = NULL, name = color_col)
    #}
}

dep_plot <- function(df, feature, clock_label, color_col = NA, x_transform = NULL) {
    d <- df %>% dplyr::filter(feature == !!feature)
    if (identical(feature, "HH_MEAN_INCOME")) {
        
        # remove invalid
        d <- d %>% filter(!is.na(x), x > 0)
        
        # cap at 5th–95th percentile
        q <- quantile(d$x, probs = c(0.01, 0.99), na.rm = TRUE)
        d <- d %>% filter(x >= q[1], x <= q[2])
    }
    
    if (!is.null(x_transform) && is.function(x_transform)) {
        d <- d %>% dplyr::mutate(x = x_transform(x))
    }
    
    map_color <- col_is_available(d, color_col)
    pal <- if (map_color) get_palette(color_col) else list(values = NULL, name = NULL)
    
    xlab <- if ("feature_label" %in% names(d)) unique(d$feature_label)[1] else feature
    xlab <- wrap_lab(xlab)
    
    p <- ggplot(d, aes(x = x, y = shap_value)) +
        { if (map_color) geom_point(aes(color = .data[[color_col]]), alpha = 0.25, size = 1.1, show.legend = TRUE)
            else            geom_point(alpha = 0.25, size = 1.1, show.legend = FALSE) } +
        geom_smooth(method = "loess", se = TRUE, linewidth = 0.9, span = 0.9, color = "black") +
        { if (map_color) geom_smooth(aes(color = .data[[color_col]]), method = "loess",
                                     se = FALSE, linetype = "dashed", linewidth = 0.9, span = 0.9) } +
        labs(
            x = xlab,
            y = shap_ylabel(clock_label),
            color = pal$name,
            title = NULL  
        ) +
        theme_minimal(base_size = 12) +
        theme(
            panel.grid.minor = element_blank(),
            plot.title = element_text(face = "bold", size = 12),
            legend.position = if (map_color) "top" else "none"
        )
    
    if (map_color && is.factor(d[[color_col]]) && !is.null(pal$values)) {
        p <- p + scale_color_manual(values = pal$values, drop = FALSE, name = pal$name)
    }
    
    p
}

# ===============================
# Build data (keep all colour vars)
# ===============================

all_color_vars <- unique(na.omit(unname(color_by)))

Xvals <- read_csv(orig_df_path, show_col_types = FALSE) %>%
    select(any_of(c("HHID","PN", features, all_color_vars))) %>%
    distinct() %>%
    mutate(
        GENDER           = factor(GENDER, levels = c(0,1), labels = c("Female","Male")),
#        EVER_SMOKED_RAND = factor(EVER_SMOKED_RAND, levels = c(0,1), labels = c("Never","Ever")),
#        WHITE_BINARY     = factor(WHITE_BINARY, levels = c(0,1), labels = c("Other","Non-Hispanic White"))
    )

grim_shap <- read_shap_long(shap_grim_path)
dune_shap <- read_shap_long(shap_dune_path)

dep_grim <- make_dep_df(grim_shap, Xvals, features, color_vars = all_color_vars) |>
    dplyr::mutate(feature_label = dplyr::recode(feature, !!!lab_map, .default = feature))

dep_dune <- make_dep_df(dune_shap, Xvals, features, color_vars = all_color_vars) |>
    dplyr::mutate(feature_label = dplyr::recode(feature, !!!lab_map, .default = feature))

# ===============================
# Make plots (GrimAge + Dunedin per feature)
# ===============================
# ===============================
# COLUMN HEADERS (VERY IMPORTANT)
# ===============================
header <- ggplot() +
    annotate("text", x = 0.25, y = 0, label = "GrimAge",
             size = 5, fontface = "bold") +
    annotate("text", x = 0.75, y = 0, label = "DunedinPoAm",
             size = 5, fontface = "bold") +
    theme_void()

# ===============================
# MAKE CLEAN PLOTS (FINAL FIX)
# ===============================
# ===============================
# MAKE CLEAN PLOTS (FINAL FIX)
# ===============================

plots <- lapply(seq_along(features), function(i) {
    
    feat <- features[i]
    colvar <- color_by[[feat]]
    xfun   <- if (feat %in% names(x_transform)) x_transform[[feat]] else NULL
    
    feat_label <- wrap_lab(lab_map[[feat]] %||% feat)
    
    # LEFT: GrimAge (ONLY place with row label)
    p_g <- dep_plot(dep_grim, feat, "GrimAge", color_col = colvar, x_transform = xfun) +
        labs(
            title = paste0(LETTERS[i], "  ", feat_label),
            subtitle = "GrimAge"
        ) +
        theme(
            plot.title    = element_text(face = "bold", size = 12, hjust = 0),
            plot.subtitle = element_text(size = 10, hjust = 0)
        )
    
    # RIGHT: Dunedin (NO feature title, just small label)
    p_d <- dep_plot(dep_dune, feat, "DunedinPoAm", color_col = colvar, x_transform = xfun) +
        labs(
            subtitle = "DunedinPoAm"
        ) +
        theme(
            plot.title    = element_blank(),
            plot.subtitle = element_text(size = 10, hjust = 0)
        )
    
    # Combine row
    p_g + p_d
})

# ===============================
# FINAL LAYOUT
# ===============================
final_plot <- wrap_plots(plots, ncol = 1, guides = "collect") &
    theme(
        legend.position = "top",
        legend.box = "horizontal",
        legend.margin = margin(t = 5, b = 10),
        legend.box.margin = margin(b = 10)
    )
ggsave(
    out_png,
    final_plot,
    width  = 12,
    height = max(8, 2.6 * length(features)),  # slightly tighter
    dpi    = 300
)

















# ===============================
# SHAP DEPENDENCE PLOTS (R) 
# HORIZONTAL PANEL
#   - GrimAge + DunedinPoAm (XGB Regressor)
#   - ONE PANEL: BMI, Income, PGI + Physical Activity
# ===============================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(rlang)
library(stringr)

# ---- paths (edit if needed) ----
orig_df_path   <- "/Users/martalens/Desktop/ML_EAA/output/dataframes/epigenetic_age_events.csv"
shap_grim_path <- "/Users/martalens/Desktop/ML_EAA/output/oct_analyses/XGB/Regressor/shap_long_EAA_GRIMAGE.csv"
shap_dune_path <- "/Users/martalens/Desktop/ML_EAA/output/oct_analyses/XGB/Regressor/shap_long_EAA_DUNEDINMPOA.csv"
out_png        <- "/Users/martalens/Desktop/ML_EAA/output/oct_analyses/plots/shap_dependence_ALL_horizontal.png"
dict_path      <- "/Users/martalens/Desktop/ML_EAA/data/metadata/variables_dictionary.csv"

# ---- FEATURES: all 6 in one go ----
features <- c(
    "BMI",
    "HH_MEAN_INCOME",
    "PGI_SCZ_PGC14",
    "PHYS_ACTIVITY_VIGOROUS",
    "PHYS_ACTIVITY_MODERATE",
    "PHYS_ACTIVITY_MILD"
)

# Map each feature to the variable used for colouring
color_by <- c(
    BMI                    = "EVER_SMOKED_RAND",
    HH_MEAN_INCOME         = "WHITE_BINARY",
    PGI_SCZ_PGC14          = "GENDER",
    PHYS_ACTIVITY_VIGOROUS = "EVER_SMOKED_RAND",
    PHYS_ACTIVITY_MODERATE = "EVER_SMOKED_RAND",
    PHYS_ACTIVITY_MILD     = "EVER_SMOKED_RAND"
)

# Feature-specific x-axis transforms
x_transform <- list(
    HH_MEAN_INCOME = function(x) log(x)   # log income (dropping zeros first)
)

# ===============================
# Labels from dictionary
# ===============================

dict <- readr::read_csv(dict_path, show_col_types = FALSE) |>
    dplyr::select(raw_var, description)

lab_map <- dict |>
    dplyr::mutate(
        description = ifelse(is.na(description) | description == "", raw_var, description)
    ) |>
    tibble::deframe()   # named chr vector: raw_var -> description

wrap_lab <- function(s, width = 22) stringr::str_wrap(s, width)

# ===============================
# Helpers
# ===============================

read_shap_long <- function(path) {
    df <- read_csv(path, show_col_types = FALSE)
    stopifnot(all(c("HHID","PN","variable","shap_value") %in% names(df)))
    df
}

make_dep_df <- function(shap_long, Xvals, features, color_vars = character()) {
    keep_cols <- unique(c("HHID","PN", features, color_vars))
    shap_long %>%
        filter(variable %in% features) %>%
        mutate(variable = as.character(variable)) %>%
        left_join(Xvals %>% select(any_of(keep_cols)), by = c("HHID","PN")) %>%
        pivot_longer(cols = all_of(features), names_to = "feature", values_to = "x") %>%
        filter(feature == variable) %>%
        select(HHID, PN, feature, x, shap_value, any_of(color_vars))
}

col_is_available <- function(df, colname) {
    !is.na(colname) && nzchar(colname) && (colname %in% names(df))
}

shap_ylabel <- function(clock) {
    if (grepl("DUNEDIN", toupper(clock))) {
        "SHAP (y/y, DunedinPoAm)"
    } else {
        "SHAP (years, GrimAge EAA)"
    }
}

get_palette <- function(color_col) {
    if (identical(color_col, "GENDER")) {
        list(values = c("Female" = "tan2", "Male" = "darkmagenta"), name = "Gender")
    } else if (identical(color_col, "EVER_SMOKED_RAND")) {
        list(values = c("Never" = "seagreen2", "Ever" = "brown"), name = "Smoking history")
    } else if (identical(color_col, "WHITE_BINARY")) {
        list(values = c("Non-Hispanic White" = "coral2", "Other" = "darkblue"), name = "Ethnicity")
    } else {
        list(values = NULL, name = color_col)
    }
}

dep_plot <- function(df, feature, clock_label, color_col = NA, x_transform = NULL) {
    d <- df %>% dplyr::filter(feature == !!feature)
    
    # special handling for income (drop 0 / NA)
    if (identical(feature, "HH_MEAN_INCOME")) {
        d <- d %>% dplyr::filter(!is.na(x), x > 0)
    }
    if (!is.null(x_transform) && is.function(x_transform)) {
        d <- d %>% dplyr::mutate(x = x_transform(x))
    }
    
    map_color <- col_is_available(d, color_col)
    pal <- if (map_color) get_palette(color_col) else list(values = NULL, name = NULL)
    
    xlab <- if ("feature_label" %in% names(d)) unique(d$feature_label)[1] else feature
    xlab <- wrap_lab(xlab)
    
    p <- ggplot(d, aes(x = x, y = shap_value)) +
        { if (map_color) geom_point(aes(color = .data[[color_col]]), alpha = 0.25, size = 1.1, show.legend = TRUE)
            else            geom_point(alpha = 0.25, size = 1.1, show.legend = FALSE) } +
        geom_smooth(method = "loess", se = TRUE, linewidth = 0.9, span = 0.9, color = "black") +
        { if (map_color) geom_smooth(aes(color = .data[[color_col]]), method = "loess",
                                     se = FALSE, linetype = "dashed", linewidth = 0.9, span = 0.9) } +
        labs(
            x = xlab,
            y = shap_ylabel(clock_label),
            color = pal$name,
            title = paste0(xlab, " — ", clock_label)
        ) +
        theme_minimal(base_size = 12) +
        theme(
            panel.grid.minor = element_blank(),
            plot.title = element_text(face = "bold", size = 12),
            legend.position = if (map_color) "top" else "none"
        )
    
    if (map_color && is.factor(d[[color_col]]) && !is.null(pal$values)) {
        p <- p + scale_color_manual(values = pal$values, drop = FALSE, name = pal$name)
    }
    
    p
}

# ===============================
# Build data (keep all colour vars)
# ===============================

all_color_vars <- unique(na.omit(unname(color_by)))

Xvals <- read_csv(orig_df_path, show_col_types = FALSE) %>%
    select(any_of(c("HHID","PN", features, all_color_vars))) %>%
    distinct() %>%
    mutate(
        GENDER           = factor(GENDER, levels = c(0,1), labels = c("Female","Male")),
        EVER_SMOKED_RAND = factor(EVER_SMOKED_RAND, levels = c(0,1), labels = c("Never","Ever")),
        WHITE_BINARY     = factor(WHITE_BINARY, levels = c(0,1), labels = c("Other","Non-Hispanic White"))
    )

grim_shap <- read_shap_long(shap_grim_path)
dune_shap <- read_shap_long(shap_dune_path)

dep_grim <- make_dep_df(grim_shap, Xvals, features, color_vars = all_color_vars) |>
    dplyr::mutate(feature_label = dplyr::recode(feature, !!!lab_map, .default = feature))

dep_dune <- make_dep_df(dune_shap, Xvals, features, color_vars = all_color_vars) |>
    dplyr::mutate(feature_label = dplyr::recode(feature, !!!lab_map, .default = feature))

# ===============================
# Make plots (GrimAge + Dunedin per feature)
# ===============================

# ===============================
# Make plots (GrimAge above Dunedin per feature)
# ===============================

# current list of 6 plots
# ===============================
# Make plots (FIXED: clean + labeled)
# ===============================

plots <- lapply(seq_along(features), function(i) {
    
    feat <- features[i]
    colvar <- color_by[[feat]]
    xfun   <- if (feat %in% names(x_transform)) x_transform[[feat]] else NULL
    
    # Build panels (NO titles inside)
    p_g <- dep_plot(dep_grim, feat, "GrimAge",     color_col = colvar, x_transform = xfun) +
        theme(plot.title = element_blank())
    
    p_d <- dep_plot(dep_dune, feat, "DunedinPoAm", color_col = colvar, x_transform = xfun) +
        theme(plot.title = element_blank())
    
    # Feature label
    feat_label <- wrap_lab(lab_map[[feat]] %||% feat)
    
    # Combine + add LETTER + feature title
    (p_g + p_d) +
        plot_annotation(
            title = paste0(LETTERS[i], "  ", feat_label)
        ) &
        theme(
            plot.title = element_text(
                size = 12,
                face = "bold",
                hjust = 0
            )
        )
})

# Final stacked panel
final_plot <- patchwork::wrap_plots(plots, ncol = 1, guides = "collect") &
    theme(
        legend.position = "top"
    )

ggsave(
    out_png,
    final_plot,
    width  = 12,
    height = max(8, 3 * length(features)),
    dpi    = 300
)

# TOP 3 FEATURES
top_plot <- wrap_plots(plots[1:3], ncol = 3, guides = "collect") &
    theme(legend.position = "top")

ggsave(
    "/Users/martalens/Desktop/ML_EAA/output/npj_aging/plots/shap_dep_TOP3.png",
    top_plot, width = 14, height = 6, dpi = 300
)

# BOTTOM 3 FEATURES
bottom_plot <- wrap_plots(plots[4:6], ncol = 3, guides = "collect") &
    theme(legend.position = "top")

ggsave(
    "/Users/martalens/Desktop/ML_EAA/output/npj_agings/plots/shap_dep_BOTTOM3.png",
    bottom_plot, width = 14, height = 6, dpi = 300
)






