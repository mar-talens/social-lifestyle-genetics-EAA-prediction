library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(rlang)
library(stringr)

orig_df_path <- "output/epigenetic_age_events.csv"
shap_grim_path <- "output/analyses/XGB/Regressor/shap_long_EAA_GRIMAGE.csv"
shap_dune_path <- "output/analyses/XGB/Regressor/shap_long_EAA_DUNEDINMPOA.csv"
dict_path <- "variables_dictionary.csv"
out_png <- "output/figures/SuppFig3_SHAP_dependence.png"

dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)

features <- c(
    "HH_MEAN_INCOME",
    "PHYS_ACTIVITY_VIGOROUS",
    "PHYS_ACTIVITY_MODERATE",
    "PHYS_ACTIVITY_MILD",
    "BMI",
    "PGI_SCZ_PGC14"
)

color_by <- c(
    BMI = "GENDER",
    HH_MEAN_INCOME = "GENDER",
    PGI_SCZ_PGC14 = "GENDER",
    PHYS_ACTIVITY_VIGOROUS = "GENDER",
    PHYS_ACTIVITY_MODERATE = "GENDER",
    PHYS_ACTIVITY_MILD = "GENDER"
)

x_transform <- list(
    HH_MEAN_INCOME = function(x) log(x)
)

dict <- readr::read_csv(dict_path, show_col_types = FALSE) |>
    dplyr::select(raw_var, description)

lab_map <- dict |>
    dplyr::mutate(
        description = ifelse(
            is.na(description) | description == "",
            raw_var,
            description
        )
    ) |>
    tibble::deframe()

wrap_lab <- function(s, width = 22) stringr::str_wrap(s, width)

read_shap_long <- function(path) {
    df <- read_csv(path, show_col_types = FALSE)
    stopifnot(all(c("HHID", "PN", "variable", "shap_value") %in% names(df)))
    df
}

make_dep_df <- function(shap_long, Xvals, features, color_vars = character()) {
    keep_cols <- unique(c("HHID", "PN", features, color_vars))
    shap_long %>%
        filter(variable %in% features) %>%
        mutate(variable = as.character(variable)) %>%
        left_join(Xvals %>% select(any_of(keep_cols)), by = c("HHID", "PN")) %>%
        pivot_longer(
            cols = all_of(features),
            names_to = "feature",
            values_to = "x"
        ) %>%
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
        list(
            values = c("Female" = "yellow3", "Male" = "darkblue"),
            name = "Gender"
        )
    }
}

dep_plot <- function(df, feature, clock_label, color_col = NA, x_transform = NULL) {
    d <- df %>% dplyr::filter(feature == !!feature)

    if (identical(feature, "HH_MEAN_INCOME")) {
        d <- d %>% filter(!is.na(x), x > 0)

        q <- quantile(d$x, probs = c(0.01, 0.99), na.rm = TRUE)
        d <- d %>% filter(x >= q[1], x <= q[2])
    }

    if (!is.null(x_transform) && is.function(x_transform)) {
        d <- d %>% dplyr::mutate(x = x_transform(x))
    }

    map_color <- col_is_available(d, color_col)
    pal <- if (map_color) {
        get_palette(color_col)
    } else {
        list(values = NULL, name = NULL)
    }

    xlab <- if ("feature_label" %in% names(d)) unique(d$feature_label)[1] else feature
    xlab <- wrap_lab(xlab)

    p <- ggplot(d, aes(x = x, y = shap_value)) +
        {
            if (map_color) {
                geom_point(
                    aes(color = .data[[color_col]]),
                    alpha = 0.25,
                    size = 1.1,
                    show.legend = TRUE
                )
            } else {
                geom_point(alpha = 0.25, size = 1.1, show.legend = FALSE)
            }
        } +
        geom_smooth(
            method = "loess",
            se = TRUE,
            linewidth = 0.9,
            span = 0.9,
            color = "black"
        ) +
        {
            if (map_color) {
                geom_smooth(
                    aes(color = .data[[color_col]]),
                    method = "loess",
                    se = FALSE,
                    linetype = "dashed",
                    linewidth = 0.9,
                    span = 0.9
                )
            }
        } +
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
        p <- p + scale_color_manual(
            values = pal$values,
            drop = FALSE,
            name = pal$name
        )
    }

    p
}

all_color_vars <- unique(na.omit(unname(color_by)))

Xvals <- read_csv(orig_df_path, show_col_types = FALSE) %>%
    select(any_of(c("HHID", "PN", features, all_color_vars))) %>%
    distinct() %>%
    mutate(
        GENDER = factor(
            GENDER,
            levels = c(0, 1),
            labels = c("Female", "Male")
        )
    )

grim_shap <- read_shap_long(shap_grim_path)
dune_shap <- read_shap_long(shap_dune_path)

dep_grim <- make_dep_df(
    grim_shap,
    Xvals,
    features,
    color_vars = all_color_vars
) |>
    dplyr::mutate(
        feature_label = dplyr::recode(feature, !!!lab_map, .default = feature)
    )

dep_dune <- make_dep_df(
    dune_shap,
    Xvals,
    features,
    color_vars = all_color_vars
) |>
    dplyr::mutate(
        feature_label = dplyr::recode(feature, !!!lab_map, .default = feature)
    )

plots <- lapply(seq_along(features), function(i) {
    feat <- features[i]
    colvar <- color_by[[feat]]
    xfun <- if (feat %in% names(x_transform)) x_transform[[feat]] else NULL
    feat_label <- wrap_lab(lab_map[[feat]] %||% feat)

    p_g <- dep_plot(
        dep_grim,
        feat,
        "GrimAge",
        color_col = colvar,
        x_transform = xfun
    ) +
        labs(
            title = paste0(LETTERS[i], "  ", feat_label),
            subtitle = "GrimAge"
        ) +
        theme(
            plot.title = element_text(face = "bold", size = 12, hjust = 0),
            plot.subtitle = element_text(size = 10, hjust = 0)
        )

    p_d <- dep_plot(
        dep_dune,
        feat,
        "DunedinPoAm",
        color_col = colvar,
        x_transform = xfun
    ) +
        labs(subtitle = "DunedinPoAm") +
        theme(
            plot.title = element_blank(),
            plot.subtitle = element_text(size = 10, hjust = 0)
        )

    p_g + p_d
})

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
    width = 12,
    height = max(8, 2.6 * length(features)),
    dpi = 300
)
