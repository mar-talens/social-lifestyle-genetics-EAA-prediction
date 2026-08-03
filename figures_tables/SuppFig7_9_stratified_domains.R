library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(scales)
library(patchwork)

group_order_raw <- c(
    "demographic",
    "education, job and socioeconomic status",
    "social network and support",
    "adult adversity and life-course events",
    "early-life conditions and family background",
    "genetics",
    "lifestyle and health behaviors"
)

label_map <- c(
    "demographic" = "Demographics",
    "education, job and socioeconomic status" = "Education, Job and Socioeconomic Status",
    "social network and support" = "Social Network and Support",
    "adult adversity and life-course events" = "Adult Adversity and Life-Course Events",
    "early-life conditions and family background" = "Early-Life Conditions and Family Background",
    "genetics" = "Polygenic Indices and Ancestry Principal Components",
    "lifestyle and health behaviors" = "Health Behaviors and Clinical Risk Factors"
)

BASE_DIR_XGB_REGR <- "output/analyses/XGB/Regressor"
CLOCKS_REGR <- c("EAA_GRIMAGE", "EAA_DUNEDINMPOA")

.make_single_domain_stratum_plot <- function(
        base_dir,
        clocks_vec,
        stratum_name,
        type = c("domain", "ablation", "permutation"),
        plot_title = NULL
) {
    type <- match.arg(type)

    perf_path <- file.path(base_dir, "model_performance.csv")
    dom_path <- switch(
        type,
        "domain" = file.path(base_dir, "domainonly_performance_stratified.csv"),
        "ablation" = file.path(base_dir, "ablation_performance_stratified.csv"),
        "permutation" = file.path(base_dir, "permutation_performance_knn_stratified.csv")
    )

    perf <- read_csv(perf_path, show_col_types = FALSE) %>%
        filter(stratum == stratum_name)

    dom <- tryCatch({
        suppressWarnings(read_csv(dom_path, show_col_types = FALSE)) %>%
            filter(stratum == stratum_name)
    }, error = function(e) {
        warning(paste("Missing data file for", type, "analysis at:", dom_path))
        return(data.frame())
    })

    if (type == "domain") {
        dom_mean_col <- "mean_r2"
        dom_sd_col <- "sd_r2"
    } else if (type == "ablation") {
        dom_mean_col <- "ablation_mean_R2"
        dom_sd_col <- "ablation_sd_R2"
    } else {
        dom_mean_col <- "permutation_mean"
        dom_sd_col <- "permutation_sd"
    }

    baseline_tbl <- perf %>%
        filter(clock %in% clocks_vec, model == "all_vars") %>%
        transmute(clock, base = mean_r2) %>%
        group_by(clock) %>%
        summarise(baseline = mean(base, na.rm = TRUE), .groups = "drop") %>%
        mutate(clock = factor(clock, levels = clocks_vec))

    if (nrow(baseline_tbl) != length(clocks_vec) || any(is.na(baseline_tbl$baseline))) {
        warning(paste("Baseline data missing for stratum:", stratum_name))
        return(
            ggplot() +
                labs(
                    title = paste(plot_title, " (Missing Baseline)"),
                    subtitle = stratum_name
                ) +
                theme_void()
        )
    }

    if (nrow(dom) == 0) {
        return(
            ggplot() +
                labs(
                    title = paste(plot_title, " (Missing Domain Data)"),
                    subtitle = stratum_name
                ) +
                theme_void()
        )
    }

    dom_only <- dom %>%
        filter(clock %in% clocks_vec) %>%
        mutate(group_l = tolower(group)) %>%
        filter(group_l %in% group_order_raw) %>%
        transmute(
            group = factor(group_l, levels = group_order_raw),
            clock = factor(clock, levels = clocks_vec),
            value = .data[[dom_mean_col]],
            sd = dplyr::coalesce(.data[[dom_sd_col]], 0)
        ) %>%
        filter(!is.na(value))

    if (nrow(dom_only) == 0) {
        warning(paste("No valid domain values for stratum:", stratum_name, "type:", type))
        return(
            ggplot() +
                labs(
                    title = paste(plot_title, " (No Valid Data)"),
                    subtitle = stratum_name
                ) +
                theme_void()
        )
    }

    baseline_df <- tidyr::expand_grid(
        group = factor(group_order_raw, levels = group_order_raw),
        clock = factor(clocks_vec, levels = clocks_vec)
    ) %>%
        left_join(baseline_tbl, by = "clock")

    y_max <- max(baseline_df$baseline, dom_only$value + dom_only$sd, na.rm = TRUE) * 1.05
    top <- ceiling(y_max * 10) / 10
    brks <- seq(0, top, by = 0.1)

    label_map_clocks <- setNames(c("GrimAge", "DunedinPoAm"), clocks_vec)
    pal <- setNames(c("#BF7C3F", "#3B74A8"), clocks_vec)
    pd <- position_dodge(width = 0.72)

    if (is.null(plot_title)) {
        plot_title <- switch(
            type,
            "domain" = "Domain-only models",
            "ablation" = "Drop-one-domain models",
            "permutation" = "Domain permutation (KNN)"
        )
    }

    ggplot() +
        scale_x_discrete(labels = label_map, drop = FALSE) +
        geom_hline(yintercept = brks, linewidth = 0.05, color = "grey85") +
        geom_col(
            data = baseline_df,
            aes(x = group, y = baseline, group = clock),
            fill = "grey90", width = 0.8, position = pd
        ) +
        geom_col(
            data = dom_only,
            aes(x = group, y = value, fill = clock),
            width = 0.26, position = pd, show.legend = FALSE
        ) +
        geom_point(
            data = dom_only,
            aes(x = group, y = value, fill = clock),
            shape = 21, size = 4, stroke = 0.2, colour = "grey70",
            position = pd
        ) +
        geom_errorbar(
            data = dom_only,
            aes(
                x = group,
                ymin = pmax(0, value - sd),
                ymax = value + sd,
                group = clock
            ),
            position = pd,
            width = 0.18,
            colour = "black",
            linewidth = 0.5,
            show.legend = FALSE
        ) +
        scale_fill_manual(values = pal, name = "Clock", labels = label_map_clocks) +
        coord_flip(clip = "on") +
        scale_y_continuous(
            limits = c(0, y_max),
            breaks = brks,
            labels = label_number(accuracy = 0.01),
            expand = expansion(mult = c(0, 0.02))
        ) +
        labs(x = NULL, y = expression(R^2), title = plot_title) +
        theme_classic(base_size = 16) +
        theme(
            legend.position = "right",
            axis.line = element_line(color = "grey80"),
            axis.ticks = element_line(color = "grey80"),
            axis.text = element_text(color = "black"),
            axis.title = element_text(color = "black"),
            axis.text.y = element_text(hjust = 1, size = 14),
            axis.text.x = element_text(size = 14),
            axis.title.y = element_blank(),
            axis.title.x = element_text(size = 16, face = "bold"),
            plot.margin = margin(t = 6, r = 16, b = 6, l = 22),
            plot.title = element_text(face = "bold", size = 18, hjust = 0),
            panel.grid.major.x = element_line(color = "grey90"),
            panel.grid.major.y = element_blank(),
            panel.grid.minor = element_blank()
        )
}

make_domain_panel_plot_stratified <- function(
        base_dir,
        clocks_vec,
        stratum_vec,
        stratum_labels,
        analysis_type_label,
        out_png
) {
    p_dom1 <- .make_single_domain_stratum_plot(
        base_dir, clocks_vec, stratum_vec[1], type = "domain",
        plot_title = "Domain-only models"
    )
    p_abl1 <- .make_single_domain_stratum_plot(
        base_dir, clocks_vec, stratum_vec[1], type = "ablation",
        plot_title = "Drop-one-domain models"
    )
    p_perm1 <- .make_single_domain_stratum_plot(
        base_dir, clocks_vec, stratum_vec[1], type = "permutation",
        plot_title = "Domain permutation (KNN)"
    )

    p_dom2 <- .make_single_domain_stratum_plot(
        base_dir, clocks_vec, stratum_vec[2], type = "domain",
        plot_title = "Domain-only models"
    )
    p_abl2 <- .make_single_domain_stratum_plot(
        base_dir, clocks_vec, stratum_vec[2], type = "ablation",
        plot_title = "Drop-one-domain models"
    )
    p_perm2 <- .make_single_domain_stratum_plot(
        base_dir, clocks_vec, stratum_vec[2], type = "permutation",
        plot_title = "Domain permutation (KNN)"
    )

    p_dom1 <- p_dom1 + theme(
        plot.margin = margin(t = 6, r = 1, b = 6, l = 62),
        axis.title.x = element_blank()
    )
    p_abl1 <- p_abl1 + theme(
        plot.margin = margin(t = 6, r = 1, b = 6, l = 62),
        axis.title.x = element_blank()
    )
    p_perm1 <- p_perm1 + theme(
        plot.margin = margin(t = 6, r = 1, b = 6, l = 62)
    )

    theme_right_col_base <- theme(
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.y = element_blank(),
        plot.margin = margin(t = 6, r = 20, b = 6, l = 20)
    )
    theme_right_col_title <- theme(
        plot.title = element_text(size = 0, colour = "white"),
        axis.title.x = element_blank()
    )
    theme_right_col_title_last <- theme(
        plot.title = element_text(size = 0, colour = "white")
    )

    p_dom2 <- p_dom2 + theme_right_col_base + theme_right_col_title
    p_abl2 <- p_abl2 + theme_right_col_base + theme_right_col_title
    p_perm2 <- p_perm2 + theme_right_col_base + theme_right_col_title_last

    title1 <- ggplot() +
        labs(title = stratum_labels[1]) +
        theme_void() +
        theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))
    title2 <- ggplot() +
        labs(title = stratum_labels[2]) +
        theme_void() +
        theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))

    row_titles <- title1 | title2
    row_dom <- p_dom1 | p_dom2
    row_abl <- p_abl1 | p_abl2
    row_perm <- p_perm1 | p_perm2

    panel <- row_titles /
        row_dom /
        row_abl /
        row_perm +
        plot_layout(
            guides = "collect",
            heights = unit(c(0.05, 1, 1, 1), "null")
        ) &
        theme(legend.position = "bottom") &
        plot_annotation(
            title = paste(
                "XGB Regressor Performance by Stratum:",
                analysis_type_label,
                " (R²)"
            ),
            theme = theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5))
        )

    ggsave(out_png, panel, width = 16, height = 18, dpi = 300)
    panel
}

dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)

make_domain_panel_plot_stratified(
    base_dir = BASE_DIR_XGB_REGR,
    clocks_vec = CLOCKS_REGR,
    stratum_vec = c("GENDER_0.0", "GENDER_1.0"),
    stratum_labels = c("Females", "Males"),
    analysis_type_label = "Sex",
    out_png = "output/figures/SuppFig7_stratified_gender_domains.png"
)

make_domain_panel_plot_stratified(
    base_dir = BASE_DIR_XGB_REGR,
    clocks_vec = CLOCKS_REGR,
    stratum_vec = c("WHITE_BINARY_1", "WHITE_BINARY_0"),
    stratum_labels = c("Non-Hispanic White", "Other ethnicities"),
    analysis_type_label = "Ethnicity",
    out_png = "output/figures/SuppFig8_stratified_ethnicity_domains.png"
)

make_domain_panel_plot_stratified(
    base_dir = BASE_DIR_XGB_REGR,
    clocks_vec = CLOCKS_REGR,
    stratum_vec = c("EVER_SMOKED_RAND_1.0", "EVER_SMOKED_RAND_0.0"),
    stratum_labels = c("Ever smoker", "Never smoked"),
    analysis_type_label = "Smoking History",
    out_png = "output/figures/SuppFig9_stratified_smoking_domains.png"
)
