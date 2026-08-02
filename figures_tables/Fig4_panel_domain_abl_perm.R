library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(patchwork)

group_order_raw <- c(
    "demographic",
    "education, job and socioeconomic status",
    "early-life conditions and family background",
    "adult adversity and life-course events",
    "social network and support",
    "lifestyle and health behaviors",
    "genetics"
)

label_map <- c(
    "demographic" = "Demographics",
    "education, job and socioeconomic status" = "Education, Job and Socioeconomic Status",
    "early-life conditions and family background" = "Early-Life Conditions and Family Background",
    "adult adversity and life-course events" = "Adult Adversity and Life-Course Events",
    "social network and support" = "Social Network and Support",
    "lifestyle and health behaviors" = "Health Behaviors and Clinical Risk Factors",
    "genetics" = "Polygenic Indices and Ancestry Principal Components"
)

.make_single_domain_plot <- function(
        base_dir,
        clocks_vec,
        type = c("domain", "ablation", "permutation"),
        panel_title = NULL
) {
    type <- match.arg(type)

    perf_path <- file.path(base_dir, "model_performance.csv")
    dom_path <- switch(
        type,
        "domain" = file.path(base_dir, "domainonly_performance.csv"),
        "ablation" = file.path(base_dir, "ablation_performance.csv"),
        "permutation" = file.path(base_dir, "permutation_performance_knn.csv")
    )

    perf <- read_csv(perf_path, show_col_types = FALSE)
    dom <- suppressWarnings(read_csv(dom_path, show_col_types = FALSE))

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
        filter(clock %in% clocks_vec, stratum == "ALL", model == "all_vars") %>%
        transmute(clock, base = mean_r2) %>%
        group_by(clock) %>%
        summarise(baseline = mean(base, na.rm = TRUE), .groups = "drop") %>%
        mutate(clock = factor(clock, levels = clocks_vec))

    stopifnot(nrow(baseline_tbl) == length(clocks_vec), !any(is.na(baseline_tbl$baseline)))

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

    if (nrow(dom_only) == 0) stop("No valid values for this analysis type.")

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

    if (is.null(panel_title)) {
        panel_title <- switch(
            type,
            "domain" = "Domain-only models",
            "ablation" = "Drop-one-domain models",
            "permutation" = "Domain permutation (KNN)"
        )
    }

    ggplot() +
        scale_x_discrete(limits = rev(group_order_raw), labels = label_map, drop = FALSE) +
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
            position = pd, width = 0.18, colour = "black", linewidth = 0.5,
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
        labs(x = NULL, y = "R²", title = panel_title) +
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

make_domain_panel_plot <- function(base_dir, clocks_vec, out_png) {
    p_dom <- .make_single_domain_plot(
        base_dir, clocks_vec, type = "domain",
        panel_title = "Domain-only models"
    )
    p_abl <- .make_single_domain_plot(
        base_dir, clocks_vec, type = "ablation",
        panel_title = "Drop-one-domain models"
    )
    p_perm <- .make_single_domain_plot(
        base_dir, clocks_vec, type = "permutation",
        panel_title = "Domain permutation (KNN)"
    )

    panel <- (p_dom / p_abl / p_perm) +
        plot_layout(ncol = 1, heights = c(1, 1, 1), guides = "collect") +
        plot_annotation(tag_levels = "A") &
        theme(
            legend.position = "right",
            plot.title = element_text(size = 18, face = "bold"),
            plot.tag = element_text(size = 22, face = "bold")
        )

    ggsave(out_png, panel, width = 11, height = 18, dpi = 600)
}

dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)

make_domain_panel_plot(
    base_dir = "output/analyses/XGB/Regressor",
    clocks_vec = c("EAA_GRIMAGE", "EAA_DUNEDINMPOA"),
    out_png = "output/figures/Fig4_domain_ablation_perm_XGBRegr.png"
)
