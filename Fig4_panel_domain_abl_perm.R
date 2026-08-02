# ================== LIBRERÍAS ==================
library(readr)
library(dplyr)
library(forcats)
library(ggplot2)
library(tidyr)
library(stringr)
library(scales)
library(patchwork)

# ================== ORDEN FIJO DE SUBGRUPOS (sin Smoking) ==================
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

# ================== FUNCIÓN GENÉRICA PARA UN TIPO (domain / ablation / permutation) ==================
.make_single_domain_plot <- function(
        base_dir,
        clocks_vec,
        METRIC = c("r2", "auc"),
        type = c("domain", "ablation", "permutation"),
        panel_title = NULL
) {
    METRIC <- match.arg(METRIC)
    type   <- match.arg(type)
    
    perf_path <- file.path(base_dir, "model_performance.csv")
    
    dom_path <- switch(
        type,
        "domain"      = file.path(base_dir, "domainonly_performance.csv"),
        "ablation"    = file.path(base_dir, "ablation_performance.csv"),
        "permutation" = file.path(base_dir, "permutation_performance_knn.csv")
    )
    
    perf <- read_csv(perf_path, show_col_types = FALSE)
    dom  <- suppressWarnings(read_csv(dom_path,  show_col_types = FALSE))
    
    # columnas según tipo
    if (type == "domain") {
        dom_mean_col <- if (METRIC == "auc") "mean_auc"        else "mean_r2"
        dom_sd_col   <- if (METRIC == "auc") "sd_auc"          else "sd_r2"
    } else if (type == "ablation") {
        dom_mean_col <- if (METRIC == "auc") "ablation_mean_AUC" else "ablation_mean_R2"
        dom_sd_col   <- if (METRIC == "auc") "ablation_sd_AUC"   else "ablation_sd_R2"
    } else { # permutation
        dom_mean_col <- "permutation_mean"
        dom_sd_col   <- "permutation_sd"
    }
    
    metric_label <- if (METRIC == "auc") "AUC" else "R²"
    
    # ===== Baseline (modelo all_vars) por clock =====
    baseline_tbl <- perf %>%
        filter(clock %in% clocks_vec, stratum == "ALL", model == "all_vars") %>%
        transmute(clock, base = if (METRIC == "auc") mean_auc else mean_r2) %>%
        group_by(clock) %>%
        summarise(baseline = mean(base, na.rm = TRUE), .groups = "drop") %>%
        mutate(clock = factor(clock, levels = clocks_vec))
    
    stopifnot(nrow(baseline_tbl) == length(clocks_vec), !any(is.na(baseline_tbl$baseline)))
    
    # ===== Tabla dominio =====
    dom_only <- dom %>%
        filter(clock %in% clocks_vec) %>%
        mutate(group_l = tolower(group)) %>%
        filter(group_l %in% group_order_raw) %>%
        transmute(
            group = factor(group_l, levels = group_order_raw),
            clock = factor(clock, levels = clocks_vec),
            value = .data[[dom_mean_col]],
            sd    = dplyr::coalesce(.data[[dom_sd_col]], 0)
        ) %>%
        filter(!is.na(value))
    
    if (nrow(dom_only) == 0) stop("No hay valores válidos para este tipo de análisis.")
    
    # duplicar baseline a (group, clock)
    baseline_df <- tidyr::expand_grid(
        group = factor(group_order_raw, levels = group_order_raw),
        clock = factor(clocks_vec, levels = clocks_vec)
    ) %>%
        left_join(baseline_tbl, by = "clock")
    
    # rango eje Y (numérico, antes del flip)
    y_max <- max(baseline_df$baseline, dom_only$value + dom_only$sd, na.rm = TRUE) * 1.05
    top   <- ceiling(y_max * 10) / 10
    brks  <- seq(0, top, by = 0.1)
    
    label_map_clocks <- setNames(c("GrimAge", "DunedinPoAm"), clocks_vec)
    pal <- setNames(c("#BF7C3F", "#3B74A8"), clocks_vec)
    pd  <- position_dodge(width = 0.72)
    
    # título por defecto del panel
    if (is.null(panel_title)) {
        panel_title <- switch(
            type,
            "domain"      = "Domain-only models",
            "ablation"    = "Drop-one-domain models",
            "permutation" = "Domain permutation (KNN)"
        )
    }
    
    p <- ggplot() +
        scale_x_discrete(limits = rev(group_order_raw), labels = label_map, drop = FALSE) +
        geom_hline(yintercept = brks, linewidth = 0.05, color = "grey85") +
        
        # barra base gris (full model)
        geom_col(
            data = baseline_df,
            aes(x = group, y = baseline, group = clock),
            fill = "grey90", width = 0.8, position = pd
        ) +
        
        # barra fina coloreada (dominio)
        geom_col(
            data = dom_only,
            aes(x = group, y = value, fill = clock),
            width = 0.26, position = pd, show.legend = FALSE
        ) +
        
        # punto en el extremo de la barra fina
        geom_point(
            data = dom_only,
            aes(x = group, y = value, fill = clock),
            shape = 21, size = 4, stroke = 0.2, colour = "grey70",
            position = pd
        ) +
        
        # barras de error
        geom_errorbar(
            data = dom_only,
            aes(x = group,
                ymin = pmax(0, value - sd),
                ymax = value + sd,
                group = clock),
            position = pd, width = 0.18, colour = "black", linewidth = 0.5, show.legend = FALSE
        ) +
        
        scale_fill_manual(values = pal, name = "Clock", labels = label_map_clocks) +
        coord_flip(clip = "on") +
        scale_y_continuous(
            limits = c(0, y_max),
            breaks = brks,
            labels = label_number(accuracy = 0.01),
            expand = expansion(mult = c(0, 0.02))
        ) +
        labs(x = NULL, y = metric_label, title = panel_title) +
        theme_classic(base_size = 16) +        # <<< fuente más grande
        theme(
            legend.position = "right",
            axis.line = element_line(color = "grey80"),
            axis.ticks = element_line(color = "grey80"),
            axis.text = element_text(color = "black"),
            axis.title = element_text(color = "black"),
            axis.text.y  = element_text(hjust = 1, size = 14),
            axis.text.x  = element_text(size = 14),
            axis.title.y = element_blank(),
            axis.title.x = element_text(size = 16, face = "bold"),
            plot.margin  = margin(t = 6, r = 16, b = 6, l = 22),
            plot.title   = element_text(face = "bold", size = 18, hjust = 0),
            panel.grid.major.x = element_line(color = "grey90"),
            panel.grid.major.y = element_blank(),
            panel.grid.minor   = element_blank()
        )
    
    return(p)
}

# ================== FUNCIÓN PANEL: UNE DOMAIN + ABLATION + PERMUTATION ==================
make_domain_panel_plot <- function(
        base_dir,
        clocks_vec,
        METRIC = c("r2","auc"),
        out_png
) {
    METRIC <- match.arg(METRIC)
    
    p_dom  <- .make_single_domain_plot(base_dir, clocks_vec, METRIC, type = "domain",
                                       panel_title = "Domain-only models")
    p_abl  <- .make_single_domain_plot(base_dir, clocks_vec, METRIC, type = "ablation",
                                       panel_title = "Drop-one-domain models")
    p_perm <- .make_single_domain_plot(base_dir, clocks_vec, METRIC, type = "permutation",
                                       panel_title = "Domain permutation (KNN)")
    
    panel <- (p_dom / p_abl / p_perm) +
        plot_layout(
            ncol = 1,
            heights = c(1,1,1),
            guides = "collect"   # <- THIS collects legends
        ) +
        plot_annotation(
            tag_levels = "A"
        ) &
        theme(
            legend.position = "right",
            plot.title = element_text(size = 18, face = "bold"),
            plot.tag = element_text(size = 22, face = "bold")
        )
    
    ggsave(out_png, panel, width = 11, height = 18, dpi = 600)
    print(panel)
    message("Panel guardado en: ", out_png)
}

# ================== LLAMADAS (4 casos) ==================

# RF — Classifier (AUC)
make_domain_panel_plot(
    base_dir   = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/RF/Classifier",
    clocks_vec = c("EAA_GRIMAGE_BINARY", "EAA_DUNEDINMPOA_BINARY"),
    METRIC     = "auc",
    out_png    = "/Users/martalens/Desktop/ML_EAA/output/npj_aging_output/plots/supplementary/domain_ablation_perm_RFBinary.png"
)

# RF — Regressor (R²)
make_domain_panel_plot(
    base_dir   = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/RF/Regressor",
    clocks_vec = c("EAA_GRIMAGE", "EAA_DUNEDINMPOA"),
    METRIC     = "r2",
    out_png    = "/Users/martalens/Desktop/ML_EAA/output/npj_aging_output/plots/supplementary/domain_ablation_perm_RFRegr.png"
)

# XGB — Classifier (AUC)
make_domain_panel_plot(
    base_dir   = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Classifier",
    clocks_vec = c("EAA_GRIMAGE_BINARY", "EAA_DUNEDINMPOA_BINARY"),
    METRIC     = "auc",
    out_png    = "/Users/martalens/Desktop/ML_EAA/output/npj_aging_output/plots/supplementary/domain_ablation_perm_XGBBin.png"
)

# XGB — Regressor (R²)
make_domain_panel_plot(
    base_dir   = "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor",
    clocks_vec = c("EAA_GRIMAGE", "EAA_DUNEDINMPOA"),
    METRIC     = "r2",
    out_png    = "/Users/martalens/Desktop/ML_EAA/output/npj_aging_output/plots/Fig4_domain_ablation_perm_XGBRegr.png"
)