# ==========================================
# XGB — Domain Performance by Strata (2-col x 3-row panels)
# ==========================================

library(readr)
library(dplyr)
library(forcats)
library(ggplot2)
library(tidyr)
library(stringr)
library(scales)
library(patchwork)
library(units) 

# ================== ORDEN FIJO DE SUBGRUPOS ==================
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
    "demographic"                               = "Demographics",
    "education, job and socioeconomic status"   = "Education, Job and Socioeconomic Status",
    "social network and support"                = "Social Network and Support",
    "adult adversity and life-course events"    = "Adult Adversity and Life-Course Events",
    "early-life conditions and family background" = "Early-Life Conditions and Family Background",
    "genetics"                                  = "Polygenic Indices and Ancestry Principal Components",
    "lifestyle and health behaviors"            = "Health Behaviors and Clinical Risk Factors"
)

# DIRECTORIO BASE
BASE_DIR_XGB_REGR <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor"

# ================== FUNCIÓN GENÉRICA PARA UN TIPO Y UN ESTRATO ==================
.make_single_domain_stratum_plot <- function(
        base_dir,
        clocks_vec,
        stratum_name,
        METRIC = c("r2", "auc"),
        type = c("domain", "ablation", "permutation"),
        plot_title = NULL,
        base_perf_file = "model_performance.csv"
) {
    METRIC <- match.arg(METRIC)
    type   <- match.arg(type)
    
    # Rutas de archivos
    perf_path <- file.path(base_dir, base_perf_file)
    dom_path <- switch(
        type,
        "domain"      = file.path(base_dir, "domainonly_performance_stratified.csv"),
        "ablation"    = file.path(base_dir, "ablation_performance_stratified.csv"),
        "permutation" = file.path(base_dir, "permutation_performance_knn_stratified.csv")
    )
    
    # Lectura de datos
    perf <- read_csv(perf_path, show_col_types = FALSE) %>%
        filter(stratum == stratum_name)
    
    dom <- tryCatch({
        suppressWarnings(read_csv(dom_path, show_col_types = FALSE)) %>%
            filter(stratum == stratum_name)
    }, error = function(e) {
        warning(paste("Missing data file for", type, "analysis at:", dom_path))
        return(data.frame())
    })
    
    # Nombres de columnas según el tipo de análisis
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
    
    metric_label <- if (METRIC == "auc") "AUC" else expression(R^2)
    
    # ===== 1. Línea Base (full model) por clock DENTRO DEL ESTRATO =====
    baseline_tbl <- perf %>%
        filter(clock %in% clocks_vec, model == "all_vars") %>%
        transmute(clock, base = if (METRIC == "auc") mean_auc else mean_r2) %>%
        group_by(clock) %>%
        summarise(baseline = mean(base, na.rm = TRUE), .groups = "drop") %>%
        mutate(clock = factor(clock, levels = clocks_vec))
    
    if (nrow(baseline_tbl) != length(clocks_vec) || any(is.na(baseline_tbl$baseline))) {
        warning(paste("Baseline data missing for stratum:", stratum_name))
        return(ggplot() + labs(title = paste(plot_title, " (Missing Baseline)"), subtitle = stratum_name) + theme_void())
    }
    
    # ===== 2. Tabla de Dominio (Data del panel principal) DENTRO DEL ESTRATO =====
    if (nrow(dom) == 0) {
        return(ggplot() + labs(title = paste(plot_title, " (Missing Domain Data)"), subtitle = stratum_name) + theme_void())
    }
    
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
    
    if (nrow(dom_only) == 0) {
        warning(paste("No valid domain values for stratum:", stratum_name, "type:", type))
        return(ggplot() + labs(title = paste(plot_title, " (No Valid Data)"), subtitle = stratum_name) + theme_void())
    }
    
    # duplicar baseline
    baseline_df <- tidyr::expand_grid(
        group = factor(group_order_raw, levels = group_order_raw),
        clock = factor(clocks_vec, levels = clocks_vec)
    ) %>%
        left_join(baseline_tbl, by = "clock")
    
    # Rango eje Y
    y_max <- max(baseline_df$baseline, dom_only$value + dom_only$sd, na.rm = TRUE) * 1.05
    top   <- ceiling(y_max * 10) / 10
    brks  <- seq(0, top, by = 0.1)
    
    label_map_clocks <- setNames(c("GrimAge", "DunedinPoAm"), clocks_vec)
    pal <- setNames(c("#BF7C3F", "#3B74A8"), clocks_vec)
    pd  <- position_dodge(width = 0.72)
    
    # Título por defecto del subplot
    if (is.null(plot_title)) {
        plot_title <- switch(
            type,
            "domain"      = "Domain-only models",
            "ablation"    = "Drop-one-domain models",
            "permutation" = "Domain permutation (KNN)"
        )
    }
    
    p <- ggplot() +
        scale_x_discrete(labels = label_map, drop = FALSE) +
        geom_hline(yintercept = brks, linewidth = 0.05, color = "grey85") +
        
        # Barra base gris (full model)
        geom_col(
            data = baseline_df,
            aes(x = group, y = baseline, group = clock),
            fill = "grey90", width = 0.8, position = pd
        ) +
        
        # Barra fina coloreada (dominio)
        geom_col(
            data = dom_only,
            aes(x = group, y = value, fill = clock),
            width = 0.26, position = pd, show.legend = FALSE
        ) +
        
        # Punto en el extremo de la barra fina
        geom_point(
            data = dom_only,
            aes(x = group, y = value, fill = clock),
            shape = 21, size = 4, stroke = 0.2, colour = "grey70",
            position = pd
        ) +
        
        # Barras de error
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
        labs(x = NULL, y = metric_label, title = plot_title) +
        theme_classic(base_size = 16) +
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


# ================== FUNCIÓN PANEL ESTRATIFICADO: UNE ESTRATOS Y TIPOS (2 Columnas x 3 Filas) ==================
make_domain_panel_plot_stratified <- function(
        base_dir,
        clocks_vec,
        stratum_vec,
        stratum_labels,
        METRIC = c("r2","auc"),
        analysis_type_label = "Stratified Analysis",
        out_png
) {
    METRIC <- match.arg(METRIC)
    
    # 1. GENERAR SUB-GRÁFICOS
    p_dom1 <- .make_single_domain_stratum_plot(base_dir, clocks_vec, stratum_vec[1], METRIC, type = "domain", plot_title = "Domain-only models")
    p_abl1 <- .make_single_domain_stratum_plot(base_dir, clocks_vec, stratum_vec[1], METRIC, type = "ablation", plot_title = "Drop-one-domain models")
    p_perm1 <- .make_single_domain_stratum_plot(base_dir, clocks_vec, stratum_vec[1], METRIC, type = "permutation", plot_title = "Domain permutation (KNN)")
    
    p_dom2 <- .make_single_domain_stratum_plot(base_dir, clocks_vec, stratum_vec[2], METRIC, type = "domain", plot_title = "Domain-only models")
    p_abl2 <- .make_single_domain_stratum_plot(base_dir, clocks_vec, stratum_vec[2], METRIC, type = "ablation", plot_title = "Drop-one-domain models")
    p_perm2 <- .make_single_domain_stratum_plot(base_dir, clocks_vec, stratum_vec[2], METRIC, type = "permutation", plot_title = "Domain permutation (KNN)")
    
    # 2. AJUSTAR TEMAS PARA EL DISEÑO DEL PANEL (ALINEACIÓN)
    
    # Columna Izquierda: Ajuste de márgenes y título X en el último.
    p_dom1  <- p_dom1  + theme(plot.margin = margin(t = 6, r = 1, b = 6, l = 62), axis.title.x = element_blank())
    p_abl1  <- p_abl1  + theme(plot.margin = margin(t = 6, r = 1, b = 6, l = 62), axis.title.x = element_blank())
    p_perm1 <- p_perm1 + theme(plot.margin = margin(t = 6, r = 1, b = 6, l = 62)) 
    
    # Columna Derecha: Ocultar texto del título manteniendo el espacio. Ocultar texto del eje Y.
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
    
    p_dom2  <- p_dom2  + theme_right_col_base + theme_right_col_title
    p_abl2  <- p_abl2  + theme_right_col_base + theme_right_col_title
    p_perm2 <- p_perm2 + theme_right_col_base + theme_right_col_title_last
    
    
    # 3. COMBINAR CON PATCHWORK (Usando la estructura correcta)
    
    # Títulos de columna (primera fila del layout)
    title1 <- ggplot() + labs(title = stratum_labels[1]) + theme_void() + 
        theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))
    title2 <- ggplot() + labs(title = stratum_labels[2]) + theme_void() + 
        theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))
    
    # Diseño de Filas
    row_titles <- title1 | title2
    row_dom <- p_dom1 | p_dom2
    row_abl <- p_abl1 | p_abl2
    row_perm <- p_perm1 | p_perm2
    
    # Combinación final (4 filas)
    panel <- row_titles /
        row_dom /
        row_abl /
        row_perm +
        plot_layout(
            guides = "collect", 
            heights = unit(c(0.05, 1, 1, 1), 'null') # Fila de títulos pequeña
        ) &
        theme(legend.position = "bottom") &
        plot_annotation(
            title = paste("XGB Regressor Performance by Stratum:", analysis_type_label, " (", if (METRIC == "auc") "AUC" else "R²", ")"),
            theme = theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5))
        )
    
    # 4. PASO CRÍTICO: CREAR EL DIRECTORIO ANTES DE GUARDAR
    out_dir <- dirname(out_png)
    if (!dir.exists(out_dir)) {
        dir.create(out_dir, recursive = TRUE)
        message("Directorio creado: ", out_dir)
    }
    
    ggsave(out_png, panel, width = 16, height = 18, dpi = 300)
    print(panel)
    message("Panel guardado en: ", out_png)
    
    return(panel)
}

# ================== LLAMADAS PARA GENERAR LOS 3 PANELES ==================

CLOCKS_REGR = c("EAA_GRIMAGE", "EAA_DUNEDINMPOA")
METRIC_REGR = "r2"

# 1) SEX
panel_gender <- make_domain_panel_plot_stratified(
    base_dir      = BASE_DIR_XGB_REGR,
    clocks_vec    = CLOCKS_REGR,
    stratum_vec   = c("GENDER_0.0", "GENDER_1.0"),
    stratum_labels = c("Females", "Males"),
    METRIC        = METRIC_REGR,
    analysis_type_label = "Sex",
    out_png       = "~/Desktop/ML_EAA/Paper_graphs/domain_ablation_perm_XGBRegr_bySEX.png"
)

# 2) ETHNICITY
panel_ethnicity <- make_domain_panel_plot_stratified(
    base_dir      = BASE_DIR_XGB_REGR,
    clocks_vec    = CLOCKS_REGR,
    stratum_vec   = c("WHITE_BINARY_1", "WHITE_BINARY_0"),
    stratum_labels = c("Non-Hispanic White", "Other ethnicities"),
    METRIC        = METRIC_REGR,
    analysis_type_label = "Ethnicity",
    out_png       = "~/Desktop/ML_EAA/Paper_graphs/domain_ablation_perm_XGBRegr_byETHNICITY.png"
)

# 3) SMOKING
panel_smoking <- make_domain_panel_plot_stratified(
    base_dir      = BASE_DIR_XGB_REGR,
    clocks_vec    = CLOCKS_REGR,
    stratum_vec   = c("EVER_SMOKED_RAND_1.0", "EVER_SMOKED_RAND_0.0"),
    stratum_labels = c("Ever smoker", "Never smoked"),
    METRIC        = METRIC_REGR,
    analysis_type_label = "Smoking History",
    out_png       = "~/Desktop/ML_EAA/Paper_graphs/domain_ablation_perm_XGBRegr_bySMOKING.png"
)