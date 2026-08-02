library(dplyr)
library(readr)
library(readxl)
library(tidyr)
library(ggplot2)
library(forcats)
library(stringr)

# ========= PATHS =========
xgb_grim  <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_EAA_GRIMAGE.csv"
xgb_dune  <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/XGB/Regressor/shap_long_EAA_DUNEDINMPOA.csv"

rf_path   <- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/RF/Regressor/permutation_feature_standard.csv"
lasso_path<- "/Users/martalens/Desktop/ML_EAA/output/nov_analyses/LASSO/lasso_coefficients.csv"

dict_path <- "/Users/martalens/Desktop/ML_EAA/reports/paper_word/npj_aging/additional_file_1.xlsx"

# ========= DICTIONARY =========
dict <- read_xlsx(dict_path) %>%
    transmute(
        variable = `VARIABLE NAME`,
        label    = DESCRIPTION,
        group    = GROUP
    )

# ========= XGB =========
get_xgb <- function(path, clock){
    read_csv(path, show_col_types = FALSE) %>%
        group_by(variable) %>%
        summarise(val = mean(abs(shap_value)), .groups="drop") %>%
        mutate(clock = clock)
}

xgb <- bind_rows(
    get_xgb(xgb_grim, "GrimAge"),
    get_xgb(xgb_dune, "DunedinPoAm")
) %>%
    group_by(clock) %>%
    mutate(prop = val / sum(val)) %>%
    ungroup() %>%
    mutate(model = "XGB")

# ========= RF =========
rf <- read_csv(rf_path, show_col_types = FALSE) %>%
    group_by(clock, feature) %>%
    summarise(val = mean(mean_drop), .groups="drop") %>%
    rename(variable = feature) %>%
    mutate(clock = recode(clock,
                          "EAA_GRIMAGE" = "GrimAge",
                          "EAA_DUNEDINMPOA" = "DunedinPoAm")) %>%
    group_by(clock) %>%
    mutate(prop = val / sum(val)) %>%
    ungroup() %>%
    mutate(model = "RF")

# ========= LASSO =========
lasso <- read_csv(lasso_path, show_col_types = FALSE) %>%
    mutate(val = abs(mean_coef) * selection_freq) %>%
    mutate(clock = recode(clock,
                          "EAA_GRIMAGE" = "GrimAge",
                          "EAA_DUNEDINMPOA" = "DunedinPoAm")) %>%
    group_by(clock) %>%
    mutate(prop = val / sum(val)) %>%
    ungroup() %>%
    select(variable, clock, prop) %>%
    mutate(model = "LASSO")

# ========= COMBINE =========
# ========= COMBINE ALL IMPORTANCE =========

all_imp <- bind_rows(
    xgb %>% select(variable, clock, prop, model),
    rf  %>% select(variable, clock, prop, model),
    lasso
)

all_imp <- all_imp %>%
    filter(clock %in% c("GrimAge", "DunedinPoAm"))

# check structure
print(head(all_imp, 20))
print(unique(all_imp$model))
print(unique(all_imp$clock))

# ========= STEP 1: TOP 15 PER MODEL × CLOCK =========


top_check <- all_imp %>%
    group_by(model, clock) %>%
    slice_max(prop, n = 15) %>%
    arrange(model, clock, desc(prop))

print(top_check, n = 100)

# ========= STEP 1: TOP 15 PER MODEL × CLOCK =========
top_check <- all_imp %>%
    group_by(model, clock) %>%
    slice_max(prop, n = 15) %>%
    arrange(model, clock, desc(prop)) %>%
    group_by(model, clock) %>%
    mutate(rank = row_number()) %>%
    ungroup()

print(top_check, n = 100)
top_check_flag <- top_check %>%
    mutate(in_top15 = TRUE) %>%
    select(variable, model, clock, in_top15)

# ========= STEP 2: APPLY HYBRID FILTER =========

# --- Top 10: always keep ---
vars_top10 <- top_check %>%
    filter(rank <= 10) %>%
    pull(variable) %>%
    unique()

# --- Rank 11–15: keep only if appears in ≥2 models ---
vars_10_15 <- top_check %>%
    filter(rank > 10 & rank <= 15) %>%
    mutate(model_clock = paste(model, clock)) %>%
    group_by(variable) %>%
    summarise(n_models = n_distinct(model_clock), .groups = "drop") %>%
    filter(n_models >= 2) %>%
    pull(variable)

# --- Final variable list ---
var_keep <- union(vars_top10, vars_10_15)

# --- Filter dataset ---
plot_df <- all_imp %>%
    filter(variable %in% var_keep)


# ========= STEP 4: ADD LABELS =========
plot_df <- plot_df %>%
    left_join(dict, by = "variable") %>%
    mutate(
        label = ifelse(is.na(label), variable, label),
        label = stringr::str_wrap(label, 30)
    )
plot_df <- plot_df %>%
    mutate(
        group = recode(
            group,
            "demographic" = "Demographics",
            "education, job and socioeconomic status" = "Education, Job and Socioeconomic Status",
            "early-life conditions and family background" = "Early-Life Conditions and Family Background",
            "adult adversity and life-course events" = "Adult Adversity and Life-Course Events",
            "social network and support" = "Social Network and Support",
            "health behaviours and clinical risk factors" = "Health Behaviors and Clinical Risk Factors",
            "genetics" = "Polygenic Indices and Ancestry Principal Components"
        )
    )

plot_df <- plot_df %>%
    left_join(top_check_flag, by = c("variable", "model", "clock"))
plot_df <- plot_df %>%
    mutate(prop_plot = ifelse(is.na(in_top15), NA, prop))
# ========= STEP 5: MODEL × CLOCK =========
plot_df <- plot_df %>%
    mutate(model_clock = paste(model, clock, sep = "_"))

plot_df$model_clock <- factor(
    plot_df$model_clock,
    levels = c(
        "XGB_GrimAge", "RF_GrimAge", "LASSO_GrimAge",
        "XGB_DunedinPoAm", "RF_DunedinPoAm", "LASSO_DunedinPoAm"
    )
)


# ========= STEP 6: ORDER VARIABLES =========
plot_df <- plot_df %>%
    group_by(variable) %>%
    mutate(mean_importance = mean(prop, na.rm = TRUE)) %>%
    ungroup()

order_vars <- plot_df %>%
    distinct(label, mean_importance) %>%
    arrange(desc(mean_importance)) %>%
    pull(label)

plot_df$label <- factor(plot_df$label, levels = rev(order_vars))

plot_df <- plot_df %>%
    mutate(
        model_pretty = recode(model,
                              "XGB" = "XGBoost",
                              "RF" = "Random Forest",
                              "LASSO" = "LASSO"
        ),
        clock_pretty = clock,
        model_clock = paste(model_pretty, clock_pretty, sep = "\n")
    )

plot_df$model_clock <- factor(
    plot_df$model_clock,
    levels = c(
        "XGBoost\nGrimAge",
        "Random Forest\nGrimAge",
        "LASSO\nGrimAge",
        "XGBoost\nDunedinPoAm",
        "Random Forest\nDunedinPoAm",
        "LASSO\nDunedinPoAm"
    )
)


library(dplyr)
library(readr)
library(readxl)
library(tidyr)
library(ggplot2)
library(forcats)
library(stringr)
library(ggnewscale)
library(scales)

# ========= DOMAIN COLORS =========
domain_colors <- c(
    "Adult Adversity and Life-Course Events" = "#DEEBF7",
    "Demographics" = "#C6DBEF",
    "Early-Life Conditions and Family Background" = "#9ECAE1",
    "Education, Job and Socioeconomic Status" = "#6BAED6",
    "Polygenic Indices and Ancestry Principal Components" = "#3182BD",
    "Health Behaviors and Clinical Risk Factors" = "#08519C",
    "Social Network and Support" = "#08306B"
)

# ========= PREP =========
plot_df <- plot_df %>%
    mutate(
        model_pretty = recode(model,
                              "XGB" = "XGBoost",
                              "RF" = "Random Forest",
                              "LASSO" = "LASSO"
        ),
        model_clock = interaction(clock, model_pretty, sep = "_")
    )

plot_df$model_clock <- factor(
    plot_df$model_clock,
    levels = c(
        "GrimAge_XGBoost",
        "GrimAge_Random Forest",
        "GrimAge_LASSO",
        "DunedinPoAm_XGBoost",
        "DunedinPoAm_Random Forest",
        "DunedinPoAm_LASSO"
    )
)

# ========= PLOT =========
p <- ggplot(plot_df, aes(x = model_clock, y = label)) +
    
    # ---- DOMAIN STRIP ----
geom_tile(aes(x = -0.5, fill = group), width = 0.18, alpha = 1) +
    scale_fill_manual(values = domain_colors, name = "Domain") +
    
    ggnewscale::new_scale_fill() +
    
    # ---- HEATMAP ----
geom_tile(aes(fill = prop_plot), color = "white", linewidth = 0.35) +
    
    scale_fill_gradient(
        low = "#F2F2F2",
        high = "#001F3F",
        limits = c(0, 0.30),
        oob = scales::squish,
        na.value = "white",
        name = "Importance (%)",
        labels = percent
    ) +
    
    # ---- VALUES ----
geom_text(
    aes(
        label = ifelse(
            is.na(prop_plot),
            "",
            paste0(round(prop_plot * 100, 2), "%")
        ),
        color = factor(prop_plot > 0.20)
    ),
    size = 3
) +
    scale_color_manual(
        values = c("FALSE" = "black", "TRUE" = "white"),
        guide = "none"
    ) +
    
    # ---- SEPARATOR ----
geom_vline(xintercept = 3.5, colour = "grey35", linewidth = 0.6) +
    
    # ---- FIXED X LABELS (ONLY MODELS) ----
scale_x_discrete(
    labels = c("XGBoost", "Random Forest", "LASSO",
               "XGBoost", "Random Forest", "LASSO"),
    expand = expansion(add = c(0.5, 0))
) +
    
    labs(
        x = "Model",
        y = "Predictors",
        subtitle = "                 GrimAge                    DunedinPoAm"
    ) +
    
    theme_minimal(base_size = 14) +
    theme(
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        
        axis.text.x = element_text(angle = 30, hjust = 1, size = 11),
        axis.text.y = element_text(size = 12),  # 🔥 bigger predictors
        
        axis.title = element_text(size = 12, face = "bold"),
        
        plot.subtitle = element_text(size = 13, face = "bold", hjust = 0.5, margin = margin(b = 10)),
        
        panel.grid = element_blank(),
        
        legend.position = "right",
        legend.box = "vertical",
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 10)
    )

print(p)

ggsave(
    "/Users/martalens/Desktop/ML_EAA/output/npj_aging_output/plots/heatmap_cross_model_final.png",
    p,
    width = 10,
    height = 12,
    dpi = 600
)





















