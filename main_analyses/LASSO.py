"""Canonical LASSO analyses accompanying the published study.

This script is the standalone form of executed cell 0 in ``LASSO.ipynb``.
Analytical statements and output schemas are preserved; only notebook scaffolding
and absolute paths are replaced.
"""

import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ANALYSIS_OUTPUT_DIR, EPIGENETIC_AGE_EVENTS_FILE

SEED = 42
N_SPLITS = 5
targets = ["EAA_GRIMAGE", "EAA_DUNEDINMPOA", "EAA_HORVATH", "EAA_HANNUM", "EAA_LEVINE"]

# Predictor domains (transferred unchanged from cell 0).
grouped_variables = {
    'adult adversity and life-course events': [
        'ALCOHOL_DRUG_FAM','ATTACK','CHLDIED','COMBAT','CURRENT_MARRIAGE_YRS_RAND','DENCARE','DENLOAN','DENPROM','DISASTER',
        'DISCRIMINATION','DISMJOB','DRUGOTH','EVER_AGE_DISCRIMINATION','EVER_ANCESTRY_DISCRIMINATION','EVER_APPAREANCE_DISCRIMINATION',
        'EVER_DISABILITY_DISCRIMINATION','EVER_FINANCIAL_DISCRIMINATION','EVER_GENDER_DISCRIMINATION','EVER_LIVED_AS_COUPLE',
        'EVER_ORIENTATION_DISCRIMINATION','EVER_RACE_DISCRIMINATION','EVER_RELIGION_DISCRIMINATION','EVER_WEIGHT_DISCRIMINATION',
        'FINANCIAL_STRAIN_ELDERLY','FINANCIAL_STRAIN_YRSLF','FRAUD_5Y_AGG','HOMELESS_AGG','HOUSE_AT40_1.0','HOUSE_AT40_2.0',
        'HOUSE_AT40_3.0','HOUSE_AT40_7.0','HOUSE_FIRSTJOB_1.0','HOUSE_FIRSTJOB_2.0','HOUSE_FIRSTJOB_3.0','HOUSE_FIRSTJOB_4.0',
        'HOUSE_FIRSTJOB_5.0','HOUSING_PROB_YOURSELF','ILLOTH','ILLSELF','JAIL_AGG','LH15','LH19','LH1B3','LH1B4','LH3A','LH4B','LH4C',
        'LH4D','LH4E','LH51','LIVING_AT40_ALONE','LIVING_AT40_BIOCHILD','LIVING_AT40_GRANDP','LIVING_AT40_INLAWS','LIVING_AT40_NONREL',
        'LIVING_AT40_OTHREL','LIVING_AT40_PARENTS','LIVING_AT40_PARTNER','LIVING_AT40_SIBS','LIVING_AT40_STEPCHILD','LIVING_AT40_CHILDREN_UNSPEC',
        'LIVING_ATFULLJOB_ALONE','LIVING_ATFULLJOB_BIOCHILD','LIVING_ATFULLJOB_GRANDP','LIVING_ATFULLJOB_INLAWS','LIVING_ATFULLJOB_MILITARY',
        'LIVING_ATFULLJOB_NONREL','LIVING_ATFULLJOB_OTHREL','LIVING_ATFULLJOB_PARENTS','LIVING_ATFULLJOB_PARTNER','LIVING_ATFULLJOB_STEPCHILD',
        'LIVING_ATFULLJOB_SIBS','LONGEST_MARRIAGE_YRS_RAND','LOST_JOB_5Y_AGG','MOVED_5Y_AGG','NEIGHBORHOOD_COHESION','NEIGHBORHOOD_DISORDER',
        'NEVER_MARRIED','NHIREDJOB','NUMBER_DIVORCES','NUMBER_LIVING_SIBLINGS','NUMBER_MARRIAGES','NUMBER_WIDOWED','PHY_EMOT_PROB_FAM',
        'PREVMOV','RELATIONSHIP_PROB','ROBBED_5Y_AGG','SICK_FAM','UNEMPLOYED_5Y_AGG','UNFPOLICE','VETERAN_RAND','WORK_DIFF_YRSLF', 'NO_LHMS_SUPP'
    ],
    'demographic': [
        'BIRTH_PLACE_1.0','BIRTH_PLACE_10.0','BIRTH_PLACE_11.0','BIRTH_PLACE_2.0','BIRTH_PLACE_3.0','BIRTH_PLACE_4.0','BIRTH_PLACE_5.0', 'BIRTH_PLACE_6.0','BIRTH_PLACE_7.0','BIRTH_PLACE_8.0','BIRTH_PLACE_9.0','BIRTHYR','GENDER','HISPANIC_BINARY','IMMGYEAR','LH1', 'NO_LHMS_SUPP',
        'MARITAL_STATUS_DIVORCED','MARITAL_STATUS_MARRIED','MARITAL_STATUS_NEVER_MARRIED','MARITAL_STATUS_PARTNERED',
        'MARITAL_STATUS_WIDOWED','RAEVBRN_CAT','RELIGION_1.0','RELIGION_2.0','RELIGION_3.0','RELIGION_4.0','RELIGION_5.0',
        'SIBLINGS_BINARY','USBORN', 'AFRICAN_AMERICAN_BINARY', 'WHITE_BINARY', 'NO_ETHNICITY_BINARY'
    ],
    'early-life conditions and family background': [
        'ATTENMO','CHMISSCH','DAD_AGE_RAND','DAD_ALIVE_RAND','DRKDRUG','DRUG','EFFMO','FAEDUC','FAMFIN','FATHER_DISABLED','FAUNEM','FMFINH', 'FJOB_ARMEDFORCES','FJOB_BLUECOLLAR', 'FJOB_SERVICE', 'FJOB_WHITECOLLAR', 'MOTHER_AGE_RAND',        'MOTHER_ALIVE_RAND', 'NUM_SCHOOLS', 'RELFA_NOTAPPLY',  'HOUSE_AT10_1.0','HOUSE_AT10_2.0','HOUSE_AT10_3.0','HOUSE_AT10_4.0','LH11','LH1B1','LH1B2','LH23','LH2A','LH2B','LH2C','LH2D','LH2E', 'NO_LHMS_SUPP',
        'LH2G','LH2H','LH2I','LH30','LH31A','LH31B','LH31C','LH31D','LH31E','LH31F','LH33','LH7','LH8','LIVEGPAR','LIVING_AT10_BIOFATH','LIVING_AT10_BIOMOTH','LIVING_AT10_GRANDP','LIVING_AT10_NONREL','LIVING_AT10_OTHREL','LIVING_AT10_SIBLINGS','LIVING_AT10_STEPFATH', 'LIVING_AT10_STEPMOTH','MOEDUC','MOVFIN','MOWORK','MULTILINGUAL_HOME','NEVER_LIVED_FATHER','NEVER_LIVED_MOTHER','PARSMOKE','PHYABUSE','PRIVATE_SCHOOL','RELWFA','RELWMO','RELWMO_NOTAPPLY','RTHLTHCH','SCHLOVER','SPOKE_ENGLISH_HOME','TEACHMO','TRPOLICE'
    ],
    'education, job and socioeconomic status': [
        'CHRONIC_UNEMPLOYMENT','CURRENTLY_PENSION','CURRENTLY_WORKING','DEGREE','FINANCIAL_CONTROL_CHANGE','HH_INCOME_DROPS','HH_MEAN_INCOME', 'HH_POVERTY_THRESHOLD_CAT',
        'HH_PERSISTENT_LOW_INCOME','HH_UNEMPLOYED_5Y_AGG','IND_PUBLIC','IND_SECONDARY','IND_TERTIARY','IND_UNKNOWN','LH13','LH34','LH38', 'NO_LHMS_SUPP',
        'LH39A','LH39B','LH39C','LH39D','LH40A','LH40B','LH40C','LH40D','LONGEST_JOB_DURATION','NUM_UNEMPLOYMENT','NUM_UNI','OCC_ARMEDFORCES',
        'OCC_BLUECOLLAR','OCC_SERVICE','OCC_WHITECOLLAR','OCC_UNKNOWN','PRIVATE_UNI','RAEDUC','SCHLYRS'
    ],
    'genetics': [
        'PC1_5A','PC1_5B','PC1_5C','PC1_5D','PC1_5E','PC6_10A','PC6_10B','PC6_10C','PC6_10D','PC6_10E','PGI_AB_BROAD17','PGI_ADHD_PGC17',
        'PGI_AFBC_SOCGEN16','PGI_AI_GSCAN19','PGI_ALC_PGC18','PGI_ANXFS_ANGST16','PGI_AUTISM_PGC17','PGI_BIP_PGC11','PGI_BMI2_GIANT18',
        'PGI_BUN_CKDGEN19','PGI_BUNTE_CKDGEN19','PGI_CAD_CARDIOGRAM11','PGI_CANNABIS_ICC18','PGI_CKD_CKDGEN19','PGI_CKDTE_CKDGEN19',
        'PGI_CRP_CHARGE22','PGI_DEPSYMP_SSGAC16','PGI_DPW_GSCAN19','PGI_EDU3_W23_SSGAC18','PGI_EGFR_CKDGEN19','PGI_EGFRTE_CKDGEN19',
        'PGI_EXTRAVERSION_GPC16','PGI_GENCOG2_CHARGE18','PGI_GWALZNA_PGC21','PGI_HBA1CEA_MAGIC17','PGI_HDL_GLGC13','PGI_HTN_COGNET17',
        'PGI_LDL_GLGC13','PGI_MENARCHE_REPROGEN17','PGI_MENOPAUSEREPROGEN21','PGI_MI_CARDIOGRAM15','PGI_NEBC_SOCGEN16',
        'PGI_NEUROTICISM_SSGAC16','PGI_OCD_IOCDF17','PGI_PTSDC_PGC18','PGI_SC_GSCAN19','PGI_SCZ_PGC14','PGI_SI_GSCAN19',
        'PGI_T2DALL_DIAGRAM24','PGI_TC_GLGC13','PGI_TG_GLGC13','PGI_WC_GIANT15','PGI_WELLBEING_SSGAC16','PGI_WHR_GIANT15','PGI_XDISORDER_PGC13'
    ],
    'lifestyle and health behaviors': [
        'BINGE_LIFECOURSE','BMI','CHSMOKE','CURRENTLY_DRINKS_RAND','CURRENTLY_SMOKING_RAND','FORMER_SMOKER','DAYS_WEEK_DRINKING_RAND', 'NO_LHMS_SUPP', 
        'DRINKS_PER_DAY_RAND','DRINKS_PER_WEEK_RAND','HEALTH_PROB_YRSLF','LH58A','LH58B','LH58C','LH59A','LH59B','LH59C',
        'PHYS_ACTIVITY_MILD','PHYS_ACTIVITY_MODERATE','PHYS_ACTIVITY_VIGOROUS'
    ],
    'social network and support': [
        'CHILDREN_NEG_SUPPORT','CHILDREN_POS_SUPPORT','CLOSE_CHILDREN','CLOSE_FAMILY','CLOSE_FRIENDS','CONTACT_CHILDREN','CONTACT_FAMILY',
        'CONTACT_FRIENDS','FAMILY_NEG_SUPPORT','FAMILY_POS_SUPPORT','FRIENDS_NEG_SUPPORT','FRIENDS_POS_SUPPORT','HAS_CHILDREN','HAS_FAMILY',
        'HAS_FRIENDS','LIVES_WITH_PARTNER','LONELINESS','RELATIONSHIP_SPOUSE','RELATIVES_NEIGHBORHOOD','SPOUSE_NEG_SUPPORT','SPOUSE_POS_SUPPORT'
    ],
    "smoking": ["CURRENTLY_SMOKING_RAND", "FORMER_SMOKER", "CHSMOKE"],

    'lifestyle without smoking': [
        'BINGE_LIFECOURSE','BMI', 'CURRENTLY_DRINKS_RAND', 'DAYS_WEEK_DRINKING_RAND', 'NO_LHMS_SUPP', 
        'DRINKS_PER_DAY_RAND','DRINKS_PER_WEEK_RAND','HEALTH_PROB_YRSLF','LH58A','LH58B','LH58C','LH59A','LH59B','LH59C',
        'PHYS_ACTIVITY_MILD','PHYS_ACTIVITY_MODERATE','PHYS_ACTIVITY_VIGOROUS']
}

all_predictors = []

def build_model(alpha):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=alpha, max_iter=20000))
    ])

def tune_alpha(X, y):
    alphas = np.logspace(-3, 1, 20)
    best_alpha, best_score = None, -np.inf

    inner = KFold(n_splits=5, shuffle=True, random_state=SEED + 1)

    for a in alphas:
        scores = []
        for tr, va in inner.split(X):
            m = build_model(a)
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[va])
            scores.append(r2_score(y.iloc[va], pred))

        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_alpha = a

    return best_alpha

# ========= KNN PERM =========
def permutation_knn(model, Xtr, Xva, yva, cols, n_repeats=30):

    imp = SimpleImputer()
    sc = StandardScaler()

    cond = [c for c in Xtr.columns if c not in cols]

    Ztr = sc.fit_transform(imp.fit_transform(Xtr[cond]))
    Zva = sc.transform(imp.transform(Xva[cond]))

    nn = NearestNeighbors(n_neighbors=min(40, len(Xtr))).fit(Ztr)
    _, idx = nn.kneighbors(Zva)

    base = r2_score(yva, model.predict(Xva))
    scores = []

    for _ in range(n_repeats):
        choice = np.random.randint(0, idx.shape[1], size=len(Xva))
        donor = idx[np.arange(len(Xva)), choice]

        Xp = Xva.copy()
        donor_vals = Xtr.iloc[donor][cols].values

        for i, c in enumerate(cols):
            Xp[c] = donor_vals[:, i]

        scores.append(r2_score(yva, model.predict(Xp)))

    return base, np.mean(scores), np.std(scores), base - np.mean(scores)
def compute_learning_curves_lasso(X, y, splits, alpha_per_fold, train_sizes=np.linspace(0.1, 1.0, 8), seed=42):

    rng = np.random.default_rng(seed)
    results = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):

        Xtr_full, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr_full, yva = y.iloc[tr_idx], y.iloc[va_idx]

        alpha = alpha_per_fold[fold_id]

        n_tr = len(Xtr_full)

        order = rng.permutation(n_tr)
        Xtr_full = Xtr_full.iloc[order].reset_index(drop=True)
        ytr_full = ytr_full.iloc[order].reset_index(drop=True)

        for frac in train_sizes:

            k = max(10, int(np.ceil(frac * n_tr)))

            Xtr = Xtr_full.iloc[:k]
            ytr = ytr_full.iloc[:k]

            model = build_model(alpha)
            model.fit(Xtr, ytr)

            train_r2 = r2_score(ytr, model.predict(Xtr))
            val_r2 = r2_score(yva, model.predict(Xva))

            results.append({
                "fold": fold_id,
                "train_size_frac": float(frac),
                "train_n": int(k),
                "train_r2": float(train_r2),
                "val_r2": float(val_r2)
            })

    df = pd.DataFrame(results)

    agg = df.groupby("train_size_frac", as_index=False).agg(
        train_mean=("train_r2", "mean"),
        train_sd=("train_r2", "std"),
        val_mean=("val_r2", "mean"),
        val_sd=("val_r2", "std"),
        train_n_mean =("train_n", "mean")  
    )  

    return df, agg


# ========= STRATIFIED =========
def run_stratified_full(X, y, strat_col, target):

    all_rows = []
    coef_rows = []
    domain_rows = []

    for val in X[strat_col].dropna().unique():

        mask = X[strat_col] == val
        Xs = X.loc[mask, all_predictors].reset_index(drop=True)
        ys = y.loc[mask].reset_index(drop=True)

        if len(ys) < 50:
            continue

        print(f"[STRAT] {strat_col}={val} (n={len(ys)})")

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

        fold_metrics = []
        coef_list = []
        dom_rows_local = []

        for fold, (tr, va) in enumerate(kf.split(Xs)):

            Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
            ytr, yva = ys.iloc[tr], ys.iloc[va]

            alpha = tune_alpha(Xtr, ytr)
            model = build_model(alpha)
            model.fit(Xtr, ytr)

            pred = model.predict(Xva)
            pred_tr = model.predict(Xtr)

            fold_metrics.append({
                "r2": r2_score(yva, pred),
                "rmse": root_mean_squared_error(yva, pred),
                "mae": mean_absolute_error(yva, pred),
                "train_r2": r2_score(ytr, pred_tr)
            })

            coef_list.append(model.named_steps["lasso"].coef_)

            # ===== DOMAIN ANALYSIS =====
            for name, cols in grouped_variables.items():

                cols = [c for c in cols if c in Xs.columns]
                if not cols:
                    continue

                # domain-only
                m_dom = build_model(alpha)
                m_dom.fit(Xtr[cols], ytr)
                r2_dom = r2_score(yva, m_dom.predict(Xva[cols]))

                # ablation
                Xtr_ab = Xtr.drop(columns=cols)
                Xva_ab = Xva.drop(columns=cols)

                m_ab = build_model(alpha)
                m_ab.fit(Xtr_ab, ytr)
                r2_ab = r2_score(yva, m_ab.predict(Xva_ab))

                # permutation
                base, pm, psd, diff = permutation_knn(model, Xtr, Xva, yva, cols)

                dom_rows_local.append({
                    "group": name,
                    "baseline_r2": base,
                    "domain_r2": r2_dom,
                    "ablation_r2": r2_ab,
                    "perm_mean": pm,
                    "perm_sd": psd,
                    "perm_drop": diff
                })

        # ===== aggregate performance =====
        folds_df = pd.DataFrame(fold_metrics)

        all_rows.append({
            "clock": target,
            "stratum": f"{strat_col}_{val}",
            "mean_r2": folds_df["r2"].mean(),
            "sd_r2": folds_df["r2"].std(),
            "mean_rmse": folds_df["rmse"].mean(),
            "sd_rmse": folds_df["rmse"].std(),
            "mean_mae": folds_df["mae"].mean(),
            "sd_mae": folds_df["mae"].std(),
            "train_mean_r2": folds_df["train_r2"].mean()
        })

        # ===== coefficients =====
        coef_arr = np.vstack(coef_list)

        coef_df = pd.DataFrame({
            "variable": all_predictors,
            "mean_coef": coef_arr.mean(0),
            "sd_coef": coef_arr.std(0),
            "selection_freq": (coef_arr != 0).mean(0),
            "clock": target,
            "stratum": f"{strat_col}_{val}"
        })

        coef_rows.append(coef_df)

        # ===== domain aggregation =====
        dom_df = pd.DataFrame(dom_rows_local)

        if not dom_df.empty:
            dom_agg = (
                dom_df.groupby("group")
                .agg(
                    baseline_mean_r2=("baseline_r2", "mean"),
                    baseline_sd_r2=("baseline_r2", "std"),

                    domain_mean_r2=("domain_r2", "mean"),
                    domain_sd_r2=("domain_r2", "std"),

                    ablation_mean_r2=("ablation_r2", "mean"),
                    ablation_sd_r2=("ablation_r2", "std"),

                    perm_mean_r2=("perm_mean", "mean"),
                    perm_between_sd=("perm_mean", "std"),   # across folds
                    perm_within_sd=("perm_sd", "mean"),     # within fold (KNN variability)
                )
                .reset_index()
            )

            dom_agg["clock"] = target
            dom_agg["stratum"] = f"{strat_col}_{val}"

            domain_rows.append(dom_agg)

    return (
        pd.DataFrame(all_rows),
        pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame(),
        pd.concat(domain_rows, ignore_index=True) if domain_rows else pd.DataFrame()
    )

# ========= MAIN =========
# ========= MAIN =========

# Main workflow: data loading, shared folds, model fitting, and output writing.
def main():
    global all_predictors

    # Final analysis dataset; its existing row order is analytical input.
    df = pd.read_csv(EPIGENETIC_AGE_EVENTS_FILE)
    outdir = ANALYSIS_OUTPUT_DIR / "LASSO"
    os.makedirs(outdir, exist_ok=True)

    # Predictor preparation and shared outer folds.
    all_predictors = sorted({
        c for cols in grouped_variables.values()
        for c in cols if c in df.columns
    })
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    _indices = np.arange(len(df))
    SPLITS = list(kf.split(_indices))

    for target in targets:

        print(f"\n==== {target} ====")

        X = df[all_predictors].reset_index(drop=True)
        y = df[target].reset_index(drop=True)

        cv_rows, coef_list, domain_rows = [], [], []

        alpha_per_fold = {}

        for fold, (tr, va) in enumerate(SPLITS):

            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]

            alpha = tune_alpha(Xtr, ytr)

            alpha_per_fold[fold] = alpha

            model = build_model(alpha)
            model.fit(Xtr, ytr)

            pred = model.predict(Xva)
            pred_tr = model.predict(Xtr)

            cv_rows.append({
                "fold": fold,
                "r2": r2_score(yva, pred),
                "rmse": root_mean_squared_error(yva, pred),
                "mae": mean_absolute_error(yva, pred),
                "train_r2": r2_score(ytr, pred_tr)
            })

            coef_list.append(model.named_steps["lasso"].coef_)

            for name, cols in grouped_variables.items():

                cols = [c for c in cols if c in X.columns]
                if not cols:
                    continue

                # domain-only
                m_dom = build_model(alpha)
                m_dom.fit(Xtr[cols], ytr)
                r2_dom = r2_score(yva, m_dom.predict(Xva[cols]))

                # ablation
                Xtr_ab = Xtr.drop(columns=cols)
                Xva_ab = Xva.drop(columns=cols)

                m_ab = build_model(alpha)
                m_ab.fit(Xtr_ab, ytr)
                r2_ab = r2_score(yva, m_ab.predict(Xva_ab))

                base, pm, psd, diff = permutation_knn(model, Xtr, Xva, yva, cols)

                domain_rows.append({
                    "fold": fold,
                    "group": name,
                    "baseline_r2": base,
                    "domain_r2": r2_dom,
                    "ablation_r2": r2_ab,
                    "perm_mean": pm,
                    "perm_sd": psd,
                    "perm_drop": diff
                })

        # ========= PERFORMANCE =========
        cv_df = pd.DataFrame(cv_rows)

        perf = {
            "clock": target,
            "mean_r2": cv_df["r2"].mean(),
            "sd_r2": cv_df["r2"].std(),
            "mean_rmse": cv_df["rmse"].mean(),
            "sd_rmse": cv_df["rmse"].std(),
            "mean_mae": cv_df["mae"].mean(),
            "sd_mae": cv_df["mae"].std(),
            "train_mean_r2": cv_df["train_r2"].mean(),
            "train_sd_r2": cv_df["train_r2"].std()
        }

        pd.DataFrame([perf]).to_csv(
            os.path.join(outdir, "model_performance.csv"),
            mode="a",
            header=not os.path.exists(os.path.join(outdir, "model_performance.csv")),
            index=False
        )

        # ========= COEFFICIENTS =========
        coef_arr = np.vstack(coef_list)

        coef_df = pd.DataFrame({
            "variable": all_predictors,
            "mean_coef": coef_arr.mean(0),
            "sd_coef": coef_arr.std(0),
            "selection_freq": (coef_arr != 0).mean(0),
            "clock": target
        })

        coef_df.to_csv(
            os.path.join(outdir, "lasso_coefficients.csv"),
            mode="a",
            header=not os.path.exists(os.path.join(outdir, "lasso_coefficients.csv")),
            index=False
        )

        # ========= DOMAIN AGG =========
        dom = pd.DataFrame(domain_rows)

        dom_agg = (
            dom.groupby("group")
            .agg(
                baseline_mean_r2=("baseline_r2", "mean"),
                baseline_sd_r2=("baseline_r2", "std"),
                domain_mean_r2=("domain_r2", "mean"),
                domain_sd_r2=("domain_r2", "std"),
                ablation_mean_r2=("ablation_r2", "mean"),
                ablation_sd_r2=("ablation_r2", "std"),
                perm_mean_r2=("perm_mean", "mean"),
                perm_sd_r2=("perm_mean", "std"),
            )
            .reset_index()
        )

        dom_agg["ablation_drop"] = dom_agg["baseline_mean_r2"] - dom_agg["ablation_mean_r2"]
        dom_agg["perm_drop"] = dom_agg["baseline_mean_r2"] - dom_agg["perm_mean_r2"]
        dom_agg["clock"] = target

        dom_agg.to_csv(
            os.path.join(outdir, "domain_analysis.csv"),
            mode="a",
            header=not os.path.exists(os.path.join(outdir, "domain_analysis.csv")),
            index=False
        )

        # ========= LEARNING CURVES (FIXED) =========
        lc_raw, lc_agg = compute_learning_curves_lasso(X, y, SPLITS, alpha_per_fold)

        lc_raw["clock"] = target
        lc_agg["clock"] = target

        lc_raw.to_csv(
            os.path.join(outdir, "learning_curve_raw.csv"),
            mode="a",
            header=not os.path.exists(os.path.join(outdir, "learning_curve_raw.csv")),
            index=False
        )

        lc_agg.to_csv(
            os.path.join(outdir, "learning_curve.csv"),
            mode="a",
            header=not os.path.exists(os.path.join(outdir, "learning_curve.csv")),
            index=False
        )

        # ========= STRATIFIED =========
        for strat_col in ["GENDER", "WHITE_BINARY", "EVER_SMOKED_RAND"]:

            if strat_col not in df.columns:
                continue

            Xs = X.copy()
            Xs[strat_col] = df[strat_col].values

            perf_strat, coef_strat, dom_strat = run_stratified_full(Xs, y, strat_col, target)

            if not perf_strat.empty:
                perf_strat.to_csv(
                    os.path.join(outdir, "stratified_performance.csv"),
                    mode="a",
                    header=not os.path.exists(os.path.join(outdir, "stratified_performance.csv")),
                    index=False
                )

            if not coef_strat.empty:
                coef_strat.to_csv(
                    os.path.join(outdir, "stratified_coefficients.csv"),
                    mode="a",
                    header=not os.path.exists(os.path.join(outdir, "stratified_coefficients.csv")),
                    index=False
                )

            if not dom_strat.empty:
                dom_strat.to_csv(
                    os.path.join(outdir, "stratified_domain_analysis.csv"),
                    mode="a",
                    header=not os.path.exists(os.path.join(outdir, "stratified_domain_analysis.csv")),
                    index=False
                )

    print("DONE")


if __name__ == "__main__":
    main()
