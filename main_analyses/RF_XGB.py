"""Canonical Random Forest/XGBoost analysis workflow.

This script is derived from the single executed cell in ``RF_XGB.ipynb``.
The analytical statements, ordering, seeds, folds, model grids, metrics, and
output schemas are preserved; only path handling, execution structure, and
explicitly unused notebook material have been cleaned.
"""


import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import ANALYSIS_OUTPUT_DIR, EPIGENETIC_AGE_EVENTS_FILE

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    r2_score, root_mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, balanced_accuracy_score,
    roc_curve, precision_recall_curve, brier_score_loss
)

def main():
    # Load the final analysis dataset without changing its row order.
    df = pd.read_csv(EPIGENETIC_AGE_EVENTS_FILE)
    outdir = ANALYSIS_OUTPUT_DIR
    os.makedirs(outdir, exist_ok=True)

    # Predictor domains, targets, and configuration
    PARALLEL_CORES = max(1, min(4, os.cpu_count() or 2))  # 2–4 on a laptop
    N_SPLITS = 5
    SEED = 42

    # Predictor-domain definitions
    # ------------------ GROUPS ------------------
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

    # Targets and tasks
    # ========= TARGETS / TASKS =========
    epigenetic_clocks = {
        'CONT':   ["EAA_GRIMAGE", "EAA_DUNEDINMPOA", "EAA_HORVATH", "EAA_HANNUM", "EAA_LEVINE"],
        'BINARY': ["EAA_GRIMAGE_BINARY", "EAA_DUNEDINMPOA_BINARY", "EAA_HORVATH_BINARY", "EAA_HANNUM_BINARY", "EAA_LEVINE_BINARY"],
    }
    limited_clocks = []
    SHAP_TARGETS = {"EAA_GRIMAGE", "EAA_DUNEDINMPOA", "EAA_HORVATH", "EAA_HANNUM", "EAA_LEVINE"}

    # Hyperparameter grids
    # ——— DEFAULT (all clocks except DunedinPoAm) ———
    rf_param_grid_strict = {
        "n_estimators": [300, 600, 900],
        "max_depth": [3, 4, 5],
        "min_samples_split": [40, 80, 120],
        "min_samples_leaf": [20, 40, 80],
        "max_features": ["log2", 0.1, 0.2],
        "max_samples": [0.4, 0.5, 0.6],
        "ccp_alpha": [0.0005, 0.0015, 0.003],
        "min_impurity_decrease": [1e-4, 5e-4, 1e-3],
    }

    xgb_param_grid_strict = {
        "n_estimators": [600, 900, 1200],
        "learning_rate": [0.01, 0.02],
        "max_depth": [2, 3],
        "min_child_weight": [30, 60, 90],
        "subsample": [0.5, 0.6],
        "colsample_bytree": [0.5, 0.6],
        "colsample_bylevel": [0.6, 0.8],
        "gamma": [5, 10, 15],
        "reg_lambda": [100, 200, 300],
        "reg_alpha": [10, 20, 40],
    }

    DUNEDIN_BASES = {"EAA_DUNEDINMPOA"}

    def base_clock_name(t: str) -> str:
        return t[:-7] if t.endswith("_BINARY") else t

    def is_dunedin(target: str) -> bool:
        return base_clock_name(target) in DUNEDIN_BASES

    # ——— DUNEDIN (less regularization) ———
    rf_param_grid_poam = {
        "n_estimators": [600, 900, 1200],
        "max_depth": [6, 8, None],
        "min_samples_split": [10, 20, 40],
        "min_samples_leaf": [5, 10, 20],
        "max_features": ["sqrt", 0.5],
        "max_samples": [0.6, 0.8, 1.0],
        "ccp_alpha": [0.0, 0.0005, 0.001],
        "min_impurity_decrease": [0.0, 1e-4, 5e-4],
    }

    xgb_param_grid_poam = {
        "n_estimators": [800, 1200, 1600],
        "learning_rate": [0.03, 0.05],
        "max_depth": [2, 3],
        "min_child_weight": [20, 40, 80],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "colsample_bylevel": [0.6, 0.8],
        "gamma": [0, 1, 3, 5],
        "reg_lambda": [60, 120, 200, 300],
        "reg_alpha": [10, 30, 60],
    }

    PERF_COLS = [
        "clock", "model", "stratum",
        "mean_r2", "sd_r2", "mean_auc", "sd_auc",
        "mean_rmse", "sd_rmse", "mean_mae", "sd_mae",
        "mean_oob", "sd_oob",
        "mean_accuracy", "sd_accuracy",
        "mean_f1", "sd_f1",
        "mean_precision", "sd_precision",
        "mean_recall", "sd_recall",
        "mean_bal_accuracy", "sd_bal_accuracy",
        "train_mean_r2", "train_sd_r2",
        "train_mean_auc", "train_sd_auc",
        "train_mean_rmse", "train_sd_rmse",
        "train_mean_mae", "train_sd_mae",
        "train_mean_accuracy", "train_sd_accuracy",
        "train_mean_f1", "train_sd_f1",
        "train_mean_precision", "train_sd_precision",
        "train_mean_recall", "train_sd_recall",
        "train_mean_bal_accuracy", "train_sd_bal_accuracy",
    ]

    dummy_cols = {
        "BIRTH_PLACE": ['BIRTH_PLACE_1.0','BIRTH_PLACE_10.0','BIRTH_PLACE_11.0','BIRTH_PLACE_2.0','BIRTH_PLACE_3.0','BIRTH_PLACE_4.0','BIRTH_PLACE_5.0','BIRTH_PLACE_6.0','BIRTH_PLACE_7.0','BIRTH_PLACE_8.0','BIRTH_PLACE_9.0'],
        'HOUSE_AT10': ["HOUSE_AT10_1.0","HOUSE_AT10_2.0","HOUSE_AT10_3.0","HOUSE_AT10_4.0"],
        'HOUSE_AT40': ["HOUSE_AT40_1.0","HOUSE_AT40_2.0","HOUSE_AT40_3.0","HOUSE_AT40_7.0"],
        'HOUSE_ATFIRSTJOB': ["HOUSE_FIRSTJOB_1.0","HOUSE_FIRSTJOB_2.0","HOUSE_FIRSTJOB_3.0","HOUSE_FIRSTJOB_4.0","HOUSE_FIRSTJOB_5.0"],
        "INDUSTRY": ["IND_PRIMARY","IND_SECONDARY","IND_TERTIARY","IND_PUBLIC","IND_UNKNOWN"],
        'LIVING_AT10': ['LIVING_AT10_BIOFATH','LIVING_AT10_BIOMOTH','LIVING_AT10_GRANDP','LIVING_AT10_NONREL','LIVING_AT10_OTHREL','LIVING_AT10_SIBLINGS','LIVING_AT10_STEPFATH','LIVING_AT10_STEPMOTH'],
        'LIVING_AT40': ['LIVING_AT40_ALONE','LIVING_AT40_BIOCHILD','LIVING_AT40_GRANDP','LIVING_AT40_INLAWS','LIVING_AT40_NONREL','LIVING_AT40_OTHREL','LIVING_AT40_PARENTS','LIVING_AT40_PARTNER','LIVING_AT40_SIBS','LIVING_AT40_STEPCHILD','LIVING_AT40_CHILDREN_UNSPEC'],
        'LIVING_ATFULLJOB': ['LIVING_ATFULLJOB_ALONE','LIVING_ATFULLJOB_BIOCHILD','LIVING_ATFULLJOB_GRANDP','LIVING_ATFULLJOB_INLAWS','LIVING_ATFULLJOB_MILITARY','LIVING_ATFULLJOB_NONREL','LIVING_ATFULLJOB_OTHREL','LIVING_ATFULLJOB_PARENTS','LIVING_ATFULLJOB_PARTNER','LIVING_ATFULLJOB_STEPCHILD','LIVING_ATFULLJOB_SIBS'],
        "MARITAL_STATUS": ["MARITAL_STATUS_MARRIED","MARITAL_STATUS_DIVORCED","MARITAL_STATUS_WIDOWED","MARITAL_STATUS_NEVER_MARRIED","MARITAL_STATUS_PARTNERED"],
        "RELIGION": ["RELIGION_1.0","RELIGION_2.0","RELIGION_3.0","RELIGION_4.0","RELIGION_5.0"],
        "OCCUPATION": ["OCC_BLUECOLLAR","OCC_WHITECOLLAR","OCC_SERVICE","OCC_ARMEDFORCES","OCC_UNKNOWN"],
        "FJOB2": ["FJOB_ARMEDFORCES","FJOB_BLUECOLLAR","FJOB_SERVICE","FJOB_WHITECOLLAR"]
    }


    # Common helper functions
    # ========= SMALL UTILITIES =========
    def drop_constant_and_near_duplicate(X: pd.DataFrame, corr_threshold=0.995, sample_for_corr=2000, verbose=True):
        X = X.copy()
        nunique = X.nunique(dropna=True)
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            X.drop(columns=const_cols, inplace=True, errors="ignore")
            if verbose:
                print(f"[PRE] Dropped {len(const_cols)} constant columns.")
        numX = X.select_dtypes(include=[np.number])
        if numX.shape[1] >= 2:
            if len(numX) > sample_for_corr:
                numX_s = numX.sample(sample_for_corr, random_state=SEED)
            else:
                numX_s = numX
            corr = numX_s.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                if col in to_drop: continue
                high_corr = upper.index[upper[col] > corr_threshold].tolist()
                to_drop.update(high_corr)
            if to_drop:
                X.drop(columns=list(to_drop), inplace=True, errors="ignore")
                if verbose:
                    print(f"[PRE] Dropped {len(to_drop)} near-duplicate (|corr|>{corr_threshold}) columns.")
        return X


    def augment_params_for_class_imbalance(model_class, params, y_train, is_classification):
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        q = dict(params) if params else {}
        if not is_classification: return q
        y_arr = np.asarray(y_train)
        classes = np.unique(y_arr)
        if len(classes) == 2:
            n_pos = np.sum(y_arr == classes.max())
            n_neg = np.sum(y_arr == classes.min())
            if getattr(model_class, "__name__", "").endswith("Classifier"):
                if model_class is XGBClassifier:
                    if n_pos > 0:
                        q["scale_pos_weight"] = float(n_neg / max(1, n_pos))
                if model_class is RandomForestClassifier:
                    q["class_weight"] = "balanced"
        else:
            if getattr(model_class, "__name__", "").endswith("Classifier"):
                if model_class is RandomForestClassifier:
                    q["class_weight"] = "balanced"
        return q

    def _coerce_param_types_for_model(model_class, p: dict) -> dict:
        from xgboost import XGBRegressor, XGBClassifier
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        def _as_none(x):
            if x is None: return None
            if isinstance(x, str) and x.strip().lower() in {"none","nan","na",""}: return None
            try:
                if pd.isna(x): return None
            except Exception:
                pass
            return x
        q = dict(p) if p else {}
        for k, v in list(q.items()): q[k] = _as_none(v)
        if model_class in (XGBRegressor, XGBClassifier):
            for k in ("n_estimators","max_depth","min_child_weight","max_bin"):
                if k in q and q[k] is not None: q[k] = int(q[k])
            for k in ("subsample","colsample_bytree","colsample_bylevel","learning_rate","gamma","reg_lambda","reg_alpha"):
                if k in q and q[k] is not None: q[k] = float(q[k])
        if model_class in (RandomForestRegressor, RandomForestClassifier):
            for k in ("n_estimators","min_samples_split","min_samples_leaf"):
                if k in q and q[k] is not None: q[k] = int(q[k])
            if "max_depth" in q and q["max_depth"] is not None: q["max_depth"] = int(q["max_depth"])
            if "max_features" in q and isinstance(q["max_features"], (int, float, np.floating)):
                q["max_features"] = float(q["max_features"]) if q["max_features"] <= 1 else int(q["max_features"])
            if "ccp_alpha" in q and q["ccp_alpha"] is not None: q["ccp_alpha"] = float(q["ccp_alpha"])
            if "min_impurity_decrease" in q and q["min_impurity_decrease"] is not None: q["min_impurity_decrease"] = float(q["min_impurity_decrease"])
        return q

    def make_model(model_class, params, n_jobs=1):
        p = _coerce_param_types_for_model(model_class, params or {})
        p["n_jobs"] = n_jobs
        from xgboost import XGBRegressor, XGBClassifier
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        if model_class in (RandomForestRegressor, RandomForestClassifier):
            p["oob_score"] = True
            p.setdefault("random_state", SEED) 
        if model_class in (XGBRegressor, XGBClassifier):
            p.setdefault("tree_method", "hist")
            p.setdefault("max_bin", 256)
            p.setdefault("random_state", SEED)
            p.setdefault("eval_metric", "auc" if model_class is XGBClassifier else "rmse")
        return model_class(**p)

    def fit_model(model_class, params, Xtr, ytr, is_classification, n_jobs=PARALLEL_CORES):
        model = make_model(model_class, params or {}, n_jobs=n_jobs)
        model.fit(Xtr, ytr)
        return model

    def tune_hyperparameters(model, param_grid, X, y, is_classification, n_iter=30):
        if is_classification:
            y_enc = LabelEncoder().fit_transform(y)
            scorer = 'roc_auc_ovr' if len(np.unique(y_enc)) > 2 else 'roc_auc'
            search = RandomizedSearchCV(
                model, param_distributions=param_grid, n_iter=n_iter,
                cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
                scoring=scorer, n_jobs=PARALLEL_CORES, random_state=SEED
            )
            print(f"[TUNE] Starting param search (Classifier) with {n_iter} iters...")
            search.fit(X, y_enc)
        else:
            search = RandomizedSearchCV(
                model, param_distributions=param_grid, n_iter=n_iter,
                cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
                scoring="r2", n_jobs=PARALLEL_CORES, random_state=SEED
            )
            print(f"[TUNE] Starting param search (Regressor) with {n_iter} iters...")
            search.fit(X, y)
        print("[TUNE] Done. Best params selected.")
        return search.best_params_, search.cv_results_

    def _cls_metrics_dict(y_true, y_pred, y_proba):
        out = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted"),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    out["auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    out["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
            except Exception:
                out["auc"] = np.nan
        else:
            out["auc"] = np.nan
        return out

    def get_metrics(y_true, y_pred, y_proba, is_classification):
        if is_classification:
            return _cls_metrics_dict(y_true, y_pred, y_proba)
        else:
            return {"r2": r2_score(y_true, y_pred),
                    "rmse": root_mean_squared_error(y_true, y_pred),
                    "mae": mean_absolute_error(y_true, y_pred)}

    def _permute_column_inplace(Xdf, col, rng):
        vals = Xdf[col].to_numpy(copy=True)
        mask = ~pd.isna(vals)
        vals[mask] = rng.permutation(vals[mask])
        Xdf[col] = vals

    def permutation_featurewise_standard(model, Xva, yva, is_classification, n_repeats=10, seed=42):
        print("[PERM] Standard per-feature permutation on OUTER-VAL ...")
        rng = np.random.default_rng(seed)
        yhat = model.predict(Xva)
        yproba = model.predict_proba(Xva) if hasattr(model, "predict_proba") else None
        base = get_metrics(yva, yhat, yproba, is_classification)
        base_main = base["auc"] if is_classification else base["r2"]
        rows = []
        for col in Xva.columns:
            drops = []
            for _ in range(n_repeats):
                Xp = Xva.copy()
                _permute_column_inplace(Xp, col, rng)
                yhatp = model.predict(Xp)
                yprop = model.predict_proba(Xp) if hasattr(model, "predict_proba") else None
                met = get_metrics(yva, yhatp, yprop, is_classification)
                drops.append(base_main - (met["auc"] if is_classification else met["r2"]))
            rows.append({"feature": col, "mean_drop": float(np.mean(drops)), "sd_drop": float(np.std(drops))})
        return pd.DataFrame(rows).sort_values("mean_drop", ascending=False, ignore_index=True)

    def permutation_grouped_categorical_standard(model, Xva, yva, is_classification,
                                                 cat_to_dummies: dict, n_repeats=10, seed=42):
        if not cat_to_dummies:
            print("[PERM] No cat_to_dummies mapping; skipping.")
            return pd.DataFrame()
        print("[PERM] Grouped-categorical permutation ...")
        rng = np.random.default_rng(seed)
        yhat = model.predict(Xva)
        yproba = model.predict_proba(Xva) if hasattr(model, "predict_proba") else None
        base = get_metrics(yva, yhat, yproba, is_classification)
        base_main = base["auc"] if is_classification else base["r2"]
        rows = []
        for orig, dums in cat_to_dummies.items():
            dcols = [d for d in (dums if isinstance(dums, (list, tuple)) else []) if d in Xva.columns]
            if not dcols: continue
            drops = []
            for _ in range(n_repeats):
                Xp = Xva.copy()
                for d in dcols: _permute_column_inplace(Xp, d, rng)
                yhatp = model.predict(Xp)
                yprop = model.predict_proba(Xp) if hasattr(model, "predict_proba") else None
                met = get_metrics(yva, yhatp, yprop, is_classification)
                drops.append(base_main - (met["auc"] if is_classification else met["r2"]))
            rows.append({"original_categorical": orig, "n_dummy_cols": len(dcols),
                         "mean_drop": float(np.mean(drops)), "sd_drop": float(np.std(drops))})
        return pd.DataFrame(rows).sort_values("mean_drop", ascending=False, ignore_index=True)

    def permutation_group_standard(model, Xva, yva, is_classification,
                                   group_to_cols: dict, n_repeats=10, seed=42):
        print("[PERM] Conceptual group permutation ...")
        rng = np.random.default_rng(seed)
        yhat = model.predict(Xva)
        yproba = model.predict_proba(Xva) if hasattr(model, "predict_proba") else None
        base = get_metrics(yva, yhat, yproba, is_classification)
        base_main = base["auc"] if is_classification else base["r2"]
        rows = []
        for g, cols in group_to_cols.items():
            gcols = [c for c in cols if c in Xva.columns]
            if not gcols: continue
            drops = []
            for _ in range(n_repeats):
                Xp = Xva.copy()
                for c in gcols: _permute_column_inplace(Xp, c, rng)
                yhatp = model.predict(Xp)
                yprop = model.predict_proba(Xp) if hasattr(model, "predict_proba") else None
                met = get_metrics(yva, yhatp, yprop, is_classification)
                drops.append(base_main - (met["auc"] if is_classification else met["r2"]))
            rows.append({"group": g, "n_cols": len(gcols),
                         "mean_drop": float(np.mean(drops)), "sd_drop": float(np.std(drops))})
        return pd.DataFrame(rows).sort_values("mean_drop", ascending=False, ignore_index=True)

    def permutation_knn(model, X_train, X_val, y_val, group_cols,
                        is_classification, n_repeats=50, k_neighbors=40, seed=42):
        rng = np.random.default_rng(seed)
        cond_cols = [c for c in X_train.columns if c not in group_cols]
        if len(cond_cols) == 0: return np.nan, np.nan, np.nan, np.nan, np.nan
        imp = SimpleImputer(strategy='median'); sclr = StandardScaler()
        Ztr = sclr.fit_transform(imp.fit_transform(X_train[cond_cols]))
        Zva = sclr.transform(imp.transform(X_val[cond_cols]))
        k_use = min(k_neighbors, len(X_train))
        if k_use < 1: return np.nan, np.nan, np.nan, np.nan, np.nan
        nn = NearestNeighbors(n_neighbors=k_use).fit(Ztr)
        _, idxs = nn.kneighbors(Zva)
        if is_classification:
            if getattr(y_val, "dtype", pd.Series(y_val).dtype).kind not in "iu":
                le = LabelEncoder(); y_eval = le.fit_transform(y_val)
            else:
                y_eval = np.asarray(y_val)
        else:
            y_eval = np.asarray(y_val, dtype=float)
        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        base_met = get_metrics(y_eval, y_pred, y_proba, is_classification)
        main_vals = []
        for _ in range(n_repeats):
            choice = rng.integers(0, idxs.shape[1], size=idxs.shape[0])
            donor_pos = idxs[np.arange(idxs.shape[0]), choice]
            X_val_perm = X_val.copy()
            valid_cols = [c for c in group_cols if (c in X_train.columns) and (c in X_val_perm.columns)]
            seen = set(); valid_cols = [c for c in valid_cols if not (c in seen or seen.add(c))]
            if len(valid_cols) == 0: continue
            donor_block = X_train.iloc[donor_pos].reindex(columns=valid_cols)
            for c in valid_cols: X_val_perm[c] = donor_block[c].to_numpy()
            y_perm  = model.predict(X_val_perm)
            yp_perm = model.predict_proba(X_val_perm) if hasattr(model, "predict_proba") else None
            met = get_metrics(y_eval, y_perm, yp_perm, is_classification)
            main_vals.append(met["auc"] if is_classification else met["r2"])
        main_vals = np.array(main_vals, dtype=float)
        baseline_main = base_met["auc"] if is_classification else base_met["r2"]
        perm_mean = float(np.nanmean(main_vals)) if main_vals.size else np.nan
        perm_sd   = float(np.nanstd(main_vals))  if main_vals.size else np.nan
        diff      = float(baseline_main - perm_mean) if pd.notna(baseline_main) and pd.notna(perm_mean) else np.nan
        loss_pct  = float(100 * diff / baseline_main) if pd.notna(diff) and baseline_main not in (0, None) else np.nan
        return baseline_main, perm_mean, perm_sd, diff, loss_pct

    def _train_metrics(model, Xtr, ytr, is_classification):
        yhat_tr = model.predict(Xtr)
        yproba_tr = model.predict_proba(Xtr) if hasattr(model, "predict_proba") else None
        return get_metrics(ytr, yhat_tr, yproba_tr, is_classification)

    def _compute_shap_matrix(model, X_background, X_eval, is_classification=False):
        from xgboost import XGBClassifier, XGBRegressor
        is_xgb = isinstance(model, (XGBClassifier, XGBRegressor))
        if not is_xgb:
            print("[SHAP] SKIPPED for non-XGB model (returning zeros).")
            return np.zeros((len(X_eval), X_eval.shape[1]), dtype=float)
        try:
            print(f"[SHAP] Using XGBoost pred_contribs on {len(X_eval)} rows...")
            dmat = xgb.DMatrix(X_eval, feature_names=list(X_eval.columns))
            contribs = model.get_booster().predict(dmat, pred_contribs=True, approx_contribs=False)
            arr = np.asarray(contribs)[:, :-1]
            print(f"[SHAP] XGB contribs shape: {arr.shape}")
            return arr
        except Exception as e:
            print(f"[SHAP][XGB] ERROR during pred_contribs. Returning zeros. Error: {e}")
            return np.zeros((len(X_eval), X_eval.shape[1]), dtype=float)

    def observed_only_delta(model_all, model_minus_g, Xva, yva, group_cols, is_classification):
        cols = [c for c in group_cols if c in Xva.columns]
        if len(cols) == 0: return np.nan, 0.0
        mask = Xva[cols].notna().any(axis=1)
        coverage = float(mask.mean() * 100.0)
        if coverage == 0.0: return np.nan, 0.0
        Xvm = Xva.loc[mask]
        yvm = yva.loc[mask]
        yhat_all = model_all.predict(Xvm)
        yprob_all = model_all.predict_proba(Xvm) if hasattr(model_all, "predict_proba") else None
        m_all = get_metrics(yvm, yhat_all, yprob_all, is_classification)
        base_main = m_all["auc"] if is_classification else m_all["r2"]
        Xvm_minus = Xvm.drop(columns=cols, errors="ignore")
        yhat_min = model_minus_g.predict(Xvm_minus)
        yprob_min = model_minus_g.predict_proba(Xvm_minus) if hasattr(model_minus_g, "predict_proba") else None
        m_min = get_metrics(yvm, yhat_min, yprob_min, is_classification)
        minus_main = m_min["auc"] if is_classification else m_min["r2"]
        return float(base_main - minus_main), coverage

    # Shared outer-fold creation
    # ========= PRECOMPUTE OUTER SPLITS =========
    _outer_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    _indices = np.arange(len(df))
    PRECOMP_SPLITS = [(tr_idx, va_idx) for tr_idx, va_idx in _outer_kf.split(_indices)]
    SPLITS = PRECOMP_SPLITS
    print(f"[SPLITS] Precomputed {N_SPLITS} shared outer folds.")

    # Predictor preparation
    # ========= PREDICTOR SET with PRE-FILTERS =========
    all_predictors_raw = sorted({c for cols in grouped_variables.values() for c in cols if c in df.columns})
    if len(all_predictors_raw) == 0:
        raise ValueError("No predictors available in df from grouped_variables.")
    _pre_X_full = df[all_predictors_raw].copy()
    _pre_X_full = drop_constant_and_near_duplicate(_pre_X_full, corr_threshold=0.995, sample_for_corr=2000, verbose=True)
    all_predictors = list(_pre_X_full.columns)

    id_df = df.loc[:, ["HHID", "PN"]].reset_index(drop=True)

    # Main RF/XGBoost CV, feature importance, SHAP, permutation, and ablation
    # ========= CORE CV (with OOF + all analyses) =========
    def foldwise_allvars_perm_ablate_domain(
        model_class, param_grid, X, y, grouped_variables, all_predictors, is_classification,
        splits, perm_repeats=50, k_neighbors=40, seed=42, do_full=True, compute_shap=False,
        inner_n_iter=30, inner_n_splits=5, cat_to_dummies: dict | None = None,
        id_df_for_oof: pd.DataFrame | None = None
    ):
        cv_rows, perm_rows_knn, ablation_rows, domain_rows = [], [], [], []
        fi_fold_list, shap_abs_means = [], []
        fi_domain_best_rows, fi_ablation_rows = [], []
        shap_test_chunks = []
        perm_feat_all, perm_cat_all, perm_group_all = [], [], []
        abl_obs_rows = []
        oof_rows = []
        hpo_rows = []

        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            print("\n" + "="*80)
            print(f"[OUTER] Fold {fold_id}: starting (HPO inside this fold)")
            print("="*80)

            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr_raw, yva_raw = y.iloc[tr_idx], y.iloc[va_idx]
            if is_classification:
                le = LabelEncoder()
                ytr = le.fit_transform(ytr_raw)
                yva = le.transform(yva_raw)
                # ===== CLASSIFIER-ONLY: Aligned series for OOF and observed_only_delta =====
                yva_ser = pd.Series(yva, index=Xva.index)
            else:
                ytr, yva = ytr_raw, yva_raw

            print(f"[HPO] Tuning hyperparameters on OUTER-TRAIN only (fold {fold_id}) ...")
            base_params_for_search = augment_params_for_class_imbalance(
                model_class, {}, (ytr if is_classification else ytr_raw), is_classification
            )
            base_model_for_search = make_model(model_class, base_params_for_search)
            best_params, _ = tune_hyperparameters(
                base_model_for_search, param_grid, Xtr, (ytr if is_classification else ytr_raw),
                is_classification=is_classification, n_iter=inner_n_iter
            )
            final_params = {**base_params_for_search, **best_params}
            print(f"[HPO] Best params (fold {fold_id}): {best_params}")

            hpo_row = {"fold": fold_id}; hpo_row.update({f"param__{k}": v for k, v in best_params.items()})
            hpo_rows.append(hpo_row)

            base_model = fit_model(model_class, final_params, Xtr, (ytr if is_classification else ytr_raw), is_classification, n_jobs=PARALLEL_CORES)

            met_tr = _train_metrics(base_model, Xtr, (ytr if is_classification else ytr_raw), is_classification)
            yhat = base_model.predict(Xva)
            yproba = base_model.predict_proba(Xva) if hasattr(base_model, "predict_proba") else None
            m_all = get_metrics((yva if is_classification else yva_raw), yhat, yproba, is_classification)
            m_all["fold"] = fold_id
            m_all["oob"] = float(getattr(base_model, "oob_score_", np.nan))
            for k, v in met_tr.items(): m_all["train_" + k] = v
            cv_rows.append(m_all)
            print(f"[BASE] OUTER-VAL baseline: {('AUC' if is_classification else 'R2')}={m_all.get('auc' if is_classification else 'r2')}")

            # ===== OOF save (ids, true, pred, proba, residual) =====
            if id_df_for_oof is not None:
                ids_fold = id_df_for_oof.iloc[va_idx].reset_index(drop=True)
                oof_df = pd.DataFrame({
                    "HHID": ids_fold["HHID"].values,
                    "PN":   ids_fold["PN"].values,
                    "fold": fold_id,
                    "y_true": (yva_ser.reset_index(drop=True).values if is_classification
                               else yva_raw.reset_index(drop=True).values),
                    "y_pred": pd.Series(yhat).values
                })
                if yproba is not None:
                    oof_df["y_proba_pos"] = (yproba[:, 1] if yproba.ndim == 2 and yproba.shape[1] >= 2 else np.nan)
                try:
                    oof_df["residual"] = oof_df["y_true"].astype(float) - oof_df["y_pred"].astype(float)
                except Exception:
                    oof_df["residual"] = np.nan
                oof_rows.append(oof_df)

            if hasattr(base_model, "feature_importances_"):
                fi_fold_list.append(base_model.feature_importances_.copy())

            if compute_shap:
                print(f"[SHAP] Computing SHAP on OUTER-VAL (fold {fold_id}) ...")
                bg = Xtr if len(Xtr) <= 5000 else Xtr.sample(5000, random_state=0)
                shap_mat = _compute_shap_matrix(base_model, bg, Xva, is_classification=is_classification)
                shap_abs_means.append(np.nanmean(np.abs(shap_mat), axis=0))
                shap_wide_fold = pd.DataFrame(shap_mat, columns=all_predictors)
                shap_wide_fold.insert(0, "row_idx", va_idx)
                shap_wide_fold.insert(1, "fold", fold_id)
                shap_test_chunks.append(shap_wide_fold)
            else:
                print("[SHAP] Skipped (compute_shap=False).")

            print("[PERM] Running standard permutations on OUTER-VAL ...")
            pf = permutation_featurewise_standard(base_model, Xva, (yva if is_classification else yva_raw),
                                                  is_classification, n_repeats=10, seed=seed)
            pf["fold"] = fold_id; perm_feat_all.append(pf)

            pc = permutation_grouped_categorical_standard(base_model, Xva, (yva if is_classification else yva_raw),
                                                          is_classification, cat_to_dummies or {}, n_repeats=10, seed=seed)
            if not pc.empty: pc["fold"] = fold_id; perm_cat_all.append(pc)

            pg = permutation_group_standard(base_model, Xva, (yva if is_classification else yva_raw),
                                            is_classification, grouped_variables, n_repeats=10, seed=seed)
            pg["fold"] = fold_id; perm_group_all.append(pg)

            print("[PERM] Running KNN-conditional permutation by conceptual group ...")
            for grp_name, cols in grouped_variables.items():
                grp_cols = [c for c in cols if c in all_predictors]
                if not grp_cols: continue
                base, perm_mean, perm_sd, diff, loss_pct = permutation_knn(
                    base_model, Xtr, Xva,
                    (yva if is_classification else yva_raw),
                    group_cols=grp_cols, is_classification=is_classification,
                    n_repeats=perm_repeats, k_neighbors=k_neighbors, seed=seed
                )
                perm_rows_knn.append({
                    "fold": fold_id, "group": grp_name,
                    "baseline": base, "perm_mean": perm_mean, "perm_sd": perm_sd,
                    "difference": diff, "loss_pct": loss_pct
                })

            if not do_full:
                print("[CFG] do_full=False → skipping ablation and domain-only for this target.")
                continue

            for grp_name, cols in grouped_variables.items():
                grp_cols = [c for c in cols if c in all_predictors]
                if not grp_cols: continue

                Xtr_abl = Xtr.drop(columns=grp_cols, errors="ignore")
                Xva_abl = Xva.drop(columns=grp_cols, errors="ignore")
                abl_model = fit_model(model_class, final_params, Xtr_abl, (ytr if is_classification else ytr_raw), is_classification, n_jobs=PARALLEL_CORES)
                yhat_abl = abl_model.predict(Xva_abl)
                yproba_abl = abl_model.predict_proba(Xva_abl) if hasattr(abl_model, "predict_proba") else None
                m_abl = get_metrics((yva if is_classification else yva_raw), yhat_abl, yproba_abl, is_classification)
                m_abl.update({"fold": fold_id, "group": grp_name})
                m_abl["oob"] = float(getattr(abl_model, "oob_score_", np.nan))
                met_tr_abl = _train_metrics(abl_model, Xtr_abl, (ytr if is_classification else ytr_raw), is_classification)
                for k, v in met_tr_abl.items(): m_abl["train_" + k] = v
                ablation_rows.append(m_abl)

                if hasattr(abl_model, "feature_importances_"):
                    kept_cols = list(Xtr_abl.columns)
                    for feat, imp in zip(kept_cols, abl_model.feature_importances_):
                        fi_ablation_rows.append({
                            "fold": fold_id, "removed_group": grp_name, "variable": feat, "importance": imp
                        })

                # --- OBSERVED-ONLY Δ + coverage (series only in classifier) ---
                delta_obs, coverage_val = observed_only_delta(
                    base_model, abl_model, Xva,
                    (yva_ser if is_classification else yva_raw),
                    grp_cols, is_classification
                )
                abl_obs_rows.append({"fold": fold_id, "group": grp_name,
                                     "delta_observed_only": delta_obs,
                                     "coverage_val_pct": coverage_val})

                Xtr_dom, Xva_dom = Xtr[grp_cols].copy(), Xva[grp_cols].copy()
                dom_model = fit_model(model_class, final_params, Xtr_dom, (ytr if is_classification else ytr_raw), is_classification, n_jobs=PARALLEL_CORES)
                yhat_dom = dom_model.predict(Xva_dom)
                yproba_dom = dom_model.predict_proba(Xva_dom) if hasattr(dom_model, "predict_proba") else None
                m_dom = get_metrics((yva if is_classification else yva_raw), yhat_dom, yproba_dom, is_classification)
                m_dom.update({"fold": fold_id, "group": grp_name})
                m_dom["oob"] = float(getattr(dom_model, "oob_score_", np.nan))
                met_tr_dom = _train_metrics(dom_model, Xtr_dom, (ytr if is_classification else ytr_raw), is_classification)
                for k, v in met_tr_dom.items(): m_dom["train_" + k] = v
                domain_rows.append(m_dom)

                if hasattr(dom_model, "feature_importances_"):
                    for feat, imp in zip(grp_cols, dom_model.feature_importances_):
                        fi_domain_best_rows.append({
                            "fold": fold_id, "group": grp_name, "variable": feat, "importance": imp
                        })

        fi_all_avg = np.mean(np.vstack(fi_fold_list), axis=0) if fi_fold_list else None
        shap_all_avg = np.mean(np.vstack(shap_abs_means), axis=0) if shap_abs_means else None
        shap_test_df = pd.concat(shap_test_chunks, ignore_index=True) if shap_test_chunks else pd.DataFrame()
        oof_all = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()

        return (
            pd.DataFrame(cv_rows),
            fi_all_avg,
            pd.DataFrame(perm_rows_knn),
            pd.DataFrame(ablation_rows),
            pd.DataFrame(domain_rows),
            fi_domain_best_rows,
            fi_ablation_rows,
            shap_all_avg,
            shap_test_df,
            (pd.concat(perm_feat_all, ignore_index=True) if perm_feat_all else pd.DataFrame()),
            (pd.concat(perm_cat_all,  ignore_index=True) if perm_cat_all  else pd.DataFrame()),
            (pd.concat(perm_group_all,ignore_index=True) if perm_group_all else pd.DataFrame()),
            pd.DataFrame(abl_obs_rows),
            oof_all,
            pd.DataFrame(hpo_rows)
        )

    def run_stratified_models_full(
        model_class,
        params_by_fold,
        X,
        y,
        stratify_col,
        is_classification,
        all_predictors,
        grouped_variables,
        splits,
        compute_shap=False,
        perm_repeats=50,
        k_neighbors=40,
        seed=SEED,
        enable_stratified=True
    ):
        """
        Estratificado con TODO:
          - Modelo ALL-VARS en cada estrato usando params_by_fold
          - Performance por estrato
          - Ablation por dominio dentro del estrato
          - Domain-only por dominio dentro del estrato
          - Permutación KNN por dominio dentro del estrato
          - SHAP (solo si compute_shap=True y modelo XGB)
        Devuelve:
          strat_perf_rows: lista de dicts con performance ALL-VARS por estrato
          strat_avg_fi_rows: lista de dicts con FI medio por estrato (ALL-VARS)
          strat_perm_knn_df: DataFrame con permutación KNN agregada (stratum, group)
          strat_abl_agg_df: DataFrame con ablation agregada (stratum, group)
          strat_dom_agg_df: DataFrame con domain-only agregada (stratum, group)
          strat_shap_avg_rows: lista de dicts con mean_abs_SHAP por (stratum, variable)
          strat_shap_test_df: SHAP por fila de validación (row_idx, fold, stratum, features)
        """
        if not enable_stratified:
            return [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame()

        strat_perf_rows = []
        strat_avg_fi_rows = []
        strat_perm_knn_list = []
        strat_abl_agg_list = []
        strat_dom_agg_list = []
        strat_shap_avg_rows = []
        all_shap_chunks = []

        for value in X[stratify_col].dropna().unique():
            stratum_name = f"{stratify_col}_{value}"
            mask = X[stratify_col] == value

            X_sub_full = X.loc[mask, all_predictors].copy()
            y_sub_full = y[mask]

            if len(y_sub_full) < 10:
                print(f"[STRATIFIED] {stratum_name}: demasiado pequeño (n={len(y_sub_full)}), se salta.")
                continue

            print(f"[STRATIFIED] Full análisis para estrato {stratum_name} (n={len(y_sub_full)})")

            sub_index = X_sub_full.index

            fi_fold_list = []
            cv_rows = []
            oob_list = []

            perm_knn_rows = []
            abl_rows = []
            dom_rows = []

            # for SHAP in this stratum
            shap_abs_folds = []
            shap_chunks_stratum = []

            for fold_id, (tr_idx, va_idx) in enumerate(splits):
                tr_mask = sub_index.isin(tr_idx)
                va_mask = sub_index.isin(va_idx)

                if tr_mask.sum() == 0 or va_mask.sum() == 0:
                    continue

                Xtr = X_sub_full.loc[tr_mask]
                Xva = X_sub_full.loc[va_mask]
                ytr_raw = y_sub_full.loc[tr_mask]
                yva_raw = y_sub_full.loc[va_mask]

                if is_classification:
                    le = LabelEncoder()
                    ytr = le.fit_transform(ytr_raw)
                    yva = le.transform(yva_raw)
                else:
                    ytr, yva = ytr_raw, yva_raw

                best_params_fold = params_by_fold.get(fold_id, {})
                imb = augment_params_for_class_imbalance(
                    model_class, {}, (ytr if is_classification else ytr_raw), is_classification
                )
                this_params = {**imb, **best_params_fold}

                model = fit_model(
                    model_class, this_params, Xtr,
                    (ytr if is_classification else ytr_raw),
                    is_classification, n_jobs=PARALLEL_CORES
                )

                # train + validation metrics (ALL-VARS)
                met_tr = _train_metrics(
                    model, Xtr,
                    (ytr if is_classification else ytr_raw),
                    is_classification
                )
                yhat = model.predict(Xva)
                yproba = model.predict_proba(Xva) if hasattr(model, "predict_proba") else None
                met_va = get_metrics(
                    (yva if is_classification else yva_raw),
                    yhat, yproba, is_classification
                )
                met_va.update({
                    "fold": fold_id,
                    "stratum": stratum_name,
                    "oob": float(getattr(model, "oob_score_", np.nan))
                })
                for k, v in met_tr.items():
                    met_va["train_" + k] = v

                cv_rows.append(met_va)
                oob_list.append(met_va["oob"])

                if hasattr(model, "feature_importances_"):
                    fi_fold_list.append(model.feature_importances_.copy())

                # ---------- SHAP within stratum ----------
                if compute_shap:
                    print(f"[SHAP|STRAT] {stratum_name} fold {fold_id} ...")
                    bg = Xtr if len(Xtr) <= 5000 else Xtr.sample(5000, random_state=0)
                    shap_mat = _compute_shap_matrix(model, bg, Xva, is_classification=is_classification)
                    shap_abs_folds.append(np.nanmean(np.abs(shap_mat), axis=0))

                    shap_wide_fold = pd.DataFrame(shap_mat, columns=all_predictors)
                    shap_wide_fold.insert(0, "row_idx", Xva.index.values)
                    shap_wide_fold.insert(1, "fold", fold_id)
                    shap_wide_fold.insert(2, "stratum", stratum_name)
                    shap_chunks_stratum.append(shap_wide_fold)

                # ---------- DOMAIN-WISE KNN PERMUTATION (within stratum) ----------
                for grp_name, cols in grouped_variables.items():
                    grp_cols = [c for c in cols if c in X_sub_full.columns]
                    if not grp_cols:
                        continue
                    base, perm_mean, perm_sd, diff, loss_pct = permutation_knn(
                        model,
                        X_train=Xtr,
                        X_val=Xva,
                        y_val=(yva if is_classification else yva_raw),
                        group_cols=grp_cols,
                        is_classification=is_classification,
                        n_repeats=perm_repeats,
                        k_neighbors=k_neighbors,
                        seed=seed
                    )
                    perm_knn_rows.append({
                        "fold": fold_id,
                        "stratum": stratum_name,
                        "group": grp_name,
                        "baseline": base,
                        "perm_mean": perm_mean,
                        "perm_sd": perm_sd,
                        "difference": diff,
                        "loss_pct": loss_pct,
                    })

                # ---------- DOMAIN-WISE ABLATION AND DOMAIN-ONLY (within stratum) ----------
                for grp_name, cols in grouped_variables.items():
                    grp_cols = [c for c in cols if c in X_sub_full.columns]
                    if not grp_cols:
                        continue

                    # ABLATION: remove columns from the domain
                    Xtr_abl = Xtr.drop(columns=grp_cols, errors="ignore")
                    Xva_abl = Xva.drop(columns=grp_cols, errors="ignore")
                    abl_model = fit_model(
                        model_class, this_params,
                        Xtr_abl,
                        (ytr if is_classification else ytr_raw),
                        is_classification, n_jobs=PARALLEL_CORES
                    )
                    yhat_abl = abl_model.predict(Xva_abl)
                    yproba_abl = abl_model.predict_proba(Xva_abl) if hasattr(abl_model, "predict_proba") else None
                    met_tr_abl = _train_metrics(
                        abl_model, Xtr_abl,
                        (ytr if is_classification else ytr_raw),
                        is_classification
                    )
                    m_abl = get_metrics(
                        (yva if is_classification else yva_raw),
                        yhat_abl, yproba_abl, is_classification
                    )
                    m_abl.update({
                        "fold": fold_id,
                        "stratum": stratum_name,
                        "group": grp_name,
                        "oob": float(getattr(abl_model, "oob_score_", np.nan))
                    })
                    for k, v in met_tr_abl.items():
                        m_abl["train_" + k] = v
                    abl_rows.append(m_abl)

                    # DOMAIN-ONLY: domain columns only
                    Xtr_dom = Xtr[grp_cols].copy()
                    Xva_dom = Xva[grp_cols].copy()
                    dom_model = fit_model(
                        model_class, this_params,
                        Xtr_dom,
                        (ytr if is_classification else ytr_raw),
                        is_classification, n_jobs=PARALLEL_CORES
                    )
                    yhat_dom = dom_model.predict(Xva_dom)
                    yproba_dom = dom_model.predict_proba(Xva_dom) if hasattr(dom_model, "predict_proba") else None
                    met_tr_dom = _train_metrics(
                        dom_model, Xtr_dom,
                        (ytr if is_classification else ytr_raw),
                        is_classification
                    )
                    m_dom = get_metrics(
                        (yva if is_classification else yva_raw),
                        yhat_dom, yproba_dom, is_classification
                    )
                    m_dom.update({
                        "fold": fold_id,
                        "stratum": stratum_name,
                        "group": grp_name,
                        "oob": float(getattr(dom_model, "oob_score_", np.nan))
                    })
                    for k, v in met_tr_dom.items():
                        m_dom["train_" + k] = v
                    dom_rows.append(m_dom)

            # ====== Aggregates WITHIN THAT STRATUM ======
            if not cv_rows:
                continue

            folds_df = pd.DataFrame(cv_rows)

            # --- ALL-VARS performance by stratum ---
            perf = {
                "group": stratum_name,
                "mean_r2":   folds_df["r2"].mean()   if ("r2" in folds_df) else np.nan,
                "sd_r2":     folds_df["r2"].std()    if ("r2" in folds_df) else np.nan,
                "mean_auc":  folds_df["auc"].mean()  if ("auc" in folds_df) else np.nan,
                "sd_auc":    folds_df["auc"].std()   if ("auc" in folds_df) else np.nan,
                "mean_rmse": folds_df["rmse"].mean() if ("rmse" in folds_df) else np.nan,
                "sd_rmse":   folds_df["rmse"].std()  if ("rmse" in folds_df) else np.nan,
                "mean_mae":  folds_df["mae"].mean()  if ("mae" in folds_df) else np.nan,
                "sd_mae":    folds_df["mae"].std()   if ("mae" in folds_df) else np.nan,
                "mean_oob":  np.nanmean(oob_list) if len(oob_list) else np.nan,
                "mean_accuracy": folds_df["accuracy"].mean() if ("accuracy" in folds_df) else np.nan,
                "sd_accuracy":   folds_df["accuracy"].std()  if ("accuracy" in folds_df) else np.nan,
                "mean_f1":       folds_df["f1"].mean()       if ("f1" in folds_df) else np.nan,
                "sd_f1":         folds_df["f1"].std()        if ("f1" in folds_df) else np.nan,
                "mean_precision": folds_df["precision"].mean() if ("precision" in folds_df) else np.nan,
                "sd_precision":   folds_df["precision"].std()  if ("precision" in folds_df) else np.nan,
                "mean_recall":    folds_df["recall"].mean()    if ("recall" in folds_df) else np.nan,
                "sd_recall":      folds_df["recall"].std()     if ("recall" in folds_df) else np.nan,
                "mean_bal_accuracy": folds_df["balanced_accuracy"].mean() if ("balanced_accuracy" in folds_df) else np.nan,
                "sd_bal_accuracy":   folds_df["balanced_accuracy"].std()  if ("balanced_accuracy" in folds_df) else np.nan,
            }
            strat_perf_rows.append(perf)

            # --- Average feature importance in the stratum ---
            if len(fi_fold_list) > 0:
                fi_avg = np.mean(np.vstack(fi_fold_list), axis=0)
                for feat, imp in zip(all_predictors, fi_avg):
                    strat_avg_fi_rows.append({
                        "source": stratum_name,
                        "variable": feat,
                        "importance": imp
                    })

            # --- Average SHAP and test values in the stratum ---
            if compute_shap and shap_abs_folds:
                arr = np.mean(np.vstack(shap_abs_folds), axis=0)
                for feat, val in zip(all_predictors, arr):
                    strat_shap_avg_rows.append({
                        "source": stratum_name,
                        "variable": feat,
                        "mean_abs_shap": float(val)
                    })
            if compute_shap and shap_chunks_stratum:
                all_shap_chunks.append(pd.concat(shap_chunks_stratum, ignore_index=True))

            # --- KNN permutation aggregated by domain in the stratum ---
            if perm_knn_rows:
                perm_knn_df = pd.DataFrame(perm_knn_rows)
                perm_agg = (
                    perm_knn_df
                    .groupby(["stratum", "group"], as_index=False)
                    .agg(
                        baseline_mean=("baseline", "mean"),
                        baseline_sd=("baseline", "std"),
                        permutation_mean=("perm_mean", "mean"),
                        permutation_sd=("perm_mean", "std"),
                        difference=("difference", "mean"),
                        loss_pct=("loss_pct", "mean")
                    )
                )
                strat_perm_knn_list.append(perm_agg)

            # --- Ablation aggregated by domain in the stratum ---
            if abl_rows:
                abl_df = pd.DataFrame(abl_rows)
                grp = abl_df.groupby(["stratum", "group"], as_index=False)

                if is_classification:
                    abl_agg = grp.agg(
                        ablation_mean_AUC=("auc", "mean"),
                        ablation_sd_AUC=("auc", "std"),
                        ablation_mean_accuracy=("accuracy", "mean"),
                        ablation_sd_accuracy=("accuracy", "std"),
                        ablation_mean_f1=("f1", "mean"),
                        ablation_sd_f1=("f1", "std"),
                    )
                    base_auc = folds_df["auc"].mean() if "auc" in folds_df else np.nan
                    abl_agg["baseline_mean_AUC"] = base_auc
                    abl_agg["baseline_sd_AUC"] = folds_df["auc"].std() if "auc" in folds_df else np.nan
                    abl_agg["difference"] = abl_agg["baseline_mean_AUC"] - abl_agg["ablation_mean_AUC"]
                    abl_agg["loss_pct"] = np.where(
                        abl_agg["baseline_mean_AUC"] != 0,
                        100.0 * abl_agg["difference"] / abl_agg["baseline_mean_AUC"],
                        np.nan
                    )
                else:
                    abl_agg = grp.agg(
                        ablation_mean_R2=("r2", "mean"),
                        ablation_sd_R2=("r2", "std"),
                        ablation_mean_RMSE=("rmse", "mean"),
                        ablation_sd_RMSE=("rmse", "std"),
                        ablation_mean_MAE=("mae", "mean"),
                        ablation_sd_MAE=("mae", "std"),
                    )
                    base_r2 = folds_df["r2"].mean() if "r2" in folds_df else np.nan
                    base_r2_sd = folds_df["r2"].std() if "r2" in folds_df else np.nan
                    abl_agg["baseline_mean_R2"] = base_r2
                    abl_agg["baseline_sd_R2"] = base_r2_sd
                    abl_agg["differenceR2"] = abl_agg["baseline_mean_R2"] - abl_agg["ablation_mean_R2"]
                    abl_agg["%lossR2"] = np.where(
                        abl_agg["baseline_mean_R2"] != 0,
                        100.0 * abl_agg["differenceR2"] / abl_agg["baseline_mean_R2"],
                        np.nan
                    )
                strat_abl_agg_list.append(abl_agg)

            # --- Domain-only aggregated by domain in the stratum ---
            if dom_rows:
                dom_df = pd.DataFrame(dom_rows)
                grp = dom_df.groupby(["stratum", "group"], as_index=False)
                if is_classification:
                    dom_agg = grp.agg(
                        mean_auc=("auc", "mean"),
                        sd_auc=("auc", "std"),
                        mean_accuracy=("accuracy", "mean"),
                        sd_accuracy=("accuracy", "std"),
                        mean_f1=("f1", "mean"),
                        sd_f1=("f1", "std")
                    )
                else:
                    dom_agg = grp.agg(
                        mean_r2=("r2", "mean"),
                        sd_r2=("r2", "std"),
                        mean_rmse=("rmse", "mean"),
                        sd_rmse=("rmse", "std"),
                        mean_mae=("mae", "mean"),
                        sd_mae=("mae", "std")
                    )
                strat_dom_agg_list.append(dom_agg)

        strat_perm_knn_df = (
            pd.concat(strat_perm_knn_list, ignore_index=True)
            if strat_perm_knn_list else pd.DataFrame()
        )
        strat_abl_agg_df = (
            pd.concat(strat_abl_agg_list, ignore_index=True)
            if strat_abl_agg_list else pd.DataFrame()
        )
        strat_dom_agg_df = (
            pd.concat(strat_dom_agg_list, ignore_index=True)
            if strat_dom_agg_list else pd.DataFrame()
        )
        strat_shap_test_df = (
            pd.concat(all_shap_chunks, ignore_index=True)
            if all_shap_chunks else pd.DataFrame()
        )

        return (
            strat_perf_rows,
            strat_avg_fi_rows,
            strat_perm_knn_df,
            strat_abl_agg_df,
            strat_dom_agg_df,
            strat_shap_avg_rows,
            strat_shap_test_df
        )



    def hpo_df_to_params_by_fold(hpo_df: pd.DataFrame) -> dict[int, dict]:
        def _as_none(x):
            if x is None: return None
            if isinstance(x, str) and x.strip().lower() in {"none","nan","na",""}: return None
            try: return None if pd.isna(x) else x
            except Exception: return x
        params_by_fold = {}
        if hpo_df is None or hpo_df.empty: return params_by_fold
        for _, row in hpo_df.iterrows():
            fold = int(row["fold"]); p = {}
            for c in row.index:
                if c.startswith("param__"):
                    key = c.replace("param__", "")
                    p[key] = _as_none(row[c])
            params_by_fold[fold] = p
        return params_by_fold

    def compute_learning_curves(model_class, params_by_fold, X, y, is_classification, splits,
                                      train_sizes=np.linspace(0.1, 1.0, 8), seed=SEED):
        rng = np.random.default_rng(seed)
        results = []
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            Xtr_full, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr_full, yva_raw = y.iloc[tr_idx], y.iloc[va_idx]
            if is_classification:
                le = LabelEncoder()
                ytr_full_enc = pd.Series(le.fit_transform(ytr_full))
                yva = le.transform(yva_raw)
            else:
                ytr_full_enc = pd.Series(ytr_full)
                yva = yva_raw
            best_params_fold = params_by_fold.get(fold_id, {})
            n_tr = len(Xtr_full)
            order = rng.permutation(n_tr)
            Xtr_full = Xtr_full.iloc[order].reset_index(drop=True)
            ytr_full_enc = ytr_full_enc.iloc[order].reset_index(drop=True)
            for frac in train_sizes:
                k = max(10, int(np.ceil(frac * n_tr)))
                Xtr = Xtr_full.iloc[:k]
                ytr = ytr_full_enc.iloc[:k]
                m = fit_model(model_class, best_params_fold, Xtr, ytr, is_classification, n_jobs=PARALLEL_CORES)
                yhat_tr = m.predict(Xtr)
                yproba_tr = m.predict_proba(Xtr) if hasattr(m, "predict_proba") else None
                met_tr = get_metrics(ytr, yhat_tr, yproba_tr, is_classification)
                train_main = met_tr["auc"] if is_classification else met_tr["r2"]
                yhat_va = m.predict(Xva)
                yproba_va = m.predict_proba(Xva) if hasattr(m, "predict_proba") else None
                met_va = get_metrics(yva, yhat_va, yproba_va, is_classification)
                val_main = met_va["auc"] if is_classification else met_va["r2"]
                results.append({
                    "fold": fold_id, "train_size_frac": float(frac), "train_n": int(k),
                    "train_score": float(train_main), "val_score": float(val_main),
                })
        res_df = pd.DataFrame(results)
        if res_df.empty:
            return res_df, pd.DataFrame()
        agg = (res_df.groupby("train_size_frac", as_index=False)
                      .agg(train_score_mean=("train_score","mean"),
                           train_score_sd=("train_score","std"),
                           val_score_mean=("val_score","mean"),
                           val_score_sd=("val_score","std"),
                           train_n_mean=("train_n","mean")))
        return res_df, agg

    # Main RF/XGBoost model loop, stratification, learning curves, and output writing
    # ========= MAIN ANALYSIS LOOP =========
    for alg_name, model_class, param_grid in [
        ("XGB", XGBRegressor,           xgb_param_grid_strict),
        ("RF",  RandomForestRegressor,  rf_param_grid_strict),
        ("RF",  RandomForestClassifier, rf_param_grid_strict),
        ("XGB", XGBClassifier,          xgb_param_grid_strict)
    ]:
        for outcome_type, targets in epigenetic_clocks.items():
            for target in targets:
                is_classification = (outcome_type == "BINARY")
                base_target_name = target.replace("_BINARY", "")
                is_limited_clock = base_target_name in limited_clocks
                wants_shap_clock = base_target_name in SHAP_TARGETS
                is_xgb_model = ("XGB" in model_class.__name__)
                compute_shap = wants_shap_clock and is_xgb_model

                if is_classification and "Regressor" in model_class.__name__: continue
                if (not is_classification) and "Classifier" in model_class.__name__: continue
                if target not in df.columns:
                    print(f"WARNING: target '{target}' does not exist. Skipping.")
                    continue

                save_dir = os.path.join(outdir, alg_name, ("Classifier" if is_classification else "Regressor"))
                os.makedirs(save_dir, exist_ok=True)

                X = df[all_predictors].copy().reset_index(drop=True)
                y = df[target].copy().reset_index(drop=True)
                print("\n=== Starting {} {} on target {} ({}) ===".format(
                    alg_name, model_class.__name__, target, "classification" if is_classification else "regression"))
                print(f"[DATA] X shape: {X.shape}, y length: {len(y)}, predictors: {len(all_predictors)}")
                print(f"[CFG] limited_clock={is_limited_clock}, wants_shap_clock={wants_shap_clock}, model_is_xgb={is_xgb_model}, compute_shap={compute_shap}")

                is_dun = is_dunedin(target)
                is_rf  = "RandomForest" in model_class.__name__
                is_xgb = "XGB" in model_class.__name__

                param_grid_use = (
                    rf_param_grid_poam if (is_dun and is_rf) else
                    xgb_param_grid_poam if (is_dun and is_xgb) else
                    param_grid
                )

                (
                    cv_allvars_df, fi_all_avg, perm_knn_df, ablation_df, domain_df,
                    fi_domain_best_rows_fold, fi_ablation_rows_fold, shap_all_avg,
                    shap_test_df, perm_feat_std_df, perm_cat_std_df, perm_group_std_df,
                    ablation_observed_df, oof_all_df, hpo_per_fold_df
                ) = foldwise_allvars_perm_ablate_domain(
                    model_class, param_grid_use, X, y, grouped_variables, all_predictors,
                    is_classification, splits=SPLITS, perm_repeats=50, k_neighbors=40,
                    seed=SEED, do_full=(not is_limited_clock), compute_shap=compute_shap,
                    inner_n_iter=30, inner_n_splits=5, cat_to_dummies=dummy_cols, id_df_for_oof=id_df
                )

                print(f"[{target}] Finished outer CV. (do_full={not is_limited_clock}, SHAP={'on' if compute_shap else 'off'})")
                params_by_fold = hpo_df_to_params_by_fold(hpo_per_fold_df)

                if not hpo_per_fold_df.empty:
                    hpo_per_fold_df.insert(0, "clock", target)
                    hpo_per_fold_df.insert(1, "model_class", model_class.__name__)
                    hpo_per_fold_path = os.path.join(save_dir, "best_hyperparameters_per_fold.csv")
                    hpo_per_fold_df.to_csv(
                        hpo_per_fold_path, mode="a",
                        header=not os.path.exists(hpo_per_fold_path), index=False
                    )
                    print(f"[{target}] Per-fold HPO appended to {hpo_per_fold_path}")
                
                if not ablation_observed_df.empty:
                    obs_path = os.path.join(save_dir, "ablation_observed_delta.csv")
                    tmp = ablation_observed_df.copy()
                    tmp["clock"] = target
                    tmp.to_csv(obs_path, mode="a",
                               header=not os.path.exists(obs_path), index=False)
                    print(f"[{target}] Observed-only Δ appended to {obs_path}")

                if not oof_all_df.empty:
                    oof_path = os.path.join(save_dir, "oof_predictions.csv")
                    tmp = oof_all_df.copy()
                    tmp["clock"] = target
                    tmp["is_classification"] = int(is_classification)
                    tmp.to_csv(oof_path, mode="a", header=not os.path.exists(oof_path), index=False)
                    print(f"[{target}] OOF predictions appended to {oof_path}")

                    if is_classification:
                        if "y_proba_pos" in tmp.columns and tmp["y_proba_pos"].notna().any():
                            try:
                                fpr, tpr, thr = roc_curve(tmp["y_true"].astype(int), tmp["y_proba_pos"])
                                roc_df = pd.DataFrame({"fpr":fpr, "tpr":tpr, "threshold":thr})
                                roc_df["clock"] = target
                                roc_df.to_csv(os.path.join(save_dir, "roc_curve_points.csv"),
                                              mode="a", header=not os.path.exists(os.path.join(save_dir, "roc_curve_points.csv")), index=False)
                            except Exception as e:
                                print(f"[{target}] ROC curve failed: {e}")
                            try:
                                prec, rec, _ = precision_recall_curve(tmp["y_true"].astype(int), tmp["y_proba_pos"])
                                pr_df = pd.DataFrame({"precision":prec, "recall":rec})
                                pr_df["clock"] = target
                                pr_df.to_csv(os.path.join(save_dir, "pr_curve_points.csv"),
                                             mode="a", header=not os.path.exists(os.path.join(save_dir, "pr_curve_points.csv")), index=False)
                            except Exception as e:
                                print(f"[{target}] PR curve failed: {e}")
                            try:
                                bins = pd.qcut(tmp["y_proba_pos"], q=10, duplicates="drop")
                                calib = tmp.groupby(bins, observed=True).agg(
                                    proba_mean=("y_proba_pos","mean"),
                                    y_rate=("y_true","mean"),
                                    n=("y_true","size")
                                ).reset_index(drop=True)
                                calib["clock"] = target
                                calib.to_csv(os.path.join(save_dir, "calibration_bins.csv"),
                                             mode="a", header=not os.path.exists(os.path.join(save_dir, "calibration_bins.csv")), index=False)
                                brier = brier_score_loss(tmp["y_true"].astype(int), tmp["y_proba_pos"])
                                pd.DataFrame([{"clock": target, "brier_score": float(brier)}]).to_csv(
                                    os.path.join(save_dir, "brier_scores.csv"),
                                    mode="a", header=not os.path.exists(os.path.join(save_dir, "brier_scores.csv")), index=False
                                )
                            except Exception as e:
                                print(f"[{target}] Calibration/Brier failed: {e}")

                shap_vec = None
                if compute_shap and (not shap_test_df.empty):
                    ids_fold = id_df.iloc[shap_test_df["row_idx"].values].reset_index(drop=True)
                    shap_wide_all = pd.concat([ids_fold, shap_test_df.drop(columns=["row_idx", "fold"])], axis=1)
                    shap_wide_all = shap_wide_all.drop_duplicates(subset=["HHID", "PN"], keep="first")
                    shap_wide_path = os.path.join(save_dir, f"shap_wide_{target}.csv")
                    shap_wide_all.to_csv(shap_wide_path, index=False)
                    value_cols = [c for c in shap_wide_all.columns if c not in ["HHID", "PN"]]
                    shap_long_all = shap_wide_all.melt(id_vars=["HHID", "PN"], value_vars=value_cols,
                                                       var_name="variable", value_name="shap_value")
                    shap_long_all["clock"] = target
                    shap_long_all["source"] = "ALL"
                    shap_long_path = os.path.join(save_dir, f"shap_long_{target}.csv")
                    shap_long_all.to_csv(shap_long_path, index=False)
                    shap_vec_map = (
                        shap_long_all.groupby("variable", as_index=False)["shap_value"]
                        .apply(lambda s: np.mean(np.abs(s)))
                        .rename(columns={"shap_value": "mean_abs_shap"})
                    )
                    shap_vec = [np.nan] * len(all_predictors)
                    var_to_meanabs = dict(zip(shap_vec_map["variable"], shap_vec_map["mean_abs_shap"]))
                    for j, feat in enumerate(all_predictors):
                        if feat in var_to_meanabs:
                            shap_vec[j] = float(var_to_meanabs[feat])
                    print(f"[{target}] SHAP saved to:\n  - {shap_wide_path}\n  - {shap_long_path}")

                row_all = {
                    "clock": target,
                    "model": "all_vars",
                    "stratum": "ALL",
                    "mean_r2":   cv_allvars_df["r2"].mean()   if ("r2" in cv_allvars_df) else np.nan,
                    "sd_r2":     cv_allvars_df["r2"].std()    if ("r2" in cv_allvars_df) else np.nan,
                    "mean_auc":  cv_allvars_df["auc"].mean()  if ("auc" in cv_allvars_df) else np.nan,
                    "sd_auc":    cv_allvars_df["auc"].std()   if ("auc" in cv_allvars_df) else np.nan,
                    "mean_rmse": cv_allvars_df["rmse"].mean() if ("rmse" in cv_allvars_df) else np.nan,
                    "sd_rmse":   cv_allvars_df["rmse"].std()  if ("rmse" in cv_allvars_df) else np.nan,
                    "mean_mae":  cv_allvars_df["mae"].mean()  if ("mae" in cv_allvars_df) else np.nan,
                    "sd_mae":    cv_allvars_df["mae"].std()   if ("mae" in cv_allvars_df) else np.nan,
                    "mean_oob":  cv_allvars_df["oob"].mean()  if "oob" in cv_allvars_df else np.nan,
                    "sd_oob":    cv_allvars_df["oob"].std()   if "oob" in cv_allvars_df else np.nan,
                    "mean_accuracy": cv_allvars_df["accuracy"].mean() if ("accuracy" in cv_allvars_df) else np.nan,
                    "sd_accuracy":   cv_allvars_df["accuracy"].std()  if ("accuracy" in cv_allvars_df) else np.nan,
                    "mean_f1":       cv_allvars_df["f1"].mean()       if ("f1" in cv_allvars_df) else np.nan,
                    "sd_f1":         cv_allvars_df["f1"].std()        if ("f1" in cv_allvars_df) else np.nan,
                    "mean_precision": cv_allvars_df["precision"].mean() if ("precision" in cv_allvars_df) else np.nan,
                    "sd_precision":   cv_allvars_df["precision"].std()  if ("precision" in cv_allvars_df) else np.nan,
                    "mean_recall":    cv_allvars_df["recall"].mean()    if ("recall" in cv_allvars_df) else np.nan,
                    "sd_recall":      cv_allvars_df["recall"].std()     if ("recall" in cv_allvars_df) else np.nan,
                    "mean_bal_accuracy": cv_allvars_df["balanced_accuracy"].mean() if ("balanced_accuracy" in cv_allvars_df) else np.nan,
                    "sd_bal_accuracy":   cv_allvars_df["balanced_accuracy"].std()  if ("balanced_accuracy" in cv_allvars_df) else np.nan,
                    "train_mean_r2":   cv_allvars_df["train_r2"].mean()   if ("train_r2" in cv_allvars_df) else np.nan,
                    "train_sd_r2":     cv_allvars_df["train_r2"].std()    if ("train_r2" in cv_allvars_df) else np.nan,
                    "train_mean_auc":  cv_allvars_df["train_auc"].mean()  if ("train_auc" in cv_allvars_df) else np.nan,
                    "train_sd_auc":    cv_allvars_df["train_auc"].std()   if ("train_auc" in cv_allvars_df) else np.nan,
                    "train_mean_rmse": cv_allvars_df["train_rmse"].mean() if ("train_rmse" in cv_allvars_df) else np.nan,
                    "train_sd_rmse":   cv_allvars_df["train_rmse"].std()  if ("train_rmse" in cv_allvars_df) else np.nan,
                    "train_mean_mae":  cv_allvars_df["train_mae"].mean()  if ("train_mae" in cv_allvars_df) else np.nan,
                    "train_sd_mae":    cv_allvars_df["train_mae"].std()   if ("train_mae" in cv_allvars_df) else np.nan,
                    "train_mean_accuracy": cv_allvars_df["train_accuracy"].mean() if ("train_accuracy" in cv_allvars_df) else np.nan,
                    "train_sd_accuracy":   cv_allvars_df["train_accuracy"].std()  if ("train_accuracy" in cv_allvars_df) else np.nan,
                    "train_mean_f1":       cv_allvars_df["train_f1"].mean()       if ("train_f1" in cv_allvars_df) else np.nan,
                    "train_sd_f1":         cv_allvars_df["train_f1"].std()        if ("train_f1" in cv_allvars_df) else np.nan,
                    "train_mean_precision": cv_allvars_df["train_precision"].mean() if ("train_precision" in cv_allvars_df) else np.nan,
                    "train_sd_precision":   cv_allvars_df["train_precision"].std()  if ("train_precision" in cv_allvars_df) else np.nan,
                    "train_mean_recall":    cv_allvars_df["train_recall"].mean()    if ("train_recall" in cv_allvars_df) else np.nan,
                    "train_sd_recall":      cv_allvars_df["train_recall"].std()     if ("train_recall" in cv_allvars_df) else np.nan,
                    "train_mean_bal_accuracy": cv_allvars_df["train_balanced_accuracy"].mean() if ("train_balanced_accuracy" in cv_allvars_df) else np.nan,
                    "train_sd_bal_accuracy":   cv_allvars_df["train_balanced_accuracy"].std()  if ("train_balanced_accuracy" in cv_allvars_df) else np.nan,
                }
                perf_path = os.path.join(save_dir, "model_performance.csv")
                pd.DataFrame([row_all]).reindex(columns=PERF_COLS).to_csv(
                    perf_path, mode="a", header=not os.path.exists(perf_path), index=False
                )
                print(f"[{target}] Performance appended to {perf_path}")

                if (base_target_name in SHAP_TARGETS) and (fi_all_avg is not None):
                    shap_vec_use = shap_vec if (compute_shap and shap_vec is not None) else (
                        shap_all_avg if (compute_shap and (shap_all_avg is not None) and (len(shap_all_avg) == len(all_predictors))) else [np.nan]*len(all_predictors)
                    )
                    rows = []
                    for feat, imp, sv in zip(all_predictors, fi_all_avg, shap_vec_use):
                        rows.append({
                            "clock": target, "source": "ALL", "variable": feat,
                            "importance": float(imp),
                            "mean_abs_shap": (float(sv) if pd.notna(sv) else np.nan),
                            "mean_r2":  row_all["mean_r2"], "sd_r2":  row_all["sd_r2"],
                            "mean_rmse":row_all["mean_rmse"], "sd_rmse":row_all["sd_rmse"],
                            "mean_mae": row_all["mean_mae"], "sd_mae": row_all["sd_mae"],
                            "mean_auc": row_all["mean_auc"], "sd_auc": row_all["sd_auc"],
                            "mean_oob": row_all["mean_oob"], "sd_oob": row_all["sd_oob"],
                        })
                    fi_avg_path = os.path.join(save_dir, "all_vars_avgimportance.csv")
                    pd.DataFrame(rows).to_csv(
                        fi_avg_path, mode="a", header=not os.path.exists(fi_avg_path), index=False
                    )
                    print(f"[{target}] Avg FI+SHAP (ALL) saved to {fi_avg_path}")
                else:
                    print(f"[{target}] Skipping FI/SHAP save (either not in SHAP_TARGETS or FI missing).")

                if not perm_knn_df.empty:
                    perm_df_agg = (
                        perm_knn_df.groupby(["group"], as_index=False)
                                   .agg(baseline_mean=("baseline", "mean"),
                                        baseline_sd=("baseline", "std"),
                                        permutation_mean=("perm_mean", "mean"),
                                        permutation_sd=("perm_mean", "std"),
                                        difference=("difference", "mean"),
                                        loss_pct=("loss_pct", "mean"))
                    )
                    perm_df_agg["clock"] = target
                    for col in ["baseline_mean_RMSE","baseline_sd_RMSE","baseline_mean_MAE","baseline_sd_MAE",
                                "permutation_mean_RMSE","permutation_sd_RMSE","permutation_mean_MAE","permutation_sd_MAE"]:
                        perm_df_agg[col] = np.nan
                    perm_path = os.path.join(save_dir, "permutation_performance_knn.csv")
                    perm_df_agg.to_csv(perm_path, mode="a", header=not os.path.exists(perm_path), index=False)
                    print(f"[{target}] KNN-conditional permutation appended to {perm_path}")

                if not perm_feat_std_df.empty:
                    pf_path = os.path.join(save_dir, "permutation_feature_standard.csv")
                    tmp = perm_feat_std_df.copy(); tmp["clock"] = target
                    tmp.to_csv(pf_path, mode="a", header=not os.path.exists(pf_path), index=False)
                    print(f"[{target}] Standard per-feature permutation appended to {pf_path}")
                if not perm_cat_std_df.empty:
                    pc_path = os.path.join(save_dir, "permutation_grouped_categorical_standard.csv")
                    tmp = perm_cat_std_df.copy(); tmp["clock"] = target
                    tmp.to_csv(pc_path, mode="a", header=not os.path.exists(pc_path), index=False)
                    print(f"[{target}] Grouped-categorical permutation appended to {pc_path}")
                if not perm_group_std_df.empty:
                    pg_path = os.path.join(save_dir, "permutation_group_standard.csv")
                    tmp = perm_group_std_df.copy(); tmp["clock"] = target
                    tmp.to_csv(pg_path, mode="a", header=not os.path.exists(pg_path), index=False)
                    print(f"[{target}] Group permutation appended to {pg_path}")

                if not ablation_df.empty:
                    abl_groups = ablation_df.groupby("group", as_index=False)
                    if is_classification:
                        abl_agg = abl_groups.agg(
                            ablation_mean_AUC=("auc", "mean"), ablation_sd_AUC=("auc", "std"),
                            ablation_mean_accuracy=("accuracy", "mean"), ablation_sd_accuracy=("accuracy", "std"),
                            ablation_mean_f1=("f1", "mean"), ablation_sd_f1=("f1", "std"),
                            ablation_mean_oob=("oob","mean"), ablation_sd_oob=("oob","std"),
                            train_mean_auc=("train_auc","mean"),   train_sd_auc=("train_auc","std"),
                            train_mean_accuracy=("train_accuracy","mean"), train_sd_accuracy=("train_accuracy","std"),
                            train_mean_f1=("train_f1","mean"),     train_sd_f1=("train_f1","std"),
                            train_mean_precision=("train_precision","mean"), train_sd_precision=("train_precision","std"),
                            train_mean_recall=("train_recall","mean"),       train_sd_recall=("train_recall","std"),
                            train_mean_bal_accuracy=("train_balanced_accuracy","mean"),
                            train_sd_bal_accuracy=("train_balanced_accuracy","std"),
                        )
                        baseline_main = cv_allvars_df["auc"]
                        abl_agg["baseline_mean_AUC"] = baseline_main.mean()
                        abl_agg["baseline_sd_AUC"]   = baseline_main.std()
                        abl_agg["difference"] = abl_agg["baseline_mean_AUC"] - abl_agg["ablation_mean_AUC"]
                        abl_agg["loss_pct"] = np.where(
                            abl_agg["baseline_mean_AUC"] != 0,
                            100.0 * abl_agg["difference"] / abl_agg["baseline_mean_AUC"], np.nan
                        )
                        abl_agg[["baseline_mean_R2","baseline_sd_R2","baseline_mean_RMSE","baseline_sd_RMSE","baseline_mean_MAE","baseline_sd_MAE"]] = np.nan
                    else:
                        abl_agg = abl_groups.agg(
                            ablation_mean_R2=("r2", "mean"), ablation_sd_R2=("r2", "std"),
                            ablation_mean_RMSE=("rmse", "mean"), ablation_sd_RMSE=("rmse", "std"),
                            ablation_mean_MAE=("mae", "mean"), ablation_sd_MAE=("mae", "std"),
                            ablation_mean_oob=("oob","mean"), ablation_sd_oob=("oob","std"),
                            train_mean_r2=("train_r2","mean"),   train_sd_r2=("train_r2","std"),
                            train_mean_rmse=("train_rmse","mean"), train_sd_rmse=("train_rmse","std"),
                            train_mean_mae=("train_mae","mean"),   train_sd_mae=("train_mae","std"),
                        )
                        abl_agg["baseline_mean_R2"] = cv_allvars_df["r2"].mean()
                        abl_agg["baseline_sd_R2"]   = cv_allvars_df["r2"].std()
                        abl_agg["differenceR2"] = abl_agg["baseline_mean_R2"] - abl_agg["ablation_mean_R2"]
                        abl_agg["%lossR2"] = np.where(
                            abl_agg["baseline_mean_R2"] != 0,
                            100.0 * abl_agg["differenceR2"] / abl_agg["baseline_mean_R2"], np.nan
                        )
                        abl_agg["baseline_mean_RMSE"] = cv_allvars_df["rmse"].mean()
                        abl_agg["baseline_sd_RMSE"]   = cv_allvars_df["rmse"].std()
                        abl_agg["baseline_mean_MAE"]  = cv_allvars_df["mae"].mean()
                        abl_agg["baseline_sd_MAE"]    = cv_allvars_df["mae"].std()
                    abl_agg["clock"] = target
                    abl_path = os.path.join(save_dir, "ablation_performance.csv")
                    abl_agg.to_csv(abl_path, mode="a", header=not os.path.exists(abl_path), index=False)
                    print(f"[{target}] Ablation (domain-drop) appended to {abl_path}")

                    if fi_ablation_rows_fold:
                        fi_abl_df = pd.DataFrame(fi_ablation_rows_fold)
                        fi_abl_avg = (fi_abl_df.groupby(["removed_group","variable"], as_index=False)
                                                 .agg(importance=("importance","mean")))
                        fi_abl_avg["clock"] = target
                        abl_metrics = abl_agg.copy().rename(columns={"group":"removed_group"})
                        abl_metrics = abl_metrics.drop(columns=["clock"], errors="ignore")
                        fi_abl_full = fi_abl_avg.merge(abl_metrics, on="removed_group", how="left")
                        keep_front = ["clock","removed_group","variable","importance"]
                        keep_extra = [c for c in fi_abl_full.columns if c not in keep_front]
                        fi_abl_out = os.path.join(save_dir, "all_vars_avg_importance_NO_SUBGROUP.csv")
                        pd.concat([fi_abl_full[keep_front], fi_abl_full[keep_extra]], axis=1).to_csv(
                            fi_abl_out, mode="a", header=not os.path.exists(fi_abl_out), index=False
                        )
                        print(f"[{target}] FI (ablation models) appended to {fi_abl_out}")

                if not domain_df.empty:
                    dom_groups = domain_df.groupby("group", as_index=False)
                    if is_classification:
                        dom_agg = dom_groups.agg(
                            mean_auc=("auc","mean"), sd_auc=("auc","std"),
                            mean_accuracy=("accuracy","mean"), sd_accuracy=("accuracy","std"),
                            mean_f1=("f1","mean"), sd_f1=("f1","std"),
                            mean_oob=("oob","mean"), sd_oob=("oob","std"),
                            train_mean_auc=("train_auc","mean"),   train_sd_auc=("train_auc","std"),
                            train_mean_accuracy=("train_accuracy","mean"), train_sd_accuracy=("train_accuracy","std"),
                            train_mean_f1=("train_f1","mean"),     train_sd_f1=("train_f1","std"),
                            train_mean_precision=("train_precision","mean"), train_sd_precision=("train_precision","std"),
                            train_mean_recall=("train_recall","mean"),       train_sd_recall=("train_recall","std"),
                            train_mean_bal_accuracy=("train_balanced_accuracy","mean"),
                            train_sd_bal_accuracy=("train_balanced_accuracy","std"),
                        )
                        for c in ["mean_r2","sd_r2","mean_rmse","sd_rmse","mean_mae","sd_mae",
                                  "train_mean_r2","train_sd_r2","train_mean_rmse","train_sd_rmse","train_mean_mae","train_sd_mae"]:
                            dom_agg[c] = np.nan
                    else:
                        dom_agg = dom_groups.agg(
                            mean_r2=("r2","mean"), sd_r2=("r2","std"),
                            mean_rmse=("rmse","mean"), sd_rmse=("rmse","std"),
                            mean_mae=("mae","mean"), sd_mae=("mae","std"),
                            mean_oob=("oob","mean"), sd_oob=("oob","std"),
                            train_mean_r2=("train_r2","mean"),   train_sd_r2=("train_r2","std"),
                            train_mean_rmse=("train_rmse","mean"), train_sd_rmse=("train_rmse","std"),
                            train_mean_mae=("train_mae","mean"),   train_sd_mae=("train_mae","std"),
                        )
                        for c in ["mean_auc","sd_auc","mean_accuracy","sd_accuracy","mean_f1","sd_f1",
                                  "train_mean_auc","train_sd_auc","train_mean_accuracy","train_sd_accuracy","train_mean_f1","train_sd_f1"]:
                            dom_agg[c] = np.nan
                    dom_agg["clock"] = target
                    dom_path = os.path.join(save_dir, "domainonly_performance.csv")
                    dom_agg.to_csv(dom_path, mode="a", header=not os.path.exists(dom_path), index=False)
                    print(f"[{target}] Domain-only metrics appended to {dom_path}")
                    if fi_domain_best_rows_fold:
                        dom_fi = pd.DataFrame(fi_domain_best_rows_fold)
                        dom_fi_avg = (dom_fi.groupby(["group","variable"], as_index=False)
                                            .agg(importance=("importance","mean")))
                        dom_metrics = dom_agg.drop(columns=["clock"], errors="ignore").copy()
                        out_rows = dom_fi_avg.merge(dom_metrics, on="group", how="left")
                        out_rows.insert(0, "clock", target)
                        out_rows.insert(1, "source", "DOMAIN_ONLY")
                        out_rows.insert(2, "subgroup", out_rows["group"])
                        out_rows = out_rows.drop(columns=["group"])
                        if "mean_abs_shap" not in out_rows.columns: out_rows["mean_abs_shap"] = np.nan
                        expected_cols = [
                            "clock","source","subgroup","variable","importance","mean_abs_shap",
                            "mean_r2","sd_r2","mean_rmse","sd_rmse","mean_mae","sd_mae",
                            "mean_auc","sd_auc","mean_oob","sd_oob",
                        ]
                        for c in expected_cols:
                            if c not in out_rows.columns: out_rows[c] = np.nan
                        out_rows = out_rows[expected_cols]
                        out_path = os.path.join(save_dir, "domainonly_avgimportance.csv")
                        out_rows.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
                        print(f"[{target}] Domain-only FI appended to {out_path}")

                # Stratified analyses
                enable_stratified = base_target_name in SHAP_TARGETS
                for strat_col, avg_name in [
                    ("GENDER","all_vars_avgimportance_GENDER.csv"),
                    ("WHITE_BINARY","all_vars_avgimportance_WHITE.csv"),
                    ("EVER_SMOKED_RAND","all_vars_avgimportance_EVER_SMOKED_RAND.csv")
                ]:
                    if strat_col in df.columns:
                        X_with_strat = X.copy()
                        X_with_strat[strat_col] = df[strat_col].values
                        print(f"[{target}] Stratified ALL-VARS + ablation/domain/permutation/SHAP by {strat_col} (enable={enable_stratified})...")

                        strat_perf, strat_avg_fi, strat_perm_knn_strat, strat_abl_agg_strat, strat_dom_agg_strat, strat_shap_avg, strat_shap_test = run_stratified_models_full(
                            model_class=model_class,
                            params_by_fold=params_by_fold,
                            X=X_with_strat,
                            y=y,
                            stratify_col=strat_col,
                            is_classification=is_classification,
                            all_predictors=all_predictors,
                            grouped_variables=grouped_variables,
                            splits=SPLITS,
                            compute_shap=compute_shap,
                            perm_repeats=50,
                            k_neighbors=40,
                            seed=SEED,
                            enable_stratified=enable_stratified
                        )

                        perf_rows = []
                        for r in strat_perf:
                            perf_rows.append({
                                "clock": target,
                                "model": "all_vars",
                                "stratum": r["group"],
                                "mean_r2":  r.get("mean_r2", np.nan),  "sd_r2":  r.get("sd_r2", np.nan),
                                "mean_auc": r.get("mean_auc", np.nan),  "sd_auc":  r.get("sd_auc", np.nan),
                                "mean_rmse": r.get("mean_rmse", np.nan),"sd_rmse": r.get("sd_rmse", np.nan),
                                "mean_mae":  r.get("mean_mae", np.nan), "sd_mae":  r.get("sd_mae", np.nan),
                                "mean_oob":  r.get("mean_oob", np.nan),
                                "mean_accuracy": r.get("mean_accuracy", np.nan),"sd_accuracy": r.get("sd_accuracy", np.nan),
                                "mean_f1": r.get("mean_f1", np.nan),"sd_f1": r.get("sd_f1", np.nan),
                                "mean_precision": r.get("mean_precision", np.nan),"sd_precision": r.get("sd_precision", np.nan),
                                "mean_recall": r.get("mean_recall", np.nan),"sd_recall": r.get("sd_recall", np.nan),
                                "mean_bal_accuracy": r.get("mean_bal_accuracy", np.nan),"sd_bal_accuracy": r.get("sd_bal_accuracy", np.nan),
                                # train_* are left as NA for strata
                                "train_mean_r2": np.nan, "train_sd_r2": np.nan,
                                "train_mean_auc": np.nan, "train_sd_auc": np.nan,
                                "train_mean_rmse": np.nan, "train_sd_rmse": np.nan,
                                "train_mean_mae": np.nan, "train_sd_mae": np.nan,
                                "train_mean_accuracy": np.nan, "train_sd_accuracy": np.nan,
                                "train_mean_f1": np.nan, "train_sd_f1": np.nan,
                                "train_mean_precision": np.nan, "train_sd_precision": np.nan,
                                "train_mean_recall": np.nan, "train_sd_recall": np.nan,
                                "train_mean_bal_accuracy": np.nan, "train_sd_bal_accuracy": np.nan,
                            })
                        if perf_rows:
                            perf_path = os.path.join(save_dir, "model_performance.csv")
                            pd.DataFrame(perf_rows).reindex(columns=PERF_COLS).to_csv(
                                perf_path, mode="a", header=not os.path.exists(perf_path), index=False
                            )
                            print(f"[{target}] Stratified performance (incl. ablation/permutation/domain-only/SHAP) appended to {perf_path}")

                        # ===== 2.2 Average FI + SHAP by stratum =====
                        if enable_stratified and strat_avg_fi:
                            df_imp = pd.DataFrame(strat_avg_fi)       # source, variable, importance
                            df_imp["clock"] = target
                            if strat_shap_avg:
                                df_shap = pd.DataFrame(strat_shap_avg)  # source, variable, mean_abs_shap
                                df_all = df_imp.merge(df_shap, on=["source", "variable"], how="left")
                            else:
                                df_all = df_imp
                            avg_path = os.path.join(save_dir, avg_name)
                            df_all.to_csv(
                                avg_path,
                                mode="a",
                                header=not os.path.exists(avg_path),
                                index=False
                            )
                            print(f"[{target}] Stratified avg FI+SHAP appended to {avg_path}")

                        # ===== 2.3 Stratified KNN permutation =====
                        if not strat_perm_knn_strat.empty:
                            tmp = strat_perm_knn_strat.copy()
                            tmp["clock"] = target
                            perm_path_strat = os.path.join(save_dir, "permutation_performance_knn_stratified.csv")
                            tmp.to_csv(
                                perm_path_strat,
                                mode="a",
                                header=not os.path.exists(perm_path_strat),
                                index=False
                            )
                            print(f"[{target}] Stratified KNN permutation appended to {perm_path_strat}")

                        # ===== 2.4 Stratified ablation =====
                        if not strat_abl_agg_strat.empty:
                            tmp = strat_abl_agg_strat.copy()
                            tmp["clock"] = target
                            abl_path_strat = os.path.join(save_dir, "ablation_performance_stratified.csv")
                            tmp.to_csv(
                                abl_path_strat,
                                mode="a",
                                header=not os.path.exists(abl_path_strat),
                                index=False
                            )
                            print(f"[{target}] Stratified ablation appended to {abl_path_strat}")

                        # ===== 2.5 Stratified domain-only =====
                        if not strat_dom_agg_strat.empty:
                            tmp = strat_dom_agg_strat.copy()
                            tmp["clock"] = target
                            dom_path_strat = os.path.join(save_dir, "domainonly_performance_stratified.csv")
                            tmp.to_csv(
                                dom_path_strat,
                                mode="a",
                                header=not os.path.exists(dom_path_strat),
                                index=False
                            )
                            print(f"[{target}] Stratified domain-only performance appended to {dom_path_strat}")

                        # ===== 2.6 Stratified SHAP: wide + long per participant =====
                        if compute_shap and (not strat_shap_test.empty):
                            tmp = strat_shap_test.copy()
                            ids = id_df.iloc[tmp["row_idx"].values].reset_index(drop=True)
                            shap_vals = tmp.drop(columns=["row_idx", "fold", "stratum"])
                            shap_wide = pd.concat(
                                [ids, tmp[["stratum"]].reset_index(drop=True), shap_vals.reset_index(drop=True)],
                                axis=1
                            )
                            shap_wide = shap_wide.drop_duplicates(subset=["HHID", "PN", "stratum"], keep="first")

                            shap_wide_path = os.path.join(save_dir, f"shap_wide_stratified_{target}_{strat_col}.csv")
                            shap_wide.to_csv(
                                shap_wide_path,
                                mode="a",
                                header=not os.path.exists(shap_wide_path),
                                index=False
                            )

                            value_cols = [c for c in shap_wide.columns if c not in ["HHID", "PN", "stratum"]]
                            shap_long = shap_wide.melt(
                                id_vars=["HHID", "PN", "stratum"],
                                value_vars=value_cols,
                                var_name="variable",
                                value_name="shap_value"
                            )
                            shap_long["clock"] = target
                            shap_long["stratify_col"] = strat_col
                            shap_long_path = os.path.join(save_dir, f"shap_long_stratified_{target}_{strat_col}.csv")
                            shap_long.to_csv(
                                shap_long_path,
                                mode="a",
                                header=not os.path.exists(shap_long_path),
                                index=False
                            )
                            print(f"[{target}] Stratified SHAP saved to {shap_wide_path} and {shap_long_path}")
            
                # Learning curves and final output writing
                lc_raw, lc_agg = compute_learning_curves(model_class, params_by_fold, X, y, is_classification, SPLITS)
                if not lc_raw.empty:
                    fn1 = os.path.join(save_dir, f"learning_curve_raw_{target}.csv")
                    lc_raw_assign = lc_raw.copy(); lc_raw_assign["clock"] = target
                    lc_raw_assign.to_csv(fn1, mode="a", header=not os.path.exists(fn1), index=False)
                    fn2 = os.path.join(save_dir, f"learning_curve_{target}.csv")
                    lc_agg_assign = lc_agg.copy(); lc_agg_assign["clock"] = target
                    lc_agg_assign.to_csv(fn2, mode="a", header=not os.path.exists(fn2), index=False)
                    print(f"[{target}] Learning curves saved to {fn1} and {fn2}")

    print("DONE!! 🎉")



if __name__ == "__main__":
    main()
