from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

from run_2013_initial_microbiome_signal_probes import (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    compute_alpha_and_genus,
    load_asv_counts,
    load_ded_2013,
    load_family_ids,
    load_mapping,
    load_taxa,
    nearest_stool_pairs,
    sample_read_depths,
    standardize,
)


TIMING_BRANCHES = {
    "within_1y": 365,
    "within_3y": 365 * 3,
}
BINARY_OUTCOMES = {
    "any_current_symptom": "symptom_any_current",
    "either_often_symptom_or": "either_often_symptom",
    "highly_symptomatic_and": "highly_symptomatic_derived",
    "hs_vs_clean_control": "hs_vs_clean_control",
}
CONTINUOUS_OUTCOMES = {
    "symptom_sum_0_6": "symptom_sum_0_6",
}
CORE_COVARIATES = ["age", "BMI", "AntibioticsPriorMonth", "gap_days_abs"]
ALPHA_FEATURES = ["alpha_richness", "alpha_shannon"]
DIALISTER = "genus__Dialister"
FROZEN_CANDIDATE_GENERA = [
    DIALISTER,
    "genus__Christensenellaceae_R_7_group",
    "genus__Bacteroides",
    "genus__Alistipes",
    "genus__Subdoligranulum",
    "genus__Lachnospiraceae_NK4A136_group",
    "genus__Ruminococcaceae_UCG_010",
    "genus__Ruminococcaceae_UCG_011",
    "genus__Erysipelatoclostridium",
    "genus__Streptococcus",
]
MIN_BINARY_CASES = 20
MIN_CONTINUOUS_N = 100


def sex_is_estimable(data: pd.DataFrame) -> bool:
    if "SEX" not in data.columns:
        return False
    return bool(data["SEX"].dropna().astype(str).nunique() >= 2)


def build_x(model: pd.DataFrame, feature: str, covariates: list[str]) -> pd.DataFrame:
    x_parts = [model[[feature]].rename(columns={feature: "feature_z"})]
    for cov in covariates:
        if cov not in model.columns:
            continue
        if model[cov].dtype == object or str(model[cov].dtype).startswith(("string", "category")):
            x_parts.append(pd.get_dummies(model[cov].fillna("Missing").astype(str), prefix=cov, drop_first=True).astype(float))
        else:
            x_parts.append(pd.DataFrame({cov: standardize(model[cov])}, index=model.index))
    return pd.concat(x_parts, axis=1).replace([np.inf, -np.inf], np.nan)


def binary_association(data: pd.DataFrame, outcome: str, feature: str, covariates: list[str]) -> dict[str, Any]:
    cols = [outcome, feature, "FamilyID", *covariates]
    model = data[[col for col in cols if col in data.columns]].copy()
    model[outcome] = pd.to_numeric(model[outcome], errors="coerce")
    model[feature] = pd.to_numeric(model[feature], errors="coerce")
    model = model.dropna(subset=[outcome, feature])
    y = model[outcome].astype(int)
    if y.nunique() < 2 or min(y.sum(), len(y) - y.sum()) < MIN_BINARY_CASES:
        return {}
    x = build_x(model, feature, covariates)
    keep = x.notna().all(axis=1)
    x = x.loc[keep].astype(float)
    x["feature_z"] = standardize(x["feature_z"])
    x = sm.add_constant(x, has_constant="add")
    y = y.loc[keep]
    groups = model.loc[keep, "FamilyID"].astype(str)
    if y.nunique() < 2 or min(y.sum(), len(y) - y.sum()) < MIN_BINARY_CASES:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    beta = float(fit.params["feature_z"])
    standard_error = float(fit.bse["feature_z"])
    ci_low_beta, ci_high_beta = [float(value) for value in fit.conf_int().loc["feature_z"]]
    return {
        "n": int(len(y)),
        "cases": int(y.sum()),
        "controls": int(len(y) - y.sum()),
        "families": int(groups.nunique()),
        "beta_per_sd": beta,
        "standard_error": standard_error,
        "ci_low_beta": ci_low_beta,
        "ci_high_beta": ci_high_beta,
        "odds_ratio_per_sd": float(math.exp(beta)),
        "ci_low_or": float(math.exp(ci_low_beta)),
        "ci_high_or": float(math.exp(ci_high_beta)),
        "p_value": float(fit.pvalues["feature_z"]),
        "covariance_method": "family-clustered robust",
        "converged": bool(getattr(fit, "converged", True)),
    }


def continuous_association(data: pd.DataFrame, outcome: str, feature: str, covariates: list[str]) -> dict[str, Any]:
    cols = [outcome, feature, "FamilyID", *covariates]
    model = data[[col for col in cols if col in data.columns]].copy()
    model[outcome] = pd.to_numeric(model[outcome], errors="coerce")
    model[feature] = pd.to_numeric(model[feature], errors="coerce")
    model = model.dropna(subset=[outcome, feature])
    if len(model) < MIN_CONTINUOUS_N or model[outcome].std(skipna=True) == 0:
        return {}
    x = build_x(model, feature, covariates)
    keep = x.notna().all(axis=1)
    x = x.loc[keep].astype(float)
    x["feature_z"] = standardize(x["feature_z"])
    x = sm.add_constant(x, has_constant="add")
    y = model.loc[keep, outcome].astype(float)
    groups = model.loc[keep, "FamilyID"].astype(str)
    if len(y) < MIN_CONTINUOUS_N or y.std() == 0:
        return {}
    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci_low_beta, ci_high_beta = [float(value) for value in fit.conf_int().loc["feature_z"]]
    return {
        "n": int(len(y)),
        "families": int(groups.nunique()),
        "outcome_mean": float(y.mean()),
        "outcome_sd": float(y.std()),
        "beta_per_sd": float(fit.params["feature_z"]),
        "standard_error": float(fit.bse["feature_z"]),
        "ci_low_beta": ci_low_beta,
        "ci_high_beta": ci_high_beta,
        "p_value": float(fit.pvalues["feature_z"]),
        "covariance_method": "family-clustered robust",
    }


def add_q_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["q_value_global"] = np.nan
    mask = out["p_value"].notna()
    if mask.any():
        out.loc[mask, "q_value_global"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    out["q_value_within_branch_outcome"] = np.nan
    if mask.any():
        grouped = out.loc[mask].groupby(["branch_id", "outcome_id", "outcome_type", "covariate_set"])
        for _, group in grouped:
            out.loc[group.index, "q_value_within_branch_outcome"] = multipletests(group["p_value"], method="fdr_bh")[1]
    out["q_value_alpha_six"] = np.nan
    alpha_six = (
        (out["branch_id"] == "within_3y")
        & (out["covariate_set"] == "primary_core")
        & (out["outcome_type"] == "binary")
        & out["outcome_id"].isin(["either_often_symptom_or", "highly_symptomatic_and", "hs_vs_clean_control"])
        & out["feature"].isin(ALPHA_FEATURES)
        & out["p_value"].notna()
    )
    if int(alpha_six.sum()) != 6:
        raise RuntimeError(f"Expected exactly six manuscript alpha-diversity tests; found {int(alpha_six.sum())}")
    out.loc[alpha_six, "q_value_alpha_six"] = multipletests(
        out.loc[alpha_six, "p_value"], method="fdr_bh"
    )[1]
    return out


def prediction_table(
    data: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    frozen_features: list[str],
    continuous: bool,
) -> list[dict[str, Any]]:
    feature_sets = {
        "covariates": covariates,
        "covariates_plus_alpha": [*covariates, *[f for f in ALPHA_FEATURES if f in data.columns]],
        "covariates_plus_alpha_dialister": [
            *covariates,
            *[f for f in ALPHA_FEATURES if f in data.columns],
            *([DIALISTER] if DIALISTER in data.columns else []),
        ],
        "covariates_plus_alpha_frozen_taxa": [
            *covariates,
            *[f for f in ALPHA_FEATURES if f in data.columns],
            *frozen_features,
        ],
    }
    cols = sorted(set([outcome, "FamilyID", *[c for values in feature_sets.values() for c in values]]))
    model = data[[col for col in cols if col in data.columns]].copy()
    model[outcome] = pd.to_numeric(model[outcome], errors="coerce")
    model = model.dropna(subset=[outcome])
    if model.empty:
        return []
    y = model[outcome].astype(float if continuous else int)
    groups = model["FamilyID"].astype(str)
    if groups.nunique() < 3:
        return []
    if continuous:
        if len(y) < MIN_CONTINUOUS_N or y.std() == 0:
            return []
        metric_name = "r2"
    else:
        if y.nunique() < 2 or min(y.sum(), len(y) - y.sum()) < MIN_BINARY_CASES:
            return []
        metric_name = "auc"
    splits = list(GroupKFold(n_splits=3).split(model, y, groups))
    rows: list[dict[str, Any]] = []
    for model_id, features in feature_sets.items():
        features = [f for f in features if f in model.columns]
        cat_cols = [col for col in features if model[col].dtype == object or str(model[col].dtype).startswith(("string", "category"))]
        num_cols = [col for col in features if col not in cat_cols]
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ],
            remainder="drop",
        )
        estimator = Ridge(alpha=10.0) if continuous else LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
        pipe = Pipeline([("preprocess", pre), ("model", estimator)])
        metrics: list[float] = []
        for train_idx, test_idx in splits:
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            if not continuous and (y_train.nunique() < 2 or y_test.nunique() < 2):
                continue
            pipe.fit(model.iloc[train_idx][features], y_train)
            if continuous:
                pred = pipe.predict(model.iloc[test_idx][features])
                metrics.append(float(r2_score(y_test, pred)))
            else:
                pred = pipe.predict_proba(model.iloc[test_idx][features])[:, 1]
                metrics.append(float(roc_auc_score(y_test, pred)))
        if metrics:
            row: dict[str, Any] = {
                "model_id": model_id,
                metric_name: float(np.mean(metrics)),
                "folds_used": int(len(metrics)),
                "n": int(len(y)),
                "n_features_input": int(len(features)),
            }
            if not continuous:
                row["cases"] = int(y.sum())
                row["controls"] = int(len(y) - y.sum())
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locked primary 16S dry-eye microbiome analysis.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "locked_primary_16s_2013",
    )
    parser.add_argument(
        "--tie-rule",
        choices=["highest_read_depth", "lowest_read_depth", "source_order"],
        default="highest_read_depth",
        help="Outcome-independent rule for sequencing profiles tied on nearest physical stool date.",
    )
    parser.add_argument(
        "--sex",
        choices=["auto", "include", "exclude"],
        default="auto",
        help="Sex covariate handling. Auto includes sex only when at least two levels are present within the 3-year cohort.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ded = load_ded_2013(args.data_root)
    mapping = load_mapping(args.data_root)
    family = load_family_ids(args.data_root)
    counts = load_asv_counts(args.data_root)
    taxa = load_taxa(args.data_root)
    pairs = nearest_stool_pairs(
        ded,
        mapping,
        family,
        sample_depths=sample_read_depths(counts),
        tie_rule=args.tie_rule,
    )
    alpha, genus = compute_alpha_and_genus(counts, taxa)
    base = pairs.merge(alpha, on="SequencingSpecificName", how="inner").merge(
        genus, on="SequencingSpecificName", how="inner"
    )
    within_3y = base[base["gap_days_abs"] <= TIMING_BRANCHES["within_3y"]]
    include_sex = args.sex == "include" or (args.sex == "auto" and sex_is_estimable(within_3y))
    primary_covariates = [*CORE_COVARIATES, *(["SEX"] if include_sex else [])]
    covariate_sets = {
        "primary_core": primary_covariates,
        "primary_core_plus_depth": [*primary_covariates, "alpha_total_reads"],
    }
    available_frozen = [f for f in FROZEN_CANDIDATE_GENERA if f in base.columns]
    association_features = [f for f in [*ALPHA_FEATURES, *available_frozen] if f in base.columns]

    branch_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for branch_id, max_gap in TIMING_BRANCHES.items():
        branch = base[base["gap_days_abs"] <= max_gap].copy()
        branch_rows.append(
            {
                "branch_id": branch_id,
                "max_gap_days": max_gap,
                "rows": int(len(branch)),
                "participants": int(branch["PublicID"].nunique()),
                "families": int(branch["FamilyID"].nunique()),
                "median_gap_days_abs": float(branch["gap_days_abs"].median()) if len(branch) else np.nan,
                "highly_symptomatic_cases": int((branch["highly_symptomatic_derived"] == 1).sum()),
                "hs_vs_clean_cases": int((branch["hs_vs_clean_control"] == 1).sum()),
                "hs_vs_clean_controls": int((branch["hs_vs_clean_control"] == 0).sum()),
            }
        )
        for covariate_set, covariates in covariate_sets.items():
            for outcome_id, outcome_col in BINARY_OUTCOMES.items():
                for feature in association_features:
                    row: dict[str, Any] = {
                        "branch_id": branch_id,
                        "outcome_id": outcome_id,
                        "outcome_type": "binary",
                        "covariate_set": covariate_set,
                        "feature": feature,
                    }
                    try:
                        row.update(binary_association(branch, outcome_col, feature, covariates))
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    association_rows.append(row)
                preds = prediction_table(branch, outcome_col, covariates, available_frozen, continuous=False)
                for pred in preds:
                    pred.update(
                        {
                            "branch_id": branch_id,
                            "outcome_id": outcome_id,
                            "outcome_type": "binary",
                            "covariate_set": covariate_set,
                        }
                    )
                prediction_rows.extend(preds)
            for outcome_id, outcome_col in CONTINUOUS_OUTCOMES.items():
                for feature in association_features:
                    row = {
                        "branch_id": branch_id,
                        "outcome_id": outcome_id,
                        "outcome_type": "continuous",
                        "covariate_set": covariate_set,
                        "feature": feature,
                    }
                    try:
                        row.update(continuous_association(branch, outcome_col, feature, covariates))
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    association_rows.append(row)
                preds = prediction_table(branch, outcome_col, covariates, available_frozen, continuous=True)
                for pred in preds:
                    pred.update(
                        {
                            "branch_id": branch_id,
                            "outcome_id": outcome_id,
                            "outcome_type": "continuous",
                            "covariate_set": covariate_set,
                        }
                    )
                prediction_rows.extend(preds)

    assoc = add_q_values(pd.DataFrame(association_rows))
    pred = pd.DataFrame(prediction_rows)
    branch = pd.DataFrame(branch_rows)
    assoc.to_csv(args.out_dir / "locked_primary_16s_association_matrix.csv", index=False)
    pred.to_csv(args.out_dir / "locked_primary_16s_prediction_metrics.csv", index=False)
    branch.to_csv(args.out_dir / "branch_summary.csv", index=False)
    if not pred.empty:
        metric_col = np.where(pred["outcome_type"] == "continuous", pred.get("r2", np.nan), pred.get("auc", np.nan))
        pred_summary = pred.assign(metric=metric_col).pivot_table(
            index=["branch_id", "outcome_id", "outcome_type", "covariate_set"],
            columns="model_id",
            values="metric",
        ).reset_index()
        if {"covariates", "covariates_plus_alpha"}.issubset(pred_summary.columns):
            pred_summary["alpha_delta"] = pred_summary["covariates_plus_alpha"] - pred_summary["covariates"]
        if {"covariates", "covariates_plus_alpha_dialister"}.issubset(pred_summary.columns):
            pred_summary["alpha_dialister_delta"] = pred_summary["covariates_plus_alpha_dialister"] - pred_summary["covariates"]
        if {"covariates", "covariates_plus_alpha_frozen_taxa"}.issubset(pred_summary.columns):
            pred_summary["alpha_frozen_taxa_delta"] = (
                pred_summary["covariates_plus_alpha_frozen_taxa"] - pred_summary["covariates"]
            )
        pred_summary.to_csv(args.out_dir / "locked_primary_16s_prediction_summary.csv", index=False)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_rows": int(len(base)),
        "base_participants": int(base["PublicID"].nunique()),
        "sex_mode": args.sex,
        "tie_rule": args.tie_rule,
        "nearest_tied_questionnaires": int((pairs["nearest_tie_count"] > 1).sum()),
        "max_nearest_tie_count": int(pairs["nearest_tie_count"].max()),
        "sex_included": include_sex,
        "covariate_sets": covariate_sets,
        "primary_branch": "within_3y",
        "primary_outcome": "highly_symptomatic_and",
        "key_supportive_outcome": "hs_vs_clean_control",
        "timing_sensitivity_branch": "within_1y",
        "association_features": association_features,
        "available_frozen_candidates": available_frozen,
        "missing_frozen_candidates": [f for f in FROZEN_CANDIDATE_GENERA if f not in base.columns],
        "prediction_guardrail": "Same-cohort exploratory candidates; no new outcome-ranked selection inside cross-validation. Feature-set names match implementation.",
        "alpha_fdr_family": "Benjamini-Hochberg across observed ASV richness and Shannon diversity for three within-3-year symptom contrasts (six tests).",
        "no_raw_data_written": True,
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
