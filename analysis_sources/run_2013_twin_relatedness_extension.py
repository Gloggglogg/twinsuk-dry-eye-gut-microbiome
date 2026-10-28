from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

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
)
from run_2013_locked_primary_16s_analysis import CORE_COVARIATES, sex_is_estimable
from run_2013_publishability_sensitivities import (
    PRIMARY_FEATURE,
    PRIMARY_MAX_GAP_DAYS,
    PRIMARY_OUTCOME,
    classify_pair_structure,
    primary_complete_cases,
)


EXPECTED_N = 1685
EXPECTED_CASES = 93
EXPECTED_FAMILIES = 1125
EXPECTED_BINARY_DISCORDANT_PAIRS = 54
OUTPUT_DIR = PROJECT_ROOT / 'runtime' / 'twin_relatedness_extension_2026_08_28'
NUMERIC_COVARIATES = ["age", "BMI", "gap_days_abs"]
CATEGORICAL_COVARIATES = ["AntibioticsPriorMonth", "SEX"]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def normal_ci(beta: float, standard_error: float) -> tuple[float, float]:
    return beta - 1.96 * standard_error, beta + 1.96 * standard_error


def normalize_zygosity(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().upper()
    if text == "MZ":
        return "MZ"
    if text == "DZ":
        return "DZ"
    return "Unknown"


def assign_pair_zygosity(data: pd.DataFrame) -> pd.Series:
    mapping: dict[str, str] = {}
    for family_id, group in data.groupby("FamilyID", sort=False):
        known = {
            normalize_zygosity(value)
            for value in group["ACTUAL_ZYGOSITY"]
            if normalize_zygosity(value) != "Unknown"
        }
        mapping[str(family_id)] = next(iter(known)) if len(known) == 1 else "Unknown"
    return data["FamilyID"].astype(str).map(mapping).fillna("Unknown")


def build_analysis_cohort(data_root: Path) -> tuple[pd.DataFrame, list[str]]:
    ded = load_ded_2013(data_root)
    mapping = load_mapping(data_root)
    family = load_family_ids(data_root)
    counts = load_asv_counts(data_root)
    taxa = load_taxa(data_root)
    linked = nearest_stool_pairs(
        ded,
        mapping,
        family,
        sample_depths=sample_read_depths(counts),
        tie_rule="highest_read_depth",
    )
    alpha, _ = compute_alpha_and_genus(counts, taxa)
    base = linked.merge(alpha, on="SequencingSpecificName", how="inner")
    within_window = base.loc[base["gap_days_abs"] <= PRIMARY_MAX_GAP_DAYS].copy()
    covariates = [
        *CORE_COVARIATES,
        *(["SEX"] if sex_is_estimable(within_window) else []),
    ]
    cohort = primary_complete_cases(within_window, covariates).copy()
    cohort["pair_zygosity"] = assign_pair_zygosity(cohort)
    cohort["richness_z"] = (
        pd.to_numeric(cohort[PRIMARY_FEATURE], errors="coerce")
        - pd.to_numeric(cohort[PRIMARY_FEATURE], errors="coerce").mean()
    ) / pd.to_numeric(cohort[PRIMARY_FEATURE], errors="coerce").std(ddof=1)
    cohort["symptom_sum_0_6"] = pd.to_numeric(
        cohort["symptom_sum_0_6"], errors="coerce"
    )

    checks = {
        "participants": int(len(cohort)),
        "cases": int(cohort[PRIMARY_OUTCOME].sum()),
        "families": int(cohort["FamilyID"].nunique()),
    }
    expected = {
        "participants": EXPECTED_N,
        "cases": EXPECTED_CASES,
        "families": EXPECTED_FAMILIES,
    }
    if checks != expected:
        raise RuntimeError(f"Primary cohort failed closed: observed {checks}, expected {expected}")
    if cohort["symptom_sum_0_6"].isna().any():
        raise RuntimeError("The primary cohort contains missing 0--6 symptom scores")
    return cohort, covariates


def two_person_pairs(cohort: pd.DataFrame) -> pd.DataFrame:
    sizes = cohort.groupby("FamilyID", sort=False).size()
    pair_ids = sizes.index[sizes == 2]
    pairs = cohort.loc[cohort["FamilyID"].isin(pair_ids)].copy()
    if not bool((pairs.groupby("FamilyID").size() == 2).all()):
        raise RuntimeError("Pair extraction did not produce exactly two participants per family")
    return pairs


def summarize_pair_structure(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for zygosity in ["All", "MZ", "DZ", "Unknown"]:
        subset = pairs if zygosity == "All" else pairs.loc[pairs["pair_zygosity"] == zygosity]
        grouped = subset.groupby("FamilyID", sort=False)
        case_counts = grouped[PRIMARY_OUTCOME].sum() if not subset.empty else pd.Series(dtype=float)
        score_unique = grouped["symptom_sum_0_6"].nunique() if not subset.empty else pd.Series(dtype=float)
        rows.append(
            {
                "zygosity": zygosity,
                "pairs": int(subset["FamilyID"].nunique()),
                "participants": int(len(subset)),
                "binary_case_concordant_pairs": int((case_counts == 2).sum()),
                "binary_discordant_pairs": int((case_counts == 1).sum()),
                "binary_control_concordant_pairs": int((case_counts == 0).sum()),
                "symptom_score_discordant_pairs": int((score_unique > 1).sum()),
                "symptom_score_concordant_pairs": int((score_unique == 1).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    observed = int(summary.loc[summary["zygosity"] == "All", "binary_discordant_pairs"].iloc[0])
    if observed != EXPECTED_BINARY_DISCORDANT_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_BINARY_DISCORDANT_PAIRS} binary-discordant pairs; found {observed}"
        )
    return summary


def fit_conditional_logit(data: pd.DataFrame, label: str) -> dict[str, Any]:
    family_cases = data.groupby("FamilyID", sort=False)[PRIMARY_OUTCOME].sum()
    discordant_ids = family_cases.index[family_cases == 1]
    model = data.loc[data["FamilyID"].isin(discordant_ids)].copy()
    pairs = int(model["FamilyID"].nunique())
    base: dict[str, Any] = {
        "analysis": "binary_conditional_logistic",
        "zygosity": label,
        "pairs": pairs,
        "participants": int(len(model)),
        "cases": int(model[PRIMARY_OUTCOME].sum()),
        "converged": False,
        "warning_count": 0,
        "beta_per_sd": np.nan,
        "standard_error": np.nan,
        "odds_ratio_per_sd": np.nan,
        "ci_low_or": np.nan,
        "ci_high_or": np.nan,
        "p_value": np.nan,
    }
    if pairs < 5:
        base["status"] = "not fitted: fewer than five informative pairs"
        return base
    y = model[PRIMARY_OUTCOME].astype(int)
    x = model[["richness_z"]].astype(float)
    groups = model["FamilyID"].astype(str)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fit = ConditionalLogit(y, x, groups=groups).fit(
                method="bfgs", maxiter=1000, disp=False
            )
        beta = float(fit.params["richness_z"])
        standard_error = float(fit.bse["richness_z"])
        low, high = normal_ci(beta, standard_error)
        base.update(
            {
                "converged": bool(getattr(fit, "mle_retvals", {}).get("converged", True)),
                "warning_count": len(captured),
                "beta_per_sd": beta,
                "standard_error": standard_error,
                "odds_ratio_per_sd": math.exp(beta),
                "ci_low_or": math.exp(low),
                "ci_high_or": math.exp(high),
                "p_value": float(fit.pvalues["richness_z"]),
                "status": "fitted",
            }
        )
    except Exception as exc:
        base["status"] = f"fit failed: {type(exc).__name__}: {exc}"
    return base


def fit_binary_zygosity_interaction(pairs: pd.DataFrame) -> pd.DataFrame:
    known = pairs.loc[pairs["pair_zygosity"].isin(["MZ", "DZ"])].copy()
    family_cases = known.groupby("FamilyID", sort=False)[PRIMARY_OUTCOME].sum()
    discordant_ids = family_cases.index[family_cases == 1]
    model = known.loc[known["FamilyID"].isin(discordant_ids)].copy()
    model["richness_z_by_dz"] = model["richness_z"] * (
        model["pair_zygosity"] == "DZ"
    ).astype(float)
    y = model[PRIMARY_OUTCOME].astype(int)
    x = model[["richness_z", "richness_z_by_dz"]].astype(float)
    groups = model["FamilyID"].astype(str)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fit = ConditionalLogit(y, x, groups=groups).fit(
            method="bfgs", maxiter=2000, disp=False
        )
    covariance = fit.cov_params()
    beta_mz = float(fit.params["richness_z"])
    beta_interaction = float(fit.params["richness_z_by_dz"])
    beta_dz = beta_mz + beta_interaction
    variance_dz = float(
        covariance.loc["richness_z", "richness_z"]
        + covariance.loc["richness_z_by_dz", "richness_z_by_dz"]
        + 2 * covariance.loc["richness_z", "richness_z_by_dz"]
    )
    se_mz = float(fit.bse["richness_z"])
    se_dz = math.sqrt(max(variance_dz, 0.0))
    rows: list[dict[str, Any]] = []
    for zygosity, beta, standard_error in [
        ("MZ", beta_mz, se_mz),
        ("DZ", beta_dz, se_dz),
    ]:
        low, high = normal_ci(beta, standard_error)
        rows.append(
            {
                "analysis": "binary_zygosity_interaction_stratum_estimate",
                "zygosity": zygosity,
                "pairs": int(model.loc[model["pair_zygosity"] == zygosity, "FamilyID"].nunique()),
                "odds_ratio_per_sd": math.exp(beta),
                "ci_low_or": math.exp(low),
                "ci_high_or": math.exp(high),
                "p_value": 2 * stats.norm.sf(abs(beta / standard_error)),
                "interaction_p_value": float(fit.pvalues["richness_z_by_dz"]),
                "warning_count": len(captured),
            }
        )
    return pd.DataFrame(rows)


def encoded_predictors(cohort: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=cohort.index)
    output["richness_z"] = cohort["richness_z"].astype(float)
    for column in NUMERIC_COVARIATES:
        values = pd.to_numeric(cohort[column], errors="coerce")
        output[f"{column}_z"] = (values - values.mean()) / values.std(ddof=1)
    for column in CATEGORICAL_COVARIATES:
        if column not in cohort.columns:
            continue
        dummies = pd.get_dummies(
            cohort[column].fillna("Missing").astype(str),
            prefix=column,
            drop_first=True,
            dtype=float,
        )
        output = output.join(dummies)
    return output


def pair_difference_frame(
    pairs: pd.DataFrame, predictors: pd.DataFrame
) -> pd.DataFrame:
    joined = pairs[["PublicID", "FamilyID", "pair_zygosity", "symptom_sum_0_6"]].join(
        predictors
    )
    rows: list[dict[str, Any]] = []
    predictor_columns = predictors.columns.tolist()
    for _, group in joined.groupby("FamilyID", sort=False):
        ordered = group.sort_values("PublicID")
        if len(ordered) != 2:
            raise RuntimeError("Pair-difference input contains a non-pair family")
        first = ordered.iloc[0]
        second = ordered.iloc[1]
        row: dict[str, Any] = {
            "pair_zygosity": first["pair_zygosity"],
            "symptom_difference": float(first["symptom_sum_0_6"] - second["symptom_sum_0_6"]),
        }
        for column in predictor_columns:
            row[f"difference__{column}"] = float(first[column] - second[column])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_score_difference_model(
    differences: pd.DataFrame,
    zygosity: str,
    adjusted: bool,
) -> dict[str, Any]:
    model = differences if zygosity == "All" else differences.loc[
        differences["pair_zygosity"] == zygosity
    ]
    richness_column = "difference__richness_z"
    predictor_columns = [richness_column]
    if adjusted:
        predictor_columns.extend(
            column
            for column in model.columns
            if column.startswith("difference__") and column != richness_column
        )
    x = model[predictor_columns].astype(float)
    informative_columns = [
        column for column in x.columns if bool((x[column].abs() > 1e-12).any())
    ]
    x = x[informative_columns]
    y = model["symptom_difference"].astype(float)
    fit = sm.OLS(y, x).fit(cov_type="HC1")
    beta = float(fit.params[richness_column])
    standard_error = float(fit.bse[richness_column])
    low, high = normal_ci(beta, standard_error)
    return {
        "analysis": "within_pair_linear_fixed_effect",
        "zygosity": zygosity,
        "adjustment": "participant-specific primary covariates" if adjusted else "pair fixed effect only",
        "pairs": int(len(model)),
        "symptom_score_discordant_pairs": int((y != 0).sum()),
        "beta_score_points_per_sd": beta,
        "standard_error": standard_error,
        "ci_low": low,
        "ci_high": high,
        "p_value": float(fit.pvalues[richness_column]),
        "included_predictors": "; ".join(informative_columns),
    }


def fit_score_zygosity_interaction(
    differences: pd.DataFrame,
) -> dict[str, Any]:
    model = differences.loc[differences["pair_zygosity"].isin(["MZ", "DZ"])].copy()
    model["richness_difference_by_dz"] = model["difference__richness_z"] * (
        model["pair_zygosity"] == "DZ"
    ).astype(float)
    covariate_columns = [
        column
        for column in model.columns
        if column.startswith("difference__")
        and column != "difference__richness_z"
        and bool((model[column].abs() > 1e-12).any())
    ]
    x = model[
        ["difference__richness_z", "richness_difference_by_dz", *covariate_columns]
    ].astype(float)
    y = model["symptom_difference"].astype(float)
    fit = sm.OLS(y, x).fit(cov_type="HC1")
    covariance = fit.cov_params()
    beta_mz = float(fit.params["difference__richness_z"])
    beta_interaction = float(fit.params["richness_difference_by_dz"])
    beta_dz = beta_mz + beta_interaction
    variance_dz = float(
        covariance.loc["difference__richness_z", "difference__richness_z"]
        + covariance.loc["richness_difference_by_dz", "richness_difference_by_dz"]
        + 2 * covariance.loc["difference__richness_z", "richness_difference_by_dz"]
    )
    return {
        "analysis": "adjusted_within_pair_score_zygosity_interaction",
        "known_zygosity_pairs": int(len(model)),
        "mz_pairs": int((model["pair_zygosity"] == "MZ").sum()),
        "dz_pairs": int((model["pair_zygosity"] == "DZ").sum()),
        "beta_mz": beta_mz,
        "beta_dz": beta_dz,
        "beta_dz_minus_mz": beta_interaction,
        "standard_error_dz": math.sqrt(max(variance_dz, 0.0)),
        "interaction_p_value": float(fit.pvalues["richness_difference_by_dz"]),
        "included_predictors": "; ".join(x.columns),
    }


def fit_within_between_model(
    pairs: pd.DataFrame,
    predictors: pd.DataFrame,
    adjusted: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model = pairs[["FamilyID", "symptom_sum_0_6"]].join(predictors).copy()
    predictor_names = ["richness_z"]
    if adjusted:
        predictor_names.extend(column for column in predictors.columns if column != "richness_z")
    design = pd.DataFrame(index=model.index)
    for column in predictor_names:
        pair_mean = model.groupby("FamilyID")[column].transform("mean")
        design[f"within__{column}"] = model[column] - pair_mean
        design[f"between__{column}"] = pair_mean
    keep_columns = [
        column for column in design.columns if float(design[column].std(ddof=1)) > 1e-12
    ]
    design = sm.add_constant(design[keep_columns].astype(float), has_constant="add")
    fit = sm.OLS(model["symptom_sum_0_6"].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": model["FamilyID"].astype(str)}
    )
    within_name = "within__richness_z"
    between_name = "between__richness_z"
    covariance = fit.cov_params()
    beta_within = float(fit.params[within_name])
    beta_between = float(fit.params[between_name])
    variance_difference = float(
        covariance.loc[within_name, within_name]
        + covariance.loc[between_name, between_name]
        - 2 * covariance.loc[within_name, between_name]
    )
    se_difference = math.sqrt(max(variance_difference, 0.0))
    difference = beta_within - beta_between
    rows: list[dict[str, Any]] = []
    for component, name in [("within pair", within_name), ("between pairs", between_name)]:
        beta = float(fit.params[name])
        standard_error = float(fit.bse[name])
        low, high = normal_ci(beta, standard_error)
        rows.append(
            {
                "analysis": "within_between_linear_decomposition",
                "adjustment": "participant-specific primary covariates" if adjusted else "unadjusted",
                "component": component,
                "pairs": int(model["FamilyID"].nunique()),
                "beta_score_points_per_sd": beta,
                "standard_error": standard_error,
                "ci_low": low,
                "ci_high": high,
                "p_value": float(fit.pvalues[name]),
            }
        )
    comparison = {
        "analysis": "within_equals_between_wald_test",
        "adjustment": "participant-specific primary covariates" if adjusted else "unadjusted",
        "within_minus_between_beta": difference,
        "standard_error": se_difference,
        "p_value": float(2 * stats.norm.sf(abs(difference / se_difference)))
        if se_difference > 0
        else np.nan,
        "included_predictors": "; ".join(keep_columns),
    }
    return pd.DataFrame(rows), comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether zygosity-stratified and symptom-score within-pair models add "
            "information beyond the existing binary discordant-pair analysis."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort, covariates = build_analysis_cohort(args.data_root)
    pairs = two_person_pairs(cohort)
    pair_structure = summarize_pair_structure(pairs)
    pair_structure.to_csv(args.out_dir / "twin_pair_structure_by_zygosity.csv", index=False)

    binary_rows = [fit_conditional_logit(pairs, "All")]
    for zygosity in ["MZ", "DZ", "Unknown"]:
        binary_rows.append(
            fit_conditional_logit(pairs.loc[pairs["pair_zygosity"] == zygosity], zygosity)
        )
    binary_results = pd.DataFrame(binary_rows)
    binary_results.to_csv(args.out_dir / "binary_conditional_models.csv", index=False)
    binary_interaction = fit_binary_zygosity_interaction(pairs)
    binary_interaction.to_csv(args.out_dir / "binary_zygosity_interaction.csv", index=False)

    predictors = encoded_predictors(cohort)
    pair_predictors = predictors.loc[pairs.index]
    differences = pair_difference_frame(pairs, pair_predictors)
    score_rows: list[dict[str, Any]] = []
    for zygosity in ["All", "MZ", "DZ"]:
        for adjusted in [False, True]:
            score_rows.append(fit_score_difference_model(differences, zygosity, adjusted))
    score_results = pd.DataFrame(score_rows)
    score_results.to_csv(args.out_dir / "symptom_score_within_pair_models.csv", index=False)
    score_interaction = fit_score_zygosity_interaction(differences)
    write_json(args.out_dir / "symptom_score_zygosity_interaction.json", score_interaction)

    decomposition_frames: list[pd.DataFrame] = []
    decomposition_tests: list[dict[str, Any]] = []
    for adjusted in [False, True]:
        frame, comparison = fit_within_between_model(pairs, pair_predictors, adjusted)
        decomposition_frames.append(frame)
        decomposition_tests.append(comparison)
    decomposition = pd.concat(decomposition_frames, ignore_index=True)
    decomposition.to_csv(args.out_dir / "symptom_score_within_between_decomposition.csv", index=False)
    pd.DataFrame(decomposition_tests).to_csv(
        args.out_dir / "symptom_score_within_between_tests.csv", index=False
    )

    binary_all = binary_results.loc[binary_results["zygosity"] == "All"].iloc[0]
    if not np.isclose(float(binary_all["odds_ratio_per_sd"]), 0.7913007948853894, atol=1e-10):
        raise RuntimeError("The existing overall binary within-pair estimate was not reproduced")
    score_unadjusted = score_results.loc[
        (score_results["zygosity"] == "All")
        & (score_results["adjustment"] == "pair fixed effect only")
    ].iloc[0]
    decomposition_unadjusted = decomposition.loc[
        (decomposition["adjustment"] == "unadjusted")
        & (decomposition["component"] == "within pair")
    ].iloc[0]
    if not np.isclose(
        float(score_unadjusted["beta_score_points_per_sd"]),
        float(decomposition_unadjusted["beta_score_points_per_sd"]),
        atol=1e-10,
    ):
        raise RuntimeError("Pair-difference and within-between estimators do not agree")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_cohort": {
            "participants": int(len(cohort)),
            "cases": int(cohort[PRIMARY_OUTCOME].sum()),
            "families": int(cohort["FamilyID"].nunique()),
            "covariates": covariates,
        },
        "pair_structure": pair_structure.to_dict(orient="records"),
        "binary_conditional_models": binary_results.to_dict(orient="records"),
        "binary_zygosity_interaction": binary_interaction.to_dict(orient="records"),
        "symptom_score_within_pair_models": score_results.to_dict(orient="records"),
        "symptom_score_zygosity_interaction": score_interaction,
        "symptom_score_within_between_decomposition": decomposition.to_dict(orient="records"),
        "symptom_score_within_between_tests": decomposition_tests,
        "interpretation_boundaries": [
            "The conditional logistic model is the primary co-twin control analysis for the binary outcome.",
            "The symptom-score fixed-effect model uses more pairs but addresses the 0--6 severity measure.",
            "Zygosity-stratified estimates are supportive and do not constitute a formal bivariate ACE model.",
            "A bivariate genetic and environmental decomposition is not supported by only six case-concordant pairs.",
        ],
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "data_root": str(args.data_root.resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "git_commit": git_value("rev-parse", "HEAD"),
            "raw_or_external_data_written": False,
            "participant_or_family_identifiers_written": False,
        },
    }
    write_json(args.out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
