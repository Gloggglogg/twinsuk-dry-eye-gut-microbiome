from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from run_2013_initial_microbiome_signal_probes import (
    DEFAULT_DATA_ROOT,
    DED_2013,
    MICROBIOME_DIR,
    PROJECT_ROOT,
    SPINE,
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
from run_2013_locked_primary_16s_analysis import (
    CORE_COVARIATES,
    binary_association,
    build_x,
    sex_is_estimable,
)


SPECIFICATION = Path(__file__).resolve().parents[1] / 'runtime' / 'documentation' / 'DOSE_RESPONSE_ANALYSIS_SPECIFICATION_2026-08-24.md'
REFERENCE_MATRIX = PROJECT_ROOT / 'runtime' / 'coauthor_review_production_2026_08_14' / 'primary' / 'locked_primary_16s_association_matrix.csv'
DEFAULT_OUT_DIR = PROJECT_ROOT / 'runtime' / 'symptom_dose_response_2026_08_24'
PRIMARY_MAX_GAP_DAYS = 365 * 3
MIN_GROUP_PARTICIPANTS = 20
MIN_GROUP_FAMILIES = 20
RICHNESS = "alpha_richness"
SCORE = "symptom_sum_0_6"
STRICT_CONTROL = "clean_control_no_ded_no_symptoms"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def validate_whs_coding(ded: pd.DataFrame) -> dict[str, Any]:
    expected = {0, 1, 2, 3}
    observed: dict[str, list[int]] = {}
    for column in ["How often dry eyes", "How often irritated eyes"]:
        values = sorted(
            set(pd.to_numeric(ded[column], errors="coerce").dropna().astype(int).tolist())
        )
        observed[column] = values
        if not set(values).issubset(expected):
            raise RuntimeError(f"Unexpected WHS coding in {column}: {values}")
    complete = ded[
        ded["How often dry eyes"].notna() & ded["How often irritated eyes"].notna()
    ].copy()
    recalculated = complete["How often dry eyes"] + complete["How often irritated eyes"]
    if not np.allclose(recalculated, complete[SCORE], rtol=0.0, atol=0.0):
        raise RuntimeError("The stored 0--6 symptom sum does not equal the two WHS items")
    strict = complete[STRICT_CONTROL] == 1
    if not bool((complete.loc[strict, SCORE] == 0).all()):
        raise RuntimeError("At least one strict control has a nonzero current symptom score")
    return {
        "expected_item_values": sorted(expected),
        "observed_item_values": observed,
        "complete_questionnaire_rows": int(len(complete)),
        "score_min": float(complete[SCORE].min()),
        "score_max": float(complete[SCORE].max()),
        "score_recalculation_exact": True,
        "strict_controls_have_score_zero": True,
    }


def assemble_analysis_population(
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    ded = load_ded_2013(data_root)
    coding_check = validate_whs_coding(ded)
    mapping = load_mapping(data_root)
    family = load_family_ids(data_root)
    counts = load_asv_counts(data_root)
    taxa = load_taxa(data_root)
    pairs = nearest_stool_pairs(
        ded,
        mapping,
        family,
        sample_depths=sample_read_depths(counts),
        tie_rule="highest_read_depth",
    )
    alpha, _ = compute_alpha_and_genus(counts, taxa)
    base = pairs.merge(alpha, on="SequencingSpecificName", how="inner")
    within_3y = base.loc[base["gap_days_abs"] <= PRIMARY_MAX_GAP_DAYS].copy()
    include_sex = sex_is_estimable(within_3y)
    covariates = [*CORE_COVARIATES, *(["SEX"] if include_sex else [])]

    score = pd.to_numeric(within_3y[SCORE], errors="coerce")
    strict = pd.to_numeric(within_3y[STRICT_CONTROL], errors="coerce") == 1
    eligible = strict | score.isin(range(1, 7))
    dose = within_3y.loc[eligible].copy()
    dose["dose_score"] = np.where(
        pd.to_numeric(dose[STRICT_CONTROL], errors="coerce") == 1,
        0,
        pd.to_numeric(dose[SCORE], errors="coerce"),
    )
    design = build_x(dose, RICHNESS, covariates)
    keep = design.notna().all(axis=1) & dose["FamilyID"].notna()
    dose = dose.loc[keep].copy()
    dose["dose_score"] = dose["dose_score"].astype(int)
    if not dose["dose_score"].isin(range(0, 7)).all():
        raise RuntimeError("Analysis score contains values outside 0--6")
    if dose["PublicID"].duplicated().any():
        raise RuntimeError("The dose-response population contains duplicate participants")
    dose["richness_z_global"] = standardize(pd.to_numeric(dose[RICHNESS], errors="coerce"))
    if dose["richness_z_global"].isna().any():
        raise RuntimeError("Global richness standardization produced missing values")

    assembly = {
        "coding_check": coding_check,
        "nearest_pair_rows": int(len(pairs)),
        "alpha_overlap_rows": int(len(base)),
        "within_3y_rows": int(len(within_3y)),
        "analysis_rows": int(len(dose)),
        "analysis_participants": int(dose["PublicID"].nunique()),
        "analysis_families": int(dose["FamilyID"].nunique()),
        "sex_included": include_sex,
        "covariates": covariates,
        "excluded_non_strict_score_zero": int(((score == 0) & ~strict).sum()),
    }
    return dose, within_3y, covariates, assembly


def group_counts(dose: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for score in range(0, 7):
        group = dose.loc[dose["dose_score"] == score]
        rows.append(
            {
                "dose_score": score,
                "group": "strict_control" if score == 0 else f"symptom_score_{score}",
                "participants": int(len(group)),
                "families": int(group["FamilyID"].nunique()),
                "mean_richness": float(group[RICHNESS].mean()) if len(group) else np.nan,
                "median_richness": float(group[RICHNESS].median()) if len(group) else np.nan,
                "richness_sd": float(group[RICHNESS].std()) if len(group) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fit_binary_fixed_richness(
    data: pd.DataFrame,
    case_score: int,
    covariates: list[str],
) -> dict[str, Any]:
    model = data.loc[data["dose_score"].isin([0, case_score])].copy()
    model["outcome"] = (model["dose_score"] == case_score).astype(int)
    x = build_x(model, "richness_z_global", covariates)
    keep = x.notna().all(axis=1)
    x = x.loc[keep].astype(float)
    x = sm.add_constant(x, has_constant="add")
    y = model.loc[keep, "outcome"].astype(int)
    groups = model.loc[keep, "FamilyID"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    beta = float(fit.params["feature_z"])
    low, high = [float(value) for value in fit.conf_int().loc["feature_z"]]
    return {
        "dose_score": case_score,
        "n": int(len(y)),
        "cases": int(y.sum()),
        "controls": int(len(y) - y.sum()),
        "case_families": int(model.loc[keep & (model["outcome"] == 1), "FamilyID"].nunique()),
        "total_families": int(groups.nunique()),
        "beta_per_global_richness_sd": beta,
        "standard_error": float(fit.bse["feature_z"]),
        "odds_ratio_per_global_richness_sd": float(math.exp(beta)),
        "ci_low_or": float(math.exp(low)),
        "ci_high_or": float(math.exp(high)),
        "p_value": float(fit.pvalues["feature_z"]),
        "converged": bool(getattr(fit, "converged", True)),
        "covariance_method": "family-clustered robust",
    }


def fit_ordered_linear_trend(data: pd.DataFrame, covariates: list[str]) -> dict[str, Any]:
    x = build_x(data, "richness_z_global", covariates)
    keep = x.notna().all(axis=1)
    x = x.loc[keep].astype(float)
    x = sm.add_constant(x, has_constant="add")
    y = data.loc[keep, "dose_score"].astype(float)
    groups = data.loc[keep, "FamilyID"].astype(str)
    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    low, high = [float(value) for value in fit.conf_int().loc["feature_z"]]
    return {
        "n": int(len(y)),
        "families": int(groups.nunique()),
        "mean_score": float(y.mean()),
        "beta_score_points_per_global_richness_sd": float(fit.params["feature_z"]),
        "standard_error": float(fit.bse["feature_z"]),
        "ci_low": low,
        "ci_high": high,
        "p_value": float(fit.pvalues["feature_z"]),
        "r_squared": float(fit.rsquared),
        "covariance_method": "family-clustered robust",
    }


def fit_threshold_ladder(data: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    dryness = pd.to_numeric(data["How often dry eyes"], errors="coerce")
    irritation = pd.to_numeric(data["How often irritated eyes"], errors="coerce")
    definitions = [
        ("either_at_least_sometimes", (dryness >= 1) | (irritation >= 1)),
        ("both_at_least_sometimes", (dryness >= 1) & (irritation >= 1)),
        ("either_at_least_often", (dryness >= 2) | (irritation >= 2)),
        ("both_at_least_often", (dryness >= 2) & (irritation >= 2)),
        ("either_always", (dryness == 3) | (irritation == 3)),
        ("both_always", (dryness == 3) & (irritation == 3)),
    ]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for label, case_mask in definitions:
        threshold = data.loc[(data["dose_score"] == 0) | case_mask].copy()
        threshold["threshold_case"] = case_mask.loc[threshold.index].astype(int)
        case_count = int(threshold["threshold_case"].sum())
        case_families = int(
            threshold.loc[threshold["threshold_case"] == 1, "FamilyID"].nunique()
        )
        if case_count < MIN_GROUP_PARTICIPANTS or case_families < MIN_GROUP_FAMILIES:
            rows.append(
                {
                    "threshold": label,
                    "cases": case_count,
                    "case_families": case_families,
                    "model_fitted": False,
                    "reason": "below prespecified participant or family minimum",
                }
            )
            failures.append(label)
            continue
        model = threshold.rename(columns={"threshold_case": "outcome"}).copy()
        x = build_x(model, "richness_z_global", covariates)
        keep = x.notna().all(axis=1)
        x = sm.add_constant(x.loc[keep].astype(float), has_constant="add")
        y = model.loc[keep, "outcome"].astype(int)
        groups = model.loc[keep, "FamilyID"].astype(str)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": groups}
                )
            beta = float(fit.params["feature_z"])
            low, high = [float(value) for value in fit.conf_int().loc["feature_z"]]
            rows.append(
                {
                    "threshold": label,
                    "n": int(len(y)),
                    "cases": int(y.sum()),
                    "controls": int(len(y) - y.sum()),
                    "case_families": case_families,
                    "total_families": int(groups.nunique()),
                    "model_fitted": True,
                    "beta_per_global_richness_sd": beta,
                    "odds_ratio_per_global_richness_sd": float(math.exp(beta)),
                    "ci_low_or": float(math.exp(low)),
                    "ci_high_or": float(math.exp(high)),
                    "p_value": float(fit.pvalues["feature_z"]),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "threshold": label,
                    "cases": case_count,
                    "case_families": case_families,
                    "model_fitted": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            failures.append(label)
    frame = pd.DataFrame(rows)
    fitted = frame["model_fitted"].fillna(False)
    frame["q_value_bh"] = np.nan
    if fitted.any():
        frame.loc[fitted, "q_value_bh"] = multipletests(
            frame.loc[fitted, "p_value"], method="fdr_bh"
        )[1]
    return frame, failures


def reference_results() -> tuple[dict[str, Any], dict[str, Any]]:
    reference = pd.read_csv(REFERENCE_MATRIX)
    common = (
        (reference["branch_id"] == "within_3y")
        & (reference["covariate_set"] == "primary_core")
        & (reference["feature"] == RICHNESS)
    )
    headline = reference.loc[
        common
        & (reference["outcome_id"] == "highly_symptomatic_and")
        & (reference["outcome_type"] == "binary")
    ]
    prior_trend = reference.loc[
        common
        & (reference["outcome_id"] == "symptom_sum_0_6")
        & (reference["outcome_type"] == "continuous")
    ]
    if len(headline) != 1 or len(prior_trend) != 1:
        raise RuntimeError(
            f"Expected one headline and one prior continuous-score reference row; "
            f"found {len(headline)} and {len(prior_trend)}"
        )
    return headline.iloc[0].dropna().to_dict(), prior_trend.iloc[0].dropna().to_dict()


def conservative_alpha_seven_sensitivity(trend_p_value: float) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_MATRIX)
    alpha_six = reference.loc[
        (reference["branch_id"] == "within_3y")
        & (reference["covariate_set"] == "primary_core")
        & (reference["outcome_type"] == "binary")
        & reference["outcome_id"].isin(
            ["either_often_symptom_or", "highly_symptomatic_and", "hs_vs_clean_control"]
        )
        & reference["feature"].isin(["alpha_richness", "alpha_shannon"]),
        ["outcome_id", "feature", "p_value"],
    ].copy()
    if len(alpha_six) != 6:
        raise RuntimeError(f"Expected the existing six alpha-diversity tests; found {len(alpha_six)}")
    expanded = pd.concat(
        [
            alpha_six,
            pd.DataFrame(
                [
                    {
                        "outcome_id": "strict_control_score_trend",
                        "feature": "alpha_richness",
                        "p_value": trend_p_value,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    expanded["q_value_alpha_seven"] = multipletests(
        expanded["p_value"], method="fdr_bh"
    )[1]
    expanded["sensitivity_status"] = (
        "conservative post-result multiplicity sensitivity; not used to select the analysis"
    )
    return expanded


def reproduce_headline(
    dose_source: pd.DataFrame,
    covariates: list[str],
    expected: dict[str, Any],
) -> dict[str, Any]:
    observed = binary_association(
        dose_source,
        "highly_symptomatic_derived",
        RICHNESS,
        covariates,
    )
    fields = ["n", "cases", "controls", "families", "odds_ratio_per_sd", "p_value"]
    checks: dict[str, Any] = {}
    for field in fields:
        actual = observed[field]
        target = expected[field]
        if field in {"n", "cases", "controls", "families"}:
            passed = int(actual) == int(target)
        else:
            passed = bool(np.isclose(float(actual), float(target), rtol=0.0, atol=1e-12))
        checks[field] = {"observed": actual, "expected": target, "pass": passed}
    return {"passed": all(item["pass"] for item in checks.values()), "checks": checks}


def save_figure(category: pd.DataFrame, counts: pd.DataFrame, path: Path) -> None:
    plot = category.loc[category["model_fitted"].fillna(False)].sort_values("dose_score").copy()
    n_lookup = counts.set_index("dose_score")["participants"].to_dict()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.5),
        gridspec_kw={"height_ratios": [1.0, 2.0]},
        constrained_layout=True,
    )
    count_ax, ax = axes
    count_ax.bar(
        counts["dose_score"],
        counts["participants"],
        color=["#6f6f6f", *(["#8bb8d1"] * 6)],
    )
    count_ax.set_xticks(range(0, 7))
    count_ax.set_ylabel("Participants")
    count_ax.set_title("A. Analysis-group sizes")
    for _, row in counts.iterrows():
        count_ax.text(
            row["dose_score"],
            row["participants"] + max(counts["participants"]) * 0.025,
            str(int(row["participants"])),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    count_ax.set_ylim(0, max(counts["participants"]) * 1.18)
    ax.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax.errorbar(
        plot["dose_score"],
        plot["odds_ratio_per_global_richness_sd"],
        yerr=[
            plot["odds_ratio_per_global_richness_sd"] - plot["ci_low_or"],
            plot["ci_high_or"] - plot["odds_ratio_per_global_richness_sd"],
        ],
        fmt="o-",
        color="#1f5a7a",
        ecolor="#1f5a7a",
        capsize=3,
        linewidth=1.5,
    )
    ax.set_yscale("log")
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Combined ocular-symptom score")
    ax.set_ylabel("Odds ratio per 1-SD higher richness")
    ax.set_title("B. Adjusted richness association vs strict controls")
    for _, row in plot.iterrows():
        ax.annotate(
            f"n={n_lookup.get(int(row['dose_score']), 0)}",
            (row["dose_score"], row["ci_high_or"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    unfitted = category.loc[~category["model_fitted"].fillna(False), "dose_score"].astype(int)
    for score in unfitted:
        ax.text(
            score,
            1.0,
            "not estimated\n(sparse)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8a3f3f",
        )
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def input_provenance(data_root: Path) -> dict[str, Any]:
    paths = {
        "ded_questionnaire": data_root / DED_2013,
        "microbiome_mapping": data_root / MICROBIOME_DIR / "TwinsUK_ASVs_MappingFile.xlsx",
        "asv_counts": data_root / MICROBIOME_DIR / "TwinsUK_ASVsCR.rds",
        "asv_taxonomy": data_root / MICROBIOME_DIR / "TwinsUK_ASVs_Taxa.csv.zip",
        "family_spine": data_root / SPINE,
        "analysis_specification": SPECIFICATION,
        "validated_reference_matrix": REFERENCE_MATRIX,
        "analysis_script": Path(__file__).resolve(),
    }
    return {
        key: {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256(path) if path.exists() else None,
        }
        for key, path in paths.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the specified strict-control ocular-symptom dose-response analysis."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Print feasibility counts without fitting or inspecting association models.",
    )
    args = parser.parse_args()

    if not SPECIFICATION.exists():
        raise FileNotFoundError(f"Dated analysis specification is missing: {SPECIFICATION}")
    dose, within_3y, covariates, assembly = assemble_analysis_population(args.data_root)
    counts = group_counts(dose)
    symptomatic = counts.loc[counts["dose_score"].between(1, 6)]
    sparse = symptomatic.loc[
        (symptomatic["participants"] < MIN_GROUP_PARTICIPANTS)
        | (symptomatic["families"] < MIN_GROUP_FAMILIES)
    ]
    feasibility = {
        "minimum_participants_per_score": MIN_GROUP_PARTICIPANTS,
        "minimum_families_per_score": MIN_GROUP_FAMILIES,
        "granular_counts_feasible": sparse.empty,
        "sparse_score_categories": sparse["dose_score"].astype(int).tolist(),
        "group_counts": counts[
            ["dose_score", "group", "participants", "families"]
        ].to_dict(orient="records"),
    }
    if args.counts_only:
        print(json.dumps({"assembly": assembly, "feasibility": feasibility}, indent=2))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.out_dir / "dose_score_group_summary.csv", index=False)
    expected_headline, prior_full_score = reference_results()
    headline_check = reproduce_headline(
        dose_source=within_3y,
        covariates=covariates,
        expected=expected_headline,
    )
    if not headline_check["passed"]:
        raise RuntimeError("The validated headline model did not reproduce in the assembled population")

    category_rows: list[dict[str, Any]] = []
    category_unfitted: list[str] = []
    category_model_failures: list[str] = []
    for score in range(1, 7):
        count_row = counts.loc[counts["dose_score"] == score].iloc[0]
        if (
            int(count_row["participants"]) < MIN_GROUP_PARTICIPANTS
            or int(count_row["families"]) < MIN_GROUP_FAMILIES
        ):
            category_rows.append(
                {
                    "dose_score": score,
                    "model_fitted": False,
                    "reason": "below prespecified participant or family minimum",
                }
            )
            category_unfitted.append(f"score_{score}: sparse")
            continue
        try:
            result = fit_binary_fixed_richness(dose, score, covariates)
            result["model_fitted"] = True
            category_rows.append(result)
            if not result["converged"] or not np.isfinite(
                result["odds_ratio_per_global_richness_sd"]
            ):
                category_model_failures.append(f"score_{score}: nonfinite or nonconverged")
        except Exception as exc:
            category_rows.append(
                {
                    "dose_score": score,
                    "model_fitted": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            category_model_failures.append(f"score_{score}: {type(exc).__name__}")

    category = pd.DataFrame(category_rows)
    fitted_category = category["model_fitted"].fillna(False)
    category["q_value_bh"] = np.nan
    if fitted_category.any():
        category.loc[fitted_category, "q_value_bh"] = multipletests(
            category.loc[fitted_category, "p_value"], method="fdr_bh"
        )[1]
    category.to_csv(args.out_dir / "strict_control_category_models.csv", index=False)

    trend = fit_ordered_linear_trend(dose, covariates)
    write_json(args.out_dir / "ordered_trend_model.json", trend)
    alpha_seven = conservative_alpha_seven_sensitivity(trend["p_value"])
    alpha_seven.to_csv(
        args.out_dir / "alpha_seven_multiplicity_sensitivity.csv", index=False
    )
    trend_alpha_seven_q = float(
        alpha_seven.loc[
            alpha_seven["outcome_id"] == "strict_control_score_trend",
            "q_value_alpha_seven",
        ].iloc[0]
    )

    fallback_triggered = (not sparse.empty) or bool(category_model_failures)
    fallback_reason = []
    if not sparse.empty:
        fallback_reason.append(
            "one or more score categories were below the prespecified participant/family minimum"
        )
    if category_model_failures:
        fallback_reason.append("one or more category-specific models failed")
    fallback = pd.DataFrame()
    fallback_failures: list[str] = []
    if fallback_triggered:
        fallback, fallback_failures = fit_threshold_ladder(dose, covariates)
        fallback.to_csv(args.out_dir / "contingent_threshold_ladder.csv", index=False)

    if fitted_category.any():
        save_figure(
            category,
            counts,
            args.out_dir / "strict_control_dose_response_forest.png",
        )

    beta_profile = category.loc[
        fitted_category, ["dose_score", "beta_per_global_richness_sd"]
    ].sort_values("dose_score")
    if len(beta_profile) >= 2:
        rho = float(
            stats.spearmanr(
                beta_profile["dose_score"], beta_profile["beta_per_global_richness_sd"]
            ).statistic
        )
        adjacent_strengthening = int(
            np.sum(np.diff(beta_profile["beta_per_global_richness_sd"].to_numpy()) < 0)
        )
    else:
        rho = np.nan
        adjacent_strengthening = 0

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "specification": {
            "path": str(SPECIFICATION),
            "sha256": sha256(SPECIFICATION),
            "fixed_before_new_results": True,
            "chronology_label": "investigator-specified before analysis of these dose-response results",
        },
        "assembly": assembly,
        "feasibility": feasibility,
        "validated_headline_reproduction": headline_check,
        "prior_locked_full_cohort_continuous_score_result": prior_full_score,
        "ordered_trend": trend,
        "category_models": category.to_dict(orient="records"),
        "category_profile_descriptors": {
            "spearman_rho_score_vs_log_odds_ratio": rho,
            "adjacent_steps_with_stronger_inverse_point_estimate": adjacent_strengthening,
            "possible_adjacent_steps": max(0, len(beta_profile) - 1),
            "descriptive_only": True,
        },
        "fallback": {
            "triggered": fallback_triggered,
            "reason": fallback_reason,
            "category_unfitted": category_unfitted,
            "category_model_failures": category_model_failures,
            "threshold_failures": fallback_failures,
            "results": fallback.to_dict(orient="records") if fallback_triggered else [],
        },
        "multiplicity": {
            "primary_dose_response_test": "single adjusted linear trend",
            "conservative_alpha_seven_sensitivity": {
                "description": (
                    "Post-result robustness check adding the trend to the existing six-test "
                    "alpha-diversity family; not used to select the analysis"
                ),
                "trend_q_value": trend_alpha_seven_q,
            },
            "category_specific_models": (
                f"Benjamini-Hochberg across {int(fitted_category.sum())} fitted score contrasts; "
                f"{int((~fitted_category).sum())} prespecified sparse contrasts were not estimated"
            ),
            "fallback_threshold_models": (
                "Benjamini-Hochberg across fitted threshold contrasts"
                if fallback_triggered
                else "not run"
            ),
        },
        "provenance": input_provenance(args.data_root),
        "runtime": {
            "python": sys.version,
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
        },
        "privacy": {
            "participant_level_output_written": False,
            "raw_data_written": False,
            "outputs_are_aggregate_only": True,
        },
    }
    write_json(args.out_dir / "run_summary.json", summary)
    write_json(args.out_dir / "source_and_runtime_provenance.json", summary["provenance"])
    print(json.dumps(summary, indent=2, default=str))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
