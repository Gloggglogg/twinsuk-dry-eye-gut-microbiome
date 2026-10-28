from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
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
    ALPHA_FEATURES,
    CORE_COVARIATES,
    binary_association,
    build_x,
    sex_is_estimable,
)


PRIMARY_OUTCOME = "highly_symptomatic_derived"
PRIMARY_OUTCOME_ID = "highly_symptomatic_and"
PRIMARY_FEATURE = "alpha_richness"
PRIMARY_MAX_GAP_DAYS = 365 * 3
ONE_YEAR_MAX_GAP_DAYS = 365
SHORT_UNMERGED_RUN = "TwinsUKPlate28_33"
ALPHA_OUTCOMES = {
    "either_often_symptom_or": "either_often_symptom",
    "highly_symptomatic_and": "highly_symptomatic_derived",
    "hs_vs_clean_control": "hs_vs_clean_control",
}
REFERENCE_PRIMARY_DIR = PROJECT_ROOT / 'runtime' / 'coauthor_review_production_2026_08_14' / 'primary'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def primary_complete_cases(data: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    cols = [PRIMARY_OUTCOME, PRIMARY_FEATURE, "FamilyID", *covariates]
    model = data[[col for col in cols if col in data.columns]].copy()
    model[PRIMARY_OUTCOME] = pd.to_numeric(model[PRIMARY_OUTCOME], errors="coerce")
    model[PRIMARY_FEATURE] = pd.to_numeric(model[PRIMARY_FEATURE], errors="coerce")
    model = model.dropna(subset=[PRIMARY_OUTCOME, PRIMARY_FEATURE])
    design = build_x(model, PRIMARY_FEATURE, covariates)
    keep = design.notna().all(axis=1)
    return data.loc[model.index[keep]].copy()


def reproduce_alpha_family(
    within_3y: pd.DataFrame,
    covariates: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome_id, outcome_col in ALPHA_OUTCOMES.items():
        for feature in ALPHA_FEATURES:
            result = binary_association(within_3y, outcome_col, feature, covariates)
            if not result:
                raise RuntimeError(f"Could not reproduce alpha-diversity model: {outcome_id}/{feature}")
            rows.append({"outcome_id": outcome_id, "feature": feature, **result})
    frame = pd.DataFrame(rows)
    if len(frame) != 6:
        raise RuntimeError(f"Expected six alpha-diversity tests; reproduced {len(frame)}")
    frame["q_value_alpha_six"] = multipletests(frame["p_value"], method="fdr_bh")[1]
    return frame


def compare_headline_to_reference(
    reproduced: pd.Series,
    reference_dir: Path,
) -> dict[str, Any]:
    reference_path = reference_dir / "locked_primary_16s_association_matrix.csv"
    if not reference_path.exists():
        raise FileNotFoundError(f"Validated production reference is missing: {reference_path}")
    reference = pd.read_csv(reference_path)
    mask = (
        (reference["branch_id"] == "within_3y")
        & (reference["outcome_id"] == PRIMARY_OUTCOME_ID)
        & (reference["outcome_type"] == "binary")
        & (reference["covariate_set"] == "primary_core")
        & (reference["feature"] == PRIMARY_FEATURE)
    )
    if int(mask.sum()) != 1:
        raise RuntimeError(f"Expected one validated headline row; found {int(mask.sum())}")
    expected = reference.loc[mask].iloc[0]
    checks: dict[str, Any] = {}
    integer_fields = ["n", "cases", "controls", "families"]
    numeric_fields = [
        "beta_per_sd",
        "standard_error",
        "ci_low_beta",
        "ci_high_beta",
        "odds_ratio_per_sd",
        "ci_low_or",
        "ci_high_or",
        "p_value",
        "q_value_alpha_six",
    ]
    for field in integer_fields:
        observed_value = int(reproduced[field])
        expected_value = int(expected[field])
        checks[field] = {
            "observed": observed_value,
            "expected": expected_value,
            "pass": observed_value == expected_value,
        }
    for field in numeric_fields:
        observed_value = float(reproduced[field])
        expected_value = float(expected[field])
        checks[field] = {
            "observed": observed_value,
            "expected": expected_value,
            "absolute_difference": abs(observed_value - expected_value),
            "pass": bool(np.isclose(observed_value, expected_value, rtol=0.0, atol=1e-12)),
        }
    passed = all(item["pass"] for item in checks.values())
    return {
        "reference_path": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "passed": passed,
        "checks": checks,
    }


def classify_pair_structure(main_complete: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = (
        main_complete.groupby("FamilyID", sort=False)
        .agg(
            participants=("FamilyID", "size"),
            cases=(PRIMARY_OUTCOME, "sum"),
            controls=(PRIMARY_OUTCOME, lambda value: len(value) - int(value.sum())),
        )
        .reset_index()
    )
    family["cases"] = family["cases"].astype(int)
    family["controls"] = family["controls"].astype(int)

    def label(row: pd.Series) -> str:
        if row["participants"] == 2 and row["cases"] == 1:
            return "phenotype_discordant_twin_pair"
        if row["participants"] == 2 and row["cases"] == 2:
            return "case_concordant_twin_pair"
        if row["participants"] == 2 and row["cases"] == 0:
            return "control_concordant_twin_pair"
        if row["participants"] == 1 and row["cases"] == 1:
            return "case_singleton"
        if row["participants"] == 1 and row["cases"] == 0:
            return "control_singleton"
        return "other_family_structure"

    family["structure"] = family.apply(label, axis=1)
    order = [
        "phenotype_discordant_twin_pair",
        "case_concordant_twin_pair",
        "control_concordant_twin_pair",
        "case_singleton",
        "control_singleton",
        "other_family_structure",
    ]
    summary = (
        family.groupby("structure", sort=False)
        .agg(families=("FamilyID", "size"), participants=("participants", "sum"), cases=("cases", "sum"))
        .reindex(order, fill_value=0)
        .reset_index()
    )
    summary["controls"] = summary["participants"] - summary["cases"]
    return family, summary


def within_pair_analysis(
    main_complete: pd.DataFrame,
    family_structure: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    discordant_ids = family_structure.loc[
        family_structure["structure"] == "phenotype_discordant_twin_pair", "FamilyID"
    ]
    pair_rows = main_complete[main_complete["FamilyID"].isin(discordant_ids)].copy()
    richness_mean = float(main_complete[PRIMARY_FEATURE].mean())
    richness_sd = float(main_complete[PRIMARY_FEATURE].std(ddof=1))
    pair_rows["richness_z_main_cohort"] = (pair_rows[PRIMARY_FEATURE] - richness_mean) / richness_sd
    pair_rows = pair_rows.sort_values(["FamilyID", PRIMARY_OUTCOME])
    if pair_rows.empty or pair_rows["FamilyID"].nunique() != len(discordant_ids):
        raise RuntimeError("Discordant-pair extraction failed")

    y = pair_rows[PRIMARY_OUTCOME].astype(int)
    x = pair_rows[["richness_z_main_cohort"]].astype(float)
    groups = pair_rows["FamilyID"].astype(str)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fit = ConditionalLogit(y, x, groups=groups).fit(method="bfgs", maxiter=1000, disp=False)
    beta = float(fit.params["richness_z_main_cohort"])
    standard_error = float(fit.bse["richness_z_main_cohort"])
    ci_low_beta = beta - 1.96 * standard_error
    ci_high_beta = beta + 1.96 * standard_error
    result = pd.DataFrame(
        [
            {
                "analysis": "conditional_logistic_discordant_twin_pairs",
                "pairs": int(groups.nunique()),
                "participants": int(len(pair_rows)),
                "cases": int(y.sum()),
                "controls": int(len(y) - y.sum()),
                "feature_scale": "per 1 SD observed ASV richness in the 1,685-participant main complete-case cohort",
                "beta_per_sd": beta,
                "standard_error": standard_error,
                "odds_ratio_per_sd": math.exp(beta),
                "ci_low_or": math.exp(ci_low_beta),
                "ci_high_or": math.exp(ci_high_beta),
                "p_value": float(fit.pvalues["richness_z_main_cohort"]),
                "converged": bool(getattr(fit, "mle_retvals", {}).get("converged", True)),
                "warning_count": len(captured),
                "warnings": " | ".join(str(item.message) for item in captured),
                "interpretation_boundary": "within-pair sensitivity; not independent replication",
            }
        ]
    )

    differences: list[float] = []
    raw_differences: list[float] = []
    for _, group in pair_rows.groupby("FamilyID", sort=False):
        case = group.loc[group[PRIMARY_OUTCOME] == 1].iloc[0]
        control = group.loc[group[PRIMARY_OUTCOME] == 0].iloc[0]
        differences.append(float(case["richness_z_main_cohort"] - control["richness_z_main_cohort"]))
        raw_differences.append(float(case[PRIMARY_FEATURE] - control[PRIMARY_FEATURE]))
    diff = np.asarray(differences, dtype=float)
    raw_diff = np.asarray(raw_differences, dtype=float)
    se = float(diff.std(ddof=1) / math.sqrt(len(diff)))
    critical = float(stats.t.ppf(0.975, df=len(diff) - 1))
    try:
        wilcoxon_p = float(stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    descriptive = pd.DataFrame(
        [
            {
                "pairs": int(len(diff)),
                "mean_case_minus_control_richness_sd": float(diff.mean()),
                "mean_difference_ci_low": float(diff.mean() - critical * se),
                "mean_difference_ci_high": float(diff.mean() + critical * se),
                "median_case_minus_control_richness_sd": float(np.median(diff)),
                "mean_case_minus_control_asv_count": float(raw_diff.mean()),
                "median_case_minus_control_asv_count": float(np.median(raw_diff)),
                "pairs_case_lower_richness": int((diff < 0).sum()),
                "pairs_equal_richness": int((diff == 0).sum()),
                "pairs_case_higher_richness": int((diff > 0).sum()),
                "wilcoxon_signed_rank_p": wilcoxon_p,
            }
        ]
    )
    return result, descriptive


def timing_summaries(base: pd.DataFrame, main_complete: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort_id, data in [("full_3y_cohort", base), ("main_complete_case", main_complete)]:
        signed_stool_minus_questionnaire = -pd.to_numeric(data["gap_days_signed"], errors="coerce")
        timing = np.select(
            [signed_stool_minus_questionnaire < 0, signed_stool_minus_questionnaire > 0],
            ["stool_before_questionnaire", "stool_after_questionnaire"],
            default="same_day",
        )
        frame = data.assign(
            signed_stool_minus_questionnaire_days=signed_stool_minus_questionnaire,
            stool_timing=timing,
        )
        for category in ["stool_before_questionnaire", "same_day", "stool_after_questionnaire"]:
            subset = frame[frame["stool_timing"] == category]
            rows.append(
                {
                    "cohort": cohort_id,
                    "timing_category": category,
                    "participants": int(len(subset)),
                    "cases": int(subset[PRIMARY_OUTCOME].fillna(0).sum()),
                    "percent_of_cohort": float(100 * len(subset) / len(frame)),
                    "median_signed_stool_minus_questionnaire_days": (
                        float(subset["signed_stool_minus_questionnaire_days"].median())
                        if len(subset)
                        else np.nan
                    ),
                    "median_absolute_interval_days": (
                        float(subset["gap_days_abs"].median()) if len(subset) else np.nan
                    ),
                }
            )
        rows.append(
            {
                "cohort": cohort_id,
                "timing_category": "all",
                "participants": int(len(frame)),
                "cases": int(frame[PRIMARY_OUTCOME].fillna(0).sum()),
                "percent_of_cohort": 100.0,
                "median_signed_stool_minus_questionnaire_days": float(
                    frame["signed_stool_minus_questionnaire_days"].median()
                ),
                "median_absolute_interval_days": float(frame["gap_days_abs"].median()),
            }
        )
    return pd.DataFrame(rows)


def timing_interaction_analysis(
    main_complete: pd.DataFrame,
    covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = main_complete[main_complete["gap_days_signed"] != 0].copy()
    model["stool_after_questionnaire"] = (model["gap_days_signed"] < 0).astype(float)
    richness_mean = float(main_complete[PRIMARY_FEATURE].mean())
    richness_sd = float(main_complete[PRIMARY_FEATURE].std(ddof=1))
    design = build_x(model, PRIMARY_FEATURE, covariates)
    keep = design.notna().all(axis=1)
    model = model.loc[keep].copy()
    design = design.loc[keep].copy()
    design["feature_z"] = (model[PRIMARY_FEATURE] - richness_mean) / richness_sd
    design["stool_after_questionnaire"] = model["stool_after_questionnaire"]
    design["feature_z_by_stool_after"] = (
        design["feature_z"] * design["stool_after_questionnaire"]
    )
    design = sm.add_constant(design.astype(float), has_constant="add")
    y = model[PRIMARY_OUTCOME].astype(int)
    groups = model["FamilyID"].astype(str)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )

    beta_before = float(fit.params["feature_z"])
    beta_interaction = float(fit.params["feature_z_by_stool_after"])
    beta_after = beta_before + beta_interaction
    covariance = fit.cov_params()
    se_before = float(fit.bse["feature_z"])
    se_after = math.sqrt(
        float(
            covariance.loc["feature_z", "feature_z"]
            + covariance.loc["feature_z_by_stool_after", "feature_z_by_stool_after"]
            + 2 * covariance.loc["feature_z", "feature_z_by_stool_after"]
        )
    )
    se_interaction = float(fit.bse["feature_z_by_stool_after"])
    results = pd.DataFrame(
        [
            {
                "timing_group": "stool_before_questionnaire",
                "participants": int((model["stool_after_questionnaire"] == 0).sum()),
                "cases": int(model.loc[model["stool_after_questionnaire"] == 0, PRIMARY_OUTCOME].sum()),
                "odds_ratio_per_sd": math.exp(beta_before),
                "ci_low_or": math.exp(beta_before - 1.96 * se_before),
                "ci_high_or": math.exp(beta_before + 1.96 * se_before),
                "timing_interaction_p": float(fit.pvalues["feature_z_by_stool_after"]),
                "ratio_of_odds_ratios_after_vs_before": math.exp(beta_interaction),
                "ratio_ci_low": math.exp(beta_interaction - 1.96 * se_interaction),
                "ratio_ci_high": math.exp(beta_interaction + 1.96 * se_interaction),
            },
            {
                "timing_group": "stool_after_questionnaire",
                "participants": int((model["stool_after_questionnaire"] == 1).sum()),
                "cases": int(model.loc[model["stool_after_questionnaire"] == 1, PRIMARY_OUTCOME].sum()),
                "odds_ratio_per_sd": math.exp(beta_after),
                "ci_low_or": math.exp(beta_after - 1.96 * se_after),
                "ci_high_or": math.exp(beta_after + 1.96 * se_after),
                "timing_interaction_p": float(fit.pvalues["feature_z_by_stool_after"]),
                "ratio_of_odds_ratios_after_vs_before": math.exp(beta_interaction),
                "ratio_ci_low": math.exp(beta_interaction - 1.96 * se_interaction),
                "ratio_ci_high": math.exp(beta_interaction + 1.96 * se_interaction),
            },
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "participants": int(len(model)),
                "cases": int(y.sum()),
                "families": int(groups.nunique()),
                "same_day_excluded": int((main_complete["gap_days_signed"] == 0).sum()),
                "covariance_method": "family-clustered robust",
                "interaction_definition": "observed richness by stool collected after versus before questionnaire",
                "feature_scale": "per 1 SD observed ASV richness in the main complete-case cohort",
                "absolute_interval_adjusted": "yes",
                "warning_count": len(captured),
                "warnings": " | ".join(str(item.message) for item in captured),
            }
        ]
    )
    return results, diagnostics


def source_provenance(data_root: Path, script_path: Path) -> dict[str, Any]:
    sources = [
        data_root / DED_2013,
        data_root / MICROBIOME_DIR / "TwinsUK_ASVs_MappingFile.xlsx",
        data_root / MICROBIOME_DIR / "TwinsUK_ASVsCR.rds",
        data_root / MICROBIOME_DIR / "TwinsUK_ASVs_Taxa.csv.zip",
        data_root / SPINE,
    ]
    records = []
    for path in sources:
        if not path.exists():
            raise FileNotFoundError(f"Required source is missing: {path}")
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256(path),
                "access_mode": "read_only",
            }
        )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": records,
        "script": {"path": str(script_path), "sha256": sha256(script_path)},
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
        "raw_or_external_data_written": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the headline richness result and run pair/timing publishability sensitivities."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "publishability_audit_2026_08_21" / "sensitivities",
    )
    parser.add_argument("--reference-primary-dir", type=Path, default=REFERENCE_PRIMARY_DIR)
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
        tie_rule="highest_read_depth",
    )
    alpha, _ = compute_alpha_and_genus(counts, taxa)
    base = pairs.merge(alpha, on="SequencingSpecificName", how="inner")
    within_3y = base[base["gap_days_abs"] <= PRIMARY_MAX_GAP_DAYS].copy()
    include_sex = sex_is_estimable(within_3y)
    covariates = [*CORE_COVARIATES, *(["SEX"] if include_sex else [])]

    alpha_family = reproduce_alpha_family(within_3y, covariates)
    headline = alpha_family.loc[
        (alpha_family["outcome_id"] == PRIMARY_OUTCOME_ID)
        & (alpha_family["feature"] == PRIMARY_FEATURE)
    ].iloc[0]
    comparison = compare_headline_to_reference(headline, args.reference_primary_dir)
    alpha_family.to_csv(args.out_dir / "headline_alpha_six_reproduction.csv", index=False)
    write_json(args.out_dir / "headline_reproduction_check.json", comparison)
    if not comparison["passed"]:
        raise RuntimeError(
            "Headline result does not reproduce the validated production reference; "
            "see headline_reproduction_check.json"
        )

    main_complete = primary_complete_cases(within_3y, covariates)
    crude_same_sample = binary_association(
        main_complete,
        PRIMARY_OUTCOME,
        PRIMARY_FEATURE,
        [],
    )
    if not crude_same_sample:
        raise RuntimeError("The same-sample unadjusted richness model could not be fitted")
    pd.DataFrame([crude_same_sample]).to_csv(
        args.out_dir / "unadjusted_same_sample_richness.csv",
        index=False,
    )
    family_structure, pair_summary = classify_pair_structure(main_complete)
    pair_summary.to_csv(args.out_dir / "pair_structure_summary.csv", index=False)
    within_pair, pair_descriptive = within_pair_analysis(main_complete, family_structure)
    within_pair.to_csv(args.out_dir / "within_pair_conditional_logistic.csv", index=False)
    pair_descriptive.to_csv(args.out_dir / "within_pair_descriptive.csv", index=False)

    timing_summary = timing_summaries(within_3y, main_complete)
    timing_summary.to_csv(args.out_dir / "timing_summary.csv", index=False)
    timing_effects, timing_diagnostics = timing_interaction_analysis(main_complete, covariates)
    timing_effects.to_csv(args.out_dir / "timing_interaction_effects.csv", index=False)
    timing_diagnostics.to_csv(args.out_dir / "timing_interaction_diagnostics.csv", index=False)

    within_1y = within_3y[within_3y["gap_days_abs"] <= ONE_YEAR_MAX_GAP_DAYS].copy()
    one_year = binary_association(within_1y, PRIMARY_OUTCOME, PRIMARY_FEATURE, covariates)
    if not one_year:
        raise RuntimeError("The prespecified 1-year richness sensitivity could not be fitted")
    pd.DataFrame([one_year]).to_csv(args.out_dir / "one_year_richness_sensitivity.csv", index=False)

    if "SequencingRunID" not in main_complete.columns:
        raise RuntimeError("SequencingRunID is missing; cannot audit the documented Plate28_33 anomaly")
    short_run = main_complete["SequencingRunID"].astype(str).eq(SHORT_UNMERGED_RUN)
    without_short_run = main_complete.loc[~short_run].copy()
    short_run_exclusion = binary_association(
        without_short_run,
        PRIMARY_OUTCOME,
        PRIMARY_FEATURE,
        covariates,
    )
    if not short_run_exclusion:
        raise RuntimeError("The documented Plate28_33 exclusion sensitivity could not be fitted")
    short_run_exclusion.update(
        {
            "excluded_sequencing_run": SHORT_UNMERGED_RUN,
            "excluded_participants": int(short_run.sum()),
            "excluded_cases": int(main_complete.loc[short_run, PRIMARY_OUTCOME].sum()),
            "reason": "README states that this run contains short ASVs that did not merge with the wider ASV dataset",
        }
    )
    pd.DataFrame([short_run_exclusion]).to_csv(
        args.out_dir / "plate28_33_exclusion_sensitivity.csv",
        index=False,
    )

    provenance = source_provenance(args.data_root, Path(__file__).resolve())
    write_json(args.out_dir / "source_and_runtime_provenance.json", provenance)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "headline_reproduced": comparison["passed"],
        "headline": headline.to_dict(),
        "main_complete_case": {
            "participants": int(len(main_complete)),
            "cases": int(main_complete[PRIMARY_OUTCOME].sum()),
            "controls": int(len(main_complete) - main_complete[PRIMARY_OUTCOME].sum()),
            "families": int(main_complete["FamilyID"].nunique()),
        },
        "unadjusted_same_sample": crude_same_sample,
        "pair_structure": pair_summary.to_dict(orient="records"),
        "within_pair": within_pair.iloc[0].to_dict(),
        "within_pair_descriptive": pair_descriptive.iloc[0].to_dict(),
        "timing": timing_summary.to_dict(orient="records"),
        "timing_interaction": timing_effects.to_dict(orient="records"),
        "one_year": one_year,
        "documented_plate28_33_exclusion": short_run_exclusion,
        "method_boundaries": {
            "within_pair": "Conditional logistic regression includes phenotype-discordant two-person twin families only and is a robustness analysis, not independent replication.",
            "timing": "One prespecified before-versus-after richness interaction was fitted; four same-day samples were excluded from that contrast and retained in all other analyses.",
            "taxa": "No additional taxon screening was performed.",
            "technical_run": "The single documented short-unmerged-ASV run was excluded as a targeted robustness check; no high-dimensional batch-adjustment model was added.",
        },
        "no_participant_level_output_written": True,
        "no_raw_data_written": True,
    }
    write_json(args.out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
