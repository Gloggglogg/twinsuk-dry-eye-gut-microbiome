from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from run_2013_adjusted_probability_contrast import (
    FEATURE,
    OUTCOME,
    assemble_main_cohort,
    fit_point_model,
    prepare_design,
)
from run_2013_initial_microbiome_signal_probes import DEFAULT_DATA_ROOT, MICROBIOME_DIR, PROJECT_ROOT


DEFAULT_OUT_DIR = PROJECT_ROOT / 'runtime' / 'antibiotic_metadata_assumption_2026_08_24'
MAPPING_FILE = "TwinsUK_ASVs_MappingFile.xlsx"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def fit_scenario(model: pd.DataFrame, covariates: list[str], scenario: str) -> dict[str, Any]:
    data = model.copy()
    scenario_covariates = list(covariates)

    if scenario == "unknown_as_no":
        data["AntibioticsPriorMonth"] = data["AntibioticsPriorMonth"].replace("Unknown", "N")
    elif scenario == "known_only":
        data = data.loc[data["AntibioticsPriorMonth"] != "Unknown"].copy()
    elif scenario == "antibiotic_omitted":
        scenario_covariates = [item for item in scenario_covariates if item != "AntibioticsPriorMonth"]
    elif scenario != "three_level_primary":
        raise ValueError(f"Unknown scenario: {scenario}")

    y, design, groups, _, richness_sd = prepare_design(data, scenario_covariates)
    fit, point = fit_point_model(y, design, groups)
    beta = float(fit.params["feature_z"])
    se = float(fit.bse["feature_z"])
    return {
        "scenario": scenario,
        "n": int(len(data)),
        "cases": int(y.sum()),
        "families": int(groups.nunique()),
        "richness_sd": richness_sd,
        "beta_per_sd": beta,
        "standard_error": se,
        "odds_ratio_per_sd": math.exp(beta),
        "ci_low_or": math.exp(beta - 1.96 * se),
        "ci_high_or": math.exp(beta + 1.96 * se),
        "p_value": float(fit.pvalues["feature_z"]),
        "converged": bool(fit.converged),
    }


def raw_mapping_summary(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = pd.read_excel(data_root / MICROBIOME_DIR / MAPPING_FILE)
    antibiotic_col = "AntibioticsPriorMonth"
    date_col = "CollectionDateCombUseReceiveDateIfNothingOnTube"
    if antibiotic_col not in raw.columns or date_col not in raw.columns:
        raise RuntimeError("Expected antibiotic or collection-date field is absent from the mapping file")

    raw_value = raw[antibiotic_col]
    status = raw_value.astype("string").str.strip()
    status = status.mask(raw_value.isna() | status.eq(""), "<blank>")
    status = status.fillna("<blank>")
    dates = pd.to_datetime(raw[date_col], errors="coerce")

    overall = (
        status.value_counts(dropna=False)
        .rename_axis("raw_antibiotic_value")
        .reset_index(name="profiles")
    )
    overall["percent"] = overall["profiles"] / len(raw) * 100

    by_year = pd.DataFrame({"collection_year": dates.dt.year, "raw_antibiotic_value": status})
    by_year = (
        by_year.dropna(subset=["collection_year"])
        .groupby(["collection_year", "raw_antibiotic_value"], dropna=False)
        .size()
        .rename("profiles")
        .reset_index()
    )
    totals = by_year.groupby("collection_year")["profiles"].transform("sum")
    by_year["percent_within_year"] = by_year["profiles"] / totals * 100

    meta = {
        "mapping_rows": int(len(raw)),
        "antibiotic_field": antibiotic_col,
        "collection_date_field": date_col,
        "raw_missing_rows": int(raw_value.isna().sum()),
        "nonmissing_distinct_values": sorted(status.loc[status != "<blank>"].unique().tolist()),
    }
    return overall, by_year, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether unknown antibiotic metadata can be recoded as no")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, covariates = assemble_main_cohort(args.data_root)
    group_summary = (
        model.groupby("AntibioticsPriorMonth", dropna=False)
        .agg(
            participants=(OUTCOME, "size"),
            cases=(OUTCOME, "sum"),
            families=("FamilyID", "nunique"),
            mean_richness=(FEATURE, "mean"),
        )
        .reset_index()
        .rename(columns={"AntibioticsPriorMonth": "antibiotic_status"})
    )
    group_summary["case_percent"] = group_summary["cases"] / group_summary["participants"] * 100

    scenarios = [
        fit_scenario(model, covariates, "three_level_primary"),
        fit_scenario(model, covariates, "unknown_as_no"),
        fit_scenario(model, covariates, "antibiotic_omitted"),
        fit_scenario(model, covariates, "known_only"),
    ]
    scenario_df = pd.DataFrame(scenarios)
    raw_overall, raw_by_year, raw_meta = raw_mapping_summary(args.data_root)

    group_summary.to_csv(args.out_dir / "analysis_cohort_by_antibiotic_status.csv", index=False)
    scenario_df.to_csv(args.out_dir / "richness_model_antibiotic_scenarios.csv", index=False)
    raw_overall.to_csv(args.out_dir / "raw_mapping_antibiotic_values.csv", index=False)
    raw_by_year.to_csv(args.out_dir / "raw_mapping_antibiotic_values_by_collection_year.csv", index=False)

    known = group_summary.loc[group_summary["antibiotic_status"].isin(["N", "Y"])]
    known_total = int(known["participants"].sum())
    known_yes = int(known.loc[known["antibiotic_status"] == "Y", "participants"].sum())
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_population": int(len(model)),
        "analysis_cases": int(model[OUTCOME].sum()),
        "analysis_families": int(model["FamilyID"].nunique()),
        "unknown_participants": int((model["AntibioticsPriorMonth"] == "Unknown").sum()),
        "unknown_percent": float((model["AntibioticsPriorMonth"] == "Unknown").mean() * 100),
        "reported_yes_participants": int((model["AntibioticsPriorMonth"] == "Y").sum()),
        "reported_yes_percent_overall": float((model["AntibioticsPriorMonth"] == "Y").mean() * 100),
        "reported_yes_percent_among_known": (known_yes / known_total * 100) if known_total else None,
        "raw_mapping": raw_meta,
        "model_scenarios": scenarios,
        "interpretation": {
            "unknown_is_observed_no": False,
            "reason": "The source field is blank for these records. No documentary source for an antibiotic-avoidance instruction was verified in the supplied metadata or cited papers. Even if such advice is later confirmed, it would not establish individual non-use.",
            "recommended_primary_handling": "Retain no, yes and unknown as separate sample-metadata categories.",
            "recommended_sensitivity": "Report the unknown-as-no and known-only model estimates as coding checks if they materially differ from the primary estimate.",
        },
        "privacy": {
            "participant_level_output_written": False,
            "raw_data_written": False,
            "outputs_are_aggregate_only": True,
        },
    }
    write_json(args.out_dir / "run_summary.json", summary)

    lines = [
        "# Antibiotic metadata assumption audit",
        "",
        f"- Analysis population: {len(model):,}; highly symptomatic cases: {int(model[OUTCOME].sum()):,}.",
        f"- Unknown metadata: {summary['unknown_participants']:,} ({summary['unknown_percent']:.1f}%).",
        f"- Reported use: {summary['reported_yes_participants']:,} ({summary['reported_yes_percent_overall']:.1f}% overall; {summary['reported_yes_percent_among_known']:.1f}% among records with known status).",
        "- The source values classified as unknown are blank, not observed negative responses.",
        "- Recommendation: retain no, yes and unknown as separate categories. The metadata support reporting recorded status, but not an unverified avoidance instruction. Confirm any instruction separately before describing it in the manuscript.",
        "",
        "## Model comparison",
        "",
        scenario_df.to_markdown(index=False),
        "",
        "## Analysis cohort by metadata category",
        "",
        group_summary.to_markdown(index=False),
        "",
    ]
    (args.out_dir / "ANTIBIOTIC_METADATA_ASSUMPTION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
