from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from run_2013_initial_microbiome_signal_probes import DEFAULT_DATA_ROOT, PROJECT_ROOT, clean_feature_name
from run_2013_locked_primary_16s_analysis import BINARY_OUTCOMES, CORE_COVARIATES, TIMING_BRANCHES, sex_is_estimable
from run_2013_method_strengthening_batch import load_joined_inputs


DEFAULT_OUT_DIR = PROJECT_ROOT / 'runtime' / 'ancombc2_2013_inputs'
PRIMARY_OUTCOMES = [
    "either_often_symptom_or",
    "highly_symptomatic_and",
    "hs_vs_clean_control",
]


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def clean_counts(genus_counts: pd.DataFrame) -> pd.DataFrame:
    renamed = genus_counts.copy()
    renamed.columns = [clean_feature_name(str(col)) for col in renamed.columns]
    renamed = renamed.T.groupby(level=0).sum().T
    return renamed


def build_outcome_inputs(
    base: pd.DataFrame,
    clean_genus_counts: pd.DataFrame,
    outcome_id: str,
    out_dir: Path,
    prevalence_cut: float,
    fixed_taxa: list[str],
    reference_prevalence: pd.Series,
) -> dict[str, Any]:
    outcome_col = BINARY_OUTCOMES[outcome_id]
    branch = base[base["gap_days_abs"] <= TIMING_BRANCHES["within_3y"]].copy()
    branch[outcome_col] = pd.to_numeric(branch[outcome_col], errors="coerce")
    branch = branch.dropna(subset=[outcome_col, *CORE_COVARIATES, "FamilyID"])
    if sex_is_estimable(branch):
        covariates = [*CORE_COVARIATES, "SEX"]
    else:
        covariates = CORE_COVARIATES.copy()
    branch = branch.dropna(subset=[c for c in covariates if c in branch.columns])
    branch["sample_id"] = branch["SequencingSpecificName"].astype(str)
    branch["dry_eye_status"] = branch[outcome_col].astype(int).map({0: "Control", 1: "Case"})
    branch = branch.dropna(subset=["dry_eye_status"]).copy()

    counts = clean_genus_counts
    counts = counts.loc[counts.index.intersection(branch["sample_id"])]
    branch = branch.set_index("sample_id").loc[counts.index].reset_index().rename(columns={"index": "sample_id"})
    if "sample_id" not in branch.columns:
        branch.insert(0, "sample_id", counts.index.astype(str))
    analysis_prevalence = (counts > 0).mean(axis=0)
    counts = counts[fixed_taxa]

    outcome_dir = out_dir / outcome_id
    outcome_dir.mkdir(parents=True, exist_ok=True)
    counts_out = counts.T.copy()
    counts_out.insert(0, "taxon", counts_out.index)
    counts_out.to_csv(outcome_dir / "genus_counts.csv", index=False)

    branch["total_reads"] = pd.to_numeric(branch["alpha_total_reads"], errors="coerce")
    meta_cols = [
        "sample_id",
        "dry_eye_status",
        "age",
        "BMI",
        "AntibioticsPriorMonth",
        "gap_days_abs",
        "total_reads",
        "FamilyID",
    ]
    if "SEX" in covariates:
        meta_cols.append("SEX")
    metadata = branch[meta_cols].copy()
    metadata["AntibioticsPriorMonth"] = metadata["AntibioticsPriorMonth"].fillna("Unknown").astype(str)
    metadata["FamilyID"] = metadata["FamilyID"].fillna(metadata["sample_id"]).astype(str)
    if "SEX" in metadata:
        metadata["SEX"] = metadata["SEX"].fillna("Unknown").astype(str)
    metadata.to_csv(outcome_dir / "metadata.csv", index=False)

    feature_manifest = pd.DataFrame(
        {
            "taxon": fixed_taxa,
            "reference_cohort_prevalence": [float(reference_prevalence[t]) for t in fixed_taxa],
            "analysis_sample_prevalence": [float(analysis_prevalence[t]) for t in fixed_taxa],
            "total_count": [float(counts[t].sum()) for t in fixed_taxa],
        }
    ).sort_values(["reference_cohort_prevalence", "total_count", "taxon"], ascending=[False, False, True])
    feature_manifest.to_csv(outcome_dir / "feature_manifest.csv", index=False)

    return {
        "outcome_id": outcome_id,
        "outcome_col": outcome_col,
        "samples": int(len(metadata)),
        "cases": int((metadata["dry_eye_status"] == "Case").sum()),
        "controls": int((metadata["dry_eye_status"] == "Control").sum()),
        "families": int(metadata["FamilyID"].nunique()),
        "taxa": int(len(fixed_taxa)),
        "prevalence_cut": prevalence_cut,
        "prevalence_reference": "complete-case within-3-year highly_symptomatic_and cohort",
        "fixed_universe_across_outcomes": True,
        "unresolved_genus_policy": "exclude ASVs without assigned genus from genus-level counts",
        "tie_rule": "highest_read_depth",
        "covariates": covariates,
        "count_file": str(outcome_dir / "genus_counts.csv"),
        "metadata_file": str(outcome_dir / "metadata.csv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 2013 dry-eye ANCOM-BC2 input tables.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prevalence-cut", type=float, default=0.10)
    parser.add_argument("--outcomes", default=",".join(PRIMARY_OUTCOMES))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base, _wide, genus_counts = load_joined_inputs(args.data_root)
    clean_genus_counts = clean_counts(genus_counts)
    reference_outcome = BINARY_OUTCOMES["highly_symptomatic_and"]
    reference = base[base["gap_days_abs"] <= TIMING_BRANCHES["within_3y"]].copy()
    reference[reference_outcome] = pd.to_numeric(reference[reference_outcome], errors="coerce")
    reference = reference.dropna(subset=[reference_outcome, *CORE_COVARIATES, "FamilyID"])
    reference_covariates = [*CORE_COVARIATES, "SEX"] if sex_is_estimable(reference) else CORE_COVARIATES.copy()
    reference = reference.dropna(subset=[c for c in reference_covariates if c in reference.columns])
    reference_samples = reference["SequencingSpecificName"].astype(str)
    reference_counts = clean_genus_counts.loc[clean_genus_counts.index.intersection(reference_samples)]
    reference_prevalence = (reference_counts > 0).mean(axis=0)
    fixed_taxa = sorted(reference_prevalence[reference_prevalence >= args.prevalence_cut].index.astype(str).tolist())
    if not fixed_taxa:
        raise RuntimeError("The fixed prevalence filter retained no genera.")
    fixed_taxa_sha256 = hashlib.sha256("\n".join(fixed_taxa).encode("utf-8")).hexdigest()
    manifests = []
    for outcome_id in [x.strip() for x in args.outcomes.split(",") if x.strip()]:
        manifests.append(
            build_outcome_inputs(
                base,
                clean_genus_counts,
                outcome_id,
                args.out_dir,
                args.prevalence_cut,
                fixed_taxa,
                reference_prevalence,
            )
        )
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.out_dir),
        "source": "2013 within-3y nearest stool 16S; genus counts collapsed from ASV counts",
        "private_row_level_inputs": True,
        "commit_guidance": "Do not commit row-level count or metadata files.",
        "prevalence_reference": "complete-case within-3-year highly_symptomatic_and cohort",
        "reference_samples": int(len(reference_counts)),
        "fixed_universe_taxa": int(len(fixed_taxa)),
        "fixed_taxa_sha256": fixed_taxa_sha256,
        "fixed_taxa": fixed_taxa,
        "outcomes": manifests,
    }
    write_json(summary, args.out_dir / "input_manifest.json")
    print(json.dumps(summary, indent=2))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
