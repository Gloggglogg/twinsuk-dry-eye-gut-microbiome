from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from run_2013_initial_microbiome_signal_probes import (
    DEFAULT_DATA_ROOT,
    DED_2013,
    MICROBIOME_DIR,
    PROJECT_ROOT,
    SPINE,
    load_ded_2013,
)
from run_2013_locked_primary_16s_analysis import binary_association
from run_2013_symptom_dose_response import (
    REFERENCE_MATRIX,
    RICHNESS,
    assemble_analysis_population,
)


SPECIFICATION = Path(__file__).resolve().parents[1] / 'runtime' / 'documentation' / 'WHS_DED_SENSITIVITY_SPECIFICATION_2026-08-24.md'
DOSE_RESULT = PROJECT_ROOT / 'runtime' / 'symptom_dose_response_2026_08_24' / 'ordered_trend_model.json'
DEFAULT_OUT_DIR = PROJECT_ROOT / 'runtime' / 'whs_ded_sensitivity_2026_08_24'
OUTCOME = "whs_ded_derived"


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


def verify_source_definition(data_root: Path) -> dict[str, Any]:
    ded = load_ded_2013(data_root)
    source = pd.to_numeric(ded["WHS DED"], errors="coerce")
    derived = pd.to_numeric(ded[OUTCOME], errors="coerce")
    comparable = source.notna() & derived.notna()
    agreement = source.loc[comparable].astype(int) == derived.loc[comparable].astype(int)
    if not bool(agreement.all()):
        raise RuntimeError("The supplied and derived WHS DED variables disagree")
    hs = pd.to_numeric(ded["highly_symptomatic_derived"], errors="coerce") == 1
    dx = pd.to_numeric(ded["Ever dx DED"], errors="coerce") == 1
    return {
        "questionnaire_rows": int(len(ded)),
        "complete_whs_inputs": int(comparable.sum()),
        "source_whs_cases": int((source.loc[comparable] == 1).sum()),
        "derived_whs_cases": int((derived.loc[comparable] == 1).sum()),
        "highly_symptomatic_cases": int((hs & comparable).sum()),
        "ever_diagnosed_cases": int((dx & comparable).sum()),
        "highly_symptomatic_only": int((hs & ~dx & comparable).sum()),
        "ever_diagnosed_only": int((dx & ~hs & comparable).sum()),
        "both_components": int((hs & dx & comparable).sum()),
        "exact_agreement": True,
    }


def alpha_eight_sensitivity(whs_p_value: float) -> pd.DataFrame:
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
        raise RuntimeError(f"Expected six existing alpha-diversity tests; found {len(alpha_six)}")
    trend = json.loads(DOSE_RESULT.read_text(encoding="utf-8"))
    expanded = pd.concat(
        [
            alpha_six,
            pd.DataFrame(
                [
                    {
                        "outcome_id": "strict_control_score_trend",
                        "feature": RICHNESS,
                        "p_value": float(trend["p_value"]),
                    },
                    {
                        "outcome_id": "whs_ded",
                        "feature": RICHNESS,
                        "p_value": float(whs_p_value),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    expanded["q_value_alpha_eight"] = multipletests(
        expanded["p_value"], method="fdr_bh"
    )[1]
    expanded["sensitivity_status"] = (
        "conservative expanded multiplicity sensitivity fixed before WHS DED estimation"
    )
    return expanded


def input_provenance(data_root: Path) -> dict[str, Any]:
    paths = {
        "ded_questionnaire": data_root / DED_2013,
        "microbiome_mapping": data_root / MICROBIOME_DIR / "TwinsUK_ASVs_MappingFile.xlsx",
        "asv_counts": data_root / MICROBIOME_DIR / "TwinsUK_ASVsCR.rds",
        "asv_taxonomy": data_root / MICROBIOME_DIR / "TwinsUK_ASVs_Taxa.csv.zip",
        "family_spine": data_root / SPINE,
        "analysis_specification": SPECIFICATION,
        "dose_trend_result": DOSE_RESULT,
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
        description="Run the prespecified WHS-defined dry-eye richness sensitivity."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Print source and linked-cohort counts without estimating the richness association.",
    )
    args = parser.parse_args()

    if not SPECIFICATION.exists():
        raise FileNotFoundError(f"Missing dated specification: {SPECIFICATION}")
    if not DOSE_RESULT.exists():
        raise FileNotFoundError(f"Missing validated dose-trend result: {DOSE_RESULT}")

    definition = verify_source_definition(args.data_root)
    _, within_3y, covariates, assembly = assemble_analysis_population(args.data_root)
    linked_outcome = pd.to_numeric(within_3y[OUTCOME], errors="coerce")
    linked_counts = {
        "eligible_with_whs_outcome_before_model_completeness": int(linked_outcome.notna().sum()),
        "cases_before_model_completeness": int((linked_outcome == 1).sum()),
        "controls_before_model_completeness": int((linked_outcome == 0).sum()),
    }
    if args.counts_only:
        print(json.dumps({"definition": definition, "linked_counts": linked_counts}, indent=2))
        return 0

    result = binary_association(within_3y, OUTCOME, RICHNESS, covariates)
    if not result:
        raise RuntimeError("The WHS DED richness model was not estimable")
    if not bool(result["converged"]) or not np.isfinite(result["odds_ratio_per_sd"]):
        raise RuntimeError("The WHS DED richness model did not produce a finite converged estimate")

    alpha_eight = alpha_eight_sensitivity(float(result["p_value"]))
    whs_q = float(
        alpha_eight.loc[
            alpha_eight["outcome_id"] == "whs_ded", "q_value_alpha_eight"
        ].iloc[0]
    )
    trend_q = float(
        alpha_eight.loc[
            alpha_eight["outcome_id"] == "strict_control_score_trend",
            "q_value_alpha_eight",
        ].iloc[0]
    )
    result["q_value_alpha_eight"] = whs_q

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "source_definition_validation.json", definition)
    write_json(args.out_dir / "whs_ded_richness_model.json", result)
    alpha_eight.to_csv(args.out_dir / "alpha_eight_multiplicity_sensitivity.csv", index=False)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "specification": {
            "path": str(SPECIFICATION),
            "sha256": sha256(SPECIFICATION),
            "fixed_before_whs_association_estimation": True,
            "outcome_hierarchy": [
                "highly symptomatic current-symptom outcome",
                "prespecified 0-6 current-symptom gradient",
                "strict asymptomatic-control sensitivity",
                "WHS-defined DED construct-validity sensitivity",
            ],
        },
        "definition_validation": definition,
        "assembly": assembly,
        "linked_counts": linked_counts,
        "whs_ded_richness_model": result,
        "multiplicity": {
            "family": "six existing alpha-diversity tests plus symptom trend and WHS DED richness",
            "tests": 8,
            "whs_ded_q_value": whs_q,
            "symptom_trend_q_value": trend_q,
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
    print(json.dumps(summary, indent=2, default=str))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
