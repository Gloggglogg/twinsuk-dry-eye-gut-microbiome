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
from statsmodels.stats.multitest import multipletests

from run_2013_locked_primary_16s_analysis import build_x
from run_2013_symptom_dose_response import (
    DEFAULT_DATA_ROOT,
    MIN_GROUP_FAMILIES,
    MIN_GROUP_PARTICIPANTS,
    PROJECT_ROOT,
    assemble_analysis_population,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / 'runtime' / 'sparse_tail_pooling_followup_2026_08_24'
PRIOR_CATEGORY_RESULTS = PROJECT_ROOT / 'runtime' / 'symptom_dose_response_2026_08_24' / 'strict_control_category_models.csv'


def fit_contrast(
    data: pd.DataFrame,
    case_scores: tuple[int, ...],
    label: str,
    covariates: list[str],
) -> dict[str, Any]:
    case_mask = data["dose_score"].isin(case_scores)
    model = data.loc[(data["dose_score"] == 0) | case_mask].copy()
    model["outcome"] = model["dose_score"].isin(case_scores).astype(int)
    case_count = int(model["outcome"].sum())
    case_families = int(
        model.loc[model["outcome"] == 1, "FamilyID"].nunique()
    )
    result: dict[str, Any] = {
        "contrast": label,
        "case_scores": ",".join(str(score) for score in case_scores),
        "cases": case_count,
        "case_families": case_families,
        "model_fitted": False,
    }
    if (
        case_count < MIN_GROUP_PARTICIPANTS
        or case_families < MIN_GROUP_FAMILIES
    ):
        result["reason"] = "below participant or family minimum"
        return result

    x = build_x(model, "richness_z_global", covariates)
    keep = x.notna().all(axis=1) & model["FamilyID"].notna()
    x = sm.add_constant(x.loc[keep].astype(float), has_constant="add")
    y = model.loc[keep, "outcome"].astype(int)
    groups = model.loc[keep, "FamilyID"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    beta = float(fit.params["feature_z"])
    low, high = [float(value) for value in fit.conf_int().loc["feature_z"]]
    result.update(
        {
            "n": int(len(y)),
            "cases": int(y.sum()),
            "controls": int(len(y) - y.sum()),
            "case_families": int(
                model.loc[keep & (model["outcome"] == 1), "FamilyID"].nunique()
            ),
            "total_families": int(groups.nunique()),
            "mean_case_richness": float(
                model.loc[keep & (model["outcome"] == 1), "alpha_richness"].mean()
            ),
            "median_case_richness": float(
                model.loc[keep & (model["outcome"] == 1), "alpha_richness"].median()
            ),
            "beta_per_global_richness_sd": beta,
            "standard_error": float(fit.bse["feature_z"]),
            "odds_ratio_per_global_richness_sd": float(math.exp(beta)),
            "ci_low_or": float(math.exp(low)),
            "ci_high_or": float(math.exp(high)),
            "p_value": float(fit.pvalues["feature_z"]),
            "converged": bool(getattr(fit, "converged", True)),
            "covariance_method": "family-clustered robust",
            "model_fitted": True,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pool sparse upper symptom-score categories in a transparent follow-up "
            "to the fixed 0-6 dose-response analysis."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    dose, _, covariates, assembly = assemble_analysis_population(args.data_root)
    contrasts = [
        ((4,), "score_4"),
        ((5, 6), "scores_5_6_pooled"),
        ((4, 5, 6), "scores_4_6_pooled"),
    ]
    results = pd.DataFrame(
        [
            fit_contrast(dose, scores, label, covariates)
            for scores, label in contrasts
        ]
    )

    pooled = results["contrast"].isin(
        ["scores_5_6_pooled", "scores_4_6_pooled"]
    ) & results["model_fitted"].fillna(False)
    results["q_value_bh_two_new_pooled_contrasts"] = np.nan
    if pooled.any():
        results.loc[pooled, "q_value_bh_two_new_pooled_contrasts"] = multipletests(
            results.loc[pooled, "p_value"], method="fdr_bh"
        )[1]

    if not PRIOR_CATEGORY_RESULTS.exists():
        raise FileNotFoundError(
            f"Prior category result is missing: {PRIOR_CATEGORY_RESULTS}"
        )
    prior = pd.read_csv(PRIOR_CATEGORY_RESULTS)
    prior_score4 = prior.loc[prior["dose_score"] == 4]
    if len(prior_score4) != 1:
        raise RuntimeError("Expected exactly one prior score-4 result")
    score4 = results.loc[results["contrast"] == "score_4"].iloc[0]
    prior_score4_row = prior_score4.iloc[0]
    score4_reproduced = all(
        np.isclose(
            float(score4[column]),
            float(prior_score4_row[column]),
            rtol=0.0,
            atol=1e-12,
        )
        for column in [
            "odds_ratio_per_global_richness_sd",
            "ci_low_or",
            "ci_high_or",
            "p_value",
        ]
    )
    if not score4_reproduced:
        raise RuntimeError("The score-4 comparator did not reproduce the prior result")

    score_counts = (
        dose.loc[dose["dose_score"].between(4, 6)]
        .groupby("dose_score", as_index=False)
        .agg(
            participants=("PublicID", "size"),
            families=("FamilyID", "nunique"),
            mean_richness=("alpha_richness", "mean"),
            median_richness=("alpha_richness", "median"),
        )
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "pooled_tail_models.csv", index=False)
    score_counts.to_csv(args.out_dir / "tail_score_counts.csv", index=False)

    result_lookup = results.set_index("contrast")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_label": (
            "investigator-requested follow-up after sparse score-5 and score-6 "
            "counts were known"
        ),
        "assembly": assembly,
        "minimum_participants": MIN_GROUP_PARTICIPANTS,
        "minimum_families": MIN_GROUP_FAMILIES,
        "score_4_comparator_reproduced": score4_reproduced,
        "results": results.to_dict(orient="records"),
        "descriptive_comparison": {
            "scores_5_6_or_minus_score_4_or": float(
                result_lookup.loc[
                    "scores_5_6_pooled", "odds_ratio_per_global_richness_sd"
                ]
                - result_lookup.loc[
                    "score_4", "odds_ratio_per_global_richness_sd"
                ]
            ),
            "scores_4_6_or_minus_score_4_or": float(
                result_lookup.loc[
                    "scores_4_6_pooled", "odds_ratio_per_global_richness_sd"
                ]
                - result_lookup.loc[
                    "score_4", "odds_ratio_per_global_richness_sd"
                ]
            ),
            "note": (
                "Differences between correlated odds-ratio point estimates are descriptive; "
                "overlapping confidence intervals do not establish that the associations differ."
            ),
        },
        "multiplicity": (
            "Benjamini-Hochberg adjustment across the two newly requested pooled contrasts; "
            "score 4 retains its original category-family q value."
        ),
        "privacy": {
            "participant_level_output_written": False,
            "raw_data_written": False,
            "outputs_are_aggregate_only": True,
        },
    }
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(results.to_string(index=False))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
