"""Prepare minimal private model inputs, preserving the established cohort."""
from pathlib import Path
import hashlib
import json
import numpy as np
from run_2013_twin_relatedness_extension import (
    build_analysis_cohort, DEFAULT_DATA_ROOT, PROJECT_ROOT, PRIMARY_OUTCOME,
    PRIMARY_FEATURE, encoded_predictors, two_person_pairs,
    pair_difference_frame, fit_score_difference_model,
)


def main():
    out = PROJECT_ROOT / 'outputs/twin_family_signal_2026_09_05'
    private = out / 'private'
    private.mkdir(parents=True, exist_ok=True)
    cohort, covariates = build_analysis_cohort(DEFAULT_DATA_ROOT)
    predictors = encoded_predictors(cohort)
    pairs = two_person_pairs(cohort)
    reproduced = fit_score_difference_model(
        pair_difference_frame(pairs, predictors.loc[pairs.index]), 'All', True)
    assert np.isclose(reproduced['beta_score_points_per_sd'], -0.047782309500829394, atol=1e-10)
    # No original participant or family identifiers enter the model-input file.
    codes = {key: i for i, key in enumerate(sorted(cohort.FamilyID.unique()))}
    result = predictors.copy()
    result['family_code'] = cohort.FamilyID.map(codes)
    result['zygosity'] = cohort.pair_zygosity
    result['score'] = cohort.symptom_sum_0_6
    result['case'] = cohort[PRIMARY_OUTCOME].astype(int)
    target = private / 'model_input.csv'
    result.to_csv(target, index=False)
    summary = dict(participants=len(cohort), families=cohort.FamilyID.nunique(),
                   cases=int(cohort[PRIMARY_OUTCOME].sum()), pairs=pairs.FamilyID.nunique(),
                   richness_sd=float(cohort[PRIMARY_FEATURE].std(ddof=1)),
                   covariates=covariates, columns=result.columns.tolist(),
                   input_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                   reproduced_score_model=reproduced,
                   raw_data_modified=False, original_identifiers_exported=False)
    (out / 'input_preparation.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


# Original standalone launch disabled. Use the public workflow entry point.
