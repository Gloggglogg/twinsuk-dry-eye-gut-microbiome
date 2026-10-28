"""Report all coefficients from the existing manuscript regressions.

No new outcome, estimator, covariate choice, or multiplicity family is introduced.
Source data are read-only. Only aggregate coefficients/diagnostics are exported.
The primary and related diversity results must reproduce their frozen outputs.
"""
import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from scipy.stats import norm

from run_2013_symptom_dose_response import (
    assemble_analysis_population, DEFAULT_DATA_ROOT, PROJECT_ROOT,
    RICHNESS, REFERENCE_MATRIX, build_x, standardize,
)
from run_2013_twin_family_signal_revision import PREDICTORS, nonconstant

PACKAGE = Path(__file__).resolve().parents[1] / 'runtime' / 'manuscript_assets'
OUT = PROJECT_ROOT / 'runtime/hugo_model_reporting_2026_09_07'
ROWS, DIAGNOSTICS, CHECKS = [], [], []


def check(model_id, field, observed, expected):
    ok = bool(np.isclose(observed, expected, atol=1e-9, rtol=1e-7))
    CHECKS.append(dict(model_id=model_id, field=field, observed=float(observed),
                       expected=float(expected), passed=ok))
    if not ok:
        raise ValueError(f'{model_id}: {field} does not reproduce: {observed} vs {expected}')


def collect(fit, model_id, y, groups, binary, diagnostics=True):
    x = fit.model.exog
    for term in fit.params.index:
        beta, se = float(fit.params[term]), float(fit.bse[term])
        z = beta / se
        low, high = beta - norm.ppf(.975)*se, beta + norm.ppf(.975)*se
        ROWS.append(dict(model_id=model_id, model_type='logistic' if binary else 'linear',
                         n=len(y), families=int(pd.Series(groups).nunique()),
                         cases=int(np.sum(y)) if binary else None, term=term,
                         coefficient=beta, standard_error=se, z_statistic=z,
                         p_value=float(2*norm.sf(abs(z))), ci_low=low, ci_high=high,
                         odds_ratio=float(np.exp(beta)) if binary else None,
                         or_ci_low=float(np.exp(low)) if binary else None,
                         or_ci_high=float(np.exp(high)) if binary else None,
                         uncertainty='family-clustered robust'))
    if not diagnostics:
        return
    info = dict(model_id=model_id, n=len(y), families=int(pd.Series(groups).nunique()),
                terms=list(fit.params.index), design_rank=int(np.linalg.matrix_rank(x)),
                design_columns=x.shape[1], finite_coefficients=bool(np.isfinite(fit.params).all()),
                finite_standard_errors=bool(np.isfinite(fit.bse).all()))
    if binary:
        # Separation exists if a direction improves at least one signed margin
        # without worsening any. Bounded LP also detects quasi-separation.
        a = (2*np.asarray(y)-1)[:, None] * x
        separation = linprog(-a.sum(axis=0), A_ub=-a, b_ub=np.zeros(len(y)),
                             bounds=[(-1, 1)]*x.shape[1], method='highs')
        if not separation.success:
            raise RuntimeError('Separation diagnostic failed: '+separation.message)
        separated = bool(-separation.fun > 1e-7)
        info.update(cases=int(np.sum(y)), controls=int(len(y)-np.sum(y)),
                    slopes=x.shape[1]-1, events_per_slope=float(min(np.sum(y), len(y)-np.sum(y))/(x.shape[1]-1)),
                    converged=bool(fit.converged), separation_detected=separated,
                    separation_margin_objective=float(-separation.fun),
                    maximum_absolute_score=float(np.abs(fit.model.score(fit.params)).max()))
    DIAGNOSTICS.append(info)


def population_fit(data, outcome, feature, covariates, model_id, binary=True, fixed=False):
    d = data.dropna(subset=[outcome, feature]).copy()
    x = build_x(d, feature, covariates)
    keep = x.notna().all(axis=1)
    x = x.loc[keep].astype(float)
    if not fixed:
        x['feature_z'] = standardize(x['feature_z'])
    x = sm.add_constant(x, has_constant='add')
    y, groups = d.loc[keep, outcome].astype(float), d.loc[keep, 'FamilyID'].astype(str)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = sm.GLM(y, x, family=sm.families.Binomial()) if binary else sm.OLS(y, x)
        fit = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
    collect(fit, model_id, y, groups, binary)
    DIAGNOSTICS[-1]['warnings'] = [str(w.message) for w in caught]
    return fit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dose, base, covs, assembly = assemble_analysis_population(DEFAULT_DATA_ROOT)
    reference = pd.read_csv(REFERENCE_MATRIX)
    for outcome, col in [('primary', 'highly_symptomatic_derived'),
                         ('strict', 'hs_vs_clean_control'), ('broader', 'either_often_symptom')]:
        oid = {'primary':'highly_symptomatic_and', 'strict':'hs_vs_clean_control',
               'broader':'either_often_symptom_or'}[outcome]
        for feature in ['alpha_richness', 'alpha_shannon']:
            mid = outcome+'_'+feature
            fit = population_fit(base, col, feature, covs, mid)
            ref = reference.loc[(reference.branch_id=='within_3y') &
                (reference.covariate_set=='primary_core') & (reference.outcome_id==oid) &
                (reference.feature==feature)].iloc[0]
            for field, actual in [('beta_per_sd', fit.params.feature_z),
                                  ('standard_error', fit.bse.feature_z), ('p_value', fit.pvalues.feature_z)]:
                check(mid, field, actual, ref[field])
    score = population_fit(dose, 'dose_score', 'richness_z_global', covs, 'symptom_score', False, True)
    ref = json.loads((PROJECT_ROOT/'outputs/symptom_dose_response_2026_08_24/ordered_trend_model.json').read_text())
    check('symptom_score', 'beta', score.params.feature_z, ref['beta_score_points_per_global_richness_sd'])
    check('symptom_score', 'se', score.bse.feature_z, ref['standard_error'])
    for category in range(1, 5):
        d = dose.loc[dose.dose_score.isin([0, category])].copy()
        d['category_case'] = (d.dose_score==category).astype(int)
        fit = population_fit(d, 'category_case', 'richness_z_global', covs,
                             f'score_{category}_vs_strict', True, True)
        ref = pd.read_csv(PROJECT_ROOT/'outputs/symptom_dose_response_2026_08_24/strict_control_category_models.csv')
        ref = ref.loc[ref.dose_score.eq(category)].iloc[0]
        check(f'score_{category}_vs_strict', 'beta', fit.params.feature_z, ref.beta_per_global_richness_sd)
        check(f'score_{category}_vs_strict', 'se', fit.bse.feature_z, ref.standard_error)
    fit = population_fit(base, 'whs_ded_derived', RICHNESS, covs, 'whs_defined_dry_eye')
    ref = json.loads((PROJECT_ROOT/'outputs/whs_ded_sensitivity_2026_08_24/whs_ded_richness_model.json').read_text())
    check('whs_defined_dry_eye', 'beta', fit.params.feature_z, ref['beta_per_sd'])
    check('whs_defined_dry_eye', 'se', fit.bse.feature_z, ref['standard_error'])
    # Reuse the exact private v41 predictor coding, including its fixed richness SD.
    d = pd.read_csv(PROJECT_ROOT/'outputs/twin_family_signal_2026_09_05/private/model_input.csv')
    paired = d.loc[d.groupby('family_code').family_code.transform('size').eq(2)]
    matched_ref = pd.read_csv(PROJECT_ROOT/'outputs/twin_family_signal_2026_09_05/matched_score_models.csv')
    for group in ['All', 'MZ', 'DZ']:
        f = paired if group=='All' else paired.loc[paired.zygosity==group]
        x = nonconstant(f[PREDICTORS].astype(float))
        means = x.groupby(f.family_code).transform('mean')
        wb = nonconstant(pd.concat([(x-means).add_prefix('within_'), means.add_prefix('between_')],axis=1))
        for label, design in [('matched_population', x), ('joint_within_between', wb)]:
            fit = sm.OLS(f.score, sm.add_constant(design)).fit(cov_type='cluster',cov_kwds={'groups':f.family_code})
            collect(fit, f'{group}_{label}', f.score, f.family_code, False)
            key = 'richness_z' if label=='matched_population' else 'within_richness_z'
            ref_label = 'matched_population' if label=='matched_population' else 'within'
            ref = matched_ref.loc[(matched_ref.group==group) & (matched_ref.model==ref_label)].iloc[0]
            check(f'{group}_{label}', 'beta', fit.params[key], ref.beta)
            check(f'{group}_{label}', 'se', fit.bse[key], ref.se)
    coeff = pd.DataFrame(ROWS)
    coeff.to_csv(OUT/'all_model_coefficients.csv', index=False)
    coeff.to_csv(PACKAGE/'supplementary_data_S2_model_coefficients.csv', index=False)
    primary = d
    cross = {col: pd.crosstab(primary[col], primary['case']).to_dict() for col in
             ['SEX_M', 'AntibioticsPriorMonth_Unknown', 'AntibioticsPriorMonth_Y']}
    report = dict(date='2026-09-07', purpose='Hugo requested full model reporting and sparse-case assessment',
                  assembly=assembly, models=DIAGNOSTICS, reproduction_checks=CHECKS,
                  primary_categorical_counts=cross, no_new_analysis_question=True,
                  no_resampling_or_class_weighting=True)
    (OUT/'model_reporting_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    render_table(coeff)
    print(json.dumps(dict(models=len(DIAGNOSTICS), coefficients=len(ROWS),
                         checks=len(CHECKS), all_reproduced=all(x['passed'] for x in CHECKS),
                         separation=[x['model_id'] for x in DIAGNOSTICS if x.get('separation_detected')]),indent=2))


def render_table(coeff):
    # Three clinically central models shown in the PDF; all 18 in the CSV.
    labels = {'const':'Intercept', 'feature_z':'Microbial richness', 'age':'Age', 'BMI':'Body mass index',
              'gap_days_abs':'Stool--questionnaire interval', 'AntibioticsPriorMonth_Unknown':'Antibiotics: unknown vs no',
              'AntibioticsPriorMonth_Y':'Antibiotics: yes vs no', 'SEX_M':'Male vs female'}
    panels = [('primary_alpha_richness','A. Highly symptomatic dry eye (1,685 participants; 93 cases)'),
              ('strict_alpha_richness','B. Strict-control sensitivity (618 participants; 93 cases)'),
              ('symptom_score','C. Symptom score (1,678 participants)')]
    tex=[]
    for pi, (mid, title) in enumerate(panels, 1):
        tex.extend([f'% SRC A3-TS6-S0{pi}', r'\textbf{'+title+'}', r'\par\smallskip',
                    r'\begin{tabular}{p{5.5cm}rrrrr}', r'\toprule', f'% SRC A3-TS6-H0{pi}',
                    r'Variable & Coefficient & SE & $z$ & p & 95\% CI \\',r'\midrule'])
        for ri, (_, row) in enumerate(coeff.loc[coeff.model_id==mid].iterrows(),1):
            p = '<0.0001' if row.p_value<.0001 else f'{row.p_value:.4f}'
            tex.extend([f'% SRC A3-TS6-P{pi}-R{ri:02d}',
                        f"{labels[row.term]} & {row.coefficient:.3f} & {row.standard_error:.3f} & {row.z_statistic:.2f} & ${p}$ & ({row.ci_low:.3f}, {row.ci_high:.3f}) \\"])
            # End rows with the two backslashes required by TeX.
            tex[-1] = tex[-1].rstrip('\\') + r'\\'
        tex.extend([r'\bottomrule',r'\end{tabular}',r'\par\medskip'])
    (PACKAGE/'generated/table_s6_model_coefficients.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')


# Original standalone launch disabled. Use the public workflow entry point.
