"""Focused, reproducible twin comparisons and bivariate covariance feasibility.

Input is a privately ignored, identifier-free extract from the unchanged cohort.
Only aggregate model estimates, diagnostics and bootstrap estimates are exported.
The continuous-score covariance model is exploratory, not a causal-effect model.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, skew
import statsmodels
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

COVS = ['age_z', 'BMI_z', 'gap_days_abs_z',
        'AntibioticsPriorMonth_Unknown', 'AntibioticsPriorMonth_Y', 'SEX_M']
PREDICTORS = ['richness_z'] + COVS
SEED = 20260905  # Date-based reproducibility constant, not selected by results.
BOUNDS = [(0, None), (None, None), (0, None)] * 3


def nonconstant(x):
    return x.loc[:, x.std(ddof=0).gt(1e-12)]


def estimate_record(fit, term, model, group, n):
    beta, se = float(fit.params[term]), float(fit.bse[term])
    return dict(group=group, model=model, pairs=n, beta=beta, se=se,
                ci_low=beta - 1.95996398454 * se,
                ci_high=beta + 1.95996398454 * se,
                p_value=float(2 * norm.sf(abs(beta/se))))


def score_models(frame, group):
    """Matched sample, outcome and richness unit; joint contrast uses covariance."""
    frame = frame.copy()
    n = frame.family_code.nunique()
    x = nonconstant(frame[PREDICTORS].astype(float))
    population = sm.OLS(frame.score, sm.add_constant(x)).fit(
        cov_type='cluster', cov_kwds={'groups': frame.family_code})
    means = x.groupby(frame.family_code).transform('mean')
    wb = pd.concat([(x-means).add_prefix('within_'), means.add_prefix('between_')], axis=1)
    wb = nonconstant(wb)
    joint = sm.OLS(frame.score, sm.add_constant(wb)).fit(
        cov_type='cluster', cov_kwds={'groups': frame.family_code})
    result = [estimate_record(population, 'richness_z', 'matched_population', group, n)]
    for label in ['within', 'between']:
        result.append(estimate_record(joint, label+'_richness_z', label, group, n))
    contrast = np.zeros(len(joint.params))
    contrast[joint.params.index.get_loc('within_richness_z')] = 1
    contrast[joint.params.index.get_loc('between_richness_z')] = -1
    beta = float(contrast @ joint.params)
    se = float(np.sqrt(contrast @ joint.cov_params() @ contrast))
    result.append(dict(group=group, model='within_minus_between', pairs=n, beta=beta,
                       se=se, ci_low=beta-1.95996398454*se, ci_high=beta+1.95996398454*se,
                       p_value=float(2*norm.sf(abs(beta/se)))))
    return result


def binary_models(frame, group):
    """Check the previously unadjusted conditional model with varying covariates."""
    chosen = frame.groupby('family_code')['case'].transform('sum').eq(1)
    d = frame.loc[chosen].copy()
    varying = d[PREDICTORS].astype(float)
    variation = varying - varying.groupby(d.family_code).transform('mean')
    varying = varying.loc[:, variation.std(ddof=0).gt(1e-12)]
    model = ConditionalLogit(d['case'], varying, groups=d.family_code)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = model.fit(method='bfgs', maxiter=2000, disp=False, gtol=1e-8)
    score = float(np.max(np.abs(model.score(fit.params)))/len(d))
    eigen = np.linalg.eigvalsh(-model.hessian(fit.params))
    row = estimate_record(fit, 'richness_z', 'covariate_adjusted_conditional', group,
                          d.family_code.nunique())
    row.update(odds_ratio=float(np.exp(row['beta'])),
               or_ci_low=float(np.exp(row['ci_low'])), or_ci_high=float(np.exp(row['ci_high'])),
               predictors=varying.columns.tolist(), warnings=[str(w.message) for w in caught],
               gradient_per_person=score, min_information_eigenvalue=float(eigen.min()),
               valid=bool(score < 1e-5 and eigen.min() > 0 and np.isfinite(fit.bse).all()))
    # For pairs with exactly one case, the conditional likelihood is exactly
    # logistic likelihood on case-minus-control covariate differences, no intercept.
    # Refit directly because ConditionalLogit.fit does not forward gtol in 0.14.2.
    dx = np.asarray([g.loc[g['case'].eq(1), varying.columns].to_numpy(float)[0]
                     - g.loc[g['case'].eq(0), varying.columns].to_numpy(float)[0]
                     for _, g in d.groupby('family_code')])
    def likelihood(beta):
        eta = dx @ beta
        return np.logaddexp(0, -eta).sum(), -(dx.T @ expit(-eta))
    tight = minimize(likelihood, fit.params.to_numpy(), jac=True, method='BFGS',
                     options={'maxiter': 3000, 'gtol': 1e-9})
    p = expit(dx @ tight.x)
    information = dx.T @ ((p*(1-p))[:, None]*dx)
    spectrum = np.linalg.eigvalsh(information)
    row['direct_pair_likelihood_check'] = dict(
        parameters=dict(zip(varying.columns, tight.x.tolist())),
        gradient=float(np.abs(tight.jac).max()),
        minimum_information_eigenvalue=float(spectrum.min()),
        information_condition_number=float(spectrum.max()/spectrum.min()),
        likelihood_agreement_at_original_parameters=float(abs(likelihood(fit.params)[0]+model.loglike(fit.params))),
        optimizer_message=str(tight.message))
    unknown = dx[:, varying.columns.get_loc('AntibioticsPriorMonth_Unknown')]
    values, counts = np.unique(unknown, return_counts=True)
    row['antibiotic_unknown_pair_counts'] = {str(int(v)): int(n) for v, n in zip(values, counts)}
    # A nearly flat likelihood means one coefficient can grow without a finite
    # supported optimum, even when the package returns no warning. Do not label
    # such a fit reliable just because its richness coefficient is finite.
    row['valid'] = bool(np.abs(tight.jac).max() < 1e-6 and spectrum.min() > 1e-6
                        and spectrum.max()/spectrum.min() < 1e8)
    if row['valid']:
        se = float(np.sqrt(np.linalg.inv(information)[0, 0]))
        beta = float(tight.x[0])
        row.update(beta=beta, se=se, ci_low=beta-1.95996398454*se,
                   ci_high=beta+1.95996398454*se, p_value=float(2*norm.sf(abs(beta/se))),
                   odds_ratio=float(np.exp(beta)), or_ci_low=float(np.exp(beta-1.95996398454*se)),
                   or_ci_high=float(np.exp(beta+1.95996398454*se)))
    return row


def paired_arrays(frame):
    """Adjust both traits for the same covariates, then standardize residuals."""
    x = sm.add_constant(nonconstant(frame[COVS].astype(float))).to_numpy()
    y = frame[['richness_z', 'score']].to_numpy(float)
    residual = y-x @ np.linalg.lstsq(x, y, rcond=None)[0]
    residual /= residual.std(axis=0, ddof=0)
    work = frame[['family_code', 'zygosity']].copy()
    work[['r', 's']] = residual
    arrays = {}
    for group in ['MZ', 'DZ']:
        pairs = [g[['r', 's']].to_numpy().reshape(4)
                 for _, g in work[work.zygosity.eq(group)].groupby('family_code', sort=True)]
        arrays[group] = np.asarray(pairs)
    return arrays, dict(score_residual_skewness=float(skew(residual[:, 1])),
                        residual_scales='Each trait has unit residual variance in known-zygosity pairs')


def moments(arrays):
    """Symmetric cross-products make likelihood invariant to twin ordering."""
    out = []
    for group, r in [('MZ', 1.0), ('DZ', .5)]:
        values = arrays[group]
        s = values.T @ values / len(values)
        swap = [2, 3, 0, 1]
        s = (s+s[np.ix_(swap, swap)])/2
        out.append((len(values), r, s))
    return out


def components(theta):
    matrices, paths = [], []
    for i in range(3):
        a, b, c = theta[3*i:3*i+3]
        path = np.array([[a, 0.], [b, c]])
        paths.append(path)
        matrices.append(path @ path.T)
    return matrices, paths


def objective(theta, stats):
    """Gaussian negative log likelihood per pair and analytic gradient."""
    (a, c, e), paths = components(theta)
    v = a+c+e
    total, gradients = 0., [np.zeros((2, 2)) for _ in range(3)]
    total_n = sum(n for n, _, _ in stats)
    for n, r, s in stats:
        k = r*a+c
        sigma = np.block([[v, k], [k, v]])
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0 or np.linalg.eigvalsh(sigma).min() < 1e-10:
            return 1e10, np.zeros(9)
        inv = np.linalg.inv(sigma)
        total += .5*n*(logdet+np.trace(inv@s)) / total_n
        g = .5*n*(inv-inv@s@inv)/total_n
        diagonal, cross = g[:2, :2]+g[2:, 2:], g[:2, 2:]+g[2:, :2]
        for target, value in zip(gradients, [diagonal+r*cross, diagonal+cross, diagonal]):
            target += value
    gradient = []
    for g, path in zip(gradients, paths):
        gp = (g+g.T) @ path
        gradient.extend([gp[0, 0], gp[1, 0], gp[1, 1]])
    return float(total), np.asarray(gradient)


def fit_covariance(stats, rng, starts=5, initial=None):
    candidates = []
    if initial is not None:
        candidates.append(np.asarray(initial))
    candidates.append(np.tile([.5, -.03, .6], 3))
    while len(candidates) < starts:
        candidates.append(np.array([[rng.uniform(.1, .8), rng.uniform(-.2, .2),
                                     rng.uniform(.1, .8)] for _ in range(3)]).ravel())
    fits = [minimize(objective, p, args=(stats,), jac=True, method='L-BFGS-B', bounds=BOUNDS,
                     options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-7}) for p in candidates]
    best = min(fits, key=lambda f: f.fun)
    gradient = best.jac.copy()
    for i, (lo, _) in enumerate(BOUNDS):
        if lo is not None and best.x[i] < 1e-8 and gradient[i] > 0:
            gradient[i] = 0
    acceptable = bool(np.isfinite(best.fun) and np.max(np.abs(gradient)) < 1e-4)
    matrices, _ = components(best.x)
    return dict(theta=best.x.tolist(), objective=float(best.fun),
                projected_gradient=float(np.max(np.abs(gradient))), valid=acceptable,
                optimizer_success=bool(best.success), optimizer_message=str(best.message),
                starts=starts, start_objectives=[float(f.fun) for f in fits],
                matrices={name: value.tolist() for name, value in zip(['A', 'C', 'E'], matrices)},
                eigenvalues={name: np.linalg.eigvalsh(value).tolist()
                             for name, value in zip(['A', 'C', 'E'], matrices)})


def resample_pairs(frame, rng):
    samples, counter = [], 0
    for group in frame.zygosity.unique():
        families = [g for _, g in frame[frame.zygosity.eq(group)].groupby('family_code')]
        for index in rng.integers(0, len(families), len(families)):
            sample = families[index].copy()
            sample['family_code'] = counter
            counter += 1
            samples.append(sample)
    return pd.concat(samples, ignore_index=True)


def bootstrap_one(args):
    index, frame, initial = args
    rng = np.random.default_rng(SEED+index+1)
    sample = resample_pairs(frame, rng)
    differences = {}
    for group in ['All', 'MZ', 'DZ']:
        selected = sample if group == 'All' else sample[sample.zygosity.eq(group)]
        rows = {r['model']: r['beta'] for r in score_models(selected, group)}
        differences[group] = rows['within']-rows['matched_population']
    known = sample[sample.zygosity.isin(['MZ', 'DZ'])]
    arrays, _ = paired_arrays(known)
    fit = fit_covariance(moments(arrays), rng, starts=3, initial=initial)
    covariance = {name: np.asarray(fit['matrices'][name])[0, 1] for name in ['A', 'C', 'E']}
    covariance['total'] = sum(covariance.values())
    return dict(replicate=index, valid=fit['valid'], projected_gradient=fit['projected_gradient'],
                **covariance, **{'difference_'+g: b for g, b in differences.items()})


def self_test():
    """Independent algebraic checks before touching study outcomes."""
    from scipy.optimize._numdiff import approx_derivative
    theta = np.array([.7, -.12, .6, .3, .08, .4, .5, -.05, .7])
    (a, c, e), _ = components(theta)
    stats = []
    for r in [1, .5]:
        v, k = a+c+e, r*a+c
        stats.append((1000, r, np.block([[v, k], [k, v]])))
    probe = theta+.02
    numerical = approx_derivative(lambda t: objective(t, stats)[0], probe).ravel()
    assert np.allclose(numerical, objective(probe, stats)[1], atol=1e-6)
    fit = fit_covariance(stats, np.random.default_rng(11), starts=6)
    assert fit['valid'], fit
    for label, truth in zip(['A', 'C', 'E'], [a, c, e]):
        assert np.allclose(fit['matrices'][label], truth, atol=2e-4), label
    rng = np.random.default_rng(9)
    arrays = {group: rng.normal(size=(100, 4)) for group in ['MZ', 'DZ']}
    swapped = {g: v[:, [2, 3, 0, 1]] for g, v in arrays.items()}
    assert np.isclose(objective(theta, moments(arrays))[0], objective(theta, moments(swapped))[0])
    return {'analytic_gradient': 'passed', 'known_population_covariance_recovery': 'passed',
            'twin_order_invariance': 'passed', 'model': 'bivariate continuous-score Cholesky ACE'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--bootstrap', type=int, default=499)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--binary-check-only', action='store_true')
    args = parser.parse_args()
    tests = self_test()
    if args.self_test:
        print(json.dumps(tests, indent=2)); return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    pairs = frame[frame.groupby('family_code').family_code.transform('size').eq(2)].copy()
    assert len(frame) == 1685 and len(pairs) == 1120 and pairs.family_code.nunique() == 560
    assert not frame[PREDICTORS+['score', 'case']].isna().any().any()
    rows, binary = [], []
    for group in ['All', 'MZ', 'DZ']:
        selected = pairs if group == 'All' else pairs[pairs.zygosity.eq(group)]
        rows.extend(score_models(selected, group))
        binary.append(binary_models(selected, group))
    score_results = pd.DataFrame(rows)
    assert np.isclose(score_results.query("group == 'All' and model == 'within'").beta.iloc[0],
                      -.047782309500829394, atol=1e-10)
    score_results.to_csv(args.out_dir/'matched_score_models.csv', index=False)
    (args.out_dir/'adjusted_binary_check.json').write_text(json.dumps(binary, indent=2)+'\n')
    if args.binary_check_only:
        print(json.dumps(binary, indent=2)); return
    known = pairs[pairs.zygosity.isin(['MZ', 'DZ'])]
    arrays, residual_info = paired_arrays(known)
    stats = moments(arrays)
    fit = fit_covariance(stats, np.random.default_rng(SEED), starts=12)
    assert fit['valid'], fit
    diagnostics = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                       self_tests=tests, participants=len(frame), complete_pairs=560,
                       known_zygosity_pairs={g: len(v) for g, v in arrays.items()},
                       residual_info=residual_info, symmetrized_cross_products={
                           g: s.tolist() for g, (_, _, s) in zip(['MZ', 'DZ'], stats)},
                       covariance_model=fit, seed=SEED, bootstrap_requested=args.bootstrap,
                       python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                       scipy=scipy.__version__, statsmodels=statsmodels.__version__)
    (args.out_dir/'model_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    print('Point estimates and self-tests complete; starting pair bootstrap.', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(bootstrap_one,
                               [(i, pairs, fit['theta']) for i in range(args.bootstrap)]))
    boot = pd.DataFrame(results)
    boot.to_csv(args.out_dir/'aggregate_bootstrap_estimates.csv', index=False)
    valid = boot[boot.valid]
    summary = dict(bootstrap_requested=len(boot), bootstrap_valid=len(valid),
                   maximum_gradient=float(boot.projected_gradient.max()), components={},
                   matched_comparisons={})
    for name in ['A', 'C', 'E', 'total']:
        matrices = {k: np.asarray(v) for k, v in fit['matrices'].items()}
        point = sum(v[0, 1] for v in matrices.values()) if name == 'total' else matrices[name][0, 1]
        summary['components'][name] = dict(estimate=float(point),
            ci_low=float(valid[name].quantile(.025)), ci_high=float(valid[name].quantile(.975)))
    for group in ['All', 'MZ', 'DZ']:
        selected = score_results[score_results.group.eq(group)].set_index('model').beta
        summary['matched_comparisons'][group] = dict(
            within_minus_population=float(selected['within']-selected['matched_population']),
            ci_low=float(boot['difference_'+group].quantile(.025)),
            ci_high=float(boot['difference_'+group].quantile(.975)))
    summary['percentage_partition_reportable'] = bool(
        summary['components']['total']['ci_high'] < 0 and len(valid) == len(boot))
    summary['percentage_rule_note'] = ('A nonzero total association is necessary, not sufficient, '
        'for interpretable covariance fractions. Component uncertainty and model assumptions also require review.')
    (args.out_dir/'bootstrap_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


# Original standalone launch disabled. Use the public workflow entry point.
