from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln
from statsmodels.stats.multitest import multipletests

from run_2013_initial_microbiome_signal_probes import (
    DEFAULT_DATA_ROOT,
    MICROBIOME_DIR,
    PROJECT_ROOT,
    clean_feature_name,
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
    BINARY_OUTCOMES,
    CORE_COVARIATES,
    DIALISTER,
    FROZEN_CANDIDATE_GENERA,
    TIMING_BRANCHES,
    binary_association,
    sex_is_estimable,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / 'runtime' / 'method_strengthening_2013'
IBS_SOURCE = Path("New March 2022/Extracted/DS00008_Irritable_Bowel_Syndrome_IBS/IBS_RomeIII.xlsx")
PRIMARY_OUTCOMES = {
    "either_often_symptom_or": BINARY_OUTCOMES["either_often_symptom_or"],
    "highly_symptomatic_and": BINARY_OUTCOMES["highly_symptomatic_and"],
    "hs_vs_clean_control": BINARY_OUTCOMES["hs_vs_clean_control"],
}
PRIMARY_FEATURES = ["alpha_richness", "alpha_shannon", DIALISTER]


def write_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def load_joined_inputs(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ded = load_ded_2013(data_root)
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

    asv_cols = [c for c in counts.columns if c != "SequencingSpecificName"]
    wide = counts.set_index("SequencingSpecificName")[asv_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    total = wide.sum(axis=1)
    relative = wide.div(total.replace(0, np.nan), axis=0).fillna(0)
    richness = (wide > 0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        shannon = -(relative.where(relative > 0) * np.log(relative.where(relative > 0))).sum(axis=1)

    alpha = pd.DataFrame(
        {
            "SequencingSpecificName": wide.index,
            "alpha_shannon": shannon.values,
            "alpha_richness": richness.values,
            "alpha_total_reads": total.values,
        }
    )
    genus_counts = collapse_genus_counts(wide, taxa)
    genus_rel = genus_counts.div(genus_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    genus_feature_frames = {}
    for col in genus_rel.columns:
        if (genus_counts[col] > 0).mean() >= 0.10:
            genus_feature_frames[f"genus__{clean_feature_name(col)}"] = np.log1p(genus_rel[col] * 1_000_000.0)
    genus_features = pd.DataFrame(genus_feature_frames, index=genus_rel.index)
    genus_features.index.name = None
    genus_features.insert(0, "SequencingSpecificName", genus_features.index.astype(str))
    base = pairs.merge(alpha, on="SequencingSpecificName", how="inner").merge(
        genus_features, on="SequencingSpecificName", how="inner"
    )
    return base, wide, genus_counts


def collapse_genus_counts(wide: pd.DataFrame, taxa: pd.DataFrame) -> pd.DataFrame:
    asv_cols = wide.columns.tolist()
    aligned = taxa.set_index("asv").reindex(asv_cols)
    aligned = aligned.loc[aligned["Genus_clean"].notna()].copy()
    frames = {}
    for genus, group in aligned.groupby("Genus_clean"):
        genus_asvs = [asv for asv in group.index if asv in wide.columns]
        if genus_asvs:
            frames[genus] = wide[genus_asvs].sum(axis=1)
    return pd.DataFrame(frames, index=wide.index)


def expected_rarefied_richness(counts: pd.DataFrame, depth: int) -> pd.Series:
    values = []
    for _, row in counts.iterrows():
        c = row.to_numpy(dtype=float)
        n = int(c.sum())
        if n < depth or depth <= 0:
            values.append(np.nan)
            continue
        positive = c[c > 0]
        # Expected observed richness after sampling depth reads without replacement:
        # sum_i 1 - C(N-c_i, d) / C(N, d). If N-c_i < d, taxon i must be observed.
        log_denom = gammaln(n + 1) - gammaln(depth + 1) - gammaln(n - depth + 1)
        absent = np.zeros_like(positive, dtype=float)
        feasible = (n - positive) >= depth
        log_num = (
            gammaln(n - positive[feasible] + 1)
            - gammaln(depth + 1)
            - gammaln(n - positive[feasible] - depth + 1)
        )
        absent[feasible] = np.exp(log_num - log_denom)
        values.append(float(np.sum(1.0 - absent)))
    return pd.Series(values, index=counts.index)


def primary_covariates(data: pd.DataFrame, extra: list[str] | None = None) -> list[str]:
    covariates = CORE_COVARIATES.copy()
    if sex_is_estimable(data):
        covariates.append("SEX")
    if extra:
        covariates.extend(extra)
    return covariates


def run_rarefaction_sensitivity(base: pd.DataFrame, wide: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    within_3y_ids = base.loc[base["gap_days_abs"] <= TIMING_BRANCHES["within_3y"], "SequencingSpecificName"]
    totals = wide.loc[wide.index.intersection(within_3y_ids)].sum(axis=1)
    depth = int(np.floor(totals.quantile(0.10)))
    rare = expected_rarefied_richness(wide, depth).rename("alpha_richness_rarefied_expected")
    dat = base.merge(rare.reset_index().rename(columns={"index": "SequencingSpecificName"}), on="SequencingSpecificName", how="left")
    branch = dat[dat["gap_days_abs"] <= TIMING_BRANCHES["within_3y"]].copy()
    rows = []
    for outcome_id, outcome_col in PRIMARY_OUTCOMES.items():
        matched = branch[branch["alpha_richness_rarefied_expected"].notna()].copy()
        for feature in ["alpha_richness", "alpha_richness_rarefied_expected"]:
            row = {
                "analysis": "rarefaction_depth_sensitivity",
                "branch_id": "within_3y",
                "outcome_id": outcome_id,
                "feature": feature,
                "rarefaction_depth": depth,
                "sample_set": "matched_rarefaction_eligible_subset",
            }
            try:
                row.update(binary_association(matched, outcome_col, feature, primary_covariates(matched)))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    mask = out["p_value"].notna() if "p_value" in out else pd.Series(False, index=out.index)
    if mask.any():
        out.loc[mask, "q_value"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    summary = {
        "rarefaction_depth": depth,
        "within_3y_total_read_p10": float(totals.quantile(0.10)),
        "within_3y_samples_below_depth": int((totals < depth).sum()),
        "within_3y_samples_total": int(len(totals)),
    }
    return out, summary


def design_matrix(data: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    parts = [pd.Series(1.0, index=data.index, name="intercept")]
    for cov in covariates:
        if cov not in data.columns:
            continue
        if data[cov].dtype == object or str(data[cov].dtype).startswith(("string", "category")):
            dummies = pd.get_dummies(data[cov].astype(str), prefix=cov, drop_first=True).astype(float)
            parts.append(dummies)
        else:
            parts.append(standardize(data[cov]).rename(cov))
    return pd.concat(parts, axis=1).astype(float)


def residualize_matrix(features: np.ndarray, design: np.ndarray) -> np.ndarray:
    coef = np.linalg.lstsq(design, features, rcond=None)[0]
    return features - design @ coef


def _orthonormal_basis(design: np.ndarray) -> tuple[np.ndarray, int]:
    q, r = np.linalg.qr(design, mode="reduced")
    tolerance = np.finfo(float).eps * max(design.shape) * abs(np.diag(r)).max()
    rank = int((abs(np.diag(r)) > tolerance).sum())
    return q[:, :rank], rank


def _residualize(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    return values - q @ (q.T @ values)


def _permute_family_blocks(
    residuals: np.ndarray,
    family_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute complete family residual blocks among families of equal size."""
    family_blocks: dict[int, list[np.ndarray]] = {}
    for family in pd.unique(family_ids):
        idx = np.flatnonzero(family_ids == family)
        family_blocks.setdefault(len(idx), []).append(idx)
    permuted = np.empty_like(residuals)
    for block_size, targets in family_blocks.items():
        sources = [targets[i] for i in rng.permutation(len(targets))]
        for target, source in zip(targets, sources, strict=True):
            source_order = source.copy()
            if block_size > 1:
                source_order = source_order[rng.permutation(block_size)]
            permuted[target] = residuals[source_order]
    return permuted


def _partial_multivariate_statistic(
    features: np.ndarray,
    reduced_design: np.ndarray,
    exposure: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, int]:
    q0, rank0 = _orthonormal_basis(reduced_design)
    residual_features = _residualize(features, q0)
    residual_exposure = _residualize(exposure[:, None], q0).ravel()
    exposure_norm = float(np.linalg.norm(residual_exposure))
    if exposure_norm == 0:
        raise ValueError("Outcome has no variation after covariate adjustment")
    exposure_unit = residual_exposure / exposure_norm
    ss_total = float(np.square(residual_features).sum())
    ss_effect = float(np.square(exposure_unit @ residual_features).sum())
    rank1 = rank0 + 1
    df_residual = int(len(exposure) - rank1)
    ss_residual = ss_total - ss_effect
    pseudo_f = float((ss_effect / 1.0) / (ss_residual / df_residual))
    partial_r2 = float(ss_effect / ss_total)
    return pseudo_f, partial_r2, residual_features, q0, exposure_unit, df_residual


def permanova_freedman_lane(
    data: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    cols = [outcome, "FamilyID", *covariates]
    dat = data[[c for c in cols if c in data.columns]].copy()
    dat = dat.join(feature_matrix, how="inner")
    dat[outcome] = pd.to_numeric(dat[outcome], errors="coerce")
    dat = dat.dropna(subset=[outcome, "FamilyID", *[c for c in covariates if c in dat.columns]])
    y = dat[outcome].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return {}
    feature_cols = feature_matrix.columns.intersection(dat.columns)
    features = dat[feature_cols].astype(float).to_numpy()
    design = design_matrix(dat, covariates).to_numpy()
    obs, partial_r2, residuals, q0, exposure_unit, df_residual = _partial_multivariate_statistic(
        features, design, y.astype(float)
    )
    family_ids = dat["FamilyID"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    perm_stats: list[float] = []
    for _ in range(permutations):
        permuted = _permute_family_blocks(residuals, family_ids, rng)
        permuted = _residualize(permuted, q0)
        ss_total = float(np.square(permuted).sum())
        ss_effect = float(np.square(exposure_unit @ permuted).sum())
        ss_residual = ss_total - ss_effect
        perm_stats.append(float((ss_effect / 1.0) / (ss_residual / df_residual)))
    perm = np.array(perm_stats, dtype=float)
    p = (np.sum(perm >= obs) + 1.0) / (np.sum(np.isfinite(perm)) + 1.0)
    return {
        "n": int(len(y)),
        "cases": int(y.sum()),
        "controls": int(len(y) - y.sum()),
        "families": int(pd.Series(family_ids).nunique()),
        "features": int(len(feature_cols)),
        "pseudo_f": obs,
        "partial_r2": partial_r2,
        "permutation_p_value": float(p),
        "permutations": int(permutations),
        "permutation_method": "Freedman-Lane residual permutation with complete family blocks permuted among families of equal size",
        "seed": int(seed),
    }


def one_per_family_dispersion(
    data: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    analytic_cols = [outcome, "FamilyID", *[c for c in covariates if c in data.columns]]
    dat = data[analytic_cols].copy()
    dat["SequencingSpecificName"] = dat.index.astype(str)
    dat = dat.join(feature_matrix, how="inner")
    dat[outcome] = pd.to_numeric(dat[outcome], errors="coerce")
    dat = dat.dropna(subset=analytic_cols).sort_values(
        ["FamilyID", "SequencingSpecificName"], kind="stable"
    )
    dat = dat.drop_duplicates("FamilyID", keep="first")
    y = dat[outcome].astype(int).to_numpy()
    feature_cols = feature_matrix.columns.intersection(dat.columns)
    values = dat[feature_cols].astype(float).to_numpy()
    def dispersion_f(labels: np.ndarray) -> float:
        distances = np.empty(len(dat), dtype=float)
        for level in (0, 1):
            mask = labels == level
            centroid = values[mask].mean(axis=0)
            distances[mask] = np.linalg.norm(values[mask] - centroid, axis=1)
        x = np.column_stack([np.ones(len(labels)), labels])
        return float(sm.OLS(distances, x).fit().fvalue)

    observed_f = dispersion_f(y)
    rng = np.random.default_rng(seed)
    permuted_f = []
    for _ in range(permutations):
        permuted_f.append(dispersion_f(rng.permutation(y)))
    p_value = float((np.sum(np.asarray(permuted_f) >= observed_f) + 1) / (permutations + 1))
    return {
        "dispersion_one_per_family_n": int(len(y)),
        "dispersion_cases": int(y.sum()),
        "dispersion_controls": int(len(y) - y.sum()),
        "dispersion_f": observed_f,
        "dispersion_p_value": p_value,
        "dispersion_permutations": int(permutations),
        "dispersion_seed": int(seed),
    }


def run_beta_diversity(base: pd.DataFrame, genus_counts: pd.DataFrame, permutations: int) -> pd.DataFrame:
    branch = base[base["gap_days_abs"] <= TIMING_BRANCHES["within_3y"]].copy().set_index("SequencingSpecificName")
    counts = genus_counts.loc[genus_counts.index.intersection(branch.index)].copy()
    prevalence = (counts > 0).mean()
    counts = counts.loc[:, prevalence >= 0.10]
    clr = np.log(counts + 0.5)
    clr = clr.sub(clr.mean(axis=1), axis=0)
    rows = []
    for outcome_id, outcome_col in PRIMARY_OUTCOMES.items():
        row = {
            "analysis": "residualized_clr_global_community",
            "branch_id": "within_3y",
            "outcome_id": outcome_id,
            "distance_space": "genus_count_clr_pseudocount_0.5_euclidean",
            "permutation_note": "Complete-case covariate-adjusted Freedman-Lane test with equal-size family-block permutation.",
        }
        try:
            row.update(
                permanova_freedman_lane(
                    branch,
                    clr,
                    outcome_col,
                    primary_covariates(branch),
                    permutations,
                    seed=20260814,
                )
            )
            row.update(
                one_per_family_dispersion(
                    branch,
                    clr,
                    outcome_col,
                    primary_covariates(branch),
                    permutations,
                    seed=20260815,
                )
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    mask = out["permutation_p_value"].notna() if "permutation_p_value" in out else pd.Series(False, index=out.index)
    if mask.any():
        out.loc[mask, "q_value"] = multipletests(out.loc[mask, "permutation_p_value"], method="fdr_bh")[1]
    return out


def run_clr_candidate_sensitivity(base: pd.DataFrame, genus_counts: pd.DataFrame) -> pd.DataFrame:
    branch = base[base["gap_days_abs"] <= TIMING_BRANCHES["within_3y"]].copy().set_index("SequencingSpecificName")
    counts = genus_counts.loc[genus_counts.index.intersection(branch.index)].copy()
    reference_outcome = BINARY_OUTCOMES["highly_symptomatic_and"]
    reference = branch.reset_index().dropna(subset=[reference_outcome, *CORE_COVARIATES, "FamilyID"])
    reference_covariates = primary_covariates(reference)
    reference = reference.dropna(subset=[c for c in reference_covariates if c in reference.columns])
    reference_counts = counts.loc[counts.index.intersection(reference["SequencingSpecificName"].astype(str))]
    prevalence = (reference_counts > 0).mean(axis=0)
    fixed_taxa = prevalence[prevalence >= 0.10].index.tolist()
    counts = counts.loc[:, fixed_taxa]
    clr = np.log(counts + 0.5)
    clr = clr.sub(clr.mean(axis=1), axis=0)
    candidate_names = [name.replace("genus__", "") for name in FROZEN_CANDIDATE_GENERA]
    reverse_lookup = {clean_feature_name(col): col for col in clr.columns}
    rows = []
    for candidate_clean in candidate_names:
        if candidate_clean not in reverse_lookup:
            continue
        genus = reverse_lookup[candidate_clean]
        feature = f"clr__{candidate_clean}"
        branch[feature] = clr[genus]
        for outcome_id, outcome_col in PRIMARY_OUTCOMES.items():
            row = {
                "analysis": "clr_pseudocount_candidate_sensitivity",
                "branch_id": "within_3y",
                "outcome_id": outcome_id,
                "feature": feature,
                "candidate": candidate_clean,
                "clr_universe_taxa": int(len(fixed_taxa)),
                "clr_universe_definition": "assigned genera present in >=10% of the complete-case within-3-year highly_symptomatic_and cohort",
            }
            try:
                row.update(binary_association(branch.reset_index(), outcome_col, feature, primary_covariates(branch.reset_index())))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    mask = out["p_value"].notna() if "p_value" in out else pd.Series(False, index=out.index)
    if mask.any():
        out.loc[mask, "q_value"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    return out


def load_ibs(data_root: Path) -> pd.DataFrame:
    path = data_root / IBS_SOURCE
    ibs = pd.read_excel(path)
    ibs = ibs.rename(columns={c: str(c).strip() for c in ibs.columns})
    if "MasterSet.ParticipantPublicID" in ibs.columns:
        ibs = ibs.rename(columns={"MasterSet.ParticipantPublicID": "ParticipantPublicID"})
    if "MasterSet.LtdCriteriaIBS" in ibs.columns:
        ibs = ibs.rename(columns={"MasterSet.LtdCriteriaIBS": "LtdCriteriaIBS"})
    ibs["PublicID"] = ibs["ParticipantPublicID"].astype(str)
    ibs["LtdCriteriaIBS"] = pd.to_numeric(ibs["LtdCriteriaIBS"], errors="coerce")
    return ibs[["PublicID", "LtdCriteriaIBS"]].drop_duplicates("PublicID")


def run_ibs_sensitivity(base: pd.DataFrame, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ibs = load_ibs(data_root)
    dat = base.merge(ibs, on="PublicID", how="left")
    branch = dat[dat["gap_days_abs"] <= TIMING_BRANCHES["within_3y"]].copy()
    coverage_rows = []
    for outcome_id, outcome_col in PRIMARY_OUTCOMES.items():
        y = pd.to_numeric(branch[outcome_col], errors="coerce")
        has = branch["LtdCriteriaIBS"].notna()
        coverage_rows.append(
            {
                "outcome_id": outcome_id,
                "n_total": int(y.notna().sum()),
                "n_with_ibs": int((y.notna() & has).sum()),
                "cases_total": int((y == 1).sum()),
                "cases_with_ibs": int(((y == 1) & has).sum()),
                "controls_with_ibs": int(((y == 0) & has).sum()),
                "ibs_positive_with_data": int(((branch["LtdCriteriaIBS"] == 1) & y.notna()).sum()),
            }
        )
    rows = []
    ibs_available = branch[branch["LtdCriteriaIBS"].notna()].copy()
    for outcome_id, outcome_col in PRIMARY_OUTCOMES.items():
        for feature in PRIMARY_FEATURES:
            if feature not in branch.columns:
                continue
            for adjustment_id, extra in [("primary_covariates", []), ("primary_plus_ibs", ["LtdCriteriaIBS"])]:
                row = {
                    "analysis": "ibs_sensitivity",
                    "branch_id": "within_3y",
                    "outcome_id": outcome_id,
                    "feature": feature,
                    "adjustment_id": adjustment_id,
                }
                try:
                    row.update(
                        binary_association(
                            ibs_available,
                            outcome_col,
                            feature,
                            primary_covariates(ibs_available, extra=extra),
                        )
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    mask = out["p_value"].notna() if "p_value" in out else pd.Series(False, index=out.index)
    if mask.any():
        out.loc[mask, "q_value"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    return pd.DataFrame(coverage_rows), out


def run_sequencing_run_sensitivity(base: pd.DataFrame) -> pd.DataFrame:
    branch = base[base["gap_days_abs"] <= TIMING_BRANCHES["within_3y"]].copy()
    rows: list[dict[str, Any]] = []
    for outcome_id, outcome_col in PRIMARY_OUTCOMES.items():
        for feature in ["alpha_richness", "alpha_shannon"]:
            for adjustment_id, extra in [
                ("primary_covariates", []),
                ("primary_plus_sequencing_run", ["SequencingRunID"]),
            ]:
                row: dict[str, Any] = {
                    "analysis": "sequencing_run_sensitivity",
                    "branch_id": "within_3y",
                    "outcome_id": outcome_id,
                    "feature": feature,
                    "adjustment_id": adjustment_id,
                    "sequencing_run_levels": int(branch["SequencingRunID"].dropna().astype(str).nunique()),
                }
                try:
                    row.update(
                        binary_association(
                            branch,
                            outcome_col,
                            feature,
                            primary_covariates(branch, extra=extra),
                        )
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    mask = out["p_value"].notna() if "p_value" in out else pd.Series(False, index=out.index)
    if mask.any():
        out.loc[mask, "q_value"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    return out


def write_summary_figure(out_dir: Path, rare: pd.DataFrame, clr: pd.DataFrame, ibs: pd.DataFrame) -> None:
    rows = []
    for label, df in [
        ("Observed richness", rare[rare["feature"] == "alpha_richness"]),
        ("Rarefied richness", rare[rare["feature"] == "alpha_richness_rarefied_expected"]),
        ("CLR Dialister", clr[clr["candidate"] == "Dialister"] if "candidate" in clr else pd.DataFrame()),
        ("IBS-adjusted richness", ibs[(ibs["feature"] == "alpha_richness") & (ibs["adjustment_id"] == "primary_plus_ibs")] if "adjustment_id" in ibs else pd.DataFrame()),
    ]:
        hit = df[df["outcome_id"] == "highly_symptomatic_and"].copy()
        if not hit.empty and "odds_ratio_per_sd" in hit:
            rows.append({"label": label, "or": float(hit["odds_ratio_per_sd"].iloc[0]), "p": float(hit["p_value"].iloc[0])})
    if not rows:
        return
    plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = ["#2f6f9f" if val < 1 else "#b64c4c" for val in plot["or"]]
    ax.barh(plot["label"], plot["or"], color=colors)
    ax.axvline(1.0, color="#333333", linewidth=1)
    ax.set_xlabel("Odds ratio per 1 SD higher feature")
    ax.set_title("Primary Highly Symptomatic Outcome: Method-Strengthening Checks")
    ax.invert_yaxis()
    for i, row in plot.iterrows():
        ax.text(row["or"], i, f"  OR {row['or']:.2f}, p {row['p']:.3g}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_method_strengthening_primary_or.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run compact method-strengthening analyses for dry-eye gut microbiome.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument(
        "--refresh-ibs-only",
        action="store_true",
        help="Refresh matched-subset IBS coverage and sensitivity files without rerunning permutation analyses.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base, wide, genus_counts = load_joined_inputs(args.data_root)
    if args.refresh_ibs_only:
        ibs_coverage, ibs_results = run_ibs_sensitivity(base, args.data_root)
        write_table(ibs_coverage, args.out_dir / "ibs_coverage")
        write_table(ibs_results, args.out_dir / "ibs_sensitivity")
        print(
            json.dumps(
                {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "refresh": "matched-subset IBS sensitivity only",
                    "rows": int(len(ibs_results)),
                    "out_dir": str(args.out_dir),
                },
                indent=2,
            )
        )
        return 0
    rare, rare_summary = run_rarefaction_sensitivity(base, wide)
    beta = run_beta_diversity(base, genus_counts, permutations=args.permutations)
    clr = run_clr_candidate_sensitivity(base, genus_counts)
    ibs_coverage, ibs_results = run_ibs_sensitivity(base, args.data_root)
    sequencing_run_results = run_sequencing_run_sensitivity(base)

    write_table(rare, args.out_dir / "rarefaction_richness_sensitivity")
    write_table(beta, args.out_dir / "beta_diversity_global_community")
    write_table(clr, args.out_dir / "clr_candidate_compositional_sensitivity")
    write_table(ibs_coverage, args.out_dir / "ibs_coverage")
    write_table(ibs_results, args.out_dir / "ibs_sensitivity")
    write_table(sequencing_run_results, args.out_dir / "sequencing_run_sensitivity")
    write_summary_figure(args.out_dir, rare, clr, ibs_results)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_rows": int(len(base)),
        "within_3y_rows": int((base["gap_days_abs"] <= TIMING_BRANCHES["within_3y"]).sum()),
        "tie_rule": "highest_read_depth",
        "nearest_tied_questionnaires": int((base["nearest_tie_count"] > 1).sum()),
        "genus_unresolved_asv_policy": "Excluded from genus-level analyses; retained in ASV alpha diversity.",
        "total_asvs": int(wide.shape[1]),
        "assigned_genus_features_before_prevalence_filter": int(genus_counts.shape[1]),
        "assigned_genus_read_fraction": float(genus_counts.to_numpy().sum() / wide.to_numpy().sum()),
        "rarefaction": rare_summary,
        "beta_diversity_rows": int(len(beta)),
        "clr_candidate_rows": int(len(clr)),
        "ibs_sensitivity_rows": int(len(ibs_results)),
        "sequencing_run_sensitivity_rows": int(len(sequencing_run_results)),
        "sequencing_location_levels": int(base["SequencingLocation"].dropna().astype(str).nunique()),
        "ancombc2_status": "not_run; Rscript/ANCOMBC not available on active PATH. CLR-pseudocount candidate sensitivity was run as local compositional sensitivity.",
        "interpretation_boundary": "Method-strengthening sensitivity checks; not new primary discovery branches.",
        "no_raw_data_written": True,
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
