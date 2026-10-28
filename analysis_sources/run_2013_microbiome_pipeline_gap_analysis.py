from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import re
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests

from run_2013_initial_microbiome_signal_probes import (
    DEFAULT_DATA_ROOT,
    MICROBIOME_DIR,
    PROJECT_ROOT,
    load_asv_counts,
    load_ded_2013,
    load_family_ids,
    load_mapping,
    load_taxa,
    nearest_stool_pairs,
    sample_read_depths,
)
from run_2013_locked_primary_16s_analysis import (
    CORE_COVARIATES,
    binary_association,
    sex_is_estimable,
)
from run_2013_method_strengthening_batch import (
    _orthonormal_basis,
    _permute_family_blocks,
    _residualize,
    design_matrix,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / 'runtime' / 'microbiome_pipeline_gap_analysis_2026_08_25'
PRIMARY_OUTCOME = "highly_symptomatic_derived"
SHORT_UNMERGED_RUN = "TwinsUKPlate28_33"
WITHIN_3Y_DAYS = 365 * 3
EXPECTED_N = 1685
EXPECTED_CASES = 93
EXPECTED_FAMILIES = 1125
EXPECTED_PRIMARY_OR = 0.7206689313902943
FORMAL_INPUT_DIR = PROJECT_ROOT / 'runtime' / 'coauthor_review_production_2026_08_14' / 'formal_inputs' / 'highly_symptomatic_and'


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_table(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primary_covariates(data: pd.DataFrame, extra: list[str] | None = None) -> list[str]:
    covariates = CORE_COVARIATES.copy()
    if sex_is_estimable(data):
        covariates.append("SEX")
    if extra:
        covariates.extend(extra)
    return covariates


def add_bh(frame: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    out = frame.copy()
    out["q_value"] = np.nan
    if p_col not in out:
        return out
    mask = pd.to_numeric(out[p_col], errors="coerce").notna()
    if mask.any():
        out.loc[mask, "q_value"] = multipletests(
            pd.to_numeric(out.loc[mask, p_col], errors="coerce"), method="fdr_bh"
        )[1]
    return out


def feature_descriptives(data: pd.DataFrame, feature: str) -> dict[str, float]:
    outcome = pd.to_numeric(data[PRIMARY_OUTCOME], errors="coerce")
    values = pd.to_numeric(data[feature], errors="coerce")
    case_values = values[outcome == 1].dropna()
    control_values = values[outcome == 0].dropna()
    return {
        "case_mean": float(case_values.mean()),
        "case_median": float(case_values.median()),
        "control_mean": float(control_values.mean()),
        "control_median": float(control_values.median()),
    }


def association_row(
    data: pd.DataFrame,
    feature: str,
    analysis: str,
    covariates: list[str],
    **metadata: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {"analysis": analysis, "feature": feature, **metadata}
    row.update(feature_descriptives(data, feature))
    try:
        row.update(binary_association(data, PRIMARY_OUTCOME, feature, covariates))
    except Exception as exc:  # fail closed but preserve an aggregate diagnostic
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def hill_numbers(counts: np.ndarray, chunk_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    q1 = np.empty(counts.shape[0], dtype=float)
    q2 = np.empty(counts.shape[0], dtype=float)
    for start in range(0, counts.shape[0], chunk_size):
        stop = min(start + chunk_size, counts.shape[0])
        block = counts[start:stop].astype(np.float64, copy=False)
        totals = block.sum(axis=1)
        proportions = np.divide(
            block,
            totals[:, None],
            out=np.zeros_like(block, dtype=np.float64),
            where=totals[:, None] > 0,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.zeros_like(proportions)
            positive = proportions > 0
            logs[positive] = np.log(proportions[positive])
            entropy = -(proportions * logs).sum(axis=1)
        concentration = np.square(proportions).sum(axis=1)
        q1[start:stop] = np.exp(entropy)
        q2[start:stop] = np.divide(
            1.0,
            concentration,
            out=np.full_like(concentration, np.nan),
            where=concentration > 0,
        )
    return q1, q2


def grouped_richness(presence: csr_matrix, labels: pd.Series, eligible: np.ndarray) -> tuple[np.ndarray, int]:
    clean = labels.astype("string").str.strip()
    valid = eligible & clean.notna().to_numpy() & ~clean.isin(
        ["", "NA", "N/A", "nan", "None", "Unassigned", "unassigned"]
    ).to_numpy()
    codes, levels = pd.factorize(clean.loc[valid], sort=True)
    feature_indices = np.flatnonzero(valid)
    membership = csr_matrix(
        (np.ones(len(feature_indices), dtype=np.int8), (feature_indices, codes)),
        shape=(presence.shape[1], len(levels)),
    )
    detected = presence @ membership
    return np.asarray((detected > 0).sum(axis=1)).ravel(), int(len(levels))


def gower_center(distance: np.ndarray) -> np.ndarray:
    squared = np.square(distance, dtype=np.float64)
    row_mean = squared.mean(axis=1, keepdims=True)
    col_mean = squared.mean(axis=0, keepdims=True)
    grand_mean = float(squared.mean())
    return -0.5 * (squared - row_mean - col_mean + grand_mean)


def distance_partial_test(
    metadata: pd.DataFrame,
    profiles: np.ndarray,
    metric: str,
    permutations: int,
    seed: int,
    covariates: list[str] | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    started = time.perf_counter()
    distances = np.sqrt(squareform(pdist(profiles, metric=metric)))
    gower = gower_center(distances)
    if covariates is None:
        covariates = ["age", "BMI", "AntibioticsPriorMonth", "gap_days_abs", "SEX"]
    reduced_design = design_matrix(metadata, covariates).to_numpy(dtype=float)
    q0, rank0 = _orthonormal_basis(reduced_design)
    y = metadata[PRIMARY_OUTCOME].astype(int).to_numpy(dtype=float)
    residual = _residualize(y[:, None], q0).ravel()
    norm = float(np.linalg.norm(residual))
    if norm == 0:
        raise RuntimeError("Primary outcome has no residual variation after covariate adjustment")
    unit = residual / norm

    total_ss = float(np.trace(gower) - np.trace(q0.T @ gower @ q0))
    effect_ss = float(unit @ gower @ unit)
    df_residual = int(len(metadata) - rank0 - 1)
    residual_ss = float(total_ss - effect_ss)
    observed_f = float(effect_ss / (residual_ss / df_residual))
    partial_r2 = float(effect_ss / total_ss)

    family_ids = metadata["FamilyID"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    exceed = 0
    finite = 0
    done = 0
    while done < permutations:
        current = min(batch_size, permutations - done)
        units = np.empty((len(metadata), current), dtype=float)
        for column in range(current):
            permuted = _permute_family_blocks(residual[:, None], family_ids, rng).ravel()
            permuted = _residualize(permuted[:, None], q0).ravel()
            permuted_norm = float(np.linalg.norm(permuted))
            units[:, column] = permuted / permuted_norm
        perm_effect = np.einsum("ij,ij->j", units, gower @ units)
        perm_residual = total_ss - perm_effect
        perm_f = perm_effect / (perm_residual / df_residual)
        valid = np.isfinite(perm_f)
        finite += int(valid.sum())
        exceed += int(np.sum(perm_f[valid] >= observed_f))
        done += current

    result = {
        "n": int(len(metadata)),
        "cases": int(y.sum()),
        "controls": int(len(y) - y.sum()),
        "families": int(pd.Series(family_ids).nunique()),
        "features": int(profiles.shape[1]),
        "pseudo_f": observed_f,
        "partial_r2": partial_r2,
        "permutation_p_value": float((exceed + 1) / (finite + 1)),
        "permutations": int(permutations),
        "permutation_method": (
            "Permutation of covariate-residualized dry-eye status with complete family blocks "
            "exchanged among families of equal size"
        ),
        "covariates": ";".join(covariates),
        "distance_transform": "square root",
        "seed": int(seed),
        "runtime_seconds_global_test": float(time.perf_counter() - started),
    }
    result.update(distance_dispersion_test(metadata, profiles, metric, permutations, seed + 1))
    return result


def distance_dispersion_test(
    metadata: pd.DataFrame,
    profiles: np.ndarray,
    metric: str,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    ordered = metadata.assign(_row=np.arange(len(metadata))).sort_values(
        ["FamilyID", "sample_id"], kind="stable"
    )
    selected = ordered.drop_duplicates("FamilyID", keep="first")
    indices = selected["_row"].to_numpy(dtype=int)
    labels = selected[PRIMARY_OUTCOME].astype(int).to_numpy()
    reduced_profiles = profiles[indices]
    distances = np.sqrt(squareform(pdist(reduced_profiles, metric=metric)))
    gower = gower_center(distances)
    eigenvalues, eigenvectors = np.linalg.eigh(gower)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = scale * 1e-10
    positive = eigenvalues > tolerance
    negative = eigenvalues < -tolerance
    pos_coords = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    neg_coords = eigenvectors[:, negative] * np.sqrt(-eigenvalues[negative])

    def distances_to_centroid(group_labels: np.ndarray) -> np.ndarray:
        values = np.empty(len(group_labels), dtype=float)
        for level in (0, 1):
            mask = group_labels == level
            pos_centroid = pos_coords[mask].mean(axis=0) if pos_coords.shape[1] else np.empty(0)
            neg_centroid = neg_coords[mask].mean(axis=0) if neg_coords.shape[1] else np.empty(0)
            pos_sq = np.square(pos_coords[mask] - pos_centroid).sum(axis=1) if pos_coords.shape[1] else 0.0
            neg_sq = np.square(neg_coords[mask] - neg_centroid).sum(axis=1) if neg_coords.shape[1] else 0.0
            values[mask] = np.sqrt(np.maximum(np.asarray(pos_sq) - np.asarray(neg_sq), 0.0))
        return values

    def dispersion_f(group_labels: np.ndarray) -> float:
        response = distances_to_centroid(group_labels)
        design = sm.add_constant(group_labels.astype(float), has_constant="add")
        return float(sm.OLS(response, design).fit().fvalue)

    observed_f = dispersion_f(labels)
    rng = np.random.default_rng(seed)
    permuted = np.array([dispersion_f(rng.permutation(labels)) for _ in range(permutations)])
    denominator = float(np.abs(eigenvalues).sum())
    negative_fraction = float(np.abs(eigenvalues[negative]).sum() / denominator) if denominator else 0.0
    return {
        "dispersion_one_per_family_n": int(len(labels)),
        "dispersion_cases": int(labels.sum()),
        "dispersion_controls": int(len(labels) - labels.sum()),
        "dispersion_f": observed_f,
        "dispersion_p_value": float((np.sum(permuted >= observed_f) + 1) / (permutations + 1)),
        "dispersion_permutations": int(permutations),
        "dispersion_seed": int(seed),
        "pcoa_positive_axes": int(positive.sum()),
        "pcoa_negative_axes": int(negative.sum()),
        "pcoa_negative_eigenvalue_fraction": negative_fraction,
    }


def run_community_tests(
    permutations: int,
    seed_offset: int = 0,
    extra_covariates: list[str] | None = None,
) -> pd.DataFrame:
    counts = pd.read_csv(FORMAL_INPUT_DIR / "genus_counts.csv", index_col=0).T
    counts.index = counts.index.astype(str)
    metadata = pd.read_csv(FORMAL_INPUT_DIR / "metadata.csv")
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata = metadata.set_index("sample_id").loc[counts.index]
    metadata.index.name = "sample_id"
    metadata = metadata.reset_index()
    metadata[PRIMARY_OUTCOME] = metadata["dry_eye_status"].map({"Control": 0, "Case": 1})
    if metadata[PRIMARY_OUTCOME].isna().any():
        raise RuntimeError("Unexpected dry-eye status in formal community metadata")
    if counts.shape != (EXPECTED_N, 142):
        raise RuntimeError(f"Expected a 1685 by 142 fixed-genus matrix, found {counts.shape}")

    raw = counts.to_numpy(dtype=float)
    relative = raw / raw.sum(axis=1, keepdims=True)
    presence = raw > 0
    covariates = ["age", "BMI", "AntibioticsPriorMonth", "gap_days_abs", "SEX"]
    if extra_covariates:
        covariates.extend(extra_covariates)
    rows: list[dict[str, Any]] = []
    for analysis, metric, profiles, seed in [
        ("genus_presence_jaccard", "jaccard", presence, 20260825 + seed_offset),
        ("genus_relative_abundance_bray_curtis", "braycurtis", relative, 20260827 + seed_offset),
    ]:
        row: dict[str, Any] = {
            "analysis": analysis,
            "distance_space": f"square_root_{metric}",
            "feature_universe": "fixed 142 genera used by the validated Aitchison analysis",
        }
        row.update(
            distance_partial_test(
                metadata,
                profiles,
                metric,
                permutations,
                seed,
                covariates=covariates,
            )
        )
        rows.append(row)
    return add_bh(pd.DataFrame(rows), p_col="permutation_p_value")


def tree_feasibility(tree_path: Path, count_asvs: set[str]) -> dict[str, Any]:
    with zipfile.ZipFile(tree_path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".tre"))
        newick = archive.read(member).decode("utf-8", errors="replace").strip()
    tips = set(re.findall(r"(?<=[(,])([ACGT]+):", newick))
    lengths = np.asarray(
        [float(value) for value in re.findall(r":([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", newick)],
        dtype=float,
    )
    rooting_marker = bool(newick.startswith("[&R]"))
    all_count_tips_present = count_asvs.issubset(tips)
    branch_lengths_valid = bool(len(lengths) and np.isfinite(lengths).all() and (lengths >= 0).all())
    return {
        "tree_member": member,
        "tree_tips": int(len(tips)),
        "count_asvs": int(len(count_asvs)),
        "count_asvs_present_in_tree": int(len(count_asvs & tips)),
        "count_asvs_missing_from_tree": int(len(count_asvs - tips)),
        "tree_only_tips": int(len(tips - count_asvs)),
        "branch_length_count": int(len(lengths)),
        "branch_length_min": float(lengths.min()) if len(lengths) else None,
        "branch_length_max": float(lengths.max()) if len(lengths) else None,
        "negative_branch_lengths": int((lengths < 0).sum()) if len(lengths) else None,
        "explicit_rooting_marker": rooting_marker,
        "rooting_documented_in_supplied_readme": False,
        "all_count_tips_present": all_count_tips_present,
        "branch_lengths_valid": branch_lengths_valid,
        "faith_pd_or_unifrac_run": False,
        "decision": (
            "Do not run manuscript-facing Faith PD or UniFrac from this delivery because the tree is aligned "
            "and has valid branch lengths, but its rooting is not documented. Arbitrary rooting would add a "
            "new analyst choice without a clear gain over the existing and newly tested non-phylogenetic metrics."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit overlooked 16S pipeline elements and run bounded sensitivities.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--skip-community", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    microbiome_root = args.data_root / MICROBIOME_DIR
    source_paths = {
        "counts": microbiome_root / "TwinsUK_ASVsCR.rds",
        "taxonomy": microbiome_root / "TwinsUK_ASVs_Taxa.csv.zip",
        "mapping": microbiome_root / "TwinsUK_ASVs_MappingFile.xlsx",
        "tree": microbiome_root / "TwinsUK_ASVsTree.tre.zip",
        "readme": microbiome_root / "ReadMe_TwinsUK_ASVs.docx",
        "column_dictionary": microbiome_root / "col_desc.csv",
        "ded_2013": args.data_root / "DED_outcomes_excel" / "DED_outcomes_2013.xlsx",
        "family_spine": args.data_root / "derived" / "publicid_spine.csv",
    }
    missing = [name for name, path in source_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source files: {missing}")

    source_manifest = pd.DataFrame(
        [
            {
                "source": name,
                "filename": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for name, path in source_paths.items()
        ]
    )
    write_table(args.out_dir / "source_manifest.csv", source_manifest)

    counts = load_asv_counts(args.data_root)
    taxa = load_taxa(args.data_root)
    mapping = load_mapping(args.data_root)
    full_mapping = pd.read_excel(source_paths["mapping"])
    extra_mapping = full_mapping[
        ["SequencingSpecificName", "CollectionMethod", "ShouldBeAnalyzed", "ShouldExcludeSample"]
    ].copy()
    extra_mapping["SequencingSpecificName"] = extra_mapping["SequencingSpecificName"].astype(str)
    mapping = mapping.merge(
        extra_mapping.drop(columns=["ShouldExcludeSample"]),
        on="SequencingSpecificName",
        how="left",
        validate="one_to_one",
    )
    ded = load_ded_2013(args.data_root)
    family = load_family_ids(args.data_root)
    depths = sample_read_depths(counts)
    pairs = nearest_stool_pairs(
        ded,
        mapping,
        family,
        sample_depths=depths,
        tie_rule="highest_read_depth",
    )
    pairs = pairs.merge(
        extra_mapping[["SequencingSpecificName", "CollectionMethod", "ShouldBeAnalyzed"]],
        on="SequencingSpecificName",
        how="left",
        validate="many_to_one",
    )
    branch = pairs.loc[pairs["gap_days_abs"] <= WITHIN_3Y_DAYS].copy().reset_index(drop=True)

    asv_columns = [column for column in counts.columns if column != "SequencingSpecificName"]
    count_asv_set = set(map(str, asv_columns))
    taxonomy_aligned = taxa.set_index("asv").reindex(asv_columns)
    if taxonomy_aligned.index.isna().any() or taxonomy_aligned["Kingdom"].isna().any():
        raise RuntimeError("Count-table ASVs did not align completely to kingdom-assigned taxonomy")

    count_table = counts.set_index("SequencingSpecificName")[asv_columns]
    count_table.index = count_table.index.astype(str)
    full_column_sums = count_table.sum(axis=0).to_numpy(dtype=np.int64)
    sample_names = branch["SequencingSpecificName"].astype(str).tolist()
    analysis_counts = count_table.loc[sample_names].to_numpy(dtype=np.int32, copy=True)
    del count_table, counts
    gc.collect()

    branch_total_reads = analysis_counts.sum(axis=1).astype(float)
    branch["alpha_total_reads"] = branch_total_reads
    observed_richness = (analysis_counts > 0).sum(axis=1).astype(float)
    branch["alpha_richness"] = observed_richness

    taxonomy_text = taxonomy_aligned.astype("string").fillna("")
    organelle_mask = np.asarray(taxonomy_text.apply(
        lambda column: column.str.contains("chloroplast|mitochond", case=False, regex=True)
    ).any(axis=1).to_numpy(), dtype=bool)
    archaea_mask = np.asarray(
        taxonomy_aligned["Kingdom"].astype("string").eq("Archaea").fillna(False).to_numpy(),
        dtype=bool,
    )
    bacterial_target_mask = np.asarray(
        taxonomy_aligned["Kingdom"].astype("string").eq("Bacteria").fillna(False).to_numpy()
        & ~organelle_mask
    , dtype=bool)
    no_organelle_mask = ~organelle_mask

    branch["richness_no_organelle"] = (analysis_counts[:, no_organelle_mask] > 0).sum(axis=1)
    branch["richness_bacterial_no_organelle"] = (analysis_counts[:, bacterial_target_mask] > 0).sum(axis=1)
    branch_column_sums = analysis_counts.sum(axis=0, dtype=np.int64)

    rank_rows: list[dict[str, Any]] = []
    total_full_reads = float(full_column_sums.sum())
    total_branch_reads = float(branch_column_sums.sum())
    for rank in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]:
        assigned = taxonomy_aligned[rank].notna().to_numpy()
        rank_rows.append(
            {
                "rank": rank.lower(),
                "assigned_asvs": int(assigned.sum()),
                "assigned_active_asvs": int((assigned & (full_column_sums > 0)).sum()),
                "asv_assignment_fraction": float(assigned.mean()),
                "full_table_assigned_read_fraction": float(full_column_sums[assigned].sum() / total_full_reads),
                "primary_cohort_assigned_read_fraction": float(
                    branch_column_sums[assigned].sum() / total_branch_reads
                ),
                "unique_assigned_labels": int(taxonomy_aligned.loc[assigned, rank].nunique()),
            }
        )
    write_table(args.out_dir / "taxonomy_assignment_summary.csv", pd.DataFrame(rank_rows))

    non_target_summary = pd.DataFrame(
        [
            {
                "feature_set": "chloroplast_or_mitochondria",
                "asvs": int(organelle_mask.sum()),
                "active_asvs": int((organelle_mask & (full_column_sums > 0)).sum()),
                "full_table_read_fraction": float(full_column_sums[organelle_mask].sum() / total_full_reads),
                "primary_cohort_read_fraction": float(
                    branch_column_sums[organelle_mask].sum() / total_branch_reads
                ),
                "primary_cohort_median_detected_asvs": float(
                    np.median((analysis_counts[:, organelle_mask] > 0).sum(axis=1))
                ),
            },
            {
                "feature_set": "archaea",
                "asvs": int(archaea_mask.sum()),
                "active_asvs": int((archaea_mask & (full_column_sums > 0)).sum()),
                "full_table_read_fraction": float(full_column_sums[archaea_mask].sum() / total_full_reads),
                "primary_cohort_read_fraction": float(branch_column_sums[archaea_mask].sum() / total_branch_reads),
                "primary_cohort_median_detected_asvs": float(
                    np.median((analysis_counts[:, archaea_mask] > 0).sum(axis=1))
                ),
            },
        ]
    )
    write_table(args.out_dir / "non_target_sequence_summary.csv", non_target_summary)

    base_covariates = primary_covariates(branch)
    target_rows = [
        association_row(branch, "alpha_richness", "primary_reproduction", base_covariates),
        association_row(
            branch,
            "richness_no_organelle",
            "remove_chloroplast_and_mitochondrial_asvs",
            base_covariates,
        ),
        association_row(
            branch,
            "richness_bacterial_no_organelle",
            "bacterial_asvs_without_organelles",
            base_covariates,
        ),
        association_row(
            branch,
            "alpha_richness",
            "primary_plus_collection_route",
            primary_covariates(branch, ["CollectionMethod"]),
            collection_route_levels=int(branch["CollectionMethod"].dropna().nunique()),
        ),
        association_row(
            branch.loc[pd.to_numeric(branch["ShouldBeAnalyzed"], errors="coerce").fillna(1).eq(1)].copy(),
            "alpha_richness",
            "exclude_should_be_analyzed_zero",
            base_covariates,
            dictionary_note="Column dictionary says ShouldBeAnalyzed can be ignored",
        ),
        association_row(
            branch.loc[~branch["SequencingRunID"].astype(str).eq(SHORT_UNMERGED_RUN)].copy(),
            "alpha_richness",
            "exclude_documented_short_unmerged_run",
            base_covariates,
            excluded_run=SHORT_UNMERGED_RUN,
        ),
    ]
    target_results = pd.DataFrame(target_rows)
    baseline_beta = float(target_results.loc[target_results["analysis"] == "primary_reproduction", "beta_per_sd"].iloc[0])
    target_results["beta_change_from_primary"] = target_results["beta_per_sd"] - baseline_beta
    write_table(args.out_dir / "target_sequence_and_handling_sensitivities.csv", target_results)

    collection_summary = (
        branch.assign(_outcome=pd.to_numeric(branch[PRIMARY_OUTCOME], errors="coerce"))
        .groupby("CollectionMethod", dropna=False)
        .agg(participants=("SequencingSpecificName", "size"), cases=("_outcome", "sum"), families=("FamilyID", "nunique"))
        .reset_index()
    )
    write_table(args.out_dir / "collection_route_summary.csv", collection_summary)

    detection_rows: list[dict[str, Any]] = []
    for threshold in [1, 2, 5, 10]:
        feature = f"asv_richness_count_ge_{threshold}"
        branch[feature] = (analysis_counts >= threshold).sum(axis=1)
        row = association_row(
            branch,
            feature,
            "detection_threshold_richness",
            primary_covariates(branch, ["alpha_total_reads"]),
            minimum_reads_per_asv=threshold,
            correlation_with_observed_richness=float(
                pd.Series(branch[feature]).corr(pd.Series(observed_richness), method="spearman")
            ),
        )
        detection_rows.append(row)
    detection_results = add_bh(pd.DataFrame(detection_rows))
    write_table(args.out_dir / "detection_threshold_richness.csv", detection_results)

    q1, q2 = hill_numbers(analysis_counts)
    branch["hill_q0_observed_richness"] = observed_richness
    branch["hill_q1_effective_shannon"] = q1
    branch["hill_q2_inverse_simpson"] = q2
    hill_rows: list[dict[str, Any]] = []
    for order, feature, interpretation in [
        (0, "hill_q0_observed_richness", "all detected ASVs weighted equally"),
        (1, "hill_q1_effective_shannon", "ASVs weighted in proportion to abundance"),
        (2, "hill_q2_inverse_simpson", "greater weight given to dominant ASVs"),
    ]:
        hill_rows.append(
            association_row(
                branch,
                feature,
                "hill_diversity_profile",
                base_covariates,
                hill_order=order,
                interpretation=interpretation,
                correlation_with_observed_richness=float(
                    pd.Series(branch[feature]).corr(pd.Series(observed_richness), method="spearman")
                ),
            )
        )
    hill_results = add_bh(pd.DataFrame(hill_rows))
    write_table(args.out_dir / "hill_diversity_profile.csv", hill_results)

    presence = csr_matrix(analysis_counts > 0, dtype=np.int8)
    genus_richness, genus_levels = grouped_richness(
        presence, taxonomy_aligned["Genus"], no_organelle_mask
    )
    family_richness, family_levels = grouped_richness(
        presence, taxonomy_aligned["Family"], no_organelle_mask
    )
    branch["asv_richness_no_organelle"] = branch["richness_no_organelle"]
    branch["assigned_genus_richness"] = genus_richness
    branch["assigned_family_richness"] = family_richness
    taxonomic_rows: list[dict[str, Any]] = []
    for rank, feature, universe in [
        ("ASV", "asv_richness_no_organelle", int(no_organelle_mask.sum())),
        ("genus", "assigned_genus_richness", genus_levels),
        ("family", "assigned_family_richness", family_levels),
    ]:
        taxonomic_rows.append(
            association_row(
                branch,
                feature,
                "taxonomic_resolution_richness",
                base_covariates,
                taxonomic_rank=rank,
                available_features_or_labels=universe,
                correlation_with_observed_richness=float(
                    pd.Series(branch[feature]).corr(pd.Series(observed_richness), method="spearman")
                ),
            )
        )
    taxonomic_results = add_bh(pd.DataFrame(taxonomic_rows))
    write_table(args.out_dir / "taxonomic_resolution_richness.csv", taxonomic_results)

    alpha_read_depth_rows: list[dict[str, Any]] = []
    for source_family, feature in [
        ("Hill q0", "hill_q0_observed_richness"),
        ("Hill q1", "hill_q1_effective_shannon"),
        ("Hill q2", "hill_q2_inverse_simpson"),
        ("ASV rank", "asv_richness_no_organelle"),
        ("genus rank", "assigned_genus_richness"),
        ("family rank", "assigned_family_richness"),
    ]:
        alpha_read_depth_rows.append(
            association_row(
                branch,
                feature,
                "read_depth_adjusted_alpha_sensitivity",
                primary_covariates(branch, ["alpha_total_reads"]),
                source_family=source_family,
            )
        )
    write_table(
        args.out_dir / "alpha_read_depth_sensitivities.csv",
        pd.DataFrame(alpha_read_depth_rows),
    )

    phylogenetic = tree_feasibility(source_paths["tree"], count_asv_set)
    write_json(args.out_dir / "phylogenetic_feasibility.json", phylogenetic)

    community_results = pd.DataFrame()
    if not args.skip_community:
        community_results = run_community_tests(args.permutations)
        write_table(args.out_dir / "community_distance_tests.csv", community_results)
        community_read_depth_results = run_community_tests(
            args.permutations,
            seed_offset=200000,
            extra_covariates=["total_reads"],
        )
        write_table(
            args.out_dir / "community_distance_read_depth_sensitivity.csv",
            community_read_depth_results,
        )

    full_count_names = set(depths.index.astype(str))
    mapping_names = set(full_mapping["SequencingSpecificName"].astype(str))
    raw_control_candidates = [
        path.name
        for path in microbiome_root.iterdir()
        if re.search(r"fastq|fq|blank|negative|mock|control", path.name, flags=re.IGNORECASE)
    ]
    integrity = {
        "count_table_samples": int(len(full_count_names)),
        "mapping_profiles": int(len(mapping_names)),
        "mapping_profiles_in_count_table": int(len(mapping_names & full_count_names)),
        "count_only_profiles": int(len(full_count_names - mapping_names)),
        "mapping_only_profiles": int(len(mapping_names - full_count_names)),
        "count_table_asvs": int(len(asv_columns)),
        "all_zero_asv_columns": int((full_column_sums == 0).sum()),
        "taxonomy_asvs": int(len(taxa)),
        "count_asvs_in_taxonomy": int(taxonomy_aligned["Kingdom"].notna().sum()),
        "mapping_rows_should_exclude": int(pd.to_numeric(full_mapping["ShouldExcludeSample"], errors="coerce").fillna(0).eq(1).sum()),
        "mapping_rows_should_be_analyzed_zero": int(pd.to_numeric(full_mapping["ShouldBeAnalyzed"], errors="coerce").fillna(1).eq(0).sum()),
        "primary_branch_rows_before_complete_case": int(len(branch)),
        "primary_branch_tied_questionnaires": int((branch["nearest_tie_count"] > 1).sum()),
        "raw_or_control_named_files_in_delivery": raw_control_candidates,
        "raw_fastq_available": any(re.search(r"fastq|\.fq", name, flags=re.IGNORECASE) for name in raw_control_candidates),
        "negative_or_mock_control_table_available": any(
            re.search(r"blank|negative|mock|control", name, flags=re.IGNORECASE)
            for name in raw_control_candidates
        ),
        "column_dictionary_should_be_analyzed_meaning": "For Cornell Use In Analysis, Can Ignore",
        "column_dictionary_should_exclude_sample_meaning": "Cornell Based Selection of Samples To Consider, Can Ignore",
    }
    write_json(args.out_dir / "input_integrity_summary.json", integrity)

    primary = target_results.loc[target_results["analysis"] == "primary_reproduction"].iloc[0]
    if int(primary["n"]) != EXPECTED_N or int(primary["cases"]) != EXPECTED_CASES:
        raise RuntimeError("Primary cohort did not reproduce expected denominator and case count")
    if int(primary["families"]) != EXPECTED_FAMILIES:
        raise RuntimeError("Primary cohort did not reproduce expected family count")
    if not math.isclose(float(primary["odds_ratio_per_sd"]), EXPECTED_PRIMARY_OR, rel_tol=0, abs_tol=1e-10):
        raise RuntimeError("Primary richness estimate did not reproduce the validated odds ratio")

    runtime_seconds = float(time.perf_counter() - started)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_specification": (
            "docs/research/active/dry_eye_gut_microbiome_scoping_2026/"
            "MICROBIOME_PIPELINE_GAP_ANALYSIS_SPECIFICATION_2026-08-25.md"
        ),
        "primary_reproduction": {
            "n": int(primary["n"]),
            "cases": int(primary["cases"]),
            "families": int(primary["families"]),
            "odds_ratio_per_sd": float(primary["odds_ratio_per_sd"]),
            "ci_low_or": float(primary["ci_low_or"]),
            "ci_high_or": float(primary["ci_high_or"]),
            "p_value": float(primary["p_value"]),
        },
        "permutations": int(args.permutations),
        "community_skipped": bool(args.skip_community),
        "runtime_seconds": runtime_seconds,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "output_files": sorted(path.name for path in args.out_dir.iterdir() if path.is_file()),
        "privacy": "Aggregate outputs only; no PublicID, sample-level feature table or participant-level row was written.",
    }
    write_json(args.out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
