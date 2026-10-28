"""Assemble prepared inputs and call the preserved paper functions.

Only the explicit prepare operation writes participant-level tables. Analysis
results contain aggregate estimates and diagnostics. Study assumptions remain
the assumptions of the preserved implementations.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.stats.multitest import multipletests

from .runtime import PACKAGE_ROOT, analysis_module, module_globals, sha256, source_fingerprints


ID_COLUMNS = ["PublicID", "FamilyID", "SequencingSpecificName"]
NUMERIC_COLUMNS = ["How often dry eyes", "How often irritated eyes", "Symptoms past 3 months",
                   "Ever dx DED", "age", "BMI", "gap_days_signed", "gap_days_abs"]
REQUIRED_COHORT = ID_COLUMNS + NUMERIC_COLUMNS + ["SEX", "ACTUAL_ZYGOSITY", "AntibioticsPriorMonth"]
INPUT_FILES = ["cohort.csv", "asv_counts.csv", "taxonomy.csv"]
METHOD_LIMITS = {
    "community_exchangeability": "Equal-size family exchanges and randomized within-pair order assume preservation of the null residual distribution; assessment remains pending.",
    "distance_tests": "Aitchison permutes microbial-profile residuals; Jaccard and Bray-Curtis permute symptom-indicator residuals against a fixed distance matrix.",
    "community_validation": "Synthetic execution and saved reference checks assess computation. Formal null-distribution calibration remains pending.",
    "community_feature_universes": "Aitchison retains genera detected in at least 10 percent of all within-three-year profiles before complete-case selection. Distance tests and R inputs use prevalence in the complete-case primary cohort. Equal feature counts alone cannot establish identical sets.",
    "twin_covariance": "Exploratory continuous-score working covariance model; equal-environment and genetic-relatedness assumptions apply.",
    "study_design": "Observational stool and questionnaire measurements may be separated in time.",
}


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_csv(path: Path, **kwargs):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(set(header)) != len(header):
        raise ValueError(f"Empty or duplicated CSV column names: {path.name}")
    return pd.read_csv(path, **kwargs)


def validate_full_rank(design) -> None:
    """Reject designs the preserved unpivoted basis helper cannot safely handle."""
    values = np.asarray(design, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Covariate design must be a finite two-dimensional matrix")
    if values.shape[0] <= values.shape[1] + 1:
        raise ValueError("Covariate design leaves insufficient residual degrees of freedom")
    if np.linalg.matrix_rank(values) != values.shape[1]:
        raise ValueError("Covariate design has redundant columns; provide a full-column-rank design before community testing")


def load_inputs(input_dir: str | Path):
    """Validate and align explicit prepared tables, recomputing outcomes and alpha features."""
    root = Path(input_dir).resolve()
    for name in INPUT_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Prepared input is missing: {name}")
    cohort = _read_csv(root / "cohort.csv", dtype={c: str for c in ID_COLUMNS})
    missing = sorted(set(REQUIRED_COHORT) - set(cohort.columns))
    if missing:
        raise ValueError("Missing cohort columns: " + ", ".join(missing))
    for column in ID_COLUMNS:
        if cohort[column].isna().any() or cohort[column].str.strip().eq("").any():
            raise ValueError(f"Missing identifiers in {column}")
    if cohort.PublicID.duplicated().any() or cohort.SequencingSpecificName.duplicated().any():
        raise ValueError("Prepared cohort requires unique participant and sample identifiers")
    if cohort.groupby("FamilyID").size().max() > 2:
        raise ValueError("This twin workflow supports singleton and two-person families")
    for column in NUMERIC_COLUMNS:
        cohort[column] = pd.to_numeric(cohort[column], errors="raise")
        if np.isinf(cohort[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite numeric value in {column}")
    for column in ["How often dry eyes", "How often irritated eyes"]:
        if not cohort[column].dropna().isin(range(4)).all():
            raise ValueError(f"Symptom frequency must be 0, 1, 2 or 3: {column}")
    for column in ["Symptoms past 3 months", "Ever dx DED", "ibs_doctor_diagnosed"]:
        if column in cohort:
            cohort[column] = pd.to_numeric(cohort[column], errors="raise")
            if not cohort[column].dropna().isin([0, 1]).all():
                raise ValueError(f"Binary questionnaire value must be zero, one or missing: {column}")
    if not cohort.AntibioticsPriorMonth.isin(["N", "Y", "Unknown"]).all():
        raise ValueError("AntibioticsPriorMonth must use N, Y or the explicit Unknown category")
    if not cohort.SEX.isin(["F", "M"]).all():
        raise ValueError("SEX must use F or M for the preserved model encoding")
    if not cohort.ACTUAL_ZYGOSITY.fillna("Unknown").isin(["MZ", "DZ", "Unknown"]).all():
        raise ValueError("ACTUAL_ZYGOSITY must use MZ, DZ or Unknown")
    if cohort[["gap_days_abs", "gap_days_signed"]].isna().any().any() or not np.allclose(cohort.gap_days_abs, cohort.gap_days_signed.abs()):
        raise ValueError("Absolute gap must equal the absolute signed questionnaire-minus-stool gap")
    counts = _read_csv(root / "asv_counts.csv", dtype={"SequencingSpecificName": str})
    if "SequencingSpecificName" not in counts or counts.SequencingSpecificName.duplicated().any():
        raise ValueError("Count table requires a unique SequencingSpecificName column")
    if set(counts.SequencingSpecificName) != set(cohort.SequencingSpecificName):
        raise ValueError("Count and cohort sample identifiers must match exactly")
    features = [c for c in counts if c != "SequencingSpecificName"]
    if not features:
        raise ValueError("Count table has no amplicon sequence variant columns")
    values = counts[features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Sequence counts must be finite nonnegative integers")
    if not (values.sum(axis=1) > 0).all():
        raise ValueError("Every sample must have a positive sequencing library size")
    counts[features] = values
    taxa = _read_csv(root / "taxonomy.csv", dtype={"asv": str})
    if not {"asv", "Genus_clean"}.issubset(taxa.columns) or taxa.asv.duplicated().any():
        raise ValueError("Taxonomy requires unique asv labels and a Genus_clean column")
    if not set(features).issubset(set(taxa.asv)):
        raise ValueError("At least one count feature has no taxonomy row")
    marker = root / "input_provenance.json"
    provenance = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
    kind = "synthetic" if provenance.get("data_kind") == "synthetic" else "provided_prepared_inputs"
    if kind == "synthetic":
        if any(not cohort[c].str.startswith("SYN_").all() for c in ID_COLUMNS) or any(not c.startswith("SYN_") for c in features):
            raise ValueError("Synthetic provenance requires visibly synthetic participant, family, sample and feature identifiers")
    initial = analysis_module("run_2013_initial_microbiome_signal_probes")
    cohort = initial.derive_symptom_outcomes(cohort)
    alpha, _ = initial.compute_alpha_and_genus(counts, taxa)
    cohort = cohort.drop(columns=[c for c in ["alpha_richness", "alpha_shannon", "alpha_total_reads"] if c in cohort])
    cohort = cohort.merge(alpha, on="SequencingSpecificName", validate="one_to_one")
    # Preserve source count-table order for the formal genus and seeded distance
    # procedures. Each model aligns its cohort explicitly through sample labels.
    counts = counts.set_index("SequencingSpecificName")
    counts.index.name = None
    return cohort, counts, taxa, {"data_kind": kind, "input_sha256": {n: sha256(root / n) for n in INPUT_FILES}}


def prepare_inputs(data_root: str | Path, output_dir: str | Path) -> dict:
    """Explicitly prepare private inputs using original loaders and pairing rules."""
    source, target = Path(data_root).resolve(), Path(output_dir).resolve()
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Prepared output must be separate from the read-only raw source tree")
    if target.is_relative_to(PACKAGE_ROOT):
        raise ValueError("Private prepared inputs must be written outside the public release folder")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Private prepared destination must be empty")
    initial = analysis_module("run_2013_initial_microbiome_signal_probes")
    ded, mapping, family = initial.load_ded_2013(source), initial.load_mapping(source), initial.load_family_ids(source)
    counts, taxa = initial.load_asv_counts(source), initial.load_taxa(source)
    pairs = initial.nearest_stool_pairs(ded, mapping, family, sample_depths=initial.sample_read_depths(counts), tie_rule="highest_read_depth")
    alpha, _ = initial.compute_alpha_and_genus(counts, taxa)
    cohort = pairs.merge(alpha, on="SequencingSpecificName", how="inner", validate="one_to_one")
    cohort = cohort.loc[cohort.gap_days_abs <= 365 * 3].copy()
    if cohort.empty:
        raise ValueError("No matched stool/questionnaire records within three years")
    columns = REQUIRED_COHORT + [c for c in ["ResponseDate", "microbiome_date", "WHS DED", "ibs_doctor_diagnosed"] if c in cohort]
    target.mkdir(parents=True, exist_ok=True)
    cohort[columns].to_csv(target / "cohort.csv", index=False)
    counts.loc[counts.SequencingSpecificName.isin(cohort.SequencingSpecificName)].to_csv(target / "asv_counts.csv", index=False)
    taxa.to_csv(target / "taxonomy.csv", index=False)
    manifest = {"data_kind": "private_prepared_inputs", "contains_participant_level_data": True,
                "publication_permission": "Private local preparation; exclude these inputs from public releases.",
                "participants": len(cohort), "families": cohort.FamilyID.nunique(),
                "pairing": "Nearest physical stool; highest read depth resolves tied sequencing profiles.",
                "interval_limit_days": 1095, "raw_sources_modified": False,
                "preparation_functions": ["load_ded_2013", "load_mapping", "load_family_ids", "load_asv_counts", "load_taxa", "nearest_stool_pairs", "compute_alpha_and_genus"],
                "input_sha256": {n: sha256(target / n) for n in INPUT_FILES}}
    _write_json(target / "input_provenance.json", manifest)
    return manifest


def prepare_genus_inputs(input_dir: str | Path, output_dir: str | Path) -> dict:
    """Export separate private R-package inputs through the preserved builder."""
    root, target = Path(input_dir).resolve(), Path(output_dir).resolve()
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("Private genus output must be separate from its immutable input tree")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Private genus destination must be empty")
    base, wide, taxa, provenance = load_inputs(root)
    if provenance["data_kind"] != "synthetic" and target.is_relative_to(PACKAGE_ROOT):
        raise ValueError("Private genus inputs must be written outside the public release folder")
    strengthening = analysis_module("run_2013_method_strengthening_batch")
    builder = analysis_module("build_2013_ancombc2_inputs")
    publication = analysis_module("run_2013_publishability_sensitivities")
    covariates = strengthening.primary_covariates(base)
    within = base.loc[base.gap_days_abs <= 1095].copy()
    reference = publication.primary_complete_cases(within, covariates)
    genus = builder.clean_counts(strengthening.collapse_genus_counts(wide, taxa))
    genus.index.name = None
    prevalence = (genus.loc[reference.SequencingSpecificName] > 0).mean()
    fixed = sorted(prevalence[prevalence >= .10].index.astype(str))
    if not fixed:
        raise ValueError("The primary-cohort 10 percent prevalence rule retained no genera")
    # Pseudonyms preserve alignment and family grouping. Count and metadata rows
    # remain private even after replacing the input identifier strings.
    samples = {value: f"R_SAMPLE{i+1:06d}" for i, value in enumerate(sorted(base.SequencingSpecificName))}
    families = {value: f"R_FAMILY{i+1:06d}" for i, value in enumerate(sorted(base.FamilyID.unique()))}
    base = base.copy()
    base["SequencingSpecificName"] = base.SequencingSpecificName.map(samples)
    base["FamilyID"] = base.FamilyID.map(families)
    genus = genus.rename(index=samples)
    target.mkdir(parents=True, exist_ok=True)
    outcomes = [builder.build_outcome_inputs(base, genus, outcome, target, .10, fixed, prevalence) for outcome in builder.PRIMARY_OUTCOMES]
    for outcome in outcomes:
        for key in ["count_file", "metadata_file"]:
            outcome[key] = Path(outcome[key]).relative_to(target).as_posix()
    result = {**provenance, "contains_participant_level_data": True,
              "publication_permission": "Keep R count and metadata inputs private, including pseudonymized family groups.",
              "fixed_universe_taxa": len(fixed), "reference_participants": len(reference), "prevalence_cut": .10,
              "prevalence_reference": "Complete-case primary highly symptomatic dry-eye cohort within three years",
              "fixed_universe_across_outcomes": True, "outcomes": outcomes,
              "paper_reference_taxa_count": 142,
              "taxa_count_note": "The historical study retained 142 genera; synthetic or other prepared inputs determine their own count using the identical prevalence rule.",
              "r_models_executed": False}
    _write_json(target / "input_manifest.json", result)
    return _json_value(result)


def _require_result(result, label):
    if not result or "p_value" not in result or not np.isfinite(float(result["p_value"])):
        raise ValueError(f"Model has insufficient information or a nonfinite estimate: {label}")
    if result.get("converged", True) is False:
        raise ValueError(f"Model failed to reach a finite supported fit: {label}")
    for key in ["beta_per_sd", "standard_error", "odds_ratio_per_sd"]:
        if key in result and not np.isfinite(result[key]):
            raise ValueError(f"Model returned a nonfinite {key}: {label}")
    return result


def _case_minimum(data, outcome, minimum=20):
    chosen = data.dropna(subset=[outcome])
    for value in [0, 1]:
        subset = chosen[chosen[outcome].eq(value)]
        if len(subset) < minimum or subset.FamilyID.nunique() < minimum:
            raise ValueError(f"{outcome} requires at least {minimum} participants and families in each outcome group")


def run_workflow(input_dir: str | Path, output_dir: str | Path, *, community_permutations: int = 0,
                 twin_bootstrap: int = 0, allow_heavy: bool = False) -> dict:
    """Recompute paper analyses from explicit prepared inputs; return aggregate provenance."""
    if community_permutations < 0 or twin_bootstrap < 0 or twin_bootstrap == 1:
        raise ValueError("Permutation count must be nonnegative; bootstrap count must be zero or at least two")
    if (community_permutations > 99 or twin_bootstrap > 0) and not allow_heavy:
        raise ValueError("Production permutations and repeated twin fits require explicit allow_heavy=True")
    root, target = Path(input_dir).resolve(), Path(output_dir).resolve()
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("Analysis output must be separate from its input directory")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Analysis output destination must be empty to preserve previous results")
    base, wide, taxa, provenance = load_inputs(root)
    locked = analysis_module("run_2013_locked_primary_16s_analysis")
    strengthening = analysis_module("run_2013_method_strengthening_batch")
    publication = analysis_module("run_2013_publishability_sensitivities")
    dose = analysis_module("run_2013_symptom_dose_response")
    pooling = analysis_module("run_2013_sparse_tail_pooling_followup")
    within = base.loc[base.gap_days_abs <= 1095].copy()
    covariates = strengthening.primary_covariates(within)
    primary = publication.primary_complete_cases(within, covariates)
    if len(primary) < 100 or primary.FamilyID.nunique() < 20:
        raise ValueError("Primary cohort requires at least 100 complete participants and 20 families")
    for outcome in ["either_often_symptom", "highly_symptomatic_derived", "hs_vs_clean_control"]:
        _case_minimum(primary, outcome)
    if community_permutations:
        validate_full_rank(strengthening.design_matrix(primary, covariates))
    target.mkdir(parents=True, exist_ok=True)
    files = []
    origin = "recomputed_from_synthetic_inputs" if provenance["data_kind"] == "synthetic" else "recomputed_from_provided_inputs"
    def table(name, frame):
        frame = frame.copy()
        forbidden = set(ID_COLUMNS + ["family_code", "sample_id", "public_id", "ParticipantPublicID"])
        if set(frame.columns) & forbidden:
            raise ValueError("Participant or family identifiers cannot enter analysis outputs")
        frame["data_kind"], frame["result_origin"] = provenance["data_kind"], origin
        frame.to_csv(target / name, index=False)
        if name not in files:
            files.append(name)
    def record(name, value):
        _write_json(target / name, {"data_kind": provenance["data_kind"], "result_origin": origin, **value})
        if name not in files:
            files.append(name)

    alpha = publication.reproduce_alpha_family(within, covariates)
    for row in alpha.to_dict("records"):
        _require_result(row, row["outcome_id"])
    alpha["branch_id"], alpha["covariate_set"], alpha["outcome_type"] = "within_3y", "primary_core", "binary"
    alpha = alpha.rename(columns={"q_value_alpha_six": "q_value_alpha_six_historical"})
    table("primary_alpha_models.csv", alpha)
    score = pd.to_numeric(within.symptom_sum_0_6)
    eligible = within.clean_control_no_ded_no_symptoms.eq(1) | score.isin(range(1, 7))
    ordered = within.loc[eligible].copy()
    ordered["dose_score"] = np.where(ordered.clean_control_no_ded_no_symptoms.eq(1), 0, ordered.symptom_sum_0_6).astype(int)
    design = locked.build_x(ordered, "alpha_richness", covariates)
    ordered = ordered.loc[design.notna().all(axis=1)].copy()
    initial = analysis_module("run_2013_initial_microbiome_signal_probes")
    ordered["richness_z_global"] = initial.standardize(ordered.alpha_richness)
    trend = _require_result(dose.fit_ordered_linear_trend(ordered, covariates), "strict-control score trend")
    record("ordered_trend_model.json", trend)
    table("score_group_counts.csv", dose.group_counts(ordered))
    categories = []
    for score_value in range(1, 7):
        group = ordered[ordered.dose_score.eq(score_value)]
        if len(group) < dose.MIN_GROUP_PARTICIPANTS or group.FamilyID.nunique() < dose.MIN_GROUP_FAMILIES:
            categories.append({"dose_score": score_value, "cases": len(group), "case_families": group.FamilyID.nunique(), "model_fitted": False, "reason": "below participant or family minimum"})
        else:
            categories.append({**_require_result(dose.fit_binary_fixed_richness(ordered, score_value, covariates), f"score {score_value}"), "model_fitted": True})
    category_frame = pd.DataFrame(categories)
    mask = category_frame.model_fitted
    category_frame["q_value_bh"] = np.nan
    if mask.any():
        category_frame.loc[mask, "q_value_bh"] = multipletests(category_frame.loc[mask, "p_value"], method="fdr_bh")[1]
    table("strict_control_category_models.csv", category_frame)
    pooled = pd.DataFrame([pooling.fit_contrast(ordered, values, label, covariates) for values, label in [((4,), "score_4"), ((5, 6), "scores_5_6_pooled"), ((4, 5, 6), "scores_4_6_pooled")]])
    mask = pooled.contrast.ne("score_4") & pooled.model_fitted
    pooled["q_value_bh_two_new_pooled_contrasts"] = np.nan
    if mask.any():
        pooled.loc[mask, "q_value_bh_two_new_pooled_contrasts"] = multipletests(pooled.loc[mask, "p_value"], method="fdr_bh")[1]
    table("pooled_tail_models.csv", pooled)
    whs = analysis_module("run_whs_ded_richness_sensitivity")
    whs_result = _require_result(locked.binary_association(within, "whs_ded_derived", "alpha_richness", covariates), "diagnosis-inclusive dry eye")
    with module_globals(whs, REFERENCE_MATRIX=target / "primary_alpha_models.csv", DOSE_RESULT=target / "ordered_trend_model.json"):
        eight = whs.alpha_eight_sensitivity(whs_result["p_value"])
    alpha = alpha.merge(eight[["outcome_id", "feature", "q_value_alpha_eight"]], on=["outcome_id", "feature"], how="left", validate="one_to_one")
    table("primary_alpha_models.csv", alpha)
    trend["q_value_alpha_eight"] = float(eight.loc[eight.outcome_id.eq("strict_control_score_trend"), "q_value_alpha_eight"].iloc[0])
    record("ordered_trend_model.json", trend)
    whs_result["q_value_alpha_eight"] = float(eight.loc[eight.outcome_id.eq("whs_ded"), "q_value_alpha_eight"].iloc[0])
    record("whs_ded_richness_model.json", whs_result)
    table("alpha_eight_multiplicity_sensitivity.csv", eight)
    antibiotic = analysis_module("audit_antibiotic_metadata_assumption")
    table("antibiotic_sensitivity.csv", pd.DataFrame([antibiotic.fit_scenario(primary, covariates, scenario) for scenario in ["three_level_primary", "unknown_as_no", "antibiotic_omitted", "known_only"]]))
    short = within.loc[within.gap_days_abs <= 365]
    short_fit = locked.binary_association(short, "highly_symptomatic_derived", "alpha_richness", covariates)
    table("one_year_sensitivity.csv", pd.DataFrame([{"analysis": "within_one_year", "model_fitted": bool(short_fit), **short_fit}]))
    timing, timing_diagnostics = publication.timing_interaction_analysis(primary, covariates)
    table("timing_interaction.csv", timing)
    table("timing_diagnostics.csv", timing_diagnostics)
    technical = []
    for threshold_value in [1, 2, 5, 10]:
        chosen = within.copy()
        feature = f"richness_minimum_{threshold_value}_reads"
        chosen[feature] = (wide.loc[chosen.SequencingSpecificName] >= threshold_value).sum(axis=1).to_numpy()
        result = locked.binary_association(chosen, "highly_symptomatic_derived", feature, covariates + ["alpha_total_reads"])
        technical.append({"minimum_reads_per_variant": threshold_value, "feature": feature, **_require_result(result, feature)})
    technical = pd.DataFrame(technical)
    technical["q_value_bh_four_depth_adjusted_checks"] = multipletests(technical.p_value, method="fdr_bh")[1]
    table("read_depth_sensitivity.csv", technical)
    detection_scope = analysis_module("build_2013_detection_threshold_active_scope")
    active = technical.loc[technical.minimum_reads_per_variant.isin(detection_scope.ACTIVE_THRESHOLDS)].copy()
    active = active.rename(columns={"q_value_bh_four_depth_adjusted_checks": "historical_four_test_q_value"})
    active["q_value"] = multipletests(active.p_value, method="fdr_bh")[1]
    active["multiplicity_family_size"] = len(detection_scope.ACTIVE_THRESHOLDS)
    table("detection_threshold_active_scope.csv", active)
    rare, rare_info = strengthening.run_rarefaction_sensitivity(within, wide)
    table("rarefaction_sensitivity.csv", rare)
    record("rarefaction_settings.json", rare_info)
    ibs_status = "skipped: prepared doctor-diagnosed irritable bowel syndrome column unavailable"
    if "ibs_doctor_diagnosed" in primary:
        same = primary.loc[primary.ibs_doctor_diagnosed.notna()].copy()
        try:
            _case_minimum(same, "highly_symptomatic_derived")
        except ValueError:
            ibs_status = "skipped: too few participants or families with usable prepared doctor-diagnosis answers"
        else:
            ibs_rows = []
            for name, extras in [("same_sample_primary_core", []), ("same_sample_primary_core_plus_doctor_diagnosed_ibs", ["ibs_doctor_diagnosed"])]:
                result = locked.binary_association(same, "highly_symptomatic_derived", "alpha_richness", covariates + extras)
                ibs_rows.append({"model": name, **_require_result(result, name)})
            table("ibs_sensitivity.csv", pd.DataFrame(ibs_rows))
            ibs_status = "recomputed in the same participants with available prepared doctor-diagnosis answers"
    twin_status = _run_twins(primary, twin_bootstrap, table, record)
    if community_permutations:
        _run_community(within, primary, wide, taxa, covariates, community_permutations, table)
    summary = {**provenance, "result_origin": origin, "participants": len(primary), "families": primary.FamilyID.nunique(),
               "cases": int(primary.highly_symptomatic_derived.sum()), "covariates": covariates,
               "community_permutations": community_permutations, "twin_bootstrap_requested": twin_bootstrap,
               "ibs_status": ibs_status, "twin_status": twin_status,
               "r_methods_status": "Optional native R scripts provided separately; no R model executed by this Python workflow.",
               "frozen_reference_results_used": False, "participant_level_output_written": False,
               "method_limits": METHOD_LIMITS, "output_files": files + ["run_provenance.json"],
               "source_fingerprints": source_fingerprints(),
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}}
    record("run_provenance.json", summary)
    return _json_value(summary)


def _run_twins(primary, bootstrap, table, record):
    related = analysis_module("run_2013_twin_relatedness_extension")
    twin = analysis_module("run_2013_twin_family_signal_revision")
    cohort = primary.copy()
    cohort["pair_zygosity"] = related.assign_pair_zygosity(cohort)
    cohort["richness_z"] = (cohort.alpha_richness - cohort.alpha_richness.mean()) / cohort.alpha_richness.std(ddof=1)
    pairs = related.two_person_pairs(cohort)
    if pairs.FamilyID.nunique() < 20:
        return "skipped: fewer than twenty complete twin pairs"
    discordant = int(pairs.groupby("FamilyID").highly_symptomatic_derived.sum().eq(1).sum())
    # The source summary asserts the historical count. Adapt that check to the
    # explicit new input, while keeping the original summary function intact.
    with module_globals(related, EXPECTED_BINARY_DISCORDANT_PAIRS=discordant):
        table("twin_pair_structure.csv", related.summarize_pair_structure(pairs))
    predictors = related.encoded_predictors(cohort)
    prepared = predictors.loc[pairs.index].reindex(columns=twin.PREDICTORS, fill_value=0).copy()
    prepared["family_code"] = pd.factorize(pairs.FamilyID, sort=True)[0]
    prepared["zygosity"] = pairs.pair_zygosity.to_numpy()
    prepared["score"] = pairs.symptom_sum_0_6.to_numpy()
    prepared["case"] = pairs.highly_symptomatic_derived.astype(int).to_numpy()
    rows, binary, adjusted = [], [], []
    for group in ["All", "MZ", "DZ"]:
        chosen = prepared if group == "All" else prepared.loc[prepared.zygosity.eq(group)]
        raw = pairs if group == "All" else pairs.loc[pairs.pair_zygosity.eq(group)]
        if chosen.family_code.nunique() < 10:
            continue
        rows.extend(twin.score_models(chosen, group))
        binary.append(related.fit_conditional_logit(raw, group))
        try:
            # Preserve the original sparse-category diagnostic and its validity flag.
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                diagnostic = twin.binary_models(chosen, group)
            diagnostic["runtime_warnings"] = [str(item.message) for item in captured]
            adjusted.append(diagnostic)
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            adjusted.append({"group": group, "valid": False, "status": "not estimable with these prepared predictors", "diagnostic": type(exc).__name__})
    table("twin_matched_score_models.csv", pd.DataFrame(rows))
    table("twin_binary_conditional_models.csv", pd.DataFrame(binary))
    record("twin_adjusted_binary_diagnostics.json", {"models": adjusted})
    known = prepared.loc[prepared.zygosity.isin(["MZ", "DZ"])]
    counts = known.groupby("zygosity").family_code.nunique().to_dict()
    if any(counts.get(group, 0) < 20 for group in ["MZ", "DZ"]):
        return "matched models recomputed; covariance model skipped because one twin type has fewer than twenty pairs"
    arrays, residual_info = twin.paired_arrays(known)
    fit = twin.fit_covariance(twin.moments(arrays), np.random.default_rng(twin.SEED), starts=12)
    if not fit["valid"]:
        raise ValueError("Twin covariance fit failed its gradient acceptance check")
    record("twin_covariance_diagnostics.json", {"known_zygosity_pairs": counts, "residual_info": residual_info, "covariance_model": fit,
                                               "percentage_partition_reportable": False, "interpretation": "Exploratory covariance components; no percentage partition reported."})
    if bootstrap:
        # Explicit opt-in only. Serial execution remains portable across operating systems.
        samples = pd.DataFrame([twin.bootstrap_one((i, prepared, fit["theta"])) for i in range(bootstrap)])
        table("twin_aggregate_bootstrap_estimates.csv", samples)
        valid = samples.loc[samples.valid]
        record("twin_bootstrap_summary.json", {"bootstrap_requested": bootstrap, "bootstrap_valid": len(valid),
            "components": {name: {"estimate": float(sum(np.asarray(x)[0, 1] for x in fit["matrices"].values())) if name == "total" else float(np.asarray(fit["matrices"][name])[0, 1]),
                                     "ci_low": valid[name].quantile(.025), "ci_high": valid[name].quantile(.975)} for name in ["A", "C", "E", "total"]},
            "percentage_partition_reportable": False})
    return "matched models and exploratory covariance point estimates recomputed; bootstrap opt-in recorded separately"


def _run_community(within, primary, wide, taxa, covariates, permutations, table):
    strengthening = analysis_module("run_2013_method_strengthening_batch")
    gap = analysis_module("run_2013_microbiome_pipeline_gap_analysis")
    all_genera = strengthening.collapse_genus_counts(wide, taxa)
    # Preserve the original Aitchison outer function's prevalence population.
    aitchison_keep = (all_genera.loc[within.SequencingSpecificName] > 0).mean().ge(.10)
    aitchison_genera = all_genera.loc[:, aitchison_keep]
    # Distance tests follow the formal input builder's name cleaning, source
    # count-table row order and separate complete-primary prevalence rule.
    builder = analysis_module("build_2013_ancombc2_inputs")
    clean_genera = builder.clean_counts(all_genera)
    retained = (clean_genera.loc[primary.SequencingSpecificName] > 0).mean().ge(.10)
    genera = clean_genera.loc[:, retained]
    selected_ids = genera.index[genera.index.isin(primary.SequencingSpecificName)]
    counts = genera.loc[selected_ids].to_numpy(dtype=float)
    if counts.shape[1] < 2 or not (counts.sum(axis=1) > 0).all():
        raise ValueError("Community tests require at least two retained genera and positive assigned counts")
    branch = within.set_index("SequencingSpecificName")
    # The source dispersion function inserts this name as a column; an unnamed
    # index avoids pandas treating its sort key as both a column and index level.
    branch.index.name = None
    clr = np.log(aitchison_genera + .5)
    clr = clr.sub(clr.mean(axis=1), axis=0)
    aitchison = []
    for outcome_id, outcome in strengthening.PRIMARY_OUTCOMES.items():
        chosen = within.dropna(subset=[outcome, "FamilyID", *covariates])
        validate_full_rank(strengthening.design_matrix(chosen, covariates))
        result = strengthening.permanova_freedman_lane(branch, clr, outcome, covariates, permutations, seed=20260814)
        result.update(strengthening.one_per_family_dispersion(branch, clr, outcome, covariates, permutations, seed=20260815))
        if not np.isfinite([result["pseudo_f"], result["permutation_p_value"], result["dispersion_p_value"]]).all():
            raise ValueError("Aitchison community test returned a nonfinite result")
        aitchison.append({"outcome_id": outcome_id,
            "feature_universe": "10_percent_prevalence_all_within_three_year_profiles",
            "feature_universe_sha256": hashlib.sha256("\n".join(sorted(aitchison_genera.columns)).encode()).hexdigest(),
            "prevalence_reference_profiles": len(within), **result})
    aitchison = pd.DataFrame(aitchison)
    aitchison["q_value"] = multipletests(aitchison.permutation_p_value, method="fdr_bh")[1]
    table("community_aitchison.csv", aitchison)
    metadata = primary.set_index("SequencingSpecificName", drop=False).loc[selected_ids].copy().reset_index(drop=True)
    metadata["sample_id"] = metadata.SequencingSpecificName
    metadata["total_reads"] = metadata.alpha_total_reads
    rows = []
    for metric, profiles, seed in [("jaccard", counts > 0, 20260825), ("braycurtis", counts / counts.sum(axis=1, keepdims=True), 20260827)]:
        for adjusted in [False, True]:
            covs = covariates + (["total_reads"] if adjusted else [])
            validate_full_rank(strengthening.design_matrix(metadata, covs))
            result = gap.distance_partial_test(metadata, profiles, metric, permutations, seed + (200000 if adjusted else 0), covariates=covs)
            rows.append({"analysis": metric, "read_depth_adjusted": adjusted,
                "feature_universe": "10_percent_prevalence_complete_case_primary_cohort",
                "feature_universe_sha256": hashlib.sha256("\n".join(sorted(genera.columns)).encode()).hexdigest(),
                "prevalence_reference_profiles": len(primary), **result})
    distances = pd.DataFrame(rows)
    distances["q_value_bh_two_distances"] = np.nan
    for _, group in distances.groupby("read_depth_adjusted"):
        distances.loc[group.index, "q_value_bh_two_distances"] = multipletests(group.permutation_p_value, method="fdr_bh")[1]
    table("community_distance_tests.csv", distances)
