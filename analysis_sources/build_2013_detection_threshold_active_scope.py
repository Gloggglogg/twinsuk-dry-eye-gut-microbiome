from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from run_2013_initial_microbiome_signal_probes import (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    load_asv_counts,
    load_ded_2013,
    load_family_ids,
    load_mapping,
    nearest_stool_pairs,
    sample_read_depths,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'runtime' / 'microbiome_pipeline_gap_analysis_2026_08_25'
ACTIVE_THRESHOLDS = (2, 5, 10)
WITHIN_3Y_DAYS = 365 * 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the publication-scope ASV detection-threshold sensitivity table. "
            "The one-read row is verified as the observed-richness baseline and is not "
            "retained as a separate sensitivity analysis."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_path = args.output_dir / "detection_threshold_richness.csv"
    read_depth_path = args.output_dir / "alpha_read_depth_sensitivities.csv"
    output_path = args.output_dir / "detection_threshold_richness_active_scope.csv"
    manifest_path = args.output_dir / "detection_threshold_richness_active_scope_manifest.json"

    source = pd.read_csv(source_path)
    read_depth = pd.read_csv(read_depth_path)
    source_thresholds = tuple(source["minimum_reads_per_asv"].astype(int))
    if source_thresholds != (1, 2, 5, 10):
        raise RuntimeError(f"Unexpected historical threshold rows: {source_thresholds}")

    baseline = source.loc[source["minimum_reads_per_asv"].eq(1)].iloc[0]
    read_depth_baseline = read_depth.loc[
        read_depth["feature"].eq("hill_q0_observed_richness")
    ].iloc[0]
    duplicate_fields = [
        "beta_per_sd",
        "standard_error",
        "odds_ratio_per_sd",
        "ci_low_or",
        "ci_high_or",
        "p_value",
    ]
    baseline_duplicate = all(
        np.isclose(
            float(baseline[field]),
            float(read_depth_baseline[field]),
            rtol=0,
            atol=1e-15,
        )
        for field in duplicate_fields
    )
    if not baseline_duplicate or float(baseline["correlation_with_observed_richness"]) != 1.0:
        raise RuntimeError(
            "The one-read row did not exactly duplicate read-depth-adjusted observed richness"
        )

    counts = load_asv_counts(args.data_root)
    mapping = load_mapping(args.data_root)
    ded = load_ded_2013(args.data_root)
    family = load_family_ids(args.data_root)
    depths = sample_read_depths(counts)
    linked = nearest_stool_pairs(
        ded,
        mapping,
        family,
        sample_depths=depths,
        tie_rule="highest_read_depth",
    )
    linked = linked.loc[linked["gap_days_abs"] <= WITHIN_3Y_DAYS].copy()
    count_table = counts.set_index("SequencingSpecificName")
    count_table.index = count_table.index.astype(str)
    sample_names = linked["SequencingSpecificName"].astype(str).tolist()
    analysis_counts = count_table.loc[sample_names].to_numpy(dtype=np.int32, copy=False)
    observed_richness = (analysis_counts > 0).sum(axis=1)

    change_rows: list[dict[str, float | int]] = []
    for threshold in ACTIVE_THRESHOLDS:
        threshold_richness = (analysis_counts >= threshold).sum(axis=1)
        detections_removed = observed_richness - threshold_richness
        changed = detections_removed > 0
        change_rows.append(
            {
                "minimum_reads_per_asv": threshold,
                "eligible_profiles_for_comparison": int(len(observed_richness)),
                "profiles_changed_from_observed_richness": int(changed.sum()),
                "profiles_changed_percent": float(100 * changed.mean()),
                "asv_detections_removed_total": int(detections_removed.sum()),
                "asv_detections_removed_mean": float(detections_removed.mean()),
                "asv_detections_removed_median": float(np.median(detections_removed)),
                "asv_detections_removed_maximum": int(detections_removed.max()),
            }
        )

    active = source.loc[
        source["minimum_reads_per_asv"].isin(ACTIVE_THRESHOLDS)
    ].copy()
    active = active.merge(
        pd.DataFrame(change_rows),
        on="minimum_reads_per_asv",
        how="left",
        validate="one_to_one",
    )
    active = active.rename(columns={"q_value": "historical_four_test_q_value"})
    active["q_value"] = multipletests(active["p_value"], method="fdr_bh")[1]
    active["active_scope_reason"] = (
        "Sensitivity threshold; the one-read rule is the observed-richness baseline"
    )
    active.to_csv(output_path, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "active_thresholds": list(ACTIVE_THRESHOLDS),
        "multiplicity_family_size": len(ACTIVE_THRESHOLDS),
        "multiplicity_method": "Benjamini-Hochberg",
        "one_read_row_excluded_reason": (
            "Exactly duplicates read-depth-adjusted observed richness and therefore is not "
            "a distinct sensitivity analysis"
        ),
        "one_read_baseline_duplicate_verified": baseline_duplicate,
        "profile_change_metrics_recomputed_from_read_only_processed_asv_matrix": True,
        "eligible_profiles_for_profile_change_comparison": int(len(observed_richness)),
        "inputs": [
            {"path": str(source_path), "sha256": sha256(source_path)},
            {"path": str(read_depth_path), "sha256": sha256(read_depth_path)},
        ],
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
        "participant_level_output_written": False,
        "external_or_raw_data_written": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
