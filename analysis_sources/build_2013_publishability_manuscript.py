from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from article_evidence_anchors import anchor_generated_tables


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = PROJECT_ROOT / 'runtime' / 'coauthor_review_production_2026_08_14'
AUDIT = PROJECT_ROOT / 'runtime' / 'publishability_audit_2026_08_21'
PACKAGE = Path(__file__).resolve().parents[1] / 'runtime' / 'manuscript_assets'
GENERATED = PACKAGE / "generated"
MANUSCRIPT_FIGURES = PROJECT_ROOT / 'runtime' / 'manuscript_figures_2026_08_29'
DOSE_RESPONSE = PROJECT_ROOT / 'runtime' / 'symptom_dose_response_2026_08_24'
WHS_SENSITIVITY = PROJECT_ROOT / 'runtime' / 'whs_ded_sensitivity_2026_08_24'
MICROBIOME_GAP = PROJECT_ROOT / 'runtime' / 'microbiome_pipeline_gap_analysis_2026_08_25'
TWIN_EXTENSION = PROJECT_ROOT / 'runtime' / 'twin_relatedness_extension_2026_08_28'
ANTIBIOTIC_AUDIT = PROJECT_ROOT / 'runtime' / 'antibiotic_metadata_assumption_2026_08_24'
ANCOM_CORRECTION = PROJECT_ROOT / 'runtime' / 'ancombc2_2013_corrected_2026_09_01'

GENUS_OUTCOME_ORDER = [
    "highly_symptomatic_and",
    "hs_vs_clean_control",
    "either_often_symptom_or",
]
GENUS_OUTCOME_LABELS = {
    "highly_symptomatic_and": "Highly symptomatic dry eye",
    "hs_vs_clean_control": "Highly symptomatic dry eye vs strict controls",
    "either_often_symptom_or": "Frequent dryness or irritation",
}
GENUS_METHOD_ORDER = ["LinDA", "ANCOM-BC2", "ALDEx2"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tex_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_table(path: Path, columns: list[str], rows: list[list[Any]], align: str) -> None:
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(tex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(tex_escape(value) for value in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt_effect(row: pd.Series) -> str:
    return f"{row.odds_ratio_per_sd:.2f} ({row.ci_low_or:.2f}--{row.ci_high_or:.2f})"


def fmt_p(value: float) -> str:
    if value < 0.0001:
        return "<0.0001"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def readable_taxon(value: str) -> str:
    return value.replace("_", " ")


def genus_direction(effect: float) -> str:
    if effect > 0:
        return "higher in cases"
    if effect < 0:
        return "lower in cases"
    return "no difference"


def main() -> int:
    global PACKAGE, GENERATED
    parser = argparse.ArgumentParser(
        description="Build aggregate manuscript assets after validated result checks."
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=PACKAGE,
        help="Manuscript package whose generated directory will receive the aggregate assets.",
    )
    args = parser.parse_args()
    PACKAGE = args.package_dir
    GENERATED = PACKAGE / "generated"
    GENERATED.mkdir(parents=True, exist_ok=True)
    release_source = PRODUCTION / "release_validation.json"
    release = json.loads(release_source.read_text(encoding="utf-8"))
    if not release.get("passed"):
        raise RuntimeError("Validated production release is not marked PASS")

    reproduction_source = AUDIT / "sensitivities" / "headline_reproduction_check.json"
    reproduction = json.loads(reproduction_source.read_text(encoding="utf-8"))
    if not reproduction.get("passed"):
        raise RuntimeError("Publishability audit did not exactly reproduce the headline result")

    sensitivity_summary_source = AUDIT / "sensitivities" / "run_summary.json"
    sensitivity_summary = json.loads(sensitivity_summary_source.read_text(encoding="utf-8"))
    headline = sensitivity_summary["headline"]
    if (headline["n"], headline["cases"], headline["families"]) != (1685, 93, 1125):
        raise RuntimeError("Headline cohort invariant failed")

    inputs: list[Path] = [release_source, reproduction_source, sensitivity_summary_source]

    table1_source = PRODUCTION / "reporting" / "table1_characteristics.tex"
    table1_text = table1_source.read_text(encoding="utf-8")
    table1_text = table1_text.replace("Sex: F", "Female")
    table1_text = table1_text.replace("Sex: M", "Male")
    table1_text = table1_text.replace("Antibiotic metadata: N", "Antibiotic use in prior month: No")
    table1_text = table1_text.replace("Antibiotic metadata: Y", "Antibiotic use in prior month: Yes")
    table1_text = table1_text.replace("Antibiotic metadata: Unknown", "Antibiotic use in prior month: Unknown")
    table1_text = table1_text.replace("No frequent dryness-and-irritation", r"\shortstack{Without highly\\symptomatic dry eye}")
    table1_text = table1_text.replace("Frequent dryness and irritation", r"\shortstack{With highly\\symptomatic dry eye}")
    (GENERATED / "table1_characteristics.tex").write_text(table1_text, encoding="utf-8")
    inputs.append(table1_source)

    alpha_table_source = PRODUCTION / "reporting" / "table2_alpha_diversity.csv"
    alpha_eight_source = WHS_SENSITIVITY / "alpha_eight_multiplicity_sensitivity.csv"
    alpha_table = pd.read_csv(alpha_table_source)
    alpha_eight = pd.read_csv(alpha_eight_source)
    expected_alpha_order = [
        ("either_often_symptom_or", "alpha_richness"),
        ("either_often_symptom_or", "alpha_shannon"),
        ("highly_symptomatic_and", "alpha_richness"),
        ("highly_symptomatic_and", "alpha_shannon"),
        ("hs_vs_clean_control", "alpha_richness"),
        ("hs_vs_clean_control", "alpha_shannon"),
    ]
    alpha_q = {
        (row.outcome_id, row.feature): float(row.q_value_alpha_eight)
        for row in alpha_eight.itertuples(index=False)
    }
    if len(alpha_table) != len(expected_alpha_order) or any(
        key not in alpha_q for key in expected_alpha_order
    ):
        raise RuntimeError("Eight-test alpha-diversity table inputs are incomplete")
    headline_current = dict(headline)
    headline_current.pop("q_value_alpha_six", None)
    headline_current["q_value_alpha_eight"] = alpha_q[
        ("highly_symptomatic_and", "alpha_richness")
    ]
    write_table(
        GENERATED / "table_s1a_alpha_diversity.tex",
        ["Outcome", "Feature", "N (cases)", "OR per SD (95% CI)", "P value", "BH q value"],
        [
            [
                row["Outcome"],
                row["Feature"],
                row["N (cases)"],
                row["OR per SD (95% CI)"],
                row["P value"],
                f"{alpha_q[key]:.4f}",
            ]
            for (_, row), key in zip(alpha_table.iterrows(), expected_alpha_order, strict=True)
        ],
        "lrrrrr",
    )
    inputs.extend([alpha_table_source, alpha_eight_source])
    story_figure_source = MANUSCRIPT_FIGURES / "figure1_richness_story_lean.png"
    story_figure_manifest_source = MANUSCRIPT_FIGURES / "figure1_richness_story_lean_manifest.json"
    if not story_figure_source.exists() or not story_figure_manifest_source.exists():
        raise RuntimeError(
            "Reader-facing richness forest plot is missing; run "
            "build_2013_lean_story_figure.py first"
        )
    shutil.copy2(story_figure_source, GENERATED / "figure1_richness_story.png")
    inputs.extend([story_figure_source, story_figure_manifest_source])

    dose_display = PROJECT_ROOT / "outputs/v47_jelle_display_2026_10_11"
    dose_figure_source = dose_display / "strict_control_dose_response_story.png"
    dose_figure_manifest_source = dose_display / "strict_control_dose_response_story_manifest.json"
    if not dose_figure_source.exists() or not dose_figure_manifest_source.exists():
        raise RuntimeError(
            "Reader-facing symptom-severity figure is missing; run "
            "build_v47_symptom_figure.py first"
        )
    shutil.copy2(dose_figure_source, GENERATED / "figure2_symptom_dose_response.png")
    inputs.extend([dose_figure_source, dose_figure_manifest_source])

    # Supplementary Table S2: severity and WHS-defined dry eye disease.
    ordered_source = DOSE_RESPONSE / "ordered_trend_model.json"
    category_source = DOSE_RESPONSE / "strict_control_category_models.csv"
    dose_groups_source = DOSE_RESPONSE / "dose_score_group_summary.csv"
    whs_source = WHS_SENSITIVITY / "whs_ded_richness_model.json"
    ordered = json.loads(ordered_source.read_text(encoding="utf-8"))
    category = pd.read_csv(category_source)
    dose_groups = pd.read_csv(dose_groups_source)
    whs = json.loads(whs_source.read_text(encoding="utf-8"))
    score_q = alpha_q[("strict_control_score_trend", "alpha_richness")]
    severity_rows: list[list[Any]] = [[
        "Ordered WHS symptom score, 0--6",
        "Mean score difference per richness SD",
        int(ordered["n"]),
        f"{ordered['beta_score_points_per_global_richness_sd']:.3f} ({ordered['ci_low']:.3f} to {ordered['ci_high']:.3f})",
        fmt_p(ordered["p_value"]),
        f"{score_q:.4f}",
    ]]
    for dose_score in range(1, 7):
        model = category.loc[category["dose_score"] == dose_score].iloc[0]
        group = dose_groups.loc[dose_groups["dose_score"] == dose_score].iloc[0]
        n = int(group["participants"] + 525)
        cases = int(group["participants"])
        if bool(model["model_fitted"]):
            severity_rows.append([
                f"Score {dose_score} vs strict controls",
                "OR per richness SD",
                f"{n} ({cases})",
                f"{model['odds_ratio_per_global_richness_sd']:.2f} ({model['ci_low_or']:.2f}--{model['ci_high_or']:.2f})",
                fmt_p(float(model["p_value"])),
                f"{float(model['q_value_bh']):.3f}",
            ])
        else:
            severity_rows.append([
                f"Score {dose_score} vs strict controls",
                "Not estimated",
                f"{n} ({cases})",
                "--",
                "--",
                "--",
            ])
    severity_rows.append([
        "WHS-defined dry eye disease",
        "OR per richness SD",
        f"{int(whs['n'])} ({int(whs['cases'])})",
        f"{whs['odds_ratio_per_sd']:.2f} ({whs['ci_low_or']:.2f}--{whs['ci_high_or']:.2f})",
        fmt_p(whs["p_value"]),
        f"{whs['q_value_alpha_eight']:.3f}",
    ])
    pooled_source = PROJECT_ROOT / "outputs/sparse_tail_pooling_followup_2026_08_24/pooled_tail_models.csv"
    pooled = pd.read_csv(pooled_source)
    for contrast, label in [("scores_5_6_pooled", "Scores 5--6 vs strict controls"),
                            ("scores_4_6_pooled", "Scores 4--6 vs strict controls")]:
        row = pooled.loc[pooled["contrast"].eq(contrast)].iloc[0]
        severity_rows.append([label, "OR per richness SD", f"{int(row['n'])} ({int(row['cases'])})",
                              f"{row['odds_ratio_per_global_richness_sd']:.2f} ({row['ci_low_or']:.2f}--{row['ci_high_or']:.2f})",
                              fmt_p(float(row['p_value'])), f"{row['q_value_bh_two_new_pooled_contrasts']:.4f}"])
    inputs.append(pooled_source)
    write_table(
        GENERATED / "table_s1b_severity_whs.tex",
        ["Outcome or contrast", "Effect measure", "N (outcome-positive)", "Estimate (95% CI)", "P value", "BH q value"],
        severity_rows,
        r">{\raggedright\arraybackslash}p{0.28\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}rrrr",
    )
    inputs.extend([ordered_source, category_source, dose_groups_source, whs_source])

    # Supplementary Table S2: selected sensitivity analyses without duplicating the main result.
    target_source = MICROBIOME_GAP / "target_sequence_and_handling_sensitivities.csv"
    target = pd.read_csv(target_source)
    detection_source = MICROBIOME_GAP / "detection_threshold_richness_active_scope.csv"
    detection = pd.read_csv(detection_source)
    if tuple(detection["minimum_reads_per_asv"].astype(int)) != (2, 5, 10):
        raise RuntimeError("Active detection-threshold scope must contain 2, 5 and 10 reads")
    if not detection["q_value"].map(lambda value: abs(float(value) - 0.009129504156598708) < 1e-12).all():
        raise RuntimeError("Detection-threshold BH q values were not recalculated across three tests")
    read_depth_source = MICROBIOME_GAP / "alpha_read_depth_sensitivities.csv"
    read_depth = pd.read_csv(read_depth_source)
    rarefaction_source = PRODUCTION / "method_strengthening" / "rarefaction_richness_sensitivity.csv"
    rarefaction = pd.read_csv(rarefaction_source)
    antibiotic_source = ANTIBIOTIC_AUDIT / "richness_model_antibiotic_scenarios.csv"
    antibiotic = pd.read_csv(antibiotic_source)
    ibs_source = AUDIT / "confounder_audit" / "canonical_ibs_adjustment_sensitivity.csv"
    ibs = pd.read_csv(ibs_source)
    one_year = pd.Series(sensitivity_summary["one_year"])
    sensitivity_rows: list[list[Any]] = [[
        "Stool within 1 year",
        f"{int(one_year.n)} ({int(one_year.cases)})",
        fmt_effect(one_year),
        fmt_p(one_year.p_value),
    ]]
    for row in sensitivity_summary["timing_interaction"]:
        series = pd.Series(row)
        label = "Stool before questionnaire" if row["timing_group"] == "stool_before_questionnaire" else "Stool after questionnaire"
        sensitivity_rows.append([
            label,
            f"{int(series.participants)} ({int(series.cases)})",
            fmt_effect(series),
            "--",
        ])
    read_depth_row = read_depth.loc[
        read_depth["feature"].eq("hill_q0_observed_richness")
    ].iloc[0]
    sensitivity_rows.append([
        "Additional adjustment for total sequencing reads",
        f"{int(read_depth_row['n'])} ({int(read_depth_row['cases'])})",
        f"{read_depth_row['odds_ratio_per_sd']:.2f} ({read_depth_row['ci_low_or']:.2f}--{read_depth_row['ci_high_or']:.2f})",
        fmt_p(float(read_depth_row["p_value"])),
    ])
    rarefaction_row = rarefaction.loc[
        rarefaction["outcome_id"].eq("highly_symptomatic_and")
        & rarefaction["feature"].eq("alpha_richness_rarefied_expected")
    ].iloc[0]
    sensitivity_rows.append([
        "Expected richness after standardizing libraries to 46,005 reads",
        f"{int(rarefaction_row['n'])} ({int(rarefaction_row['cases'])})",
        f"{rarefaction_row['odds_ratio_per_sd']:.2f} ({rarefaction_row['ci_low_or']:.2f}--{rarefaction_row['ci_high_or']:.2f})",
        fmt_p(float(rarefaction_row["p_value"])),
    ])
    bacterial_only = target.loc[
        target["analysis"].eq("bacterial_asvs_without_organelles")
    ].iloc[0]
    sensitivity_rows.append([
        "Removal of organellar and non-bacterial sequences",
        f"{int(bacterial_only['n'])} ({int(bacterial_only['cases'])})",
        f"{bacterial_only['odds_ratio_per_sd']:.2f} ({bacterial_only['ci_low_or']:.2f}--{bacterial_only['ci_high_or']:.2f})",
        fmt_p(float(bacterial_only["p_value"])),
    ])
    sensitivity_rows.append([
        "Minimum ASV counts of 2, 5 or 10 reads",
        "1685 (93)",
        f"{detection['odds_ratio_per_sd'].min():.2f}--{detection['odds_ratio_per_sd'].max():.2f}",
        f"{detection['p_value'].min():.4f}--{detection['p_value'].max():.4f}",
    ])
    for analysis_id, label in [
        ("primary_plus_collection_route", "Additional collection-route adjustment"),
        ("exclude_documented_short_unmerged_run", "Exclude short-ASV sequencing run"),
    ]:
        row = target.loc[target["analysis"] == analysis_id].iloc[0]
        sensitivity_rows.append([
            label,
            f"{int(row['n'])} ({int(row['cases'])})",
            f"{row['odds_ratio_per_sd']:.2f} ({row['ci_low_or']:.2f}--{row['ci_high_or']:.2f})",
            fmt_p(float(row["p_value"])),
        ])
    unknown_no = antibiotic.loc[antibiotic["scenario"] == "unknown_as_no"].iloc[0]
    sensitivity_rows.append([
        "Unknown antibiotic status recoded as no",
        f"{int(unknown_no['n'])} ({int(unknown_no['cases'])})",
        f"{unknown_no['odds_ratio_per_sd']:.2f} ({unknown_no['ci_low_or']:.2f}--{unknown_no['ci_high_or']:.2f})",
        fmt_p(float(unknown_no["p_value"])),
    ])
    known = antibiotic.loc[antibiotic["scenario"] == "known_only"].iloc[0]
    sensitivity_rows.append([
        "Known antibiotic use only",
        f"{int(known['n'])} ({int(known['cases'])})",
        f"{known['odds_ratio_per_sd']:.2f} ({known['ci_low_or']:.2f}--{known['ci_high_or']:.2f})",
        fmt_p(float(known['p_value'])),
    ])
    for row in ibs.itertuples(index=False):
        label = "IBS-complete sample, primary covariates"
        if "plus_doctor" in row.model:
            label = "Same sample plus doctor-diagnosed IBS"
        sensitivity_rows.append([
            label,
            f"{int(row.n)} ({int(row.cases)})",
            f"{row.odds_ratio_per_sd:.2f} ({row.ci_low_or:.2f}--{row.ci_high_or:.2f})",
            fmt_p(row.p_value),
        ])
    write_table(
        GENERATED / "table_s2_richness_robustness.tex",
        ["Analysis", "N (cases)", "OR per SD (95% CI)", "P value"],
        sensitivity_rows,
        r">{\raggedright\arraybackslash}p{0.47\textwidth}rrr",
    )
    inputs.extend([
        target_source,
        detection_source,
        read_depth_source,
        rarefaction_source,
        antibiotic_source,
        ibs_source,
    ])

    # Supplementary Table S3: non-duplicative ecological and community context.
    hill_source = MICROBIOME_GAP / "hill_diversity_profile.csv"
    distance_source = MICROBIOME_GAP / "community_distance_read_depth_sensitivity.csv"
    community_source = PRODUCTION / "reporting" / "table_s2_community.csv"
    hill = pd.read_csv(hill_source)
    distance = pd.read_csv(distance_source)
    community = pd.read_csv(community_source)
    ecological_rows: list[list[Any]] = []
    inverse_simpson = hill.loc[hill["hill_order"].eq(2)].iloc[0]
    ecological_rows.append([
        "Inverse Simpson diversity (effective number of equally abundant ASVs)",
        f"{int(inverse_simpson['n'])} ({int(inverse_simpson['cases'])})",
        f"OR {inverse_simpson['odds_ratio_per_sd']:.2f} ({inverse_simpson['ci_low_or']:.2f}--{inverse_simpson['ci_high_or']:.2f})",
        fmt_p(float(inverse_simpson["p_value"])),
        "--",
    ])
    primary_community = community.loc[
        community["outcome_id"].eq("highly_symptomatic_and")
    ].iloc[0]
    ecological_rows.append([
        "Overall genus composition, Aitchison distance",
        f"{int(primary_community['n'])} ({int(primary_community['cases'])})",
        f"Partial R-squared {100 * primary_community['partial_r2']:.2f}%",
        fmt_p(float(primary_community["permutation_p_value"])),
        fmt_p(float(primary_community["q_value"])),
    ])
    distance_labels = {
        "genus_presence_jaccard": "Read-depth-adjusted genus presence or absence, Jaccard",
        "genus_relative_abundance_bray_curtis": "Read-depth-adjusted genus relative abundance, Bray--Curtis",
    }
    for row in distance.itertuples(index=False):
        ecological_rows.append([
            distance_labels[row.analysis],
            f"{int(row.n)} ({int(row.cases)})",
            f"Partial R-squared {100 * row.partial_r2:.2f}%",
            fmt_p(row.permutation_p_value),
            fmt_p(row.q_value),
        ])
    write_table(
        GENERATED / "table_s3_ecological_context.tex",
        ["Analysis", "N (cases)", "Association measure", "P value", "BH q"],
        ecological_rows,
        r">{\raggedright\arraybackslash}p{0.44\textwidth}rrrr",
    )
    inputs.extend([hill_source, distance_source, community_source])

    # Supplementary Data S1 and Table S4: complete and concise genus-level reporting.
    linda_source = PRODUCTION / "formal_methods" / "microbiomestat_linda_dry_eye_status_case_results.csv"
    aldex_source = PRODUCTION / "formal_methods" / "aldex2_glm_dry_eye_status_case_results.csv"
    formal_manifest_source = PRODUCTION / "formal_methods" / "run_summary.json"
    ancom_source = PRODUCTION / "ancombc2_summary" / "ancombc2_all_case_results.csv"
    ancom_corrected_strict_source = (
        ANCOM_CORRECTION / "full" / "hs_vs_clean_control" / "ancombc2_case_result.csv"
    )
    ancom_correction_validation_source = ANCOM_CORRECTION / "strict_control_correction_validation.json"
    ancom_manifest_sources = {
        outcome_id: (
            ANCOM_CORRECTION / "full" / outcome_id / "manifest.json"
            if outcome_id == "hs_vs_clean_control"
            else PRODUCTION / "ancombc2" / "full" / outcome_id / "manifest.json"
        )
        for outcome_id in GENUS_OUTCOME_ORDER
    }
    linda = pd.read_csv(linda_source)
    aldex = pd.read_csv(aldex_source)
    ancom = pd.read_csv(ancom_source)
    ancom_corrected_strict = pd.read_csv(ancom_corrected_strict_source)
    ancom_corrected_strict.insert(0, "outcome_id", "hs_vs_clean_control")
    ancom = pd.concat(
        [
            ancom.loc[~ancom["outcome_id"].eq("hs_vs_clean_control")],
            ancom_corrected_strict,
        ],
        ignore_index=True,
    )
    formal_manifest = json.loads(formal_manifest_source.read_text(encoding="utf-8"))
    ancom_correction_validation = json.loads(
        ancom_correction_validation_source.read_text(encoding="utf-8")
    )
    if not ancom_correction_validation.get("passed"):
        raise RuntimeError("Corrected strict-control ANCOM-BC2 output has not passed validation")
    ancom_manifests = {
        outcome_id: json.loads(path.read_text(encoding="utf-8"))
        for outcome_id, path in ancom_manifest_sources.items()
    }
    expected_genus_keys = {
        (outcome_id, taxon)
        for outcome_id in GENUS_OUTCOME_ORDER
        for taxon in linda.loc[linda["outcome_id"].eq(outcome_id), "taxon"]
    }
    if len(expected_genus_keys) != 426:
        raise RuntimeError("LinDA output must contain 142 genera for each of three outcomes")
    for method_name, frame in [("ALDEx2", aldex), ("ANCOM-BC2", ancom)]:
        keys = set(zip(frame["outcome_id"], frame["taxon"], strict=True))
        if keys != expected_genus_keys:
            raise RuntimeError(f"{method_name} genus universe does not match LinDA")

    outcome_sizes = {
        outcome_id: formal_manifest["outcomes"][outcome_id]
        for outcome_id in GENUS_OUTCOME_ORDER
    }
    genus_frames: list[pd.DataFrame] = []

    linda_long = pd.DataFrame({
        "outcome_id": linda["outcome_id"],
        "genus": linda["taxon"],
        "method": "LinDA",
        "family_aware": True,
        "effect_measure": "log2 fold change",
        "effect_estimate": linda["log2FoldChange"],
        "standard_error": linda["lfcSE"],
        "test_statistic": linda["stat"],
        "p_value": linda["pvalue"],
        "q_value": linda["padj"],
        "pseudocount_sensitivity_passed": pd.NA,
        "valid_for_inference": True,
        "analysis_status": "included",
        "fit_note": "Family random intercept included",
    })
    genus_frames.append(linda_long)

    aldex_long = pd.DataFrame({
        "outcome_id": aldex["outcome_id"],
        "genus": aldex["taxon"],
        "method": "ALDEx2",
        "family_aware": False,
        "effect_measure": "centered log-ratio coefficient",
        "effect_estimate": aldex["dry_eye_statusCase:Est"],
        "standard_error": aldex["dry_eye_statusCase:SE"],
        "test_statistic": aldex["dry_eye_statusCase:t.val"],
        "p_value": aldex["dry_eye_statusCase:pval"],
        "q_value": aldex["dry_eye_statusCase:pval.padj"],
        "pseudocount_sensitivity_passed": pd.NA,
        "valid_for_inference": True,
        "analysis_status": "included as a sensitivity analysis",
        "fit_note": "No family term",
    })
    genus_frames.append(aldex_long)

    ancom_status = {
        outcome_id: manifest["status"] == "ok"
        for outcome_id, manifest in ancom_manifests.items()
    }
    ancom_long = pd.DataFrame({
        "outcome_id": ancom["outcome_id"],
        "genus": ancom["taxon"],
        "method": "ANCOM-BC2",
        "family_aware": True,
        "effect_measure": "bias-corrected log abundance coefficient",
        "effect_estimate": ancom["lfc_case"],
        "standard_error": ancom["se_case"],
        "test_statistic": ancom["W_case"],
        "p_value": ancom["p_case"],
        "q_value": ancom["q_case"],
        "pseudocount_sensitivity_passed": ancom["passed_ss_case"],
        "valid_for_inference": ancom["outcome_id"].map(ancom_status),
        "analysis_status": ancom["outcome_id"].map(
            lambda outcome_id: "included"
            if ancom_status[outcome_id]
            else "excluded after model warning"
        ),
        "fit_note": ancom["outcome_id"].map(
            lambda outcome_id: "Family random intercept included"
            if ancom_status[outcome_id]
            else ancom_manifests[outcome_id]["warning"]
        ),
    })
    genus_frames.append(ancom_long)

    genus_results = pd.concat(genus_frames, ignore_index=True)
    genus_results["outcome_label"] = genus_results["outcome_id"].map(GENUS_OUTCOME_LABELS)
    genus_results["participants"] = genus_results["outcome_id"].map(
        lambda outcome_id: outcome_sizes[outcome_id]["samples"]
    )
    genus_results["cases"] = genus_results["outcome_id"].map(
        lambda outcome_id: outcome_sizes[outcome_id]["cases"]
    )
    genus_results["controls"] = genus_results["outcome_id"].map(
        lambda outcome_id: outcome_sizes[outcome_id]["controls"]
    )
    genus_results["direction_case_vs_comparison"] = genus_results["effect_estimate"].map(genus_direction)
    genus_results["q_below_0_05"] = (
        genus_results["valid_for_inference"] & genus_results["q_value"].lt(0.05)
    )
    genus_results["q_below_0_10"] = (
        genus_results["valid_for_inference"] & genus_results["q_value"].lt(0.10)
    )
    outcome_rank = {value: rank for rank, value in enumerate(GENUS_OUTCOME_ORDER)}
    method_rank = {value: rank for rank, value in enumerate(GENUS_METHOD_ORDER)}
    genus_results["_outcome_rank"] = genus_results["outcome_id"].map(outcome_rank)
    genus_results["_method_rank"] = genus_results["method"].map(method_rank)
    genus_results = genus_results.sort_values(
        ["_outcome_rank", "_method_rank", "q_value", "p_value", "genus"]
    ).drop(columns=["_outcome_rank", "_method_rank"])
    genus_results = genus_results[[
        "outcome_id",
        "outcome_label",
        "participants",
        "cases",
        "controls",
        "method",
        "family_aware",
        "genus",
        "effect_measure",
        "effect_estimate",
        "standard_error",
        "test_statistic",
        "p_value",
        "q_value",
        "direction_case_vs_comparison",
        "q_below_0_05",
        "q_below_0_10",
        "pseudocount_sensitivity_passed",
        "valid_for_inference",
        "analysis_status",
        "fit_note",
    ]]
    genus_data_path = GENERATED / "supplementary_data_s1_genus_results.csv"
    genus_results.to_csv(genus_data_path, index=False, float_format="%.12g")

    genus_summary_rows: list[list[Any]] = []
    for outcome_id in GENUS_OUTCOME_ORDER:
        for method in GENUS_METHOD_ORDER:
            subset = genus_results.loc[
                genus_results["outcome_id"].eq(outcome_id)
                & genus_results["method"].eq(method)
            ].copy()
            valid = bool(subset["valid_for_inference"].iloc[0])
            if not valid:
                genus_summary_rows.append([
                    GENUS_OUTCOME_LABELS[outcome_id],
                    method,
                    "Yes",
                    "Excluded after model warning",
                    "Not evaluated",
                ])
                continue
            hits = subset.loc[subset["q_below_0_05"]].sort_values(["q_value", "p_value", "genus"])
            if hits.empty:
                hit_text = "None"
            else:
                hit_text = "; ".join(
                    f"{readable_taxon(row.genus)}, "
                    f"{'lower' if row.effect_estimate < 0 else 'higher'} relative abundance "
                    f"in symptomatic participants (q={row.q_value:.4f})"
                    for row in hits.itertuples(index=False)
                )
            minimum_q = float(subset["q_value"].min())
            lowest = subset.loc[subset["q_value"].le(minimum_q + 1e-12)].sort_values(
                ["p_value", "genus"]
            )
            lowest_text = "; ".join(
                f"{readable_taxon(row.genus)} (q={row.q_value:.4f})"
                for row in lowest.head(2).itertuples(index=False)
            )
            genus_summary_rows.append([
                GENUS_OUTCOME_LABELS[outcome_id],
                method,
                "Yes" if bool(subset["family_aware"].iloc[0]) else "No",
                hit_text,
                lowest_text,
            ])
    write_table(
        GENERATED / "table_s4_genus_results.tex",
        ["Outcome", "Method", "Accounts for relatives", "Genera at q below 0.05", "Smallest q value"],
        genus_summary_rows,
        r">{\raggedright\arraybackslash}p{0.22\textwidth}l>{\centering\arraybackslash}p{0.12\textwidth}>{\raggedright\arraybackslash}p{0.38\textwidth}>{\raggedright\arraybackslash}p{0.23\textwidth}",
    )
    inputs.extend([
        linda_source,
        aldex_source,
        formal_manifest_source,
        ancom_source,
        ancom_corrected_strict_source,
        ancom_correction_validation_source,
        *ancom_manifest_sources.values(),
    ])

    # Supplementary Table S5: all displayed twin estimates generated from aggregate outputs.
    binary_source = TWIN_EXTENSION / "binary_conditional_models.csv"
    score_source = TWIN_EXTENSION / "symptom_score_within_pair_models.csv"
    binary = pd.read_csv(binary_source)
    score = pd.read_csv(score_source)
    twin_rows: list[list[Any]] = []
    for zygosity in ["All", "MZ", "DZ"]:
        row = binary.loc[binary["zygosity"] == zygosity].iloc[0]
        label = "Highly symptomatic dry eye, within-pair comparison" if zygosity == "All" else ""
        twin_rows.append([
            label,
            {"All": "All", "MZ": "Monozygotic", "DZ": "Dizygotic"}[zygosity],
            int(row["pairs"]),
            f"OR {row['odds_ratio_per_sd']:.2f} ({row['ci_low_or']:.2f}--{row['ci_high_or']:.2f})",
            fmt_p(float(row["p_value"])),
        ])
    matched_source = PROJECT_ROOT / "outputs" / "twin_family_signal_2026_09_05" / "matched_score_models.csv"
    matched_score = pd.read_csv(matched_source)
    for zygosity in ["All", "MZ", "DZ"]:
        row = matched_score.loc[(matched_score["group"] == zygosity) & (matched_score["model"] == "within")].iloc[0]
        label = "0--6 symptom score, within-pair difference" if zygosity == "All" else ""
        twin_rows.append([
            label,
            {"All": "All", "MZ": "Monozygotic", "DZ": "Dizygotic"}[zygosity],
            int(row["pairs"]),
            f"Score difference {row['beta']:.3f} ({row['ci_low']:.3f} to {row['ci_high']:.3f})",
            fmt_p(float(row["p_value"])),
        ])
    for zygosity in ["All", "MZ", "DZ"]:
        row = matched_score.loc[(matched_score["group"] == zygosity) & (matched_score["model"] == "matched_population")].iloc[0]
        twin_rows.append([
            "0--6 symptom score, matched population model" if zygosity == "All" else "",
            {"All": "All", "MZ": "Monozygotic", "DZ": "Dizygotic"}[zygosity],
            int(row["pairs"]),
            f"Score difference {row['beta']:.3f} ({row['ci_low']:.3f} to {row['ci_high']:.3f})",
            fmt_p(float(row["p_value"])),
        ])
    for model, label in [("between", "0--6 symptom score, between-pair model"),
                         ("within_minus_between", "Within-minus-between score coefficient")]:
        row = matched_score.loc[(matched_score["group"] == "All") & (matched_score["model"] == model)].iloc[0]
        twin_rows.append([label, "All", int(row["pairs"]),
                          f"Score difference {row['beta']:.3f} ({row['ci_low']:.3f} to {row['ci_high']:.3f})",
                          fmt_p(float(row["p_value"]))])
    write_table(
        GENERATED / "table_s5_twin.tex",
        ["Outcome and model", "Zygosity", "Pairs", "Effect per SD (95% CI)", "P value"],
        twin_rows,
        r">{\raggedright\arraybackslash}p{0.31\textwidth}lrrr",
    )

    supplementary_sources = [
        binary_source, score_source, matched_source,
        matched_source.parent / "bootstrap_summary.json",
        matched_source.parent / "model_diagnostics.json",
        matched_source.parent / "binary_numerical_check" / "adjusted_binary_check.json",
        TWIN_EXTENSION / "binary_zygosity_interaction.csv",
        TWIN_EXTENSION / "symptom_score_within_between_decomposition.csv",
        TWIN_EXTENSION / "symptom_score_within_between_tests.csv",
        TWIN_EXTENSION / "symptom_score_zygosity_interaction.json",
        TWIN_EXTENSION / "twin_pair_structure_by_zygosity.csv",
    ]
    missing_supplementary_sources = [str(path) for path in supplementary_sources if not path.exists()]
    if missing_supplementary_sources:
        raise RuntimeError(
            "Current supplementary source outputs are missing: "
            + ", ".join(missing_supplementary_sources)
        )
    inputs.extend(supplementary_sources)

    obsolete_generated = [
        "figure1_richness_forest.png",
        "missingness.tex",
        "participant_flow.tex",
        "table_s1_new_robustness.tex",
        "table_s2_pair_structure.tex",
        "table_s3_community.tex",
        "table_s4_confounder_audit.tex",
        "table_s5_timing.tex",
        "table_s8_whs_phenotype_sensitivities.tex",
        "table_s9_threshold_ladder.tex",
        "table_s10_ecological_robustness.tex",
        "table_s11_twin_relatedness.tex",
        "table2_alpha_diversity.tex",
        "table_s2_severity_whs.tex",
        "table_s3_richness_robustness.tex",
        "table_s4_ecological_context.tex",
        "table_s5_flow_missingness.tex",
        "table_s6_twin.tex",
        "table_s4_twin.tex",
    ]
    for name in obsolete_generated:
        (GENERATED / name).unlink(missing_ok=True)

    # Keep stable, invisible claim anchors whenever aggregate tables are rebuilt.
    anchor_generated_tables(PACKAGE)

    expected_output_names = [
        "figure1_richness_story.png",
        "figure2_symptom_dose_response.png",
        "table1_characteristics.tex",
        "table_s1a_alpha_diversity.tex",
        "table_s1b_severity_whs.tex",
        "table_s2_richness_robustness.tex",
        "table_s3_ecological_context.tex",
        "table_s4_genus_results.tex",
        "table_s5_twin.tex",
        "supplementary_data_s1_genus_results.csv",
    ]
    outputs = [GENERATED / name for name in expected_output_names]
    # Hugo's full coefficient table is generated by export_v42_model_coefficients.py.
    # Rebuilding manuscript assets must not silently rerun participant-level models.
    outputs += [GENERATED / 'table_s6_model_coefficients.tex',
                PACKAGE / 'supplementary_data_S2_model_coefficients.csv']
    inputs += [PROJECT_ROOT / 'outputs/hugo_model_reporting_2026_09_07/all_model_coefficients.csv',
               PROJECT_ROOT / 'outputs/hugo_model_reporting_2026_09_07/model_reporting_checks.json']
    missing_outputs = [str(path) for path in outputs if not path.exists()]
    if missing_outputs:
        raise RuntimeError("Expected manuscript assets were not generated: " + ", ".join(missing_outputs))
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "validated_release_pass": True,
        "headline_exact_reproduction_pass": True,
        "headline": headline_current,
        "publication_scope_exclusions": [
            "standardized_probability_contrast",
            "one_read_detection_threshold_duplicate",
        ],
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in sorted(set(inputs))],
        "outputs": [{"path": str(path), "sha256": sha256(path)} for path in outputs],
        "participant_level_output_written": False,
        "external_or_raw_data_written": False,
    }
    (GENERATED / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
