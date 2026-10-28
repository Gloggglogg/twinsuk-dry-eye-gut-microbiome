from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_2013_initial_microbiome_signal_probes import (
    DED_2013,
    MICROBIOME_DIR,
    SPINE,
    DEFAULT_DATA_ROOT,
    load_asv_counts,
    load_ded_2013,
    load_family_ids,
    load_mapping,
    load_taxa,
    nearest_stool_pairs,
    sample_read_depths,
)
from run_2013_method_strengthening_batch import collapse_genus_counts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION_COMMANDS = Path(__file__).resolve().parents[1] / 'runtime' / 'documentation' / 'COAUTHOR_REVIEW_REPRODUCTION_COMMANDS_2026-08-14.md'
DEFAULT_PRODUCTION_DIR = PROJECT_ROOT / 'runtime' / 'coauthor_review_production_2026_08_14'
DEFAULT_OUT_DIR = DEFAULT_PRODUCTION_DIR / "reporting"
PRIMARY_COVARIATES = ["age", "BMI", "AntibioticsPriorMonth", "gap_days_abs", "SEX"]
OUTCOME_LABELS = {
    "either_often_symptom_or": "Dryness or irritation reported at least often",
    "highly_symptomatic_and": "Dryness and irritation reported at least often",
    "hs_vs_clean_control": "Frequent dryness and irritation vs strict asymptomatic comparison",
}
FEATURE_LABELS = {
    "alpha_richness": "Observed ASV richness",
    "alpha_shannon": "Shannon diversity",
    "alpha_richness_rarefied_expected": "Expected rarefied ASV richness",
}


def write_table(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")


def latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_")]:
        text = text.replace(old, new)
    return text


def write_latex_tabular(frame: pd.DataFrame, path: Path, alignment: str | None = None) -> None:
    align = alignment or ("l" + "r" * (len(frame.columns) - 1))
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(latex_escape(col) for col in frame.columns) + r" \\")
    lines.append("\\midrule")
    for _, row in frame.iterrows():
        lines.append(" & ".join(latex_escape(value) for value in row.tolist()) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_p(value: float) -> str:
    if pd.isna(value):
        return "--"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def mean_sd(series: pd.Series, digits: int = 1) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return f"{values.mean():.{digits}f} ({values.std():.{digits}f})"


def median_iqr(series: pd.Series, digits: int = 1) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return f"{values.median():.{digits}f} [{values.quantile(.25):.{digits}f}, {values.quantile(.75):.{digits}f}]"


def n_percent(mask: pd.Series, denominator: int) -> str:
    count = int(mask.sum())
    return f"{count} ({100 * count / denominator:.1f}%)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def build_analysis_data(data_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    ded = load_ded_2013(data_root)
    mapping = load_mapping(data_root)
    family = load_family_ids(data_root)
    counts = load_asv_counts(data_root)
    taxa = load_taxa(data_root)
    depths = sample_read_depths(counts)
    pairs = nearest_stool_pairs(
        ded,
        mapping,
        family,
        sample_depths=depths,
        tie_rule="highest_read_depth",
    )
    asv_cols = [column for column in counts.columns if column != "SequencingSpecificName"]
    wide = counts.set_index("SequencingSpecificName")[asv_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    totals = wide.sum(axis=1)
    relative = wide.div(totals.replace(0, np.nan), axis=0).fillna(0)
    alpha = pd.DataFrame(
        {
            "SequencingSpecificName": wide.index.astype(str),
            "alpha_richness": (wide > 0).sum(axis=1).to_numpy(),
            "alpha_shannon": (-(relative.where(relative > 0) * np.log(relative.where(relative > 0))).sum(axis=1)).to_numpy(),
            "alpha_total_reads": totals.to_numpy(),
        }
    )
    base = pairs.merge(alpha, on="SequencingSpecificName", how="inner")
    eligible_mapping = mapping[(mapping["ShouldExcludeSample"] == 0) & mapping["SequencingSpecificName"].isin(wide.index)]
    assigned_genus_counts = collapse_genus_counts(wide, taxa)
    audit = {
        "questionnaire_rows": int(len(ded)),
        "questionnaire_participants": int(ded["PublicID"].nunique()),
        "eligible_nonexcluded_sequencing_profiles": int(len(eligible_mapping)),
        "eligible_nonexcluded_microbiome_participants": int(eligible_mapping["PublicID"].nunique()),
        "questionnaire_participants_with_microbiome_profile": int(base["PublicID"].nunique()),
        "within_3y_participants": int((base["gap_days_abs"] <= 1095).sum()),
        "tied_nearest_questionnaires": int((base["nearest_tie_count"] > 1).sum()),
        "total_asvs": int(wide.shape[1]),
        "assigned_genera_before_prevalence_filter": int(assigned_genus_counts.shape[1]),
        "assigned_genus_read_fraction": float(assigned_genus_counts.to_numpy().sum() / wide.to_numpy().sum()),
    }
    return base, audit


def build_flow(base: pd.DataFrame, audit: dict[str, Any]) -> pd.DataFrame:
    within = base[base["gap_days_abs"] <= 1095].copy()
    complete = within.dropna(subset=["highly_symptomatic_derived", *PRIMARY_COVARIATES])
    clean = complete[complete["hs_vs_clean_control"].notna()]
    return pd.DataFrame(
        [
            ("2013 dry-eye questionnaire participants", audit["questionnaire_participants"]),
            ("Questionnaire participants with an eligible nonexcluded stool 16S profile", audit["questionnaire_participants_with_microbiome_profile"]),
            ("Participants whose selected stool profile was within 3 years", len(within)),
            ("Complete primary symptom/covariate analysis", len(complete)),
            ("Frequent dryness and irritation cases in complete primary analysis", int((complete["highly_symptomatic_derived"] == 1).sum())),
            ("Strict asymptomatic comparison subset", len(clean)),
        ],
        columns=["Stage", "Participants"],
    )


def build_table1(base: pd.DataFrame) -> pd.DataFrame:
    dat = base[base["gap_days_abs"] <= 1095].dropna(subset=["highly_symptomatic_derived", *PRIMARY_COVARIATES]).copy()
    groups = [
        ("Overall", dat),
        ("No frequent dryness-and-irritation", dat[dat["highly_symptomatic_derived"] == 0]),
        ("Frequent dryness and irritation", dat[dat["highly_symptomatic_derived"] == 1]),
    ]
    rows: list[dict[str, str]] = []

    def add(label: str, function) -> None:
        row = {"Characteristic": label}
        for name, group in groups:
            row[name] = function(group)
        rows.append(row)

    add("Participants, n", lambda group: str(len(group)))
    add("Age, years, mean (SD)", lambda group: mean_sd(group["age"]))
    add("Body mass index, kg/m$^2$, mean (SD)", lambda group: mean_sd(group["BMI"]))
    for level in sorted(dat["SEX"].astype(str).unique()):
        add(f"Sex: {level}, n (%)", lambda group, level=level: n_percent(group["SEX"].astype(str) == level, len(group)))
    for level in sorted(dat["AntibioticsPriorMonth"].astype(str).unique()):
        add(
            f"Antibiotic metadata: {level}, n (%)",
            lambda group, level=level: n_percent(group["AntibioticsPriorMonth"].astype(str) == level, len(group)),
        )
    add("Absolute stool-questionnaire interval, days, median [IQR]", lambda group: median_iqr(group["gap_days_abs"], 0))
    add("Stool collected before questionnaire, n (%)", lambda group: n_percent(group["gap_days_signed"] > 0, len(group)))
    add("Stool collected after questionnaire, n (%)", lambda group: n_percent(group["gap_days_signed"] < 0, len(group)))
    add("Stool and questionnaire on the same day, n (%)", lambda group: n_percent(group["gap_days_signed"] == 0, len(group)))
    add("Sequencing reads, median [IQR]", lambda group: median_iqr(group["alpha_total_reads"], 0))
    add("Observed ASV richness, mean (SD)", lambda group: mean_sd(group["alpha_richness"]))
    add("Shannon diversity, mean (SD)", lambda group: mean_sd(group["alpha_shannon"], 2))
    return pd.DataFrame(rows)


def build_missingness(base: pd.DataFrame) -> pd.DataFrame:
    dat = base[base["gap_days_abs"] <= 1095].copy()
    fields = [
        "highly_symptomatic_derived",
        "either_often_symptom",
        "hs_vs_clean_control",
        "age",
        "BMI",
        "SEX",
        "AntibioticsPriorMonth",
        "gap_days_abs",
        "alpha_total_reads",
    ]
    rows = []
    for field in fields:
        missing = int(dat[field].isna().sum())
        unknown = int((dat[field].astype(str) == "Unknown").sum()) if field == "AntibioticsPriorMonth" else 0
        rows.append(
            {
                "Variable": field,
                "Cohort N": len(dat),
                "Missing n": missing,
                "Missing percent": round(100 * missing / len(dat), 1),
                "Explicit unknown category n": unknown,
            }
        )
    return pd.DataFrame(rows)


def build_alpha_table(production_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(production_dir / "primary" / "locked_primary_16s_association_matrix.csv")
    keep = data[
        (data["branch_id"] == "within_3y")
        & (data["covariate_set"] == "primary_core")
        & data["outcome_id"].isin(OUTCOME_LABELS)
        & data["feature"].isin(FEATURE_LABELS)
    ].copy()
    keep["Outcome"] = keep["outcome_id"].map(OUTCOME_LABELS)
    keep["Feature"] = keep["feature"].map(FEATURE_LABELS)
    keep["OR per SD (95% CI)"] = keep.apply(
        lambda row: f"{row.odds_ratio_per_sd:.2f} ({row.ci_low_or:.2f}--{row.ci_high_or:.2f})", axis=1
    )
    keep["P value"] = keep["p_value"].map(format_p)
    keep["BH q value"] = keep["q_value_alpha_six"].map(format_p)
    keep["N (cases)"] = keep.apply(lambda row: f"{int(row.n)} ({int(row.cases)})", axis=1)
    return keep[["Outcome", "Feature", "N (cases)", "OR per SD (95% CI)", "P value", "BH q value"]]


def build_sensitivity_table(production_dir: Path) -> pd.DataFrame:
    primary = pd.read_csv(production_dir / "primary" / "locked_primary_16s_association_matrix.csv")
    rare = pd.read_csv(production_dir / "method_strengthening" / "rarefaction_richness_sensitivity.csv")
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOME_LABELS:
        selections = [
            ("Primary observed richness", primary[(primary.branch_id == "within_3y") & (primary.covariate_set == "primary_core") & (primary.outcome_id == outcome) & (primary.feature == "alpha_richness")]),
            ("Observed richness plus read-depth covariate", primary[(primary.branch_id == "within_3y") & (primary.covariate_set == "primary_core_plus_depth") & (primary.outcome_id == outcome) & (primary.feature == "alpha_richness")]),
            ("Observed richness in rarefaction-eligible subset", rare[(rare.outcome_id == outcome) & (rare.feature == "alpha_richness")]),
            ("Expected rarefied richness in matched subset", rare[(rare.outcome_id == outcome) & (rare.feature == "alpha_richness_rarefied_expected")]),
            ("Observed richness within 1 year", primary[(primary.branch_id == "within_1y") & (primary.covariate_set == "primary_core") & (primary.outcome_id == outcome) & (primary.feature == "alpha_richness")]),
        ]
        for analysis, hit in selections:
            if hit.empty or pd.isna(hit.iloc[0].get("odds_ratio_per_sd")):
                continue
            row = hit.iloc[0]
            rows.append(
                {
                    "Outcome": OUTCOME_LABELS[outcome],
                    "Analysis": analysis,
                    "N (cases)": f"{int(row.n)} ({int(row.cases)})",
                    "OR per SD (95% CI)": f"{row.odds_ratio_per_sd:.2f} ({row.ci_low_or:.2f}--{row.ci_high_or:.2f})",
                    "P value": format_p(row.p_value),
                }
            )
    return pd.DataFrame(rows)


def build_tie_sensitivity_table(production_dir: Path) -> pd.DataFrame:
    runs = [
        ("Highest read depth (production rule)", production_dir / "primary"),
        ("Lowest read depth", production_dir / "tie_sensitivity_lowest_depth"),
        ("Source-file order", production_dir / "tie_sensitivity_source_order"),
    ]
    rows: list[dict[str, Any]] = []
    for rule, directory in runs:
        data = pd.read_csv(directory / "locked_primary_16s_association_matrix.csv")
        selected = data[
            (data.branch_id == "within_3y")
            & (data.covariate_set == "primary_core")
            & data.outcome_id.isin(["highly_symptomatic_and", "hs_vs_clean_control"])
            & (data.feature == "alpha_richness")
        ]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "Tied-profile rule": rule,
                    "Outcome": OUTCOME_LABELS[row.outcome_id],
                    "N (cases)": f"{int(row.n)} ({int(row.cases)})",
                    "OR per SD (95% CI)": f"{row.odds_ratio_per_sd:.2f} ({row.ci_low_or:.2f}--{row.ci_high_or:.2f})",
                    "P value": format_p(row.p_value),
                }
            )
    return pd.DataFrame(rows)


def build_traceability_table(production_dir: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Manuscript claim": "Primary cohort and case/control denominators",
                "Canonical source": "primary/locked_primary_16s_association_matrix.csv",
                "Selection": "within_3y; primary_core; highly_symptomatic_and; alpha_richness",
                "Manuscript location": "Results: Participants; Table 2",
            },
            {
                "Manuscript claim": "Six alpha-diversity q values",
                "Canonical source": "primary/locked_primary_16s_association_matrix.csv",
                "Selection": "q_value_alpha_six for two alpha features and three within-3y contrasts",
                "Manuscript location": "Results: Alpha diversity; Table 2",
            },
            {
                "Manuscript claim": "Matched observed and expected-rarefied richness",
                "Canonical source": "method_strengthening/rarefaction_richness_sensitivity.csv",
                "Selection": "matched_rarefaction_eligible_subset",
                "Manuscript location": "Results: Robustness",
            },
            {
                "Manuscript claim": "Community pseudo-F, partial R2, p and dispersion",
                "Canonical source": "method_strengthening/beta_diversity_global_community.csv",
                "Selection": "all three outcome_id rows",
                "Manuscript location": "Results: Genus-level community structure",
            },
            {
                "Manuscript claim": "Tied-profile robustness",
                "Canonical source": "primary and tie_sensitivity_*/locked_primary_16s_association_matrix.csv",
                "Selection": "within_3y; primary_core; highly_symptomatic_and; alpha_richness",
                "Manuscript location": "Results: Robustness; Supplementary Table S3",
            },
            {
                "Manuscript claim": "Exploratory Dialister candidate estimates",
                "Canonical source": "method_strengthening/clr_candidate_compositional_sensitivity.csv",
                "Selection": "candidate=Dialister; three outcome_id rows",
                "Manuscript location": "Results: Exploratory genus-level findings",
            },
            {
                "Manuscript claim": "ANCOM-BC2 fixed-universe results",
                "Canonical source": "ancombc2_summary/ancombc2_outcome_summary.csv",
                "Selection": "all three outcome_id rows",
                "Manuscript location": "Results: Exploratory genus-level findings",
            },
            {
                "Manuscript claim": "Formal package diagnostic-valid results",
                "Canonical source": "formal_methods/formal_method_min_q_summary.csv",
                "Selection": "valid_taxa only; method/outcome/component q family",
                "Manuscript location": "Results: Exploratory genus-level findings",
            },
        ]
    )


def build_forest_plot(alpha: pd.DataFrame, path: Path) -> None:
    raw = pd.read_csv(path.parents[1] / "primary" / "locked_primary_16s_association_matrix.csv")
    plot = raw[
        (raw.branch_id == "within_3y")
        & (raw.covariate_set == "primary_core")
        & raw.outcome_id.isin(OUTCOME_LABELS)
        & (raw.feature == "alpha_richness")
    ].copy()
    plot["label"] = plot.outcome_id.map(
        {
            "either_often_symptom_or": "Frequent dryness or irritation",
            "highly_symptomatic_and": "Frequent dryness and irritation",
            "hs_vs_clean_control": "Frequent dryness and irritation vs asymptomatic",
        }
    )
    plot = plot.iloc[::-1]
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.errorbar(
        plot.odds_ratio_per_sd,
        y,
        xerr=[plot.odds_ratio_per_sd - plot.ci_low_or, plot.ci_high_or - plot.odds_ratio_per_sd],
        fmt="o",
        color="#24577A",
        capsize=4,
    )
    ax.axvline(1, color="#555555", linewidth=1, linestyle="--")
    ax.set_yticks(y, plot.label)
    ax.set_xlabel("Odds ratio per SD higher richness (95% CI)")
    ax.set_xlim(0.45, 1.05)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_provenance(data_root: Path, production_dir: Path, out_dir: Path) -> dict[str, Any]:
    source_paths = {
        "dry_eye_questionnaire": data_root / DED_2013,
        "microbiome_mapping": data_root / MICROBIOME_DIR / "TwinsUK_ASVs_MappingFile.xlsx",
        "asv_counts": data_root / MICROBIOME_DIR / "TwinsUK_ASVsCR.rds",
        "asv_taxonomy": data_root / MICROBIOME_DIR / "TwinsUK_ASVs_Taxa.csv.zip",
        "family_spine": data_root / SPINE,
    }
    script_paths = [
        PROJECT_ROOT / "scripts" / name
        for name in [
            "run_2013_initial_microbiome_signal_probes.py",
            "run_2013_locked_primary_16s_analysis.py",
            "run_2013_method_strengthening_batch.py",
            "run_2013_alcohol_medication_sensitivity.py",
            "build_2013_ancombc2_inputs.py",
            "run_2013_ancombc2_analysis.R",
            "summarize_2013_ancombc2_results.py",
            "run_2013_formal_microbiome_package_sensitivity.R",
            "build_2013_coauthor_review_artifacts.py",
            "validate_2013_coauthor_review_release.py",
        ]
    ]
    exact_analysis_commands = [
        "run_2013_locked_primary_16s_analysis.py --tie-rule highest_read_depth",
        "run_2013_locked_primary_16s_analysis.py --tie-rule lowest_read_depth",
        "run_2013_locked_primary_16s_analysis.py --tie-rule source_order",
        "run_2013_method_strengthening_batch.py --permutations 1999",
        "run_2013_alcohol_medication_sensitivity.py",
        "build_2013_ancombc2_inputs.py --prevalence-cut 0.10",
        "run_2013_ancombc2_analysis.R --outcomes either_often_symptom_or",
        "run_2013_ancombc2_analysis.R --outcomes highly_symptomatic_and",
        "run_2013_ancombc2_analysis.R --outcomes hs_vs_clean_control",
        "run_2013_ancombc2_analysis.R --outcomes hs_vs_clean_control --fixed-only",
        "summarize_2013_ancombc2_results.py",
        "run_2013_formal_microbiome_package_sensitivity.R --omit-maaslin3-by-decision --aldex-mc-samples 128 --workers 1 --seed-base 20260814",
        "build_2013_coauthor_review_artifacts.py",
        "validate_2013_coauthor_review_release.py --require-formal",
    ]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "production_dir": str(production_dir),
        "tie_rule": "highest_read_depth; source-order and lowest-depth sensitivity outputs required",
        "unresolved_genus_policy": "exclude ASVs without assigned genus from genus-level analysis; retain all ASVs for alpha diversity",
        "source_files": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
        "analysis_scripts": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in script_paths
        },
        "exact_analysis_commands": exact_analysis_commands,
        "reproduction_commands": {
            "path": str(REPRODUCTION_COMMANDS),
            "bytes": REPRODUCTION_COMMANDS.stat().st_size,
            "sha256": sha256(REPRODUCTION_COMMANDS),
        },
        "reporting_output_dir": str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build aggregate coauthor-review reporting artifacts from canonical outputs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--production-dir", type=Path, default=DEFAULT_PRODUCTION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base, audit = build_analysis_data(args.data_root)
    flow = build_flow(base, audit)
    table1 = build_table1(base)
    missingness = build_missingness(base)
    alpha = build_alpha_table(args.production_dir)
    sensitivity = build_sensitivity_table(args.production_dir)
    tie_sensitivity = build_tie_sensitivity_table(args.production_dir)
    traceability = build_traceability_table(args.production_dir)
    community = pd.read_csv(args.production_dir / "method_strengthening" / "beta_diversity_global_community.csv")
    community["Outcome"] = community["outcome_id"].map(OUTCOME_LABELS)

    for name, table in [
        ("participant_flow", flow),
        ("table1_characteristics", table1),
        ("missingness", missingness),
        ("table2_alpha_diversity", alpha),
        ("table_s1_richness_sensitivities", sensitivity),
        ("table_s2_community", community),
        ("table_s3_tied_profile_sensitivity", tie_sensitivity),
        ("source_to_prose_traceability", traceability),
    ]:
        write_table(table, args.out_dir / name)
        write_latex_tabular(table, args.out_dir / f"{name}.tex")

    timing = base[base["gap_days_abs"] <= 1095]
    timing_summary = {
        "n": int(len(timing)),
        "stool_before_questionnaire": int((timing["gap_days_signed"] > 0).sum()),
        "stool_after_questionnaire": int((timing["gap_days_signed"] < 0).sum()),
        "same_day": int((timing["gap_days_signed"] == 0).sum()),
        "absolute_gap_median": float(timing["gap_days_abs"].median()),
        "absolute_gap_q1": float(timing["gap_days_abs"].quantile(.25)),
        "absolute_gap_q3": float(timing["gap_days_abs"].quantile(.75)),
        "absolute_gap_min": int(timing["gap_days_abs"].min()),
        "absolute_gap_max": int(timing["gap_days_abs"].max()),
    }
    (args.out_dir / "cohort_audit.json").write_text(json.dumps({**audit, "timing": timing_summary}, indent=2) + "\n", encoding="utf-8")
    build_forest_plot(alpha, args.out_dir / "figure1_richness_forest.png")
    provenance = build_provenance(args.data_root, args.production_dir, args.out_dir)
    (args.out_dir / "provenance_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cohort": audit, "timing": timing_summary, "outputs": str(args.out_dir)}, indent=2))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
