from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from run_2013_initial_microbiome_signal_probes import (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    compute_alpha_and_genus,
    load_asv_counts,
    load_ded_2013,
    load_family_ids,
    load_mapping,
    load_taxa,
    nearest_stool_pairs,
    sample_read_depths,
)
from run_2013_locked_primary_16s_analysis import CORE_COVARIATES, build_x, sex_is_estimable


OUTCOME = "highly_symptomatic_derived"
FEATURE = "alpha_richness"
MAX_GAP_DAYS = 365 * 3
EXPECTED_N = 1685
EXPECTED_CASES = 93
EXPECTED_FAMILIES = 1125
EXPECTED_OR = 0.7206689313902943
DEFAULT_SEED = 20260823
DEFAULT_BOOTSTRAPS = 2000
ALPHA_RESULTS = PROJECT_ROOT / 'runtime' / 'coauthor_review_production_2026_08_14' / 'primary' / 'locked_primary_16s_association_matrix.csv'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def assemble_main_cohort(data_root: Path) -> tuple[pd.DataFrame, list[str]]:
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
    alpha, _ = compute_alpha_and_genus(counts, taxa)
    base = pairs.merge(alpha, on="SequencingSpecificName", how="inner")
    within_3y = base.loc[base["gap_days_abs"] <= MAX_GAP_DAYS].copy()
    covariates = [*CORE_COVARIATES, *(["SEX"] if sex_is_estimable(within_3y) else [])]

    model = within_3y[[OUTCOME, FEATURE, "FamilyID", *covariates]].copy()
    model[OUTCOME] = pd.to_numeric(model[OUTCOME], errors="coerce")
    model[FEATURE] = pd.to_numeric(model[FEATURE], errors="coerce")
    model = model.dropna(subset=[OUTCOME, FEATURE])
    design = build_x(model, FEATURE, covariates)
    keep = design.notna().all(axis=1)
    return model.loc[keep].copy(), covariates


def prepare_design(
    model: pd.DataFrame,
    covariates: list[str],
) -> tuple[pd.Series, pd.DataFrame, pd.Series, float, float]:
    design = build_x(model, FEATURE, covariates).astype(float)
    richness_mean = float(model[FEATURE].mean())
    richness_sd = float(model[FEATURE].std(ddof=1))
    if not math.isfinite(richness_sd) or richness_sd <= 0:
        raise RuntimeError("Observed richness SD is unavailable or non-positive")
    design["feature_z"] = (model[FEATURE] - richness_mean) / richness_sd
    design = sm.add_constant(design, has_constant="add")
    y = model[OUTCOME].astype(int)
    groups = model["FamilyID"].astype(str)
    return y, design, groups, richness_mean, richness_sd


def standardized_probabilities(params: pd.Series, target_design: pd.DataFrame) -> dict[str, float]:
    low = target_design.copy()
    high = target_design.copy()
    low["feature_z"] = -1.0
    high["feature_z"] = 1.0
    p_low = float(np.mean(sm.families.Binomial().link.inverse(np.asarray(low @ params))))
    p_high = float(np.mean(sm.families.Binomial().link.inverse(np.asarray(high @ params))))
    return {
        "probability_at_minus_1_sd": p_low,
        "probability_at_plus_1_sd": p_high,
        "difference_plus_1_minus_minus_1": p_high - p_low,
    }


def fit_point_model(
    y: pd.Series,
    design: pd.DataFrame,
    groups: pd.Series,
) -> tuple[Any, dict[str, float]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
    probabilities = standardized_probabilities(fit.params, design)
    probabilities["odds_ratio_per_sd"] = float(math.exp(fit.params["feature_z"]))
    return fit, probabilities


def family_cluster_bootstrap(
    y: pd.Series,
    design: pd.DataFrame,
    groups: pd.Series,
    *,
    n_bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    family_rows = {
        family: np.flatnonzero(groups.to_numpy() == family)
        for family in pd.unique(groups)
    }
    family_ids = np.asarray(list(family_rows), dtype=object)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    failures: dict[str, int] = {}

    for _ in range(n_bootstraps):
        sampled = rng.choice(family_ids, size=len(family_ids), replace=True)
        indices = np.concatenate([family_rows[family] for family in sampled])
        y_boot = y.iloc[indices]
        if y_boot.nunique() < 2:
            failures["single_outcome_level"] = failures.get("single_outcome_level", 0) + 1
            continue
        x_boot = design.iloc[indices]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.GLM(y_boot, x_boot, family=sm.families.Binomial()).fit()
            values = standardized_probabilities(fit.params, design)
            if all(math.isfinite(value) for value in values.values()):
                rows.append(values)
            else:
                failures["non_finite_probability"] = failures.get("non_finite_probability", 0) + 1
        except Exception as exc:
            name = type(exc).__name__
            failures[name] = failures.get(name, 0) + 1

    frame = pd.DataFrame(rows)
    diagnostics = {
        "requested_resamples": n_bootstraps,
        "successful_resamples": int(len(frame)),
        "success_percent": 100.0 * len(frame) / n_bootstraps,
        "seed": seed,
        "failures": failures,
        "resampling_unit": "FamilyID",
        "target_covariate_distribution": "observed complete main analysis cohort",
    }
    if len(frame) < 0.95 * n_bootstraps:
        raise RuntimeError(
            f"Only {len(frame)} of {n_bootstraps} family-cluster bootstrap fits succeeded"
        )
    return frame, diagnostics


def percentile_interval(values: pd.Series) -> tuple[float, float]:
    low, high = np.quantile(values.to_numpy(dtype=float), [0.025, 0.975])
    return float(low), float(high)


def build_story_figure(display: pd.DataFrame, output_path: Path) -> None:
    alpha = pd.read_csv(ALPHA_RESULTS)
    alpha = alpha.loc[
        alpha["branch_id"].eq("within_3y")
        & alpha["covariate_set"].eq("primary_core")
        & alpha["feature"].eq(FEATURE)
        & alpha["outcome_id"].isin(
            ["either_often_symptom_or", "highly_symptomatic_and", "hs_vs_clean_control"]
        )
    ].copy()
    label_map = {
        "either_often_symptom_or": "Dryness or irritation\nreported often or always",
        "highly_symptomatic_and": "Dryness and irritation\nreported often or always",
        "hs_vs_clean_control": "Frequent dryness and irritation\nversus strictly asymptomatic controls",
    }
    order = list(label_map)
    alpha["order"] = alpha["outcome_id"].map({label: index for index, label in enumerate(order)})
    alpha = alpha.sort_values("order")
    if len(alpha) != 3:
        raise RuntimeError(f"Expected three observed-richness rows for the story figure; found {len(alpha)}")

    probability = display.loc[
        display["estimand"].isin(["probability_at_minus_1_sd", "probability_at_plus_1_sd"])
    ].copy()
    probability = probability.sort_values("richness_z")
    difference = display.loc[
        display["estimand"].eq("difference_plus_1_minus_minus_1")
    ].iloc[0]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.35), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    y_pos = np.arange(len(alpha))[::-1]
    colors = ["#6B7280", "#176B87", "#6B7280"]
    for y, (_, row), color in zip(y_pos, alpha.iterrows(), colors, strict=True):
        estimate = float(row["odds_ratio_per_sd"])
        low = float(row["ci_low_or"])
        high = float(row["ci_high_or"])
        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
            linewidth=1.6,
        )
    ax.axvline(1.0, color="#9CA3AF", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(0.48, 1.12)
    ax.set_xticks([0.5, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.5", "0.6", "0.8", "1.0"])
    ax.minorticks_off()
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label_map[value] for value in alpha["outcome_id"]])
    ax.set_xlabel("Adjusted odds ratio per SD higher richness")
    ax.set_title("A  Association across symptom contrasts", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    x = np.arange(2)
    estimates = probability["estimate"].to_numpy(dtype=float)
    lows = probability["ci_low"].to_numpy(dtype=float)
    highs = probability["ci_high"].to_numpy(dtype=float)
    ax.plot(x, estimates, color="#176B87", linewidth=1.8, zorder=1)
    ax.errorbar(
        x,
        estimates,
        yerr=[estimates - lows, highs - estimates],
        fmt="o",
        color="#176B87",
        ecolor="#176B87",
        capsize=4,
        markersize=7,
        linewidth=1.6,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["1 SD below\nmean richness", "1 SD above\nmean richness"])
    ax.set_ylim(0, max(0.105, float(highs.max()) * 1.2))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Standardized adjusted symptom probability")
    ax.set_title("B  Absolute interpretation of the primary model", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.5,
        0.97,
        (
            f"Difference {100 * float(difference['estimate']):.1f} percentage points\n"
            f"95% CI {100 * float(difference['ci_low']):.1f} to "
            f"{100 * float(difference['ci_high']):.1f}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="#374151",
    )

    fig.suptitle(
        "Observed stool ASV richness and frequent ocular dryness and irritation",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate standardized adjusted probabilities of frequent dryness and irritation "
            "at one SD below versus one SD above mean observed ASV richness."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "adjusted_probability_contrast_2026_08_23",
    )
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.bootstraps < 200:
        raise SystemExit("Use at least 200 bootstrap resamples")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, covariates = assemble_main_cohort(args.data_root)
    y, design, groups, richness_mean, richness_sd = prepare_design(model, covariates)
    invariants = (len(y), int(y.sum()), int(groups.nunique()))
    if invariants != (EXPECTED_N, EXPECTED_CASES, EXPECTED_FAMILIES):
        raise RuntimeError(
            "Main-cohort invariant failed: "
            f"observed n/cases/families={invariants}, "
            f"expected={(EXPECTED_N, EXPECTED_CASES, EXPECTED_FAMILIES)}"
        )

    fit, point = fit_point_model(y, design, groups)
    if not math.isclose(point["odds_ratio_per_sd"], EXPECTED_OR, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(
            "Adjusted-probability model did not reproduce the locked richness odds ratio: "
            f"observed={point['odds_ratio_per_sd']:.16g}, expected={EXPECTED_OR:.16g}"
        )

    bootstrap, diagnostics = family_cluster_bootstrap(
        y,
        design,
        groups,
        n_bootstraps=args.bootstraps,
        seed=args.seed,
    )
    display_rows = []
    for key, label, richness_z in [
        ("probability_at_minus_1_sd", "One SD below mean richness", -1.0),
        ("probability_at_plus_1_sd", "One SD above mean richness", 1.0),
        (
            "difference_plus_1_minus_minus_1",
            "Adjusted probability difference: +1 SD minus -1 SD",
            np.nan,
        ),
    ]:
        ci_low, ci_high = percentile_interval(bootstrap[key])
        display_rows.append(
            {
                "estimand": key,
                "label": label,
                "richness_z": richness_z,
                "estimate": point[key],
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    display = pd.DataFrame(display_rows)
    display.to_csv(args.out_dir / "adjusted_probability_contrast.csv", index=False)
    write_json(args.out_dir / "bootstrap_diagnostics.json", diagnostics)
    figure_path = args.out_dir / "figure1_richness_story.png"
    build_story_figure(display, figure_path)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "approved interpretability analysis of the locked primary model; no new hypothesis test",
        "outcome": "both ocular dryness and irritation reported often or always",
        "feature": "observed ASV richness",
        "n": int(len(y)),
        "cases": int(y.sum()),
        "controls": int(len(y) - y.sum()),
        "families": int(groups.nunique()),
        "covariates": covariates,
        "richness_mean_asv_count": richness_mean,
        "richness_sd_asv_count": richness_sd,
        "odds_ratio_per_sd": point["odds_ratio_per_sd"],
        "locked_odds_ratio_reproduced": True,
        "adjusted_probabilities": point,
        "interval_method": "percentile intervals from family-cluster bootstrap",
        "bootstrap_diagnostics": diagnostics,
        "output_csv": str(args.out_dir / "adjusted_probability_contrast.csv"),
        "story_figure": str(figure_path),
        "story_figure_sha256": sha256(figure_path),
        "script_sha256": sha256(Path(__file__).resolve()),
        "participant_level_output_written": False,
        "raw_data_written": False,
    }
    write_json(args.out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
