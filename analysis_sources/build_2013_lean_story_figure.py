from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / 'runtime' / 'manuscript_figures_2026_08_29'
ALPHA_RESULTS = PROJECT_ROOT / 'runtime' / 'coauthor_review_production_2026_08_14' / 'primary' / 'locked_primary_16s_association_matrix.csv'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a focused manuscript forest plot from existing aggregate alpha-diversity "
            "results without rerunning statistical models."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not ALPHA_RESULTS.exists():
        raise RuntimeError("Required aggregate alpha-diversity results are missing")

    alpha = pd.read_csv(ALPHA_RESULTS)
    label_map = {
        "highly_symptomatic_and": "Primary analysis\nfrequent dryness and irritation",
        "hs_vs_clean_control": "Strict-control sensitivity\nsame cases versus strictly asymptomatic controls",
    }
    alpha = alpha.loc[
        alpha["branch_id"].eq("within_3y")
        & alpha["covariate_set"].eq("primary_core")
        & alpha["feature"].eq("alpha_richness")
        & alpha["outcome_id"].isin(label_map)
    ].copy()
    alpha["order"] = alpha["outcome_id"].map(
        {outcome_id: index for index, outcome_id in enumerate(label_map)}
    )
    alpha = alpha.sort_values("order")
    if len(alpha) != 2:
        raise RuntimeError(f"Expected two richness rows; found {len(alpha)}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 3.15))
    y_pos = np.array([1.0, 0.0])
    colors = ["#176B87", "#6B7280"]
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
    ax.set_xlim(0.48, 1.22)
    ax.set_xticks([0.5, 0.6, 0.8, 1.0, 1.2])
    ax.set_xticklabels(["0.5", "0.6", "0.8", "1.0", "1.2"])
    ax.minorticks_off()
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label_map[value] for value in alpha["outcome_id"]])
    ax.set_ylim(-0.65, 1.65)
    ax.set_xlabel("Adjusted odds ratio per SD higher richness")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(
        1.04,
        1.08,
        "Adjusted OR (95% CI)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#374151",
        fontweight="bold",
    )
    for y, (_, row) in zip(y_pos, alpha.iterrows(), strict=True):
        ax.text(
            1.04,
            y,
            (
                f"{float(row['odds_ratio_per_sd']):.2f} "
                f"({float(row['ci_low_or']):.2f} to {float(row['ci_high_or']):.2f})"
            ),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            color="#1F2937",
        )

    fig.suptitle(
        "Lower stool ASV richness was associated with frequent ocular dryness and irritation",
        fontsize=11.5,
        fontweight="bold",
        y=0.96,
    )
    fig.subplots_adjust(left=0.36, right=0.72, top=0.77, bottom=0.24)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "figure1_richness_story_lean.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest_path = args.out_dir / "figure1_richness_story_lean_manifest.json"
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "role": "focused reader-facing forest plot built from existing aggregate results",
        "statistical_models_rerun": False,
        "outcomes_displayed": list(label_map),
        "inputs": [
            {"path": str(ALPHA_RESULTS), "sha256": sha256(ALPHA_RESULTS)},
        ],
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
        "script_sha256": sha256(Path(__file__).resolve()),
        "participant_level_output_written": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
