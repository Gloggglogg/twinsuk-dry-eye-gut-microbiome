"""Render the grouped-score figure from aggregate reference results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "published_results/grouped_score_display.csv"
    data = pd.read_csv(source)
    assert data.score_group.astype(str).tolist() == ["1", "2", "3", "4-6"]
    assert data.cases.astype(int).tolist() == [486, 496, 74, 97]
    assert data.controls.eq(525).all()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="constrained")
    y = data.odds_ratio_per_global_richness_sd.to_numpy()
    ax.errorbar(range(4), y, yerr=[y - data.ci_low_or, data.ci_high_or - y], fmt="o-", color="#176B87", capsize=4, lw=1.7)
    ax.axhline(1, color="#6B7280", ls="--", lw=1)
    ax.set_xticks(range(4), [f"{label}\nn={int(n)}" for label, n in zip(["1", "2", "3", "4–6"], data.cases)])
    ax.set_xlabel("Combined dryness and irritation frequency score")
    ax.set_ylabel("Adjusted odds ratio\nper standard deviation higher richness")
    ax.set_ylim(.50, 1.18)
    ax.set_xlim(-.25, 3.25)
    ax.set_title("Reference: 525 strictly asymptomatic participants", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB")
    target = args.output_dir / "grouped_symptom_score.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    receipt = {"input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "source_data_kind": "saved aggregate study estimates", "models_refitted": False}
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
