"""Generate a fixed, wholly artificial 600-person twin fixture."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 20261028


def create_synthetic_inputs(output_dir: str | Path, seed: int = DEFAULT_SEED) -> dict:
    """Write clearly labelled synthetic input CSVs, with no study-data dependency."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    names = ["cohort.csv", "asv_counts.csv", "taxonomy.csv", "input_provenance.json"]
    if any((target / name).exists() for name in names):
        raise FileExistsError("Synthetic input destination already contains input files")
    rng = np.random.default_rng(seed)
    n, pairs, asvs = 600, 300, 120
    scores = np.repeat(np.arange(7), [180, 75, 85, 85, 75, 60, 40])
    rng.shuffle(scores)
    dryness = np.empty(n, dtype=int)
    irritation = np.empty(n, dtype=int)
    for i, score in enumerate(scores):
        low, high = max(0, int(score) - 3), min(3, int(score))
        dryness[i] = rng.integers(low, high + 1)
        irritation[i] = int(score) - dryness[i]
    family_age = rng.uniform(35, 78, pairs)
    family_bmi = rng.normal(25, 3.1, pairs)
    family_sex = rng.choice(["F", "M"], pairs, p=[0.7, 0.3])
    sex = np.repeat(family_sex, 2)
    sex[301::2] = rng.choice(["F", "M"], len(sex[301::2]), p=[0.7, 0.3])
    signed_gap = rng.integers(1, 1001, n) * rng.choice([-1, 1], n, p=[0.65, 0.35])
    diagnosis = rng.binomial(1, 0.12, n)
    recent = np.where(scores > 0, 1, rng.binomial(1, 0.1, n))
    cohort = pd.DataFrame({
        "PublicID": [f"SYN_P{i+1:04d}" for i in range(n)],
        "FamilyID": np.repeat([f"SYN_F{i+1:04d}" for i in range(pairs)], 2),
        "SequencingSpecificName": [f"SYN_S{i+1:04d}" for i in range(n)],
        "How often dry eyes": dryness,
        "How often irritated eyes": irritation,
        "Symptoms past 3 months": recent,
        "Ever dx DED": diagnosis,
        "age": np.repeat(family_age, 2),
        "BMI": np.repeat(family_bmi, 2) + rng.normal(0, 1.2, n),
        "SEX": sex,
        "ACTUAL_ZYGOSITY": np.repeat(["MZ"] * 150 + ["DZ"] * 150, 2),
        "AntibioticsPriorMonth": rng.choice(["N", "Y", "Unknown"], n, p=[0.7, 0.15, 0.15]),
        "gap_days_signed": signed_gap,
        "gap_days_abs": np.abs(signed_gap),
        "ResponseDate": "2013-07-01",
        "ibs_doctor_diagnosed": rng.choice([0., 1., np.nan], n, p=[0.7, 0.2, 0.1]),
    })
    family_signal = np.repeat(rng.normal(0, 0.35, pairs), 2)
    probability = 1 / (1 + np.exp(-(.5 - .12 * scores + family_signal)))
    detection = rng.random((n, asvs)) < probability[:, None]
    values = rng.negative_binomial(6, .12, size=(n, asvs)) + 1
    counts = pd.DataFrame(values * detection, columns=[f"SYN_ASV{i+1:03d}" for i in range(asvs)])
    counts.insert(0, "SequencingSpecificName", cohort.SequencingSpecificName)
    taxa = pd.DataFrame({
        "asv": [f"SYN_ASV{i+1:03d}" for i in range(asvs)],
        "Kingdom": "Bacteria",
        "Phylum": [f"SYN_Phylum{i % 4 + 1}" for i in range(asvs)],
        "Family": [f"SYN_Family{i % 12 + 1}" for i in range(asvs)],
        "Genus_clean": [f"SYN_Genus{i // 3 + 1:02d}" for i in range(asvs)],
    })
    cohort.to_csv(target / "cohort.csv", index=False)
    counts.to_csv(target / "asv_counts.csv", index=False)
    taxa.to_csv(target / "taxonomy.csv", index=False)
    manifest = {
        "data_kind": "synthetic", "seed": int(seed), "participants": n,
        "families": pairs, "asv_features": asvs, "genus_features": 40,
        "description": "Wholly artificial software fixture; every identifier starts SYN_.",
        "source_study_data_used": False,
        "scientific_interpretation": "Illustrates execution only; values are unrelated to study estimates.",
    }
    (target / "input_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
