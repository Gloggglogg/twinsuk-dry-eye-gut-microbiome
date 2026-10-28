from __future__ import annotations

import argparse
import json
import math
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / 'authorized_inputs'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DED_2013 = Path("DED_outcomes_excel/DED_outcomes_2013.xlsx")
MICROBIOME_DIR = Path("New March 2022/Extracted/16S_Microbiome_Gut_ASVs")
SPINE = Path("derived/publicid_spine.csv")

OUTCOMES = {
    "any_current_symptom_2013": "symptom_any_current",
    "either_often_symptom_2013": "either_often_symptom",
    "hs_dry_eye_2013": "highly_symptomatic_derived",
    "hs_clean_control_2013": "hs_vs_clean_control",
}
CONTINUOUS_OUTCOMES = {
    "symptom_sum_0_6_2013": "symptom_sum_0_6",
}
TIMING_BRANCHES = {
    "within_1y": 365,
    "within_3y": 365 * 3,
}
GENUS_PREVALENCE_THRESHOLD = 0.10
MAX_GENUS_FEATURES_FOR_PREDICTION = 80
MIN_BINARY_CASES = 20


@dataclass(frozen=True)
class BranchSummary:
    branch_id: str
    max_gap_days: int
    rows: int
    unique_participants: int
    median_abs_gap_days: float | None
    hs_cases: int
    any_symptom_cases: int
    any_symptom_controls: int
    either_often_cases: int
    either_often_controls: int
    hs_all_other_controls: int
    hs_clean_cases: int
    hs_clean_controls: int
    symptom_sum_nonmissing: int
    symptom_sum_mean: float | None
    genus_features: int


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [" ".join(str(col).split()) for col in df.columns]
    return df


def load_ded_2013(data_root: Path) -> pd.DataFrame:
    ded = normalize_columns(pd.read_excel(data_root / DED_2013))
    keep = [
        "PublicID",
        "ResponseDate",
        "Symptoms past 3 months",
        "Ever dx DED",
        "WHS DED",
        "highly symptomatic dry eye",
        "How often dry eyes",
        "How often irritated eyes",
        "age",
        "SEX",
        "ACTUAL_ZYGOSITY",
    ]
    ded = ded[keep].copy()
    ded["PublicID"] = ded["PublicID"].astype(str)
    ded["ResponseDate"] = pd.to_datetime(ded["ResponseDate"], errors="coerce")
    ded["age"] = pd.to_numeric(ded["age"], errors="coerce")
    for col in [
        "Symptoms past 3 months",
        "Ever dx DED",
        "WHS DED",
        "highly symptomatic dry eye",
        "How often dry eyes",
        "How often irritated eyes",
    ]:
        ded[col] = pd.to_numeric(ded[col], errors="coerce")
    ded = derive_symptom_outcomes(ded)
    return ded.dropna(subset=["PublicID", "ResponseDate"])


def derive_symptom_outcomes(ded: pd.DataFrame) -> pd.DataFrame:
    out = ded.copy()
    dryness = out["How often dry eyes"]
    irritation = out["How often irritated eyes"]
    both_symptom_questions = dryness.notna() & irritation.notna()
    no_current_symptoms = both_symptom_questions & (dryness == 0) & (irritation == 0)
    no_recent_symptoms = out["Symptoms past 3 months"].fillna(1) == 0
    no_ded_diagnosis = out["Ever dx DED"].fillna(1) == 0

    out["symptom_sum_0_6"] = dryness + irritation
    out.loc[~both_symptom_questions, "symptom_sum_0_6"] = np.nan
    out["symptom_any_current"] = np.where(
        both_symptom_questions,
        ((dryness > 0) | (irritation > 0)).astype(float),
        np.nan,
    )
    out["highly_symptomatic_derived"] = np.where(
        both_symptom_questions,
        ((dryness >= 2) & (irritation >= 2)).astype(float),
        np.nan,
    )
    complete_whs_inputs = both_symptom_questions & out["Ever dx DED"].notna()
    out["whs_ded_derived"] = np.where(
        complete_whs_inputs,
        (
            (out["highly_symptomatic_derived"] == 1)
            | (out["Ever dx DED"] == 1)
        ).astype(float),
        np.nan,
    )
    if "WHS DED" in out.columns:
        source_whs = pd.to_numeric(out["WHS DED"], errors="coerce")
        comparable = source_whs.notna() & out["whs_ded_derived"].notna()
        if not bool(
            (
                source_whs.loc[comparable].astype(int)
                == out.loc[comparable, "whs_ded_derived"].astype(int)
            ).all()
        ):
            raise RuntimeError("Derived WHS DED does not match the supplied source variable")
    out["either_often_symptom"] = np.where(
        both_symptom_questions,
        ((dryness >= 2) | (irritation >= 2)).astype(float),
        np.nan,
    )
    out["clean_control_no_ded_no_symptoms"] = np.where(
        both_symptom_questions & out["Symptoms past 3 months"].notna() & out["Ever dx DED"].notna(),
        (no_current_symptoms & no_recent_symptoms & no_ded_diagnosis).astype(float),
        np.nan,
    )
    out["hs_vs_clean_control"] = np.nan
    out.loc[out["highly_symptomatic_derived"] == 1, "hs_vs_clean_control"] = 1.0
    out.loc[out["clean_control_no_ded_no_symptoms"] == 1, "hs_vs_clean_control"] = 0.0
    return out


def load_mapping(data_root: Path) -> pd.DataFrame:
    mapping = normalize_columns(pd.read_excel(data_root / MICROBIOME_DIR / "TwinsUK_ASVs_MappingFile.xlsx"))
    keep = [
        "SequencingSpecificName",
        "PublicID",
        "FPUniqueSampleID",
        "CollectionDateCombUseReceiveDateIfNothingOnTube",
        "Age",
        "BMI",
        "AntibioticsPriorMonth",
        "ShouldExcludeSample",
        "PCRPlate",
        "ExtractionPlate",
        "SequencingRunID",
        "SequencingLocation",
    ]
    mapping = mapping[[col for col in keep if col in mapping.columns]].copy()
    mapping["PublicID"] = mapping["PublicID"].astype(str)
    mapping["SequencingSpecificName"] = mapping["SequencingSpecificName"].astype(str)
    mapping["microbiome_date"] = pd.to_datetime(
        mapping["CollectionDateCombUseReceiveDateIfNothingOnTube"], errors="coerce"
    )
    mapping["microbiome_age"] = pd.to_numeric(mapping.get("Age"), errors="coerce")
    mapping["BMI"] = pd.to_numeric(mapping.get("BMI"), errors="coerce")
    mapping["ShouldExcludeSample"] = pd.to_numeric(mapping.get("ShouldExcludeSample"), errors="coerce").fillna(0)
    mapping["AntibioticsPriorMonth"] = mapping.get("AntibioticsPriorMonth", pd.Series(index=mapping.index)).fillna(
        "Unknown"
    ).astype(str)
    return mapping.dropna(subset=["PublicID", "SequencingSpecificName", "microbiome_date"])


def load_family_ids(data_root: Path) -> pd.DataFrame:
    spine_path = data_root / SPINE
    if not spine_path.exists():
        return pd.DataFrame(columns=["PublicID", "FamilyID"])
    spine = pd.read_csv(spine_path, usecols=["PublicID", "FamilyID"], low_memory=False)
    spine["PublicID"] = spine["PublicID"].astype(str)
    spine["FamilyID"] = spine["FamilyID"].astype("string")
    return spine.dropna(subset=["PublicID"]).drop_duplicates("PublicID")


def load_asv_counts(data_root: Path) -> pd.DataFrame:
    objects = pyreadr.read_r(str(data_root / MICROBIOME_DIR / "TwinsUK_ASVsCR.rds"))
    counts = next(iter(objects.values())).copy()
    counts.index = counts.index.astype(str)
    counts = counts.reset_index().rename(columns={"rownames": "SequencingSpecificName"})
    counts["SequencingSpecificName"] = counts["SequencingSpecificName"].astype(str)
    return counts


def load_taxa(data_root: Path) -> pd.DataFrame:
    with zipfile.ZipFile(data_root / MICROBIOME_DIR / "TwinsUK_ASVs_Taxa.csv.zip") as zf:
        with zf.open("TwinsUK_ASVs_Taxa.csv") as handle:
            taxa = pd.read_csv(handle)
    taxa = taxa.rename(columns={"Unnamed: 0": "asv"})
    genus = taxa["Genus"].astype("string").str.strip()
    genus = genus.mask(genus.isin(["", "NA", "N/A", "nan", "None", "Unassigned", "unassigned"]))
    taxa["Genus_clean"] = genus
    return taxa


def sample_read_depths(counts: pd.DataFrame) -> pd.Series:
    """Return total ASV reads by sequencing profile without changing source counts."""
    asv_cols = [col for col in counts.columns if col != "SequencingSpecificName"]
    values = counts.set_index("SequencingSpecificName")[asv_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    depth = values.sum(axis=1)
    depth.index = depth.index.astype(str)
    return depth.rename("selection_total_reads")


def nearest_stool_pairs(
    ded: pd.DataFrame,
    mapping: pd.DataFrame,
    family: pd.DataFrame,
    sample_depths: pd.Series | None = None,
    tie_rule: str = "source_order",
) -> pd.DataFrame:
    """Select the nearest physical stool and resolve tied sequencing profiles deterministically.

    The production rule selects the highest-read-depth nonexcluded sequencing profile among
    profiles tied on the nearest physical collection date. Alternative rules are retained for
    outcome-independent sensitivity analyses.
    """
    if tie_rule not in {"highest_read_depth", "lowest_read_depth", "source_order"}:
        raise ValueError(f"Unsupported tie rule: {tie_rule}")
    mapping = mapping[mapping["ShouldExcludeSample"] == 0].copy()
    mapping["_source_order"] = np.arange(len(mapping), dtype=int)
    if sample_depths is not None:
        mapping["selection_total_reads"] = mapping["SequencingSpecificName"].map(sample_depths)
    elif "selection_total_reads" not in mapping.columns:
        mapping["selection_total_reads"] = np.nan
    records: list[dict[str, Any]] = []
    grouped = {pid: grp for pid, grp in mapping.groupby("PublicID", sort=False)}
    for _, row in ded.iterrows():
        stool_rows = grouped.get(row["PublicID"])
        if stool_rows is None or stool_rows.empty:
            continue
        deltas = (row["ResponseDate"] - stool_rows["microbiome_date"]).dt.days
        min_abs_delta = deltas.abs().min()
        tied = stool_rows.loc[deltas.abs() == min_abs_delta].copy()
        tied["_gap_days_signed"] = deltas.loc[tied.index]
        if tie_rule == "highest_read_depth":
            tied = tied.sort_values(
                ["selection_total_reads", "SequencingSpecificName"],
                ascending=[False, True],
                na_position="last",
                kind="stable",
            )
        elif tie_rule == "lowest_read_depth":
            tied = tied.sort_values(
                ["selection_total_reads", "SequencingSpecificName"],
                ascending=[True, True],
                na_position="last",
                kind="stable",
            )
        else:
            tied = tied.sort_values("_source_order", kind="stable")
        stool = tied.iloc[0]
        record = row.to_dict()
        record.update(
            {
                "SequencingSpecificName": stool["SequencingSpecificName"],
                "microbiome_date": stool["microbiome_date"],
                "microbiome_age": stool.get("microbiome_age"),
                "BMI": stool.get("BMI"),
                "AntibioticsPriorMonth": stool.get("AntibioticsPriorMonth"),
                "PCRPlate": stool.get("PCRPlate"),
                "ExtractionPlate": stool.get("ExtractionPlate"),
                "SequencingRunID": stool.get("SequencingRunID"),
                "SequencingLocation": stool.get("SequencingLocation"),
                "FPUniqueSampleID": stool.get("FPUniqueSampleID"),
                "selection_total_reads": stool.get("selection_total_reads"),
                "nearest_tie_count": int(len(tied)),
                "nearest_tie_rule": tie_rule,
                "gap_days_signed": int(stool["_gap_days_signed"]),
                "gap_days_abs": int(abs(stool["_gap_days_signed"])),
            }
        )
        records.append(record)
    pairs = pd.DataFrame.from_records(records)
    if pairs.empty:
        return pairs
    pairs = pairs.merge(family, on="PublicID", how="left")
    pairs["FamilyID"] = pairs["FamilyID"].fillna(pairs["PublicID"]).astype(str)
    return pairs


def compute_alpha_and_genus(counts: pd.DataFrame, taxa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    asv_cols = [col for col in counts.columns if col != "SequencingSpecificName"]
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

    aligned = taxa.set_index("asv").reindex(asv_cols)
    assigned = aligned["Genus_clean"].notna()
    aligned = aligned.loc[assigned].copy()
    genus_frames = {}
    for genus, group in aligned.groupby("Genus_clean"):
        genus_asvs = group.index.tolist()
        prevalence = (wide[genus_asvs].sum(axis=1) > 0).mean()
        if prevalence >= GENUS_PREVALENCE_THRESHOLD:
            genus_frames[f"genus__{clean_feature_name(genus)}"] = relative[genus_asvs].sum(axis=1)
    genus = pd.DataFrame(genus_frames, index=wide.index)
    genus = np.log1p(genus * 1_000_000.0)
    genus.insert(0, "SequencingSpecificName", genus.index)
    return alpha.reset_index(drop=True), genus.reset_index(drop=True)


def clean_feature_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unclassified"


def standardize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean(skipna=True)) / std


def association_screen(data: pd.DataFrame, outcome: str, features: list[str], covariates: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        cols = [outcome, feature, *covariates, "FamilyID"]
        model_data = data[cols].copy()
        model_data[outcome] = pd.to_numeric(model_data[outcome], errors="coerce")
        model_data[feature] = standardize(model_data[feature])
        model_data = model_data.dropna(subset=[outcome, feature])
        if model_data[outcome].nunique() < 2:
            continue
        y = model_data[outcome].astype(int)
        if min(y.sum(), len(y) - y.sum()) < MIN_BINARY_CASES:
            continue
        x_parts = [model_data[[feature]].rename(columns={feature: "feature_z"})]
        for cov in covariates:
            if cov not in model_data:
                continue
            if model_data[cov].dtype == object or str(model_data[cov].dtype).startswith("category"):
                x_parts.append(
                    pd.get_dummies(model_data[cov].fillna("Missing"), prefix=cov, drop_first=True).astype(float)
                )
            else:
                x_parts.append(pd.DataFrame({cov: standardize(model_data[cov])}))
        x = pd.concat(x_parts, axis=1).replace([np.inf, -np.inf], np.nan)
        keep = x.notna().all(axis=1)
        x = sm.add_constant(x.loc[keep].astype(float), has_constant="add")
        y = y.loc[keep]
        groups = model_data.loc[keep, "FamilyID"].astype(str)
        if min(y.sum(), len(y) - y.sum()) < MIN_BINARY_CASES:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": groups}
                )
            beta = float(fit.params["feature_z"])
            p_value = float(fit.pvalues["feature_z"])
            rows.append(
                {
                    "feature": feature,
                    "n": int(len(y)),
                    "cases": int(y.sum()),
                    "controls": int(len(y) - y.sum()),
                    "beta_per_sd": beta,
                    "odds_ratio_per_sd": float(math.exp(beta)),
                    "p_value": p_value,
                    "converged": True,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "feature": feature,
                    "n": int(len(y)),
                    "cases": int(y.sum()),
                    "controls": int(len(y) - y.sum()),
                    "beta_per_sd": np.nan,
                    "odds_ratio_per_sd": np.nan,
                    "p_value": np.nan,
                    "converged": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty and "q_value" not in out.columns:
        out["q_value"] = np.nan
    if not out.empty and out["p_value"].notna().any():
        mask = out["p_value"].notna()
        out.loc[mask, "q_value"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    return out.sort_values(["q_value", "p_value"], na_position="last") if not out.empty else out


def continuous_association_screen(
    data: pd.DataFrame, outcome: str, features: list[str], covariates: list[str]
) -> pd.DataFrame:
    rows = []
    for feature in features:
        cols = [outcome, feature, *covariates, "FamilyID"]
        model_data = data[cols].copy()
        model_data[outcome] = pd.to_numeric(model_data[outcome], errors="coerce")
        model_data[feature] = standardize(model_data[feature])
        model_data = model_data.dropna(subset=[outcome, feature])
        if len(model_data) < 100:
            continue
        x_parts = [model_data[[feature]].rename(columns={feature: "feature_z"})]
        for cov in covariates:
            if cov not in model_data:
                continue
            if model_data[cov].dtype == object or str(model_data[cov].dtype).startswith("category"):
                x_parts.append(
                    pd.get_dummies(model_data[cov].fillna("Missing"), prefix=cov, drop_first=True).astype(float)
                )
            else:
                x_parts.append(pd.DataFrame({cov: standardize(model_data[cov])}))
        x = pd.concat(x_parts, axis=1).replace([np.inf, -np.inf], np.nan)
        keep = x.notna().all(axis=1)
        x = sm.add_constant(x.loc[keep].astype(float), has_constant="add")
        y = model_data.loc[keep, outcome].astype(float)
        groups = model_data.loc[keep, "FamilyID"].astype(str)
        if len(y) < 100 or y.std() == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
            rows.append(
                {
                    "feature": feature,
                    "n": int(len(y)),
                    "beta_per_sd": float(fit.params["feature_z"]),
                    "p_value": float(fit.pvalues["feature_z"]),
                    "converged": True,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "feature": feature,
                    "n": int(len(y)),
                    "beta_per_sd": np.nan,
                    "p_value": np.nan,
                    "converged": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty and "q_value" not in out.columns:
        out["q_value"] = np.nan
    if not out.empty and out["p_value"].notna().any():
        mask = out["p_value"].notna()
        out.loc[mask, "q_value"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    return out.sort_values(["q_value", "p_value"], na_position="last") if not out.empty else out


def predictive_check(
    data: pd.DataFrame,
    branch_id: str,
    outcome: str,
    microbiome_features: list[str],
    covariates: list[str],
    groups: pd.Series,
) -> list[dict[str, Any]]:
    model_data = data[[outcome, *covariates, *microbiome_features]].copy()
    model_data[outcome] = pd.to_numeric(model_data[outcome], errors="coerce")
    complete_y = model_data[outcome].notna()
    model_data = model_data.loc[complete_y].copy()
    y = model_data[outcome].astype(int)
    groups = groups.loc[complete_y].astype(str)
    if y.nunique() < 2 or min(y.sum(), len(y) - y.sum()) < MIN_BINARY_CASES:
        return []

    categorical = [col for col in covariates if model_data[col].dtype == object]
    numeric_cov = [col for col in covariates if col not in categorical]
    alpha_features = [c for c in microbiome_features if c.startswith("alpha_")]
    genus_features = [c for c in microbiome_features if c.startswith("genus__")]

    configs = {
        "covariate_only": numeric_cov + categorical,
        "covariate_plus_alpha": numeric_cov + categorical + alpha_features,
        "covariate_plus_alpha_genus_all": numeric_cov + categorical + alpha_features + genus_features,
    }
    unique_groups = groups.nunique()
    if unique_groups >= 3:
        splitter = GroupKFold(n_splits=3)
        splits = splitter.split(model_data, y, groups)
        split_type = "group_kfold_family"
    else:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260609)
        splits = splitter.split(model_data, y)
        split_type = "stratified_kfold"

    cached_splits = list(splits)
    rows = []
    for model_id, columns in configs.items():
        columns = [col for col in columns if col in model_data.columns]
        cat_cols = [col for col in columns if model_data[col].dtype == object]
        num_cols = [col for col in columns if col not in cat_cols]
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ],
            remainder="drop",
        )
        clf = Pipeline(
            [
                ("preprocess", pre),
                ("model", LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")),
            ]
        )
        aucs = []
        for fold_idx, (train_idx, test_idx) in enumerate(cached_splits, start=1):
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue
            clf.fit(model_data.iloc[train_idx][columns], y_train)
            pred = clf.predict_proba(model_data.iloc[test_idx][columns])[:, 1]
            aucs.append(float(roc_auc_score(y_test, pred)))
            rows.append(
                {
                    "branch_id": branch_id,
                    "outcome": outcome,
                    "model_id": model_id,
                    "split_type": split_type,
                    "fold": fold_idx,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_cases": int(y_test.sum()),
                    "test_controls": int(len(y_test) - y_test.sum()),
                    "auc": float(aucs[-1]),
                    "n_features_input": int(len(columns)),
                }
            )
        if aucs:
            rows.append(
                {
                    "branch_id": branch_id,
                    "outcome": outcome,
                    "model_id": model_id,
                    "split_type": split_type,
                    "fold": "mean",
                    "n_train": "",
                    "n_test": int(len(y)),
                    "test_cases": int(y.sum()),
                    "test_controls": int(len(y) - y.sum()),
                    "auc": float(np.mean(aucs)),
                    "n_features_input": int(len(columns)),
                }
            )
    return rows


def continuous_predictive_check(
    data: pd.DataFrame,
    branch_id: str,
    outcome: str,
    microbiome_features: list[str],
    covariates: list[str],
    groups: pd.Series,
) -> list[dict[str, Any]]:
    model_data = data[[outcome, *covariates, *microbiome_features]].copy()
    model_data[outcome] = pd.to_numeric(model_data[outcome], errors="coerce")
    complete_y = model_data[outcome].notna()
    model_data = model_data.loc[complete_y].copy()
    y = model_data[outcome].astype(float)
    groups = groups.loc[complete_y].astype(str)
    if len(y) < 100 or y.std() == 0:
        return []

    categorical = [col for col in covariates if model_data[col].dtype == object]
    numeric_cov = [col for col in covariates if col not in categorical]
    alpha_features = [c for c in microbiome_features if c.startswith("alpha_")]
    genus_features = [c for c in microbiome_features if c.startswith("genus__")]
    configs = {
        "covariate_only": numeric_cov + categorical,
        "covariate_plus_alpha": numeric_cov + categorical + alpha_features,
        "covariate_plus_alpha_genus_all": numeric_cov + categorical + alpha_features + genus_features,
    }
    splitter = GroupKFold(n_splits=3) if groups.nunique() >= 3 else None
    splits = list(splitter.split(model_data, y, groups)) if splitter else list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=20260609).split(
            model_data, pd.qcut(y, q=3, duplicates="drop").astype(str)
        )
    )
    split_type = "group_kfold_family" if splitter else "stratified_score_tertile_kfold"
    rows = []
    for model_id, columns in configs.items():
        columns = [col for col in columns if col in model_data.columns]
        cat_cols = [col for col in columns if model_data[col].dtype == object]
        num_cols = [col for col in columns if col not in cat_cols]
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ],
            remainder="drop",
        )
        reg = Pipeline([("preprocess", pre), ("model", Ridge(alpha=10.0))])
        r2s = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            reg.fit(model_data.iloc[train_idx][columns], y.iloc[train_idx])
            pred = reg.predict(model_data.iloc[test_idx][columns])
            r2 = float(r2_score(y.iloc[test_idx], pred))
            r2s.append(r2)
            rows.append(
                {
                    "branch_id": branch_id,
                    "outcome": outcome,
                    "model_id": model_id,
                    "split_type": split_type,
                    "fold": fold_idx,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "r2": r2,
                    "n_features_input": int(len(columns)),
                }
            )
        rows.append(
            {
                "branch_id": branch_id,
                "outcome": outcome,
                "model_id": model_id,
                "split_type": split_type,
                "fold": "mean",
                "n_train": "",
                "n_test": int(len(y)),
                "r2": float(np.mean(r2s)),
                "n_features_input": int(len(columns)),
            }
        )
    return rows


def select_prediction_features(data: pd.DataFrame, y: pd.Series, features: list[str]) -> list[str]:
    genus_features = [col for col in features if col.startswith("genus__")]
    scores = []
    for col in genus_features:
        x = pd.to_numeric(data[col], errors="coerce")
        if x.notna().sum() == 0 or x.std(skipna=True) == 0:
            continue
        score = abs(x.corr(y, method="spearman"))
        if pd.notna(score):
            scores.append((score, col))
    return [col for _, col in sorted(scores, reverse=True)[:MAX_GENUS_FEATURES_FOR_PREDICTION]]


def run_branch(base: pd.DataFrame, branch_id: str, max_gap: int, output_dir: Path) -> tuple[BranchSummary, list[dict[str, Any]]]:
    branch = base[base["gap_days_abs"] <= max_gap].copy()
    alpha_features = [col for col in branch.columns if col.startswith("alpha_")]
    genus_features = [col for col in branch.columns if col.startswith("genus__")]
    covariates = ["age", "BMI", "AntibioticsPriorMonth", "gap_days_abs"]
    summary = BranchSummary(
        branch_id=branch_id,
        max_gap_days=max_gap,
        rows=int(len(branch)),
        unique_participants=int(branch["PublicID"].nunique()),
        median_abs_gap_days=float(branch["gap_days_abs"].median()) if not branch.empty else None,
        hs_cases=int((branch["highly_symptomatic_derived"] == 1).sum()),
        any_symptom_cases=int((branch["symptom_any_current"] == 1).sum()),
        any_symptom_controls=int((branch["symptom_any_current"] == 0).sum()),
        either_often_cases=int((branch["either_often_symptom"] == 1).sum()),
        either_often_controls=int((branch["either_often_symptom"] == 0).sum()),
        hs_all_other_controls=int((branch["highly_symptomatic_derived"] == 0).sum()),
        hs_clean_cases=int((branch["hs_vs_clean_control"] == 1).sum()),
        hs_clean_controls=int((branch["hs_vs_clean_control"] == 0).sum()),
        symptom_sum_nonmissing=int(branch["symptom_sum_0_6"].notna().sum()),
        symptom_sum_mean=float(branch["symptom_sum_0_6"].mean(skipna=True)),
        genus_features=len(genus_features),
    )

    branch_dir = output_dir / branch_id
    branch_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    continuous_prediction_rows: list[dict[str, Any]] = []
    for outcome_id, outcome_col in OUTCOMES.items():
        alpha_assoc = association_screen(branch, outcome_col, alpha_features, covariates)
        genus_assoc = association_screen(branch, outcome_col, genus_features, covariates)
        alpha_assoc.insert(0, "outcome_id", outcome_id)
        genus_assoc.insert(0, "outcome_id", outcome_id)
        alpha_assoc.to_csv(branch_dir / f"{outcome_id}_alpha_association.csv", index=False)
        genus_assoc.to_csv(branch_dir / f"{outcome_id}_genus_association.csv", index=False)
        prediction_rows.extend(
            predictive_check(
                branch,
                branch_id,
                outcome_col,
                alpha_features + genus_features,
                covariates,
                branch["FamilyID"],
            )
        )
    for outcome_id, outcome_col in CONTINUOUS_OUTCOMES.items():
        alpha_assoc = continuous_association_screen(branch, outcome_col, alpha_features, covariates)
        genus_assoc = continuous_association_screen(branch, outcome_col, genus_features, covariates)
        alpha_assoc.insert(0, "outcome_id", outcome_id)
        genus_assoc.insert(0, "outcome_id", outcome_id)
        alpha_assoc.to_csv(branch_dir / f"{outcome_id}_alpha_linear_association.csv", index=False)
        genus_assoc.to_csv(branch_dir / f"{outcome_id}_genus_linear_association.csv", index=False)
        continuous_prediction_rows.extend(
            continuous_predictive_check(
                branch,
                branch_id,
                outcome_col,
                alpha_features + genus_features,
                covariates,
                branch["FamilyID"],
            )
        )
    return summary, prediction_rows + continuous_prediction_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run initial 2013 dry-eye gut microbiome signal probes.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "initial_2013_signal_probes",
    )
    args = parser.parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ded = load_ded_2013(args.data_root)
    mapping = load_mapping(args.data_root)
    family = load_family_ids(args.data_root)
    counts = load_asv_counts(args.data_root)
    taxa = load_taxa(args.data_root)
    pairs = nearest_stool_pairs(
        ded,
        mapping,
        family,
        sample_depths=sample_read_depths(counts),
        tie_rule="highest_read_depth",
    )
    alpha, genus = compute_alpha_and_genus(counts, taxa)
    base = pairs.merge(alpha, on="SequencingSpecificName", how="inner").merge(
        genus, on="SequencingSpecificName", how="inner"
    )

    branch_summaries: list[BranchSummary] = []
    prediction_rows: list[dict[str, Any]] = []
    for branch_id, max_gap in TIMING_BRANCHES.items():
        summary, pred = run_branch(base, branch_id, max_gap, args.output_dir)
        branch_summaries.append(summary)
        prediction_rows.extend(pred)

    branch_df = pd.DataFrame([asdict(row) for row in branch_summaries])
    branch_df.to_csv(args.output_dir / "branch_summary.csv", index=False)
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(args.output_dir / "predictive_signal_metrics.csv", index=False)

    run_summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - start, 3),
        "ded_2013_rows": int(len(ded)),
        "mapping_rows_after_basic_filters": int(len(mapping[mapping["ShouldExcludeSample"] == 0])),
        "nearest_pair_rows": int(len(pairs)),
        "base_rows_with_microbiome_features": int(len(base)),
        "alpha_features": 3,
        "genus_prevalence_threshold": GENUS_PREVALENCE_THRESHOLD,
        "genus_features": int(len([col for col in base.columns if col.startswith("genus__")])),
        "branch_summaries": [asdict(row) for row in branch_summaries],
        "output_dir": str(args.output_dir),
        "interpretation_guardrail": "Exploratory signal probes only; no method lock or final inference.",
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
