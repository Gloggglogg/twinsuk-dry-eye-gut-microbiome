from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from run_2013_locked_primary_16s_analysis import (
    CORE_COVARIATES,
    binary_association,
    sex_is_estimable,
)
from run_2013_publishability_sensitivities import primary_complete_cases


CANONICAL_ROOT = Path(__file__).resolve().parents[1] / 'authorized_inputs' / 'canonical_questionnaires'
QUESTION_DICTIONARY = CANONICAL_ROOT / "question_dictionary.csv"
RESPONSES_LONG = CANONICAL_ROOT / "responses_long.csv"
MISSING_DICTIONARY = CANONICAL_ROOT / "codebooks" / "global_missing_code_dictionary.csv"
PRIMARY_OUTCOME = "highly_symptomatic_derived"
PRIMARY_FEATURE = "alpha_richness"
PRIMARY_MAX_GAP_DAYS = 365 * 3
TARGET_IBS_QUESTION_ID = "d52b975fec600d3c1a41"


CANDIDATE_PATTERNS = {
    "stool_consistency_or_transit": re.compile(
        r"bristol|stool|bowel movement|bowel habit|constipat|diarrh|defecat|intestinal transit",
        re.I,
    ),
    "diet": re.compile(
        r"what is (?:your|the) main diet|main diet that best describes|types of food and drink do you consume|fibre supplements?",
        re.I,
    ),
    "probiotics_or_fermented_food": re.compile(
        r"probiotic|live food products?|fermented food",
        re.I,
    ),
    "gastrointestinal_disease": re.compile(
        r"irritable bowel|inflammatory bowel|ulcerative colitis|crohn|coeliac|celiac|diverticulitis|gastroesophageal reflux|bowel / intestine disease",
        re.I,
    ),
    "autoimmune_disease_or_sjogren": re.compile(
        r"sj.?gren|rheumatoid arthritis|systemic lupus|\blupus\b|autoimmune",
        re.I,
    ),
    "proton_pump_inhibitor": re.compile(
        r"proton[- ]pump|omepraz|lansopraz|pantopraz|esomepraz|rabepraz|type of drug\\antacids",
        re.I,
    ),
    "metformin": re.compile(r"metformin|type of drug\\diabetic tablets", re.I),
    "laxatives": re.compile(r"laxative", re.I),
    "antidepressants": re.compile(r"anti[- ]?depressants?", re.I),
    "hormone_replacement_therapy": re.compile(
        r"hormone replacement|\bHRT\b|female hormones?|including hormone treatments",
        re.I,
    ),
    "ocular_treatment": re.compile(
        r"eye drops|drops prescribed or other treatment for dry eye|treatment for dry eyes",
        re.I,
    ),
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def build_main_cohort(data_root: Path) -> pd.DataFrame:
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
    within_3y = base[base["gap_days_abs"] <= PRIMARY_MAX_GAP_DAYS].copy()
    covariates = [*CORE_COVARIATES, *(["SEX"] if sex_is_estimable(within_3y) else [])]
    complete = primary_complete_cases(within_3y, covariates)
    keep = [
        "PublicID",
        "ResponseDate",
        "microbiome_date",
        PRIMARY_OUTCOME,
        PRIMARY_FEATURE,
        "FamilyID",
        *covariates,
    ]
    return complete[keep].rename(columns={"PublicID": "public_id"}).copy()


def fit_targeted_ibs_sensitivity(
    responses: pd.DataFrame,
    cohort: pd.DataFrame,
    missing_codes: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = responses[responses["question_id"] == TARGET_IBS_QUESTION_ID].copy()
    data = data[meaningful_mask(data, missing_codes)].copy()
    data["ibs_doctor_diagnosed"] = pd.to_numeric(data["raw_response"], errors="coerce")
    data = data[data["ibs_doctor_diagnosed"].isin([0, 1])].copy()
    data["response_date"] = pd.to_datetime(data["response_date"], errors="coerce")
    data = data.merge(
        cohort[["public_id", "ResponseDate"]],
        on="public_id",
        how="inner",
        suffixes=("", "_cohort"),
    )
    data["abs_days_to_2013_questionnaire"] = (
        data["response_date"] - data["ResponseDate"]
    ).abs().dt.days
    value_counts = data.groupby("public_id")["ibs_doctor_diagnosed"].nunique()
    conflicting_participants = int((value_counts > 1).sum())
    data = data.sort_values(
        ["public_id", "abs_days_to_2013_questionnaire"],
        na_position="last",
    ).drop_duplicates("public_id", keep="first")
    model = cohort.merge(
        data[["public_id", "ibs_doctor_diagnosed"]],
        on="public_id",
        how="inner",
    )
    include_sex = sex_is_estimable(model)
    core_covariates = [*CORE_COVARIATES, *(["SEX"] if include_sex else [])]
    rows: list[dict[str, Any]] = []
    for label, covariates in [
        ("same_sample_primary_core", core_covariates),
        ("same_sample_primary_core_plus_doctor_diagnosed_ibs", [*core_covariates, "ibs_doctor_diagnosed"]),
    ]:
        result = binary_association(model, PRIMARY_OUTCOME, PRIMARY_FEATURE, covariates)
        if not result:
            raise RuntimeError(f"Targeted IBS sensitivity did not fit: {label}")
        rows.append({"model": label, "covariates": " | ".join(covariates), **result})
    diagnostics = {
        "question_id": TARGET_IBS_QUESTION_ID,
        "question_text": "Has a doctor EVER diagnosed or treated you for Irritable Bowel Syndrome (IBS)?",
        "coding": {"0": "No", "1": "Yes"},
        "model_n": int(len(model)),
        "model_cases": int(model[PRIMARY_OUTCOME].sum()),
        "ibs_yes": int(model["ibs_doctor_diagnosed"].sum()),
        "ibs_yes_cases": int(
            model.loc[model[PRIMARY_OUTCOME] == 1, "ibs_doctor_diagnosed"].sum()
        ),
        "conflicting_participants_before_nearest_date_selection": conflicting_participants,
        "selection_rule": "nearest dated response to the 2013 dry-eye questionnaire; ties retain first canonical row",
        "role": "post hoc sensitivity covariate; not a replacement for the locked primary model",
    }
    return pd.DataFrame(rows), diagnostics


def select_candidate_questions(dictionary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    dictionary = dictionary.copy()
    dictionary["question_text"] = dictionary["question_text"].fillna("").astype(str)
    dictionary["question_code"] = dictionary["question_code"].fillna("").astype(str)
    dictionary["search_text"] = dictionary["question_text"] + " " + dictionary["question_code"]
    rows: list[pd.DataFrame] = []
    mapping: dict[str, set[str]] = {}
    for candidate, pattern in CANDIDATE_PATTERNS.items():
        selected = dictionary[dictionary["search_text"].str.contains(pattern, na=False)].copy()
        selected.insert(0, "candidate", candidate)
        mapping[candidate] = set(selected["question_id"].astype(str))
        rows.append(selected)
    catalog = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return catalog, mapping


def load_relevant_responses(
    question_ids: set[str],
    cohort_ids: set[str],
    chunk_size: int,
) -> tuple[pd.DataFrame, int]:
    usecols = [
        "public_id",
        "response_date",
        "date_precision",
        "date_provenance",
        "question_id",
        "raw_response",
        "response_label",
        "source_id",
        "source_file",
        "source_sheet",
    ]
    retained: list[pd.DataFrame] = []
    rows_scanned = 0
    for chunk in pd.read_csv(
        RESPONSES_LONG,
        usecols=usecols,
        dtype=str,
        chunksize=chunk_size,
        low_memory=False,
    ):
        rows_scanned += len(chunk)
        mask = chunk["question_id"].isin(question_ids) & chunk["public_id"].isin(cohort_ids)
        if mask.any():
            retained.append(chunk.loc[mask].copy())
    if not retained:
        return pd.DataFrame(columns=usecols), rows_scanned
    return pd.concat(retained, ignore_index=True), rows_scanned


def meaningful_mask(responses: pd.DataFrame, missing_codes: set[str]) -> pd.Series:
    raw = responses["raw_response"].fillna("").astype(str).str.strip()
    label = responses["response_label"].fillna("").astype(str).str.strip()
    missing_labels = re.compile(
        r"^$|no data available|not applicable|question skipped|not completed|refused|declined|don.?t know|unable to determine|undefined response",
        re.I,
    )
    raw_valid = (~raw.isin(missing_codes)) & (~raw.str.match(missing_labels))
    label_valid = ~label.str.match(missing_labels)
    return raw_valid | label_valid


def summarize_candidates(
    catalog: pd.DataFrame,
    question_map: dict[str, set[str]],
    responses: pd.DataFrame,
    cohort: pd.DataFrame,
    missing_codes: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cohort_n = len(cohort)
    cohort_cases = int(cohort[PRIMARY_OUTCOME].sum())
    for candidate in CANDIDATE_PATTERNS:
        questions = catalog[catalog["candidate"] == candidate].copy()
        ids = question_map[candidate]
        data = responses[responses["question_id"].isin(ids)].copy()
        if not data.empty:
            data = data[meaningful_mask(data, missing_codes)].copy()
        participants = set(data["public_id"].dropna().astype(str)) if not data.empty else set()
        participant_rows = cohort[cohort["public_id"].isin(participants)].copy()
        dated = data.dropna(subset=["response_date"]).copy() if not data.empty else data.copy()
        if not dated.empty:
            dated["response_date"] = pd.to_datetime(dated["response_date"], errors="coerce")
            dated = dated.dropna(subset=["response_date"])
            dated = dated.merge(cohort, on="public_id", how="inner")
            dated["abs_days_to_stool"] = (dated["response_date"] - dated["microbiome_date"]).abs().dt.days
            dated["abs_days_to_questionnaire"] = (
                dated["response_date"] - dated["ResponseDate"]
            ).abs().dt.days
            dated = dated.sort_values(["public_id", "abs_days_to_stool"]).drop_duplicates("public_id")
        source_files = sorted(questions["source_file"].dropna().astype(str).unique())
        source_ids = sorted(questions["source_id"].dropna().astype(str).unique())
        source_sheets = sorted(questions["source_sheet"].dropna().astype(str).unique())
        examples = questions["question_text"].dropna().astype(str).drop_duplicates().head(4).tolist()
        dated_n = int(dated["public_id"].nunique()) if not dated.empty else 0
        rows.append(
            {
                "candidate": candidate,
                "canonical_question_count": int(len(questions)),
                "source_ids": " | ".join(source_ids),
                "source_files": " | ".join(source_files),
                "source_sheets": " | ".join(source_sheets),
                "join_key": "public_id (canonicalized from PublicID/ParticipantPublicID)",
                "question_examples": " | ".join(examples),
                "main_cohort_n": cohort_n,
                "main_cohort_cases": cohort_cases,
                "participants_with_nonmissing_candidate_response": int(len(participants)),
                "coverage_percent": float(100 * len(participants) / cohort_n),
                "cases_with_nonmissing_candidate_response": int(participant_rows[PRIMARY_OUTCOME].sum()),
                "participants_with_dated_candidate_response": dated_n,
                "dated_coverage_percent": float(100 * dated_n / cohort_n),
                "median_abs_days_candidate_to_stool": (
                    float(dated["abs_days_to_stool"].median()) if dated_n else np.nan
                ),
                "median_abs_days_candidate_to_2013_questionnaire": (
                    float(dated["abs_days_to_questionnaire"].median()) if dated_n else np.nan
                ),
                "dated_within_1y_of_stool": (
                    int((dated["abs_days_to_stool"] <= 365).sum()) if dated_n else 0
                ),
                "dated_within_3y_of_stool": (
                    int((dated["abs_days_to_stool"] <= 365 * 3).sum()) if dated_n else 0
                ),
                "date_precision_values": (
                    " | ".join(sorted(dated["date_precision"].dropna().astype(str).unique()))
                    if dated_n
                    else ""
                ),
                "date_provenance_values": (
                    " | ".join(sorted(dated["date_provenance"].dropna().astype(str).unique()))
                    if dated_n
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_questions(
    catalog: pd.DataFrame,
    responses: pd.DataFrame,
    cohort: pd.DataFrame,
    missing_codes: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question in catalog.drop_duplicates(["candidate", "question_id"]).itertuples(index=False):
        data = responses[responses["question_id"] == question.question_id].copy()
        if not data.empty:
            data = data[meaningful_mask(data, missing_codes)].copy()
        participants = set(data["public_id"].dropna().astype(str)) if not data.empty else set()
        participant_rows = cohort[cohort["public_id"].isin(participants)]
        dated = data.dropna(subset=["response_date"]).copy() if not data.empty else data.copy()
        if not dated.empty:
            dated["response_date"] = pd.to_datetime(dated["response_date"], errors="coerce")
            dated = dated.dropna(subset=["response_date"]).merge(cohort, on="public_id", how="inner")
            dated["abs_days_to_stool"] = (dated["response_date"] - dated["microbiome_date"]).abs().dt.days
            dated["abs_days_to_questionnaire"] = (
                dated["response_date"] - dated["ResponseDate"]
            ).abs().dt.days
            dated = dated.sort_values(["public_id", "abs_days_to_stool"]).drop_duplicates("public_id")
        text = str(question.question_text)
        if re.search(r"specific name|names of your currently prescribed", text, flags=re.I):
            top_values = "suppressed_free_text"
            unique_values: int | str = "suppressed_free_text"
        else:
            value = data["response_label"].fillna("").astype(str).str.strip()
            fallback = data["raw_response"].fillna("").astype(str).str.strip()
            value = value.mask(value == "", fallback)
            counts = value.value_counts(dropna=False).head(6)
            top_values = " | ".join(f"{key}:{int(count)}" for key, count in counts.items())
            unique_values = int(value.nunique())
        rows.append(
            {
                "candidate": question.candidate,
                "question_id": question.question_id,
                "source_id": question.source_id,
                "source_sheet": question.source_sheet,
                "question_code": question.question_code,
                "question_text": text,
                "participants_with_nonmissing_response": int(len(participants)),
                "coverage_percent": float(100 * len(participants) / len(cohort)),
                "cases_with_nonmissing_response": int(participant_rows[PRIMARY_OUTCOME].sum()),
                "participants_with_dated_response": int(len(dated)),
                "median_abs_days_to_stool": (
                    float(dated["abs_days_to_stool"].median()) if len(dated) else np.nan
                ),
                "median_abs_days_to_2013_questionnaire": (
                    float(dated["abs_days_to_questionnaire"].median()) if len(dated) else np.nan
                ),
                "dated_within_1y_of_stool": (
                    int((dated["abs_days_to_stool"] <= 365).sum()) if len(dated) else 0
                ),
                "dated_within_3y_of_stool": (
                    int((dated["abs_days_to_stool"] <= 365 * 3).sum()) if len(dated) else 0
                ),
                "unique_response_values": unique_values,
                "top_response_counts": top_values,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["candidate", "participants_with_nonmissing_response", "question_id"],
        ascending=[True, False, True],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit canonical questionnaire sources for time-near microbiome confounders."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "publishability_audit_2026_08_21" / "confounder_audit",
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in [QUESTION_DICTIONARY, RESPONSES_LONG, MISSING_DICTIONARY]:
        if not path.exists():
            raise FileNotFoundError(f"Canonical questionnaire asset is missing: {path}")

    cohort = build_main_cohort(args.data_root)
    if len(cohort) != 1685 or int(cohort[PRIMARY_OUTCOME].sum()) != 93:
        raise RuntimeError(
            f"Confounder audit cohort mismatch: n={len(cohort)}, cases={int(cohort[PRIMARY_OUTCOME].sum())}"
        )
    dictionary = pd.read_csv(QUESTION_DICTIONARY, dtype=str, low_memory=False)
    catalog, question_map = select_candidate_questions(dictionary)
    catalog_columns = [
        "candidate",
        "question_id",
        "source_id",
        "source_file",
        "source_sheet",
        "questionnaire",
        "question_code",
        "source_column",
        "question_text",
        "item_category",
    ]
    catalog[catalog_columns].to_csv(args.out_dir / "canonical_candidate_question_catalog.csv", index=False)

    all_question_ids = set().union(*question_map.values())
    responses, rows_scanned = load_relevant_responses(
        all_question_ids,
        set(cohort["public_id"].astype(str)),
        args.chunk_size,
    )
    missing_codes = set(
        pd.read_csv(MISSING_DICTIONARY, dtype=str)["response_code"].dropna().astype(str)
    )
    summary = summarize_candidates(catalog, question_map, responses, cohort, missing_codes)
    summary.to_csv(args.out_dir / "canonical_time_near_confounder_audit.csv", index=False)
    question_summary = summarize_questions(catalog, responses, cohort, missing_codes)
    question_summary.to_csv(args.out_dir / "canonical_candidate_question_coverage.csv", index=False)
    ibs_results, ibs_diagnostics = fit_targeted_ibs_sensitivity(
        responses,
        cohort,
        missing_codes,
    )
    ibs_results.to_csv(args.out_dir / "canonical_ibs_adjustment_sensitivity.csv", index=False)
    write_json(args.out_dir / "canonical_ibs_adjustment_diagnostics.json", ibs_diagnostics)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"participants": len(cohort), "cases": int(cohort[PRIMARY_OUTCOME].sum())},
        "canonical_assets": {
            "question_dictionary": str(QUESTION_DICTIONARY),
            "responses_long": str(RESPONSES_LONG),
            "missing_code_dictionary": str(MISSING_DICTIONARY),
        },
        "candidate_domains": list(CANDIDATE_PATTERNS),
        "selected_question_ids": len(all_question_ids),
        "canonical_response_rows_scanned": rows_scanned,
        "canonical_response_rows_retained_in_memory": len(responses),
        "question_level_rows_written": len(question_summary),
        "targeted_ibs_sensitivity": ibs_diagnostics,
        "raw_or_external_data_written": False,
        "participant_level_output_written": False,
        "selection_note": "Candidate questions were selected by explicit text patterns before response coverage was summarized.",
        "interpretation_note": "This is a provenance, timing, and completeness audit. It does not establish valid covariate coding or authorize adjustment.",
    }
    write_json(args.out_dir / "audit_manifest.json", manifest)
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))
    return 0


# Original standalone launch disabled. Use the public workflow entry point.
