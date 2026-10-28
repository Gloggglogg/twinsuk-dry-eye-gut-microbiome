"""Meaningful input, reproducibility and privacy checks using artificial data."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinsuk_microbiome import create_synthetic_inputs, prepare_genus_inputs, prepare_inputs, run_workflow
from twinsuk_microbiome.runtime import sha256
from twinsuk_microbiome.workflow import ID_COLUMNS, load_inputs, validate_full_rank


class PreparedInputTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "runtime")
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        create_synthetic_inputs(self.inputs)

    def tearDown(self):
        self.temporary.cleanup()

    def test_fixed_seed_and_visible_synthetic_labels(self):
        other = self.root / "same_seed"
        create_synthetic_inputs(other)
        for name in ["cohort.csv", "asv_counts.csv", "taxonomy.csv"]:
            self.assertEqual(sha256(self.inputs / name), sha256(other / name))
        cohort, counts, _, provenance = load_inputs(self.inputs)
        self.assertEqual(provenance["data_kind"], "synthetic")
        self.assertEqual((len(cohort), cohort.FamilyID.nunique()), (600, 300))
        self.assertEqual(counts.shape, (600, 120))
        self.assertTrue(all(cohort[c].str.startswith("SYN_").all() for c in ID_COLUMNS))

    def test_missing_schema_and_misaligned_samples_fail(self):
        path = self.inputs / "cohort.csv"
        cohort = pd.read_csv(path)
        cohort.drop(columns="BMI").to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Missing cohort columns"):
            load_inputs(self.inputs)
        cohort.loc[0, "SequencingSpecificName"] = "SYN_missing_sample"
        cohort.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "sample identifiers must match"):
            load_inputs(self.inputs)

    def test_fractional_or_missing_counts_fail(self):
        path = self.inputs / "asv_counts.csv"
        counts = pd.read_csv(path)
        counts.iloc[0, 1] = .5
        counts.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            load_inputs(self.inputs)

    def test_synthetic_marker_cannot_label_other_identifiers(self):
        path = self.inputs / "cohort.csv"
        cohort = pd.read_csv(path)
        cohort.loc[0, "PublicID"] = "unlabelled_identifier"
        cohort.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "visibly synthetic"):
            load_inputs(self.inputs)

    def test_input_count_order_is_preserved_for_seeded_models(self):
        path = self.inputs / "asv_counts.csv"
        counts = pd.read_csv(path).iloc[::-1].copy()
        counts.to_csv(path, index=False)
        cohort, wide, _, _ = load_inputs(self.inputs)
        self.assertEqual(list(wide.index), list(counts.SequencingSpecificName))
        self.assertNotEqual(list(wide.index), list(cohort.SequencingSpecificName))

    def test_redundant_covariate_guard(self):
        validate_full_rank(np.column_stack([np.ones(8), np.arange(8.)]))
        with self.assertRaisesRegex(ValueError, "redundant columns"):
            validate_full_rank(np.column_stack([np.ones(8), np.ones(8), np.arange(8.)]))

    def test_heavy_execution_and_private_prepare_guards(self):
        with self.assertRaisesRegex(ValueError, "allow_heavy"):
            run_workflow(self.inputs, self.root / "output", community_permutations=1999)
        with self.assertRaisesRegex(ValueError, "allow_heavy"):
            run_workflow(self.inputs, self.root / "output", twin_bootstrap=499)
        with self.assertRaisesRegex(ValueError, "read-only raw source"):
            prepare_inputs(self.inputs, self.inputs / "derived")
        with self.assertRaisesRegex(ValueError, "public release"):
            prepare_inputs(self.root / "pretend_raw", ROOT / "runtime" / "private_should_fail")
        with self.assertRaisesRegex(ValueError, "separate"):
            run_workflow(self.inputs, self.inputs / "results")
        with self.assertRaisesRegex(ValueError, "separate"):
            run_workflow(self.inputs, self.root)

    def test_private_genus_export_preserves_universe_and_alignment(self):
        output = self.root / "private_genus"
        result = prepare_genus_inputs(self.inputs, output)
        self.assertTrue(result["contains_participant_level_data"])
        self.assertFalse(result["r_models_executed"])
        taxa = []
        for entry in result["outcomes"]:
            counts = pd.read_csv(output / entry["count_file"])
            metadata = pd.read_csv(output / entry["metadata_file"])
            self.assertEqual(list(counts.columns[1:]), list(metadata.sample_id))
            self.assertTrue(metadata.FamilyID.str.startswith("R_FAMILY").all())
            taxa.append(set(counts.taxon))
        self.assertTrue(all(value == taxa[0] for value in taxa))

    def test_insufficient_case_families_fail_before_output(self):
        path = self.inputs / "cohort.csv"
        cohort = pd.read_csv(path)
        cohort["How often dry eyes"] = 0
        cohort["How often irritated eyes"] = 0
        cohort.to_csv(path, index=False)
        output = self.root / "result"
        with self.assertRaisesRegex(ValueError, "at least 20"):
            run_workflow(self.inputs, output)
        self.assertFalse(output.exists())


class EndToEndTests(unittest.TestCase):
    def test_aggregate_models_and_repeated_run_reproducibility(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime") as temporary:
            work = Path(temporary)
            inputs = work / "input"
            create_synthetic_inputs(inputs)
            outputs = [work / "first", work / "second"]
            summaries = [run_workflow(inputs, destination) for destination in outputs]
            for summary in summaries:
                self.assertEqual(summary["participants"], 600)
                self.assertEqual(summary["families"], 300)
                self.assertFalse(summary["participant_level_output_written"])
                self.assertFalse(summary["frozen_reference_results_used"])
                self.assertEqual(summary["data_kind"], "synthetic")
            alpha = pd.read_csv(outputs[0] / "primary_alpha_models.csv")
            self.assertEqual(len(alpha), 6)
            self.assertTrue(np.isfinite(alpha[["beta_per_sd", "standard_error", "p_value", "q_value_alpha_eight", "q_value_alpha_six_historical"]]).all().all())
            eight = pd.read_csv(outputs[0] / "alpha_eight_multiplicity_sensitivity.csv")
            self.assertEqual(len(eight), 8)
            self.assertTrue(eight.q_value_alpha_eight.between(0, 1).all())
            active = pd.read_csv(outputs[0] / "detection_threshold_active_scope.csv")
            self.assertEqual(set(active.minimum_reads_per_variant), {2, 5, 10})
            self.assertEqual(set(active.multiplicity_family_size), {3})
            for first in outputs[0].iterdir():
                second = outputs[1] / first.name
                self.assertTrue(second.is_file())
                self.assertEqual(first.read_bytes(), second.read_bytes(), first.name)
                if first.suffix == ".csv":
                    frame = pd.read_csv(first)
                    self.assertFalse(set(ID_COLUMNS + ["family_code", "sample_id"]) & set(frame.columns))
                    self.assertEqual(set(frame.result_origin), {"recomputed_from_synthetic_inputs"})
                    self.assertNotIn("SYN_P", first.read_text(encoding="utf-8"))
                else:
                    json.loads(first.read_text(encoding="utf-8"))


if __name__ == "__main__":
    (ROOT / "runtime").mkdir(exist_ok=True)
    unittest.main()
