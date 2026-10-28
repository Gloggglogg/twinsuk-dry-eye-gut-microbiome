"""Check the public archive's source integrity and aggregate reference scope."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_source_manifest_matches_files(self):
        manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
        count = 0
        for record in manifest["source_files"]:
            path = ROOT / record["public"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["public_sha256"])
            if path.suffix == ".py":
                tree = ast.parse(path.read_text(encoding="utf-8"))
                defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                for name, entry in record["functions"].items():
                    fingerprint = hashlib.sha256(ast.dump(defs[name], include_attributes=False).encode()).hexdigest()
                    self.assertEqual(fingerprint, entry["source_ast_sha256"])
                    count += 1
        self.assertEqual(count, 195)

    def test_aggregate_provenance(self):
        root = ROOT / "published_results"
        manifest = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["participant_data_read"])
        for record in manifest["files"]:
            self.assertEqual(hashlib.sha256((root / record["public"]).read_bytes()).hexdigest(), record["public_sha256"])
        coeffs = pd.read_csv(root / "model_coefficients.csv")
        self.assertEqual(len(coeffs), 159)
        self.assertEqual(coeffs.model_id.nunique(), 18)
        self.assertEqual(len(pd.read_csv(root / "genus_method_results.csv")), 1278)
        for file in root.rglob("*.csv"):
            columns = pd.read_csv(file, nrows=0).columns
            self.assertFalse(set(columns) & {"PublicID", "FamilyID", "SequencingSpecificName", "ResponseDate"})

    def test_primary_and_display_reference_estimates(self):
        coeffs = pd.read_csv(ROOT / "published_results/model_coefficients.csv")
        primary = coeffs.loc[coeffs.model_id.eq("primary_alpha_richness") & coeffs.term.eq("feature_z")].iloc[0]
        self.assertAlmostEqual(primary.odds_ratio, .7206689313902943, places=12)
        display = pd.read_csv(ROOT / "published_results/grouped_score_display.csv")
        self.assertEqual(display.cases.astype(int).tolist(), [486, 496, 74, 97])
        self.assertTrue(display.controls.eq(525).all())
        self.assertAlmostEqual(display.iloc[-1].odds_ratio_per_global_richness_sd, .6854054527073973, places=12)


if __name__ == "__main__":
    unittest.main()
