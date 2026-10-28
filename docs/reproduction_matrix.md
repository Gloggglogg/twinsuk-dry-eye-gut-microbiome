# Reproduction map

The package separates new fits on supplied data from saved study reference results. `demo` fits artificial data. `run` fits the approved prepared inputs supplied by the user. Neither command silently opens a study dataset.

| Article component | Implementation in `analysis_sources/` | Public route and reference |
| --- | --- | --- |
| Questionnaire definitions, nearest stool pairing, diversity | `run_2013_initial_microbiome_signal_probes.py` | `prepare`, followed by `run`; prepared input contract |
| Primary richness and Shannon regression | `run_2013_locked_primary_16s_analysis.py` | Lightweight runner; final eight-test adjustment, historical six-test intermediate, coefficient reference and Table S1a |
| Continuous score, separate categories and strict controls | `run_2013_symptom_dose_response.py` | Lightweight runner; 18-model coefficient reference |
| Scores 4 to 6 pooled | `run_2013_sparse_tail_pooling_followup.py` | Lightweight runner; saved grouped-score display inputs |
| Final eight-test correction including diagnosis-inclusive richness and continuous score | `run_whs_ded_richness_sensitivity.py` | Lightweight runner; Tables S1a and S1b |
| Timing and read-depth sensitivities | `run_2013_publishability_sensitivities.py`, `audit_2013_time_near_confounders.py` | Lightweight runner; Table S2 |
| Known antibiotic-use status and bowel-syndrome sensitivity | `audit_antibiotic_metadata_assumption.py`, `audit_2013_time_near_confounders.py` | Lightweight runner; bowel diagnosis is skipped if unavailable |
| Matched score, binary and covariance twin models | `run_2013_twin_family_signal_revision.py`, `run_2013_twin_relatedness_extension.py` | Lightweight paired comparisons; covariance bootstrap opt-in; Table S5 |
| Aitchison profile and distance community tests | `run_2013_method_strengthening_batch.py`, `run_2013_microbiome_pipeline_gap_analysis.py` | Permutations opt-in; Table S3 gives final read-depth-adjusted distance sensitivities |
| Detection thresholds | `build_2013_detection_threshold_active_scope.py` | Preserved 2-, 5- and 10-read sensitivity implementation |
| Fixed genus universe and formal genus tests | `build_2013_ancombc2_inputs.py`, the two scripts in `r/` | Prepared private R inputs; explicit R jobs; 1,278 aggregate rows |
| Coefficient export | `export_v42_model_coefficients.py` | Saved 159 coefficients from 18 retained models |
| Manuscript tables and grouped-score figure | `published_results/`, `scripts/render_published_results.py` | Readable saved tables and aggregate-only figure regeneration |

The source manifest identifies dependency functions and historical entry points retained for provenance. Original launchers contain fixed study-size assertions and some superseded analyses. Use the portable entry point for a new run. Inspect `--help` for expensive-run arguments. Production settings were 1,999 community permutations and 499 twin covariance bootstrap repetitions; those counts require explicit opt-in.

All reference tables are from manuscript version 49. Table S3's read-depth-adjusted square-root Jaccard result has p=0.022 and multiplicity-adjusted q=0.045; the corresponding Bray-Curtis result has p=0.225. Earlier base-covariate analyses have different values. The package preserves their source without treating those historical estimates as the final table.
