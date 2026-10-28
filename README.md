# TwinsUK dry-eye gut-microbiome analysis code

This software accompanies **Lower gut microbiome richness is associated with more frequent dry eye symptoms: a twin study**. It preserves the implementations used for manuscript version 49 and provides a portable runner, synthetic example inputs and aggregate reference results.

The analysis relates stool microbial richness to ocular dryness and irritation in the TwinsUK cohort. It accounts for related participants through family-clustered standard errors and includes paired-twin comparisons. Richness is the number of distinct amplicon sequence variants (ASVs), which are microbial sequence types detected in a sample.

The workflow starts from the provider's processed sequence-count and taxonomy tables. Upstream sequence processing requires its own source data and documentation.

## Start with the synthetic example

Use Python 3.11 and the dependencies in `requirements.txt`. From this directory:

```sh
python -m pip install -r requirements.txt
python scripts/run_pipeline.py demo --output-dir runtime/demo
python scripts/render_published_results.py --output-dir runtime/published
python -m unittest discover -s tests -v
```

The demonstration generates artificial participants and microbial counts. Its estimates demonstrate execution and have no biomedical interpretation. The figure command redraws the manuscript's grouped symptom-score display from the saved aggregate estimates. It performs no model fitting.

## Reproduce analyses with approved data

TwinsUK participant records and microbial count tables require approved access from [TwinsUK](https://twinsuk.ac.uk/resources-for-researchers/). Arrange access with the data custodian and follow the applicable agreement. The package contains no study participant records. Synthetic identifiers begin with `SYN`.

```sh
python scripts/run_pipeline.py prepare --data-root /private/approved-source --output-dir /private/prepared
python scripts/run_pipeline.py run --input-dir /private/prepared --output-dir /private/results
```

The `prepare` command expects the original documented source layout and performs the study's questionnaire-to-stool pairing. Alternatively, supply the prepared files described in [the input contract](docs/data_contract.md). Store all prepared files, genus-method inputs and new participant-derived results in a private location. An ordinary `run` performs the lightweight models. Expensive community permutations and twin bootstrap resampling require explicit options and `--allow-heavy`; see `--help` and [the reproduction map](docs/reproduction_matrix.md).

The two scripts in `r/` retain the genus-level methods. They require the recorded R packages and private inputs produced by the separate `prepare-genus` command. The public release does not automatically install R packages or launch expensive resampling.

```sh
python scripts/run_pipeline.py prepare-genus --input-dir /private/prepared --output-dir /private/genus
Rscript r/run_2013_ancombc2_analysis.R --input-dir /private/genus --output-dir /private/ancombc2
Rscript r/run_2013_formal_microbiome_package_sensitivity.R --input-dir /private/genus --output-dir /private/formal --workers 1 --aldex-mc-samples 128 --seed-base 20260814 --omit-maaslin3-by-decision
```

The final option records the study decision to omit MaAsLin3 as an analysis. The preserved R runner still loads its package as a dependency. Genus inputs remain participant-level data after identifier replacement and must remain private.

## What the release establishes

`analysis_sources/` contains the original study functions. The export changes module-level paths and disables historical standalone launchers. The source manifest records each change and matching fingerprints for all 195 Python functions and classes. The runner calls selected retained functions and guards its inputs. Some preserved historical functions belong to dependencies or earlier analyses; their presence does not make them part of the final paper.

`published_results/` contains the manuscript's aggregate tables, 159 coefficients from 18 regression models, and 1,278 genus-method result rows. These are saved reference results. The public synthetic demonstration does not independently reproduce the participant-data estimates. The full production permutation, bootstrap and R jobs require approved inputs and the recorded environments.

The exploratory community tests use custom family-preserving permutation procedures. Their statistical assumptions still require assessment before submission, including whether exchanging equally sized families adequately preserves the residual distribution. False-positive calibration requires a separate statistical assessment. The public runner rejects redundant community covariates because the preserved residualization helper expects a full-rank design. Read [the methods and verification limits](docs/methods_and_limits.md) before reusing these procedures.

## Files and citation

| Location | Contents |
| --- | --- |
| `analysis_sources/` | Preserved Python analysis functions |
| `src/twinsuk_microbiome/` | Portable input, execution and synthetic-data adapters |
| `r/` | ANCOM-BC2, LinDA and ALDEx2 genus analyses |
| `example_data/` | Artificial inputs with a recorded generator seed |
| `published_results/` | Aggregate study reference results and provenance |
| `tests/` | Source-integrity, input and execution checks |
| `source_manifest.json` | Source fingerprints and exact export changes |

Use `CITATION.cff` to cite this software. Software authorship records the code contributor; the manuscript's scientific author list is separate. The software is released under the MIT license. Dependency licenses continue to apply to their respective packages.
