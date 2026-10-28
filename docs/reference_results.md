# Reading the aggregate reference results

The saved tables report study-level estimates. They support comparison after an approved-data run. `provenance.json` identifies each source and its checksum. The grouped-score figure command uses the four saved model estimates in `grouped_score_display.csv`.

The tables use these abbreviations:

| Label | Meaning |
| --- | --- |
| ASV | Amplicon sequence variant, a distinct microbial sequence type |
| BH | Benjamini-Hochberg adjustment of p values for multiple testing |
| BMI | Body mass index, in kilograms per square metre |
| CI | Confidence interval |
| CLR | Centered log-ratio transformation of genus counts after adding the specified pseudocount |
| DZ | Dizygotic, or non-identical, twins |
| IQR | Interquartile range, from the 25th to the 75th percentile |
| MZ | Monozygotic, or identical, twins |
| N | Number of included participants or pairs, as specified in the row |
| OR | Odds ratio |
| SD | Standard deviation |
| SE | Standard error |
| WHS | Women's Health Study questionnaire, assessing dry-eye symptoms and reported diagnosis |

The genus result file names three analysis packages: ANCOM-BC2, LinDA and ALDEx2. The `family_aware` field identifies whether the fitted method accounts for related participants. `valid_for_inference`, `analysis_status` and `fit_note` preserve method-specific interpretation limits. Read those fields together with the effect estimate and adjusted p value.
