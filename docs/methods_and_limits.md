# Methods and verification limits

The primary cohort pairs the 2013 dry-eye questionnaire with the nearest stool profile within 1,095 days, using read depth to resolve a sample-date tie. The complete analysis contains 1,685 participants in 1,125 families. The highly symptomatic outcome requires both dryness and irritation often or constantly. The combined frequency score sums the two questions and ranges from zero to six.

Primary regression adjusts for age, body mass index, antibiotic use and the absolute questionnaire-to-stool interval. Sex is included when estimable. Family-clustered standard errors account for related participants. The final multiplicity correction jointly covers eight retained tests: two diversity measures across three binary outcomes, the continuous symptom score and the diagnosis-inclusive richness association. The six-test correction is retained as a historical intermediate. The strict-control score analysis includes strictly asymptomatic participants and participants with scores one to six, totalling 1,678 in the study. Its separate score comparisons share the richness standard deviation from that analysis sample. Twin models use the richness scale from the 1,685-person primary cohort.

## Community procedures

The Aitchison profile test permutes covariate-adjusted microbial profile residuals and removes covariate effects again. The distance tests hold the centered distance matrix fixed, permute the covariate-adjusted symptom indicator and adjust that indicator again. These are different implemented procedures. Both exchange complete families only among equally sized families and change within-family order. They do not restrict exchanges by identical versus non-identical twin status.

The historical Aitchison source selects genera detected in at least 10% of all eligible within-three-year participants before each outcome's complete-case selection. The genus-method and distance-test inputs select their fixed universe from the complete primary cohort. Both historical branches report 142 genera. Their exact set identity requires a protected-data comparison. The public runner preserves each source rule. Confirm the exact feature sets with approved data before claiming a common universe across these branches.

Such exchanges assume that the residual distributions are compatible across exchanged families. Family size alone does not verify this assumption. The validity of the outcome-residual distance procedure and the family-exchange restrictions remains under assessment. Saved checks establish computational consistency and agreement in significance classification with an alternate seed. They do not measure false-positive rates under realistic null data.

The preserved residualization helper can select an incorrect adjustment space when covariates are redundant. The portable runner checks for independent covariate columns and rejects a deficient design before calling it. This hypothetical-input failure has not established that the original participant design was affected. Finite-data, residual-variation and residual-degrees-of-freedom requirements also apply. The historical permutation functions do not return the count of finite permutation statistics, which limits retrospective checking of numerical edge cases.

The final manuscript's ecological-context table reports the read-depth-adjusted Jaccard and Bray-Curtis sensitivities. Historical outputs using other adjustment sets are distinct analyses. Consult `reproduction_matrix.md` for the current result scope.

## Twin comparisons

The matched symptom-score models compare population, within-pair and between-pair associations in the same complete-pair sample. Their covariance is used for the within-minus-between comparison. Highly symptomatic binary comparisons use discordant pairs. The adjusted binary models split by twin type have invalid-fit flags because a covariate is informed by only one discordant pair in each subgroup; preserve these flags.

The working Gaussian covariance model partitions richness-score covariance into additive-genetic, shared-environmental and individual-specific components. It assumes cross-twin genetic correlations of one for identical twins and one-half for non-identical twins. Bootstrap resampling samples whole pairs within twin type. Its algebra checks cover analytic gradients, recovery of known covariance matrices and reversal of twin order. The saved 499 bootstrap fits passed the recorded computational criterion. The component intervals include zero, leaving percentage partitioning too uncertain to report.

## Environments and reproduction

The public Python requirements describe the primary-analysis environment. The later twin run used Python 3.10.12, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1 and statsmodels 0.14.2. Floating-point differences across environments may occur. The public verification file records the actual environment used for this release's checks.

Original genus-method records used R 4.5.3, ANCOMBC 2.12.1, MicrobiomeStat 1.4 and ALDEx2 1.40.0. The formal sensitivity runner also loads maaslin3, jsonlite and knitr when the MaAsLin3 analysis is omitted. Use explicit input and output arguments and one worker. The ANCOM-BC2 script retains the named degrees-of-freedom alignment fix. R package execution and production resampling require their own reproduction checks with approved inputs.
