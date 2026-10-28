# Prepared input contract

The portable workflow reads three comma-separated files from the explicit input directory. Treat approved-study inputs and their derived results as private. The supplied example files are entirely artificial.

## cohort.csv

One row represents one selected participant and stool sample. Participant and sequencing identifiers must be unique. A family may have one or two rows. The two frequency questions use zero for never, one for sometimes, two for often and three for constantly.

| Column | Required values or meaning |
| --- | --- |
| `PublicID` | Participant identifier |
| `FamilyID` | Family identifier shared by related twins |
| `SequencingSpecificName` | Selected stool sample identifier |
| `How often dry eyes` | Integer 0 to 3 |
| `How often irritated eyes` | Integer 0 to 3 |
| `Symptoms past 3 months` | Recent symptom indicator, 0 or 1 |
| `Ever dx DED` | Ever diagnosed dry eye indicator, 0 or 1 |
| `age` | Age in years at questionnaire |
| `BMI` | Body mass index, kilograms per square metre |
| `SEX` | `F` or `M` |
| `ACTUAL_ZYGOSITY` | `MZ` for identical twins, `DZ` for non-identical twins, or `Unknown` |
| `AntibioticsPriorMonth` | `N`, `Y` or `Unknown` |
| `gap_days_signed` | Questionnaire date minus stool date, in days |
| `gap_days_abs` | Absolute value of that interval, at most 1,095 days |
| `ibs_doctor_diagnosed` | Optional doctor-diagnosed irritable bowel syndrome, 0, 1 or missing |

`ResponseDate` and `microbiome_date` may be retained as context. The workflow derives the symptom outcomes and microbial diversity using the preserved study functions. If the optional bowel-diagnosis column is absent, the corresponding sensitivity is recorded as skipped.

## asv_counts.csv

The first column is `SequencingSpecificName`. All remaining columns contain non-negative integer counts and are named by amplicon sequence variant (ASV), a microbial sequence type. Sample identifiers must match `cohort.csv` exactly. Each sample must have a positive total count. Do not rarefy or rescale this input before running the study richness calculation.

## taxonomy.csv

Supply one row per ASV, with columns `asv` and `Genus_clean`. Every count-table ASV must have a matching taxonomy row. `Genus_clean` supplies the cleaned genus name for genus aggregation. The original taxonomy contains additional rank columns that may be retained.

## Original source preparation

`prepare --data-root ... --output-dir ...` adapts the original source layout. Required relative locations are recorded in the preserved loaders: the 2013 questionnaire under `DED_outcomes_excel`, the microbiome directory under `New March 2022/Extracted/16S_Microbiome_Gut_ASVs`, and the participant-family spine under `derived`. This route requires the approved original files and their schema. Public-release testing covered source inspection; execution with protected source data remains pending.

Preparation selects the nearest eligible stool sample and resolves tied sample dates using read depth. It writes a new prepared folder and records provenance without embedding the user's absolute source path. It rejects destinations within the raw source tree or the public package. Use a private output directory separate from both.

## Private genus-method inputs

`prepare-genus --input-dir ... --output-dir ...` exports count and metadata files for the two native R scripts. It replaces sample and family identifiers consistently and retains a fixed genus universe across the three outcomes. The prevalence rule requires detection in at least 10% of the complete-case primary cohort. The study retained 142 genera; another input dataset has its own count under that rule. These exported counts and participant covariates require the same access protection as the prepared source files.
