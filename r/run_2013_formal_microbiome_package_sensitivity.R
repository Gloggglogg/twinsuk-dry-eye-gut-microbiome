options(stringsAsFactors = FALSE)

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    input_dir = "authorized_inputs/genus",
    output_dir = "runtime/run_2013_formal_microbiome_package_sensitivity",
    outcomes = c("either_often_symptom_or", "highly_symptomatic_and", "hs_vs_clean_control"),
    smoke = FALSE,
    max_taxa = 8,
    aldex_mc_samples = 128,
    workers = 4,
    seed_base = 20260814,
    skip_maaslin3 = FALSE,
    maaslin3_omitted_by_decision = FALSE,
    skip_linda = FALSE,
    skip_aldex2 = FALSE
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--input-dir") {
      i <- i + 1; out$input_dir <- args[[i]]
    } else if (key == "--output-dir") {
      i <- i + 1; out$output_dir <- args[[i]]
    } else if (key == "--outcomes") {
      i <- i + 1; out$outcomes <- strsplit(args[[i]], ",", fixed = TRUE)[[1]]
    } else if (key == "--smoke") {
      out$smoke <- TRUE
    } else if (key == "--max-taxa") {
      i <- i + 1; out$max_taxa <- as.integer(args[[i]])
    } else if (key == "--aldex-mc-samples") {
      i <- i + 1; out$aldex_mc_samples <- as.integer(args[[i]])
    } else if (key == "--workers") {
      i <- i + 1; out$workers <- as.integer(args[[i]])
    } else if (key == "--seed-base") {
      i <- i + 1; out$seed_base <- as.integer(args[[i]])
    } else if (key == "--skip-maaslin3") {
      out$skip_maaslin3 <- TRUE
    } else if (key == "--omit-maaslin3-by-decision") {
      out$skip_maaslin3 <- TRUE
      out$maaslin3_omitted_by_decision <- TRUE
    } else if (key == "--skip-linda") {
      out$skip_linda <- TRUE
    } else if (key == "--skip-aldex2") {
      out$skip_aldex2 <- TRUE
    } else {
      stop("Unknown argument: ", key)
    }
    i <- i + 1
  }
  out
}

write_json <- function(x, path) {
  jsonlite::write_json(x, path, auto_unbox = TRUE, pretty = TRUE, null = "null")
}

write_table <- function(x, path) {
  write.csv(x, paste0(path, ".csv"), row.names = FALSE, na = "")
  writeLines(capture.output(knitr::kable(x, format = "pipe")), paste0(path, ".md"))
}

read_inputs <- function(input_dir, outcome, smoke, max_taxa) {
  outcome_dir <- file.path(input_dir, outcome)
  counts <- read.csv(file.path(outcome_dir, "genus_counts.csv"), check.names = FALSE)
  rownames(counts) <- counts$taxon
  counts$taxon <- NULL
  counts <- as.matrix(counts)
  storage.mode(counts) <- "numeric"
  if (smoke) {
    counts <- counts[seq_len(min(max_taxa, nrow(counts))), , drop = FALSE]
  }
  meta <- read.csv(file.path(outcome_dir, "metadata.csv"), check.names = FALSE)
  meta <- meta[match(colnames(counts), meta$sample_id), , drop = FALSE]
  if (any(is.na(meta$sample_id))) {
    stop("Metadata is missing sample IDs for outcome: ", outcome)
  }
  rownames(meta) <- meta$sample_id
  meta$dry_eye_status <- factor(meta$dry_eye_status, levels = c("Control", "Case"))
  meta$AntibioticsPriorMonth <- factor(meta$AntibioticsPriorMonth)
  meta$FamilyID <- factor(meta$FamilyID)
  if ("SEX" %in% names(meta)) meta$SEX <- factor(meta$SEX)
  meta$age <- as.numeric(meta$age)
  meta$BMI <- as.numeric(meta$BMI)
  meta$gap_days_abs <- as.numeric(meta$gap_days_abs)
  meta$total_reads <- as.numeric(meta$total_reads)
  meta$log_total_reads <- log1p(meta$total_reads)
  keep <- complete.cases(meta[, c("dry_eye_status", "age", "BMI", "gap_days_abs", "AntibioticsPriorMonth", "FamilyID", "SEX", "log_total_reads")])
  meta <- meta[keep, , drop = FALSE]
  counts <- counts[, meta$sample_id, drop = FALSE]
  list(counts = counts, metadata = meta)
}

case_rows_maaslin <- function(path, outcome) {
  dat <- read.delim(path, check.names = FALSE)
  if (!nrow(dat)) return(data.frame())
  dat <- dat[dat$metadata == "dry_eye_status" & dat$value == "Case", , drop = FALSE]
  if (!nrow(dat)) return(data.frame())
  dat$outcome_id <- outcome
  dat$package <- "maaslin3"
  if (!"error" %in% names(dat)) dat$error <- NA_character_
  dat$diagnostic <- trimws(ifelse(is.na(dat$error), "", as.character(dat$error)))
  dat$diagnostic_ok <- dat$diagnostic == ""
  dat$q_dry_eye_component_bh <- NA_real_
  for (model_name in unique(dat$model)) {
    use <- dat$model == model_name & dat$diagnostic_ok & is.finite(dat$pval_individual)
    dat$q_dry_eye_component_bh[use] <- p.adjust(dat$pval_individual[use], method = "BH")
  }
  dat
}

run_maaslin3_one <- function(counts, meta, out_dir, outcome) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  data <- as.data.frame(t(counts))
  fit <- maaslin3::maaslin3(
    input_data = data,
    input_metadata = meta,
    output = out_dir,
    formula = "~ dry_eye_status + age + BMI + gap_days_abs + AntibioticsPriorMonth + SEX + log_total_reads + (1 | FamilyID)",
    min_abundance = 0,
    min_prevalence = 0,
    max_prevalence = 1.01,
    zero_threshold = 0,
    min_variance = 0,
    max_significance = 1,
    normalization = "TSS",
    transform = "LOG",
    correction = "BH",
    standardize = TRUE,
    plot_summary_plot = FALSE,
    plot_associations = FALSE,
    save_models = FALSE,
    save_plots_rds = FALSE,
    small_random_effects = TRUE,
    bypass_small_group_warning = FALSE,
    verbosity = "WARN"
  )
  maaslin3::maaslin_log_reset()
  invisible(fit)
  case_rows_maaslin(file.path(out_dir, "all_results.tsv"), outcome)
}

run_linda_one <- function(counts, meta, outcome) {
  fit <- MicrobiomeStat::linda(
    feature.dat = counts,
    meta.dat = meta,
    formula = "~ dry_eye_status + age + BMI + gap_days_abs + AntibioticsPriorMonth + SEX + (1|FamilyID)",
    feature.dat.type = "count",
    prev.filter = 0,
    mean.abund.filter = 0,
    max.abund.filter = 0,
    is.winsor = TRUE,
    outlier.pct = 0.03,
    adaptive = TRUE,
    zero.handling = "pseudo-count",
    pseudo.cnt = 0.5,
    p.adj.method = "BH",
    alpha = 0.05,
    n.cores = 1,
    verbose = FALSE
  )
  key <- grep("^dry_eye_status", names(fit$output), value = TRUE)[1]
  if (is.na(key)) return(data.frame())
  dat <- fit$output[[key]]
  dat$taxon <- rownames(dat)
  dat$outcome_id <- outcome
  dat$package <- "MicrobiomeStat_LinDA"
  dat$term <- key
  dat
}

run_aldex_one <- function(counts, meta, outcome, mc_samples, seed) {
  mm <- model.matrix(~ dry_eye_status + age + BMI + gap_days_abs + AntibioticsPriorMonth + SEX, data = meta)
  set.seed(seed)
  clr <- ALDEx2::aldex.clr(
    reads = counts,
    conds = mm,
    mc.samples = mc_samples,
    denom = "all",
    verbose = FALSE,
    useMC = FALSE
  )
  dat <- ALDEx2::aldex.glm(clr, verbose = FALSE, fdr.method = "BH")
  dat$taxon <- rownames(dat)
  dat$outcome_id <- outcome
  dat$package <- "ALDEx2_glm"
  dat$seed <- seed
  dat
}

extract_aldex_case <- function(dat) {
  if (!nrow(dat)) return(data.frame())
  cols <- names(dat)
  prefix <- grep("^dry_eye_statusCase:", cols, value = TRUE)
  keep <- unique(c("taxon", "outcome_id", "package", prefix))
  dat[, keep, drop = FALSE]
}

summarize_results <- function(maaslin, linda, aldex_case, focal_taxa) {
  rows <- list()
  if (nrow(maaslin)) {
    m <- maaslin[maaslin$model %in% c("abundance", "prevalence"), , drop = FALSE]
    for (outcome in unique(m$outcome_id)) {
      for (model in unique(m$model)) {
        sub <- m[m$outcome_id == outcome & m$model == model, , drop = FALSE]
        valid <- sub[sub$diagnostic_ok & is.finite(sub$q_dry_eye_component_bh), , drop = FALSE]
        rows[[length(rows) + 1]] <- data.frame(
          package = "maaslin3",
          model = model,
          outcome_id = outcome,
          tested_taxa = length(unique(sub$feature)),
          valid_taxa = length(unique(valid$feature)),
          flagged_taxa = length(unique(sub$feature[!sub$diagnostic_ok | !is.finite(sub$pval_individual)])),
          min_q = if (nrow(valid)) min(valid$q_dry_eye_component_bh) else NA_real_,
          min_q_taxon = if (nrow(valid)) valid$feature[which.min(valid$q_dry_eye_component_bh)] else NA_character_,
          q_lt_0_05 = if (nrow(valid)) sum(valid$q_dry_eye_component_bh < 0.05) else 0L,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  if (nrow(linda)) {
    for (outcome in unique(linda$outcome_id)) {
      sub <- linda[linda$outcome_id == outcome, , drop = FALSE]
      valid <- sub[is.finite(sub$pvalue) & is.finite(sub$padj), , drop = FALSE]
      rows[[length(rows) + 1]] <- data.frame(
        package = "MicrobiomeStat_LinDA",
        model = "mixed_effect_clr",
        outcome_id = outcome,
        tested_taxa = length(unique(sub$taxon)),
        valid_taxa = length(unique(valid$taxon)),
        flagged_taxa = length(unique(sub$taxon[!is.finite(sub$pvalue) | !is.finite(sub$padj)])),
        min_q = if (nrow(valid)) min(valid$padj) else NA_real_,
        min_q_taxon = if (nrow(valid)) valid$taxon[which.min(valid$padj)] else NA_character_,
        q_lt_0_05 = if (nrow(valid)) sum(valid$padj < 0.05) else 0L,
        stringsAsFactors = FALSE
      )
    }
  }
  if (nrow(aldex_case)) {
    q_col <- grep(":(BH|pval\\.padj)$", names(aldex_case), value = TRUE)[1]
    if (!is.na(q_col)) {
      for (outcome in unique(aldex_case$outcome_id)) {
        sub <- aldex_case[aldex_case$outcome_id == outcome, , drop = FALSE]
        valid <- sub[is.finite(sub[[q_col]]), , drop = FALSE]
        rows[[length(rows) + 1]] <- data.frame(
          package = "ALDEx2_glm",
          model = "fixed_effect_clr",
          outcome_id = outcome,
          tested_taxa = length(unique(sub$taxon)),
          valid_taxa = length(unique(valid$taxon)),
          flagged_taxa = length(unique(sub$taxon[!is.finite(sub[[q_col]])])),
          min_q = if (nrow(valid)) min(valid[[q_col]]) else NA_real_,
          min_q_taxon = if (nrow(valid)) valid$taxon[which.min(valid[[q_col]])] else NA_character_,
          q_lt_0_05 = if (nrow(valid)) sum(valid[[q_col]] < 0.05) else 0L,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  summary <- if (length(rows)) do.call(rbind, rows) else data.frame()
  focal <- list()
  if (nrow(maaslin)) {
    sub <- maaslin[maaslin$feature %in% focal_taxa, , drop = FALSE]
    if (nrow(sub)) {
      focal[[length(focal) + 1]] <- data.frame(
        package = "maaslin3",
        model = sub$model,
        outcome_id = sub$outcome_id,
        taxon = sub$feature,
        coef = sub$coef,
        p = sub$pval_individual,
        q = sub$q_dry_eye_component_bh,
        diagnostic_ok = sub$diagnostic_ok,
        diagnostic = sub$diagnostic,
        q_family = "BH within outcome and MaAsLin3 model component among diagnostic-valid dry-eye rows",
        stringsAsFactors = FALSE
      )
    }
  }
  if (nrow(linda)) {
    sub <- linda[linda$taxon %in% focal_taxa, , drop = FALSE]
    if (nrow(sub)) {
      focal[[length(focal) + 1]] <- data.frame(
        package = "MicrobiomeStat_LinDA",
        model = "mixed_effect_clr",
        outcome_id = sub$outcome_id,
        taxon = sub$taxon,
        coef = sub$log2FoldChange,
        p = sub$pvalue,
        q = sub$padj,
        diagnostic_ok = is.finite(sub$pvalue) & is.finite(sub$padj),
        diagnostic = ifelse(is.finite(sub$pvalue) & is.finite(sub$padj), "", "missing or non-finite inferential output"),
        q_family = "LinDA package BH within dry-eye term",
        stringsAsFactors = FALSE
      )
    }
  }
  if (nrow(aldex_case)) {
    coef_col <- grep(":Est$", names(aldex_case), value = TRUE)[1]
    p_col <- grep(":pval$", names(aldex_case), value = TRUE)[1]
    q_col <- grep(":(BH|pval\\.padj)$", names(aldex_case), value = TRUE)[1]
    sub <- aldex_case[aldex_case$taxon %in% focal_taxa, , drop = FALSE]
    if (nrow(sub) && !any(is.na(c(coef_col, p_col, q_col)))) {
      focal[[length(focal) + 1]] <- data.frame(
        package = "ALDEx2_glm",
        model = "fixed_effect_clr",
        outcome_id = sub$outcome_id,
        taxon = sub$taxon,
        coef = sub[[coef_col]],
        p = sub[[p_col]],
        q = sub[[q_col]],
        diagnostic_ok = is.finite(sub[[p_col]]) & is.finite(sub[[q_col]]),
        diagnostic = ifelse(is.finite(sub[[p_col]]) & is.finite(sub[[q_col]]), "", "missing or non-finite inferential output"),
        q_family = "ALDEx2 package BH within dry-eye term",
        stringsAsFactors = FALSE
      )
    }
  }
  focal_df <- if (length(focal)) do.call(rbind, focal) else data.frame()
  list(summary = summary, focal = focal_df)
}

main <- function() {
  args <- parse_args()
  dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
  suppressPackageStartupMessages({
    library(jsonlite)
    library(knitr)
    library(maaslin3)
    library(MicrobiomeStat)
    library(ALDEx2)
  })
  if (args$workers > 1) {
    mirai::daemons(args$workers)
    on.exit(mirai::daemons(0), add = TRUE)
  }

  focal_taxa <- c("Dialister", "Lachnospiraceae_AC2044_group", "Sutterella", "Campylobacter", "Alistipes", "Streptococcus")
  maaslin_rows <- list()
  linda_rows <- list()
  aldex_rows <- list()
  outcome_status <- list()

  for (outcome in args$outcomes) {
    inputs <- read_inputs(args$input_dir, outcome, args$smoke, args$max_taxa)
    counts <- inputs$counts
    meta <- inputs$metadata
    outcome_out <- file.path(args$output_dir, if (args$smoke) "smoke" else "full", outcome)
    dir.create(outcome_out, recursive = TRUE, showWarnings = FALSE)

    maaslin_dir <- file.path(outcome_out, "maaslin3")
    maaslin_rows[[outcome]] <- if (args$skip_maaslin3) data.frame() else run_maaslin3_one(counts, meta, maaslin_dir, outcome)

    linda_rows[[outcome]] <- if (args$skip_linda) data.frame() else run_linda_one(counts, meta, outcome)
    write.csv(linda_rows[[outcome]], file.path(outcome_out, "microbiomestat_linda_dry_eye_status_case.csv"), row.names = FALSE, na = "")

    outcome_seed <- args$seed_base + match(outcome, args$outcomes) - 1L
    aldex_full <- if (args$skip_aldex2) data.frame() else run_aldex_one(counts, meta, outcome, args$aldex_mc_samples, outcome_seed)
    aldex_rows[[outcome]] <- extract_aldex_case(aldex_full)
    write.csv(aldex_full, file.path(outcome_out, "aldex2_glm_all_terms.csv"), row.names = FALSE, na = "")
    write.csv(aldex_rows[[outcome]], file.path(outcome_out, "aldex2_glm_dry_eye_status_case.csv"), row.names = FALSE, na = "")

    outcome_status[[outcome]] <- list(
      samples = nrow(meta),
      cases = sum(meta$dry_eye_status == "Case"),
      controls = sum(meta$dry_eye_status == "Control"),
      taxa = nrow(counts),
      families = length(unique(meta$FamilyID)),
      total_reads_min = min(meta$total_reads),
      total_reads_median = median(meta$total_reads),
      total_reads_max = max(meta$total_reads),
      aldex2_seed = outcome_seed
    )
  }

  maaslin <- do.call(rbind, maaslin_rows)
  linda <- do.call(rbind, linda_rows)
  aldex_case <- do.call(rbind, aldex_rows)

  write.csv(maaslin, file.path(args$output_dir, "maaslin3_dry_eye_status_case_results.csv"), row.names = FALSE, na = "")
  write.csv(linda, file.path(args$output_dir, "microbiomestat_linda_dry_eye_status_case_results.csv"), row.names = FALSE, na = "")
  write.csv(aldex_case, file.path(args$output_dir, "aldex2_glm_dry_eye_status_case_results.csv"), row.names = FALSE, na = "")

  summaries <- summarize_results(maaslin, linda, aldex_case, focal_taxa)
  write_table(summaries$summary, file.path(args$output_dir, "formal_method_min_q_summary"))
  write_table(summaries$focal, file.path(args$output_dir, "formal_method_focal_taxa_results"))

  run_summary <- list(
    generated_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    input_dir = args$input_dir,
    output_dir = args$output_dir,
    smoke = args$smoke,
    executed = list(maaslin3 = !args$skip_maaslin3, microbiomestat_linda = !args$skip_linda, aldex2 = !args$skip_aldex2),
    maaslin3_strategy_status = if (!args$skip_maaslin3) "executed" else if (args$maaslin3_omitted_by_decision) "omitted_by_investigator_decision" else "not_executed",
    outcomes = outcome_status,
    methods = list(
      maaslin3 = list(version = as.character(packageVersion("maaslin3")), formula = "~ dry_eye_status + age + BMI + gap_days_abs + AntibioticsPriorMonth + SEX + log_total_reads + (1 | FamilyID)", family_aware = TRUE, small_random_effects = TRUE, bypass_small_group_warning = FALSE, q_family = "BH within outcome and abundance/prevalence component among diagnostic-valid dry-eye rows"),
      microbiomestat_linda = list(version = as.character(packageVersion("MicrobiomeStat")), formula = "~ dry_eye_status + age + BMI + gap_days_abs + AntibioticsPriorMonth + SEX + (1|FamilyID)", family_aware = TRUE),
      aldex2_glm = list(version = as.character(packageVersion("ALDEx2")), formula = "~ dry_eye_status + age + BMI + gap_days_abs + AntibioticsPriorMonth + SEX", family_aware = FALSE, mc_samples = args$aldex_mc_samples, seed_base = args$seed_base)
    ),
    maaslin3_diagnostic_rule = "Rows with a nonempty MaAsLin3 error/diagnostic field are retained for audit but excluded from inferential summaries and the manuscript evidence count.",
    interpretation_boundary = "Formal package sensitivities do not override the locked claim boundary unless they provide coherent cross-method support. ANCOM-BC2 remains a conservative null taxon-level sensitivity."
  )
  write_json(run_summary, file.path(args$output_dir, "run_summary.json"))
  cat(jsonlite::toJSON(run_summary, auto_unbox = TRUE, pretty = TRUE), "\n")
  0
}

if (sys.nframe() == 0L) quit(status = main(), save = "no")
