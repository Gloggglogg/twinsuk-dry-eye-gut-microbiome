options(stringsAsFactors = FALSE)

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    input_dir = "authorized_inputs/genus",
    output_dir = "runtime/run_2013_ancombc2_analysis",
    outcomes = c("either_often_symptom_or", "highly_symptomatic_and", "hs_vs_clean_control"),
    smoke = FALSE,
    max_taxa = 8,
    random_family = TRUE
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
    } else if (key == "--fixed-only") {
      out$random_family <- FALSE
    } else {
      stop("Unknown argument: ", key)
    }
    i <- i + 1
  }
  out
}

read_count_matrix <- function(path, max_taxa = NULL) {
  dat <- read.csv(path, check.names = FALSE)
  rownames(dat) <- dat$taxon
  dat$taxon <- NULL
  mat <- as.matrix(dat)
  storage.mode(mat) <- "numeric"
  if (!is.null(max_taxa)) {
    mat <- mat[seq_len(min(max_taxa, nrow(mat))), , drop = FALSE]
  }
  mat
}

read_metadata <- function(path, sample_ids) {
  meta <- read.csv(path, check.names = FALSE)
  meta <- meta[match(sample_ids, meta$sample_id), , drop = FALSE]
  if (any(is.na(meta$sample_id))) {
    stop("Metadata is missing sample IDs present in count matrix.")
  }
  rownames(meta) <- meta$sample_id
  meta$dry_eye_status <- factor(meta$dry_eye_status, levels = c("Control", "Case"))
  meta$AntibioticsPriorMonth <- factor(meta$AntibioticsPriorMonth)
  meta$FamilyID <- factor(meta$FamilyID)
  if ("SEX" %in% names(meta)) {
    meta$SEX <- factor(meta$SEX)
  }
  meta$age <- as.numeric(meta$age)
  meta$BMI <- as.numeric(meta$BMI)
  meta$gap_days_abs <- as.numeric(meta$gap_days_abs)
  meta
}

write_json <- function(x, path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    txt <- paste(c("{", paste(sprintf('  "%s": "%s"', names(x), unlist(x)), collapse = ",\n"), "}"), collapse = "\n")
    writeLines(txt, path)
  } else {
    jsonlite::write_json(x, path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  }
}

apply_named_dof_alignment <- function() {
  # ANCOM-BC2 2.12.1 binds genus-specific degrees-of-freedom vectors by
  # position. A sparse genus can omit an unestimable nuisance coefficient,
  # which shifts later coefficients. Aligning by coefficient name preserves
  # estimable terms and records NA for the omitted term.
  namespace <- asNamespace("ANCOMBC")
  iter_remle <- get(".iter_remle", envir = namespace)
  body_lines <- deparse(body(iter_remle), width.cutoff = 500L)
  target <- which(trimws(body_lines) == 'dof = do.call("rbind", dof)')
  if (length(target) != 1L) {
    stop("Could not identify the ANCOM-BC2 degrees-of-freedom row-binding statement uniquely.")
  }
  replacement <- c(
    "        dof = lapply(dof, function(dof_i) {",
    "            if (is.null(names(dof_i))) {",
    "                return(stats::setNames(as.numeric(dof_i), fix_eff))",
    "            }",
    "            dof_aligned = rep(NA_real_, n_fix_eff)",
    "            names(dof_aligned) = fix_eff",
    "            dof_aligned[match(names(dof_i), fix_eff)] = dof_i",
    "            return(dof_aligned)",
    "        })",
    "        dof = do.call(\"rbind\", dof)"
  )
  body_lines <- append(body_lines[-target], replacement, after = target - 1L)
  body(iter_remle) <- parse(text = paste(body_lines, collapse = "\n"))[[1L]]
  environment(iter_remle) <- namespace
  assignInNamespace(".iter_remle", iter_remle, ns = "ANCOMBC")
  invisible(TRUE)
}

extract_case_results <- function(res) {
  if (is.null(res)) {
    return(data.frame())
  }
  out <- as.data.frame(res)
  if (!"taxon" %in% names(out)) {
    out$taxon <- rownames(out)
  }
  case_cols <- grep("dry_eye_statusCase", names(out), value = TRUE)
  keep <- unique(c("taxon", case_cols))
  keep <- keep[keep %in% names(out)]
  out[, keep, drop = FALSE]
}

standardize_case_columns <- function(dat) {
  names(dat) <- sub("dry_eye_statusCase$", "case", names(dat))
  names(dat) <- sub("dry_eye_statusCase", "case", names(dat))
  dat
}

run_one <- function(input_dir, output_dir, outcome, smoke, max_taxa, random_family) {
  suppressPackageStartupMessages(library(ANCOMBC))
  named_dof_alignment <- apply_named_dof_alignment()
  outcome_dir <- file.path(input_dir, outcome)
  count_path <- file.path(outcome_dir, "genus_counts.csv")
  meta_path <- file.path(outcome_dir, "metadata.csv")
  if (!file.exists(count_path)) stop("Missing count file: ", count_path)
  if (!file.exists(meta_path)) stop("Missing metadata file: ", meta_path)

  counts <- read_count_matrix(count_path, if (smoke) max_taxa else NULL)
  meta <- read_metadata(meta_path, colnames(counts))
  out_dir <- file.path(output_dir, if (smoke) "smoke" else "full", outcome)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  fixed_terms <- c("dry_eye_status", "age", "BMI", "gap_days_abs", "AntibioticsPriorMonth")
  if ("SEX" %in% names(meta) && length(unique(meta$SEX)) > 1) {
    fixed_terms <- c(fixed_terms, "SEX")
  }
  fix_formula <- paste(fixed_terms, collapse = " + ")
  rand_formula <- if (random_family) "(1 | FamilyID)" else NULL
  status <- "ok"
  warning_text <- NULL
  started <- Sys.time()
  fit <- tryCatch(
    withCallingHandlers(
      ANCOMBC::ancombc2(
        data = counts,
        taxa_are_rows = TRUE,
        meta_data = meta,
        fix_formula = fix_formula,
        rand_formula = rand_formula,
        p_adj_method = "BH",
        pseudo = 0,
        pseudo_sens = TRUE,
        prv_cut = 0,
        lib_cut = 0,
        group = NULL,
        struc_zero = FALSE,
        neg_lb = FALSE,
        alpha = 0.05,
        n_cl = 1,
        verbose = FALSE,
        global = FALSE,
        pairwise = FALSE,
        dunnet = FALSE,
        trend = FALSE
      ),
      warning = function(w) {
        if (status == "ok") status <<- "warning"
        warning_text <<- paste(c(warning_text, conditionMessage(w)), collapse = " | ")
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      status <<- "error"
      warning_text <<- conditionMessage(e)
      NULL
    }
  )
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  if (status != "error" && !is.null(warning_text)) {
    status <- "warning"
  }

  full_res <- if (!is.null(fit) && !is.null(fit$res)) as.data.frame(fit$res) else data.frame()
  case_res <- standardize_case_columns(extract_case_results(if (!is.null(fit)) fit$res else NULL))
  write.csv(full_res, file.path(out_dir, "ancombc2_full_result.csv"), row.names = FALSE)
  write.csv(case_res, file.path(out_dir, "ancombc2_case_result.csv"), row.names = FALSE)

  manifest <- list(
    outcome = outcome,
    smoke = smoke,
    samples = ncol(counts),
    taxa = nrow(counts),
    cases = sum(meta$dry_eye_status == "Case"),
    controls = sum(meta$dry_eye_status == "Control"),
    families = length(unique(meta$FamilyID)),
    fix_formula = fix_formula,
    rand_formula = if (is.null(rand_formula)) "" else rand_formula,
    p_adj_method = "BH",
    pseudo_sens = TRUE,
    status = status,
    warning = if (is.null(warning_text)) "" else warning_text,
    elapsed_seconds = round(elapsed, 3),
    r_version = as.character(getRversion()),
    ancombc_version = as.character(utils::packageVersion("ANCOMBC")),
    named_dof_alignment = named_dof_alignment
  )
  write_json(manifest, file.path(out_dir, "manifest.json"))
  invisible(manifest)
}

main <- function() {
  args <- parse_args()
  dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
  manifests <- list()
  for (outcome in args$outcomes) {
    message("Running ANCOM-BC2: outcome=", outcome, " smoke=", args$smoke)
    manifests[[length(manifests) + 1]] <- run_one(
      args$input_dir, args$output_dir, outcome, args$smoke, args$max_taxa, args$random_family
    )
  }
  write_json(
    list(
      generated_utc = format(as.POSIXct(Sys.time(), tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ"),
      output_dir = normalizePath(args$output_dir, winslash = "/", mustWork = FALSE),
      smoke = args$smoke,
      runs = manifests
    ),
    file.path(args$output_dir, if (args$smoke) "smoke_run_manifest.json" else "full_run_manifest.json")
  )
  message("Wrote ANCOM-BC2 outputs to ", args$output_dir)
}

if (sys.nframe() == 0L) main()
