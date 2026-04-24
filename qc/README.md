# QC Report

## Overview

`qc_report.py` is a self-contained Python script that reads FragPipe immunopeptidomics
output and produces a single-file interactive HTML report (`qc_report.html`).

The report is **stateful**: each run appends metrics to `qc_state.json`, enabling
longitudinal QC tracking and automatic diff highlighting between runs.

---

## Usage

```bash
uv run qc/qc_report.py [options]
```

All dependencies are declared inline; `uv` installs them automatically on first run.
No virtual environment setup is required.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--label TEXT` | `Run YYYY-MM-DD HH:MM` | Human-readable label for this run |
| `--data PATH` | `../circrna/data/ip/combined_peptide.tsv` | FragPipe combined peptide table |
| `--msstats PATH` | `../circrna/data/ip/msstats.csv` | MSstats intensity table |
| `--out PATH` | `qc_report.html` | Output HTML file |
| `--state PATH` | `qc_state.json` | Run history state file |
| `--reset` | — | Clear all prior run history and start fresh |

### Examples

```bash
# Baseline run
uv run qc/qc_report.py --label "Baseline"

# Update with new data — diffs vs baseline appear automatically
uv run qc/qc_report.py --label "Week 4" --data /path/to/new_combined_peptide.tsv

# Point to a custom state file (e.g. per-project history)
uv run qc/qc_report.py --label "Cohort B" --state cohort_b_state.json

# Wipe history and restart
uv run qc/qc_report.py --reset --label "Re-analysis"
```

---

## Inputs

| File | Description |
|------|-------------|
| `combined_peptide.tsv` | FragPipe combined peptide table (all samples) |
| `msstats.csv` | MSstats-formatted intensity table for quantitative analysis |
| `iedb/iedb_public.sql` | Local IEDB database dump (see **IEDB Setup** below) |

Sample names are derived automatically from columns ending in `" Match Type"` in the
combined peptide table — no manual configuration needed when samples change.

---

## Output

`qc_report.html` — a fully self-contained HTML file (no external dependencies at view
time; Plotly JS is loaded from CDN). Open in any modern browser.

`qc_state.json` — JSON run-history file. Accumulates per-run and per-sample metrics
across executions. Safe to commit to version control.

---

## Report structure

The report is a **sidebar-navigated multi-tab document**:

| Tab | Contents |
|-----|----------|
| **Glossary** | Definitions of all terms and acronyms |
| **Overview** | All cross-sample QC sections (see below) |
| **History** | Run history table + longitudinal trend charts |
| **[Sample name]** | One tab per sample with sample-specific plots |

### Overview sections

| # | Section | What it shows |
|---|---------|---------------|
| 1 | Summary | Per-sample MS/MS vs MBR counts, MBR rate, contaminant rate |
| 2 | Peptide Length Distribution | Dataset-wide and per-sample histograms; MHC-I (8–11 aa) window highlighted |
| 3 | Spectral Count Distribution | Per-sample violin plots (log scale) and MBR rate bar chart |
| 4 | Intensity Distribution | Log₂ intensity violins per condition, missing value rates, replicate correlations |
| 5 | Contaminants | Per-sample contaminant rate bar chart and top contaminant proteins |
| 6 | Peptide Overlap | Pairwise Jaccard similarity heatmap, shared peptide counts, prevalence histogram |
| 7 | Protein Source | SwissProt vs TrEMBL breakdown; unique canonical genes per sample |
| 8 | Amino Acid Composition | Position-specific AA frequency heatmap for 9-mers; immunopeptidome vs proteome background |
| 9 | Sequence Logo | Frequency and information-content logos for 9-mers; P2/P9 anchor positions highlighted |
| 10 | Charge State Distribution | Dataset-wide charge pie and per-sample stacked bar |
| 11 | Sample Clustering / PCA | PCA scatter, explained variance, hierarchically clustered correlation heatmap |
| 12 | IEDB Cross-reference | Known vs novel MHC-I peptides — Venn diagram, donut chart, top known epitopes table |
| 13 | GO / Pathway Enrichment | Top GO:BP, KEGG, and Reactome terms via g:Profiler |

### Per-sample tabs

Each sample tab shows:

| Section | Contents |
|---------|----------|
| Summary | MS/MS, MBR, total, contaminant counts and rates — with ↑↓ diff badges vs previous run |
| Peptide Length Distribution | Per-sample histogram with MHC-I window highlighted |
| Protein Source & Charge State | Source breakdown pie and charge state pie side by side |
| Spectral Count & Intensity | Spectral count histogram (log x) and log₂ intensity histogram |
| Binding Motif (9-mers) | Frequency + information content sequence logos; AA position frequency heatmap |
| Peptide Overlap | Shared peptide count with every other sample |
| Top Proteins | Top 15 source proteins by peptide count (contaminants excluded) |
| IEDB Cross-reference | MHC-I peptides vs full IEDB — Venn diagram |

### History tab

Appears automatically; shows:
- **Run history table** — one row per run with key global metrics; current run highlighted
- **Longitudinal trend charts** — per-sample line charts for total detected, MBR rate,
  contaminant rate, MHC-I count, MHC-I %, plus global total peptides and novel candidate count

---

## Run history and diffs

Each execution appends a snapshot to `qc_state.json`:

```json
{
  "runs": [
    {
      "run_id": "20240115_103000",
      "label": "Baseline",
      "timestamp": "2024-01-15T10:30:00",
      "samples": ["A549_1", "PC3_100M_1"],
      "global": { "total_peptides": 45000, "mhci_count": 39825, ... },
      "per_sample": {
        "A549_1": { "total_detected": 4200, "mbr_rate_num": 9.5, ... }
      }
    }
  ]
}
```

When a previous run exists, each metric in the per-sample summary table shows a
coloured badge:
- **Green ↑** — metric improved (more MHC-I peptides, lower contaminant/MBR rate)
- **Red ↓** — metric regressed

---

## IEDB Setup

Download the SQL export from the IEDB website:

```
https://www.iedb.org/database_export_v3.php
Download: "SQL Statement Export" (~600 MB compressed, ~13 GB uncompressed)
```

Place the uncompressed file at:

```
qc/iedb/iedb_public.sql
```

On first run the script streams through the SQL file, extracts all
`linear_peptide_seq` values from the `epitope` table, and caches the result to
`qc/.iedb_seqs.pkl`. Subsequent runs load from the pickle and are fast.

---

## Caching

| Cache file | Contents | Safe to delete? |
|------------|----------|-----------------|
| `qc/.iedb_seqs.pkl` | Parsed IEDB peptide sequence set | Yes — rebuilt on next run (slow) |
| `qc/qc_state.json` | Run history and metrics | Only if you want to reset history (`--reset`) |

---

## Dependencies

Managed automatically by `uv` via the inline script metadata:

```
pandas, plotly, scipy, numpy, matplotlib, logomaker, scikit-learn, requests
```

Python ≥ 3.11 required.
