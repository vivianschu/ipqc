# Immunopeptidomics QC Webapp

Interactive Streamlit application for QC reporting of immunopeptidomics data from
[FragPipe](https://fragpipe.nesvilab.org/) peptide table outputs.

## Features

- **Upload & preview** — drag-and-drop FragPipe `combined_peptide.tsv`, auto-detects delimiter
- **Flexible column mapping** — pick peptide, protein, gene, charge, and length columns
- **Sample editor** — auto-detect from FragPipe convention or map manually; paste names from clipboard
- **12 QC sections**, all conditionally rendered if the required columns are present:
  - Dataset summary
  - MS/MS vs MBR detection counts and rates
  - Peptide length distribution (global + per sample), with MHC-I / MHC-II highlighting
  - Spectral count violin plots
  - Intensity distribution (requires optional `msstats.csv` upload)
  - Contaminant rate per sample + top contaminant protein table
  - Pairwise peptide overlap (Jaccard similarity and shared counts)
  - Protein source breakdown (SwissProt / TrEMBL / Contaminant)
  - Amino acid composition vs human proteome background
  - 9-mer sequence logos (frequency + information content, P2/P9 anchor highlight)
  - Charge state distribution
  - PCA and hierarchical sample correlation heatmap
- **Per-sample detail** — expandable sections with per-sample length, source, spectral count,
  intensity, top proteins, overlap bar chart, and 9-mer motif
- **Downloads** — self-contained HTML report and summary CSV

## Quickstart

### Using uv (recommended)

```bash
cd qc/webapp
uv run streamlit run app.py
```

### Using pip

```bash
cd qc/webapp
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Usage

1. **Upload** — drop your `combined_peptide.tsv` on the upload screen. Optionally upload
   `msstats.csv` to enable the intensity section.
2. **Configure** — the app auto-detects sample columns from the FragPipe `<sample> Match Type`
   convention. Adjust the column dropdowns, add/rename samples as needed, then click
   **Run QC Analysis**.
3. **Report** — browse each QC section in the tab strip. Use the **Download** buttons at the
   bottom for a self-contained HTML report or a summary CSV.

## Input file format

The primary input is the FragPipe `combined_peptide.tsv`. The app expects:

| Column type | Example name | Required? |
|---|---|---|
| Peptide sequence | `Peptide Sequence` | **Yes** |
| Protein | `Protein` | Optional (needed for contaminant & source sections) |
| Gene | `Gene` | Optional |
| Peptide length | `Peptide Length` | Optional (auto-derived if absent) |
| Charge | `Charges` | Optional |
| Per-sample Match Type | `PBMC_1 Match Type` | Required for per-sample stats |
| Per-sample Spectral Count | `PBMC_1 Spectral Count` | Optional |
| Per-sample Intensity | `PBMC_1 Intensity` | Optional (needed for PCA) |

The MSstats file must have at minimum `Intensity`, `Run`, and `Condition` columns.

## Column mapping schema

The internal representation of a fully configured mapping looks like:

```python
from modules.mapping import ColumnMapping, SampleDef

mapping = ColumnMapping(
    peptide_col="Peptide Sequence",      # required
    protein_col="Protein",               # optional
    gene_col="Gene",                     # optional
    length_col=None,                     # None → derived from peptide string length
    charge_col="Charges",                # optional
    entry_name_col="Entry Name",         # optional
    protein_desc_col="Protein Description",  # optional
    samples=[
        SampleDef(
            name="PBMC_1",
            match_col="PBMC_1 Match Type",
            spectral_col="PBMC_1 Spectral Count",
            intensity_col="PBMC_1 Intensity",
        ),
        SampleDef(
            name="PBMC_2",
            match_col="PBMC_2 Match Type",
            spectral_col="PBMC_2 Spectral Count",
            intensity_col="PBMC_2 Intensity",
        ),
    ],
)
```

## Project structure

```
webapp/
├── app.py               # Streamlit entry point (3 screens: Upload → Map → Report)
├── pyproject.toml
├── README.md
└── modules/
    ├── __init__.py
    ├── parsing.py        # Delimiter detection, DataFrame loading
    ├── mapping.py        # ColumnMapping / SampleDef types, auto-detect, validation
    ├── metrics.py        # All analysis computations (pure functions)
    ├── charts.py         # All Plotly / matplotlib chart generation
    └── report.py         # Self-contained HTML report assembly
```

## What changed from the original script

| Original | Webapp |
|---|---|
| Hardcoded paths (`combined_peptide.tsv`, `msstats.csv`) | File upload UI; msstats is optional |
| CLI arguments (`--data`, `--label`, etc.) | Replaced by Streamlit UI controls |
| Sample names inferred from `Match Type` columns only | Auto-detect + manual editor + paste-in flow |
| Ahmed replicate correlation block (hardcoded sample names) | Generalised to all sample pairs with intensity columns |
| IEDB SQL cross-reference | Removed (requires large local SQL dump not bundled) |
| g:Profiler GO enrichment API call | Removed (can be re-added as optional section) |
| QC run history / `qc_state.json` | Removed (webapp is stateless; run history is not meaningful in browser context) |
| Monolithic 1400-line script | Split into `parsing`, `mapping`, `metrics`, `charts`, `report` modules |
| Static HTML file written to disk | Live Streamlit report with download button |
