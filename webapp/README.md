# Immunopeptidomics Platform

Interactive Streamlit application for immunopeptidomics data analysis, HLA typing, and
neoantigen prioritisation from [FragPipe](https://fragpipe.nesvilab.org/) peptide table outputs.

## Modules

| Module | Description |
|--------|-------------|
| **MS Analysis** | Upload → configure → QC report for FragPipe peptide tables |
| **HLA Typing** | HLA allele inference from the immunopeptidome via motif deconvolution |
| **MHC Prediction** | Binding and presentation prediction using NetMHCpan 4.2 and MHCflurry |
| **My Runs** | Save, revisit, and manage completed analyses |
| **Compare Runs** | Side-by-side metric comparison across saved runs |
| **Glossary** | Reference definitions for immunopeptidomics QC concepts |

## Quickstart

### Using uv (recommended)

```bash
cd webapp
uv run streamlit run app.py
```

### Using pip

```bash
cd webapp
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## MS Analysis

### Inputs

| Column type | Example name | Required? |
|---|---|---|
| Peptide sequence | `Peptide Sequence` | **Yes** |
| Protein | `Protein` | Optional |
| Gene | `Gene` | Optional |
| Peptide length | `Peptide Length` | Optional (auto-derived) |
| Charge | `Charges` | Optional |
| Per-sample Match Type | `PBMC_1 Match Type` | Required for per-sample stats |
| Per-sample Spectral Count | `PBMC_1 Spectral Count` | Optional |
| Per-sample Intensity | `PBMC_1 Intensity` | Optional (needed for PCA) |

The MSstats file requires at minimum `Intensity`, `Run`, and `Condition` columns.

### QC sections

- Dataset summary (peptide counts, sample count, MBR rates)
- MS/MS vs. MBR detection counts and rates
- Peptide length distribution (global + per sample), with MHC-I / MHC-II highlighting
- Spectral count violin plots
- Intensity distributions (requires `msstats.csv`)
- Contaminant rate per sample and top contaminant protein table
- Pairwise peptide overlap (Jaccard similarity and shared counts)
- Protein source breakdown (SwissProt / TrEMBL / contaminant)
- Amino acid composition vs. human proteome background
- 9-mer sequence logos (frequency + information content, P2/P9 anchor highlight)
- Charge state distribution
- PCA and hierarchical sample correlation heatmap
- Per-sample detail panels

## MHC Prediction

Accepts a list of peptide sequences and predicts MHC class I binding affinity and
presentation scores. NetMHCpan 4.2 must be installed separately (academic license required;
binary bundled under `tools/netMHCpan-4.2/`). MHCflurry is installed via PyPI.

## Project structure

```
webapp/
├── app.py               # Streamlit entry point — navigation shell
├── pyproject.toml
├── README.md
├── modules/
│   ├── auth.py          # Password hashing and verification
│   ├── charts.py        # Plotly / matplotlib chart generation
│   ├── database.py      # SQLite user and run storage
│   ├── hla_typing.py    # HLA inference from peptide motifs
│   ├── mapping.py       # ColumnMapping / SampleDef types, auto-detect
│   ├── metrics.py       # Analysis computations (pure functions)
│   ├── motif_decon.py   # Unsupervised motif deconvolution
│   ├── parsing.py       # Delimiter detection, DataFrame loading
│   ├── prediction.py    # MHC prediction orchestration
│   ├── predictors/      # NetMHCpan and MHCflurry predictor wrappers
│   ├── report.py        # Self-contained HTML report assembly
│   ├── storage.py       # Run serialisation / deserialisation
│   └── ui_utils.py      # Sidebar CSS, prototype gate, auth block
└── pages/
    ├── analysis.py      # MS Analysis workflow (Upload → Configure → Report)
    ├── 2_About.py
    ├── 3_Glossary.py
    ├── 4_My_Runs.py
    ├── 5_Compare_Runs.py
    ├── 6_MHC_Prediction.py
    ├── 7_Diagnostics.py
    └── 8_HLA_Typing.py
```

## Data storage

Saved runs are written to `webapp/data/runs/<user_id>/<run_id>/`. The SQLite database
is at `webapp/data/qc.db` and is created automatically on first startup.
