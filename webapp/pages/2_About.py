"""About page for the Immunopeptidomics Platform."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.title("About")
st.markdown(
    "The Immunopeptidomics Platform supports end-to-end analysis of MHC-presented peptides "
    "identified by mass spectrometry, from quality control and exploratory data analysis through "
    "to HLA typing, MHC binding prediction, and neoantigen prioritisation. It is designed for "
    "researchers conducting HLA immunopeptidomics experiments in academic or translational settings."
)

st.markdown("---")

st.subheader("Platform Modules")
st.markdown(
    """
**MS Analysis** — quality control and exploratory analysis of FragPipe peptide table outputs:

- Identification overview (MS/MS vs. match-between-runs counts per sample)
- Peptide length distributions compared against MHC class I and class II ranges
- Spectral count and intensity distributions
- Contaminant profiling
- Cross-sample peptide overlap (Jaccard similarity and shared peptide counts)
- Protein source classification (SwissProt / TrEMBL / contaminant)
- Amino acid composition and 9-mer sequence logos
- Charge state distributions
- PCA and hierarchical sample correlation

**HLA Typing** — HLA allele inference from the immunopeptidome using motif deconvolution.

**MHC Prediction** — binding and presentation prediction for peptide lists using NetMHCpan and MHCflurry.

**My Runs / Compare Runs** — save and revisit completed analyses; compare metrics across runs side-by-side.
"""
)

st.subheader("Intended Use")
st.markdown(
    """
This platform is intended for researchers evaluating immunopeptidomics data quality, identifying
outlier samples, and performing downstream analyses including neoantigen prioritisation, differential
peptide presentation, and motif-based HLA typing.

Users are expected to have familiarity with mass spectrometry-based proteomics workflows. This
platform is **not** a clinical diagnostic product and must not be used as the basis for clinical decisions.
"""
)

st.subheader("Inputs Accepted")
st.markdown(
    """
| Input | Module | Description |
|-------|--------|-------------|
| FragPipe peptide table | MS Analysis | `combined_peptide.tsv` — one row per unique peptide |
| MSstats intensity file | MS Analysis | `msstats.csv` with `Intensity`, `Run`, `Condition` columns |
| Peptide list | MHC Prediction | Plain list or CSV of peptide sequences |

The peptide table must contain at minimum a peptide sequence column and at least one
sample-level Match Type column following the FragPipe naming convention
(`<sample> Match Type`, `<sample> Spectral Count`, `<sample> Intensity`).
"""
)

st.subheader("Interpreting the Output")
st.markdown(
    """
Each section of the MS Analysis report highlights one aspect of data quality. Key heuristics and
caveats are described inline in the report tabs and in the [Glossary](/Glossary) page.

Some figures (including PCA and overlap heatmaps) are computationally derived and depend
on the number and nature of samples present. Plots that cannot be meaningfully generated
for a given dataset are suppressed with an explanatory note.

All figures are interactive (pan, zoom, hover). The HTML report download produces a
self-contained file suitable for sharing with collaborators.
"""
)

st.subheader("User Accounts and Run History")
st.markdown(
    """
Registered users can save completed analyses to their account and revisit them in future
sessions via the **My Runs** page. Saved runs can also be compared side-by-side to assess
batch-to-batch or replicate-to-replicate consistency.

Run data is stored locally on the server in a per-user directory. Account passwords are
stored as bcrypt hashes; no plain-text credentials are retained.
"""
)

st.subheader("Technical Notes")
st.markdown(
    """
- Built with [Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/)
- PCA uses scikit-learn's `PCA` on log-transformed, min-value-imputed intensity matrices
- Sequence logos rendered via [logomaker](https://logomaker.readthedocs.io/)
- Hierarchical clustering uses Ward linkage with Euclidean distance
- MHC binding prediction via [NetMHCpan 4.2](https://services.healthtech.dtu.dk/services/NetMHCpan-4.2/) and [MHCflurry](https://github.com/openvax/mhcflurry)
- Contaminant detection based on the `Cont_` prefix convention in FragPipe FASTA databases
"""
)
