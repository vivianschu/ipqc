"""Glossary page for immunopeptidomics QC concepts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st


def _term(name: str, body: str) -> None:
    st.markdown(f"**{name}**")
    st.markdown(body)
    st.markdown("")


st.title("Glossary")
st.markdown(
    "Definitions of key terms used throughout the application. "
    "Where relevant, notes indicate when a value or interpretation is a heuristic "
    "guideline rather than a universal biological rule."
)

# ── Experimental Biology ──────────────────────────────────────────────────────

st.header("Experimental Biology")

_term(
    "Immunopeptidomics",
    "The large-scale study of peptides presented on MHC molecules at the cell surface, "
    "typically profiled by affinity purification of HLA-peptide complexes followed by "
    "liquid chromatography-tandem mass spectrometry (LC-MS/MS). The resulting data describe "
    "the repertoire of antigenic peptides displayed under a given biological condition.",
)

_term(
    "MHC (Major Histocompatibility Complex)",
    "A family of cell-surface glycoproteins that bind and display peptide fragments, "
    "enabling immune surveillance of intracellular protein content. In humans, MHC molecules "
    "are encoded by the highly polymorphic HLA gene cluster on chromosome 6.",
)

_term(
    "HLA (Human Leukocyte Antigen)",
    "The human form of MHC. HLA class I molecules (HLA-A, -B, -C) present peptides to "
    "CD8+ cytotoxic T cells; HLA class II molecules (HLA-DR, -DQ, -DP) present peptides "
    "to CD4+ helper T cells. Hundreds of alleles exist for each locus, each with a "
    "distinct peptide-binding specificity.",
)

_term(
    "MHC class I / HLA class I",
    "Binds peptides typically 8–11 amino acids in length, with 9-mers most common in many "
    "alleles. The binding groove accommodates peptides via interactions at anchor positions "
    "(most commonly P2 and the C-terminal residue). Present on virtually all nucleated cells. "
    "Note: the exact length range varies by allele; some alleles bind peptides outside "
    "the 8–11 aa window.",
)

_term(
    "MHC class II / HLA class II",
    "Binds longer peptides whose core register is commonly 9 amino acids but whose full "
    "sequence, including flanking residues, typically extends to 13–25 aa or more. The "
    "open-ended binding groove permits variable overhang at both termini, so detected "
    "MHC-II peptides span a wider length range than MHC-I peptides and the distribution "
    "is more diffuse. Present primarily on professional antigen-presenting cells (dendritic "
    "cells, macrophages, B cells). Length cutoffs presented in this application are "
    "heuristic guidelines only and should be interpreted in the context of the specific "
    "assay and HLA alleles present.",
)

_term(
    "Immunoprecipitation (IP)",
    "Antibody-based enrichment used in immunopeptidomics to pull down HLA molecules "
    "and their associated peptides from cell lysates. The specificity of the antibody "
    "determines whether class I, class II, or both are enriched. Pan-class I antibodies "
    "(e.g. W6/32) capture all HLA-A, -B, and -C molecules; allele- or locus-specific "
    "reagents narrow the target further.",
)

_term(
    "Neoantigen",
    "A tumor-specific peptide derived from a somatic mutation in a cancer cell. Because "
    "neoantigens are absent from normal tissues, they are candidate targets for personalized "
    "cancer immunotherapy. Immunopeptidomics data are commonly used to confirm that "
    "mutation-derived peptides are actually presented at the cell surface.",
)

_term(
    "Anchor residues",
    "Amino acid positions within an MHC-bound peptide that insert into specificity pockets "
    "in the binding groove, largely determining allele-level binding preference. For 9-mer "
    "MHC-I peptides, P2 and P9 (the C-terminal residue) are the primary anchors for most "
    "alleles. Secondary anchors at other positions contribute additional selectivity. "
    "Anchor patterns vary across alleles and are the basis for peptide-MHC binding "
    "prediction tools.",
)

_term(
    "Immunopeptidome",
    "The complete set of peptides presented by MHC molecules on a given cell type under "
    "specific conditions. The immunopeptidome reflects the cell's protein expression, "
    "protein degradation machinery, and antigen-processing pathway, and can change "
    "substantially in response to inflammation, infection, or oncogenic transformation.",
)

# ── Mass Spectrometry & Quantification ────────────────────────────────────────

st.header("Mass Spectrometry and Quantification")

_term(
    "MS/MS (Tandem Mass Spectrometry)",
    "A peptide precursor ion is isolated, fragmented (typically by collision-induced "
    "dissociation or HCD), and the resulting fragment ions are measured. The fragmentation "
    "pattern provides a sequence-based identification with high confidence. MS/MS-identified "
    "peptides are the most reliable class of identification in a FragPipe output.",
)

_term(
    "MBR (Match Between Runs)",
    "A computational technique in which a peptide identified by MS/MS in one run is "
    "transferred to other runs in the experiment based on accurate precursor mass and "
    "retention time alignment, without requiring a directly observed MS/MS spectrum. "
    "MBR increases sensitivity and reduces missing values in multi-run experiments. "
    "A high MBR proportion (commonly flagged above approximately 30%) may indicate "
    "runs with low spectral quality or poorly overlapping peptide sets, and should be "
    "reviewed alongside other QC metrics. The 30% threshold is a heuristic guideline, "
    "not a universal rule.",
)

_term(
    "Spectral Count",
    "The number of MS/MS spectra matched to a peptide across a run. A semi-quantitative "
    "measure of peptide abundance; higher counts generally indicate more confident and "
    "more abundant detections. Spectral counts are subject to saturation for highly "
    "abundant peptides and are less precise than intensity-based quantification.",
)

_term(
    "Intensity",
    "The integrated ion signal (peak area) extracted from the extracted ion chromatogram "
    "for a peptide precursor. More sensitive and dynamic than spectral count; the preferred "
    "basis for differential abundance testing. Intensities are log-transformed (typically "
    "log base 2) before statistical analysis to reduce the effects of the wide dynamic "
    "range of MS detection.",
)

_term(
    "Charge State",
    "The number of protons carried by a peptide ion during MS analysis. In immunopeptidomics, "
    "the charge state distribution reflects the length and amino acid composition of the "
    "detected peptide population. Shorter peptides (such as typical MHC-I 9-mers) more "
    "commonly appear as doubly charged (2+) ions, while longer peptides tend toward higher "
    "charge states. The exact distribution depends on peptide length range, ionisation "
    "conditions, and instrument settings, and should be interpreted in context rather than "
    "against absolute expected values.",
)

_term(
    "Log₂ (Log base 2) transformation",
    "A monotone transformation applied to MS intensities to compress their wide dynamic "
    "range into a near-normal distribution. Log₂ is preferred over natural log because "
    "a one-unit difference corresponds to a two-fold change, which is intuitive for "
    "comparing peptide abundances across conditions.",
)

_term(
    "Missing Value",
    "A peptide detected in some runs but absent from others, resulting in a gap in the "
    "intensity matrix. Missing values are common in immunopeptidomics because the "
    "immunopeptidome is typically sampled stochastically, and detection is not exhaustive "
    "across runs. High missing rates reduce statistical power in downstream differential "
    "analysis. MSstats and related tools provide normalisation and imputation strategies "
    "for handling missing values.",
)

_term(
    "Contaminant",
    "A non-cellular protein co-purified during the HLA immunoprecipitation step. Common "
    "contaminants include keratins (from skin cells or hair), trypsin (used for digestion), "
    "albumin (from serum), and immunoglobulins. In FragPipe outputs, contaminant proteins "
    "are flagged with the `Cont_` prefix. Contaminant rates above approximately 1–2% of "
    "detected peptides can indicate sample handling issues, though the threshold depends "
    "on sample type and preparation protocol.",
)

_term(
    "FDR (False Discovery Rate)",
    "The expected proportion of incorrect identifications among all called results at a "
    "given score cutoff. FragPipe applies FDR control at the peptide-spectrum match, "
    "peptide, and protein levels, typically at a 1% threshold, using target-decoy "
    "database searching. FDR control does not eliminate false positives; it bounds "
    "their expected rate.",
)

# ── Software and Databases ─────────────────────────────────────────────────────

st.header("Software and Databases")

_term(
    "FragPipe",
    "A computational pipeline for LC-MS/MS data analysis combining MSFragger (a fast "
    "database search engine) with downstream tools for peptide-spectrum match filtering, "
    "protein inference, label-free quantification, and MBR. The `combined_peptide.tsv` "
    "output used by this application is generated by the Philosopher and IonQuant "
    "components of FragPipe.",
)

_term(
    "MSstats",
    "An R package (and Python port) for statistical analysis of quantitative proteomics "
    "data, including normalisation, summarisation to protein level, and differential "
    "abundance testing across experimental conditions. The optional MSstats input to this "
    "application enables intensity distribution and missing value visualisation "
    "at the run and condition level.",
)

_term(
    "UniProt / SwissProt (sp|)",
    "UniProt is the primary reference protein sequence database used in proteomics database "
    "searches. SwissProt (identifiers beginning with `sp|`) is the manually reviewed, "
    "high-quality section of UniProt. Peptides from SwissProt entries originate from "
    "well-characterised canonical protein sequences.",
)

_term(
    "TrEMBL (tr|)",
    "The computationally annotated, unreviewed section of UniProt (identifiers beginning "
    "with `tr|`). TrEMBL entries include predicted open reading frames, isoforms, and "
    "sequences with limited experimental support. They are generally lower confidence than "
    "SwissProt entries.",
)

# ── Statistics and Visualisation ──────────────────────────────────────────────

st.header("Statistics and Visualisation")

_term(
    "PCA (Principal Component Analysis)",
    "A linear dimensionality-reduction technique applied here to the matrix of log₂ "
    "peptide intensities across samples. Samples are projected onto the axes of maximum "
    "variance (principal components), enabling 2D visualisation of sample clustering. "
    "Missing intensity values are imputed with the per-peptide minimum before PCA. "
    "Biological or technical replicates are expected to cluster together; outlier samples "
    "may indicate quality or batch issues.",
)

_term(
    "Jaccard Similarity",
    "A set-overlap metric defined as |A ∩ B| / |A ∪ B|. Ranges from 0 (no shared "
    "peptides between two samples) to 1 (identical detected peptide sets). Higher values "
    "indicate greater overlap in detected peptide repertoires. Expected similarity levels "
    "depend on whether samples are replicates, the same cell type, or biologially "
    "distinct conditions.",
)

_term(
    "Pearson Correlation (r)",
    "Measures the linear correlation between two samples' log₂ intensity profiles "
    "computed over peptides detected in both samples. Values closer to 1 indicate high "
    "reproducibility. Values commonly above 0.85 are reported in good technical replicates, "
    "though the expected range depends on the experimental design and biological variability. "
    "This threshold is a heuristic, not a universal standard.",
)

_term(
    "Information Content (bits)",
    "An entropy-based measure of amino acid constraint at a given position in a sequence "
    "logo, computed from the position-specific probability matrix relative to a uniform "
    "background. Higher bits indicate stronger amino acid conservation at that position "
    "and typically reflect biologically important residues such as anchor positions in "
    "MHC-bound peptides.",
)

_term(
    "Sequence Logo",
    "A graphical representation of positional amino acid frequencies (or information content) "
    "for a set of aligned sequences of the same length. Taller letters at a given position "
    "indicate stronger amino acid preference. In this application, logos are computed for "
    "9-mer peptides as a representation of HLA-I binding motifs. The gold-shaded columns "
    "at positions P2 and P9 mark the most commonly reported primary anchor positions for "
    "HLA-A and HLA-B alleles, though anchor positions vary by allele.",
)

_term(
    "Venn Diagram",
    "A figure showing the overlap between two sets by representing each set as a circle. "
    "The intersection area (where circles overlap) represents elements shared by both sets. "
    "In this application, Venn diagrams can be used to compare the detected peptide sets "
    "between pairs of samples.",
)
