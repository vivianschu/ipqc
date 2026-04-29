"""Glossary page for immunopeptidomics QC concepts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as _components

# ── Term data ─────────────────────────────────────────────────────────────────

TERMS: list[dict] = [
    # Experimental Biology
    {
        "name": "Immunopeptidomics",
        "category": "Experimental Biology",
        "body": (
            "The large-scale study of peptides presented on MHC molecules at the cell surface, "
            "typically profiled by affinity purification of HLA-peptide complexes followed by "
            "liquid chromatography-tandem mass spectrometry (LC-MS/MS). The resulting data describe "
            "the repertoire of antigenic peptides displayed under a given biological condition."
        ),
    },
    {
        "name": "MHC (Major Histocompatibility Complex)",
        "category": "Experimental Biology",
        "body": (
            "A family of cell-surface glycoproteins that bind and display peptide fragments, "
            "enabling immune surveillance of intracellular protein content. In humans, MHC molecules "
            "are encoded by the highly polymorphic HLA gene cluster on chromosome 6."
        ),
    },
    {
        "name": "HLA (Human Leukocyte Antigen)",
        "category": "Experimental Biology",
        "body": (
            "The human form of MHC. HLA class I molecules (HLA-A, -B, -C) present peptides to "
            "CD8+ cytotoxic T cells; HLA class II molecules (HLA-DR, -DQ, -DP) present peptides "
            "to CD4+ helper T cells. Hundreds of alleles exist for each locus, each with a "
            "distinct peptide-binding specificity."
        ),
    },
    {
        "name": "MHC class I / HLA class I",
        "category": "Experimental Biology",
        "body": (
            "Binds peptides typically 8–11 amino acids in length, with 9-mers most common in many "
            "alleles. The binding groove accommodates peptides via interactions at anchor positions "
            "(most commonly P2 and the C-terminal residue). Present on virtually all nucleated cells. "
            "Note: the exact length range varies by allele; some alleles bind peptides outside "
            "the 8–11 aa window."
        ),
    },
    {
        "name": "MHC class II / HLA class II",
        "category": "Experimental Biology",
        "body": (
            "Binds longer peptides whose core register is commonly 9 amino acids but whose full "
            "sequence, including flanking residues, typically extends to 13–25 aa or more. The "
            "open-ended binding groove permits variable overhang at both termini, so detected "
            "MHC-II peptides span a wider length range than MHC-I peptides and the distribution "
            "is more diffuse. Present primarily on professional antigen-presenting cells (dendritic "
            "cells, macrophages, B cells). Length cutoffs in this application are heuristic "
            "guidelines and should be interpreted in context."
        ),
    },
    {
        "name": "Immunoprecipitation (IP)",
        "category": "Experimental Biology",
        "body": (
            "Antibody-based enrichment used in immunopeptidomics to pull down HLA molecules "
            "and their associated peptides from cell lysates. The specificity of the antibody "
            "determines whether class I, class II, or both are enriched. Pan-class I antibodies "
            "(e.g. W6/32) capture all HLA-A, -B, and -C molecules; allele- or locus-specific "
            "reagents narrow the target further."
        ),
    },
    {
        "name": "Neoantigen",
        "category": "Experimental Biology",
        "body": (
            "A tumor-specific peptide derived from a somatic mutation in a cancer cell. Because "
            "neoantigens are absent from normal tissues, they are candidate targets for personalized "
            "cancer immunotherapy. Immunopeptidomics data are commonly used to confirm that "
            "mutation-derived peptides are actually presented at the cell surface."
        ),
    },
    {
        "name": "Anchor Residues",
        "category": "Experimental Biology",
        "body": (
            "Amino acid positions within an MHC-bound peptide that insert into specificity pockets "
            "in the binding groove, largely determining allele-level binding preference. For 9-mer "
            "MHC-I peptides, P2 and P9 (the C-terminal residue) are the primary anchors for most "
            "alleles. Secondary anchors at other positions contribute additional selectivity. "
            "Anchor patterns vary across alleles and are the basis for peptide-MHC binding "
            "prediction tools."
        ),
    },
    {
        "name": "Immunopeptidome",
        "category": "Experimental Biology",
        "body": (
            "The complete set of peptides presented by MHC molecules on a given cell type under "
            "specific conditions. The immunopeptidome reflects the cell's protein expression, "
            "protein degradation machinery, and antigen-processing pathway, and can change "
            "substantially in response to inflammation, infection, or oncogenic transformation."
        ),
    },
    # Mass Spectrometry
    {
        "name": "MS/MS (Tandem Mass Spectrometry)",
        "category": "Mass Spectrometry",
        "body": (
            "A peptide precursor ion is isolated, fragmented (typically by collision-induced "
            "dissociation or HCD), and the resulting fragment ions are measured. The fragmentation "
            "pattern provides a sequence-based identification with high confidence. MS/MS-identified "
            "peptides are the most reliable class of identification in a FragPipe output."
        ),
    },
    {
        "name": "MBR (Match Between Runs)",
        "category": "Mass Spectrometry",
        "body": (
            "A computational technique in which a peptide identified by MS/MS in one run is "
            "transferred to other runs in the experiment based on accurate precursor mass and "
            "retention time alignment, without requiring a directly observed MS/MS spectrum. "
            "MBR increases sensitivity and reduces missing values in multi-run experiments. "
            "A high MBR proportion (commonly flagged above ~30%) may indicate runs with low "
            "spectral quality or poorly overlapping peptide sets."
        ),
    },
    {
        "name": "Spectral Count",
        "category": "Mass Spectrometry",
        "body": (
            "The number of MS/MS spectra matched to a peptide across a run. A semi-quantitative "
            "measure of peptide abundance; higher counts generally indicate more confident and "
            "more abundant detections. Spectral counts are subject to saturation for highly "
            "abundant peptides and are less precise than intensity-based quantification."
        ),
    },
    {
        "name": "Intensity",
        "category": "Mass Spectrometry",
        "body": (
            "The integrated ion signal (peak area) extracted from the extracted ion chromatogram "
            "for a peptide precursor. More sensitive and dynamic than spectral count; the preferred "
            "basis for differential abundance testing. Intensities are log-transformed (typically "
            "log base 2) before statistical analysis to reduce the effects of the wide dynamic "
            "range of MS detection."
        ),
    },
    {
        "name": "Charge State",
        "category": "Mass Spectrometry",
        "body": (
            "The number of protons carried by a peptide ion during MS analysis. In immunopeptidomics, "
            "the charge state distribution reflects the length and amino acid composition of the "
            "detected peptide population. Shorter peptides (such as typical MHC-I 9-mers) more "
            "commonly appear as doubly charged (2+) ions, while longer peptides tend toward higher "
            "charge states."
        ),
    },
    {
        "name": "Log₂ Transformation",
        "category": "Mass Spectrometry",
        "body": (
            "A monotone transformation applied to MS intensities to compress their wide dynamic "
            "range into a near-normal distribution. Log₂ is preferred over natural log because "
            "a one-unit difference corresponds to a two-fold change, which is intuitive for "
            "comparing peptide abundances across conditions."
        ),
    },
    {
        "name": "Missing Value",
        "category": "Mass Spectrometry",
        "body": (
            "A peptide detected in some runs but absent from others, resulting in a gap in the "
            "intensity matrix. Missing values are common in immunopeptidomics because the "
            "immunopeptidome is typically sampled stochastically. High missing rates reduce "
            "statistical power in downstream differential analysis."
        ),
    },
    {
        "name": "Contaminant",
        "category": "Mass Spectrometry",
        "body": (
            "A non-cellular protein co-purified during the HLA immunoprecipitation step. Common "
            "contaminants include keratins, trypsin, albumin, and immunoglobulins. In FragPipe "
            "outputs, contaminant proteins are flagged with the `Cont_` prefix. Contaminant rates "
            "above ~1–2% of detected peptides can indicate sample handling issues."
        ),
    },
    {
        "name": "FDR (False Discovery Rate)",
        "category": "Mass Spectrometry",
        "body": (
            "The expected proportion of incorrect identifications among all called results at a "
            "given score cutoff. FragPipe applies FDR control at the peptide-spectrum match, "
            "peptide, and protein levels, typically at a 1% threshold, using target-decoy "
            "database searching."
        ),
    },
    # Software & Databases
    {
        "name": "FragPipe",
        "category": "Software & Databases",
        "body": (
            "A computational pipeline for LC-MS/MS data analysis combining MSFragger (a fast "
            "database search engine) with downstream tools for peptide-spectrum match filtering, "
            "protein inference, label-free quantification, and MBR. The `combined_peptide.tsv` "
            "output used by this application is generated by the Philosopher and IonQuant "
            "components of FragPipe."
        ),
    },
    {
        "name": "MSstats",
        "category": "Software & Databases",
        "body": (
            "An R package (and Python port) for statistical analysis of quantitative proteomics "
            "data, including normalisation, summarisation to protein level, and differential "
            "abundance testing across experimental conditions. The optional MSstats input to this "
            "application enables intensity distribution and missing value visualisation."
        ),
    },
    {
        "name": "UniProt / SwissProt (sp|)",
        "category": "Software & Databases",
        "body": (
            "UniProt is the primary reference protein sequence database used in proteomics database "
            "searches. SwissProt (identifiers beginning with `sp|`) is the manually reviewed, "
            "high-quality section of UniProt. Peptides from SwissProt entries originate from "
            "well-characterised canonical protein sequences."
        ),
    },
    {
        "name": "TrEMBL (tr|)",
        "category": "Software & Databases",
        "body": (
            "The computationally annotated, unreviewed section of UniProt (identifiers beginning "
            "with `tr|`). TrEMBL entries include predicted open reading frames, isoforms, and "
            "sequences with limited experimental support. They are generally lower confidence than "
            "SwissProt entries."
        ),
    },
    # Statistics & Visualisation
    {
        "name": "PCA (Principal Component Analysis)",
        "category": "Statistics & Visualisation",
        "body": (
            "A linear dimensionality-reduction technique applied here to the matrix of log₂ "
            "peptide intensities across samples. Samples are projected onto the axes of maximum "
            "variance (principal components), enabling 2D visualisation of sample clustering. "
            "Missing intensity values are imputed with the per-peptide minimum before PCA. "
            "Biological or technical replicates are expected to cluster together."
        ),
    },
    {
        "name": "Jaccard Similarity",
        "category": "Statistics & Visualisation",
        "body": (
            "A set-overlap metric defined as |A ∩ B| / |A ∪ B|. Ranges from 0 (no shared "
            "peptides between two samples) to 1 (identical detected peptide sets). Higher values "
            "indicate greater overlap in detected peptide repertoires."
        ),
    },
    {
        "name": "Pearson Correlation (r)",
        "category": "Statistics & Visualisation",
        "body": (
            "Measures the linear correlation between two samples' log₂ intensity profiles "
            "computed over peptides detected in both samples. Values closer to 1 indicate high "
            "reproducibility. Values commonly above 0.85 are reported in good technical replicates, "
            "though the expected range depends on the experimental design."
        ),
    },
    {
        "name": "Information Content (bits)",
        "category": "Statistics & Visualisation",
        "body": (
            "An entropy-based measure of amino acid constraint at a given position in a sequence "
            "logo, computed from the position-specific probability matrix relative to a uniform "
            "background. Higher bits indicate stronger amino acid conservation at that position "
            "and typically reflect biologically important residues such as anchor positions."
        ),
    },
    {
        "name": "Sequence Logo",
        "category": "Statistics & Visualisation",
        "body": (
            "A graphical representation of positional amino acid frequencies (or information content) "
            "for a set of aligned sequences of the same length. Taller letters at a given position "
            "indicate stronger amino acid preference. In this application, logos are computed for "
            "9-mer peptides as a representation of HLA-I binding motifs."
        ),
    },
    {
        "name": "Venn Diagram",
        "category": "Statistics & Visualisation",
        "body": (
            "A figure showing the overlap between two sets by representing each set as a circle. "
            "The intersection area represents elements shared by both sets. In this application, "
            "Venn diagrams can be used to compare the detected peptide sets between pairs of samples."
        ),
    },
]

CATEGORIES = ["Experimental Biology", "Mass Spectrometry", "Software & Databases", "Statistics & Visualisation"]

CATEGORY_COLORS: dict[str, tuple[str, str]] = {
    "Experimental Biology":     ("#dbeafe", "#1d4ed8"),
    "Mass Spectrometry":        ("#fce7f3", "#be185d"),
    "Software & Databases":     ("#d1fae5", "#065f46"),
    "Statistics & Visualisation": ("#ede9fe", "#5b21b6"),
}

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Filter pill buttons */
div[data-testid="stPills"] button {
    border-radius: 9999px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Card — hover handled directly on the element, perfectly per-card scoped */
.glossary-card {
    background: white;
    border: 1.5px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px 18px 16px 18px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    cursor: pointer;
    transition: border-color 0.15s, box-shadow 0.15s;
    user-select: none;
}
.glossary-card:hover:not(.open) {
    border-color: #9ca3af;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.glossary-card.open {
    border-color: #ea580c;
    box-shadow: 0 0 0 2px #ea580c18;
}

/* Card title */
.card-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #111827;
    line-height: 1.35;
}

/* Category badge */
.category-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 500;
    width: fit-content;
}

/* Snippet */
.card-snippet {
    font-size: 0.82rem;
    color: #6b7280;
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* Hide the Streamlit button completely — JS handles the click */
div[data-testid="stColumn"]:has(.glossary-card) div[data-testid="stButton"] {
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Count line */
.term-count {
    font-size: 0.82rem;
    color: #6b7280;
    margin-bottom: 4px;
}

/* Detail panel */
.detail-panel {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 6px;
}
.detail-panel h4 {
    margin: 0 0 6px 0;
    font-size: 0.95rem;
    color: #111827;
}
.detail-panel p {
    margin: 0;
    font-size: 0.85rem;
    color: #374151;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────

st.title("Glossary")
st.markdown(
    "Definitions of key terms used throughout the application. "
    "Where relevant, notes indicate when a value is a heuristic guideline "
    "rather than a universal biological rule."
)

# ── Category filter ───────────────────────────────────────────────────────────

selected_cat = st.pills(
    "Filter by category",
    options=["All"] + CATEGORIES,
    default="All",
    key="glossary_cat_filter",
    label_visibility="collapsed",
)

# ── Filtered terms ────────────────────────────────────────────────────────────

filtered = TERMS if selected_cat == "All" else [t for t in TERMS if t["category"] == selected_cat]
st.markdown(f'<div class="term-count">{len(filtered)} term{"s" if len(filtered) != 1 else ""}</div>', unsafe_allow_html=True)

# ── JS: wire card clicks to their hidden Streamlit buttons ────────────────────
# srcdoc iframes share the parent's origin so window.parent.document is accessible.
_components.html("""
<script>
(function () {
    function attach() {
        var doc = window.parent.document;
        doc.querySelectorAll('.glossary-card').forEach(function (card) {
            if (card._gcAttached) return;
            card._gcAttached = true;
            card.addEventListener('click', function () {
                var col = card.closest('[data-testid="stColumn"]');
                if (!col) return;
                var btn = col.querySelector('[data-testid="stButton"] button');
                if (btn) btn.click();
            });
        });
    }

    attach();

    // Re-attach after Streamlit re-renders the DOM
    var timer;
    new MutationObserver(function () {
        clearTimeout(timer);
        timer = setTimeout(attach, 80);
    }).observe(window.parent.document.body, { childList: true, subtree: true });
})();
</script>
""", height=0)

# ── Session state ─────────────────────────────────────────────────────────────

if "glossary_selected" not in st.session_state:
    st.session_state["glossary_selected"] = None


def _toggle(name: str) -> None:
    st.session_state["glossary_selected"] = (
        None if st.session_state["glossary_selected"] == name else name
    )


# ── Card grid ─────────────────────────────────────────────────────────────────

COLS = 3
rows = [filtered[i : i + COLS] for i in range(0, len(filtered), COLS)]

for row in rows:
    cols = st.columns(COLS)
    for col, term in zip(cols, row):
        cat = term["category"]
        bg, fg = CATEGORY_COLORS.get(cat, ("#f3f4f6", "#374151"))
        snippet = term["body"][:160].rstrip() + ("…" if len(term["body"]) > 160 else "")
        is_open = st.session_state["glossary_selected"] == term["name"]
        badge = f'<span class="category-badge" style="background:{bg};color:{fg};">{cat}</span>'
        card_cls = "glossary-card open" if is_open else "glossary-card"
        card_html = f"""
        <div class="{card_cls}">
            <div class="card-title">{term["name"]}</div>
            {badge}
            <div class="card-snippet">{snippet}</div>
        </div>
        """
        with col:
            st.markdown(card_html, unsafe_allow_html=True)
            st.button(
                " ",
                key=f"gloss_{term['name']}",
                on_click=_toggle,
                args=(term["name"],),
            )

    selected_in_row = next(
        (t for t in row if t["name"] == st.session_state["glossary_selected"]), None
    )
    if selected_in_row:
        cat = selected_in_row["category"]
        bg, fg = CATEGORY_COLORS.get(cat, ("#f3f4f6", "#374151"))
        badge_html = f'<span class="category-badge" style="background:{bg};color:{fg};margin-bottom:8px;display:inline-block;">{cat}</span>'
        st.markdown(
            f"""<div class="detail-panel">
                {badge_html}
                <h4>{selected_in_row["name"]}</h4>
                <p>{selected_in_row["body"]}</p>
            </div>""",
            unsafe_allow_html=True,
        )
