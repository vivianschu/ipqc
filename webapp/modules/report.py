"""HTML report assembly for immunopeptidomics QC.

Builds a self-contained, downloadable HTML file from pre-computed metrics
and chart HTML fragments.  The Streamlit UI calls :func:`build_html_report`
and offers the result as a file download.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go

PLOTLY_CDN = (
    '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"'
    " charset=\"utf-8\"></script>"
)

_CSS = """
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body   { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 0; padding: 0; background: #f8f9fb; color: #1a1f36; }
  .wrap  { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1     { font-size: 1.7rem; font-weight: 700; margin-bottom: .3rem; }
  h2     { font-size: 1.25rem; font-weight: 600; color: #1a1f36;
           border-bottom: 2px solid #e4e8f0; padding-bottom: .4rem;
           margin: 2rem 0 1rem; }
  h3     { font-size: 1rem; font-weight: 600; color: #444; margin: 1.2rem 0 .5rem; }
  .meta  { color: #666; font-size: .9rem; margin-bottom: 1.5rem; }
  .section { background: #fff; border-radius: 10px; padding: 1.5rem;
             box-shadow: 0 1px 4px rgba(0,0,0,.07); margin-bottom: 1.5rem; }
  .note  { color: #666; font-style: italic; font-size: .9rem; }
  .help  { background: #f0f4ff; border-left: 3px solid #4C72B0;
           padding: .6rem 1rem; border-radius: 4px; font-size: .88rem;
           margin-bottom: 1rem; color: #333; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
  .stats-table { border-collapse: collapse; width: 100%; font-size: .88rem; }
  .stats-table th, .stats-table td
               { padding: .45rem .8rem; border: 1px solid #e4e8f0; text-align: left; }
  .stats-table thead { background: #f0f4ff; font-weight: 600; }
  .stats-table tbody tr:nth-child(even) { background: #fafbff; }
  .badge { display: inline-block; border-radius: 4px; padding: .15em .5em;
           font-size: .8rem; font-weight: 600; }
  .badge-green  { background: #dcfce7; color: #166534; }
  .badge-yellow { background: #fef9c3; color: #854d0e; }
  .badge-red    { background: #fee2e2; color: #991b1b; }
  @media (max-width: 700px) { .grid-2 { grid-template-columns: 1fr; } }
</style>
"""

_HELP_TEXTS: dict[str, str] = {
    "msms_mbr": (
        "<strong>MS/MS</strong> identifications are directly sequenced spectra — "
        "the gold standard. <strong>MBR (Match Between Runs)</strong> transfers "
        "identifications across samples based on accurate mass and retention time. "
        "A high MBR rate (>30%) can indicate low sample quality or poor LC reproducibility."
    ),
    "length": (
        "MHC class I peptides are typically <strong>8–11 amino acids</strong> (median 9 aa). "
        "MHC class II peptides range from <strong>13–25 aa</strong>. "
        "A sharp peak at 9 aa is a hallmark of a clean MHC-I immunopeptidomics experiment."
    ),
    "contam": (
        "Contaminants (tagged with <code>Cont_</code> in the protein database) typically "
        "include keratins and common lab proteins. "
        "Rates above ~1–2% may indicate sample handling issues."
    ),
    "pca": (
        "PCA is computed on log₂ intensities of peptides detected in ≥ min(3, n_samples) "
        "samples; missing values are min-value imputed per peptide. "
        "Biological replicates should cluster together."
    ),
    "aa_motif": (
        "The sequence logo shows enrichment at each position relative to uniform background. "
        "For 9-mers, <strong>P2 and P9</strong> (gold highlight) are primary anchor residues "
        "for most MHC-I alleles. Strong signal at these positions confirms HLA-presented peptides."
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def _table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='note'>No data available.</p>"
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return f'<table class="stats-table"><thead>{header}</thead><tbody>{body}</tbody></table>'


def _img_html(b64: str, alt: str = "") -> str:
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:100%;height:auto;" alt="{alt}">'
    )


def _section(title: str, *content: str, help_key: str | None = None) -> str:
    help_block = (
        f'<div class="help">{_HELP_TEXTS[help_key]}</div>'
        if help_key and help_key in _HELP_TEXTS
        else ""
    )
    body = "\n".join(content)
    return f'<div class="section"><h2>{title}</h2>{help_block}{body}</div>'


def _grid(*items: str) -> str:
    return '<div class="grid-2">' + "".join(f"<div>{it}</div>" for it in items) + "</div>"


# ── Public API ────────────────────────────────────────────────────────────────

def build_html_report(
    run_label: str,
    dataset_stats: list[tuple[str, str]],
    summary_df: pd.DataFrame,
    figures: dict[str, Any],  # key → go.Figure or base64 str or DataFrame
    sample_names: list[str],
) -> str:
    """Assemble and return a complete self-contained HTML report string.

    Parameters
    ----------
    run_label     : Human-readable title/label for this run.
    dataset_stats : List of (label, value) tuples for the dataset overview card.
    summary_df    : Per-sample summary DataFrame from metrics.compute_sample_summary.
    figures       : Dict of named figures/images keyed by section identifier.
                    Values may be go.Figure, base64 str, or pd.DataFrame.
    sample_names  : Ordered list of sample names for per-sample sections.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections: list[str] = []

    # ── Dataset overview ──────────────────────────────────────────────────────
    stats_rows = "".join(
        f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
        for k, v in dataset_stats
    )
    stats_table = f'<table class="stats-table small-table"><tbody>{stats_rows}</tbody></table>'

    sections.append(
        _section(
            "1. Dataset Overview",
            stats_table,
            "<h3>Per-Sample Summary</h3>",
            _table_html(summary_df.drop(columns=["MBR Rate %", "Contam Rate %"], errors="ignore")),
        )
    )

    # ── MS/MS vs MBR ─────────────────────────────────────────────────────────
    s1_parts: list[str] = []
    if "msms_mbr" in figures:
        s1_parts.append(_fig_html(figures["msms_mbr"]))
    if "mbr_rate" in figures:
        s1_parts.append(_fig_html(figures["mbr_rate"]))
    if s1_parts:
        sections.append(_section("2. MS/MS vs MBR", *s1_parts, help_key="msms_mbr"))

    # ── Peptide length ────────────────────────────────────────────────────────
    s2_parts: list[str] = []
    if "length_all" in figures:
        s2_parts.append(_fig_html(figures["length_all"]))
    if "length_per_sample" in figures:
        s2_parts.append(_fig_html(figures["length_per_sample"]))
    if s2_parts:
        sections.append(_section("3. Peptide Length Distribution", *s2_parts, help_key="length"))

    # ── Spectral count ────────────────────────────────────────────────────────
    if "spectral_violin" in figures:
        sections.append(
            _section("4. Spectral Count Distribution", _fig_html(figures["spectral_violin"]))
        )

    # ── MSstats (optional) ────────────────────────────────────────────────────
    ms_parts: list[str] = []
    for key in ("msstats_missing", "msstats_violin", "intensity_corr"):
        if key in figures:
            ms_parts.append(_fig_html(figures[key]))
    if ms_parts:
        sections.append(_section("5. Intensity Distribution (MSstats)", *ms_parts))
    else:
        sections.append(
            _section(
                "5. Intensity Distribution (MSstats)",
                "<p class='note'>MSstats file not uploaded. Upload msstats.csv to enable this section.</p>",
            )
        )

    # ── Contaminants ──────────────────────────────────────────────────────────
    contam_parts: list[str] = []
    if "contam_rate" in figures:
        contam_parts.append(_fig_html(figures["contam_rate"]))
    if "contam_proteins" in figures and isinstance(figures["contam_proteins"], pd.DataFrame):
        contam_parts.append("<h3>Top Contaminant Proteins</h3>")
        contam_parts.append(_table_html(figures["contam_proteins"]))
    if contam_parts:
        sections.append(_section("6. Contaminant Summary", *contam_parts, help_key="contam"))
    else:
        sections.append(
            _section(
                "6. Contaminant Summary",
                "<p class='note'>Protein column not mapped — contaminant analysis unavailable.</p>",
            )
        )

    # ── Peptide overlap ───────────────────────────────────────────────────────
    ov_parts: list[str] = []
    for key in ("jaccard", "shared", "prevalence"):
        if key in figures:
            ov_parts.append(_fig_html(figures[key]))
    if ov_parts:
        sections.append(_section("7. Peptide Overlap", *ov_parts))

    # ── Protein source ────────────────────────────────────────────────────────
    src_parts: list[str] = []
    for key in ("source_pie", "source_per_sample", "genes_per_sample"):
        if key in figures:
            src_parts.append(_fig_html(figures[key]))
    if src_parts:
        sections.append(_section("8. Protein Source", *src_parts))
    else:
        sections.append(
            _section(
                "8. Protein Source",
                "<p class='note'>Protein column not mapped — source analysis unavailable.</p>",
            )
        )

    # ── Amino acid composition ────────────────────────────────────────────────
    aa_parts: list[str] = []
    if "aa_heatmap" in figures:
        aa_parts.append(_fig_html(figures["aa_heatmap"]))
    if "aa_freq" in figures:
        aa_parts.append(_fig_html(figures["aa_freq"]))
    if aa_parts:
        sections.append(_section("9. Amino Acid Composition", *aa_parts))

    # ── Sequence logo ─────────────────────────────────────────────────────────
    if "seq_logo" in figures and figures["seq_logo"]:
        sections.append(
            _section(
                "10. Sequence Logo (9-mer Binding Motif)",
                _img_html(figures["seq_logo"], "Sequence Logo"),
                help_key="aa_motif",
            )
        )

    # ── Charge state ─────────────────────────────────────────────────────────
    chg_parts: list[str] = []
    for key in ("charge_pie", "charge_per_sample"):
        if key in figures:
            chg_parts.append(_fig_html(figures[key]))
    if chg_parts:
        sections.append(_section("11. Charge State Distribution", *chg_parts))
    else:
        sections.append(
            _section(
                "11. Charge State Distribution",
                "<p class='note'>Charge column not mapped — charge analysis unavailable.</p>",
            )
        )

    # ── PCA / clustering ──────────────────────────────────────────────────────
    pca_parts: list[str] = []
    for key in ("pca", "pca_variance", "correlation"):
        if key in figures:
            pca_parts.append(_fig_html(figures[key]))
    if pca_parts:
        sections.append(_section("12. PCA & Sample Clustering", *pca_parts, help_key="pca"))
    else:
        sections.append(
            _section(
                "12. PCA & Sample Clustering",
                "<p class='note'>Intensity columns not available for ≥2 samples — PCA unavailable.</p>",
            )
        )

    # ── Per-sample detail pages ───────────────────────────────────────────────
    per_sample_html = _build_per_sample_sections(figures, sample_names)
    if per_sample_html:
        sections.append(
            _section("13. Per-Sample Detail", per_sample_html)
        )

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Immunopeptidomics QC — {run_label}</title>
{PLOTLY_CDN}
{_CSS}
</head>
<body>
<div class="wrap">
  <h1>Immunopeptidomics QC Report</h1>
  <p class="meta">Run: <strong>{run_label}</strong> &nbsp;·&nbsp; Generated: {timestamp}</p>
  {body}
</div>
</body>
</html>"""


def _build_per_sample_sections(
    figures: dict[str, Any], sample_names: list[str]
) -> str:
    """Build collapsible per-sample detail blocks."""
    blocks: list[str] = []
    for s in sample_names:
        sid = _safe_id(s)
        parts: list[str] = []
        for suffix in ("len", "src", "chg", "sc", "int", "overlap", "prot", "logo", "heatmap"):
            key = f"sample_{sid}_{suffix}"
            if key not in figures:
                continue
            val = figures[key]
            if isinstance(val, go.Figure):
                parts.append(_fig_html(val))
            elif isinstance(val, str) and val:
                # base64 image or raw HTML
                if val.startswith("iVBOR") or val.startswith("/9j/"):
                    parts.append(_img_html(val, f"{s} logo"))
                else:
                    parts.append(val)

        if parts:
            inner = "\n".join(parts)
            blocks.append(
                f'<details><summary><strong>{s}</strong></summary>'
                f'<div style="padding:.8rem 0">{inner}</div></details>'
            )

    return "\n".join(blocks)


def build_csv_summary(summary_df: pd.DataFrame) -> str:
    """Return the per-sample summary as a CSV string."""
    return summary_df.to_csv(index=False)
