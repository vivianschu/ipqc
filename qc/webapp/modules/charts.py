"""Chart generation functions for immunopeptidomics QC.

Every function accepts pre-computed data (DataFrames, arrays, dicts from
metrics.py) and returns either a ``plotly.graph_objects.Figure`` or a
base64-encoded PNG string for matplotlib figures.

None of these functions perform analysis; they only visualise.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .mapping import SampleDef
from .metrics import AAS, HUMAN_BG, parse_charges

SAMPLE_COLORS: list[str] = (
    px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
)


def _bar_color(length: int) -> str:
    return "#2ca02c" if 8 <= length <= 11 else "#aec7e8"


# ── Section 1: MS/MS vs MBR ──────────────────────────────────────────────────

def chart_msms_mbr(summary_df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of MS/MS vs MBR detections per sample."""
    fig = go.Figure()
    fig.add_bar(
        name="MS/MS",
        x=summary_df["Sample"],
        y=summary_df["MS/MS"],
        marker_color="#4C72B0",
    )
    fig.add_bar(
        name="MBR",
        x=summary_df["Sample"],
        y=summary_df["MBR"],
        marker_color="#DD8452",
    )
    fig.update_layout(
        barmode="stack",
        title="Detected Peptides per Sample (MS/MS vs MBR)",
        xaxis_title="Sample",
        yaxis_title="Peptide Count",
        legend_title="Detection Type",
        template="plotly_white",
        height=420,
    )
    return fig


def chart_mbr_rate(summary_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of MBR rate per sample with 30% warning threshold."""
    df = summary_df.sort_values("MBR Rate %", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df["MBR Rate %"],
            y=df["Sample"],
            orientation="h",
            marker=dict(
                color=df["MBR Rate %"],
                colorscale="RdYlGn_r",
                cmin=0,
                cmax=50,
                colorbar=dict(title="MBR %"),
            ),
            text=df["MBR Rate"],
            textposition="outside",
        )
    )
    fig.add_vline(
        x=30,
        line_dash="dash",
        line_color="red",
        annotation_text="30% threshold",
        annotation_position="top right",
    )
    fig.update_layout(
        title="MBR Rate per Sample (high MBR may indicate quality issues)",
        xaxis_title="MBR Rate (%)",
        yaxis_title="Sample",
        template="plotly_white",
        height=max(300, 60 + 40 * len(df)),
        margin=dict(r=80),
    )
    return fig


# ── Section 2: Peptide length distribution ───────────────────────────────────

def chart_length_all(df: pd.DataFrame) -> go.Figure:
    """Bar chart of peptide length distribution across all peptides."""
    len_counts = df["_length"].value_counts().sort_index().dropna()
    lengths = [int(l) for l in len_counts.index]
    counts = len_counts.values.tolist()
    colors = [_bar_color(l) for l in lengths]

    fig = go.Figure()
    fig.add_bar(x=lengths, y=counts, marker_color=colors, showlegend=False)
    for label, color in [("MHC-I (8–11 aa)", "#2ca02c"), ("Other", "#aec7e8")]:
        fig.add_bar(x=[None], y=[None], name=label, marker_color=color, showlegend=True)
    if lengths:
        fig.add_vrect(x0=7.5, x1=11.5, fillcolor="#2ca02c", opacity=0.07, line_width=0, layer="below")
        fig.update_layout(
            xaxis=dict(tickmode="linear", tick0=min(lengths), dtick=1),
        )
    fig.update_layout(
        title="Peptide Length Distribution (All Peptides)",
        xaxis_title="Peptide Length (aa)",
        yaxis_title="Number of Peptides",
        template="plotly_white",
        legend_title="MHC Class",
        height=420,
    )
    return fig


def chart_length_per_sample(df: pd.DataFrame, samples: list[SampleDef]) -> go.Figure:
    """Stacked bar chart of peptide lengths 8–24 aa per sample (detected only)."""
    focus_lengths = list(range(8, 25))
    rows: list[dict[str, Any]] = []
    for sd in samples:
        if not sd.match_col or sd.match_col not in df.columns:
            continue
        detected = df[df[sd.match_col] != "unmatched"]
        for l in focus_lengths:
            rows.append({
                "Sample": sd.name,
                "Length": str(l),
                "Count": int((detected["_length"] == l).sum()),
            })

    if not rows:
        return go.Figure()

    psl_df = pd.DataFrame(rows)
    length_colors = px.colors.sequential.Viridis_r
    fig = go.Figure()
    for i, l in enumerate(focus_lengths):
        sub = psl_df[psl_df["Length"] == str(l)]
        fig.add_bar(
            name=f"{l} aa",
            x=sub["Sample"],
            y=sub["Count"],
            marker_color=length_colors[i % len(length_colors)],
        )
    fig.update_layout(
        barmode="stack",
        title="Peptide Length Distribution per Sample (Detected Peptides, 8–24 aa)",
        xaxis_title="Sample",
        yaxis_title="Peptide Count",
        legend_title="Length",
        template="plotly_white",
        height=450,
    )
    return fig


# ── Section 3: Spectral count distribution ───────────────────────────────────

def chart_spectral_violin(df: pd.DataFrame, samples: list[SampleDef]) -> go.Figure:
    """Violin plot of spectral count distribution per sample (non-zero only)."""
    fig = go.Figure()
    for i, sd in enumerate(samples):
        if not sd.match_col or sd.match_col not in df.columns:
            continue
        if not sd.spectral_col or sd.spectral_col not in df.columns:
            continue
        mask = df[sd.match_col] != "unmatched"
        vals = df.loc[mask, sd.spectral_col]
        vals = vals[vals > 0]
        if vals.empty:
            continue
        fig.add_trace(
            go.Violin(
                y=vals,
                name=sd.name,
                box_visible=True,
                meanline_visible=True,
                points=False,
                marker_color=SAMPLE_COLORS[i % len(SAMPLE_COLORS)],
            )
        )
    fig.update_layout(
        title="Spectral Count Distribution per Sample (Detected, Non-Zero)",
        yaxis_title="Spectral Count",
        yaxis_type="log",
        xaxis_title="Sample",
        showlegend=False,
        template="plotly_white",
        height=460,
    )
    return fig


# ── Section 4: MSstats intensity (optional) ──────────────────────────────────

def chart_msstats_missing(ms_df: pd.DataFrame) -> go.Figure:
    """Missing value rate per run from MSstats data."""
    run_stats = (
        ms_df.groupby("Run")["Intensity"]
        .agg(total="count", missing=lambda x: x.isna().sum())
        .reset_index()
    )
    run_stats["missing_pct"] = run_stats["missing"] / run_stats["total"] * 100
    run_stats = run_stats.sort_values("missing_pct", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=run_stats["missing_pct"],
            y=run_stats["Run"],
            orientation="h",
            marker=dict(
                color=run_stats["missing_pct"],
                colorscale="RdYlGn_r",
                cmin=0,
                cmax=100,
                colorbar=dict(title="Missing %"),
            ),
            text=run_stats["missing_pct"].map("{:.1f}%".format),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Missing Value Rate per Run (MSstats)",
        xaxis_title="Missing Intensity (%)",
        yaxis_title="Run",
        template="plotly_white",
        height=max(400, 60 + 30 * len(run_stats)),
        margin=dict(r=80),
    )
    return fig


def chart_msstats_intensity_violin(ms_df: pd.DataFrame) -> go.Figure:
    """Log₂ intensity violin per condition from MSstats data."""
    ms_df = ms_df.copy()
    ms_df["log2_intensity"] = np.where(
        ms_df["Intensity"].notna() & (ms_df["Intensity"] > 0),
        np.log2(ms_df["Intensity"]),
        np.nan,
    )
    conditions = sorted(ms_df["Condition"].dropna().unique().tolist())
    fig = go.Figure()
    for i, cond in enumerate(conditions):
        vals = ms_df.loc[ms_df["Condition"] == cond, "log2_intensity"].dropna()
        fig.add_trace(
            go.Violin(
                y=vals,
                name=cond,
                box_visible=True,
                meanline_visible=True,
                points=False,
                marker_color=SAMPLE_COLORS[i % len(SAMPLE_COLORS)],
            )
        )
    fig.update_layout(
        title="Log₂ Intensity Distribution per Condition (MSstats)",
        yaxis_title="Log₂ Intensity",
        xaxis_title="Condition",
        showlegend=False,
        template="plotly_white",
        height=460,
    )
    return fig


def chart_sample_intensity_correlation(
    df: pd.DataFrame, samples: list[SampleDef]
) -> go.Figure:
    """Pairwise scatter plots of log₂ intensity between all sample pairs."""
    valid = [sd for sd in samples if sd.intensity_col and sd.intensity_col in df.columns]
    if len(valid) < 2:
        return go.Figure()

    from itertools import combinations
    from plotly.subplots import make_subplots
    from scipy.stats import pearsonr

    pairs = list(combinations(range(len(valid)), 2))
    n_pairs = len(pairs)
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    titles = [f"{valid[a].name} vs {valid[b].name}" for a, b in pairs]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles)

    for idx, (a, b) in enumerate(pairs):
        row = idx // ncols + 1
        col = idx % ncols + 1
        ca, cb = valid[a].intensity_col, valid[b].intensity_col
        sub = df[[ca, cb]].copy()
        sub = sub[(sub[ca] > 0) & (sub[cb] > 0)]
        if sub.empty:
            continue
        xa = np.log2(sub[ca])
        ya = np.log2(sub[cb])
        r, _ = pearsonr(xa, ya)

        mn = min(xa.min(), ya.min())
        mx = max(xa.max(), ya.max())
        pad = (mx - mn) * 0.05
        axis_range = [mn - pad, mx + pad]

        fig.add_trace(
            go.Scatter(
                x=xa, y=ya, mode="markers",
                marker=dict(size=3, opacity=0.4, color=SAMPLE_COLORS[idx % len(SAMPLE_COLORS)]),
                name=f"r={r:.3f}", showlegend=True,
                cliponaxis=True,
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                       line=dict(color="red", dash="dash", width=1), showlegend=False),
            row=row, col=col,
        )
        fig.update_xaxes(range=axis_range, row=row, col=col)
        fig.update_yaxes(range=axis_range, row=row, col=col)

    fig.update_layout(
        title="Sample Intensity Correlations (Log₂, peptides detected in both samples)",
        template="plotly_white",
        height=max(380, 380 * nrows),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Log₂ Intensity")
    fig.update_yaxes(title_text="Log₂ Intensity")
    return fig


# ── Section 5: Contaminants ───────────────────────────────────────────────────

def chart_contaminant_rate(contam_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of contaminant rate per sample."""
    max_pct = contam_df["Contam %"].max()
    fig = go.Figure(
        go.Bar(
            x=contam_df["Contam %"],
            y=contam_df["Sample"],
            orientation="h",
            marker=dict(
                color=contam_df["Contam %"],
                colorscale="OrRd",
                cmin=0,
                cmax=max(max_pct * 1.2, 1.0),
                colorbar=dict(title="Contam %"),
            ),
            text=contam_df["Contam %"].map("{:.2f}%".format),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Contaminant Rate per Sample (% of Detected Peptides)",
        xaxis_title="Contaminant Peptides (%)",
        yaxis_title="Sample",
        template="plotly_white",
        height=max(300, 60 + 40 * len(contam_df)),
        margin=dict(r=80),
    )
    return fig


# ── Section 6: Peptide overlap ────────────────────────────────────────────────

def chart_jaccard_heatmap(jaccard: np.ndarray, sample_names: list[str]) -> go.Figure:
    """Annotated heatmap of pairwise Jaccard similarity."""
    fig = go.Figure(
        go.Heatmap(
            z=jaccard,
            x=sample_names,
            y=sample_names,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=np.round(jaccard, 2),
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="Jaccard"),
        )
    )
    fig.update_layout(
        title="Pairwise Jaccard Similarity (Detected Peptides)",
        xaxis=dict(tickangle=45),
        template="plotly_white",
        height=max(400, 80 + 40 * len(sample_names)),
    )
    return fig


def chart_shared_heatmap(shared: np.ndarray, sample_names: list[str]) -> go.Figure:
    """Annotated heatmap of absolute shared peptide counts."""
    fig = go.Figure(
        go.Heatmap(
            z=shared,
            x=sample_names,
            y=sample_names,
            colorscale="Purples",
            text=shared,
            texttemplate="%{text}",
            textfont=dict(size=8),
            colorbar=dict(title="Shared<br>Peptides"),
        )
    )
    fig.update_layout(
        title="Shared Peptide Counts Between Samples",
        xaxis=dict(tickangle=45),
        template="plotly_white",
        height=max(400, 80 + 40 * len(sample_names)),
    )
    return fig


def chart_peptide_prevalence(
    detected_sets: dict[str, set[str]], all_peptides: pd.Series
) -> go.Figure:
    """Bar chart: how many samples each peptide is detected in."""
    pep_counts = pd.Series(
        {pep: sum(1 for s in detected_sets.values() if pep in s) for pep in all_peptides}
    )
    prev = pep_counts.value_counts().sort_index()
    colors = px.colors.sequential.Viridis[
        :: max(1, len(px.colors.sequential.Viridis) // max(len(prev), 1))
    ]
    fig = go.Figure(
        go.Bar(
            x=prev.index,
            y=prev.values,
            marker_color=colors[: len(prev)],
            text=prev.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Peptide Prevalence Across Samples",
        xaxis_title="Number of Samples Peptide Is Detected In",
        yaxis_title="Number of Peptides",
        xaxis=dict(tickmode="linear", dtick=1),
        template="plotly_white",
        height=400,
    )
    return fig


# ── Section 7: Protein source ─────────────────────────────────────────────────

def chart_protein_source_pie(df: pd.DataFrame) -> go.Figure:
    """Donut chart of protein source breakdown across all peptides."""
    source_counts = df["_source"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=source_counts.index,
            values=source_counts.values,
            hole=0.4,
            marker=dict(colors=["#4C72B0", "#DD8452", "#c44e52", "#8c8c8c"]),
            textinfo="label+percent+value",
        )
    )
    fig.update_layout(
        title="Protein Source Distribution (All Peptides)",
        template="plotly_white",
        height=400,
    )
    return fig


def chart_protein_source_per_sample(
    df: pd.DataFrame, samples: list[SampleDef]
) -> go.Figure:
    """Stacked bar chart of protein source breakdown per sample."""
    source_order = ["SwissProt (sp)", "TrEMBL (tr)", "Contaminant", "Other", "Unknown"]
    colors_map = {
        "SwissProt (sp)": "#4C72B0",
        "TrEMBL (tr)": "#DD8452",
        "Contaminant": "#c44e52",
        "Other": "#8c8c8c",
        "Unknown": "#cccccc",
    }
    fig = go.Figure()
    for src in source_order:
        ys, xs = [], []
        for sd in samples:
            if not sd.match_col or sd.match_col not in df.columns:
                continue
            detected = df[df[sd.match_col] != "unmatched"]
            ys.append(int((detected["_source"] == src).sum()))
            xs.append(sd.name)
        if any(y > 0 for y in ys):
            fig.add_bar(name=src, x=xs, y=ys, marker_color=colors_map.get(src, "#aaaaaa"))
    fig.update_layout(
        barmode="stack",
        title="Protein Source per Sample (Detected Peptides)",
        xaxis_title="Sample",
        yaxis_title="Peptide Count",
        legend_title="Source",
        template="plotly_white",
        height=440,
    )
    return fig


def chart_genes_per_sample(df: pd.DataFrame, samples: list[SampleDef]) -> go.Figure:
    """Horizontal bar chart of unique canonical genes per sample."""
    if "_gene" not in df.columns:
        return go.Figure()

    rows: list[dict[str, Any]] = []
    for sd in samples:
        if not sd.match_col or sd.match_col not in df.columns:
            continue
        detected = df[(df[sd.match_col] != "unmatched") & (~df["_is_contam"])]
        genes = detected["_gene"].dropna()
        genes = genes[genes.astype(str).str.strip() != ""]
        rows.append({"Sample": sd.name, "Unique Genes": int(genes.nunique())})

    if not rows:
        return go.Figure()

    gene_df = pd.DataFrame(rows).sort_values("Unique Genes", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=gene_df["Unique Genes"],
            y=gene_df["Sample"],
            orientation="h",
            marker_color="#4C72B0",
            text=gene_df["Unique Genes"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Unique Canonical Genes per Sample",
        xaxis_title="Number of Unique Genes",
        yaxis_title="Sample",
        template="plotly_white",
        height=max(300, 60 + 40 * len(gene_df)),
        margin=dict(r=60),
    )
    return fig


# ── Section 8: Amino acid composition ────────────────────────────────────────

def chart_aa_heatmap(pos_freq: np.ndarray, mer_count: int) -> go.Figure:
    """Heatmap of amino acid frequency by position for fixed-length peptides."""
    n_pos = pos_freq.shape[0]
    fig = go.Figure(
        go.Heatmap(
            z=pos_freq.T,
            x=[f"P{i + 1}" for i in range(n_pos)],
            y=AAS,
            colorscale="RdBu_r",
            zmid=1 / 20,
            colorbar=dict(title="Frequency"),
            text=np.round(pos_freq.T, 3),
            texttemplate="%{text}",
            textfont=dict(size=8),
        )
    )
    fig.update_layout(
        title=f"Amino Acid Position Frequency — {n_pos}-mers (n={mer_count:,})",
        xaxis_title="Position",
        yaxis_title="Amino Acid",
        template="plotly_white",
        height=600,
    )
    return fig


def chart_aa_frequency(all_aa_freq: pd.Series) -> go.Figure:
    """Grouped bar comparing immunopeptidome AA frequency vs human proteome background."""
    bg_freq = pd.Series(HUMAN_BG).reindex(AAS, fill_value=0)
    fig = go.Figure()
    fig.add_bar(name="Immunopeptidome", x=AAS, y=all_aa_freq.values, marker_color="#4C72B0")
    fig.add_bar(
        name="Human proteome (background)",
        x=AAS,
        y=bg_freq.values,
        marker_color="#DD8452",
        opacity=0.7,
    )
    fig.update_layout(
        barmode="group",
        title="Amino Acid Frequency: Immunopeptidome vs Human Proteome Background",
        xaxis_title="Amino Acid",
        yaxis_title="Relative Frequency",
        legend_title="Dataset",
        template="plotly_white",
        height=400,
    )
    return fig


# ── Section 9: Sequence logo ──────────────────────────────────────────────────

def chart_sequence_logo(pos_freq: np.ndarray, mer_count: int) -> str:
    """Return a base64-encoded PNG of frequency + information content logos.

    Highlights P2 and P9 anchor positions for 9-mers.
    """
    n_pos = pos_freq.shape[0]
    logo_df = pd.DataFrame(pos_freq, columns=AAS, index=list(range(n_pos)))
    info_df = logomaker.transform_matrix(
        logo_df, from_type="probability", to_type="information"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 3.5))
    fig.patch.set_facecolor("white")

    for ax, matrix, title, ylabel in [
        (axes[0], logo_df, f"Frequency Logo — {n_pos}-mers (n={mer_count:,})", "Frequency"),
        (axes[1], info_df, f"Information Content Logo — {n_pos}-mers (n={mer_count:,})", "Bits"),
    ]:
        logomaker.Logo(matrix, ax=ax, color_scheme="chemistry")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Position", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if n_pos == 9:
            for pos in [1, 8]:  # P2 and P9 anchor residues
                ax.axvspan(pos - 0.5, pos + 0.5, color="gold", alpha=0.18, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=2.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ── Section 10: Charge state ──────────────────────────────────────────────────

def chart_charge_pie(charge_series: pd.Series) -> go.Figure:
    """Donut chart of overall charge state distribution."""
    fig = go.Figure(
        go.Pie(
            labels=[f"{c}+" for c in charge_series.index],
            values=charge_series.values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Plotly),
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        title="Overall Charge State Distribution (All Peptides)",
        template="plotly_white",
        height=380,
    )
    return fig


def chart_charge_per_sample(
    df: pd.DataFrame,
    samples: list[SampleDef],
    charge_col: str,
    charge_vals: list[int],
) -> go.Figure:
    """Stacked bar chart of charge state distribution per sample."""
    fig = go.Figure()
    for c in charge_vals:
        ys, xs = [], []
        for sd in samples:
            if not sd.match_col or sd.match_col not in df.columns:
                continue
            detected = df[df[sd.match_col] != "unmatched"]
            count = int(detected[charge_col].apply(lambda v: c in parse_charges(v)).sum())
            ys.append(count)
            xs.append(sd.name)
        fig.add_bar(name=f"{c}+", x=xs, y=ys)
    fig.update_layout(
        barmode="stack",
        title="Charge State Distribution per Sample (Detected Peptides)",
        xaxis_title="Sample",
        yaxis_title="Peptide Count",
        legend_title="Charge",
        template="plotly_white",
        height=430,
    )
    return fig


# ── Section 11: PCA / clustering ─────────────────────────────────────────────

def chart_pca(pca_data: dict[str, Any]) -> go.Figure:
    """Scatter plot of samples in PCA space (PC1 vs PC2)."""
    samples = pca_data["samples"]
    coords = pca_data["coords"]
    var_exp = pca_data["var_exp"]

    fig = go.Figure()
    for i, s in enumerate(samples):
        fig.add_trace(
            go.Scatter(
                x=[coords[i, 0]],
                y=[coords[i, 1]],
                mode="markers+text",
                name=s,
                text=[s],
                textposition="top center",
                marker=dict(size=12, color=SAMPLE_COLORS[i % len(SAMPLE_COLORS)]),
                showlegend=False,
            )
        )
    pc2_label = f"PC2 ({var_exp[1]:.1f}% variance)" if len(var_exp) > 1 else "PC2"
    fig.update_layout(
        title="PCA — Samples (peptides detected in ≥ min threshold, min-value imputed)",
        xaxis_title=f"PC1 ({var_exp[0]:.1f}% variance)",
        yaxis_title=pc2_label,
        template="plotly_white",
        height=500,
    )
    return fig


def chart_pca_variance(pca_data: dict[str, Any]) -> go.Figure:
    """Bar chart of explained variance per principal component."""
    var_exp = pca_data["var_exp"]
    n_comp = pca_data["n_comp"]
    fig = go.Figure(
        go.Bar(
            x=[f"PC{i + 1}" for i in range(n_comp)],
            y=var_exp,
            marker_color="#4C72B0",
            text=[f"{v:.1f}%" for v in var_exp],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        template="plotly_white",
        height=380,
    )
    return fig


def chart_correlation_heatmap(pca_data: dict[str, Any]) -> go.Figure:
    """Hierarchically clustered sample Pearson correlation heatmap."""
    corr = pca_data["corr_ordered"]
    sord = pca_data["samples_ordered"]
    fig = go.Figure(
        go.Heatmap(
            z=np.round(corr, 3),
            x=sord,
            y=sord,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=np.round(corr, 2),
            texttemplate="%{text}",
            textfont=dict(size=8),
            colorbar=dict(title="Pearson r"),
        )
    )
    fig.update_layout(
        title="Sample Correlation Heatmap (hierarchical clustering order, log₂ intensity)",
        xaxis=dict(tickangle=45),
        template="plotly_white",
        height=max(400, 80 + 40 * len(sord)),
    )
    return fig


# ── Per-sample charts ─────────────────────────────────────────────────────────

def chart_sample_overlap_bar(
    sample_name: str,
    detected_sets: dict[str, set[str]],
    color: str,
) -> go.Figure:
    """Horizontal bar chart of shared peptides between one sample and all others."""
    s_peps = detected_sets.get(sample_name, set())
    rows = sorted(
        [{"sample": s2, "shared": len(s_peps & s2_peps)}
         for s2, s2_peps in detected_sets.items() if s2 != sample_name],
        key=lambda r: r["shared"],
    )
    if not rows:
        return go.Figure()
    ov_df = pd.DataFrame(rows)
    fig = go.Figure(
        go.Bar(
            x=ov_df["shared"],
            y=ov_df["sample"],
            orientation="h",
            marker_color=color,
            text=ov_df["shared"].map("{:,}".format),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Peptide Overlap with Other Samples — {sample_name}",
        xaxis_title="Shared Peptides",
        yaxis_title="Sample",
        template="plotly_white",
        height=max(300, 80 + 40 * len(rows)),
        margin=dict(r=80),
    )
    return fig


def chart_top_proteins(
    df: pd.DataFrame, sample_name: str, color: str
) -> go.Figure:
    """Horizontal bar chart of top 15 proteins by peptide count for one sample."""
    if "_protein" not in df.columns or "_is_contam" not in df.columns:
        return go.Figure()
    top = (
        df.loc[~df["_is_contam"], "_protein"]
        .value_counts()
        .head(15)
        .sort_values(ascending=True)
    )
    if top.empty:
        return go.Figure()
    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=top.index.tolist(),
            orientation="h",
            marker_color=color,
            text=top.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top 15 Proteins by Peptide Count — {sample_name}",
        xaxis_title="Peptide Count",
        yaxis_title="Protein",
        template="plotly_white",
        height=500,
        margin=dict(r=80),
    )
    return fig


def chart_sample_spectral_histogram(
    df: pd.DataFrame, sd: SampleDef, color: str
) -> go.Figure:
    """Histogram of spectral count distribution for a single sample."""
    if not sd.match_col or not sd.spectral_col:
        return go.Figure()
    mask = df[sd.match_col] != "unmatched"
    vals = df.loc[mask, sd.spectral_col]
    vals = vals[vals > 0]
    fig = go.Figure(
        go.Histogram(x=vals, nbinsx=40, marker_color=color, opacity=0.85)
    )
    fig.update_layout(
        title=f"Spectral Count Distribution — {sd.name}",
        xaxis_title="Spectral Count",
        yaxis_title="# Peptides",
        xaxis_type="log",
        template="plotly_white",
        height=340,
    )
    return fig


def chart_sample_intensity_histogram(
    df: pd.DataFrame, sd: SampleDef, color: str
) -> go.Figure:
    """Histogram of log₂ intensity for a single sample."""
    if not sd.intensity_col or sd.intensity_col not in df.columns:
        return go.Figure()
    if not sd.match_col or sd.match_col not in df.columns:
        return go.Figure()
    mask = df[sd.match_col] != "unmatched"
    raw = df.loc[mask, sd.intensity_col]
    vals = np.log2(raw[raw > 0])
    if vals.empty:
        return go.Figure()
    fig = go.Figure(
        go.Histogram(x=vals, nbinsx=40, marker_color=color, opacity=0.85)
    )
    fig.update_layout(
        title=f"Log₂ Intensity Distribution — {sd.name}",
        xaxis_title="Log₂ Intensity",
        yaxis_title="# Peptides",
        template="plotly_white",
        height=340,
    )
    return fig


# ── Venn diagram ──────────────────────────────────────────────────────────────

def chart_venn2(
    set_a: set[str],
    set_b: set[str],
    label_a: str,
    label_b: str,
    title: str,
) -> str:
    """Return a base64-encoded PNG of a two-set Venn diagram."""
    from matplotlib.patches import Circle

    a_only = len(set_a - set_b)
    b_only = len(set_b - set_a)
    ab = len(set_a & set_b)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Circle((0.37, 0.5), 0.27, color="#4C72B0", alpha=0.45))
    ax.add_patch(Circle((0.63, 0.5), 0.27, color="#DD8452", alpha=0.45))

    kw = dict(ha="center", va="center", fontsize=12, color="#1a1f36")
    ax.text(0.22, 0.5, f"{a_only:,}", **kw)
    ax.text(0.78, 0.5, f"{b_only:,}", **kw)
    ax.text(0.50, 0.5, f"{ab:,}", **kw)
    ax.text(0.29, 0.01, label_a, ha="center", va="center", fontsize=9, color="#4C72B0")
    ax.text(0.72, 0.01, label_b, ha="center", va="center", fontsize=9, color="#DD8452")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color="#444", pad=6)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64
