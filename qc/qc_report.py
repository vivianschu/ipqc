#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "plotly", "scipy", "numpy", "matplotlib", "logomaker", "scikit-learn", "requests"]
# ///

"""
Immunopeptidomics QC Report — Sections 1–5
Reads:  ../circrna/data/ip/combined_peptide.tsv
        ../circrna/data/ip/msstats.csv
Output: qc_report.html
"""

import re
import io
import json
import argparse
import base64
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── paths (defaults) ─────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
_DATA     = HERE / "../circrna/data/ip/combined_peptide.tsv"
_MSSTATS  = HERE / "../circrna/data/ip/msstats.csv"
_OUT      = HERE / "qc_report.html"

# ── CLI ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Immunopeptidomics QC Report")
_parser.add_argument("--data",    default=str(_DATA),    help="Path to combined_peptide.tsv")
_parser.add_argument("--msstats", default=str(_MSSTATS), help="Path to msstats.csv")
_parser.add_argument("--out",     default=str(_OUT),     help="Output HTML path")
_parser.add_argument("--label",   default=None,          help="Human-readable run label")
_parser.add_argument("--state",   default=str(HERE / "qc_state.json"),
                     help="Path to run-history state file")
_parser.add_argument("--reset",   action="store_true",   help="Clear all prior run history")
args = _parser.parse_args()

DATA    = Path(args.data)
MSSTATS = Path(args.msstats)
OUT     = Path(args.out)

# ── load data ────────────────────────────────────────────────────────────────
print("Loading data…")
df = pd.read_csv(DATA, sep="\t", low_memory=False)

# Derive sample names from Match Type columns
match_cols = [c for c in df.columns if c.endswith(" Match Type")]
samples = [c.replace(" Match Type", "") for c in match_cols]
spectral_cols  = [f"{s} Spectral Count" for s in samples]
intensity_cols = [f"{s} Intensity"      for s in samples]

# ── helpers ──────────────────────────────────────────────────────────────────
def is_contaminant(protein: str) -> bool:
    return bool(re.search(r"Cont_", str(protein)))

df["_is_contam"] = df["Protein"].apply(is_contaminant)

PLOTLY_CDN = (
    '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"'
    ' charset="utf-8"></script>'
)

# colour palette for samples
SAMPLE_COLORS = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24


def venn2_img(set_a, set_b, label_a="A", label_b="B", title="") -> str:
    """Return an HTML snippet with a 2-set Venn diagram (title rendered as HTML)."""
    from matplotlib.patches import Circle
    a_only = len(set_a - set_b)
    b_only = len(set_b - set_a)
    ab     = len(set_a & set_b)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Circle((0.37, 0.5), 0.27, color="#4C72B0", alpha=0.45))
    ax.add_patch(Circle((0.63, 0.5), 0.27, color="#DD8452", alpha=0.45))

    kw = dict(ha="center", va="center", fontsize=12, color="#1a1f36")
    ax.text(0.22, 0.5,  f"{a_only:,}",  **kw)
    ax.text(0.78, 0.5,  f"{b_only:,}",  **kw)
    ax.text(0.50, 0.5,  f"{ab:,}",      **kw)

    ax.text(0.29, 0.01, label_a, ha="center", va="center", fontsize=9,
            color="#4C72B0")
    ax.text(0.72, 0.01, label_b, ha="center", va="center", fontsize=9,
            color="#DD8452")

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color="#444", pad=6)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    return (f'<div style="display:inline-block;text-align:center;">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:440px;width:100%;height:auto;" alt="Venn diagram">'
            f'</div>')


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class QCState:
    """JSON-backed store that accumulates QC metrics across runs."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._data: dict = {"runs": []}

    def load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def reset(self) -> None:
        self._data = {"runs": []}

    def add_run(self, run: dict) -> None:
        self._data["runs"].append(run)

    def runs(self) -> list:
        return self._data["runs"]

    def previous_run(self) -> dict | None:
        """Most recent run already in history (before current is appended)."""
        runs = self._data["runs"]
        return runs[-1] if runs else None


def diff_badge(current, prev, higher_is_better: bool = True, is_pct: bool = False) -> str:
    """Return a coloured HTML delta badge, or '' if no change / no prior value."""
    if prev is None or current is None:
        return ""
    try:
        delta = float(current) - float(prev)
    except (TypeError, ValueError):
        return ""
    if abs(delta) < 0.001:
        return ""
    sign  = "\u2191" if delta > 0 else "\u2193"
    good  = (delta > 0) == higher_is_better
    cls   = "diff-good" if good else "diff-bad"
    if is_pct:
        val = f"{abs(delta):.1f}\u00a0pp"
    elif isinstance(current, float) or isinstance(prev, float):
        val = f"{abs(delta):.1f}"
    else:
        val = f"{abs(int(delta)):,}"
    return f'<span class="diff-badge {cls}">{sign}\u00a0{val}</span>'


# ── run state ─────────────────────────────────────────────────────────────────
state = QCState(args.state)
if not args.reset:
    state.load()
prev_run: dict | None = state.previous_run()   # snapshot before this run is added


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Summary Table
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 1 — Summary Table…")

rows = []
for s, mc, sc in zip(samples, match_cols, spectral_cols):
    col = df[mc]
    msms  = (col == "MS/MS").sum()
    mbr   = (col == "MBR").sum()
    total = msms + mbr
    # contaminant: detected (not unmatched) AND contaminant protein
    contam = ((col != "unmatched") & df["_is_contam"]).sum()
    mbr_pct = f"{mbr/total*100:.1f}%" if total > 0 else "—"
    contam_pct = f"{contam/total*100:.1f}%" if total > 0 else "—"
    rows.append(dict(
        Sample=s,
        MS_MS=msms,
        MBR=mbr,
        Total_Detected=total,
        MBR_Rate=mbr_pct,
        Contaminants=contam,
        Contam_Rate=contam_pct,
    ))

summary_df = pd.DataFrame(rows)

# dataset-level stats
total_peptides   = len(df)
unique_canonical = (~df["_is_contam"]).sum()
pep_8_11 = ((df["Peptide Length"] >= 8) & (df["Peptide Length"] <= 11)).sum()
pep_13_25 = ((df["Peptide Length"] >= 13) & (df["Peptide Length"] <= 25)).sum()

dataset_stats = [
    ("Total unique peptides", f"{total_peptides:,}"),
    ("Canonical (non-contaminant)", f"{unique_canonical:,}"),
    ("MHC-I length range (8–11 aa)", f"{pep_8_11:,} ({pep_8_11/total_peptides*100:.1f}%)"),
    ("MHC-II length range (13–25 aa)", f"{pep_13_25:,} ({pep_13_25/total_peptides*100:.1f}%)"),
    ("Number of samples", str(len(samples))),
]

def dataset_stats_html(stats):
    rows_html = "".join(
        f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
        for k, v in stats
    )
    return f"""
<table class="stats-table small-table">
  <tbody>{rows_html}</tbody>
</table>"""

def summary_table_html(df_s):
    header = "<tr>" + "".join(
        f"<th>{c.replace('_',' ')}</th>" for c in df_s.columns
    ) + "</tr>"
    body = ""
    for _, r in df_s.iterrows():
        cells = "".join(f"<td>{v:,}" if isinstance(v, int) else f"<td>{v}" for v in r)
        body += f"<tr>{cells}</tr>"
    return f"""
<table class="stats-table">
  <thead>{header}</thead>
  <tbody>{body}</tbody>
</table>"""

# Bar chart: MS/MS vs MBR per sample
fig1_bar = go.Figure()
fig1_bar.add_bar(
    name="MS/MS",
    x=summary_df["Sample"],
    y=summary_df["MS_MS"],
    marker_color="#4C72B0",
)
fig1_bar.add_bar(
    name="MBR",
    x=summary_df["Sample"],
    y=summary_df["MBR"],
    marker_color="#DD8452",
)
fig1_bar.update_layout(
    barmode="stack",
    title="Detected Peptides per Sample (MS/MS vs MBR)",
    xaxis_title="Sample",
    yaxis_title="Peptide Count",
    legend_title="Detection Type",
    template="plotly_white",
    height=420,
)
fig1_bar_html = fig1_bar.to_html(full_html=False, include_plotlyjs=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Peptide Length Distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 2 — Peptide Length Distribution…")

len_counts = df["Peptide Length"].value_counts().sort_index()
lengths = len_counts.index.tolist()
counts  = len_counts.values.tolist()

def bar_color(length):
    if 8 <= length <= 11:
        return "#2ca02c"   # MHC-I green
    else:
        return "#aec7e8"   # other grey-blue

bar_colors = [bar_color(l) for l in lengths]

fig2_all = go.Figure()
fig2_all.add_bar(
    x=lengths,
    y=counts,
    marker_color=bar_colors,
    showlegend=False,
)
# invisible traces for legend
for label, color in [("MHC-I (8–11 aa)", "#2ca02c"),
                     ("Other", "#aec7e8")]:
    fig2_all.add_bar(x=[None], y=[None], name=label,
                     marker_color=color, showlegend=True)
fig2_all.add_vrect(x0=7.5, x1=11.5, fillcolor="#2ca02c",
                   opacity=0.07, line_width=0, layer="below")
fig2_all.update_layout(
    title="Peptide Length Distribution (All Peptides)",
    xaxis_title="Peptide Length (aa)",
    yaxis_title="Number of Peptides",
    xaxis=dict(tickmode="linear", tick0=min(lengths), dtick=1),
    template="plotly_white",
    legend_title="MHC Class",
    height=420,
)
fig2_all_html = fig2_all.to_html(full_html=False, include_plotlyjs=False)

# Per-sample length stacked bar
focus_lengths = list(range(8, 25))
per_sample_len = []
for s, mc in zip(samples, match_cols):
    detected = df[df[mc] != "unmatched"]
    for l in focus_lengths:
        n = (detected["Peptide Length"] == l).sum()
        per_sample_len.append(dict(Sample=s, Length=str(l), Count=n))

psl_df = pd.DataFrame(per_sample_len)

fig2_sample = go.Figure()
length_colors = px.colors.sequential.Viridis_r
for i, l in enumerate(focus_lengths):
    sub = psl_df[psl_df["Length"] == str(l)]
    color = length_colors[i % len(length_colors)]
    fig2_sample.add_bar(
        name=f"{l} aa",
        x=sub["Sample"],
        y=sub["Count"],
        marker_color=color,
    )
fig2_sample.update_layout(
    barmode="stack",
    title="Peptide Length Distribution per Sample (Detected Peptides, 8–15 aa)",
    xaxis_title="Sample",
    yaxis_title="Peptide Count",
    legend_title="Length",
    template="plotly_white",
    height=450,
)
fig2_sample_html = fig2_sample.to_html(full_html=False, include_plotlyjs=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Spectral Count Distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 3 — Spectral Count Distribution…")

# Violin: non-zero spectral counts per sample (detected peptides only)
fig3_violin = go.Figure()
for i, (s, mc, sc) in enumerate(zip(samples, match_cols, spectral_cols)):
    detected_mask = df[mc] != "unmatched"
    vals = df.loc[detected_mask, sc]
    vals = vals[vals > 0]
    fig3_violin.add_trace(go.Violin(
        y=vals,
        name=s,
        box_visible=True,
        meanline_visible=True,
        points=False,
        marker_color=SAMPLE_COLORS[i % len(SAMPLE_COLORS)],
    ))
fig3_violin.update_layout(
    title="Spectral Count Distribution per Sample (Detected, Non-Zero)",
    yaxis_title="Spectral Count",
    yaxis_type="log",
    xaxis_title="Sample",
    showlegend=False,
    template="plotly_white",
    height=460,
)
fig3_violin_html = fig3_violin.to_html(full_html=False, include_plotlyjs=False)

# MBR rate bar (sorted descending)
mbr_df = summary_df[["Sample", "MBR_Rate"]].copy()
mbr_df["MBR_Rate_Num"] = mbr_df["MBR_Rate"].str.rstrip("%").replace("—", "0").astype(float)
mbr_df = mbr_df.sort_values("MBR_Rate_Num", ascending=True)

fig3_mbr = go.Figure(go.Bar(
    x=mbr_df["MBR_Rate_Num"],
    y=mbr_df["Sample"],
    orientation="h",
    marker=dict(
        color=mbr_df["MBR_Rate_Num"],
        colorscale="RdYlGn_r",
        cmin=0, cmax=50,
        colorbar=dict(title="MBR %"),
    ),
    text=mbr_df["MBR_Rate"],
    textposition="outside",
))
fig3_mbr.add_vline(x=30, line_dash="dash", line_color="red",
                   annotation_text="30% threshold", annotation_position="top right")
fig3_mbr.update_layout(
    title="MBR Rate per Sample (high MBR may indicate quality issues)",
    xaxis_title="MBR Rate (%)",
    yaxis_title="Sample",
    template="plotly_white",
    height=460,
    margin=dict(r=80),
)
fig3_mbr_html = fig3_mbr.to_html(full_html=False, include_plotlyjs=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Intensity Distribution (msstats.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 4 — Intensity Distribution…")

ms = pd.read_csv(MSSTATS)
ms["log2_intensity"] = np.where(
    ms["Intensity"].notna() & (ms["Intensity"] > 0),
    np.log2(ms["Intensity"]),
    np.nan,
)

# missing value rate per run
run_stats = (
    ms.groupby("Run")["Intensity"]
    .agg(total="count", missing=lambda x: x.isna().sum())
    .reset_index()
)
run_stats["missing_pct"] = run_stats["missing"] / run_stats["total"] * 100
run_stats = run_stats.sort_values("missing_pct", ascending=True)

fig4_missing = go.Figure(go.Bar(
    x=run_stats["missing_pct"],
    y=run_stats["Run"],
    orientation="h",
    marker=dict(
        color=run_stats["missing_pct"],
        colorscale="RdYlGn_r",
        cmin=0, cmax=100,
        colorbar=dict(title="Missing %"),
    ),
    text=run_stats["missing_pct"].map("{:.1f}%".format),
    textposition="outside",
))
fig4_missing.update_layout(
    title="Missing Value Rate per Run (MSstats)",
    xaxis_title="Missing Intensity (%)",
    yaxis_title="Run",
    template="plotly_white",
    height=500,
    margin=dict(r=80),
)
fig4_missing_html = fig4_missing.to_html(full_html=False, include_plotlyjs=False)

# log2 intensity violin per condition
conditions = ms["Condition"].unique().tolist()
fig4_violin = go.Figure()
for i, cond in enumerate(sorted(conditions)):
    vals = ms.loc[ms["Condition"] == cond, "log2_intensity"].dropna()
    fig4_violin.add_trace(go.Violin(
        y=vals,
        name=cond,
        box_visible=True,
        meanline_visible=True,
        points=False,
        marker_color=SAMPLE_COLORS[i % len(SAMPLE_COLORS)],
    ))
fig4_violin.update_layout(
    title="Log₂ Intensity Distribution per Condition (MSstats)",
    yaxis_title="Log₂ Intensity",
    xaxis_title="Condition",
    showlegend=False,
    template="plotly_white",
    height=460,
)
fig4_violin_html = fig4_violin.to_html(full_html=False, include_plotlyjs=False)

# Ahmed replicate correlations (Ahmed_1, Ahmed_2, Ahmed_3 intensity columns)
ahmed_cols = ["Ahmed_1 Intensity", "Ahmed_2 Intensity", "Ahmed_3 Intensity"]
ahmed_labels = ["Ahmed_1", "Ahmed_2", "Ahmed_3"]
ahmed_pairs = [(0, 1), (0, 2), (1, 2)]

fig4_corr = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        f"{ahmed_labels[a]} vs {ahmed_labels[b]}" for a, b in ahmed_pairs
    ],
)
for col_idx, (a, b) in enumerate(ahmed_pairs, start=1):
    ca, cb = ahmed_cols[a], ahmed_cols[b]
    sub = df[[ca, cb]].copy()
    sub = sub[(sub[ca] > 0) & (sub[cb] > 0)]
    xa = np.log2(sub[ca])
    ya = np.log2(sub[cb])
    r, _ = pearsonr(xa, ya)
    fig4_corr.add_trace(
        go.Scattergl(
            x=xa, y=ya,
            mode="markers",
            marker=dict(size=3, opacity=0.4, color=SAMPLE_COLORS[col_idx - 1]),
            name=f"r={r:.3f}",
            showlegend=True,
        ),
        row=1, col=col_idx,
    )
    # diagonal
    mn = min(xa.min(), ya.min())
    mx = max(xa.max(), ya.max())
    fig4_corr.add_trace(
        go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                   line=dict(color="red", dash="dash", width=1),
                   showlegend=False),
        row=1, col=col_idx,
    )
    fig4_corr.add_annotation(
        text=f"r = {r:.3f}", xref=f"x{col_idx}", yref=f"y{col_idx}",
        x=0.05, y=0.95, xanchor="left", yanchor="top",
        showarrow=False, font=dict(size=12, color="black"),
    )

fig4_corr.update_layout(
    title="Ahmed Replicate Correlations (Log₂ Intensity, peptides detected in both)",
    template="plotly_white",
    height=420,
    showlegend=False,
)
fig4_corr.update_xaxes(title_text="Log₂ Intensity")
fig4_corr.update_yaxes(title_text="Log₂ Intensity")
fig4_corr_html = fig4_corr.to_html(full_html=False, include_plotlyjs=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Contaminants
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 5 — Contaminants…")

contam_df = df[df["_is_contam"]].copy()

# per-sample contaminant rate
contam_rows = []
for s, mc in zip(samples, match_cols):
    col = df[mc]
    total_detected = (col != "unmatched").sum()
    n_contam = ((col != "unmatched") & df["_is_contam"]).sum()
    pct = n_contam / total_detected * 100 if total_detected > 0 else 0
    contam_rows.append(dict(Sample=s, Contaminant_Peptides=n_contam,
                            Total_Detected=total_detected, Contam_Pct=pct))

contam_sample_df = pd.DataFrame(contam_rows).sort_values("Contam_Pct", ascending=True)

fig5_bar = go.Figure(go.Bar(
    x=contam_sample_df["Contam_Pct"],
    y=contam_sample_df["Sample"],
    orientation="h",
    marker=dict(
        color=contam_sample_df["Contam_Pct"],
        colorscale="OrRd",
        cmin=0, cmax=contam_sample_df["Contam_Pct"].max() * 1.2,
        colorbar=dict(title="Contam %"),
    ),
    text=contam_sample_df["Contam_Pct"].map("{:.2f}%".format),
    textposition="outside",
))
fig5_bar.update_layout(
    title="Contaminant Rate per Sample (% of Detected Peptides)",
    xaxis_title="Contaminant Peptides (%)",
    yaxis_title="Sample",
    template="plotly_white",
    height=460,
    margin=dict(r=80),
)
fig5_bar_html = fig5_bar.to_html(full_html=False, include_plotlyjs=False)

# Top contaminant proteins table
contam_protein_counts = (
    contam_df.groupby(["Protein", "Entry Name", "Protein Description"])
    .size()
    .reset_index(name="Peptide_Count")
    .sort_values("Peptide_Count", ascending=False)
    .head(20)
)

def contam_table_html(df_c):
    header = "<tr><th>Protein</th><th>Entry Name</th><th>Description</th><th>Peptide Count</th></tr>"
    body = ""
    for _, r in df_c.iterrows():
        body += (
            f"<tr><td>{r['Protein']}</td><td>{r['Entry Name']}</td>"
            f"<td>{r['Protein Description']}</td><td>{r['Peptide_Count']}</td></tr>"
        )
    return f"""
<table class="stats-table">
  <thead>{header}</thead>
  <tbody>{body}</tbody>
</table>"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Peptide Overlap Between Samples
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 6 — Peptide Overlap…")

# Detected peptide sets per sample (MS/MS or MBR)
detected_sets = {
    s: set(df.loc[df[mc] != "unmatched", "Peptide Sequence"])
    for s, mc in zip(samples, match_cols)
}

# Jaccard similarity matrix
n = len(samples)
jaccard = np.zeros((n, n))
shared  = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(n):
        a, b = detected_sets[samples[i]], detected_sets[samples[j]]
        union = len(a | b)
        inter = len(a & b)
        jaccard[i, j] = inter / union if union > 0 else 0
        shared[i, j]  = inter

# Heatmap — Jaccard
fig6_jaccard = go.Figure(go.Heatmap(
    z=jaccard,
    x=samples, y=samples,
    colorscale="Blues",
    zmin=0, zmax=1,
    text=np.round(jaccard, 2),
    texttemplate="%{text}",
    textfont=dict(size=9),
    colorbar=dict(title="Jaccard"),
))
fig6_jaccard.update_layout(
    title="Pairwise Jaccard Similarity (Detected Peptides)",
    xaxis=dict(tickangle=45),
    template="plotly_white",
    height=560,
)
fig6_jaccard_html = fig6_jaccard.to_html(full_html=False, include_plotlyjs=False)

# Heatmap — shared peptide counts (upper triangle annotation)
fig6_shared = go.Figure(go.Heatmap(
    z=shared,
    x=samples, y=samples,
    colorscale="Purples",
    text=shared,
    texttemplate="%{text}",
    textfont=dict(size=8),
    colorbar=dict(title="Shared<br>Peptides"),
))
fig6_shared.update_layout(
    title="Shared Peptide Counts Between Samples",
    xaxis=dict(tickangle=45),
    template="plotly_white",
    height=560,
)
fig6_shared_html = fig6_shared.to_html(full_html=False, include_plotlyjs=False)

# Peptide prevalence — how many samples each peptide appears in
pep_sample_counts = pd.Series(
    {pep: sum(1 for s in detected_sets.values() if pep in s)
     for pep in df["Peptide Sequence"]}
)
prev_counts = pep_sample_counts.value_counts().sort_index()

fig6_prev = go.Figure(go.Bar(
    x=prev_counts.index,
    y=prev_counts.values,
    marker_color=px.colors.sequential.Viridis[::max(1, len(px.colors.sequential.Viridis) // len(prev_counts))],
    text=prev_counts.values,
    textposition="outside",
))
fig6_prev.update_layout(
    title="Peptide Prevalence Across Samples",
    xaxis_title="Number of Samples Peptide Is Detected In",
    yaxis_title="Number of Peptides",
    xaxis=dict(tickmode="linear", dtick=1),
    template="plotly_white",
    height=400,
)
fig6_prev_html = fig6_prev.to_html(full_html=False, include_plotlyjs=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Protein Source
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 7 — Protein Source…")

def protein_source(protein: str) -> str:
    if re.search(r"Cont_", str(protein)):
        return "Contaminant"
    if str(protein).startswith("sp|"):
        return "SwissProt (sp)"
    if str(protein).startswith("tr|"):
        return "TrEMBL (tr)"
    return "Other"

df["_source"] = df["Protein"].apply(protein_source)

# Dataset-wide pie
source_counts = df["_source"].value_counts()
fig7_pie = go.Figure(go.Pie(
    labels=source_counts.index,
    values=source_counts.values,
    hole=0.4,
    marker=dict(colors=["#4C72B0", "#DD8452", "#c44e52", "#8c8c8c"]),
    textinfo="label+percent+value",
))
fig7_pie.update_layout(
    title="Protein Source Distribution (All Peptides)",
    template="plotly_white",
    height=400,
)
fig7_pie_html = fig7_pie.to_html(full_html=False, include_plotlyjs=False)

# Unique genes per sample (canonical, non-contaminant)
gene_rows = []
for s, mc in zip(samples, match_cols):
    detected = df[(df[mc] != "unmatched") & (~df["_is_contam"])]
    genes = detected["Gene"].dropna()
    genes = genes[genes.str.strip() != ""]
    gene_rows.append(dict(Sample=s, Unique_Genes=genes.nunique()))

gene_df = pd.DataFrame(gene_rows).sort_values("Unique_Genes", ascending=True)

fig7_genes = go.Figure(go.Bar(
    x=gene_df["Unique_Genes"],
    y=gene_df["Sample"],
    orientation="h",
    marker_color="#4C72B0",
    text=gene_df["Unique_Genes"],
    textposition="outside",
))
fig7_genes.update_layout(
    title="Unique Canonical Genes per Sample",
    xaxis_title="Number of Unique Genes",
    yaxis_title="Sample",
    template="plotly_white",
    height=460,
    margin=dict(r=60),
)
fig7_genes_html = fig7_genes.to_html(full_html=False, include_plotlyjs=False)

# Source breakdown per sample (stacked bar)
source_order = ["SwissProt (sp)", "TrEMBL (tr)", "Contaminant", "Other"]
source_colors_map = {
    "SwissProt (sp)": "#4C72B0",
    "TrEMBL (tr)":    "#DD8452",
    "Contaminant":    "#c44e52",
    "Other":          "#8c8c8c",
}
fig7_src_sample = go.Figure()
for src in source_order:
    ys = []
    for s, mc in zip(samples, match_cols):
        detected = df[df[mc] != "unmatched"]
        ys.append((detected["_source"] == src).sum())
    fig7_src_sample.add_bar(
        name=src, x=samples, y=ys,
        marker_color=source_colors_map[src],
    )
fig7_src_sample.update_layout(
    barmode="stack",
    title="Protein Source per Sample (Detected Peptides)",
    xaxis_title="Sample",
    yaxis_title="Peptide Count",
    legend_title="Source",
    template="plotly_white",
    height=440,
)
fig7_src_sample_html = fig7_src_sample.to_html(full_html=False, include_plotlyjs=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Amino Acid Composition
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 8 — Amino Acid Composition…")

AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Frequency of each AA at each position for 9-mers (MHC-I focus)
mers9 = df.loc[df["Peptide Length"] == 9, "Peptide Sequence"].dropna().tolist()

pos_aa = np.zeros((9, 20))  # positions x amino acids
for pep in mers9:
    for pos, aa in enumerate(pep):
        if aa in AAS:
            pos_aa[pos, AAS.index(aa)] += 1

# Normalise to frequency
pos_freq = pos_aa / pos_aa.sum(axis=1, keepdims=True)

fig8_heatmap = go.Figure(go.Heatmap(
    z=pos_freq.T,           # AA × position
    x=[f"P{i+1}" for i in range(9)],
    y=AAS,
    colorscale="RdBu_r",
    zmid=1 / 20,            # expected uniform frequency
    colorbar=dict(title="Frequency"),
    text=np.round(pos_freq.T, 3),
    texttemplate="%{text}",
    textfont=dict(size=8),
))
fig8_heatmap.update_layout(
    title=f"Amino Acid Position Frequency — 9-mers (n={len(mers9):,})",
    xaxis_title="Position",
    yaxis_title="Amino Acid",
    template="plotly_white",
    height=600,
)
fig8_heatmap_html = fig8_heatmap.to_html(full_html=False, include_plotlyjs=False)

# Overall AA frequency vs background (all detected canonical peptides)
canonical_detected = df[
    (~df["_is_contam"]) &
    df[match_cols].apply(lambda row: any(v != "unmatched" for v in row), axis=1)
]["Peptide Sequence"].dropna()

all_aa_counts = pd.Series(list("".join(canonical_detected))).value_counts()
all_aa_counts = all_aa_counts.reindex(AAS, fill_value=0)
all_aa_freq   = all_aa_counts / all_aa_counts.sum()

# UniProt human proteome background (approximate)
HUMAN_BG = {
    "A": 0.0707, "C": 0.0227, "D": 0.0526, "E": 0.0628, "F": 0.0391,
    "G": 0.0695, "H": 0.0228, "I": 0.0591, "K": 0.0577, "L": 0.0988,
    "M": 0.0228, "N": 0.0405, "P": 0.0472, "Q": 0.0397, "R": 0.0553,
    "S": 0.0694, "T": 0.0550, "V": 0.0687, "W": 0.0120, "Y": 0.0293,
}
bg_freq = pd.Series(HUMAN_BG).reindex(AAS, fill_value=0)

fig8_freq = go.Figure()
fig8_freq.add_bar(
    name="Immunopeptidome",
    x=AAS,
    y=all_aa_freq.values,
    marker_color="#4C72B0",
)
fig8_freq.add_bar(
    name="Human proteome (background)",
    x=AAS,
    y=bg_freq.values,
    marker_color="#DD8452",
    opacity=0.7,
)
fig8_freq.update_layout(
    barmode="group",
    title="Amino Acid Frequency: Immunopeptidome vs Human Proteome Background",
    xaxis_title="Amino Acid",
    yaxis_title="Relative Frequency",
    legend_title="Dataset",
    template="plotly_white",
    height=400,
)
fig8_freq_html = fig8_freq.to_html(full_html=False, include_plotlyjs=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Sequence Logo / Binding Motif
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 9 — Sequence Logo…")

# pos_freq (9×20) already built in Section 8
logo_df = pd.DataFrame(pos_freq, columns=AAS, index=list(range(9)))
info_df = logomaker.transform_matrix(logo_df,
                                     from_type="probability",
                                     to_type="information")

fig_logo, axes = plt.subplots(1, 2, figsize=(16, 3.5))
fig_logo.patch.set_facecolor("white")
for ax, matrix, title, ylabel in [
    (axes[0], logo_df, f"Frequency Logo — 9-mers (n={len(mers9):,})", "Frequency"),
    (axes[1], info_df, f"Information Content Logo — 9-mers (n={len(mers9):,})", "Bits"),
]:
    logomaker.Logo(matrix, ax=ax, color_scheme="chemistry")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Position", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    for pos in [1, 8]:          # P2, P9 anchor residues
        ax.axvspan(pos - 0.5, pos + 0.5, color="gold", alpha=0.18, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout(pad=2.0)
_buf = io.BytesIO()
fig_logo.savefig(_buf, format="png", bbox_inches="tight", dpi=150)
_buf.seek(0)
logo_b64 = base64.b64encode(_buf.read()).decode()
plt.close(fig_logo)
sec9_logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" '
    f'style="max-width:100%;height:auto;" alt="Sequence Logo">'
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Charge State Distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 10 — Charge State Distribution…")

# Charges column may contain comma-separated values e.g. "2,3"
def parse_charges(val):
    return [int(c.strip()) for c in str(val).split(",") if c.strip().isdigit()]

# Dataset-wide charge distribution
all_charges = []
for raw in df["Charges"]:
    all_charges.extend(parse_charges(raw))
charge_overall = pd.Series(all_charges).value_counts().sort_index()

fig10_pie = go.Figure(go.Pie(
    labels=[f"{c}+" for c in charge_overall.index],
    values=charge_overall.values,
    hole=0.4,
    marker=dict(colors=px.colors.qualitative.Plotly),
    textinfo="label+percent",
))
fig10_pie.update_layout(
    title="Overall Charge State Distribution (All Peptides)",
    template="plotly_white",
    height=380,
)
fig10_pie_html = fig10_pie.to_html(full_html=False, include_plotlyjs=False)

# Per-sample charge distribution (detected only)
charge_vals = sorted(charge_overall.index.tolist())
fig10_sample = go.Figure()
for c in charge_vals:
    ys = []
    for s, mc in zip(samples, match_cols):
        detected = df[df[mc] != "unmatched"]
        count = detected["Charges"].apply(
            lambda v: c in parse_charges(v)
        ).sum()
        ys.append(count)
    fig10_sample.add_bar(name=f"{c}+", x=samples, y=ys)

fig10_sample.update_layout(
    barmode="stack",
    title="Charge State Distribution per Sample (Detected Peptides)",
    xaxis_title="Sample",
    yaxis_title="Peptide Count",
    legend_title="Charge",
    template="plotly_white",
    height=430,
)
fig10_sample_html = fig10_sample.to_html(full_html=False, include_plotlyjs=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Sample Clustering / PCA
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 11 — PCA & Sample Clustering…")

# Build log2 intensity matrix; 0 → NaN
int_mat = df[intensity_cols].copy().replace(0, np.nan)
int_mat.columns = samples
log2_mat = np.log2(int_mat)

# Keep peptides detected in ≥3 samples
n_detected = (int_mat > 0).sum(axis=1)
log2_filt = log2_mat[n_detected >= 3]

# Min-value imputation per sample (below-detection assumption)
X = log2_filt.values.T          # shape: n_samples × n_peptides
col_mins = np.nanmin(X, axis=0)
nan_mask = np.isnan(X)
X_imp = X.copy()
X_imp[nan_mask] = np.take(col_mins, np.where(nan_mask)[1])

# PCA
n_comp = min(5, X_imp.shape[0] - 1)
X_scaled = StandardScaler().fit_transform(X_imp)
pca = PCA(n_components=n_comp)
coords = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_ * 100

fig11_pca = go.Figure()
for i, s in enumerate(samples):
    fig11_pca.add_trace(go.Scatter(
        x=[coords[i, 0]], y=[coords[i, 1]],
        mode="markers+text",
        name=s,
        text=[s],
        textposition="top center",
        marker=dict(size=12, color=SAMPLE_COLORS[i % len(SAMPLE_COLORS)]),
        showlegend=False,
    ))
fig11_pca.update_layout(
    title=f"PCA — Samples (peptides detected in ≥3 samples, min-imputed)",
    xaxis_title=f"PC1 ({var_exp[0]:.1f}% variance)",
    yaxis_title=f"PC2 ({var_exp[1]:.1f}% variance)",
    template="plotly_white",
    height=500,
)
fig11_pca_html = fig11_pca.to_html(full_html=False, include_plotlyjs=False)

# Explained variance bar
fig11_var = go.Figure(go.Bar(
    x=[f"PC{i+1}" for i in range(n_comp)],
    y=var_exp,
    marker_color="#4C72B0",
    text=[f"{v:.1f}%" for v in var_exp],
    textposition="outside",
))
fig11_var.update_layout(
    title="PCA Explained Variance",
    xaxis_title="Principal Component",
    yaxis_title="Variance Explained (%)",
    template="plotly_white",
    height=420,
)
fig11_var_html = fig11_var.to_html(full_html=False, include_plotlyjs=False)

# Sample correlation heatmap (hierarchical clustering order)
corr_mat = np.corrcoef(X_imp)
order = leaves_list(linkage(X_imp, method="ward", metric="euclidean"))
corr_ordered = corr_mat[np.ix_(order, order)]
samples_ordered = [samples[i] for i in order]

fig11_corr = go.Figure(go.Heatmap(
    z=np.round(corr_ordered, 3),
    x=samples_ordered,
    y=samples_ordered,
    colorscale="RdBu",
    zmin=-1, zmax=1,
    text=np.round(corr_ordered, 2),
    texttemplate="%{text}",
    textfont=dict(size=8),
    colorbar=dict(title="Pearson r"),
))
fig11_corr.update_layout(
    title="Sample Correlation Heatmap (hierarchical clustering order, log₂ intensity)",
    xaxis=dict(tickangle=45),
    template="plotly_white",
    height=560,
)
fig11_corr_html = fig11_corr.to_html(full_html=False, include_plotlyjs=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — IEDB Cross-reference (local SQL)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 12 — IEDB Cross-reference (local SQL)…")

import pickle as _pickle
import re as _re

IEDB_SQL   = HERE / "iedb/iedb_public.sql"
IEDB_CACHE = HERE / ".iedb_seqs.pkl"

def _load_iedb_sequences() -> set:
    if IEDB_CACHE.exists():
        print("  Loading IEDB sequences from cache…")
        with open(IEDB_CACHE, "rb") as f:
            return _pickle.load(f)
    print("  Parsing IEDB SQL dump — runs once, then cached…")
    # Extract linear_peptide_seq (3rd column) from INSERT INTO `epitope` rows.
    # Format: (id, 'description', 'SEQ', ...) — SEQ is all uppercase amino-acid letters.
    pat = _re.compile(rb"\(\d+,'[^']*','([A-Z]+)',")
    seqs: set = set()
    with open(IEDB_SQL, "rb") as f:
        for line in f:
            if b"INSERT INTO `epitope`" not in line:
                continue
            for m in pat.finditer(line):
                seqs.add(m.group(1).decode("ascii"))
    with open(IEDB_CACHE, "wb") as f:
        _pickle.dump(seqs, f)
    print(f"  Cached {len(seqs):,} IEDB epitope sequences")
    return seqs

iedb_seqs = _load_iedb_sequences()

# ── Detected canonical MHC-I peptides ────────────────────────────────────────
mhci_pool = df[
    (~df["_is_contam"]) & df["Peptide Length"].between(8, 11)
].copy()
mhci_pool["_total_sc"] = mhci_pool[spectral_cols].sum(axis=1)

detected_mask_any = mhci_pool[match_cols].apply(
    lambda r: any(v != "unmatched" for v in r), axis=1
)
mhci_detected = set(mhci_pool.loc[detected_mask_any, "Peptide Sequence"].dropna())
mhci_iedb     = {s for s in iedb_seqs if 8 <= len(s) <= 11}

n_detected  = len(mhci_detected)
n_known     = len(mhci_detected & mhci_iedb)
n_unique    = len(mhci_detected - mhci_iedb)
known_pct   = n_known / n_detected * 100 if n_detected else 0

iedb_summary = [
    ("Canonical MHC-I peptides detected (8–11 aa)", f"{n_detected:,}"),
    ("Known IEDB epitopes",                          f"{n_known:,} ({known_pct:.1f}%)"),
    ("Novel candidates (not in IEDB)",               f"{n_unique:,}"),
    ("Total IEDB MHC-I sequences (8–11 aa)",         f"{len(mhci_iedb):,}"),
]

fig12_pie = go.Figure(go.Pie(
    labels=["Known (IEDB)", "Novel candidates"],
    values=[max(n_known, 0), max(n_unique, 0)],
    hole=0.4,
    marker=dict(colors=["#2ca02c", "#4C72B0"]),
    textinfo="label+percent+value",
))
fig12_pie.update_layout(
    title=f"IEDB Cross-reference — {n_detected:,} canonical MHC-I peptides",
    template="plotly_white",
    height=380,
)
fig12_pie_html = fig12_pie.to_html(full_html=False, include_plotlyjs=False)

# Overall Venn
sec12_venn_html = venn2_img(
    mhci_detected, mhci_iedb,
    label_a="Detected\n(MHC-I)",
    label_b="IEDB",
    title="Detected MHC-I Peptides vs IEDB (8–11 aa)",
)

# Known peptides table (top 30 by total spectral count)
known_peps_set = mhci_detected & mhci_iedb
known_df = mhci_pool[mhci_pool["Peptide Sequence"].isin(known_peps_set)].copy()
known_df = (
    known_df[["Peptide Sequence", "Gene", "Protein Description", "_total_sc", "Peptide Length"]]
    .sort_values("_total_sc", ascending=False)
    .head(30)
    .rename(columns={"_total_sc": "Total Spectral Count", "Peptide Length": "Length"})
)

def iedb_table_html(df_k):
    if df_k.empty:
        return '<p class="note">No known IEDB epitopes found.</p>'
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df_k.columns) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in df_k.itertuples(index=False)
    )
    return (f'<table class="stats-table">'
            f'<thead>{header}</thead><tbody>{body}</tbody></table>')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — GO / Pathway Enrichment
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Section 13 — GO/Pathway Enrichment…")

GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

# Gene list: canonical detected genes (any sample)
enrich_genes = (
    df.loc[
        (~df["_is_contam"]) &
        df[match_cols].apply(lambda r: any(v != "unmatched" for v in r), axis=1),
        "Gene"
    ]
    .dropna()
    .str.strip()
    .pipe(lambda s: s[s != ""])
    .unique()
    .tolist()
)

go_results_by_source = {}
try:
    resp = requests.post(
        GPROFILER_URL,
        json={
            "organism": "hsapiens",
            "query": enrich_genes,
            "sources": ["GO:BP", "KEGG", "REAC"],
            "user_threshold": 0.05,
            "all_results": False,
            "no_evidences": True,
            "ordered": False,
        },
        timeout=60,
    )
    if resp.status_code == 200:
        gp_data = resp.json()
        results = gp_data.get("result", [])   # flat list of term objects
        for row in results:
            src = row["source"]
            go_results_by_source.setdefault(src, []).append({
                "Term":        row["name"],
                "ID":          row["native"],
                "p_adj":       row["p_value"],
                "neg_log10_p": -np.log10(row["p_value"] + 1e-300),
                "Term Size":   row["term_size"],
                "Overlap":     row["intersection_size"],
            })
        print(f"  g:Profiler returned {len(results)} enriched terms")
    else:
        print(f"  g:Profiler returned status {resp.status_code}")
except Exception as exc:
    print(f"  g:Profiler error: {exc}")

def enrich_bar(source_key, title, color, top_n=20):
    rows = go_results_by_source.get(source_key, [])
    if not rows:
        return go.Figure().update_layout(
            title=f"{title} — no results", template="plotly_white", height=80)
    top = sorted(rows, key=lambda x: x["neg_log10_p"], reverse=True)[:top_n]
    top = top[::-1]   # ascending for horizontal bar
    fig = go.Figure(go.Bar(
        y=[r["Term"] for r in top],
        x=[r["neg_log10_p"] for r in top],
        orientation="h",
        marker_color=color,
        text=[f'p={r["p_adj"]:.1e}  overlap={r["Overlap"]}' for r in top],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="-log₁₀(p adjusted)",
        template="plotly_white",
        height=max(350, top_n * 22),
        margin=dict(l=320, r=160),
    )
    return fig

fig13_gobp  = enrich_bar("GO:BP", "GO Biological Process (Top 20)",  "#4C72B0")
fig13_kegg  = enrich_bar("KEGG",  "KEGG Pathways (Top 20)",           "#DD8452")
fig13_reac  = enrich_bar("REAC",  "Reactome Pathways (Top 20)",       "#2ca02c")

fig13_gobp_html = fig13_gobp.to_html(full_html=False, include_plotlyjs=False)
fig13_kegg_html = fig13_kegg.to_html(full_html=False, include_plotlyjs=False)
fig13_reac_html = fig13_reac.to_html(full_html=False, include_plotlyjs=False)

# ═══════════════════════════════════════════════════════════════════════════════
# PER-SAMPLE FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print("Building per-sample figures…")

# Pre-compute peptide sets for all samples (used in overlap chart)
sample_pep_sets = {}
for _s, _mc in zip(samples, match_cols):
    _mask = df[_mc] != "unmatched"
    sample_pep_sets[_s] = set(df.loc[_mask, "Peptide Sequence"].dropna())

sample_figs = {}
for i, (s, mc, sc) in enumerate(zip(samples, match_cols, spectral_cols)):
    s_color = SAMPLE_COLORS[i % len(SAMPLE_COLORS)]
    detected_mask = df[mc] != "unmatched"
    detected = df[detected_mask]

    # Length distribution (MHC-I only — no MHC-II annotation)
    len_counts_s = detected["Peptide Length"].value_counts().sort_index()
    fig_s_len = go.Figure()
    fig_s_len.add_bar(
        x=len_counts_s.index.tolist(),
        y=len_counts_s.values.tolist(),
        marker_color=[bar_color(l) for l in len_counts_s.index],
        showlegend=False,
    )
    for label, lc in [("MHC-I (8\u201311 aa)", "#2ca02c"), ("Other", "#aec7e8")]:
        fig_s_len.add_bar(x=[None], y=[None], name=label,
                          marker_color=lc, showlegend=True)
    fig_s_len.add_vrect(x0=7.5, x1=11.5, fillcolor="#2ca02c",
                        opacity=0.07, line_width=0, layer="below")
    fig_s_len.update_layout(
        title=f"Peptide Length Distribution \u2014 {s}",
        xaxis_title="Peptide Length (aa)", yaxis_title="Count",
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        legend_title="MHC Class", template="plotly_white", height=420,
    )

    # Protein source pie
    src_counts_s = detected["_source"].value_counts()
    fig_s_src = go.Figure(go.Pie(
        labels=src_counts_s.index.tolist(),
        values=src_counts_s.values.tolist(),
        hole=0.4,
        marker=dict(colors=["#4C72B0", "#DD8452", "#c44e52", "#8c8c8c"]),
        textinfo="label+percent+value",
    ))
    fig_s_src.update_layout(
        title=f"Protein Source \u2014 {s}", template="plotly_white", height=320,
    )

    # Charge state pie
    sample_charges_list = []
    for raw in detected["Charges"]:
        sample_charges_list.extend(parse_charges(raw))
    if sample_charges_list:
        charge_s = pd.Series(sample_charges_list).value_counts().sort_index()
        fig_s_chg = go.Figure(go.Pie(
            labels=[f"{c}+" for c in charge_s.index],
            values=charge_s.values.tolist(),
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Plotly),
            textinfo="label+percent",
        ))
    else:
        fig_s_chg = go.Figure()
    fig_s_chg.update_layout(
        title=f"Charge State Distribution \u2014 {s}",
        template="plotly_white", height=320,
    )

    # Per-sample IEDB Venn (MHC-I detected peptides vs IEDB)
    s_mhci = set(
        detected.loc[detected["Peptide Length"].between(8, 11) & (~detected["_is_contam"]),
                     "Peptide Sequence"].dropna()
    )
    s_venn_html = venn2_img(
        s_mhci, mhci_iedb,
        label_a=f"{s}\n(MHC-I)",
        label_b="IEDB",
        title=f"MHC-I Peptides vs IEDB \u2014 {s}",
    )

    # ── Sequence logo + AA position frequency heatmap (9-mers, non-contam) ──
    s_mers9 = detected.loc[
        (detected["Peptide Length"] == 9) & (~detected["_is_contam"]),
        "Peptide Sequence"
    ].dropna().tolist()

    if s_mers9:
        s_pos_aa = np.zeros((9, 20))
        for pep in s_mers9:
            for pos, aa in enumerate(pep):
                if aa in AAS:
                    s_pos_aa[pos, AAS.index(aa)] += 1
        s_pos_freq = s_pos_aa / s_pos_aa.sum(axis=1, keepdims=True)

        fig_s_heatmap = go.Figure(go.Heatmap(
            z=s_pos_freq.T,
            x=[f"P{i+1}" for i in range(9)],
            y=AAS,
            colorscale="RdBu_r",
            zmid=1 / 20,
            colorbar=dict(title="Frequency"),
            text=np.round(s_pos_freq.T, 3),
            texttemplate="%{text}",
            textfont=dict(size=8),
        ))
        fig_s_heatmap.update_layout(
            title=f"AA Position Frequency \u2014 9-mers (n={len(s_mers9):,}) \u2014 {s}",
            xaxis_title="Position", yaxis_title="Amino Acid",
            template="plotly_white", height=550,
        )
        s_heatmap_html = fig_s_heatmap.to_html(full_html=False, include_plotlyjs=False,
                                                config={"responsive": True})

        s_logo_df = pd.DataFrame(s_pos_freq, columns=AAS, index=list(range(9)))
        s_info_df = logomaker.transform_matrix(s_logo_df, from_type="probability",
                                               to_type="information")
        fig_s_logo, s_axes = plt.subplots(1, 2, figsize=(16, 3.5))
        fig_s_logo.patch.set_facecolor("white")
        for ax, matrix, t, ylabel in [
            (s_axes[0], s_logo_df, f"Frequency Logo \u2014 9-mers (n={len(s_mers9):,})", "Frequency"),
            (s_axes[1], s_info_df, f"Information Content Logo \u2014 9-mers (n={len(s_mers9):,})", "Bits"),
        ]:
            logomaker.Logo(matrix, ax=ax, color_scheme="chemistry")
            ax.set_title(t, fontsize=11)
            ax.set_xlabel("Position", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            for pos in [1, 8]:
                ax.axvspan(pos - 0.5, pos + 0.5, color="gold", alpha=0.18, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.tight_layout(pad=2.0)
        _sbuf = io.BytesIO()
        fig_s_logo.savefig(_sbuf, format="png", bbox_inches="tight", dpi=150)
        _sbuf.seek(0)
        s_logo_b64 = base64.b64encode(_sbuf.read()).decode()
        plt.close(fig_s_logo)
        s_logo_html = (
            f'<img src="data:image/png;base64,{s_logo_b64}" '
            f'style="max-width:100%;height:auto;" alt="Sequence Logo \u2014 {s}">'
        )
    else:
        s_heatmap_html = "<p class='note'>No 9-mer peptides detected.</p>"
        s_logo_html    = "<p class='note'>No 9-mer peptides detected.</p>"

    # ── Spectral count histogram ──────────────────────────────────────────────
    sc_vals = detected[sc]
    sc_vals = sc_vals[sc_vals > 0]
    fig_s_sc = go.Figure(go.Histogram(
        x=sc_vals, nbinsx=40,
        marker_color=s_color, opacity=0.85,
    ))
    fig_s_sc.update_layout(
        title=f"Spectral Count Distribution \u2014 {s}",
        xaxis_title="Spectral Count", yaxis_title="# Peptides",
        xaxis_type="log", template="plotly_white", height=340,
    )
    s_sc_html = fig_s_sc.to_html(full_html=False, include_plotlyjs=False,
                                  config={"responsive": True})

    # ── Log2 intensity histogram ──────────────────────────────────────────────
    int_col = f"{s} Intensity"
    if int_col in df.columns:
        int_raw = df.loc[detected_mask, int_col]
        int_vals = np.log2(int_raw[int_raw > 0])
        fig_s_int = go.Figure(go.Histogram(
            x=int_vals, nbinsx=40,
            marker_color=s_color, opacity=0.85,
        ))
        fig_s_int.update_layout(
            title=f"Log\u2082 Intensity Distribution \u2014 {s}",
            xaxis_title="Log\u2082 Intensity", yaxis_title="# Peptides",
            template="plotly_white", height=340,
        )
        s_int_html = fig_s_int.to_html(full_html=False, include_plotlyjs=False,
                                        config={"responsive": True})
    else:
        s_int_html = "<p class='note'>Intensity data not available for this sample.</p>"

    # ── Peptide overlap with other samples ────────────────────────────────────
    s_peps = sample_pep_sets[s]
    overlap_rows = sorted(
        [{"sample": s2, "shared": len(s_peps & sample_pep_sets[s2])}
         for s2 in samples if s2 != s],
        key=lambda r: r["shared"]
    )
    overlap_df = pd.DataFrame(overlap_rows)
    fig_s_overlap = go.Figure(go.Bar(
        x=overlap_df["shared"], y=overlap_df["sample"],
        orientation="h",
        marker_color=s_color,
        text=overlap_df["shared"].map("{:,}".format),
        textposition="outside",
    ))
    fig_s_overlap.update_layout(
        title=f"Peptide Overlap with Other Samples \u2014 {s}",
        xaxis_title="Shared Peptides", yaxis_title="Sample",
        template="plotly_white",
        height=max(300, 80 + 40 * (len(samples) - 1)),
        margin=dict(r=80),
    )
    s_overlap_html = fig_s_overlap.to_html(full_html=False, include_plotlyjs=False,
                                            config={"responsive": True})

    # ── Top 15 proteins by peptide count ─────────────────────────────────────
    top_prots = (
        detected.loc[~detected["_is_contam"], "Protein"]
        .value_counts().head(15).sort_values(ascending=True)
    )
    fig_s_prot = go.Figure(go.Bar(
        x=top_prots.values, y=top_prots.index.tolist(),
        orientation="h",
        marker_color=s_color,
        text=top_prots.values, textposition="outside",
    ))
    fig_s_prot.update_layout(
        title=f"Top 15 Proteins by Peptide Count \u2014 {s}",
        xaxis_title="Peptide Count", yaxis_title="Protein",
        template="plotly_white", height=500, margin=dict(r=80),
    )
    s_prot_html = fig_s_prot.to_html(full_html=False, include_plotlyjs=False,
                                      config={"responsive": True})

    row = summary_df[summary_df["Sample"] == s].iloc[0]
    sample_figs[s] = dict(
        stats=row,
        len_html=fig_s_len.to_html(full_html=False, include_plotlyjs=False,
                                   config={"responsive": True}),
        src_html=fig_s_src.to_html(full_html=False, include_plotlyjs=False,
                                   config={"responsive": True}),
        chg_html=fig_s_chg.to_html(full_html=False, include_plotlyjs=False,
                                   config={"responsive": True}),
        venn_html=s_venn_html,
        logo_html=s_logo_html,
        heatmap_html=s_heatmap_html,
        sc_html=s_sc_html,
        int_html=s_int_html,
        overlap_html=s_overlap_html,
        prot_html=s_prot_html,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT RUN METRICS → STATE
# ═══════════════════════════════════════════════════════════════════════════════
print("Saving run metrics to state…")

_run_ts    = datetime.now()
_run_id    = _run_ts.strftime("%Y%m%d_%H%M%S")
_run_label = args.label or _run_ts.strftime("Run %Y-%m-%d %H:%M")

_per_sample: dict = {}
for _s, _mc in zip(samples, match_cols):
    _det   = df[_mc] != "unmatched"
    _total = int(_det.sum())
    _msms  = int((df[_mc] == "MS/MS").sum())
    _mbr   = int((df[_mc] == "MBR").sum())
    _contam = int((_det & df["_is_contam"]).sum())
    _mhci  = int((_det & df["Peptide Length"].between(8, 11) & ~df["_is_contam"]).sum())
    _per_sample[_s] = {
        "total_detected":  _total,
        "msms":            _msms,
        "mbr":             _mbr,
        "mbr_rate_num":    round(_mbr  / _total * 100, 1) if _total > 0 else 0.0,
        "contam_count":    _contam,
        "contam_rate_num": round(_contam / _total * 100, 1) if _total > 0 else 0.0,
        "mhci_count":      _mhci,
        "mhci_pct":        round(_mhci / _total * 100, 1) if _total > 0 else 0.0,
    }

_global: dict = {
    "total_peptides":  int(total_peptides),
    "canonical_pct":   round(unique_canonical / total_peptides * 100, 1) if total_peptides else 0.0,
    "mhci_count":      int(n_detected),
    "novel_count":     int(n_unique),
    "novel_pct":       round(n_unique / n_detected * 100, 1) if n_detected else 0.0,
    "iedb_known_pct":  round(known_pct, 1),
    "n_samples":       len(samples),
}

state.add_run({
    "run_id":       _run_id,
    "label":        _run_label,
    "timestamp":    _run_ts.isoformat(),
    "data_path":    str(args.data),
    "msstats_path": str(args.msstats),
    "samples":      samples,
    "global":       _global,
    "per_sample":   _per_sample,
})
state.save()

# Propagate per-sample metrics into sample_figs so build_sample_page can read them
for _s in samples:
    sample_figs[_s]["mhci_count"] = _per_sample[_s]["mhci_count"]

# ═══════════════════════════════════════════════════════════════════════════════
# Assemble HTML
# ═══════════════════════════════════════════════════════════════════════════════
print("Writing HTML…")


def safe_id(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


# ─── longitudinal / history helpers ──────────────────────────────────────────

def _run_history_table_html(runs: list) -> str:
    if not runs:
        return "<p class='note'>No run history yet.</p>"
    header = (
        "<tr><th>#</th><th>Label</th><th>Date</th><th>Samples</th>"
        "<th>Total Peptides</th><th>MHC-I (8–11 aa)</th>"
        "<th>Novel (not IEDB)</th><th>IEDB Known %</th></tr>"
    )
    rows_html = ""
    for i, r in enumerate(runs, 1):
        g   = r.get("global", {})
        ts  = r.get("timestamp", "")[:10]
        is_current = (i == len(runs))
        bg  = ' style="background:#eef3ff;"' if is_current else ""
        tag = ' <span class="badge badge-purple">current</span>' if is_current else ""
        rows_html += (
            f"<tr{bg}>"
            f"<td>{i}</td>"
            f"<td><strong>{r.get('label','')}</strong>{tag}</td>"
            f"<td>{ts}</td>"
            f"<td>{len(r.get('samples', []))}</td>"
            f"<td>{g.get('total_peptides', '—'):,}</td>"
            f"<td>{g.get('mhci_count', '—'):,}</td>"
            f"<td>{g.get('novel_count', '—'):,} ({g.get('novel_pct', '—')}%)</td>"
            f"<td>{g.get('iedb_known_pct', '—')}%</td>"
            f"</tr>"
        )
    return f'<table class="stats-table"><thead>{header}</thead><tbody>{rows_html}</tbody></table>'


def _build_trend_charts_html(runs: list) -> str:
    """Return HTML with Plotly line charts of per-sample QC metrics over runs."""
    if len(runs) < 2:
        return "<p class='note'>Trend charts appear after two or more runs.</p>"

    all_samples = sorted({s for r in runs for s in r.get("samples", [])})
    x_labels    = [r.get("label", r.get("run_id", "")) for r in runs]

    TREND_METRICS = [
        ("total_detected",  "Total Detected",       True),
        ("mbr_rate_num",    "MBR Rate (%)",          False),
        ("contam_rate_num", "Contaminant Rate (%)",  False),
        ("mhci_count",      "MHC-I Peptides (8–11 aa)", True),
        ("mhci_pct",        "MHC-I %",               True),
    ]

    charts_html = ""
    for metric_key, metric_label, higher_is_better in TREND_METRICS:
        fig = go.Figure()
        for j, samp in enumerate(all_samples):
            y_vals = []
            x_vals = []
            for r in runs:
                ps = r.get("per_sample", {})
                if samp in ps:
                    y_vals.append(ps[samp].get(metric_key))
                    x_vals.append(r.get("label", r.get("run_id", "")))
            if y_vals:
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode="lines+markers",
                    name=samp,
                    marker_color=SAMPLE_COLORS[j % len(SAMPLE_COLORS)],
                    marker_size=8,
                ))
        fig.update_layout(
            title=metric_label,
            xaxis_title="Run",
            yaxis_title=metric_label,
            template="plotly_white",
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        charts_html += f'<div class="plot-grid" style="margin-bottom:28px;">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    # Global trend: total peptides + novel count
    fig_g = make_subplots(rows=1, cols=2,
                          subplot_titles=["Total Unique Peptides", "Novel Candidates (not in IEDB)"])
    for col_i, key in enumerate(["total_peptides", "novel_count"], start=1):
        y_g = [r.get("global", {}).get(key) for r in runs]
        fig_g.add_trace(go.Scatter(
            x=x_labels, y=y_g,
            mode="lines+markers+text",
            text=[str(v) if v is not None else "" for v in y_g],
            textposition="top center",
            marker_size=9,
            line_color="#4C72B0",
            showlegend=False,
        ), row=1, col=col_i)
    fig_g.update_layout(template="plotly_white", height=320)
    charts_html += f'<div class="plot-grid">{fig_g.to_html(full_html=False, include_plotlyjs=False)}</div>'

    return charts_html


def build_history_tab_html(runs: list) -> str:
    table_html  = _run_history_table_html(runs)
    trends_html = _build_trend_charts_html(runs)
    return f"""<div class="tab-content hidden" id="tab-history">
<div class="container">
<div class="section">
  <h2>Run History</h2>
  <p class="desc">Each row represents one execution of the pipeline. The <strong>current</strong> run
  is highlighted. Re-run the script with <code>--label</code> to tag each update.</p>
  {table_html}
</div>
<div class="section">
  <h2>Longitudinal QC Trends</h2>
  <p class="desc">Per-sample metrics across all recorded runs.
  Upward trends in MHC-I count and downward trends in MBR / contaminant rates indicate improving data quality.</p>
  {trends_html}
</div>
</div>
</div><!-- /tab-history -->"""


def build_sample_page(s, figs, prev_metrics: dict | None = None):
    row = figs["stats"]
    p   = prev_metrics or {}
    mbr_num     = float(row["MBR_Rate"].rstrip("%"))     if row["MBR_Rate"]    != "—" else None
    contam_num  = float(row["Contam_Rate"].rstrip("%"))  if row["Contam_Rate"] != "—" else None
    run_badge   = f'<span class="badge badge-purple" style="float:right;margin-top:2px;">{_run_label}</span>'
    return f"""<div class="container">
<div class="section">
  <h2>{s} \u2014 Summary {run_badge}</h2>
  <table class="stats-table small-table"><tbody>
    <tr><td>MS/MS detected</td><td><strong>{row['MS_MS']:,}</strong>{diff_badge(row['MS_MS'], p.get('msms'))}</td></tr>
    <tr><td>MBR detected</td><td><strong>{row['MBR']:,}</strong>{diff_badge(row['MBR'], p.get('mbr'), higher_is_better=False)}</td></tr>
    <tr><td>Total detected</td><td><strong>{row['Total_Detected']:,}</strong>{diff_badge(row['Total_Detected'], p.get('total_detected'))}</td></tr>
    <tr><td>MBR rate</td><td><strong>{row['MBR_Rate']}</strong>{diff_badge(mbr_num, p.get('mbr_rate_num'), higher_is_better=False, is_pct=True)}</td></tr>
    <tr><td>Contaminants</td><td><strong>{row['Contaminants']:,}</strong>{diff_badge(row['Contaminants'], p.get('contam_count'), higher_is_better=False)}</td></tr>
    <tr><td>Contaminant rate</td><td><strong>{row['Contam_Rate']}</strong>{diff_badge(contam_num, p.get('contam_rate_num'), higher_is_better=False, is_pct=True)}</td></tr>
    <tr><td>MHC-I peptides (8–11 aa)</td><td><strong>{figs.get('mhci_count', 0):,}</strong>{diff_badge(figs.get('mhci_count'), p.get('mhci_count'))}</td></tr>
  </tbody></table>
</div>
<div class="section">
  <h2>{s} \u2014 Peptide Length Distribution</h2>
  <div class="plot-grid">{figs['len_html']}</div>
</div>
<div class="section">
  <h2>{s} \u2014 Protein Source &amp; Charge State</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start;">
    <div style="min-width:0;overflow:hidden;">{figs['src_html']}</div>
    <div style="min-width:0;overflow:hidden;">{figs['chg_html']}</div>
  </div>
</div>
<div class="section">
  <h2>{s} \u2014 Spectral Count &amp; Intensity</h2>
  <p class="note">Distribution of spectral counts and log\u2082 intensities for detected peptides in this sample.</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start;">
    <div style="min-width:0;overflow:hidden;">{figs['sc_html']}</div>
    <div style="min-width:0;overflow:hidden;">{figs['int_html']}</div>
  </div>
</div>
<div class="section">
  <h2>{s} \u2014 Binding Motif (9-mers)</h2>
  <p class="note">Sequence logo and position-specific amino acid frequency for 9-mer MHC-I peptides (non-contaminant). Gold highlights mark P2 and P9 anchor positions.</p>
  <div style="margin-top:16px;">{figs['logo_html']}</div>
  <div class="plot-grid" style="margin-top:24px;">{figs['heatmap_html']}</div>
</div>
<div class="section">
  <h2>{s} \u2014 Peptide Overlap with Other Samples</h2>
  <p class="note">Number of peptides shared between this sample and each other sample (all detected, non-filtered).</p>
  <div class="plot-grid">{figs['overlap_html']}</div>
</div>
<div class="section">
  <h2>{s} \u2014 Top Proteins</h2>
  <p class="note">Top 15 source proteins by number of detected peptides (contaminants excluded).</p>
  <div class="plot-grid">{figs['prot_html']}</div>
</div>
<div class="section">
  <h2>{s} \u2014 IEDB Cross-reference (MHC-I)</h2>
  <p class="note">MHC-I peptides (8\u201311 aa, non-contaminant) detected in this sample vs. the full IEDB epitope database.</p>
  <div style="display:flex;justify-content:center;margin-top:12px;">{figs['venn_html']}</div>
</div>
</div>"""


# Build tab navigation buttons
tab_buttons  = "<button class=\"tab-btn\" data-tab=\"glossary\" onclick=\"switchTab('glossary')\">Glossary</button>\n"
tab_buttons += "<button class=\"tab-btn active\" data-tab=\"overview\" onclick=\"switchTab('overview')\">Overview</button>\n"
tab_buttons += "<button class=\"tab-btn\" data-tab=\"history\" onclick=\"switchTab('history')\">&#128337; History</button>\n"
for s in samples:
    sid = safe_id(s)
    tab_buttons += (
        f"<button class=\"tab-btn\" data-tab=\"{sid}\" onclick=\"switchTab('{sid}')\">{s}</button>\n"
    )

# Build per-sample tab content blocks
sample_tab_contents = ""
for s in samples:
    sid = safe_id(s)
    sample_tab_contents += f"<div class=\"tab-content hidden\" id=\"tab-{sid}\">\n"
    _prev_s = prev_run["per_sample"].get(s) if prev_run else None
    sample_tab_contents += build_sample_page(s, sample_figs[s], _prev_s)
    sample_tab_contents += "\n</div>\n"

CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; background: #f7f8fa; color: #222; }
  .page-header { background: #1a1f36; color: #fff; padding: 32px 48px; }
  .page-header h1 { margin: 0 0 4px; font-size: 1.7rem; }
  .page-header p  { margin: 0; opacity: .7; font-size: .9rem; }
  /* ── sidebar navigation ── */
  .app-body { display: flex; align-items: stretch; }
  .sidebar { width: 190px; min-width: 190px; background: #fff;
             border-right: 1px solid #e0e4ef;
             position: sticky; top: 0; height: 100vh; overflow-y: auto;
             padding: 16px 0; box-shadow: 2px 0 6px rgba(0,0,0,.04); z-index: 100; }
  .sidebar-label { font-size: .7rem; font-weight: 700; letter-spacing: .08em;
                   text-transform: uppercase; color: #9aa; padding: 10px 18px 4px; }
  .tab-btn { display: block; width: 100%; text-align: left; background: none;
             border: none; border-left: 3px solid transparent;
             padding: 9px 18px; cursor: pointer; font-size: .85rem; color: #555;
             font-weight: 500; transition: background .12s, color .12s;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .tab-btn:hover  { color: #1a1f36; background: #f0f2f8; }
  .tab-btn.active { color: #1a1f36; border-left-color: #4C72B0;
                    background: #eef1ff; font-weight: 700; }
  .content-area { flex: 1; min-width: 0; }
  .tab-content.hidden { display: none; }
  /* ── layout ── */
  .container { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }
  .section { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
             padding: 28px 32px; margin-bottom: 32px; }
  .section h2 { margin-top: 0; font-size: 1.2rem; border-bottom: 2px solid #e0e4ef;
                padding-bottom: 10px; color: #1a1f36; }
  .section h3 { font-size: 1rem; color: #444; margin: 24px 0 8px; }
  .stats-table { border-collapse: collapse; width: 100%; font-size: .88rem; margin-top: 8px; }
  .stats-table th { background: #1a1f36; color: #fff; padding: 9px 14px;
                    text-align: left; font-weight: 600; white-space: nowrap; }
  .stats-table td { padding: 7px 14px; border-bottom: 1px solid #edf0f7; }
  .stats-table tr:nth-child(even) td { background: #f7f8fc; }
  .stats-table tr:hover td { background: #eef1ff; }
  .small-table { max-width: 520px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: .78rem; font-weight: 600; }
  .badge-green  { background: #d4edda; color: #155724; }
  .badge-purple { background: #e2d9f3; color: #4a1d96; }
  .badge-grey   { background: #e9ecef; color: #495057; }
  .note { font-size: .82rem; color: #666; margin-top: 10px; }
  .desc { font-size: .9rem; color: #444; line-height: 1.65; margin: 10px 0 18px; }
  .plot-grid { display: grid; grid-template-columns: 1fr; gap: 24px; margin-top: 16px; }
  .site-footer { background: #f7f8fc; border-top: 1px solid #dde1f0; padding: 18px 40px;
                 font-size: .8rem; color: #888; display: flex; justify-content: space-between;
                 align-items: center; flex-wrap: wrap; gap: 8px; margin-top: 40px; }
  /* ── diff badges ── */
  .diff-badge { display: inline-block; padding: 1px 7px; border-radius: 8px;
                font-size: .75rem; font-weight: 600; margin-left: 6px; vertical-align: middle; }
  .diff-good  { background: #d4edda; color: #155724; }
  .diff-bad   { background: #f8d7da; color: #721c24; }
  /* ── glossary ── */
  .glossary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 20px; }
  .glossary-card { background: #f7f8fc; border-left: 3px solid #4C72B0;
                   border-radius: 6px; padding: 12px 16px; }
  .glossary-card .term { display: block; font-weight: 700; font-size: .93rem;
                         color: #1a1f36; margin-bottom: 4px; }
  .glossary-card .def  { font-size: .82rem; color: #555; line-height: 1.55; }
  .glossary-section-title { font-size: .75rem; font-weight: 700; letter-spacing: .07em;
                             text-transform: uppercase; color: #9aa; margin: 28px 0 10px; }
</style>
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Immunopeptidomics QC Report</title>
{PLOTLY_CDN}
{CSS}
</head>
<body>

<div class="page-header">
  <h1>Immunopeptidomics QC Report</h1>
  <p>FragPipe output &mdash; Neoantigen discovery pipeline &mdash; {len(samples)} samples
     &nbsp;&nbsp;|&nbsp;&nbsp; <strong>{_run_label}</strong>
     &nbsp;&nbsp;|&nbsp;&nbsp; {len(state.runs())} run(s) on record</p>
</div>

<div class="app-body">
<nav class="sidebar">
  <div class="sidebar-label">Pages</div>
{tab_buttons}
</nav>
<div class="content-area">

<!-- ══════════════════════════════════════════════════════ GLOSSARY TAB -->
<div class="tab-content hidden" id="tab-glossary">
<div class="container">
<div class="section">
  <h2>Glossary &mdash; Terms &amp; Acronyms</h2>
  <p class="desc">
    A reference guide to the terminology used throughout this report.
    Hover over any term for context, or use this page before diving into the results.
  </p>

  <div class="glossary-section-title">Experimental biology</div>
  <div class="glossary-grid">
    <div class="glossary-card">
      <span class="term">Immunopeptidomics</span>
      <span class="def">The large-scale study of peptides presented on MHC molecules.
        Uses mass spectrometry to identify the full repertoire of peptides displayed on a cell surface.</span>
    </div>
    <div class="glossary-card">
      <span class="term">MHC (Major Histocompatibility Complex)</span>
      <span class="def">Cell-surface proteins that bind and display peptide fragments for immune surveillance.
        In humans, the MHC is encoded by the HLA gene cluster.</span>
    </div>
    <div class="glossary-card">
      <span class="term">HLA (Human Leukocyte Antigen)</span>
      <span class="def">The human form of MHC. HLA class I (A, B, C) presents peptides to CD8+ cytotoxic T cells;
        HLA class II (DR, DQ, DP) presents to CD4+ helper T cells.</span>
    </div>
    <div class="glossary-card">
      <span class="term">MHC-I / HLA class I</span>
      <span class="def">Binds short peptides of 8&ndash;11 amino acids. Present on virtually all nucleated cells.
        The primary target in this pipeline for neoantigen discovery.</span>
    </div>
    <div class="glossary-card">
      <span class="term">MHC-II / HLA class II</span>
      <span class="def">Binds longer peptides greater than 13 amino acids. Present mainly on professional
        antigen-presenting cells (dendritic cells, macrophages, B cells).</span>
    </div>
    <div class="glossary-card">
      <span class="term">Immunoprecipitation (IP)</span>
      <span class="def">Antibody-based enrichment step that pulls down HLA molecules and their associated
        peptides from cell lysates. Specificity of the antibody determines which HLA class is enriched.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Neoantigen</span>
      <span class="def">A tumor-specific peptide arising from somatic mutations in cancer cells.
        Not present in normal tissues, making it a target for personalized immunotherapy.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Anchor residues</span>
      <span class="def">Specific amino acid positions (P2 and P9 for 9-mer MHC-I peptides) that insert
        into pockets in the MHC binding groove. Strongly conserved within an HLA allele&rsquo;s motif.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Immunopeptidome</span>
      <span class="def">The complete set of peptides presented by MHC molecules on a given cell type
        under specific conditions. Reflects the cell&rsquo;s protein expression and processing state.</span>
    </div>
  </div>

  <div class="glossary-section-title">Mass spectrometry &amp; quantification</div>
  <div class="glossary-grid">
    <div class="glossary-card">
      <span class="term">MS/MS (Tandem Mass Spectrometry)</span>
      <span class="def">A peptide is isolated, fragmented, and the resulting fragment ions are measured.
        The fragmentation pattern provides a direct spectral identification with high confidence.</span>
    </div>
    <div class="glossary-card">
      <span class="term">MBR (Match Between Runs)</span>
      <span class="def">A peptide detected by MS/MS in one run is &ldquo;transferred&rdquo; to other runs
        using retention time alignment. No direct spectrum &mdash; identification is inferred.
        High MBR rates (&gt;30%) may indicate quality concerns.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Spectral Count</span>
      <span class="def">The number of MS/MS spectra matched to a peptide. A semi-quantitative
        measure of abundance; higher counts indicate more confident and more abundant detections.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Intensity</span>
      <span class="def">The integrated ion peak area from the extracted ion chromatogram.
        More sensitive and continuous than spectral count; used for quantitative comparisons.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Charge State</span>
      <span class="def">The number of protons on a peptide ion during MS analysis.
        Short MHC-I peptides typically appear as 2+ ions; longer MHC-II peptides as 3+.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Log&#8322; (Log base 2)</span>
      <span class="def">A log transformation that compresses the wide dynamic range of MS intensities
        into a near-normal distribution, making statistical analysis and visualisation more tractable.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Missing Value</span>
      <span class="def">A peptide detected in some runs but not others, resulting in no intensity value.
        High missing rates reduce statistical power and may indicate run-to-run variability.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Contaminant</span>
      <span class="def">Non-cellular proteins co-purified during IP (e.g. keratins, albumin, serum proteins).
        Identified by the <code>Cont_</code> prefix in FragPipe output; excluded from neoantigen analysis.</span>
    </div>
  </div>

  <div class="glossary-section-title">Software &amp; databases</div>
  <div class="glossary-grid">
    <div class="glossary-card">
      <span class="term">FragPipe</span>
      <span class="def">A mass spectrometry data analysis pipeline (MSFragger + downstream tools).
        Performs peptide database search, FDR filtering, and label-free quantification including MBR.</span>
    </div>
    <div class="glossary-card">
      <span class="term">MSstats</span>
      <span class="def">An R/Python package for statistical analysis of quantitative proteomics data.
        Provides normalisation, summarisation, and differential abundance testing across conditions.</span>
    </div>
    <div class="glossary-card">
      <span class="term">IEDB (Immune Epitope Database)</span>
      <span class="def">A public repository of experimentally validated immune epitopes from the literature.
        Used here to distinguish known self/pathogen epitopes from novel neoantigen candidates.</span>
    </div>
    <div class="glossary-card">
      <span class="term">UniProt / SwissProt (sp|)</span>
      <span class="def">SwissProt is the manually reviewed, high-quality section of UniProt.
        Peptides from <code>sp|</code> entries originate from well-characterised canonical proteins.</span>
    </div>
    <div class="glossary-card">
      <span class="term">TrEMBL (tr|)</span>
      <span class="def">The computationally annotated, unreviewed section of UniProt.
        Lower confidence than SwissProt; may include isoforms, predicted ORFs, or novel sequences.</span>
    </div>
    <div class="glossary-card">
      <span class="term">g:Profiler</span>
      <span class="def">A web tool for functional enrichment analysis. Maps gene lists to GO terms,
        KEGG pathways, and Reactome pathways, applying Benjamini-Hochberg FDR correction.</span>
    </div>
    <div class="glossary-card">
      <span class="term">GO (Gene Ontology)</span>
      <span class="def">A standardised vocabulary describing biological processes (BP), molecular functions (MF),
        and cellular components (CC). GO:BP enrichment is most relevant for immunopeptidome interpretation.</span>
    </div>
    <div class="glossary-card">
      <span class="term">KEGG / Reactome</span>
      <span class="def">Curated pathway databases. KEGG covers metabolic and signalling pathways;
        Reactome focuses on human biological reactions. Both complement GO for pathway-level interpretation.</span>
    </div>
  </div>

  <div class="glossary-section-title">Statistics &amp; visualisation</div>
  <div class="glossary-grid">
    <div class="glossary-card">
      <span class="term">PCA (Principal Component Analysis)</span>
      <span class="def">A dimensionality-reduction technique. Projects the high-dimensional peptide
        intensity matrix onto axes of maximum variance (principal components), enabling 2D sample clustering.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Jaccard Similarity</span>
      <span class="def">|A &cap; B| &divide; |A &cup; B|. Ranges from 0 (no shared peptides) to 1 (identical sets).
        Measures how similar two samples&rsquo; detected peptide repertoires are.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Information Content (bits)</span>
      <span class="def">An entropy-based measure of amino acid constraint at a given position in a sequence logo.
        Higher bits = stronger conservation = more important for binding specificity.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Pearson Correlation (r)</span>
      <span class="def">Measures linear correlation between two samples&rsquo; log&#8322; intensity profiles.
        Values above ~0.85 indicate good run-to-run reproducibility for biological or technical replicates.</span>
    </div>
    <div class="glossary-card">
      <span class="term">FDR (False Discovery Rate)</span>
      <span class="def">The expected proportion of false positives among all called results.
        Controlled at 1% (0.01) for peptide identification in FragPipe; at 5% for pathway enrichment.</span>
    </div>
    <div class="glossary-card">
      <span class="term">Venn Diagram</span>
      <span class="def">Shows overlap between two sets. Here used to compare detected peptides against
        IEDB: the intersection represents known epitopes; the left-only region represents novel candidates.</span>
    </div>
  </div>
</div>
</div>
</div><!-- /tab-glossary -->

<!-- ══════════════════════════════════════════════════════ OVERVIEW TAB -->
<div class="tab-content" id="tab-overview">
<div class="container">

<!-- ── SECTION 1 ─────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>1 &mdash; Summary</h2>
  <p class="desc">
    High-level detection statistics for each sample. The table distinguishes between
    <strong>MS/MS</strong> identifications (confirmed by a direct fragmentation spectrum)
    and <strong>MBR</strong> identifications (Match Between Runs &mdash; inferred by aligning
    retention times across runs with no direct spectrum). MBR increases peptide recovery
    but reduces confidence; rates above 30% warrant caution. Contaminant peptides originate
    from non-cellular proteins co-purified during immunoprecipitation and must be excluded
    from downstream neoantigen analysis.
  </p>

  <h3>Dataset overview</h3>
  {dataset_stats_html(dataset_stats)}

  <h3>Per-sample detection summary</h3>
  <p class="note">
    <span class="badge badge-green">MS/MS</span> direct spectral match &nbsp;|&nbsp;
    <span class="badge badge-purple">MBR</span> match between runs (inferred) &nbsp;|&nbsp;
    <strong>Total Detected</strong> = MS/MS + MBR &nbsp;|&nbsp;
    <strong>Contam Rate</strong> = contaminant peptides among detected
  </p>
  {summary_table_html(summary_df)}

  <h3>MS/MS vs MBR per sample</h3>
  <div class="plot-grid">
    {fig1_bar_html}
  </div>
</div>

<!-- ── SECTION 2 ─────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>2 &mdash; Peptide Length Distribution</h2>
  <p class="desc">
    The length distribution of detected peptides is a primary quality indicator.
    MHC-I molecules (HLA-A, B, C) present 8&ndash;11 amino acid peptides, with a strong
    peak at 9-mers being the hallmark of a successful HLA-I immunoprecipitation. A broad, 
    flat, or shifted distribution may indicate non-specific peptide capture, degradation,
    or protocol issues.
  </p>
  <p class="note">
    <span class="badge badge-green">MHC-I (8–11 aa)</span> &nbsp;
    The expected MHC-I window (8–11 aa) is the primary target for neoantigen discovery.
  </p>

  <h3>All peptides</h3>
  <div class="plot-grid">
    {fig2_all_html}
  </div>

  <h3>Per sample (detected, 8–15 aa)</h3>
  <div class="plot-grid">
    {fig2_sample_html}
  </div>
</div>

<!-- ── SECTION 3 ─────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>3 &mdash; Spectral Count Distribution</h2>
  <p class="desc">
    Spectral count is a semi-quantitative proxy for peptide abundance &mdash; the more spectra
    matched to a peptide, the more confidently and abundantly it is detected. Violin plots
    show the per-sample distribution on a log scale. Samples with systematically lower counts
    or a larger proportion of single-spectrum peptides may have lower input material or worse
    chromatography. The MBR rate bar highlights samples where a high fraction of peptides
    lack direct spectral evidence; these samples carry greater uncertainty in quantification.
  </p>
  <p class="note">
    Violin plots show the distribution of spectral counts for detected peptides (non-zero only, log scale).
    High MBR rate (&gt;30%) may indicate poor raw file quality or aggressive imputation.
  </p>

  <h3>Spectral count distribution per sample</h3>
  <div class="plot-grid">
    {fig3_violin_html}
  </div>

  <h3>MBR rate per sample</h3>
  <div class="plot-grid">
    {fig3_mbr_html}
  </div>
</div>

<!-- ── SECTION 4 ─────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>4 &mdash; Intensity Distribution</h2>
  <p class="desc">
    Peptide intensity (integrated ion peak area) is a continuous, more sensitive quantitative
    measure than spectral count. Log&#8322;-transformed intensities from MSstats should follow
    a roughly bell-shaped distribution; shifts between conditions suggest systematic differences
    in abundance or sample loading. Missing value rates per run reflect how many peptides
    were detected but not quantified &mdash; rates above 40&ndash;50% indicate that many peptides
    are near the detection limit. Replicate correlation scatter plots (r values) confirm
    run-to-run reproducibility; r &gt; 0.9 is expected for technical replicates.
  </p>
  <p class="note">
    Intensity values from MSstats. Violins show log&#8322; intensity per condition.
    Ahmed replicates (1/2/3) are correlated pairwise to assess run reproducibility.
  </p>

  <h3>Log&#8322; intensity per condition</h3>
  <div class="plot-grid">
    {fig4_violin_html}
  </div>

  <h3>Missing value rate per run</h3>
  <div class="plot-grid">
    {fig4_missing_html}
  </div>

  <h3>Ahmed replicate correlations (log&#8322; intensity)</h3>
  <div class="plot-grid">
    {fig4_corr_html}
  </div>
</div>

<!-- ── SECTION 5 ─────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>5 &mdash; Contaminants</h2>
  <p class="desc">
    Contaminant proteins enter the sample during cell lysis and immunoprecipitation: common
    culprits include human keratins (from skin cells), bovine serum albumin (from culture media),
    and other abundant cellular housekeeping proteins. FragPipe flags these with a
    <code>Cont_</code> prefix. A contaminant rate below 5% is typical for a clean IP;
    higher rates suggest reagent contamination or non-specific antibody binding.
    Contaminant peptides must be removed before calling neoantigens.
  </p>
  <p class="note">
    Contaminant proteins are identified by the <code>Cont_</code> prefix in the Protein column.
    These should be excluded from neoantigen candidate lists.
  </p>

  <h3>Contaminant rate per sample</h3>
  <div class="plot-grid">
    {fig5_bar_html}
  </div>

  <h3>Top contaminant proteins (dataset-wide)</h3>
  {contam_table_html(contam_protein_counts)}
</div>

<!-- ── SECTION 9 ─────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>9 &mdash; Sequence Logo / Binding Motif</h2>
  <p class="desc">
    A sequence logo is a graphical summary of amino acid preferences at each position
    of the peptide. The <strong>frequency logo</strong> shows raw probabilities; the
    <strong>information content logo</strong> (in bits) weights each position by how
    strongly constrained it is &mdash; positions with low entropy (high bits) are the
    most biologically meaningful. The two gold-shaded columns at P2 and P9 are the
    canonical MHC-I anchor positions. Strong hydrophobic enrichment there (L, V, I, M)
    is the hallmark of a successful HLA-I immunoprecipitation and confirms the correct
    allele-specific motif.
  </p>
  <p class="note">
    Built from {len(mers9):,} 9-mer peptides.
    <span style="background:#ffd700;padding:1px 6px;border-radius:3px;">Gold shading</span>
    marks canonical MHC-I anchor positions P2 and P9, where hydrophobic residues
    (L, V, I, M) are expected to be enriched.
    Frequency logo shows raw probability; information content logo (bits) highlights
    positions with strongest amino-acid preference.
  </p>
  <div class="plot-grid" style="margin-top:16px;">
    {sec9_logo_html}
  </div>
</div>

<!-- ── SECTION 10 ────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>10 &mdash; Charge State Distribution</h2>
  <p class="desc">
    During electrospray ionisation, peptides acquire protons and enter the mass spectrometer
    as multiply-charged ions. Short MHC-I peptides (8&ndash;11 aa) typically carry a
    <strong>2+ charge</strong> due to one basic N-terminus and one basic C-terminal residue.
    Longer MHC-II peptides (13&ndash;25 aa) often carry <strong>3+</strong> due to additional
    internal basic residues (K, R, H). A dominant 2+ population confirms a clean MHC-I
    immunopeptidome. Unexpected 1+ or &ge;4+ enrichment may indicate co-purified small
    molecules, non-HLA peptides, or in-source fragmentation artifacts.
  </p>
  <p class="note">
    HLA-I peptides are predominantly <strong>2+</strong>; HLA-II peptides skew toward
    <strong>3+</strong>. Unexpected enrichment of 1+ or 4+ may indicate IP quality issues
    or co-purified non-HLA peptides.
    Some peptides are observed at multiple charge states (e.g. 2,3 — counted for each).
  </p>
  <div class="plot-grid" style="display:grid;grid-template-columns:1fr 2fr;gap:24px;margin-top:16px;">
    <div>{fig10_pie_html}</div>
    <div>{fig10_sample_html}</div>
  </div>
</div>

<!-- ── SECTION 11 ────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>11 &mdash; Sample Clustering / PCA</h2>
  <p class="desc">
    Principal Component Analysis (PCA) reduces the high-dimensional peptide intensity matrix
    to a 2D scatter plot, revealing overall sample structure. Samples with similar
    immunopeptidomes (shared HLA alleles, same cell background) will cluster together.
    Biological replicates should overlap; large separation may indicate batch effects or
    true biological differences. PC1 and PC2 capture the largest axes of variance &mdash;
    their explained variance percentages indicate how much of the total information is shown.
    The correlation heatmap (hierarchically clustered) quantifies pairwise Pearson
    correlations: values &gt; 0.85 indicate good reproducibility between runs.
  </p>
  <p class="note">
    PCA on log&#8322; intensities of peptides detected in &ge;3 samples (missing values
    imputed with per-peptide minimum). Samples that cluster together share similar
    immunopeptidomes. Outlier samples warrant investigation before neoantigen calling.
  </p>
  <h3>PCA Scatter</h3>
  <div class="plot-grid">
    {fig11_pca_html}
  </div>
  <div class="plot-grid">
    {fig11_var_html}
  </div>
  <h3>Sample correlation heatmap (hierarchically clustered)</h3>
  <div class="plot-grid">
    {fig11_corr_html}
  </div>
</div>

<!-- ── SECTION 12 ────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>12 &mdash; IEDB Cross-reference</h2>
  <p class="desc">
    Cross-referencing detected peptides against the Immune Epitope Database (IEDB) serves
    two purposes: <strong>validation</strong> (finding known epitopes confirms the IP protocol
    is working correctly) and <strong>discovery</strong> (peptides absent from IEDB are novel
    candidates). A healthy immunopeptidome experiment typically recovers a substantial fraction
    of known HLA-I epitopes. Peptides unique to this dataset &mdash; especially those from
    mutated proteins &mdash; are high-priority neoantigen candidates requiring further
    validation (mutation evidence, MHC binding prediction, T-cell reactivity assays).
  </p>
  <p class="note">
    All canonical MHC-I peptides (8&ndash;11 aa) detected across any sample are compared
    against the local IEDB epitope database (<code>iedb_public.sql</code>).
    A high fraction of <strong>known</strong> epitopes validates the IP protocol.
    <strong>Novel candidates</strong> are peptides not previously described in IEDB
    and warrant further validation (mutation evidence, expression, MHC binding prediction).
    Sequence set cached in <code>.iedb_seqs.pkl</code> for fast re-runs.
  </p>
  {dataset_stats_html(iedb_summary)}
  <h3>Peptide overlap &mdash; Detected MHC-I vs IEDB</h3>
  <div style="display:flex;justify-content:center;margin-top:16px;">{sec12_venn_html}</div>
  <h3>Donut chart breakdown</h3>
  <div class="plot-grid" style="margin-top:16px;">{fig12_pie_html}</div>
  <h3>Known IEDB epitopes detected (top 30 by spectral count)</h3>
  {iedb_table_html(known_df)}
</div>

<!-- ── SECTION 13 ────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>13 &mdash; GO / Pathway Enrichment</h2>
  <p class="desc">
    Pathway enrichment analysis maps the source genes of detected peptides onto biological
    processes and pathways. In a successful HLA-I immunopeptidomics experiment, the top
    enriched terms should include <em>antigen processing and presentation</em>
    (GO:0019882) and MHC class I-related terms, confirming that the pipeline is capturing
    bona fide HLA-presented peptides. Unexpected dominant pathways (e.g. mitochondrial,
    extracellular matrix) may indicate co-purified material or biological phenomena worth
    investigating. Terms are filtered at FDR &lt; 5% (Benjamini-Hochberg correction).
  </p>
  <p class="note">
    Gene set: {len(enrich_genes):,} unique canonical genes detected across all samples.
    Enrichment via <a href="https://biit.cs.ut.ee/gprofiler/" target="_blank">g:Profiler</a>
    (g:GOSt, Benjamini-Hochberg FDR &lt; 0.05).
    Dominant antigen processing &amp; presentation terms confirm successful immunopeptidomics;
    unexpected pathways may indicate contamination or co-purified material.
  </p>
  <h3>GO Biological Process</h3>
  <div class="plot-grid">{fig13_gobp_html}</div>
  <h3>KEGG Pathways</h3>
  <div class="plot-grid">{fig13_kegg_html}</div>
  <h3>Reactome Pathways</h3>
  <div class="plot-grid">{fig13_reac_html}</div>
</div>

<!-- ── SECTION 6 ─────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>6 &mdash; Peptide Overlap Between Samples</h2>
  <p class="desc">
    Compares the detected peptide repertoires across samples. High Jaccard similarity between
    samples typically reflects shared HLA alleles, the same cell line background, or similar
    biological conditions. Low similarity may indicate different HLA types or distinct tumour
    biology. Peptides detected in only one sample (<em>singletons</em>) are of particular
    interest as private neoantigens or rare self-peptides. The prevalence histogram shows
    how many peptides are broadly shared vs. sample-specific.
  </p>
  <p class="note">
    Jaccard similarity = |A &cap; B| / |A &cup; B|. Values close to 1 indicate high overlap.
    Singleton peptides (detected in only one sample) are neoantigen candidates of interest.
  </p>

  <h3>Jaccard similarity heatmap</h3>
  <div class="plot-grid">
    {fig6_jaccard_html}
  </div>

  <h3>Shared peptide count heatmap</h3>
  <div class="plot-grid">
    {fig6_shared_html}
  </div>

  <h3>Peptide prevalence (how many samples each peptide appears in)</h3>
  <div class="plot-grid">
    {fig6_prev_html}
  </div>
</div>

<!-- ── SECTION 7 ─────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>7 &mdash; Protein Source</h2>
  <p class="desc">
    Shows where the detected peptides originate in the protein database. The majority should
    derive from <strong>SwissProt</strong> (manually reviewed, high-confidence canonical proteins),
    which confirms that the immunopeptidome is dominated by well-characterised cellular proteins.
    <strong>TrEMBL</strong> entries are computationally predicted and less curated; peptides
    from TrEMBL may represent isoforms, predicted ORFs, or non-canonical sources such as
    circRNA-derived proteins. The unique gene count per sample provides a measure of proteome
    coverage, roughly reflecting sample depth and HLA allele diversity.
  </p>
  <p class="note">
    <strong>SwissProt (sp)</strong> = reviewed canonical proteins &nbsp;|&nbsp;
    <strong>TrEMBL (tr)</strong> = unreviewed &nbsp;|&nbsp;
    <strong>Contaminant</strong> = <code>Cont_</code> prefixed entries.
    Neoantigens derive from canonical or non-canonical (e.g. circRNA) protein sources.
  </p>

  <h3>Dataset-wide protein source</h3>
  <div class="plot-grid">
    {fig7_pie_html}
  </div>

  <h3>Protein source per sample</h3>
  <div class="plot-grid">
    {fig7_src_sample_html}
  </div>

  <h3>Unique canonical genes per sample</h3>
  <div class="plot-grid">
    {fig7_genes_html}
  </div>
</div>

<!-- ── SECTION 8 ─────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>8 &mdash; Amino Acid Composition</h2>
  <p class="desc">
    Characterises the amino acid (AA) makeup of the immunopeptidome. The position-specific
    heatmap for 9-mers reveals the HLA binding motif: anchor positions P2 and P9 (positions 2
    and 9 in the peptide) interact directly with the HLA binding groove and are strongly
    constrained to specific residues (typically hydrophobic: L, V, I, M). Strong enrichment
    at these positions confirms MHC-I binding. The overall AA frequency comparison to the
    human proteome background reveals biases introduced by HLA selection &mdash; for example,
    basic residues at the C-terminus (K, R) are enriched in many HLA-A/B alleles.
  </p>
  <p class="note">
    Position frequency heatmap for 9-mer peptides (primary MHC-I length).
    Anchor residues at P2 and P9 typically show strong hydrophobic enrichment (L, V, I, M).
    The bar chart compares the overall immunopeptidome AA frequency to the human proteome background.
  </p>

  <h3>Position-specific AA frequency — 9-mers</h3>
  <div class="plot-grid">
    {fig8_heatmap_html}
  </div>

  <h3>Overall AA frequency vs human proteome background</h3>
  <div class="plot-grid">
    {fig8_freq_html}
  </div>
</div>

</div><!-- /container overview -->
</div><!-- /tab-overview -->

<!-- ══════════════════════════════════════════════ HISTORY TAB -->
{build_history_tab_html(state.runs())}

<!-- ══════════════════════════════════════════════ PER-SAMPLE TABS -->
{sample_tab_contents}

</div><!-- /content-area -->
</div><!-- /app-body -->

<footer class="site-footer">
  <span>Developed by the He Lab at the Princess Margaret Cancer Centre</span>
  <span>Last updated: {datetime.now().strftime("%B %d, %Y")}</span>
</footer>

<script>
function switchTab(tabId) {{
  document.querySelectorAll('.tab-content').forEach(function(el) {{
    el.classList.add('hidden');
  }});
  document.querySelectorAll('.tab-btn').forEach(function(el) {{
    el.classList.remove('active');
  }});
  document.getElementById('tab-' + tabId).classList.remove('hidden');
  document.querySelector('[data-tab="' + tabId + '"]').classList.add('active');
  window.dispatchEvent(new Event('resize'));
}}
</script>

</body>
</html>
"""

OUT.write_text(html)
print(f"Done → {OUT.resolve()}")
