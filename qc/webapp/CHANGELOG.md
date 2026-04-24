# Changelog

## [0.3.0] — 2026-04-24

### Summary

UI and navigation overhaul. No changes to data processing, metrics, charts, or
the Upload → Map → Report workflow behaviour.

---

### Added

**Prototype password gate** (item 5)
- All pages are now protected by a shared password gate before any content renders.
- The gate shows only a centred form with the application title, a password input,
  and an Enter button. No sidebar, navigation, or footer is visible until authenticated.
- Authentication persists for the duration of the browser session.
- Implemented in `modules/ui_utils.py` (`check_prototype_gate()`); called from `app.py`
  before navigation renders.

**Persistent sidebar login / user display** (item 6)
- Logged-out state: compact Username and Password inputs plus a Log in button are
  always visible in the sidebar on every page, including About and Glossary.
- A "Create account" expander below the login fields allows account creation without
  leaving the current page.
- Logged-in state: "Logged in as: [username]" text and a Log out button replace the
  login form on every page.
- Implemented in `modules/ui_utils.py` (`render_sidebar_auth()`); called from the
  `app.py` shell so the auth block appears on all pages without per-page duplication.
- Sidebar layout order (top to bottom): application title → navigation links → divider
  → auth block → divider → footer.

**Shared UI utility module** (`modules/ui_utils.py`)
- `inject_sidebar_css()` — injects CSS that hides the sidebar collapse/toggle button
  and enforces a minimum sidebar width of 220 px.
- `check_prototype_gate()` — renders the prototype password gate and calls `st.stop()`
  if the user has not yet authenticated.
- `render_sidebar_auth()` — renders the login/logout block inside the sidebar context.

---

### Changed

**Always-visible sidebar** (item 1)
- `initial_sidebar_state` changed from `"collapsed"` to `"expanded"` in
  `st.set_page_config()`.
- CSS injected via `inject_sidebar_css()` hides the `[data-testid="collapsedControl"]`
  element so users cannot collapse the sidebar.

**Main page renamed to "Analysis"** (item 2)
- Analysis workflow content moved from `app.py` to a new `pages/analysis.py` page file.
- Page title in the sidebar navigation is "Analysis".
- `st.title()` and step indicator on the analysis page reflect the "Analysis" context.

**Sidebar navigation order** (item 3)
- Navigation now uses `st.navigation()` in `app.py` to define page order explicitly:
  About → Glossary → Analysis → My Runs → Compare Runs.
- `app.py` is now a navigation shell; all page content lives in individual page files.

**Sample mapping remove button** (item 4)
- Replaced the "Remove" `st.button` label with "×" (multiplication sign U+00D7) in
  `pages/analysis.py` (`render_mapping()`).
- Column proportions updated from `[2, 2, 2, 2, 0.5]` to `[2, 3, 3, 3, 0.5]` to give
  the selectbox columns more width.
- The button retains its `help="Remove this sample"` tooltip and full functionality.

**Page file cleanup**
- Removed `st.set_page_config()`, per-page `_render_sidebar()`, and per-page
  `_render_footer()` functions from all page files (`2_About.py`, `3_Glossary.py`,
  `4_My_Runs.py`, `5_Compare_Runs.py`). These are now handled once in `app.py`.
- Removed hard-coded sidebar links (e.g. `/4_My_Runs`, `/5_Compare_Runs`) from page
  content; navigation is handled by the sidebar nav.
- "Sign in as" / "Sign out" wording updated to "Logged in as" / "Log out" throughout.
- "Sign in via the sidebar" prompt in the save-run section updated to "Log in via the
  sidebar".

---

## [0.2.0] — 2026-04-24

### Summary

Major feature expansion of the Immunopeptidomics QC Webapp. All changes preserve
the existing Upload → Map → Report workflow and are backward-compatible with existing
data files.

---

### Added

**Authentication and user accounts**
- `modules/auth.py`: `hash_password`, `verify_password`, `password_strength_error`
  using bcrypt (already present; now wired into the app)
- `modules/database.py`: SQLite-backed user and run storage with ownership enforcement
  (already present; now wired into the app). Added `update_run_data_dir` function.
- Sign-in / Create account UI in the sidebar of every page
- Logout button visible when signed in

**Run persistence**
- "Save Run" panel at the bottom of the Report screen (visible to signed-in users)
- Saves run name, upload timestamp, sample count, summary metrics, and full parsed
  data (parquet + JSON via `modules/storage.py`) to a per-user local directory
  (`data/runs/<user_id>/<run_id>/`)

**My Runs page** (`pages/4_My_Runs.py`)
- Table of all saved runs with: name, date, sample count, median peptide count,
  median MBR rate
- Per-run actions: Open (loads run into Report screen), Delete (with confirmation),
  Compare (mark as run A or B for side-by-side comparison)
- Link to Compare Runs page when two runs are selected

**Compare Runs page** (`pages/5_Compare_Runs.py`)
- Side-by-side dataset-level summary metrics table
- Side-by-side MS/MS vs MBR charts and MBR rate charts
- Side-by-side peptide length distributions
- Side-by-side contaminant rate charts
- Per-sample delta metrics table for samples with matching names across runs
- Explanatory note on why cross-run PCA is not shown

**Per-plot sample filtering**
- Every report tab that shows per-sample data has its own `st.multiselect` widget
- "All" and "Clear" convenience buttons adjacent to each multiselect
- Caption showing "Showing N of M samples" when a subset is active
- Minimum sample enforcement with informative message (2 required for Jaccard,
  shared heatmap, and PCA; 1 for all other sections)
- Filters reset automatically when a new dataset is loaded
- Applies to: MS/MS & MBR, Length, Spectral Counts, Intensity correlations,
  Contaminants, Overlap, Protein Source, Charge, PCA

**Footer**
- "Developed by the He Lab at the Princess Margaret Cancer Centre 2026"
  appears on all pages (app.py, About, Glossary, My Runs, Compare Runs)

---

### Changed

**Emoji removal**
- Removed all emoji from app.py, pages/2_About.py, pages/3_Glossary.py
- Replaced emoji tab labels with plain text: "Summary", "MS/MS and MBR", "Length",
  "Spectral Counts", "Intensity", "Contaminants", "Overlap", "Protein Source",
  "Amino Acids", "Charge", "PCA", "Per-Sample"
- Replaced emoji in buttons, expander labels, help text, step indicator, and
  download buttons
- Replaced page_icon emoji with no icon

**Scientific content**
- Length tab: reworded to use hedged language ("most commonly", "varies by allele")
  consistent with the Glossary
- MS/MS & MBR tab: reworded to clarify the 30% threshold is a heuristic
- Contaminants tab: reworded to note threshold depends on sample type and protocol
- PCA tab: reworded to note biological replicates are "expected" rather than
  "required" to cluster
- Amino Acids tab: reworded anchor position note to indicate allele-level variability

**Glossary** (`pages/3_Glossary.py`)
- MHC class I: length described as "most commonly 8-11 aa, with 9-mers
  predominating in many alleles; optimal length range varies by allele"
- MHC class II: described as "typically 13-25 aa; core binding epitope generally
  9 aa; varies by allele and cleavage context"
- Charge state: describes tendency (not rule) for shorter peptides to appear as
  doubly charged; notes dependence on sequence and instrument settings
- FDR: explicitly distinguishes PSM-level, peptide-level, and protein-level FDR;
  identifies the method as target-decoy estimation
- MBR: explains the transfer mechanism accurately; notes sensitivity increase and
  false transfer risk; identifies 30% threshold as heuristic

**About page** (`pages/2_About.py`)
- Non-clinical disclaimer present and prominent
- User accounts and run history section added
- Technical notes section present

---

### Dependencies added

- `bcrypt>=4.0` — password hashing (was used but missing from pyproject.toml)
- `pyarrow>=14.0` — required by pandas for `.parquet` read/write in storage.py

---

### Setup notes

**First run**

The SQLite database is created automatically on first startup. No manual
migration is needed.

```
cd qc/webapp
uv sync
streamlit run app.py
```

**Run data storage**

Saved runs are written to `qc/webapp/data/runs/<user_id>/<run_id>/`. This
directory is created automatically. To back up or migrate run data, copy the
entire `data/` directory.

**Database location**

`qc/webapp/data/qc.db` — SQLite file, created on first startup.
