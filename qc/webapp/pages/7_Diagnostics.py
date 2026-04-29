"""Server diagnostics — predictor availability, versions, disk space, and env vars.

Only safe, non-secret information is displayed here. Secrets and password hashes
are never shown. This page is useful for verifying a fresh self-hosted deployment.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st

from modules.predictors.registry import ALL_PREDICTORS
from modules.predictors.netmhcpan_predictor import _BINARY as _NETMHCPAN_BINARY, _BUNDLED_BINARY as _NETMHCPAN_BUNDLED

st.set_page_config(page_title="Diagnostics — IPQC", layout="wide")
st.title("Server Diagnostics")
st.caption(
    "Read-only view of the server environment. "
    "No credentials or user data are shown here."
)

# ── 1. Runtime versions ───────────────────────────────────────────────────────

st.subheader("Runtime")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Python", platform.python_version())

with col2:
    try:
        import streamlit as _st
        st_ver = _st.__version__
    except Exception:
        st_ver = "unknown"
    st.metric("Streamlit", st_ver)

with col3:
    git_commit = "unknown"
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).parent.parent,
        )
        if r.returncode == 0:
            git_commit = r.stdout.strip()
    except Exception:
        pass
    st.metric("Git commit", git_commit)

# ── 2. Predictor status ───────────────────────────────────────────────────────

st.subheader("MHC-I Predictors")

_ENV_VARS: dict[str, list[str]] = {
    "MHCflurry": ["MHCFLURRY_DATA_PATH"],
    "NetMHCpan": [],
}

rows = []
for cls in ALL_PREDICTORS:
    available = cls.is_available()
    version = cls.version() if available else "not installed"
    env_vars = _ENV_VARS.get(cls.name, [])
    env_status = ", ".join(
        f"`{v}`={os.environ.get(v, '(unset)')}"
        for v in env_vars
    ) if env_vars else "n/a"

    rows.append({
        "Tool": cls.name,
        "Available": "✅ Yes" if available else "❌ No",
        "Version": version,
        "Env vars": env_status,
    })

import pandas as pd
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── 3. Install hints for missing predictors ───────────────────────────────────

missing = [cls for cls in ALL_PREDICTORS if not cls.is_available()]
if missing:
    with st.expander("Install hints for missing predictors"):
        for cls in missing:
            st.markdown(f"**{cls.name}**")
            st.code(cls.install_hint, language="bash")
else:
    st.success("All predictors are available.")

# ── 4. Environment variables (non-secret) ────────────────────────────────────

st.subheader("Environment Variables")

_WATCHED_VARS = [
    "MHCFLURRY_DATA_PATH",
    "PATH",
]

env_rows = []
for var in _WATCHED_VARS:
    val = os.environ.get(var, "")
    if var == "PATH":
        val = val[:120] + "…" if len(val) > 120 else val
    env_rows.append({"Variable": var, "Value": val or "(unset)"})

# Show the resolved NetMHCpan binary path (bundled or PATH)
_bundled_exists = _NETMHCPAN_BUNDLED.is_file()
env_rows.append({
    "Variable": "netMHCpan binary (resolved)",
    "Value": f"{_NETMHCPAN_BINARY}  {'✅ exists' if Path(_NETMHCPAN_BINARY).exists() else '❌ not found'}",
})

st.dataframe(pd.DataFrame(env_rows), use_container_width=True, hide_index=True)

# ── 5. Disk space ─────────────────────────────────────────────────────────────

st.subheader("Disk Space")

_CHECK_PATHS: list[tuple[str, str]] = [
    ("App data (runs + DB)", str(Path(__file__).parent.parent / "data")),
    ("MHCflurry models", os.environ.get("MHCFLURRY_DATA_PATH", str(Path.home() / ".local/share/mhcflurry"))),
    ("NetMHCpan install", str(_NETMHCPAN_BUNDLED.parent)),
    ("/tmp", "/tmp"),
]

disk_rows = []
for label, path_str in _CHECK_PATHS:
    if not path_str:
        disk_rows.append({"Location": label, "Path": "(not configured)", "Used": "—", "Free": "—", "Total": "—"})
        continue
    p = Path(path_str)
    if not p.exists():
        disk_rows.append({"Location": label, "Path": path_str, "Used": "—", "Free": "path not found", "Total": "—"})
        continue
    try:
        usage = shutil.disk_usage(p)
        def _fmt(n: int) -> str:
            for unit in ("B", "KB", "MB", "GB", "TB"):
                if n < 1024:
                    return f"{n:.1f} {unit}"
                n //= 1024
            return f"{n} TB"
        disk_rows.append({
            "Location": label,
            "Path": path_str,
            "Used": _fmt(usage.used),
            "Free": _fmt(usage.free),
            "Total": _fmt(usage.total),
        })
    except Exception as e:
        disk_rows.append({"Location": label, "Path": path_str, "Used": "—", "Free": str(e), "Total": "—"})

st.dataframe(pd.DataFrame(disk_rows), use_container_width=True, hide_index=True)

# ── 6. CLI sanity checks ──────────────────────────────────────────────────────

st.subheader("CLI Checks")

_CLI_CHECKS: list[tuple[str, list[str]]] = [
    ("netMHCpan", [_NETMHCPAN_BINARY, "-h"]),
    ("git", ["git", "--version"]),
]

cli_rows = []
for label, cmd in _CLI_CHECKS:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        first_line = (r.stdout + r.stderr).strip().splitlines()[0] if (r.stdout + r.stderr).strip() else "(no output)"
        ok = r.returncode == 0
        cli_rows.append({"Check": label, "Result": "✅ Found" if ok else "❌ Failed", "Output": first_line})
    except FileNotFoundError:
        cli_rows.append({"Check": label, "Result": "❌ Not Found", "Output": f"{cmd[0]!r} not on PATH"})
    except subprocess.TimeoutExpired:
        cli_rows.append({"Check": label, "Result": "⚠️ Timeout", "Output": "did not respond in 10 s"})
    except Exception as e:
        cli_rows.append({"Check": label, "Result": "⚠️ Error", "Output": str(e)})

st.dataframe(pd.DataFrame(cli_rows), use_container_width=True, hide_index=True)
