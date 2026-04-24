"""Shared UI utilities: prototype gate, sidebar auth, CSS injection."""
from __future__ import annotations

import streamlit as st

from modules.auth import hash_password, password_strength_error, verify_password
from modules.database import create_user, get_user_by_username, username_exists

_PROTOTYPE_PASSWORD = "helab"

_SIDEBAR_CSS = """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { min-width: 220px !important; }
</style>
"""

_GATE_CSS = """
<style>
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; min-width: 0 !important; }
</style>
"""


def inject_sidebar_css() -> None:
    """Inject CSS that keeps the sidebar always visible and non-collapsible."""
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)


def check_prototype_gate() -> None:
    """Show a password gate before any app content if not yet authenticated.

    Calls st.stop() when the gate is active so nothing else renders.
    """
    if st.session_state.get("prototype_authenticated"):
        return

    st.markdown(_GATE_CSS, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("## Immunopeptidomics QC")
        st.markdown("This application is a prototype for internal use.")
        pwd = st.text_input("Password", type="password", key="_gate_pwd")
        if st.button("Enter", key="_gate_btn"):
            if pwd == _PROTOTYPE_PASSWORD:
                st.session_state["prototype_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")

    st.stop()


def render_sidebar_auth() -> None:
    """Render the login/logout block inside the current sidebar context.

    Call this inside a ``with st.sidebar:`` block (or from within app.py where
    the sidebar context is already active).
    """
    if st.session_state.get("user_id"):
        st.markdown(f"Logged in as: **{st.session_state['username']}**")
        if st.button("Log out", use_container_width=True, key="_sb_logout"):
            for k in ["user_id", "username"]:
                st.session_state.pop(k, None)
            st.rerun()
    else:
        u = st.text_input(
            "Username",
            key="_sb_login_user",
            label_visibility="collapsed",
            placeholder="Username",
        )
        p = st.text_input(
            "Password",
            type="password",
            key="_sb_login_pass",
            label_visibility="collapsed",
            placeholder="Password",
        )
        if st.button("Log in", use_container_width=True, key="_sb_login_btn"):
            user = get_user_by_username(u.strip())
            if user and verify_password(p, user["password_hash"]):
                st.session_state["user_id"] = user["id"]
                st.session_state["username"] = user["username"]
                st.rerun()
            else:
                st.error("Invalid username or password.")

        with st.expander("Create account"):
            nu = st.text_input("Username", key="_sb_signup_user")
            ne = st.text_input("Email", key="_sb_signup_email")
            np_ = st.text_input("Password", type="password", key="_sb_signup_pass")
            if st.button("Create account", use_container_width=True, key="_sb_signup_btn"):
                err = password_strength_error(np_)
                if err:
                    st.error(err)
                elif username_exists(nu.strip()):
                    st.error("Username already taken.")
                else:
                    uid = create_user(nu.strip(), ne.strip(), hash_password(np_))
                    st.session_state["user_id"] = uid
                    st.session_state["username"] = nu.strip()
                    st.success("Account created.")
                    st.rerun()
