"""Shared page header: top-left app logo + title + optional caption."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_LOGO = _ASSETS / "logo.png"
_LOGO_MARK = _ASSETS / "logo-mark.png"


def render_header(title: str = "OrgGraph", subtitle: str | None = None) -> None:
    """Render the standard OrgGraph page header.

    Call this at the top of every page (after st.set_page_config). Pins the
    OrgGraph logo to the top-left (above the sidebar nav) on every page.
    """
    if _LOGO.is_file():
        st.logo(
            str(_LOGO),
            size="large",
            link="https://github.com/agentroute-ai/orggraph",
            icon_image=str(_LOGO_MARK) if _LOGO_MARK.is_file() else None,
        )
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()
