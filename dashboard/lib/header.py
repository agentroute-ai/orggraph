"""Shared page header: title + optional caption."""
from __future__ import annotations

import streamlit as st


def render_header(title: str = "OrgGraph", subtitle: str | None = None) -> None:
    """Render the standard OrgGraph page header.

    Call this at the top of every page (after st.set_page_config).
    """
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()
