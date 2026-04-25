"""
frontend/app.py
----------------
Streamlit frontend for DocuSeek.

Talks to the FastAPI backend at DOCUSEEK_API_URL (default: http://localhost:8000).
Set the env var to point at a remote server if needed.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

_API_URL = os.environ.get("DOCUSEEK_API_URL", "http://localhost:8000")
_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocuSeek",
    page_icon="🔍",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Sidebar — experiment info from /health/ready
# ---------------------------------------------------------------------------


def _fetch_health() -> dict | None:
    try:
        r = httpx.get(f"{_API_URL}/health/ready", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


with st.sidebar:
    st.title("DocuSeek")
    st.caption("RAG over ML framework documentation")
    st.divider()

    health = _fetch_health()
    if health:
        st.success("API online", icon="✅")
        st.markdown(f"**Experiment**  \n`{health['experiment']}`")
        st.markdown(f"**Qdrant**  \n{'connected' if health['qdrant'] else '⚠️ unreachable'}")
    else:
        st.error("API offline", icon="🔴")
        st.caption(f"Expected at `{_API_URL}`")

    st.divider()
    st.caption("Docs: HuggingFace · PyTorch · PEFT · Diffusers · Tokenizers · Accelerate")


# ---------------------------------------------------------------------------
# Main — query interface
# ---------------------------------------------------------------------------

st.header("Ask about ML framework docs")

question = st.text_area(
    label="Question",
    placeholder="How do I use LoRA with PEFT for fine-tuning?",
    height=100,
    label_visibility="collapsed",
)

submit = st.button("Ask", type="primary", disabled=not question.strip())

if submit and question.strip():
    with st.spinner("Retrieving and generating…"):
        try:
            response = httpx.post(
                f"{_API_URL}/query",
                json={"question": question.strip()},
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text}")
            st.stop()
        except httpx.RequestError:
            st.error(f"Could not reach the API at `{_API_URL}`. Is the server running?")
            st.stop()

    # ── Answer ───────────────────────────────────────────────────────────────
    st.markdown("### Answer")
    st.markdown(data["answer"])

    # ── Query variants (only shown when rewriting was active) ────────────────
    if data.get("query_variants"):
        with st.expander("Query variants used for retrieval"):
            for i, variant in enumerate(data["query_variants"], start=1):
                st.markdown(f"**{i}.** {variant}")

    # ── Sources ──────────────────────────────────────────────────────────────
    st.divider()
    sources = data.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for src in sources:
                st.markdown(
                    f"**{src['title']}** · `{src['source']}`  \n[{src['url']}]({src['url']})"
                )
    else:
        st.caption("No sources returned.")

    # ── Latency ──────────────────────────────────────────────────────────────
    st.caption(f"⏱ {data['latency_ms']:.0f} ms end-to-end")
