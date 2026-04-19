"""
docuseek/frontend/app.py
----------------
Minimal Streamlit UI for interacting with DocuSeek.
"""

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="DocuSeek", page_icon="🔍")
st.title("🔍 DocuSeek")
st.caption("RAG over ML framework documentation")

question = st.text_input("Ask a question:", placeholder="How do I load a dataset from the hub?")

if st.button("Search", disabled=not question):
    with st.spinner("Retrieving and generating..."):
        try:
            response = httpx.post(f"{API_URL}/query", json={"question": question}, timeout=60.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            st.error(f"API error: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Sources")
    for src in data["sources"]:
        st.markdown(f"- [{src['title']}]({src['url']}) · `{src['source']}`")
