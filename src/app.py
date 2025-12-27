# src/app.py
import streamlit as st

st.title("LegalDoc QA ⚖️")
uploaded_file = st.file_path("Upload Contract (PDF)")

if uploaded_file:
    # 1. Ingest & Index
    # 2. Query Input
    query = st.text_input("Ask about governing law, termination, etc.")
    if st.button("Analyze"):
        response = query_engine.query(query)
        st.write(response.response)
        with st.expander("View Source Excerpts"):
            for node in response.source_nodes:
                st.info(f"Score: {node.score:.2f} \n\n {node.text}")
