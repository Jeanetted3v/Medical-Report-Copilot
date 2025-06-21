"""To run:
PYTHONPATH=. streamlit run src/app.py
"""
from hydra import compose, initialize
import streamlit as st
import asyncio
import tempfile
from src.pipeline import MainPipeline
from omegaconf import OmegaConf

st.set_page_config(page_title="Medical Report Assistant", layout="wide")

with initialize(config_path="../config", version_base=None):
    cfg = compose(config_name="config")

pipeline = MainPipeline(cfg)

st.title("📄 Medical Report Analyzer")
st.markdown("Upload one or more PDF reports. The assistant will extract relevant information and provide interpretations.")

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Analyze Reports"):
        with st.spinner("Processing reports..."):
            # Save uploaded files to temporary files
            temp_paths = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.read())
                    temp_paths.append(tmp_file.name)

            # Run processing loop
            results = asyncio.run(pipeline.run_batch_pdfs(temp_paths))

            # Display results
            for result in results:
                st.subheader(f"📄 {result.source_pdf_filename}")
                st.markdown(f"**Report Type**: {result.report_type.value}")
                st.markdown(f"**Interpretation**: {result.overall_interpretation}")
                
                if result.raw_text:
                    with st.expander("Raw Extracted Text"):
                        st.text(result.raw_text)

                if result.lab_results:
                    with st.expander("Lab Results"):
                        for lab in result.lab_results:
                            st.markdown(f"- `{lab.test_name}`: **{lab.result_value}** {lab.unit or ''} ({lab.interpretation})")

                if result.medical_images:
                    with st.expander("Medical Images"):
                        for i, img in enumerate(result.medical_images):
                            st.markdown(f"**Image Type**: {img.image_type}")
                            st.markdown(f"**Description**: {img.image_descriptions}")
                            st.markdown(f"**Interpretation**: {img.image_interpretation, 'N/A'}")

