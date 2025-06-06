import streamlit as st
import asyncio
from pathlib import Path
from src.pipeline import MainPipeline
from src.utils.settings import SETTINGS
from src.utils.cls_LLM import build_settings_dict

st.title("Medical Report Chatbot")
st.write("Upload your medical report (PDF) to get interpretations.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to the uploads directory
    upload_path = Path("data/uploads") / uploaded_file.name
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File uploaded successfully!")

    # Run the processing pipeline
    async def process_file():
        settings_dict = build_settings_dict()
        pipeline = MainPipeline(SETTINGS, settings_dict)
        
        pages = pipeline.pdf_to_images(str(upload_path))
        is_medical_image = await pipeline.check_medical_images(pages)
        
        if is_medical_image:
            interpretation = await pipeline.generate_image_interpretation(pages)
            st.write("Interpretation:")
            st.write(interpretation)
        else:
            st.write("The uploaded document is not recognized as a medical image.")

    asyncio.run(process_file())