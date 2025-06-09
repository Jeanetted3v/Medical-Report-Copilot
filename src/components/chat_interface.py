from pathlib import Path
import streamlit as st
import asyncio
from src.pipeline import MainPipeline
from src.utils.cls_LLM import build_settings_dict


class ChatInterface:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = MainPipeline(cfg, build_settings_dict())
    
    def upload_file(self):
        uploaded_file = st.file_uploader("Upload your medical report (PDF)", type=["pdf"])
        return uploaded_file

    async def process_file(self, uploaded_file):
        if uploaded_file is not None:
            pdf_path = f"./data/uploads/{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            pages = self.pipeline.pdf_to_images(pdf_path)
            is_medical_image = await self.pipeline.check_medical_images(pages)
            interpretation = ""
            if is_medical_image:
                interpretation = await self.pipeline.generate_image_interpretation(pages)
            return interpretation
        return "No file uploaded."

    def display_chat_interface(self):
        st.title("Medical Report Chatbot")
        uploaded_file = self.upload_file()
        
        if st.button("Generate Interpretation"):
            with st.spinner("Processing..."):
                interpretation = asyncio.run(self.process_file(uploaded_file))
                st.success("Interpretation generated!")
                st.write(interpretation)