"""To run:
poetry run python -m src.pipeline
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from enum import Enum
import base64
import pymupdf
from pydantic import BaseModel
from src.utils.cls_LLM import build_llm_client


logger = logging.getLogger(__name__)


class ReportType(Enum):
    TEXT_ONLY = "text_only"
    MEDICAL_IMAGE = "medical_image"


class PageMetadata(BaseModel):
    source_pdf_filename: str
    page_number: int
    image_captions: Optional[List[str]] = []
    report_type: Optional[ReportType] = None
    page_image_data: bytes

    @property
    def image_url(self) -> str:
        """Generate data URL from image data on demand"""
        if self.page_image_data:
            img_base64 = base64.b64encode(self.page_image_data).decode()
            return f"data:image/png;base64,{img_base64}"
        return None


class SinglePDFResult(BaseModel):
    source_pdf_filename: str
    image_captions: List[str]
    text_content: str
    pages: List[PageMetadata]
    interpretation: str


class MedicalImageCheck(BaseModel):
    """Response model for agent checking if a PDF is a medical image problem"""
    is_medical_image: bool
    report_type: ReportType
    image_captions: List[str] = []  # Captions for medical images if applicable
    image_types: List[str] = []  # Types of images found (e.g., X-ray, MRI)
    # confidence_score: Optional[float] = None  # Confidence score for the classification


class LabTest(BaseModel):
    test_name: str
    result: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None


class Interpretation(BaseModel):
    datetime: datetime
    lab_result: Optional[List[LabTest]]
    interpretation: str
    reasoning_interpretation: Optional[str] = None


class ImageReportInterpretation(BaseModel):
    image_types: List[str]
    image_captions: List[str]
    interpretation: str
    reasoning: Optional[str] = None
    report_information: Optional[str] = None
    recommendations: Optional[str] = None


class MainPipeline:
    def __init__(self, cfg, settings_dict: dict):
        self.cfg = cfg
        self.medical_image_agent = build_llm_client(settings_dict, "")
    
    async def _send_images_with_prompt(
        self, images: List[str], prompt: str, response_model, agent
    ):
        """Helper method to send images with prompt - hides the complexity"""
        image_contents = [{"type": "image_url", "image_url": {"url": img}} for img in images]
        message_content = [{"type": "text", "text": prompt}] + image_contents
        
        # Temporarily update agent messages
        original_messages = agent.messages
        agent.messages = [{"role": "user", "content": message_content}]
        
        try:
            result = await agent.async_structured_completion(
                response_model=response_model, temperature=0.1
            )
            return result
        finally:
            # Restore original messages
            agent.messages = original_messages

    def pdf_to_images(self, pdf_path: str) -> List[PageMetadata]:
        """Convert PDF files to multiple images, one per page."""
        pdf_filename = Path(pdf_path).name
        pages_metadata = []

        # Logic to convert PDF to images and populate pages_metadata
        try:
            doc = pymupdf.open(pdf_path)
            logger.info(f"Converting {len(doc)} pages from {pdf_filename} to images")
            
            for page_number in range(len(doc)):
                page = doc[page_number]
                # Higher DPI (300 DPI) - better quality
                mat = pymupdf.Matrix(300/72, 300/72)
                pix = page.get_pixmap(matrix=mat)
                image_data = pix.tobytes("png")
                page_metadata = PageMetadata(
                    source_pdf_filename=pdf_filename,
                    page_number=page_number + 1,
                    image_captions=[],
                    report_type=None,
                    page_image_data=image_data
                )
                pages_metadata.append(page_metadata)
                logger.info(f"Converted page {page_number + 1} to memory "
                            f"({len(image_data)} bytes)")
            doc.close()  # PyMuPDF keeps the PDF file open in memory. Without closing, might hit OS limits on open files
            return pages_metadata
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise e
                
    async def check_medical_images(self, page_metadata: list[PageMetadata]) -> Optional[bool]:
        """check if the pdf file is a medical image or a text-only problem."""
        pdf_filename = page_metadata[0].source_pdf_filename
        try:
            image_data_urls = [page.image_url for page in page_metadata if page.image_url]
            result = await self._send_images_with_prompt(
                images=image_data_urls,
                prompt=self.cfg.prompts.medical_image_agent,
                response_model=MedicalImageCheck,
                agent=self.medical_image_agent
            )
            logger.info(f"Result of Checking Medical Images: {result}")
            is_medical_image = result.is_medical_image
            # Update ALL pages with the same result (whole PDF classification)
            for page in page_metadata:
                page.report_type = result.report_type
                page.image_captions = result.image_captions if is_medical_image else []

            logger.info(f"Medical image analysis for {pdf_filename}: {is_medical_image}")
            if result.image_types:  # Fix: check image_types not image_captions
                logger.info(f"Found image types: {', '.join(result.image_types)}")
            return result.is_medical_image
        except Exception as e:
            logger.error(f"Error checking for medical images: {str(e)}")
            # Default to treating as having images for safety
            return

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        pass

    def extract_images(self, pdf_path: str) -> List[PageMetadata]:
        """Extract images from a PDF file and return metadata."""
        pass

    async def generate_image_interpretation(self, pages_with_images: List[PageMetadata]) -> str:
        """For medical image problems, generate an interpretation of the report."""
        try:
            pdf_filename = pages_with_images[0].source_pdf_filename
            image_data_urls = [page.image_url for page in pages_with_images if page.image_url]
            if not image_data_urls:
                logger.warning(f"No valid image data found in {pdf_filename}")
                return ""
            all_captions = []
            for page in pages_with_images:
                if page.image_captions:
                    all_captions.extend(page.image_captions)

            if all_captions:
                caption_context = f"\n\nImage Captions identified: {', '.join(all_captions)}"
                self.cfg.prompts.medical_image_interpretation_agent += caption_context

            result = await self._send_images_with_prompt(
                images=image_data_urls,
                prompt=self.cfg.prompts.medical_image_interpretation_agent,
                response_model=ImageReportInterpretation,
                agent=self.medical_image_agent
            )
            logger.info(f"Results from medical image interpretation agent: {result}")
            return result.interpretation if result else ""
        except Exception as e:
            logger.error(f"Error generating image interpretation: {str(e)}")
            return ""

    async def generate_text_interpretation(self, text_content: str, pdf_name: str) -> str:
        """For text-only problems, generate an interpretation of the report.
        CK
        """
        pass

    async def run_single_pdf(self, pdf_path: str) -> SinglePDFResult:
        """Process a single PDF file and return the results."""
        pass

    def save_results(self, result: SinglePDFResult) -> None:
        """Save the results of processing a single PDF file."""
        pass

    async def run_batch_pdfs(self, pdf_paths: List[str]) -> List[SinglePDFResult]:
        """Process a batch of PDF files and return the results."""
        results = []
        for pdf_path in pdf_paths:
            result = await self.run_single_pdf(pdf_path)
            results.append(result)
            self.save_results(result)
        return results

