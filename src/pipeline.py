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
import pdfplumber
from pydantic_ai import Agent, ImageUrl
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from src.utils.llm_model_factory import LLMModelFactory
from src.database.psql_schema import User, Upload, LabResult, Image, Embedding
from src.database.psql_manager import PSQL


logger = logging.getLogger(__name__)

Traceloop.init(disable_batch=True)


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
    
    class Config:
        json_encoders = {
            bytes: lambda v: f"<{len(v)} bytes>"
        }
        fields = {
            "page_image_data": {"exclude": True}
        }


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
    def __init__(self, cfg):
        self.cfg = cfg
        self.azure_model = LLMModelFactory.create_model(dict(self.cfg.azure))
        self.check_medical_images_agent = Agent(
                model=self.azure_model,
                result_type=MedicalImageCheck,
            )
        self.extract_markdown_text_agent = Agent(
                model=self.azure_model,
                result_type=str
            )

    def _pdf_to_images(self, pdf_path: str) -> List[PageMetadata]:
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

    @workflow(name="check_medical_images")
    async def _check_medical_images(self, page_metadata: list[PageMetadata]) -> Optional[bool]:
        """check if the pdf file is a medical image or a text-only problem."""
        pdf_filename = page_metadata[0].source_pdf_filename
        try:
            message_content = [self.cfg.prompts.check_medical_images_agent]
            for page in page_metadata:
                if page.image_url:
                    message_content.append(ImageUrl(url=page.image_url))

            result = await self.check_medical_images_agent.run(message_content)
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

    @workflow(name="extract_markdown_text")
    async def _extract_markdown_text(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            extracted_text = []
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Extracted text from {len(pdf.pages)} pages "
                            f"in {pdf_path}")
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text.append(f"--Page {page_num}")
                        extracted_text.append(page_text)
            full_text = "\n".join(extracted_text)
            result = await self.extract_markdown_text_agent.run(full_text)
            formatted_text = result.data
            return formatted_text.strip() if formatted_text else ""
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    async def extract_text_from_tables(self, page_meta_data: list[PageMetadata]) -> str:
        """Extract text from tables in the PDF content."""

    def _extract_medical_images(self, pdf_path: str) -> List[PageMetadata]:
        """Extract images from a PDF file and return metadata."""
        pass

    @workflow(name="generate_image_interpretation")
    async def _generate_image_interpretation(self, pages_with_images: List[PageMetadata]) -> str:
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
                agent=self.llm
            )
            logger.info(f"Results from medical image interpretation agent: {result}")
            return result.interpretation if result else ""
        except Exception as e:
            logger.error(f"Error generating image interpretation: {str(e)}")
            return ""

    async def run_single_pdf(self, pdf_path: str) -> SinglePDFResult:
        """Process a single PDF file and return the results."""
        pass

    async def _create_embeddings(self, session: AsyncSession, upload: Upload, result: SinglePDFResult) -> None:
        """Create embeddings for the uploaded content."""
        try:
            # Create embedding for main text content
            main_embedding = Embedding(
                upload_id=upload.id,
                source_table='uploads',
                source_row_id=upload.id,
                content=result.text_content,
                embedding=self._generate_embedding(result.text_content),
                chunk_index=0
            )
            session.add(main_embedding)
            
            # Create embedding for interpretation
            if result.interpretation:
                interpretation_embedding = Embedding(
                    upload_id=upload.id,
                    source_table='uploads',
                    source_row_id=upload.id,
                    content=f"Interpretation: {result.interpretation}",
                    embedding=self._generate_embedding(result.interpretation),
                    chunk_index=1
                )
                session.add(interpretation_embedding)
            
            # Create embeddings for image captions if any
            for i, caption in enumerate(result.image_captions):
                if caption:
                    caption_embedding = Embedding(
                        upload_id=upload.id,
                        source_table='uploads',
                        source_row_id=upload.id,
                        content=f"Image caption: {caption}",
                        embedding=self._generate_embedding(caption),
                        chunk_index=i + 2
                    )
                    session.add(caption_embedding)
                    
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")

    async def save_results(self, result: SinglePDFResult, user_id: int) -> None:
        """Save the results of processing a single PDF file to database."""
        try:
            async for session in PSQL.get_session():
                # 1. Create Report record
                report_type = "text" if result.pages[0].report_type == ReportType.TEXT_ONLY else "image"
                
                report = Report(
                    user_id=user_id,
                    report_type=report_type,
                    raw_text=result.text_content,
                    image=result.source_pdf_filename if report_type == "image" else None,
                )
                session.add(report)
                await session.flush()  # Get the report.id
                
                # 2. Save lab results if text report
                if report_type == "text":
                    # Parse the interpretation to extract lab tests
                    # Assuming generate_text_interpretation returns an Interpretation object
                    interpretation_result = await self.generate_text_interpretation(
                        result.text_content, result.source_pdf_filename
                    )
                    
                    if interpretation_result and interpretation_result.lab_result:
                        for lab_test in interpretation_result.lab_result:
                            # Extract numeric value and ranges if possible
                            numeric_value, lower_range, upper_range = self._parse_lab_values(
                                lab_test.result, lab_test.reference_range
                            )
                            
                            lab_result = LabResult(
                                report_id=report.id,
                                test_name=lab_test.test_name,
                                consolidated_test_name=await self._consolidate_test_name(lab_test.test_name),
                                result_value=lab_test.result,
                                unit=lab_test.unit,
                                lower_range=lower_range,
                                upper_range=upper_range,
                                interpretation=await self._determine_interpretation(
                                    lab_test.test_name, lab_test.result, lab_test.reference_range, lab_test.unit
                                ),
                                test_date=interpretation_result.datetime if hasattr(interpretation_result, 'datetime') else None,
                            )
                            session.add(lab_result)
                
                # 3. Save medical images if image report
                if report_type == "image":
                    # Group image types and descriptions
                    image_types = []
                    image_descriptions = []
                    
                    for page in result.pages:
                        if page.image_captions:
                            image_descriptions.extend(page.image_captions)
                    
                    # Extract image types from captions or use a default
                    image_types = await self._extract_image_types(image_descriptions)
                    
                    medical_image = MedicalImage(
                        report_id=report.id,
                        image_type=", ".join(image_types) if image_types else "Unknown",
                        image_descriptions="\n".join(image_descriptions),
                        image_data=result.source_pdf_filename,  # Store PDF filename
                        extracted_text=result.text_content,
                    )
                    session.add(medical_image)
                
                # 4. Create embeddings for RAG
                await self._create_embeddings(session, report, result)
                
                await session.commit()
                logger.info(f"Successfully saved results for {result.source_pdf_filename}")
                
        except Exception as e:
            logger.error(f"Error saving results to database: {str(e)}")
            raise


    async def run_batch_pdfs(self, pdf_paths: List[str]) -> List[SinglePDFResult]:
        """Process a batch of PDF files and return the results."""
        results = []
        for pdf_path in pdf_paths:
            result = await self.run_single_pdf(pdf_path)
            results.append(result)
            self.save_results(result)
        return results

