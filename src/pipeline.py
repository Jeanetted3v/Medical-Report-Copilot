"""To run:
poetry run python -m src.pipeline
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import io
import json
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
from src.database.psql_schema import User, Report, LabResult, MedicalImage, Embeddings
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

class MedicalImageLocation(BaseModel):
    image_type: str
    caption: str
    bounding_box: List[float]
    page_number: int

class MedicalImageCheck(BaseModel):
    """Response model for agent checking if a PDF is a medical image problem"""
    is_medical_image: bool
    report_type: ReportType
    image_captions: List[str] = []  # Captions for medical images if applicable
    image_types: List[str] = []  # Types of images found (e.g., X-ray, MRI)
    medical_image_locations: List[MedicalImageLocation] = [] 



class MedicalImage(BaseModel):
    image_type: str  # e.g., 'X-ray', 'MRI', 'CT', 'Graph'
    image_descriptions: str
    image_interpretation: Optional[str] = None

class LabTest(BaseModel):
    test_name: str
    consolidated_test_name: Optional[str] = None  # LLM suggested consolidated name
    result_value: str
    doc_comment: Optional[str] = None
    unit: Optional[str] = None
    lower_range: Optional[float] = None  # LLM suggested
    upper_range: Optional[float] = None  # LLM suggested
    interpretation: str  # 'normal', 'high', 'low'
    test_date: Optional[datetime] = None


class Interpretation(BaseModel):
    datetime: datetime
    lab_result: Optional[List[LabTest]]
    interpretation: str
    reasoning_interpretation: Optional[str] = None


class SinglePDFResult(BaseModel):
    source_pdf_filename: str
    report_type: ReportType
    raw_text: Optional[str] = None  # Raw text content extracted from PDF
    overall_interpretation: Optional[str] = None  # LLM generated interpretation for the entire report
    pages: List[PageMetadata]
    lab_results: Optional[List[LabTest]] = []
    medical_images: Optional[List[MedicalImage]] = []


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
        self.image_interpretor_agent = Agent(
                model=self.azure_model,
                result_type=MedicalImage,
            )
        self.lab_result_agent = Agent(
                model=self.azure_model,
                result_type=LabTest,
            )
        self.report_interpretation_agent = Agent(
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
    async def _check_medical_images(self, page_metadata: list[PageMetadata]) -> MedicalImageCheck:
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
            return result.data
        except Exception as e:
            logger.error(f"Error checking for medical images: {str(e)}")
            return MedicalImageCheck(
                is_medical_image=False,  # Default to text-only on error
                report_type=ReportType.TEXT_ONLY,
                image_captions=[],
                image_types=[],
                medical_image_locations=[]
            )

    def _extract_text_and_tables(self, pdf_path: str) -> str:
        """Extract text and tables from PDF, preparing them for LLM processing."""
        extracted_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"Processing {len(pdf.pages)} pages from {pdf_path}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_content = []
                page_content.append(f"--- PAGE {page_num} ---")
                tables = page.extract_tables()
                table_areas = []
                
                if tables:
                    page_content.append("\n[TABLES ON THIS PAGE:]")
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            page_content.append(f"\nTable {table_idx + 1}:")
                            # Format table for LLM
                            for row_idx, row in enumerate(table):
                                if row:  # Skip empty rows
                                    cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                    page_content.append(" | ".join(cleaned_row))
                            # Get table bounding box to potentially exclude from text
                            try:
                                table_obj = page.find_tables()[table_idx]
                                table_areas.append(table_obj.bbox)
                            except (IndexError, AttributeError):
                                pass
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    page_content.append("\n[TEXT CONTENT:]")
                    page_content.append(page_text)
                
                if len(page_content) > 1:  # More than just the page header
                    extracted_content.extend(page_content)
                    extracted_content.append("")  # Add spacing between pages
        
        return "\n".join(extracted_content)

    @workflow(name="extract_markdown_text")
    async def _extract_markdown_text(self, pdf_path: str) -> str:
        """Extract text content from a PDF file and convert to structured markdown."""
        try:
            extracted_content = self._extract_text_and_tables(pdf_path)
            if not extracted_content:
                logger.warning(f"No content extracted from {pdf_path}")
                return ""

            result = await self.extract_markdown_text_agent.run(extracted_content)
            formatted_text = result.data if hasattr(result, 'data') else str(result)
            return formatted_text.strip() if formatted_text else ""
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    async def _extract_medical_images(
        self,
        pdf_path: str,
        medical_check_result: MedicalImageCheck
    ) -> List[Dict[str, Any]]:
        """Extract medical images from PDF and return structured data for PSQL"""
        try:
            medical_images = []

            all_locations = medical_check_result.medical_image_locations
            if not all_locations:
                logger.info("No medical image locations found in PDF")
                return medical_images
            
            # Extract images based on locations
            with pdfplumber.open(pdf_path) as pdf:
                for location in all_locations:
                    try:
                        page = pdf.pages[location.page_number - 1]  # Page numbers are 1-based in metadata
                        page_width = page.width
                        page_height = page.height
                        # Convert percentage coordinates to absolute coordinates
                        x1_pct, y1_pct, x2_pct, y2_pct = location.bounding_box
                        
                        bbox = (
                            x1_pct * page_width,
                            y1_pct * page_height,
                            x2_pct * page_width,
                            y2_pct * page_height
                        )
                        
                        # Extract the medical image region
                        cropped_page = page.crop(bbox)
                        region_image = cropped_page.to_image()
                        
                        # Convert to bytes
                        region_bytes = io.BytesIO()
                        region_image.save(region_bytes, format='PNG')
                        
                        # Optional: Get detailed interpretation of the extracted image
                        region_base64 = base64.b64encode(region_bytes.getvalue()).decode()
                        region_url = f"data:image/png;base64,{region_base64}"
                        
                        result = await self.image_interpretor_agent.run([
                            self.cfg.prompts.analyze_medical_image,
                            ImageUrl(url=region_url)
                        ])
                        # Handle result based on result_type (should be MedicalImage)
                        if hasattr(result, 'image_type'):
                            # Direct access if result is MedicalImage object
                            image_type = result.image_type
                            image_descriptions = result.image_descriptions
                            image_interpretation = result.image_interpretation
                        else:
                            # Fallback if result has .data attribute
                            image_type = result.data.image_type if hasattr(result, 'data') else location.image_type
                            image_descriptions = result.data.image_descriptions if hasattr(result, 'data') else location.caption
                            image_interpretation = result.data.image_interpretation if hasattr(result, 'data') else None
                        
                        medical_images.append({
                            'page_number': location.page_number,
                            'image_type': image_type,
                            'image_descriptions': image_descriptions,
                            'image_data': region_bytes.getvalue(),
                            'image_interpretation': image_interpretation
                        })
                        logger.info(f"Extracted {location.image_type} from page {location.page_number}")
                    except Exception as e:
                        logger.error(f"Error extracting medical image from page {location.page_number}: {e}")
                        continue
            logger.info(f"Successfully extracted {len(medical_images)} medical images")
            return medical_images
        except Exception as e:
            logger.error(f"Error in medical image extraction: {e}")
            return []

    async def _extract_lab_results(self, text: str) -> List[LabTest]:
        """Process text to extract lab results"""
        formatted_prompt = self.cfg.prompts.lab_results_agent.format(text=text)
        result = await self.lab_result_agent.run(formatted_prompt)
        lab_test = json.loads(result.data)
        logger.info(f"Extracted lab results: {lab_test}")
        return [LabTest(**item) for item in lab_test.get('data', [])]
            
    @workflow(name="generate_overall_interpretation")
    async def _generate_overall_interpretation(
        self, pages_metadata: List[PageMetadata]
    ) -> str:
        """Generate overall interpretation by letting LLM directly read the PDF pages"""
        try:
            message_content = [self.cfg.prompts.report_interpretation_agent]

            for page in pages_metadata:
                if page.image_url:
                    message_content.append(ImageUrl(url=page.image_url))

            result = await self.extract_markdown_text_agent.run(message_content)
            interpretation = result.data if hasattr(result, 'data') else str(result)
            return interpretation.strip() if interpretation else "No significant findings noted."
            
        except Exception as e:
            logger.error(f"Error generating overall interpretation: {e}")
            return "Error generating interpretation."
        
    async def run_single_pdf(self, pdf_path: str) -> SinglePDFResult:
        """Process a single PDF file and return structured results."""
        pdf_filename = Path(pdf_path).name
        logger.info(f"Starting processing of PDF: {pdf_filename}")
        try:
            logger.info("Step 1: Converting PDF to images...")
            pages_metadata = self._pdf_to_images(pdf_path)
            if not pages_metadata:
                raise ValueError(f"Failed to convert PDF to images: "
                                 f"{pdf_filename}")
            
            # Step 2: Check if PDF contains medical images
            logger.info("Step 2: Checking for medical images...")
            is_medical_image = await self._check_medical_images(pages_metadata)
            
            # Step 3: Extract text content
            logger.info("Step 3: Extracting text content...")
            raw_text = await self._extract_markdown_text(pdf_path)
            
            # Step 4: Process based on content type
            lab_results = []
            medical_images = []

            if raw_text.strip():
                logger.info("Step 4: Processing lab results from text...")
                lab_results = await self._extract_lab_results(raw_text)
                logger.info(f"Extracted {len(lab_results)} lab results")
            
            # Step 5: Extract medical images if present
            if is_medical_image:
                logger.info("Step 4a: Processing medical images...")
                medical_images = await self._extract_medical_images(pages_metadata, pdf_path)
                logger.info(f"Found {len(medical_images)} medical images")

            # Step 6: Generate overall interpretation                   
            logger.info("Step 6: Generating overall interpretation...")
            overall_interpretation = await self._generate_overall_interpretation(pages_metadata)
            
            # Step 7: Create final result with all extracted data
            single_pdf_result = SinglePDFResult(
                source_pdf_filename=pdf_filename,
                report_type="medical_image" if is_medical_image else "text",
                raw_text=raw_text,
                overall_interpretation=overall_interpretation,
                pages=pages_metadata,
                lab_results=lab_results,
                medical_images=medical_images
            )
            return single_pdf_result
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_filename}: {e}")
            # Return a minimal result with error information
            return SinglePDFResult(
                source_pdf_filename=pdf_filename,
                report_type="text",
                raw_text="",
                overall_interpretation=f"Error processing PDF: {str(e)}",
                pages=[],
                lab_results=[],
                medical_images=[]
            )

    async def _create_embeddings(self):
        pass

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

