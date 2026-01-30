"""
PDF extractor module for the Multi-Modal RAG system.
Extracts text, images, tables, and charts from PDF documents.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import fitz  # PyMuPDF
import pdfplumber
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
import io

from src.utils.logger import get_logger
from src.utils.file_utils import FileUtils, ensure_dir
from src.utils.image_utils import ImageUtils
from src.utils.config_loader import get_config_value

logger = get_logger(__name__)


class PDFExtractor:
    """
    Extract content from PDF documents including text, images, tables, and charts.
    """

    def __init__(
        self,
        dpi: int = 300,
        extract_images: bool = True,
        extract_tables: bool = True,
        min_image_size: int = 100,
        output_dir: Optional[str] = None
    ):
        """
        Initialize PDF extractor.

        Args:
            dpi: DPI for image extraction
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            min_image_size: Minimum image size in pixels
            output_dir: Directory to save extracted content
        """
        self.dpi = dpi
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.min_image_size = min_image_size

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(get_config_value("paths.processed_data", "data/processed"))

        ensure_dir(self.output_dir)

        logger.info(f"PDFExtractor initialized with DPI={dpi}, extract_images={extract_images}")

    def extract(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract all content from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing extracted content
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting content from: {pdf_path}")

        # Initialize result structure
        result = {
            "metadata": {
                "source": str(pdf_path),
                "filename": pdf_path.name,
                "pages": 0,
                "total_images": 0,
                "total_tables": 0
            },
            "text": {},
            "images": {},
            "tables": {},
            "charts": {},
            "pages": []
        }

        try:
            # Extract with PyMuPDF (fast for text and images)
            fitz_result = self._extract_with_fitz(pdf_path)
            result["text"].update(fitz_result["text"])
            result["images"].update(fitz_result["images"])
            result["metadata"]["pages"] = fitz_result["pages"]

            # Extract tables with pdfplumber (better for tables)
            if self.extract_tables:
                table_result = self._extract_tables(pdf_path)
                result["tables"].update(table_result["tables"])
                result["metadata"]["total_tables"] = table_result["total_tables"]

            # Extract charts (if any)
            chart_result = self._extract_charts(pdf_path)
            result["charts"].update(chart_result["charts"])

            # Update metadata
            result["metadata"]["total_images"] = len(result["images"])

            # Create page summaries
            result["pages"] = self._create_page_summaries(result)

            logger.success(f"Successfully extracted content from {pdf_path.name}")
            logger.info(f"Pages: {result['metadata']['pages']}, Images: {result['metadata']['total_images']}, Tables: {result['metadata']['total_tables']}")

        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            raise

        return result

    def _extract_with_fitz(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text and images using PyMuPDF.

        Args:
            pdf_path: Path to PDF

        Returns:
            Extracted text and images
        """
        result = {
            "text": {},
            "images": {},
            "pages": 0
        }

        doc = fitz.open(pdf_path)

        try:
            result["pages"] = len(doc)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text
                text = page.get_text()
                if text.strip():
                    result["text"][page_num + 1] = {
                        "content": text.strip(),
                        "bbox": None,
                        "font_info": None
                    }

                # Extract images
                if self.extract_images:
                    page_images = self._extract_images_from_page(page, page_num + 1)
                    if page_images:
                        result["images"][page_num + 1] = page_images

        finally:
            doc.close()

        return result

    def _extract_images_from_page(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract images from a single page.

        Args:
            page: PyMuPDF page object
            page_num: Page number

        Returns:
            List of image dictionaries
        """
        images = []

        # Get images from page
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))

                # Check minimum size
                if image.width < self.min_image_size or image.height < self.min_image_size:
                    continue

                # Save image
                image_filename = f"page{page_num}_img{img_index + 1}.{image_ext}"
                image_path = self.output_dir / "images" / image_filename
                ensure_dir(image_path.parent)

                ImageUtils.save_image(image, image_path)

                # Store image info
                image_info = {
                    "filename": image_filename,
                    "path": str(image_path),
                    "bbox": img[1:5] if len(img) > 4 else None,
                    "width": image.width,
                    "height": image.height,
                    "format": image_ext,
                    "size_bytes": len(image_bytes)
                }

                images.append(image_info)

            except Exception as e:
                logger.warning(f"Error extracting image {img_index} from page {page_num}: {e}")
                continue

        return images

    def _extract_tables(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract tables using pdfplumber.

        Args:
            pdf_path: Path to PDF

        Returns:
            Extracted tables
        """
        result = {
            "tables": {},
            "total_tables": 0
        }

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract tables
                    tables = page.extract_tables()

                    if tables:
                        page_tables = []
                        for table_idx, table in enumerate(tables):
                            # Convert to DataFrame for easier handling
                            df = pd.DataFrame(table)

                            # Clean table (remove empty rows/columns)
                            df = df.dropna(how='all').dropna(axis=1, how='all')

                            if not df.empty:
                                table_info = {
                                    "data": df.values.tolist(),
                                    "headers": df.iloc[0].tolist() if len(df) > 0 else [],
                                    "rows": len(df),
                                    "columns": len(df.columns),
                                    "bbox": None  # Could be extracted if needed
                                }

                                page_tables.append(table_info)

                        if page_tables:
                            result["tables"][page_num] = page_tables
                            result["total_tables"] += len(page_tables)

                except Exception as e:
                    logger.warning(f"Error extracting tables from page {page_num}: {e}")
                    continue

        return result

    def _extract_charts(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract charts and diagrams (placeholder for future implementation).

        Args:
            pdf_path: Path to PDF

        Returns:
            Extracted charts
        """
        # This is a placeholder for chart extraction
        # Could use computer vision models to detect and extract charts
        result = {
            "charts": {}
        }

        logger.info("Chart extraction not yet implemented - placeholder")
        return result

    def _create_page_summaries(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create summaries for each page.

        Args:
            result: Full extraction result

        Returns:
            List of page summaries
        """
        pages = []

        for page_num in range(1, result["metadata"]["pages"] + 1):
            page_info = {
                "page_number": page_num,
                "has_text": page_num in result["text"],
                "text_length": len(result["text"].get(page_num, {}).get("content", "")),
                "image_count": len(result["images"].get(page_num, [])),
                "table_count": len(result["tables"].get(page_num, [])),
                "chart_count": len(result["charts"].get(page_num, []))
            }
            pages.append(page_info)

        return pages

    def extract_page_as_image(self, pdf_path: Union[str, Path], page_num: int) -> Optional[Image.Image]:
        """
        Extract a specific page as an image.

        Args:
            pdf_path: Path to PDF
            page_num: Page number (1-based)

        Returns:
            PIL Image of the page
        """
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=page_num,
                last_page=page_num
            )

            if images:
                return images[0]

        except Exception as e:
            logger.error(f"Error converting page {page_num} to image: {e}")

        return None

    def get_document_info(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic document information.

        Args:
            pdf_path: Path to PDF

        Returns:
            Document metadata
        """
        try:
            with fitz.open(pdf_path) as doc:
                info = doc.metadata
                return {
                    "title": info.get("title", ""),
                    "author": info.get("author", ""),
                    "subject": info.get("subject", ""),
                    "creator": info.get("creator", ""),
                    "producer": info.get("producer", ""),
                    "creation_date": info.get("creationDate", ""),
                    "modification_date": info.get("modDate", ""),
                    "pages": len(doc)
                }
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return {}

    def save_extracted_content(self, result: Dict[str, Any], output_dir: Optional[str] = None):
        """
        Save extracted content to files.

        Args:
            result: Extraction result
            output_dir: Output directory
        """
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = self.output_dir

        ensure_dir(output_dir)

        # Save text
        text_dir = output_dir / "text"
        ensure_dir(text_dir)

        for page_num, text_data in result["text"].items():
            text_file = text_dir / f"page_{page_num}.txt"
            FileUtils.write_text(text_data["content"], text_file)

        # Save tables as CSV
        tables_dir = output_dir / "tables"
        ensure_dir(tables_dir)

        for page_num, page_tables in result["tables"].items():
            for table_idx, table_data in enumerate(page_tables):
                df = pd.DataFrame(table_data["data"])
                csv_file = tables_dir / f"page_{page_num}_table_{table_idx + 1}.csv"
                df.to_csv(csv_file, index=False)

        # Save metadata
        metadata_file = output_dir / "metadata.json"
        FileUtils.save_json(result["metadata"], metadata_file)

        logger.info(f"Saved extracted content to: {output_dir}")


# Convenience functions
def extract_pdf_content(pdf_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to extract PDF content.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted content
    """
    extractor = PDFExtractor()
    return extractor.extract(pdf_path)


def extract_pdf_text(pdf_path: Union[str, Path]) -> str:
    """
    Extract only text from PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Combined text from all pages
    """
    result = extract_pdf_content(pdf_path)
    texts = []

    for page_data in result["text"].values():
        texts.append(page_data["content"])

    return "\n\n".join(texts)
