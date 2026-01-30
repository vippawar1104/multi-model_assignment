"""
Image extractor module for the Multi-Modal RAG system.
Processes images with OCR, caption generation, and feature extraction.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image
import easyocr
import pytesseract
from transformers import pipeline
import torch

from src.utils.logger import get_logger
from src.utils.file_utils import FileUtils, ensure_dir
from src.utils.image_utils import ImageUtils
from src.utils.config_loader import get_config_value

logger = get_logger(__name__)


class ImageExtractor:
    """
    Extract content from images including OCR text and captions.
    """

    def __init__(
        self,
        generate_captions: bool = True,
        perform_ocr: bool = True,
        ocr_engine: str = "easyocr",  # easyocr or tesseract
        caption_model: str = "Salesforce/blip-image-captioning-base",
        device: str = "cpu"
    ):
        """
        Initialize image extractor.

        Args:
            generate_captions: Whether to generate image captions
            perform_ocr: Whether to perform OCR
            ocr_engine: OCR engine to use ('easyocr' or 'tesseract')
            caption_model: HuggingFace model for caption generation
            device: Device for model inference
        """
        self.generate_captions = generate_captions
        self.perform_ocr = perform_ocr
        self.ocr_engine = ocr_engine
        self.caption_model = caption_model
        self.device = device

        # Initialize OCR readers
        self._easyocr_reader = None
        self._caption_pipeline = None

        # Initialize components
        self._init_ocr()
        if self.generate_captions:
            self._init_caption_model()

        logger.info(f"ImageExtractor initialized with OCR={ocr_engine}, captions={generate_captions}")

    def _init_ocr(self):
        """Initialize OCR engines."""
        if self.perform_ocr:
            if self.ocr_engine == "easyocr":
                try:
                    self._easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                    logger.info("EasyOCR initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize EasyOCR: {e}")
                    self._easyocr_reader = None

            elif self.ocr_engine == "tesseract":
                try:
                    # Configure tesseract
                    pytesseract.get_tesseract_version()
                    logger.info("Tesseract initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Tesseract: {e}")

    def _init_caption_model(self):
        """Initialize caption generation model."""
        try:
            self._caption_pipeline = pipeline(
                "image-to-text",
                model=self.caption_model,
                device=self.device
            )
            logger.info(f"Caption model initialized: {self.caption_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize caption model: {e}")
            self._caption_pipeline = None

    def extract(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract content from an image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary containing extracted content
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Extracting content from image: {image_path}")

        # Load image
        image = ImageUtils.load_image(image_path)

        # Initialize result
        result = {
            "metadata": {
                "source": str(image_path),
                "filename": image_path.name,
                "width": image.width,
                "height": image.height,
                "format": image.format or image_path.suffix[1:].upper(),
                "size_bytes": image_path.stat().st_size
            },
            "ocr_text": "",
            "caption": "",
            "text_regions": [],
            "confidence": 0.0
        }

        try:
            # Perform OCR
            if self.perform_ocr:
                ocr_result = self._perform_ocr(image)
                result.update(ocr_result)

            # Generate caption
            if self.generate_captions and self._caption_pipeline:
                caption_result = self._generate_caption(image)
                result.update(caption_result)

            # Calculate overall confidence
            result["confidence"] = self._calculate_confidence(result)

            logger.success(f"Successfully extracted content from {image_path.name}")

        except Exception as e:
            logger.error(f"Error extracting image {image_path}: {e}")
            raise

        return result

    def _perform_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """
        Perform OCR on image.

        Args:
            image: PIL Image

        Returns:
            OCR results
        """
        result = {
            "ocr_text": "",
            "text_regions": [],
            "ocr_confidence": 0.0
        }

        try:
            if self.ocr_engine == "easyocr" and self._easyocr_reader:
                # EasyOCR
                results = self._easyocr_reader.readtext(
                    ImageUtils.image_to_numpy(image),
                    detail=1,  # Get bounding boxes
                    paragraph=False
                )

                text_parts = []
                total_confidence = 0.0
                region_count = 0

                for detection in results:
                    bbox, text, confidence = detection

                    if confidence > 0.1:  # Filter low confidence
                        text_parts.append(text)
                        total_confidence += confidence
                        region_count += 1

                        # Store text region
                        region = {
                            "text": text,
                            "bbox": [int(coord) for coord in bbox.flatten()],
                            "confidence": float(confidence)
                        }
                        result["text_regions"].append(region)

                result["ocr_text"] = " ".join(text_parts)
                result["ocr_confidence"] = total_confidence / max(region_count, 1)

            elif self.ocr_engine == "tesseract":
                # Tesseract
                text = pytesseract.image_to_string(image)
                confidence_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

                result["ocr_text"] = text.strip()

                # Extract confidence and regions
                confidences = []
                for i, conf in enumerate(confidence_data['conf']):
                    if int(conf) > 0:  # Valid confidence
                        confidences.append(int(conf))

                        # Add text region
                        region = {
                            "text": confidence_data['text'][i],
                            "bbox": [
                                confidence_data['left'][i],
                                confidence_data['top'][i],
                                confidence_data['left'][i] + confidence_data['width'][i],
                                confidence_data['top'][i] + confidence_data['height'][i]
                            ],
                            "confidence": int(conf)
                        }
                        result["text_regions"].append(region)

                result["ocr_confidence"] = sum(confidences) / max(len(confidences), 1) / 100.0

        except Exception as e:
            logger.warning(f"OCR failed: {e}")

        return result

    def _generate_caption(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate caption for image.

        Args:
            image: PIL Image

        Returns:
            Caption results
        """
        result = {
            "caption": "",
            "caption_confidence": 0.0
        }

        try:
            if self._caption_pipeline:
                # Generate caption
                outputs = self._caption_pipeline(image)

                if outputs and len(outputs) > 0:
                    result["caption"] = outputs[0].get("generated_text", "")
                    # Note: BLIP doesn't provide confidence scores
                    result["caption_confidence"] = 0.8  # Default confidence

        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")

        return result

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate overall extraction confidence.

        Args:
            result: Extraction results

        Returns:
            Overall confidence score
        """
        confidences = []

        if result.get("ocr_confidence", 0) > 0:
            confidences.append(result["ocr_confidence"])

        if result.get("caption_confidence", 0) > 0:
            confidences.append(result["caption_confidence"])

        if confidences:
            return sum(confidences) / len(confidences)

        return 0.0

    def extract_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Extract content from multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            List of extraction results
        """
        results = []

        for image_path in image_paths:
            try:
                result = self.extract(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract {image_path}: {e}")
                # Add error result
                results.append({
                    "metadata": {"source": str(image_path), "error": str(e)},
                    "ocr_text": "",
                    "caption": "",
                    "confidence": 0.0
                })

        logger.info(f"Batch extracted {len(results)} images")
        return results

    def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.

        Args:
            image: PIL Image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # Resize if too small
        min_size = 300
        if image.width < min_size or image.height < min_size:
            scale = max(min_size / image.width, min_size / image.height)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Enhance contrast (optional)
        # image = ImageEnhance.Contrast(image).enhance(2.0)

        return image

    def detect_text_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect text regions in image without full OCR.

        Args:
            image: PIL Image

        Returns:
            List of text region bounding boxes
        """
        regions = []

        try:
            if self.ocr_engine == "easyocr" and self._easyocr_reader:
                results = self._easyocr_reader.readtext(
                    ImageUtils.image_to_numpy(image),
                    detail=1
                )

                for detection in results:
                    bbox, text, confidence = detection
                    if confidence > 0.3:  # Higher threshold for detection
                        regions.append({
                            "bbox": [int(coord) for coord in bbox.flatten()],
                            "confidence": float(confidence)
                        })

        except Exception as e:
            logger.warning(f"Text region detection failed: {e}")

        return regions

    def save_results(self, results: List[Dict[str, Any]], output_dir: Union[str, Path]):
        """
        Save extraction results to files.

        Args:
            results: List of extraction results
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        # Save individual results
        for result in results:
            filename = Path(result["metadata"]["filename"]).stem
            result_file = output_dir / f"{filename}_extracted.json"
            FileUtils.save_json(result, result_file)

        # Save summary
        summary = {
            "total_images": len(results),
            "successful_extractions": len([r for r in results if r.get("confidence", 0) > 0]),
            "average_confidence": sum(r.get("confidence", 0) for r in results) / max(len(results), 1),
            "results": [
                {
                    "filename": r["metadata"]["filename"],
                    "confidence": r.get("confidence", 0),
                    "has_text": bool(r.get("ocr_text")),
                    "has_caption": bool(r.get("caption"))
                }
                for r in results
            ]
        }

        summary_file = output_dir / "extraction_summary.json"
        FileUtils.save_json(summary, summary_file)

        logger.info(f"Saved extraction results to: {output_dir}")


# Convenience functions
def extract_image_content(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to extract image content.

    Args:
        image_path: Path to image file

    Returns:
        Extracted content
    """
    extractor = ImageExtractor()
    return extractor.extract(image_path)


def extract_image_text(image_path: Union[str, Path]) -> str:
    """
    Extract only text from image.

    Args:
        image_path: Path to image file

    Returns:
        Extracted text
    """
    result = extract_image_content(image_path)
    return result.get("ocr_text", "")


def batch_extract_images(image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
    """
    Batch extract content from multiple images.

    Args:
        image_paths: List of image paths

    Returns:
        List of extraction results
    """
    extractor = ImageExtractor()
    return extractor.extract_batch(image_paths)
