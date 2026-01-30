"""
Image processor module for the Multi-Modal RAG system.
Handles image preprocessing, enhancement, and feature extraction.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

from src.utils.logger import get_logger
from src.utils.config_loader import get_config_value
from src.utils.image_utils import ImageUtils
from src.utils.file_utils import FileUtils

logger = get_logger(__name__)


class ImageProcessor:
    """
    Advanced image processing for multi-modal RAG systems.
    """

    def __init__(
        self,
        max_dimension: Optional[int] = None,
        generate_captions: bool = True,
        perform_ocr: bool = True,
        ocr_engine: Optional[str] = None,
        enhance_images: bool = True,
        extract_features: bool = True
    ):
        """
        Initialize image processor.

        Args:
            max_dimension: Maximum dimension for resizing
            generate_captions: Whether to generate image captions
            perform_ocr: Whether to perform OCR
            ocr_engine: OCR engine to use
            enhance_images: Whether to enhance images
            extract_features: Whether to extract visual features
        """
        self.max_dimension = max_dimension or get_config_value("preprocessing.image.resize_max_dimension", 1024)
        self.generate_captions = generate_captions and get_config_value("preprocessing.image.generate_captions", True)
        self.perform_ocr = perform_ocr and get_config_value("preprocessing.image.perform_ocr", True)
        self.ocr_engine = ocr_engine or get_config_value("preprocessing.image.ocr_engine", "easyocr")
        self.enhance_images = enhance_images
        self.extract_features = extract_features

        # Initialize CLIP model for captioning and feature extraction
        self.clip_model = None
        self.clip_processor = None
        self.clip_tokenizer = None

        if self.generate_captions or self.extract_features:
            self._load_clip_model()

        # OCR components will be loaded as needed
        self.ocr_reader = None

        logger.info(f"ImageProcessor initialized with max_dimension={self.max_dimension}")

    def _load_clip_model(self):
        """Load CLIP model for image understanding."""
        try:
            model_name = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to('cuda')

            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
            self.generate_captions = False
            self.extract_features = False

    def _load_ocr_engine(self):
        """Load OCR engine."""
        if self.ocr_reader is not None:
            return

        try:
            if self.ocr_engine == "easyocr":
                import easyocr
                self.ocr_reader = easyocr.Reader(['en'])
            elif self.ocr_engine == "tesseract":
                import pytesseract
                # pytesseract is configured globally
                self.ocr_reader = pytesseract
            else:
                logger.warning(f"Unsupported OCR engine: {self.ocr_engine}")
                self.perform_ocr = False
        except ImportError as e:
            logger.warning(f"OCR engine {self.ocr_engine} not available: {e}")
            self.perform_ocr = False

    def process_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_processed: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single image with all enabled operations.

        Args:
            image_path: Path to input image
            output_dir: Directory to save processed images
            save_processed: Whether to save processed image

        Returns:
            Dictionary containing processed image data
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Processing image: {image_path}")

        # Load image
        image = ImageUtils.load_image(image_path)

        # Initialize result
        result = {
            "source_path": str(image_path),
            "filename": image_path.name,
            "original_size": image.size,
            "processed_size": None,
            "format": image.format or "Unknown",
            "mode": image.mode,
            "metadata": {}
        }

        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)

            if processed_image:
                result["processed_size"] = processed_image.size

                # Save processed image if requested
                if save_processed and output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    processed_path = output_dir / f"processed_{image_path.name}"
                    ImageUtils.save_image(processed_image, processed_path)
                    result["processed_path"] = str(processed_path)

            # Generate caption
            if self.generate_captions:
                caption = self.generate_caption(processed_image or image)
                result["caption"] = caption

            # Perform OCR
            if self.perform_ocr:
                ocr_result = self.perform_ocr_on_image(processed_image or image)
                result["ocr_text"] = ocr_result.get("text", "")
                result["ocr_confidence"] = ocr_result.get("confidence", 0.0)
                result["ocr_bounding_boxes"] = ocr_result.get("bounding_boxes", [])

            # Extract features
            if self.extract_features:
                features = self.extract_image_features(processed_image or image)
                result["features"] = features

            # Additional metadata
            result["metadata"].update({
                "aspect_ratio": image.size[0] / image.size[1] if image.size[1] > 0 else 0,
                "has_alpha": image.mode == "RGBA",
                "estimated_file_size": self._estimate_image_size(processed_image or image)
            })

            logger.success(f"Successfully processed image: {image_path.name}")
            return result

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            result["error"] = str(e)
            return result

    def preprocess_image(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Preprocess image with resizing, enhancement, and cleaning.

        Args:
            image: PIL Image to preprocess

        Returns:
            Processed PIL Image
        """
        try:
            # Convert to RGB if necessary
            if image.mode not in ["RGB", "L"]:
                image = image.convert("RGB")

            # Resize if too large
            if max(image.size) > self.max_dimension:
                image = ImageUtils.resize_image(image, max_size=self.max_dimension)

            # Enhance image if requested
            if self.enhance_images:
                image = self._enhance_image(image)

            return image

        except Exception as e:
            logger.warning(f"Error preprocessing image: {e}")
            return None

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality.

        Args:
            image: PIL Image to enhance

        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)

            # Denoise
            img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_array = cv2.filter2D(img_array, -1, kernel)

            # Convert back to PIL
            enhanced_image = Image.fromarray(img_array)

            # Adjust contrast and brightness
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(1.1)

            enhancer = ImageEnhance.Brightness(enhanced_image)
            enhanced_image = enhancer.enhance(1.05)

            return enhanced_image

        except Exception as e:
            logger.warning(f"Error enhancing image: {e}")
            return image

    def generate_caption(self, image: Image.Image) -> Optional[str]:
        """
        Generate descriptive caption for image using CLIP.

        Args:
            image: PIL Image

        Returns:
            Generated caption
        """
        if not self.clip_model or not self.generate_captions:
            return None

        try:
            # Prepare image for CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # Generate features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)

            # For captioning, we'd typically use a separate captioning model
            # For now, return a placeholder
            return f"Image of size {image.size} with {image.mode} mode"

        except Exception as e:
            logger.warning(f"Error generating caption: {e}")
            return None

    def perform_ocr_on_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Perform OCR on image.

        Args:
            image: PIL Image

        Returns:
            OCR results
        """
        if not self.perform_ocr:
            return {"text": "", "confidence": 0.0, "bounding_boxes": []}

        self._load_ocr_engine()

        if not self.ocr_reader:
            return {"text": "", "confidence": 0.0, "bounding_boxes": []}

        try:
            # Convert PIL to numpy array
            img_array = np.array(image)

            if self.ocr_engine == "easyocr":
                results = self.ocr_reader.readtext(img_array)

                text_parts = []
                confidences = []
                bounding_boxes = []

                for (bbox, text, confidence) in results:
                    text_parts.append(text)
                    confidences.append(confidence)
                    bounding_boxes.append(bbox)

                full_text = " ".join(text_parts)
                avg_confidence = np.mean(confidences) if confidences else 0.0

                return {
                    "text": full_text,
                    "confidence": float(avg_confidence),
                    "bounding_boxes": bounding_boxes
                }

            elif self.ocr_engine == "tesseract":
                import pytesseract
                text = pytesseract.image_to_string(img_array)

                # Get confidence (approximate)
                data = pytesseract.image_to_data(img_array, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0

                return {
                    "text": text,
                    "confidence": float(avg_confidence),
                    "bounding_boxes": []  # Tesseract doesn't provide easy bbox access
                }

        except Exception as e:
            logger.error(f"Error performing OCR: {e}")
            return {"text": "", "confidence": 0.0, "bounding_boxes": []}

    def extract_image_features(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Extract visual features from image using CLIP.

        Args:
            image: PIL Image

        Returns:
            Feature vector
        """
        if not self.clip_model or not self.extract_features:
            return None

        try:
            # Prepare image for CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # Extract features
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)

            # Convert to numpy and normalize
            features = features.cpu().numpy()
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

            return features.flatten()

        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            return None

    def _estimate_image_size(self, image: Image.Image) -> int:
        """
        Estimate file size of image.

        Args:
            image: PIL Image

        Returns:
            Estimated file size in bytes
        """
        try:
            # Rough estimation based on dimensions and mode
            pixels = image.size[0] * image.size[1]
            bytes_per_pixel = 3 if image.mode == "RGB" else 1  # RGB = 3 bytes, L = 1 byte
            compression_factor = 0.3  # JPEG compression
            return int(pixels * bytes_per_pixel * compression_factor)
        except:
            return 0

    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batches.

        Args:
            image_paths: List of image paths
            output_dir: Output directory
            batch_size: Number of images to process at once

        Returns:
            List of processing results
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")

            for path in batch:
                try:
                    result = self.process_image(path, output_dir)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    results.append({
                        "source_path": str(path),
                        "error": str(e)
                    })

        logger.info(f"Processed {len(results)} images in total")
        return results

    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ):
        """
        Save processing results to file.

        Args:
            results: List of processing results
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        FileUtils.save_json({"images": results}, output_path)
        logger.info(f"Saved image processing results to {output_path}")


# Convenience functions
def process_image_simple(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Simple image processing function.

    Args:
        image_path: Path to image

    Returns:
        Processing results
    """
    processor = ImageProcessor()
    return processor.process_image(image_path)


def extract_image_features_simple(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Simple feature extraction function.

    Args:
        image_path: Path to image

    Returns:
        Feature vector
    """
    processor = ImageProcessor(generate_captions=False, perform_ocr=False)
    image = ImageUtils.load_image(image_path)
    return processor.extract_image_features(image)
