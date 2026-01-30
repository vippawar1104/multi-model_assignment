"""
Chart processor module for the Multi-Modal RAG system.
Handles chart and diagram extraction, analysis, and description.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.utils.logger import get_logger
from src.utils.config_loader import get_config_value
from src.utils.image_utils import ImageUtils
from src.utils.file_utils import FileUtils

logger = get_logger(__name__)


class ChartProcessor:
    """
    Advanced chart and diagram processing for multi-modal RAG systems.
    """

    def __init__(
        self,
        extract_text: bool = True,
        analyze_layout: bool = True,
        generate_descriptions: bool = True,
        detect_chart_type: bool = True,
        extract_data_points: bool = False
    ):
        """
        Initialize chart processor.

        Args:
            extract_text: Whether to extract text from charts
            analyze_layout: Whether to analyze chart layout
            generate_descriptions: Whether to generate descriptions
            detect_chart_type: Whether to detect chart type
            extract_data_points: Whether to extract data points
        """
        self.extract_text = extract_text
        self.analyze_layout = analyze_layout
        self.generate_descriptions = generate_descriptions
        self.detect_chart_type = detect_chart_type
        self.extract_data_points = extract_data_points

        # Initialize TrOCR for text extraction from images
        self.trocr_processor = None
        self.trocr_model = None

        if self.extract_text:
            self._load_trocr_model()

        logger.info("ChartProcessor initialized")

    def _load_trocr_model(self):
        """Load TrOCR model for text extraction."""
        try:
            model_name = "microsoft/trocr-base-printed"
            self.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.trocr_model = self.trocr_model.to('cuda')

            logger.info("TrOCR model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load TrOCR model: {e}")
            self.extract_text = False

    def process_chart(
        self,
        image_path: Union[str, Path],
        chart_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a chart or diagram image.

        Args:
            image_path: Path to chart image
            chart_type: Optional chart type hint

        Returns:
            Processed chart data
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Chart image not found: {image_path}")

        logger.info(f"Processing chart: {image_path}")

        # Load image
        image = ImageUtils.load_image(image_path)

        # Initialize result
        result = {
            "source_path": str(image_path),
            "filename": image_path.name,
            "image_size": image.size,
            "detected_type": chart_type,
            "metadata": {}
        }

        try:
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Detect chart type if not provided
            if not chart_type and self.detect_chart_type:
                detected_type = self._detect_chart_type(image)
                result["detected_type"] = detected_type

            # Extract text
            if self.extract_text:
                extracted_text = self._extract_text_from_chart(image)
                result["extracted_text"] = extracted_text

            # Analyze layout
            if self.analyze_layout:
                layout_analysis = self._analyze_chart_layout(image)
                result["layout_analysis"] = layout_analysis

            # Generate description
            if self.generate_descriptions:
                description = self._generate_chart_description(image, result)
                result["description"] = description

            # Extract data points (advanced feature)
            if self.extract_data_points:
                data_points = self._extract_data_points(image, result.get("detected_type"))
                result["data_points"] = data_points

            # Create text representation for embedding
            result["text_representation"] = self._create_text_representation(result)

            logger.success(f"Successfully processed chart: {image_path.name}")
            return result

        except Exception as e:
            logger.error(f"Error processing chart {image_path}: {e}")
            result["error"] = str(e)
            return result

    def _detect_chart_type(self, image: Image.Image) -> Optional[str]:
        """
        Detect the type of chart in the image.

        Args:
            image: Chart image

        Returns:
            Detected chart type
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Simple heuristics for chart type detection

            # Check for pie chart (circular shapes)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=10, maxRadius=200
            )
            if circles is not None and len(circles[0]) > 0:
                return "pie_chart"

            # Check for line/bar chart (straight lines and bars)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

            if lines is not None:
                horizontal_lines = sum(1 for line in lines[0] if abs(line[1] - line[3]) < 10)
                vertical_lines = sum(1 for line in lines[0] if abs(line[0] - line[2]) < 10)

                if vertical_lines > horizontal_lines * 2:
                    return "bar_chart"
                elif horizontal_lines > 5:
                    return "line_chart"

            # Check for scatter plot (dots/patterns)
            # This is a simplified check
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 20:  # Many small contours might indicate scatter plot
                return "scatter_plot"

            # Default
            return "unknown_chart"

        except Exception as e:
            logger.warning(f"Error detecting chart type: {e}")
            return "unknown_chart"

    def _extract_text_from_chart(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract text from chart using OCR.

        Args:
            image: Chart image

        Returns:
            Extracted text information
        """
        if not self.trocr_processor or not self.trocr_model:
            return {"text": "", "confidence": 0.0}

        try:
            # Prepare image for TrOCR
            pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values

            if torch.cuda.is_available():
                pixel_values = pixel_values.to('cuda')

            # Generate text
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(
                    pixel_values,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )

            generated_text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return {
                "text": generated_text,
                "confidence": 0.8  # Placeholder confidence
            }

        except Exception as e:
            logger.warning(f"Error extracting text from chart: {e}")
            return {"text": "", "confidence": 0.0}

    def _analyze_chart_layout(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze the layout of the chart.

        Args:
            image: Chart image

        Returns:
            Layout analysis results
        """
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze contours
            contour_info = []
            for contour in contours[:10]:  # Limit to top 10 contours
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                x, y, w, h = cv2.boundingRect(contour)

                contour_info.append({
                    "area": area,
                    "perimeter": perimeter,
                    "bounding_box": (x, y, w, h),
                    "aspect_ratio": w / h if h > 0 else 0
                })

            # Color analysis
            colors = self._analyze_colors(img_array)

            return {
                "num_contours": len(contours),
                "contour_info": contour_info,
                "colors": colors,
                "image_dimensions": image.size
            }

        except Exception as e:
            logger.warning(f"Error analyzing chart layout: {e}")
            return {}

    def _analyze_colors(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze colors in the chart.

        Args:
            img_array: Image array

        Returns:
            Color analysis
        """
        try:
            # Reshape image to be a list of pixels
            pixels = img_array.reshape(-1, 3)

            # Calculate unique colors (simplified)
            unique_colors = np.unique(pixels, axis=0)

            # Calculate dominant colors using k-means (simplified)
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=min(5, len(unique_colors)), n_init=10)
            kmeans.fit(pixels)

            dominant_colors = kmeans.cluster_centers_.astype(int)

            return {
                "num_unique_colors": len(unique_colors),
                "dominant_colors": dominant_colors.tolist()
            }

        except Exception as e:
            logger.warning(f"Error analyzing colors: {e}")
            return {"num_unique_colors": 0, "dominant_colors": []}

    def _generate_chart_description(self, image: Image.Image, chart_info: Dict[str, Any]) -> str:
        """
        Generate a natural language description of the chart.

        Args:
            image: Chart image
            chart_info: Chart analysis information

        Returns:
            Chart description
        """
        try:
            description_parts = []

            # Chart type
            chart_type = chart_info.get("detected_type", "unknown")
            if chart_type != "unknown_chart":
                description_parts.append(f"This appears to be a {chart_type.replace('_', ' ')}.")
            else:
                description_parts.append("This appears to be a chart or diagram.")

            # Size information
            size = chart_info.get("image_size", (0, 0))
            description_parts.append(f"The image has dimensions {size[0]}x{size[1]} pixels.")

            # Text content
            extracted_text = chart_info.get("extracted_text", {})
            if extracted_text.get("text"):
                description_parts.append(f"It contains the following text: {extracted_text['text'][:200]}...")

            # Layout information
            layout = chart_info.get("layout_analysis", {})
            num_contours = layout.get("num_contours", 0)
            if num_contours > 0:
                description_parts.append(f"The chart contains {num_contours} distinct visual elements.")

            # Colors
            colors = layout.get("colors", {})
            num_colors = colors.get("num_unique_colors", 0)
            if num_colors > 0:
                description_parts.append(f"The chart uses approximately {num_colors} different colors.")

            return " ".join(description_parts)

        except Exception as e:
            logger.warning(f"Error generating chart description: {e}")
            return "A chart or diagram image."

    def _extract_data_points(self, image: Image.Image, chart_type: Optional[str]) -> List[Dict[str, Any]]:
        """
        Extract data points from chart (advanced feature).

        Args:
            image: Chart image
            chart_type: Type of chart

        Returns:
            List of data points
        """
        # This is a placeholder for advanced data point extraction
        # In a full implementation, this would use computer vision techniques
        # to identify and extract numerical data from charts

        logger.info("Data point extraction is not fully implemented yet")
        return []

    def _create_text_representation(self, chart_info: Dict[str, Any]) -> str:
        """
        Create text representation for embedding.

        Args:
            chart_info: Chart information

        Returns:
            Text representation
        """
        text_parts = []

        # Basic information
        text_parts.append(f"Chart: {chart_info.get('filename', 'unknown')}")
        text_parts.append(f"Type: {chart_info.get('detected_type', 'unknown')}")

        # Description
        if chart_info.get("description"):
            text_parts.append(f"Description: {chart_info['description']}")

        # Extracted text
        extracted = chart_info.get("extracted_text", {})
        if extracted.get("text"):
            text_parts.append(f"Text content: {extracted['text']}")

        return "\n".join(text_parts)

    def process_charts_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple charts in batch.

        Args:
            image_paths: List of chart image paths
            output_dir: Output directory

        Returns:
            List of processing results
        """
        results = []

        for path in image_paths:
            try:
                result = self.process_chart(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process chart {path}: {e}")
                results.append({
                    "source_path": str(path),
                    "error": str(e)
                })

        logger.info(f"Processed {len(results)} charts")
        return results

    def save_processed_charts(
        self,
        charts: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ):
        """
        Save processed charts to file.

        Args:
            charts: List of processed charts
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        FileUtils.save_json({"charts": charts}, output_path)
        logger.info(f"Saved {len(charts)} processed charts to {output_path}")


# Convenience functions
def process_chart_simple(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Simple chart processing function.

    Args:
        image_path: Path to chart image

    Returns:
        Processed chart data
    """
    processor = ChartProcessor()
    return processor.process_chart(image_path)


def detect_chart_type_simple(image_path: Union[str, Path]) -> Optional[str]:
    """
    Simple chart type detection function.

    Args:
        image_path: Path to chart image

    Returns:
        Detected chart type
    """
    processor = ChartProcessor(
        extract_text=False,
        analyze_layout=False,
        generate_descriptions=False,
        extract_data_points=False
    )
    image = ImageUtils.load_image(image_path)
    return processor._detect_chart_type(image)
