"""
Test file for preprocessing modules.
Tests text chunking, image processing, table processing, and chart processing.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_chunker import TextChunker, chunk_text_simple
from src.preprocessing.image_processor import ImageProcessor
from src.preprocessing.table_processor import TableProcessor, process_table_simple
from src.preprocessing.chart_processor import ChartProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_text_chunker():
    """Test text chunking functionality."""
    logger.info("Testing TextChunker...")

    # Test basic chunking
    text = "This is a test document. " * 100  # Long text
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)

    chunks = chunker.chunk_text(text)
    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("chunk_index" in chunk for chunk in chunks)

    # Test simple function
    simple_chunks = chunk_text_simple(text, chunk_size=50)
    assert len(simple_chunks) > 0
    assert isinstance(simple_chunks[0], str)

    logger.success("TextChunker tests passed")


def test_table_processor():
    """Test table processing functionality."""
    logger.info("Testing TableProcessor...")

    # Test with DataFrame
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "City": ["New York", "London", "Tokyo"]
    })

    processor = TableProcessor()
    result = processor.process_table(df)

    assert result["table_id"].startswith("table_")
    assert result["processed_shape"] == (3, 3)
    assert "markdown" in result
    assert "text_representation" in result
    assert len(result["chunks"]) > 0

    # Test with list of lists
    table_data = [
        ["Name", "Age", "City"],
        ["Alice", 25, "New York"],
        ["Bob", 30, "London"]
    ]

    result2 = process_table_simple(table_data)
    assert result2["processed_shape"] == (2, 3)

    logger.success("TableProcessor tests passed")


def test_image_processor():
    """Test image processing functionality."""
    logger.info("Testing ImageProcessor...")

    # Create a simple test image if none exists
    from PIL import Image
    import numpy as np

    # Create a test image
    test_image_path = Path("data/processed/images/test_image.png")
    test_image_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a simple RGB image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    test_image.save(test_image_path)

    try:
        processor = ImageProcessor(
            generate_captions=False,  # Skip CLIP for testing
            perform_ocr=False,       # Skip OCR for testing
            extract_features=False   # Skip feature extraction for testing
        )

        result = processor.process_image(test_image_path)

        assert result["filename"] == "test_image.png"
        assert "processed_size" in result
        assert "metadata" in result

        logger.success("ImageProcessor tests passed")

    finally:
        # Clean up
        if test_image_path.exists():
            test_image_path.unlink()


def test_chart_processor():
    """Test chart processing functionality."""
    logger.info("Testing ChartProcessor...")

    # Create a simple test chart image
    from PIL import Image, ImageDraw
    import numpy as np

    test_chart_path = Path("data/processed/images/test_chart.png")
    test_chart_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a simple chart-like image with some shapes
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)

    # Draw some rectangles (like bars in a bar chart)
    draw.rectangle([50, 50, 80, 150], fill='blue')
    draw.rectangle([100, 30, 130, 150], fill='red')
    draw.rectangle([150, 70, 180, 150], fill='green')

    img.save(test_chart_path)

    try:
        processor = ChartProcessor(
            extract_text=False,      # Skip TrOCR for testing
            generate_descriptions=True,
            detect_chart_type=True
        )

        result = processor.process_chart(test_chart_path)

        assert result["filename"] == "test_chart.png"
        assert "detected_type" in result
        assert "description" in result
        assert "text_representation" in result

        logger.success("ChartProcessor tests passed")

    finally:
        # Clean up
        if test_chart_path.exists():
            test_chart_path.unlink()


def test_preprocessing_integration():
    """Test integration of preprocessing modules."""
    logger.info("Testing preprocessing integration...")

    # Test that all modules can be imported
    from src.preprocessing import (
        TextChunker, ImageProcessor, TableProcessor, ChartProcessor
    )

    # Test basic functionality
    text = "This is a sample document for testing."
    chunker = TextChunker(chunk_size=20)
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 0

    # Test table processing
    table_data = [["A", "B"], [1, 2], [3, 4]]
    table_result = process_table_simple(table_data)
    assert table_result["processed_shape"] == (2, 2)

    logger.success("Preprocessing integration tests passed")


if __name__ == "__main__":
    # Run tests
    try:
        test_text_chunker()
        test_table_processor()
        test_image_processor()
        test_chart_processor()
        test_preprocessing_integration()

        logger.success("All preprocessing tests passed!")

    except Exception as e:
        logger.error(f"Preprocessing tests failed: {e}")
        raise
