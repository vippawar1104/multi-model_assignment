"""
Tests for data ingestion modules.
"""

import pytest
from pathlib import Path
import tempfile
from PIL import Image

from src.data_ingestion.pdf_extractor import PDFExtractor
from src.data_ingestion.image_extractor import ImageExtractor
from src.data_ingestion.audio_extractor import AudioExtractor
from src.data_ingestion.video_extractor import VideoExtractor
from src.utils.file_utils import FileUtils


class TestPDFExtractor:
    """Test PDF extraction functionality."""

    def test_pdf_extractor_init(self):
        """Test PDF extractor initialization."""
        extractor = PDFExtractor()
        assert extractor.dpi == 300
        assert extractor.extract_images == True

    def test_extract_pdf_metadata(self):
        """Test PDF metadata extraction."""
        extractor = PDFExtractor()

        # This would need a test PDF file
        # For now, just test that the extractor initializes
        assert extractor is not None


class TestImageExtractor:
    """Test image extraction functionality."""

    def test_image_extractor_init(self):
        """Test image extractor initialization."""
        extractor = ImageExtractor()
        assert extractor.generate_captions == True
        assert extractor.perform_ocr == True

    def test_create_test_image(self):
        """Create a test image for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "test_image.png"

            # Create a simple test image
            image = Image.new('RGB', (100, 100), color='red')
            image.save(image_path)

            assert image_path.exists()
            return image_path

    def test_image_validation(self):
        """Test image validation."""
        extractor = ImageExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            image_path = self.test_create_test_image()

            # Test valid image
            is_valid = extractor._ImageExtractor__class__.is_valid_image(image_path)
            assert is_valid == True

            # Test invalid path
            invalid_path = Path(tmpdir) / "nonexistent.png"
            is_valid = extractor._ImageExtractor__class__.is_valid_image(invalid_path)
            assert is_valid == False


class TestAudioExtractor:
    """Test audio extraction functionality."""

    def test_audio_extractor_init(self):
        """Test audio extractor initialization."""
        extractor = AudioExtractor()
        assert extractor.whisper_model_size == "base"
        assert extractor.language == "en"

    def test_get_audio_duration(self):
        """Test audio duration extraction."""
        extractor = AudioExtractor()

        # This would need a test audio file
        # For now, just test that the method exists
        assert hasattr(extractor, 'get_audio_duration')


class TestVideoExtractor:
    """Test video extraction functionality."""

    def test_video_extractor_init(self):
        """Test video extractor initialization."""
        extractor = VideoExtractor()
        assert extractor.frame_extraction_rate == 1.0
        assert extractor.extract_audio == True

    def test_video_metadata_extraction(self):
        """Test video metadata extraction."""
        extractor = VideoExtractor()

        # This would need a test video file
        # For now, just test that the method exists
        assert hasattr(extractor, '_extract_metadata')


class TestIntegration:
    """Integration tests for data ingestion."""

    def test_import_all_extractors(self):
        """Test that all extractors can be imported."""
        try:
            from src.data_ingestion import (
                PDFExtractor,
                ImageExtractor,
                VideoExtractor,
                AudioExtractor
            )
            assert True
        except ImportError:
            assert False, "Failed to import extractors"

    def test_convenience_functions(self):
        """Test convenience functions."""
        from src.data_ingestion import (
            extract_pdf_content,
            extract_image_content,
            extract_video_content,
            extract_audio_content
        )

        # Just test that functions exist
        assert callable(extract_pdf_content)
        assert callable(extract_image_content)
        assert callable(extract_video_content)
        assert callable(extract_audio_content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
