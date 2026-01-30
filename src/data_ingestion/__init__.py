"""
Data ingestion module for the Multi-Modal RAG system.
Handles extraction of content from various modalities.
"""

from src.data_ingestion.pdf_extractor import PDFExtractor, extract_pdf_content, extract_pdf_text
from src.data_ingestion.image_extractor import ImageExtractor, extract_image_content, extract_image_text, batch_extract_images
from src.data_ingestion.video_extractor import VideoExtractor, extract_video_content, extract_video_frames, extract_video_audio
from src.data_ingestion.audio_extractor import AudioExtractor, extract_audio_content, transcribe_audio, batch_transcribe_audio

__all__ = [
    # PDF Extractor
    "PDFExtractor",
    "extract_pdf_content",
    "extract_pdf_text",

    # Image Extractor
    "ImageExtractor",
    "extract_image_content",
    "extract_image_text",
    "batch_extract_images",

    # Video Extractor
    "VideoExtractor",
    "extract_video_content",
    "extract_video_frames",
    "extract_video_audio",

    # Audio Extractor
    "AudioExtractor",
    "extract_audio_content",
    "transcribe_audio",
    "batch_transcribe_audio",
]
