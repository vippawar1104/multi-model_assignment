"""
Example script demonstrating data ingestion modules.
Shows how to extract content from different modalities.
"""

from pathlib import Path
from src.data_ingestion import (
    PDFExtractor,
    ImageExtractor,
    VideoExtractor,
    AudioExtractor
)
from src.utils import get_logger, FileUtils

logger = get_logger(__name__)


def demonstrate_pdf_extraction():
    """Demonstrate PDF content extraction."""
    print("=" * 60)
    print("1. PDF Extraction Demo")
    print("=" * 60)

    extractor = PDFExtractor()

    # Check if test PDF exists
    pdf_path = Path("data/raw/qatar_test_doc.pdf")
    if pdf_path.exists():
        try:
            logger.info(f"Extracting content from PDF: {pdf_path}")
            result = extractor.extract(pdf_path)

            print(f"✅ PDF extracted successfully!")
            print(f"   Pages: {result['metadata']['pages']}")
            print(f"   Images: {result['metadata']['total_images']}")
            print(f"   Tables: {result['metadata']['total_tables']}")

            # Show sample text
            if result['text']:
                first_page = list(result['text'].keys())[0]
                text_preview = result['text'][first_page]['content'][:200]
                print(f"   Sample text: {text_preview}...")

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            print(f"❌ PDF extraction failed: {e}")
    else:
        print(f"⚠️  Test PDF not found: {pdf_path}")
        print("   Please place a PDF file in data/raw/ to test PDF extraction")


def demonstrate_image_extraction():
    """Demonstrate image content extraction."""
    print("\n" + "=" * 60)
    print("2. Image Extraction Demo")
    print("=" * 60)

    extractor = ImageExtractor()

    # Check for test images
    image_dir = Path("data/processed/images")
    if image_dir.exists():
        image_files = list(image_dir.glob("*.png"))[:1]  # Take first image

        if image_files:
            image_path = image_files[0]
            try:
                logger.info(f"Extracting content from image: {image_path}")
                result = extractor.extract(image_path)

                print(f"✅ Image extracted successfully!")
                print(f"   Dimensions: {result['metadata']['width']}x{result['metadata']['height']}")
                print(f"   OCR Text: {'Yes' if result['ocr_text'] else 'No'}")
                print(f"   Caption: {'Yes' if result['caption'] else 'No'}")
                print(f"   Confidence: {result['confidence']:.2f}")

                if result['ocr_text']:
                    text_preview = result['ocr_text'][:100]
                    print(f"   Sample OCR: {text_preview}...")

            except Exception as e:
                logger.error(f"Image extraction failed: {e}")
                print(f"❌ Image extraction failed: {e}")
        else:
            print("⚠️  No images found in data/processed/images/")
            print("   Run PDF extraction first to generate images")
    else:
        print("⚠️  Image directory not found")


def demonstrate_video_extraction():
    """Demonstrate video content extraction."""
    print("\n" + "=" * 60)
    print("3. Video Extraction Demo")
    print("=" * 60)

    extractor = VideoExtractor()

    # Check for test video
    video_path = Path("data/raw/sample_video.mp4")
    if video_path.exists():
        try:
            logger.info(f"Extracting content from video: {video_path}")
            result = extractor.extract(video_path)

            print(f"✅ Video extracted successfully!")
            print(f"   Duration: {result['metadata']['duration']:.1f}s")
            print(f"   FPS: {result['metadata']['fps']:.1f}")
            print(f"   Frames extracted: {result['metadata']['extracted_frames']}")
            print(f"   Audio extracted: {result['metadata']['audio_extracted']}")

        except Exception as e:
            logger.error(f"Video extraction failed: {e}")
            print(f"❌ Video extraction failed: {e}")
    else:
        print("⚠️  Test video not found: data/raw/sample_video.mp4")
        print("   Please place a video file to test video extraction")


def demonstrate_audio_extraction():
    """Demonstrate audio content extraction."""
    print("\n" + "=" * 60)
    print("4. Audio Extraction Demo")
    print("=" * 60)

    extractor = AudioExtractor()

    # Check for test audio
    audio_path = Path("data/raw/sample_audio.wav")
    if audio_path.exists():
        try:
            logger.info(f"Extracting content from audio: {audio_path}")
            result = extractor.extract(audio_path)

            print(f"✅ Audio extracted successfully!")
            print(f"   Duration: {result['metadata']['duration']:.1f}s")
            print(f"   Sample rate: {result['metadata']['sample_rate']}Hz")
            print(f"   Channels: {result['metadata']['channels']}")
            print(f"   Transcription: {'Yes' if result['transcription']['text'] else 'No'}")
            print(f"   Confidence: {result['transcription']['confidence']:.2f}")

            if result['transcription']['text']:
                text_preview = result['transcription']['text'][:100]
                print(f"   Sample text: {text_preview}...")

        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            print(f"❌ Audio extraction failed: {e}")
    else:
        print("⚠️  Test audio not found: data/raw/sample_audio.wav")
        print("   Please place an audio file to test audio extraction")


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n" + "=" * 60)
    print("5. Batch Processing Demo")
    print("=" * 60)

    # Check for multiple images
    image_dir = Path("data/processed/images")
    if image_dir.exists():
        image_files = list(image_dir.glob("*.png"))[:3]  # Take up to 3 images

        if len(image_files) > 1:
            try:
                logger.info(f"Batch extracting {len(image_files)} images")
                results = ImageExtractor().extract_batch(image_files)

                print(f"✅ Batch extraction successful!")
                print(f"   Processed: {len(results)} images")

                successful = sum(1 for r in results if r.get('confidence', 0) > 0)
                print(f"   Successful: {successful}/{len(results)}")

            except Exception as e:
                logger.error(f"Batch extraction failed: {e}")
                print(f"❌ Batch extraction failed: {e}")
        else:
            print("⚠️  Need at least 2 images for batch processing demo")
    else:
        print("⚠️  No images available for batch processing")


def main():
    """Main demonstration function."""
    print("🚀 Multi-Modal Data Ingestion Demo")
    print("===================================")

    # Run all demonstrations
    demonstrate_pdf_extraction()
    demonstrate_image_extraction()
    demonstrate_video_extraction()
    demonstrate_audio_extraction()
    demonstrate_batch_processing()

    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add test files to data/raw/ directory")
    print("2. Run individual extractors for detailed testing")
    print("3. Check extracted content in data/processed/")
    print("4. Proceed to Phase 4: Preprocessing modules")


if __name__ == "__main__":
    main()
