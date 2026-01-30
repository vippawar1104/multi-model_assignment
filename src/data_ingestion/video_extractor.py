"""
Video extractor module for the Multi-Modal RAG system.
Extracts frames, audio, and metadata from video files.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
from moviepy import VideoFileClip
from PIL import Image
import numpy as np

from src.utils.logger import get_logger
from src.utils.file_utils import FileUtils, ensure_dir
from src.utils.image_utils import ImageUtils
from src.utils.config_loader import get_config_value

logger = get_logger(__name__)


class VideoExtractor:
    """
    Extract content from video files including frames, audio, and metadata.
    """

    def __init__(
        self,
        frame_extraction_rate: float = 1.0,  # frames per second
        extract_audio: bool = True,
        max_frames: Optional[int] = None,
        frame_quality: int = 95,
        audio_format: str = "wav"
    ):
        """
        Initialize video extractor.

        Args:
            frame_extraction_rate: Frames to extract per second
            extract_audio: Whether to extract audio
            max_frames: Maximum number of frames to extract
            frame_quality: JPEG quality for saved frames
            audio_format: Audio export format
        """
        self.frame_extraction_rate = frame_extraction_rate
        self.extract_audio = extract_audio
        self.max_frames = max_frames
        self.frame_quality = frame_quality
        self.audio_format = audio_format

        # Output directories will be set per extraction
        self.output_dir = None

        logger.info(f"VideoExtractor initialized with rate={frame_extraction_rate} fps, audio={extract_audio}")

    def extract(self, video_path: Union[str, Path], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract content from video file.

        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted content

        Returns:
            Dictionary containing extracted content
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(get_config_value("paths.processed_data", "data/processed")) / "video"

        ensure_dir(self.output_dir)

        logger.info(f"Extracting content from video: {video_path}")

        # Initialize result structure
        result = {
            "metadata": {
                "source": str(video_path),
                "filename": video_path.name,
                "duration": 0.0,
                "fps": 0.0,
                "width": 0,
                "height": 0,
                "total_frames": 0,
                "extracted_frames": 0,
                "audio_extracted": False
            },
            "frames": {},
            "audio": {},
            "keyframes": [],
            "scenes": []
        }

        try:
            # Extract metadata first
            metadata = self._extract_metadata(video_path)
            result["metadata"].update(metadata)

            # Extract frames
            frames_result = self._extract_frames(video_path, result["metadata"])
            result["frames"] = frames_result["frames"]
            result["metadata"]["extracted_frames"] = frames_result["count"]

            # Extract audio
            if self.extract_audio:
                audio_result = self._extract_audio(video_path)
                result["audio"] = audio_result
                result["metadata"]["audio_extracted"] = audio_result["extracted"]

            # Detect keyframes (optional)
            keyframes = self._detect_keyframes(video_path)
            result["keyframes"] = keyframes

            logger.success(f"Successfully extracted content from {video_path.name}")
            logger.info(f"Duration: {result['metadata']['duration']:.1f}s, Frames: {result['metadata']['extracted_frames']}, Audio: {result['metadata']['audio_extracted']}")

        except Exception as e:
            logger.error(f"Error extracting video {video_path}: {e}")
            raise

        return result

    def _extract_metadata(self, video_path: Path) -> Dict[str, Any]:
        """
        Extract video metadata.

        Args:
            video_path: Path to video file

        Returns:
            Video metadata
        """
        metadata = {}

        try:
            # Use OpenCV to get basic metadata
            cap = cv2.VideoCapture(str(video_path))

            if cap.isOpened():
                metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
                metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata["duration"] = metadata["total_frames"] / metadata["fps"] if metadata["fps"] > 0 else 0

                cap.release()

            # Use moviepy for additional metadata
            clip = VideoFileClip(str(video_path))
            metadata["duration"] = clip.duration
            metadata["audio_channels"] = clip.audio.nchannels if clip.audio else 0
            metadata["audio_fps"] = clip.audio.fps if clip.audio else 0
            clip.close()

        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")

        return metadata

    def _extract_frames(self, video_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract frames from video.

        Args:
            video_path: Path to video file
            metadata: Video metadata

        Returns:
            Extracted frames information
        """
        result = {
            "frames": {},
            "count": 0
        }

        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                logger.error("Could not open video file")
                return result

            fps = metadata.get("fps", 30)
            duration = metadata.get("duration", 0)

            # Calculate frame extraction interval
            if self.frame_extraction_rate <= 0:
                frame_interval = 1  # Extract every frame
            else:
                frame_interval = int(fps / self.frame_extraction_rate)

            frame_count = 0
            extracted_count = 0

            # Create frames directory
            frames_dir = self.output_dir / "frames" / video_path.stem
            ensure_dir(frames_dir)

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)

                    # Calculate timestamp
                    timestamp = frame_count / fps

                    # Save frame
                    frame_filename = "04d"
                    frame_path = frames_dir / frame_filename

                    ImageUtils.save_image(image, frame_path, quality=self.frame_quality)

                    # Store frame info
                    result["frames"][extracted_count] = {
                        "filename": frame_filename,
                        "path": str(frame_path),
                        "timestamp": timestamp,
                        "frame_number": frame_count,
                        "width": image.width,
                        "height": image.height
                    }

                    extracted_count += 1

                    # Check max frames limit
                    if self.max_frames and extracted_count >= self.max_frames:
                        break

                frame_count += 1

            result["count"] = extracted_count
            cap.release()

        except Exception as e:
            logger.error(f"Error extracting frames: {e}")

        return result

    def _extract_audio(self, video_path: Path) -> Dict[str, Any]:
        """
        Extract audio from video.

        Args:
            video_path: Path to video file

        Returns:
            Audio extraction results
        """
        result = {
            "extracted": False,
            "audio_path": "",
            "duration": 0.0,
            "format": self.audio_format
        }

        try:
            clip = VideoFileClip(str(video_path))

            if clip.audio is None:
                logger.warning("No audio track found in video")
                clip.close()
                return result

            # Create audio directory
            audio_dir = self.output_dir / "audio"
            ensure_dir(audio_dir)

            # Export audio
            audio_filename = f"{video_path.stem}_audio.{self.audio_format}"
            audio_path = audio_dir / audio_filename

            clip.audio.write_audiofile(
                str(audio_path),
                verbose=False,
                logger=None
            )

            result["extracted"] = True
            result["audio_path"] = str(audio_path)
            result["duration"] = clip.duration

            clip.close()

            logger.info(f"Extracted audio to: {audio_path}")

        except Exception as e:
            logger.error(f"Error extracting audio: {e}")

        return result

    def _detect_keyframes(self, video_path: Path) -> List[Dict[str, Any]]:
        """
        Detect keyframes in video (simplified implementation).

        Args:
            video_path: Path to video file

        Returns:
            List of keyframe information
        """
        keyframes = []

        try:
            # This is a simplified keyframe detection
            # In a production system, you might use more sophisticated algorithms
            # For now, we'll just mark the first frame and frames at scene changes

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                return keyframes

            # Get first frame
            ret, prev_frame = cap.read()
            if ret:
                keyframes.append({
                    "frame_number": 0,
                    "timestamp": 0.0,
                    "type": "first_frame"
                })

            # Simple scene change detection (basic histogram difference)
            frame_count = 1
            fps = cap.get(cv2.CAP_PROP_FPS)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate histogram difference
                hist_prev = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
                hist_curr = cv2.calcHist([frame], [0], None, [256], [0, 256])

                hist_diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)

                # If correlation is low (significant difference), mark as keyframe
                if hist_diff < 0.9:  # Threshold for scene change
                    keyframes.append({
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps,
                        "type": "scene_change",
                        "confidence": 1.0 - hist_diff
                    })

                prev_frame = frame
                frame_count += 1

                # Limit keyframes
                if len(keyframes) > 20:  # Max 20 keyframes
                    break

            cap.release()

        except Exception as e:
            logger.warning(f"Keyframe detection failed: {e}")

        return keyframes

    def extract_frame_at_time(self, video_path: Union[str, Path], timestamp: float) -> Optional[Image.Image]:
        """
        Extract a single frame at specific timestamp.

        Args:
            video_path: Path to video file
            timestamp: Time in seconds

        Returns:
            PIL Image of the frame
        """
        try:
            clip = VideoFileClip(str(video_path))
            frame = clip.get_frame(timestamp)
            clip.close()

            # Convert to PIL Image
            return Image.fromarray(frame)

        except Exception as e:
            logger.error(f"Error extracting frame at {timestamp}s: {e}")
            return None

    def get_video_thumbnail(self, video_path: Union[str, Path], timestamp: float = 1.0) -> Optional[Image.Image]:
        """
        Get video thumbnail at specific timestamp.

        Args:
            video_path: Path to video file
            timestamp: Time in seconds (default: 1.0)

        Returns:
            PIL Image thumbnail
        """
        image = self.extract_frame_at_time(video_path, timestamp)

        if image:
            # Create thumbnail
            return ImageUtils.create_thumbnail(image, size=(320, 180))

        return None

    def save_extracted_content(self, result: Dict[str, Any], output_dir: Optional[str] = None):
        """
        Save extracted content metadata.

        Args:
            result: Extraction result
            output_dir: Output directory
        """
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = self.output_dir

        ensure_dir(output_dir)

        # Save metadata
        metadata_file = output_dir / f"{result['metadata']['filename']}_metadata.json"
        FileUtils.save_json(result["metadata"], metadata_file)

        # Save keyframes
        if result["keyframes"]:
            keyframes_file = output_dir / f"{result['metadata']['filename']}_keyframes.json"
            FileUtils.save_json(result["keyframes"], keyframes_file)

        logger.info(f"Saved video extraction metadata to: {output_dir}")


# Convenience functions
def extract_video_content(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to extract video content.

    Args:
        video_path: Path to video file

    Returns:
        Extracted content
    """
    extractor = VideoExtractor()
    return extractor.extract(video_path)


def extract_video_frames(video_path: Union[str, Path], fps: float = 1.0) -> List[Image.Image]:
    """
    Extract frames from video at specified rate.

    Args:
        video_path: Path to video file
        fps: Frames per second to extract

    Returns:
        List of PIL Images
    """
    extractor = VideoExtractor(frame_extraction_rate=fps)
    result = extractor.extract(video_path)

    frames = []
    for frame_info in result["frames"].values():
        try:
            image = ImageUtils.load_image(frame_info["path"])
            frames.append(image)
        except Exception as e:
            logger.warning(f"Could not load frame {frame_info['path']}: {e}")

    return frames


def extract_video_audio(video_path: Union[str, Path]) -> Optional[str]:
    """
    Extract audio from video.

    Args:
        video_path: Path to video file

    Returns:
        Path to extracted audio file
    """
    extractor = VideoExtractor()
    result = extractor.extract(video_path)

    if result["metadata"]["audio_extracted"]:
        return result["audio"]["audio_path"]

    return None
