"""
Audio extractor module for the Multi-Modal RAG system.
Transcribes audio files using Whisper and extracts audio features.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import whisper
import librosa
import numpy as np
from pydub import AudioSegment
import torch

from src.utils.logger import get_logger
from src.utils.file_utils import FileUtils, ensure_dir
from src.utils.config_loader import get_config_value

logger = get_logger(__name__)


class AudioExtractor:
    """
    Extract content from audio files including transcription and features.
    """

    def __init__(
        self,
        whisper_model: str = "base",
        language: str = "en",
        extract_features: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize audio extractor.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            language: Language for transcription
            extract_features: Whether to extract audio features
            device: Device for model inference
        """
        self.whisper_model_size = whisper_model
        self.language = language
        self.extract_features = extract_features
        self.device = device

        # Initialize models
        self._whisper_model = None
        self._init_whisper()

        logger.info(f"AudioExtractor initialized with model={whisper_model}, language={language}")

    def _init_whisper(self):
        """Initialize Whisper model."""
        try:
            self._whisper_model = whisper.load_model(
                self.whisper_model_size,
                device=self.device
            )
            logger.info(f"Whisper model '{self.whisper_model_size}' loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self._whisper_model = None

    def extract(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract content from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary containing extracted content
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Extracting content from audio: {audio_path}")

        # Initialize result structure
        result = {
            "metadata": {
                "source": str(audio_path),
                "filename": audio_path.name,
                "duration": 0.0,
                "sample_rate": 0,
                "channels": 0,
                "format": audio_path.suffix[1:].lower()
            },
            "transcription": {
                "text": "",
                "segments": [],
                "language": self.language,
                "confidence": 0.0
            },
            "features": {}
        }

        try:
            # Load audio
            audio_data, sample_rate = self._load_audio(audio_path)
            result["metadata"].update({
                "duration": len(audio_data) / sample_rate,
                "sample_rate": sample_rate,
                "channels": 1 if audio_data.ndim == 1 else audio_data.shape[0]
            })

            # Transcribe audio
            if self._whisper_model:
                transcription = self._transcribe_audio(audio_path)
                result["transcription"] = transcription

            # Extract audio features
            if self.extract_features:
                features = self._extract_audio_features(audio_data, sample_rate)
                result["features"] = features

            logger.success(f"Successfully extracted content from {audio_path.name}")
            logger.info(f"Duration: {result['metadata']['duration']:.1f}s, Text length: {len(result['transcription']['text'])}")

        except Exception as e:
            logger.error(f"Error extracting audio {audio_path}: {e}")
            raise

        return result

    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa
            audio_data, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
            return audio_data, sample_rate

        except Exception as e:
            logger.warning(f"Librosa failed to load {audio_path}, trying pydub: {e}")

            # Fallback to pydub for unsupported formats
            try:
                audio = AudioSegment.from_file(str(audio_path))

                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples())
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)  # Convert to mono

                # Normalize to [-1, 1]
                samples = samples.astype(np.float32) / (2**15 if audio.sample_width == 2 else 2**31)

                return samples, audio.frame_rate

            except Exception as e2:
                logger.error(f"Both librosa and pydub failed to load audio: {e2}")
                raise

    def _transcribe_audio(self, audio_path: Path) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription results
        """
        transcription = {
            "text": "",
            "segments": [],
            "language": self.language,
            "confidence": 0.0
        }

        try:
            # Transcribe with Whisper
            result = self._whisper_model.transcribe(
                str(audio_path),
                language=self.language,
                verbose=False
            )

            transcription["text"] = result["text"].strip()
            transcription["language"] = result.get("language", self.language)

            # Process segments
            if "segments" in result:
                segments = []
                confidences = []

                for segment in result["segments"]:
                    segment_data = {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip(),
                        "confidence": segment.get("confidence", 0.0)
                    }
                    segments.append(segment_data)
                    confidences.append(segment_data["confidence"])

                transcription["segments"] = segments
                transcription["confidence"] = np.mean(confidences) if confidences else 0.0

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")

        return transcription

    def _extract_audio_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract audio features using librosa.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate

        Returns:
            Audio features
        """
        features = {}

        try:
            # MFCC (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            features["mfcc"] = mfccs.tolist()
            features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
            features["mfcc_std"] = np.std(mfccs, axis=1).tolist()

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            features["chroma"] = chroma.tolist()
            features["chroma_mean"] = np.mean(chroma, axis=1).tolist()

            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            features["spectral_centroid"] = np.mean(centroid).item()

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features["zero_crossing_rate"] = np.mean(zcr).item()

            # RMS energy
            rms = librosa.feature.rms(y=audio_data)
            features["rms_energy"] = np.mean(rms).item()

            # Tempo (BPM)
            tempo, _ = librosa.beat.tempo(y=audio_data, sr=sample_rate)
            features["tempo"] = tempo.item()

            logger.debug("Extracted audio features successfully")

        except Exception as e:
            logger.warning(f"Audio feature extraction failed: {e}")

        return features

    def extract_batch(self, audio_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Extract content from multiple audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            List of extraction results
        """
        results = []

        for audio_path in audio_paths:
            try:
                result = self.extract(audio_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract {audio_path}: {e}")
                # Add error result
                results.append({
                    "metadata": {"source": str(audio_path), "error": str(e)},
                    "transcription": {"text": "", "confidence": 0.0},
                    "features": {}
                })

        logger.info(f"Batch extracted {len(results)} audio files")
        return results

    def get_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """
        Get audio duration without full extraction.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            audio_data, sample_rate = self._load_audio(Path(audio_path))
            return len(audio_data) / sample_rate
        except Exception as e:
            logger.warning(f"Could not get duration for {audio_path}: {e}")
            return 0.0

    def convert_audio_format(
        self,
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        format: str = "wav"
    ):
        """
        Convert audio to different format.

        Args:
            audio_path: Input audio path
            output_path: Output audio path
            format: Output format
        """
        try:
            audio = AudioSegment.from_file(str(audio_path))
            audio.export(str(output_path), format=format)
            logger.info(f"Converted audio to {format}: {output_path}")
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")

    def save_transcription(self, result: Dict[str, Any], output_path: Union[str, Path]):
        """
        Save transcription to file.

        Args:
            result: Extraction result
            output_path: Output file path
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        # Save full transcription
        transcription_data = {
            "metadata": result["metadata"],
            "transcription": result["transcription"]
        }

        FileUtils.save_json(transcription_data, output_path)

        # Save plain text version
        text_file = output_path.with_suffix('.txt')
        FileUtils.write_text(result["transcription"]["text"], text_file)

        logger.info(f"Saved transcription to: {output_path}")

    def save_features(self, result: Dict[str, Any], output_path: Union[str, Path]):
        """
        Save audio features to file.

        Args:
            result: Extraction result
            output_path: Output file path
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        features_data = {
            "metadata": result["metadata"],
            "features": result["features"]
        }

        FileUtils.save_json(features_data, output_path)
        logger.info(f"Saved features to: {output_path}")


# Convenience functions
def extract_audio_content(audio_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to extract audio content.

    Args:
        audio_path: Path to audio file

    Returns:
        Extracted content
    """
    extractor = AudioExtractor()
    return extractor.extract(audio_path)


def transcribe_audio(audio_path: Union[str, Path]) -> str:
    """
    Transcribe audio to text.

    Args:
        audio_path: Path to audio file

    Returns:
        Transcribed text
    """
    result = extract_audio_content(audio_path)
    return result["transcription"]["text"]


def batch_transcribe_audio(audio_paths: List[Union[str, Path]]) -> List[str]:
    """
    Batch transcribe multiple audio files.

    Args:
        audio_paths: List of audio file paths

    Returns:
        List of transcribed texts
    """
    extractor = AudioExtractor()
    results = extractor.extract_batch(audio_paths)
    return [result["transcription"]["text"] for result in results]
