"""
Image utility functions for preprocessing and manipulation.
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Union, List
import numpy as np
from PIL import Image
import base64

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImageUtils:
    """
    Utility class for image operations.
    """
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Image.Image:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path)
            logger.debug(f"Loaded image: {image_path} (size: {image.size})")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    @staticmethod
    def save_image(
        image: Image.Image,
        output_path: Union[str, Path],
        format: Optional[str] = None,
        quality: int = 95
    ):
        """
        Save image to file.
        
        Args:
            image: PIL Image object
            output_path: Output file path
            format: Image format (PNG, JPEG, etc.)
            quality: Quality for JPEG (1-100)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format is None:
            format = output_path.suffix[1:].upper()
            if format == "JPG":
                format = "JPEG"
        
        save_kwargs = {}
        if format == "JPEG":
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
            
            # Convert RGBA to RGB for JPEG
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
        
        image.save(output_path, format=format, **save_kwargs)
        logger.debug(f"Saved image to: {output_path}")
    
    @staticmethod
    def resize_image(
        image: Image.Image,
        max_dimension: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        maintain_aspect_ratio: bool = True
    ) -> Image.Image:
        """
        Resize image.
        
        Args:
            image: PIL Image object
            max_dimension: Maximum dimension (width or height)
            size: Exact size as (width, height)
            maintain_aspect_ratio: Maintain aspect ratio
            
        Returns:
            Resized image
        """
        if size:
            if maintain_aspect_ratio:
                image.thumbnail(size, Image.Resampling.LANCZOS)
            else:
                image = image.resize(size, Image.Resampling.LANCZOS)
        
        elif max_dimension:
            width, height = image.size
            if max(width, height) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {(width, height)} to {(new_width, new_height)}")
        
        return image
    
    @staticmethod
    def crop_image(
        image: Image.Image,
        box: Tuple[int, int, int, int]
    ) -> Image.Image:
        """
        Crop image to box.
        
        Args:
            image: PIL Image object
            box: (left, top, right, bottom)
            
        Returns:
            Cropped image
        """
        return image.crop(box)
    
    @staticmethod
    def image_to_bytes(
        image: Image.Image,
        format: str = "PNG"
    ) -> bytes:
        """
        Convert image to bytes.
        
        Args:
            image: PIL Image object
            format: Output format
            
        Returns:
            Image as bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    @staticmethod
    def bytes_to_image(image_bytes: bytes) -> Image.Image:
        """
        Convert bytes to image.
        
        Args:
            image_bytes: Image as bytes
            
        Returns:
            PIL Image object
        """
        return Image.open(io.BytesIO(image_bytes))
    
    @staticmethod
    def image_to_base64(
        image: Image.Image,
        format: str = "PNG"
    ) -> str:
        """
        Convert image to base64 string.
        
        Args:
            image: PIL Image object
            format: Output format
            
        Returns:
            Base64 encoded string
        """
        image_bytes = ImageUtils.image_to_bytes(image, format)
        return base64.b64encode(image_bytes).decode("utf-8")
    
    @staticmethod
    def base64_to_image(base64_string: str) -> Image.Image:
        """
        Convert base64 string to image.
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            PIL Image object
        """
        image_bytes = base64.b64decode(base64_string)
        return ImageUtils.bytes_to_image(image_bytes)
    
    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        """
        Convert image to RGB mode.
        
        Args:
            image: PIL Image object
            
        Returns:
            RGB image
        """
        if image.mode != "RGB":
            return image.convert("RGB")
        return image
    
    @staticmethod
    def get_image_dimensions(image: Union[Image.Image, str, Path]) -> Tuple[int, int]:
        """
        Get image dimensions.
        
        Args:
            image: PIL Image object or path to image
            
        Returns:
            (width, height)
        """
        if isinstance(image, (str, Path)):
            image = ImageUtils.load_image(image)
        
        return image.size
    
    @staticmethod
    def is_valid_image(
        image_path: Union[str, Path],
        min_size: Optional[Tuple[int, int]] = None,
        max_size_mb: Optional[float] = None
    ) -> bool:
        """
        Validate image file.
        
        Args:
            image_path: Path to image
            min_size: Minimum dimensions (width, height)
            max_size_mb: Maximum file size in MB
            
        Returns:
            True if valid, False otherwise
        """
        try:
            image_path = Path(image_path)
            
            # Check file exists
            if not image_path.exists():
                logger.warning(f"Image does not exist: {image_path}")
                return False
            
            # Check file size
            if max_size_mb:
                size_mb = image_path.stat().st_size / (1024 * 1024)
                if size_mb > max_size_mb:
                    logger.warning(f"Image too large: {size_mb:.2f}MB")
                    return False
            
            # Try to open image
            image = Image.open(image_path)
            
            # Check dimensions
            if min_size:
                width, height = image.size
                if width < min_size[0] or height < min_size[1]:
                    logger.warning(f"Image too small: {image.size}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Invalid image {image_path}: {e}")
            return False
    
    @staticmethod
    def create_thumbnail(
        image: Image.Image,
        size: Tuple[int, int] = (256, 256)
    ) -> Image.Image:
        """
        Create thumbnail of image.
        
        Args:
            image: PIL Image object
            size: Thumbnail size
            
        Returns:
            Thumbnail image
        """
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image array to [0, 1].
        
        Args:
            image: Image as numpy array
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """
        Denormalize image from [0, 1] to [0, 255].
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image
        """
        return (image * 255).astype(np.uint8)
    
    @staticmethod
    def image_to_numpy(image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to numpy array.
        
        Args:
            image: PIL Image object
            
        Returns:
            Numpy array
        """
        return np.array(image)
    
    @staticmethod
    def numpy_to_image(array: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image.
        
        Args:
            array: Numpy array
            
        Returns:
            PIL Image object
        """
        if array.dtype == np.float32 or array.dtype == np.float64:
            array = ImageUtils.denormalize_image(array)
        
        return Image.fromarray(array)
    
    @staticmethod
    def batch_resize_images(
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        max_dimension: int = 1024
    ) -> List[Path]:
        """
        Batch resize images.
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory
            max_dimension: Maximum dimension
            
        Returns:
            List of output paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        for image_path in image_paths:
            image_path = Path(image_path)
            image = ImageUtils.load_image(image_path)
            resized = ImageUtils.resize_image(image, max_dimension=max_dimension)
            
            output_path = output_dir / image_path.name
            ImageUtils.save_image(resized, output_path)
            output_paths.append(output_path)
        
        logger.info(f"Resized {len(output_paths)} images")
        return output_paths


# Convenience functions
load_image = ImageUtils.load_image
save_image = ImageUtils.save_image
resize_image = ImageUtils.resize_image
