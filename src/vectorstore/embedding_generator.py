"""
Embedding generator for the Multi-Modal RAG system.
Handles text and image embedding generation using various models.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import hashlib
import json

from src.utils.logger import get_logger
from src.utils.config_loader import get_config_value
from src.utils.file_utils import FileUtils

logger = get_logger(__name__)

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Try importing CLIP for image embeddings
try:
    from PIL import Image
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available. Install with: pip install transformers torch pillow")


class EmbeddingGenerator:
    """
    Generate embeddings for text and images using various models.
    Supports both text-only and multi-modal embeddings.
    """

    def __init__(
        self,
        text_model_name: Optional[str] = None,
        image_model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize embedding generator.

        Args:
            text_model_name: Name of text embedding model
            image_model_name: Name of image embedding model
            device: Device to use ('cpu', 'cuda', 'mps')
            cache_dir: Directory to cache embeddings
            batch_size: Batch size for processing
        """
        self.text_model_name = text_model_name or get_config_value(
            "embeddings.text_model", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.image_model_name = image_model_name or get_config_value(
            "embeddings.image_model",
            "openai/clip-vit-base-patch32"
        )
        self.device = device or get_config_value("embeddings.device", "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/embeddings")
        self.batch_size = batch_size
        
        self.text_model = None
        self.image_model = None
        self.image_processor = None
        
        # Initialize models
        self._initialize_models()
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"EmbeddingGenerator initialized with text_model={self.text_model_name}, device={self.device}")

    def _initialize_models(self):
        """Initialize embedding models."""
        # Initialize text model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.text_model = SentenceTransformer(self.text_model_name, device=self.device)
                logger.info(f"Loaded text embedding model: {self.text_model_name}")
            except Exception as e:
                logger.error(f"Failed to load text model: {e}")
                self.text_model = None
        else:
            logger.warning("Text embeddings not available - sentence-transformers not installed")
        
        # Initialize image model (CLIP)
        if CLIP_AVAILABLE:
            try:
                import torch
                self.image_model = CLIPModel.from_pretrained(self.image_model_name)
                self.image_processor = CLIPProcessor.from_pretrained(self.image_model_name)
                
                # Move to device
                if self.device == "cuda" and torch.cuda.is_available():
                    self.image_model = self.image_model.to("cuda")
                elif self.device == "mps" and torch.backends.mps.is_available():
                    self.image_model = self.image_model.to("mps")
                
                logger.info(f"Loaded image embedding model: {self.image_model_name}")
            except Exception as e:
                logger.error(f"Failed to load image model: {e}")
                self.image_model = None
                self.image_processor = None
        else:
            logger.warning("Image embeddings not available - transformers/torch not installed")

    def generate_text_embeddings(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        if not self.text_model:
            logger.error("Text model not available")
            # Return dummy embeddings
            if isinstance(texts, str):
                return np.random.randn(384).astype(np.float32)
            else:
                return np.random.randn(len(texts), 384).astype(np.float32)
        
        try:
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]
                single_text = True
            else:
                single_text = False
            
            # Generate embeddings
            embeddings = self.text_model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                batch_size=self.batch_size
            )
            
            # Return single embedding if input was single text
            if single_text:
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            # Return dummy embeddings on error
            if isinstance(texts, str):
                return np.random.randn(384).astype(np.float32)
            else:
                return np.random.randn(len(texts), 384).astype(np.float32)

    def generate_image_embeddings(
        self,
        images: Union[str, Path, List[Union[str, Path]]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for image(s).

        Args:
            images: Single image path or list of image paths
            normalize: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if not self.image_model or not self.image_processor:
            logger.error("Image model not available")
            # Return dummy embeddings
            if isinstance(images, (str, Path)):
                return np.random.randn(512).astype(np.float32)
            else:
                return np.random.randn(len(images), 512).astype(np.float32)
        
        try:
            import torch
            from PIL import Image
            
            # Convert single image to list
            if isinstance(images, (str, Path)):
                images = [images]
                single_image = True
            else:
                single_image = False
            
            # Load images
            pil_images = []
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    pil_images.append(img)
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {e}")
                    # Use dummy image
                    pil_images.append(Image.new("RGB", (224, 224)))
            
            # Process images
            inputs = self.image_processor(images=pil_images, return_tensors="pt", padding=True)
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            elif self.device == "mps" and torch.backends.mps.is_available():
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.image_model.get_image_features(**inputs)
            
            # Convert to numpy
            embeddings = outputs.cpu().numpy()
            
            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)
            
            # Return single embedding if input was single image
            if single_image:
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating image embeddings: {e}")
            # Return dummy embeddings on error
            if isinstance(images, (str, Path)):
                return np.random.randn(512).astype(np.float32)
            else:
                return np.random.randn(len(images), 512).astype(np.float32)

    def generate_hybrid_embeddings(
        self,
        text: str,
        image_path: Optional[Union[str, Path]] = None,
        text_weight: float = 0.7,
        image_weight: float = 0.3
    ) -> np.ndarray:
        """
        Generate hybrid embeddings combining text and image.

        Args:
            text: Text to embed
            image_path: Optional image path
            text_weight: Weight for text embedding
            image_weight: Weight for image embedding

        Returns:
            Combined embedding
        """
        embeddings = []
        weights = []
        
        # Generate text embedding
        if text:
            text_emb = self.generate_text_embeddings(text)
            embeddings.append(text_emb)
            weights.append(text_weight)
        
        # Generate image embedding
        if image_path:
            image_emb = self.generate_image_embeddings(image_path)
            embeddings.append(image_emb)
            weights.append(image_weight)
        
        if not embeddings:
            logger.warning("No valid embeddings generated")
            return np.random.randn(384).astype(np.float32)
        
        # Pad embeddings to same size
        max_dim = max(emb.shape[0] for emb in embeddings)
        padded_embeddings = []
        for emb in embeddings:
            if emb.shape[0] < max_dim:
                padding = np.zeros(max_dim - emb.shape[0])
                emb = np.concatenate([emb, padding])
            padded_embeddings.append(emb)
        
        # Weighted combination
        combined = np.zeros(max_dim)
        for emb, weight in zip(padded_embeddings, weights):
            combined += emb * weight
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined

    def generate_chunk_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of chunk dictionaries
            use_cache: Whether to use cached embeddings

        Returns:
            List of chunks with embeddings added
        """
        results = []
        texts = []
        cache_keys = []
        
        for chunk in chunks:
            # Generate cache key
            chunk_text = chunk.get("text", "")
            cache_key = self._generate_cache_key(chunk_text)
            cache_keys.append(cache_key)
            
            # Check cache
            if use_cache:
                cached_emb = self._load_from_cache(cache_key)
                if cached_emb is not None:
                    chunk["embedding"] = cached_emb
                    results.append(chunk)
                    continue
            
            texts.append(chunk_text)
        
        # Generate embeddings for uncached texts
        if texts:
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings = self.generate_text_embeddings(texts, show_progress=True)
            
            # Add embeddings to chunks
            emb_idx = 0
            for i, chunk in enumerate(chunks):
                if "embedding" not in chunk:
                    chunk["embedding"] = embeddings[emb_idx]
                    
                    # Cache embedding
                    if use_cache:
                        self._save_to_cache(cache_keys[i], embeddings[emb_idx])
                    
                    emb_idx += 1
                    results.append(chunk)
        
        logger.info(f"Generated embeddings for {len(results)} chunks")
        return results

    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache."""
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"Error loading from cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache."""
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def get_embedding_dim(self, modality: str = "text") -> int:
        """
        Get embedding dimension for a modality.

        Args:
            modality: 'text' or 'image'

        Returns:
            Embedding dimension
        """
        if modality == "text":
            if self.text_model:
                return self.text_model.get_sentence_embedding_dimension()
            return 384  # Default dimension
        elif modality == "image":
            if self.image_model:
                return self.image_model.config.projection_dim
            return 512  # Default CLIP dimension
        else:
            raise ValueError(f"Unknown modality: {modality}")


def generate_text_embedding(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to generate text embedding.

    Args:
        text: Text to embed
        model_name: Optional model name

    Returns:
        Text embedding
    """
    generator = EmbeddingGenerator(text_model_name=model_name)
    return generator.generate_text_embeddings(text)


def generate_image_embedding(image_path: Union[str, Path], model_name: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to generate image embedding.

    Args:
        image_path: Path to image
        model_name: Optional model name

    Returns:
        Image embedding
    """
    generator = EmbeddingGenerator(image_model_name=model_name)
    return generator.generate_image_embeddings(image_path)
