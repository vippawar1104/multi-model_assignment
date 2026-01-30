"""
FAISS vector store implementation for the Multi-Modal RAG system.
Provides efficient similarity search using FAISS indices.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json

from src.utils.logger import get_logger
from src.utils.config_loader import get_config_value
from src.utils.file_utils import FileUtils

logger = get_logger(__name__)

# Try importing FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    Supports multiple index types and metadata storage.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flatl2",
        metric: str = "l2",
        use_gpu: bool = False,
        nlist: int = 100
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('flatl2', 'flatip', 'ivfflat', 'hnsw')
            metric: Distance metric ('l2' or 'ip' for inner product)
            use_gpu: Whether to use GPU acceleration
            nlist: Number of clusters for IVF index
        """
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.metric = metric
        self.use_gpu = use_gpu
        self.nlist = nlist
        
        self.index = None
        self.metadata = []  # Store metadata for each vector
        self.id_to_index = {}  # Map chunk IDs to index positions
        
        # Initialize index
        self._initialize_index()
        
        logger.info(f"FAISSVectorStore initialized with dimension={dimension}, index_type={index_type}")

    def _initialize_index(self):
        """Initialize FAISS index."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - using simple numpy-based search")
            self.index = None
            self._vectors = []  # Fallback storage
            return
        
        try:
            if self.index_type == "flatl2":
                # Flat L2 index (exact search)
                self.index = faiss.IndexFlatL2(self.dimension)
                
            elif self.index_type == "flatip":
                # Flat inner product index
                self.index = faiss.IndexFlatIP(self.dimension)
                
            elif self.index_type == "ivfflat":
                # IVF Flat index (faster but approximate)
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                
            elif self.index_type == "hnsw":
                # HNSW index (fast approximate search)
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                
            else:
                logger.warning(f"Unknown index type: {self.index_type}, using FlatL2")
                self.index = faiss.IndexFlatL2(self.dimension)
            
            # GPU acceleration if requested
            if self.use_gpu and faiss.get_num_gpus() > 0:
                try:
                    self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                    logger.info("Using GPU acceleration for FAISS")
                except Exception as e:
                    logger.warning(f"GPU acceleration failed: {e}")
            
            logger.info(f"Initialized {self.index_type} index")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            self.index = None
            self._vectors = []

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add vectors to the index.

        Args:
            vectors: Numpy array of vectors (n_vectors, dimension)
            metadata: Optional list of metadata dictionaries
            ids: Optional list of chunk IDs
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        n_vectors = vectors.shape[0]
        
        # Validate dimension
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")
        
        # Convert to float32 if needed
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # Add to index
        if self.index is not None and FAISS_AVAILABLE:
            # Train index if needed (for IVF)
            if self.index_type == "ivfflat" and not self.index.is_trained:
                logger.info("Training IVF index...")
                self.index.train(vectors)
            
            # Add vectors
            self.index.add(vectors)
        else:
            # Fallback: store vectors directly
            if not hasattr(self, '_vectors'):
                self._vectors = []
            self._vectors.extend(vectors)
        
        # Store metadata
        current_index = len(self.metadata)
        for i in range(n_vectors):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            self.metadata.append(meta)
            
            # Map ID to index
            if ids and i < len(ids):
                self.id_to_index[ids[i]] = current_index + i
        
        logger.info(f"Added {n_vectors} vectors to index. Total: {len(self.metadata)}")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            Tuple of (indices, distances, metadata)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Convert to float32
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        
        # Search
        if self.index is not None and FAISS_AVAILABLE:
            distances, indices = self.index.search(query_vector, k)
            distances = distances[0].tolist()
            indices = indices[0].tolist()
        else:
            # Fallback: numpy-based search
            if not hasattr(self, '_vectors') or not self._vectors:
                return [], [], []
            
            vectors = np.array(self._vectors)
            # Compute L2 distances
            dists = np.linalg.norm(vectors - query_vector, axis=1)
            indices = np.argsort(dists)[:k].tolist()
            distances = dists[indices].tolist()
        
        # Get metadata
        results_metadata = []
        valid_indices = []
        valid_distances = []
        
        for idx, dist in zip(indices, distances):
            if idx >= 0 and idx < len(self.metadata):
                meta = self.metadata[idx]
                
                # Apply metadata filtering
                if filter_metadata:
                    if self._matches_filter(meta, filter_metadata):
                        results_metadata.append(meta)
                        valid_indices.append(idx)
                        valid_distances.append(dist)
                else:
                    results_metadata.append(meta)
                    valid_indices.append(idx)
                    valid_distances.append(dist)
        
        return valid_indices, valid_distances, results_metadata

    def search_by_id(
        self,
        chunk_id: str,
        k: int = 10
    ) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
        """
        Search for similar vectors by chunk ID.

        Args:
            chunk_id: ID of chunk to use as query
            k: Number of results to return

        Returns:
            Tuple of (indices, distances, metadata)
        """
        if chunk_id not in self.id_to_index:
            logger.warning(f"Chunk ID not found: {chunk_id}")
            return [], [], []
        
        idx = self.id_to_index[chunk_id]
        
        # Get vector
        if self.index is not None and FAISS_AVAILABLE:
            vector = self.index.reconstruct(int(idx))
        else:
            vector = np.array(self._vectors[idx])
        
        return self.search(vector, k)

    def _matches_filter(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def save(self, save_path: Union[str, Path]):
        """
        Save index and metadata to disk.

        Args:
            save_path: Directory to save to
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        if self.index is not None and FAISS_AVAILABLE:
            index_file = save_path / "faiss.index"
            faiss.write_index(self.index, str(index_file))
            logger.info(f"Saved FAISS index to {index_file}")
        else:
            # Save vectors as numpy array
            vectors_file = save_path / "vectors.npy"
            if hasattr(self, '_vectors') and self._vectors:
                np.save(vectors_file, np.array(self._vectors))
                logger.info(f"Saved vectors to {vectors_file}")
        
        # Save metadata
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'id_to_index': self.id_to_index,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
        logger.info(f"Saved metadata to {metadata_file}")

    def load(self, load_path: Union[str, Path]):
        """
        Load index and metadata from disk.

        Args:
            load_path: Directory to load from
        """
        load_path = Path(load_path)
        
        # Load index
        index_file = load_path / "faiss.index"
        if index_file.exists() and FAISS_AVAILABLE:
            self.index = faiss.read_index(str(index_file))
            logger.info(f"Loaded FAISS index from {index_file}")
        else:
            # Load vectors
            vectors_file = load_path / "vectors.npy"
            if vectors_file.exists():
                self._vectors = list(np.load(vectors_file))
                logger.info(f"Loaded vectors from {vectors_file}")
        
        # Load metadata
        metadata_file = load_path / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.id_to_index = data['id_to_index']
                self.dimension = data['dimension']
                self.index_type = data['index_type']
            logger.info(f"Loaded metadata from {metadata_file}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            'total_vectors': len(self.metadata),
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric
        }
        
        if self.index is not None and FAISS_AVAILABLE:
            stats['index_trained'] = self.index.is_trained if hasattr(self.index, 'is_trained') else True
            stats['index_total'] = self.index.ntotal
        
        return stats

    def clear(self):
        """Clear all vectors and metadata."""
        self._initialize_index()
        self.metadata = []
        self.id_to_index = {}
        logger.info("Cleared vector store")


def create_vector_store(
    dimension: int,
    index_type: str = "flatl2"
) -> FAISSVectorStore:
    """
    Convenience function to create a vector store.

    Args:
        dimension: Embedding dimension
        index_type: Type of FAISS index

    Returns:
        FAISSVectorStore instance
    """
    return FAISSVectorStore(dimension=dimension, index_type=index_type)
