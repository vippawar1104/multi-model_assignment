"""
Text chunker module for the Multi-Modal RAG system.
Handles semantic text chunking for optimal retrieval performance.
"""

import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config_loader import get_config_value
from src.utils.file_utils import FileUtils

logger = get_logger(__name__)


class TextChunker:
    """
    Advanced text chunking for semantic segmentation and retrieval optimization.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        preserve_metadata: bool = True
    ):
        """
        Initialize text chunker.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            separators: List of separators for splitting (in order of preference)
            preserve_metadata: Whether to preserve source metadata
        """
        self.chunk_size = chunk_size or get_config_value("preprocessing.text.chunk_size", 512)
        self.chunk_overlap = chunk_overlap or get_config_value("preprocessing.text.chunk_overlap", 50)
        self.separators = separators or get_config_value("preprocessing.text.separators", ["\n\n", "\n", ". ", " ", ""])
        self.preserve_metadata = preserve_metadata

        logger.info(f"TextChunker initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into semantically meaningful segments.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to preserve
            source_id: Optional source identifier

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunk_text")
            return []

        try:
            # Simple chunking implementation
            chunks = self._simple_chunk_text(text)

            # Convert to our format
            chunk_list = []
            for i, chunk_text in enumerate(chunks):
                chunk_dict = {
                    "id": f"{source_id}_chunk_{i}" if source_id else f"chunk_{i}",
                    "text": chunk_text,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_length": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "metadata": metadata or {}
                }

                # Add position information
                if i > 0:
                    chunk_dict["prev_chunk_id"] = f"{source_id}_chunk_{i-1}" if source_id else f"chunk_{i-1}"
                if i < len(chunks) - 1:
                    chunk_dict["next_chunk_id"] = f"{source_id}_chunk_{i+1}" if source_id else f"chunk_{i+1}"

                chunk_list.append(chunk_dict)

            logger.info(f"Chunked text into {len(chunk_list)} chunks")
            return chunk_list

        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            # Fallback: return single chunk
            return [{
                "id": f"{source_id}_chunk_0" if source_id else "chunk_0",
                "text": text,
                "chunk_index": 0,
                "total_chunks": 1,
                "char_length": len(text),
                "word_count": len(text.split()),
                "metadata": metadata or {}
            }]

    def _simple_chunk_text(self, text: str) -> List[str]:
        """
        Simple text chunking implementation.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to find a good break point
            chunk = text[start:end]

            # Look for sentence endings
            sentence_end = -1
            for sep in self.separators:
                pos = chunk.rfind(sep)
                if pos > 0 and (sentence_end == -1 or pos > sentence_end):
                    sentence_end = pos + len(sep)

            if sentence_end > 0:
                end = start + sentence_end
                chunk = text[start:end]
            else:
                # Force break at word boundary
                last_space = chunk.rfind(" ")
                if last_space > 0:
                    end = start + last_space + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())

            # Move start position with overlap
            start = end - self.chunk_overlap
            if start < 0:
                start = end

        return chunks

    def chunk_file(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text from a file.

        Args:
            file_path: Path to text file
            encoding: File encoding
            metadata: Optional metadata

        Returns:
            List of chunk dictionaries
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read file
            text = FileUtils.read_text_file(file_path, encoding=encoding)

            # Prepare metadata
            file_metadata = metadata or {}
            file_metadata.update({
                "source_file": str(file_path),
                "filename": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix
            })

            # Chunk the text
            return self.chunk_text(text, file_metadata, source_id=file_path.stem)

        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            raise

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        metadata_key: str = "metadata"
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries
            text_key: Key for text content
            metadata_key: Key for metadata

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            text = doc.get(text_key, "")
            metadata = doc.get(metadata_key, {})

            # Add document-level metadata
            doc_metadata = metadata.copy()
            doc_metadata["document_index"] = len(all_chunks)

            chunks = self.chunk_text(text, doc_metadata)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks

    def merge_small_chunks(
        self,
        chunks: List[Dict[str, Any]],
        min_chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Merge chunks that are smaller than minimum size.

        Args:
            chunks: List of chunk dictionaries
            min_chunk_size: Minimum chunk size (defaults to chunk_size/4)

        Returns:
            List of merged chunks
        """
        if not chunks:
            return chunks

        min_size = min_chunk_size or (self.chunk_size // 4)
        merged_chunks = []
        current_chunk = None

        for chunk in chunks:
            chunk_text = chunk["text"]

            if current_chunk is None:
                current_chunk = chunk.copy()
            elif len(current_chunk["text"]) < min_size:
                # Merge with previous chunk
                current_chunk["text"] += " " + chunk_text
                current_chunk["char_length"] = len(current_chunk["text"])
                current_chunk["word_count"] = len(current_chunk["text"].split())

                # Update metadata
                if "merged_chunks" not in current_chunk["metadata"]:
                    current_chunk["metadata"]["merged_chunks"] = []
                current_chunk["metadata"]["merged_chunks"].append(chunk["id"])

            else:
                # Current chunk is big enough, save it and start new one
                merged_chunks.append(current_chunk)
                current_chunk = chunk.copy()

        # Add the last chunk
        if current_chunk:
            merged_chunks.append(current_chunk)

        # Update chunk indices
        for i, chunk in enumerate(merged_chunks):
            chunk["chunk_index"] = i
            chunk["total_chunks"] = len(merged_chunks)
            chunk["id"] = re.sub(r'_chunk_\d+', f'_chunk_{i}', chunk["id"])

        logger.info(f"Merged small chunks: {len(chunks)} -> {len(merged_chunks)}")
        return merged_chunks

    def filter_chunks(
        self,
        chunks: List[Dict[str, Any]],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        remove_empty: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Filter chunks based on length criteria.

        Args:
            chunks: List of chunk dictionaries
            min_length: Minimum character length
            max_length: Maximum character length
            remove_empty: Whether to remove empty chunks

        Returns:
            Filtered list of chunks
        """
        filtered = []

        for chunk in chunks:
            text = chunk["text"].strip()
            length = len(text)

            # Remove empty chunks
            if remove_empty and not text:
                continue

            # Check length constraints
            if min_length and length < min_length:
                continue
            if max_length and length > max_length:
                continue

            filtered.append(chunk)

        logger.info(f"Filtered chunks: {len(chunks)} -> {len(filtered)}")
        return filtered

    def save_chunks(
        self,
        chunks: List[Dict[str, Any]],
        output_path: Union[str, Path],
        format: str = "json"
    ):
        """
        Save chunks to file.

        Args:
            chunks: List of chunk dictionaries
            output_path: Output file path
            format: Output format (json, jsonl)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            FileUtils.save_json({"chunks": chunks}, output_path)
        elif format == "jsonl":
            # Save as JSON Lines
            lines = [FileUtils.to_json(chunk) for chunk in chunks]
            FileUtils.write_text_file("\n".join(lines), output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(
        self,
        input_path: Union[str, Path],
        format: str = "json"
    ) -> List[Dict[str, Any]]:
        """
        Load chunks from file.

        Args:
            input_path: Input file path
            format: Input format (json, jsonl)

        Returns:
            List of chunk dictionaries
        """
        input_path = Path(input_path)

        if format == "json":
            data = FileUtils.load_json(input_path)
            return data.get("chunks", [])
        elif format == "jsonl":
            # Load JSON Lines
            content = FileUtils.read_text_file(input_path)
            lines = content.strip().split("\n")
            return [FileUtils.from_json(line) for line in lines if line.strip()]
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience functions
def chunk_text_simple(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Simple text chunking function.

    Args:
        text: Input text
        chunk_size: Maximum chunk size
        overlap: Chunk overlap

    Returns:
        List of text chunks
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = chunker.chunk_text(text)
    return [chunk["text"] for chunk in chunks]


def chunk_file_simple(file_path: Union[str, Path], chunk_size: int = 512) -> List[str]:
    """
    Simple file chunking function.

    Args:
        file_path: Path to text file
        chunk_size: Maximum chunk size

    Returns:
        List of text chunks
    """
    chunker = TextChunker(chunk_size=chunk_size)
    chunks = chunker.chunk_file(file_path)
    return [chunk["text"] for chunk in chunks]
