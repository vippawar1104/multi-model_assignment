"""
File utility functions for the Multi-Modal RAG system.
Handles file operations, path management, and I/O.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Optional, Union, Any, Dict
import hashlib
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileUtils:
    """
    Utility class for file operations.
    """
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
        """
        Calculate file hash.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha256, etc.)
            
        Returns:
            File hash as hex string
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size
    
    @staticmethod
    def get_file_extension(file_path: Union[str, Path]) -> str:
        """
        Get file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            File extension (lowercase, with dot)
        """
        return Path(file_path).suffix.lower()
    
    @staticmethod
    def list_files(
        directory: Union[str, Path],
        extensions: Optional[List[str]] = None,
        recursive: bool = False
    ) -> List[Path]:
        """
        List files in directory with optional filtering.
        
        Args:
            directory: Directory to search
            extensions: List of extensions to filter (e.g., ['.pdf', '.txt'])
            recursive: Search recursively
            
        Returns:
            List of file paths
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        pattern = "**/*" if recursive else "*"
        files = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                if extensions is None or file_path.suffix.lower() in extensions:
                    files.append(file_path)
        
        logger.info(f"Found {len(files)} files in {directory}")
        return files
    
    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
        """
        Copy file from source to destination.
        
        Args:
            src: Source file path
            dst: Destination file path
            
        Returns:
            Destination path
        """
        src = Path(src)
        dst = Path(dst)
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        
        logger.debug(f"Copied file: {src} -> {dst}")
        return dst
    
    @staticmethod
    def move_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
        """
        Move file from source to destination.
        
        Args:
            src: Source file path
            dst: Destination file path
            
        Returns:
            Destination path
        """
        src = Path(src)
        dst = Path(dst)
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        
        logger.debug(f"Moved file: {src} -> {dst}")
        return dst
    
    @staticmethod
    def delete_file(file_path: Union[str, Path]) -> bool:
        """
        Delete file if it exists.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if deleted, False if file didn't exist
        """
        file_path = Path(file_path)
        
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted file: {file_path}")
            return True
        
        return False
    
    @staticmethod
    def save_json(data: Any, file_path: Union[str, Path], indent: int = 2):
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.debug(f"Saved JSON to: {file_path}")
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Any:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON from: {file_path}")
        return data
    
    @staticmethod
    def read_text(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
        """
        Read text from file.
        
        Args:
            file_path: Path to text file
            encoding: File encoding
            
        Returns:
            File content as string
        """
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        
        return content
    
    @staticmethod
    def write_text(
        content: str,
        file_path: Union[str, Path],
        encoding: str = "utf-8"
    ):
        """
        Write text to file.
        
        Args:
            content: Text content
            file_path: Output file path
            encoding: File encoding
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
        
        logger.debug(f"Wrote text to: {file_path}")
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp as string.
        
        Returns:
            Timestamp in format: YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def create_unique_filename(
        base_name: str,
        extension: str,
        directory: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Create unique filename by adding timestamp.
        
        Args:
            base_name: Base filename
            extension: File extension (with or without dot)
            directory: Optional directory path
            
        Returns:
            Unique file path
        """
        if not extension.startswith("."):
            extension = f".{extension}"
        
        timestamp = FileUtils.get_timestamp()
        filename = f"{base_name}_{timestamp}{extension}"
        
        if directory:
            return Path(directory) / filename
        
        return Path(filename)
    
    @staticmethod
    def clean_directory(
        directory: Union[str, Path],
        keep_subdirs: bool = True
    ):
        """
        Clean all files in directory.
        
        Args:
            directory: Directory to clean
            keep_subdirs: Keep subdirectories
        """
        directory = Path(directory)
        
        if not directory.exists():
            return
        
        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir() and not keep_subdirs:
                shutil.rmtree(item)
        
        logger.info(f"Cleaned directory: {directory}")
    
    @staticmethod
    def get_relative_path(
        path: Union[str, Path],
        base: Union[str, Path]
    ) -> Path:
        """
        Get relative path from base.
        
        Args:
            path: Absolute path
            base: Base path
            
        Returns:
            Relative path
        """
        return Path(path).relative_to(base)
    
    @staticmethod
    def is_valid_file(
        file_path: Union[str, Path],
        extensions: Optional[List[str]] = None,
        max_size_mb: Optional[float] = None
    ) -> bool:
        """
        Validate file exists, has correct extension, and size.
        
        Args:
            file_path: Path to file
            extensions: Allowed extensions
            max_size_mb: Maximum file size in MB
            
        Returns:
            True if valid, False otherwise
        """
        file_path = Path(file_path)
        
        # Check exists
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return False
        
        # Check extension
        if extensions and file_path.suffix.lower() not in extensions:
            logger.warning(f"Invalid file extension: {file_path.suffix}")
            return False
        
        # Check size
        if max_size_mb:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                logger.warning(f"File too large: {size_mb:.2f}MB > {max_size_mb}MB")
                return False
        
        return True


# Convenience functions
ensure_dir = FileUtils.ensure_dir
save_json = FileUtils.save_json
load_json = FileUtils.load_json
read_text = FileUtils.read_text
write_text = FileUtils.write_text
