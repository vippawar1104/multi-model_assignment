"""
Tests for utility modules.
"""

import pytest
from pathlib import Path
import tempfile
import json

from src.utils.logger import get_logger
from src.utils.file_utils import FileUtils
from src.utils.config_loader import ConfigLoader


class TestLogger:
    """Test logger functionality."""
    
    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger(__name__)
        assert logger is not None
        
    def test_logger_info(self):
        """Test logging info message."""
        logger = get_logger(__name__)
        logger.info("Test info message")


class TestFileUtils:
    """Test file utilities."""
    
    def test_ensure_dir(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test" / "nested" / "dir"
            result = FileUtils.ensure_dir(test_dir)
            assert result.exists()
            assert result.is_dir()
    
    def test_save_and_load_json(self):
        """Test JSON save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            test_data = {"key": "value", "number": 42}
            
            FileUtils.save_json(test_data, test_file)
            loaded_data = FileUtils.load_json(test_file)
            
            assert loaded_data == test_data
    
    def test_write_and_read_text(self):
        """Test text write and read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_content = "Hello, World!"
            
            FileUtils.write_text(test_content, test_file)
            loaded_content = FileUtils.read_text(test_file)
            
            assert loaded_content == test_content
    
    def test_get_file_extension(self):
        """Test getting file extension."""
        assert FileUtils.get_file_extension("test.pdf") == ".pdf"
        assert FileUtils.get_file_extension("test.PDF") == ".pdf"
        assert FileUtils.get_file_extension("test.tar.gz") == ".gz"
    
    def test_list_files(self):
        """Test listing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test files
            (tmpdir / "test1.txt").touch()
            (tmpdir / "test2.txt").touch()
            (tmpdir / "test.pdf").touch()
            
            # List all files
            all_files = FileUtils.list_files(tmpdir)
            assert len(all_files) == 3
            
            # List only txt files
            txt_files = FileUtils.list_files(tmpdir, extensions=[".txt"])
            assert len(txt_files) == 2


class TestConfigLoader:
    """Test configuration loader."""
    
    def test_config_loader_init(self):
        """Test config loader initialization."""
        loader = ConfigLoader(config_dir="configs")
        assert loader.config_dir.name == "configs"
    
    def test_load_config(self):
        """Test loading configuration."""
        loader = ConfigLoader(config_dir="configs")
        
        try:
            config = loader.load("config")
            assert config is not None
            assert "project" in config
        except FileNotFoundError:
            pytest.skip("Config file not found")
    
    def test_from_dict(self):
        """Test creating config from dict."""
        test_dict = {"key": "value", "nested": {"key2": "value2"}}
        config = ConfigLoader.from_dict(test_dict)
        
        assert config.key == "value"
        assert config.nested.key2 == "value2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
