"""
Configuration loader module.
Handles loading and merging YAML configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """
    Configuration loader for YAML files with environment variable support.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, DictConfig] = {}
        
        # Load environment variables
        load_dotenv()
        
    def load(self, config_name: str) -> DictConfig:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of config file (without .yaml extension)
            
        Returns:
            Configuration as OmegaConf DictConfig
            
        Example:
            >>> loader = ConfigLoader()
            >>> config = loader.load("config")
            >>> print(config.project.name)
        """
        if config_name in self._configs:
            logger.debug(f"Returning cached config: {config_name}")
            return self._configs[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading configuration from: {config_path}")
        
        # Load YAML file
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to OmegaConf
        config = OmegaConf.create(config_dict)
        
        # Resolve environment variables
        config = self._resolve_env_vars(config)
        
        # Cache config
        self._configs[config_name] = config
        
        logger.info(f"Successfully loaded config: {config_name}")
        return config
    
    def load_all(self) -> DictConfig:
        """
        Load all configuration files and merge them.
        
        Returns:
            Merged configuration
        """
        config_files = ["config", "model_config", "retrieval_config", "generation_config"]
        configs = []
        
        for config_name in config_files:
            try:
                config = self.load(config_name)
                configs.append(config)
            except FileNotFoundError:
                logger.warning(f"Config file not found, skipping: {config_name}")
        
        # Merge all configs
        merged_config = OmegaConf.merge(*configs)
        
        logger.info(f"Merged {len(configs)} configuration files")
        return merged_config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., "llm.temperature")
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> loader = ConfigLoader()
            >>> temp = loader.get("llm.temperature", default=0.7)
        """
        if not self._configs:
            self.load_all()
        
        merged_config = OmegaConf.merge(*self._configs.values())
        
        try:
            return OmegaConf.select(merged_config, key_path, default=default)
        except Exception as e:
            logger.warning(f"Error getting config key '{key_path}': {e}")
            return default
    
    def _resolve_env_vars(self, config: DictConfig) -> DictConfig:
        """
        Resolve environment variables in configuration.
        
        Args:
            config: Configuration to resolve
            
        Returns:
            Resolved configuration
        """
        # OmegaConf automatically resolves ${env:VAR_NAME}
        OmegaConf.resolve(config)
        return config
    
    def save(self, config: DictConfig, output_path: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            OmegaConf.save(config, f)
        
        logger.info(f"Saved configuration to: {output_path}")
    
    @staticmethod
    def from_dict(config_dict: Dict) -> DictConfig:
        """
        Create OmegaConf config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            OmegaConf DictConfig
        """
        return OmegaConf.create(config_dict)
    
    @staticmethod
    def to_dict(config: DictConfig) -> Dict:
        """
        Convert OmegaConf config to dictionary.
        
        Args:
            config: OmegaConf configuration
            
        Returns:
            Dictionary
        """
        return OmegaConf.to_container(config, resolve=True)


# Global config loader instance
_config_loader = None


def get_config_loader(config_dir: str = "configs") -> ConfigLoader:
    """
    Get global config loader instance.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader


def load_config(config_name: str = "config") -> DictConfig:
    """
    Convenience function to load a configuration file.
    
    Args:
        config_name: Name of config file (without .yaml extension)
        
    Returns:
        Configuration as DictConfig
    """
    loader = get_config_loader()
    return loader.load(config_name)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Convenience function to get a configuration value.
    
    Args:
        key_path: Dot-separated path to config value
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    loader = get_config_loader()
    return loader.get(key_path, default)
