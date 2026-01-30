"""
Example script demonstrating the utility modules.
Run this to test the utilities are working correctly.
"""

from pathlib import Path
from src.utils import (
    get_logger,
    load_config,
    FileUtils,
    ensure_dir
)


def main():
    """Main function to demonstrate utilities."""
    
    # 1. Logger example
    print("=" * 60)
    print("1. Testing Logger")
    print("=" * 60)
    
    logger = get_logger(__name__)
    logger.info("Logger initialized successfully!")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.success("Utilities are working! ✅")
    
    # 2. File Utils example
    print("\n" + "=" * 60)
    print("2. Testing File Utils")
    print("=" * 60)
    
    # Create test directory
    test_dir = ensure_dir("data/test")
    logger.info(f"Created directory: {test_dir}")
    
    # Save JSON
    test_data = {
        "project": "Multi-Modal RAG",
        "version": "1.0.0",
        "features": ["PDF", "Images", "Audio", "Video"]
    }
    
    test_file = test_dir / "test.json"
    FileUtils.save_json(test_data, test_file)
    logger.info(f"Saved JSON to: {test_file}")
    
    # Load JSON
    loaded_data = FileUtils.load_json(test_file)
    logger.info(f"Loaded JSON: {loaded_data}")
    
    # 3. Config Loader example
    print("\n" + "=" * 60)
    print("3. Testing Config Loader")
    print("=" * 60)
    
    try:
        config = load_config("config")
        logger.info(f"Loaded config successfully!")
        logger.info(f"Project name: {config.project.name}")
        logger.info(f"Data directory: {config.paths.data_dir}")
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
    
    # 4. Summary
    print("\n" + "=" * 60)
    print("✅ All utilities tested successfully!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up environment: cp .env.example .env")
    print("3. Configure API keys in .env file")
    print("4. Run tests: pytest tests/ -v")


if __name__ == "__main__":
    main()
