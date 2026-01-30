"""
Multi-Modal RAG QA System
Setup configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multimodal-rag-qa",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Modal Retrieval Augmented Generation QA System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multimodal-rag-qa",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "isort>=5.13.0",
        ],
        "docs": [
            "mkdocs>=1.5.3",
            "mkdocs-material>=9.4.14",
            "mkdocstrings>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multimodal-rag=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
