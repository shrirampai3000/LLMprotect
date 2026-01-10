"""Setup script for Cryptographic Intent Binding package."""
from setuptools import setup, find_packages

setup(
    name="anti-llm",
    version="1.0.0",
    description="Cryptographic Intent Binding for Adversarial Manipulation Detection in Agentic AI Systems",
    author="Research Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.15.0",
        "cryptography>=41.0.0",
        "PyNaCl>=1.5.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "matplotlib>=3.8.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.21.0", "httpx>=0.25.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
