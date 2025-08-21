#!/usr/bin/env python3
"""Setup script for Ultra AI Project."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-r')]

install_requires = read_requirements('requirements.txt')
dev_requires = read_requirements('requirements-dev.txt')

setup(
    name="ultra-ai-project",
    version="1.0.0",
    author="Ultra AI Team",
    author_email="team@ultraai.dev",
    description="Advanced AI system with multi-agent capabilities and comprehensive tooling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ultraai/ultra-ai-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "gpu": [
            "torch[cuda]",
            "tensorflow-gpu",
        ],
        "audio": [
            "whisper",
            "soundfile",
            "pyaudio",
        ],
        "vision": [
            "mediapipe",
            "face-recognition",
            "dlib",
        ],
    },
    entry_points={
        "console_scripts": [
            "ultra-ai=main:main",
            "ultra-cli=ui.cli_interface:main",
            "ultra-server=api.routes:start_server",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
        "config": ["*.yaml", "*.yml"],
        "docs": ["*.md"],
    },
    zip_safe=False,
    keywords="ai artificial-intelligence machine-learning agents nlp computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/ultraai/ultra-ai-project/issues",
        "Source": "https://github.com/ultraai/ultra-ai-project",
        "Documentation": "https://ultra-ai-project.readthedocs.io/",
    },
)
