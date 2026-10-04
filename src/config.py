# src/config.py
import os
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
ANNOTATED_DATA_DIR = DATA_DIR / "annotated"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, CLEANED_DATA_DIR, ANNOTATED_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File paths
DOMAIN_REGISTRY_PATH = DATA_DIR / "domain_registry.csv"

# Model settings
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQUENCE_LENGTH = 512