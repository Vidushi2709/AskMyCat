"""
Central configuration for EBM project paths and constants.
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA = DATA_DIR / "train.json"
TEST_DATA = DATA_DIR / "test.json"
DEV_DATA = DATA_DIR / "dev.json"

# ChromaDB paths
COLLECTIONS_DIR = PROJECT_ROOT / "collections"
CHROMA_PATH = COLLECTIONS_DIR / "ebm"
CHROMA_COLLECTION_NAME = "ebm_passages"

# Model checkpoint paths
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RANKING_SCORER_CKPT = CHECKPOINTS_DIR / "best_EBM_scorer.ckpt"  # Updated to new trained model
SCORER_CKPT = CHECKPOINTS_DIR / "scorer.ckpt"

# Cache directory
CACHE_DIR = PROJECT_ROOT / "cache"

# Model configuration
MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding parameters
CHUNK_SIZE = 512
OVERLAP = 128
EMBEDDING_BATCH_SIZE = 128

# Device configuration
DEVICE = "cuda" if os.environ.get("DEVICE", "cuda") == "cuda" else "cpu"

# Ensure critical directories exist
CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
