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
VERIFICATION_SCORER_CKPT = CHECKPOINTS_DIR / "ranking_scorer.ckpt"  # Separate model for verification
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

# Thresholds and pipeline configuration for backend/retreiver.py

THRESHOLDS = {
    "embedding_dim": 128,  # Must match checkpoint (was 768)
	"cache_size_limit": 1 * 1024 ** 3,  # 1GB
	"cache_ttl_seconds": 7 * 24 * 3600,  # 7 days
	"text_chunk_size": 128,  # words per chunk
	"rerank_max_len": 256,
	"rerank_batch_size": 16,
	"similarity_threshold": 0.5,
	"sim_threshold": 0.5,
	"gate1": 0.5,
	"gate2": 0.4,  # Lowered from 0.5 to be less restrictive
	"gate3": 0.5,
	"min_strong_evidence": 2,  # Lowered from 3 to be less restrictive
	"min_coverage": 0.5,  # Lowered from 0.67 to be less restrictive
	"default_top_k": 10,
	"min_entailment_score": 0.7,
	"llm_temperature": 0.3,  # Default temperature for LLM calls
}

# Ensure critical directories exist
CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
