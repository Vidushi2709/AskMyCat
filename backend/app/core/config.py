from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "EBM RAG API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Evidence-Based Medicine Retrieval-Augmented Generation System"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:3001", "http://localhost:8501", "http://localhost:8502"]
    
    # Model Settings
    DEVICE: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    MODEL_PATH: str = "../checkpoints/best_EBM_scorer.ckpt"
    CHROMA_PATH: str = "../data/chroma"
    
    # Cache Settings
    ENABLE_CACHE: bool = True
    CACHE_DIR: str = "../cache"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # OpenRouter API
    OPENROUTER_API_KEY: Optional[str] = os.getenv("API_KEY")
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"  # Allow extra fields from .env

settings = Settings()
