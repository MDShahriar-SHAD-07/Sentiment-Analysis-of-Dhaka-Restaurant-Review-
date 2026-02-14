"""
Configuration Management for FastAPI Application
================================================
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Info
    APP_NAME: str = "Restaurant Intelligence API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI-powered restaurant recommendation system based on sentiment analysis"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True  # Set to False in production
    
    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Data Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DATA_PATH: Path = DATA_DIR / "processed_reviews.csv"
    VECTOR_DB_PATH: Path = BASE_DIR / "restaurant_vector_db"
    
    # Model Configuration
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    DEVICE: str = "cuda"  # or "cpu"
    
    # RAG Configuration
    RETRIEVAL_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Recommendation Settings
    MIN_REVIEWS_DEFAULT: int = 3
    MAX_RECOMMENDATIONS: int = 20
    
    # Aspect Keywords
    ASPECTS: dict = {
        "food": ["food", "meal", "dish", "cuisine", "taste", "flavor", "delicious", 
                 "খাবার", "স্বাদ", "খাদ্য", "রান্না"],
        "service": ["service", "staff", "waiter", "waitress", "server", "manager",
                   "সার্ভিস", "কর্মী", "সেবা"],
        "price": ["price", "cost", "expensive", "cheap", "affordable", "value", "budget",
                 "দাম", "মূল্য", "খরচ"],
        "ambience": ["ambience", "atmosphere", "environment", "decor", "vibe", "setting",
                    "পরিবেশ", "আবহাওয়া"],
        "cleanliness": ["clean", "hygiene", "sanitary", "dirty", "neat",
                       "পরিষ্কার", "স্বাস্থ্যকর"]
    }
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Session Management
    SESSION_TIMEOUT_MINUTES: int = 30
    MAX_CONVERSATION_HISTORY: int = 10
    
    # Rate Limiting (optional)
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Create data directory if it doesn't exist
def ensure_directories():
    """Ensure required directories exist"""
    settings = get_settings()
    settings.DATA_DIR.mkdir(exist_ok=True)
    settings.VECTOR_DB_PATH.mkdir(exist_ok=True, parents=True)
