"""
Core configuration settings
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./retail_analytics.db"
    
    # Data Settings
    DATA_PATH: str = "data"
    RAW_DATA_PATH: str = "data/raw"
    PROCESSED_DATA_PATH: str = "data/processed"
    MODELS_PATH: str = "models"
    
    # ML Model Settings
    MODEL_RETRAIN_DAYS: int = 30
    MODEL_PERFORMANCE_THRESHOLD: float = 0.8
    
    # Monitoring Settings
    PROMETHEUS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    # Business Settings
    HIGH_VALUE_CUSTOMER_THRESHOLD: float = 1000.0
    RFM_QUANTILES: int = 5
    SEASONAL_ANALYSIS_MONTHS: int = 12
    
    class Config:
        env_file = ".env"

settings = Settings()
