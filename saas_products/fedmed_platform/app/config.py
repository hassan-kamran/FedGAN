"""
Application configuration management using Pydantic Settings.
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings and configuration."""

    # Application
    app_name: str = "FedMed Platform"
    app_version: str = "1.0.0"
    debug: bool = False

    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Database
    database_url: str
    database_url_async: Optional[str] = None

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # CORS
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # File Upload
    max_upload_size: int = 104857600  # 100MB
    upload_dir: str = "./uploads"

    # Model Storage
    model_storage_path: str = "./models"
    checkpoint_dir: str = "./checkpoints"

    # Federated Learning
    max_clients: int = 100
    default_rounds: int = 10
    default_local_epochs: int = 5
    default_batch_size: int = 16

    # Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: str = "noreply@fedmed.ai"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/fedmed.log"

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
