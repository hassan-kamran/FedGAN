"""
MedSynth Platform - Core Configuration
Centralized settings management for the application
"""
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator, PostgresDsn


class Settings(BaseSettings):
    """Application settings and configuration"""

    # Application
    APP_NAME: str = "MedSynth Platform"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production-use-openssl-rand-hex-32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost",
    ]

    # Database
    POSTGRES_SERVER: str = "postgres"
    POSTGRES_USER: str = "medsynth"
    POSTGRES_PASSWORD: str = "medsynth_password"
    POSTGRES_DB: str = "medsynth_db"
    POSTGRES_PORT: str = "5432"
    DATABASE_URL: Optional[PostgresDsn] = None

    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return f"postgresql://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}:{values.get('POSTGRES_PORT')}/{values.get('POSTGRES_DB')}"

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_URL: Optional[str] = None

    @validator("REDIS_URL", pre=True)
    def assemble_redis_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return f"redis://{values.get('REDIS_HOST')}:{values.get('REDIS_PORT')}/{values.get('REDIS_DB')}"

    # Celery
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    @validator("CELERY_BROKER_URL", pre=True)
    def assemble_celery_broker(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return values.get("REDIS_URL", "redis://redis:6379/0")

    @validator("CELERY_RESULT_BACKEND", pre=True)
    def assemble_celery_backend(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return values.get("REDIS_URL", "redis://redis:6379/0")

    # File Storage
    UPLOAD_DIR: str = "/app/uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024 * 1024  # 10GB
    ALLOWED_EXTENSIONS: List[str] = [".tfrecord", ".png", ".jpg", ".jpeg", ".dcm"]

    # Training Configuration
    DEFAULT_LATENT_DIM: int = 200
    DEFAULT_IMAGE_SIZE: int = 128
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_EPOCHS: int = 100

    # Subscription Limits
    BASIC_MAX_IMAGES: int = 10000
    BASIC_MAX_CLIENTS: int = 3
    PROFESSIONAL_MAX_IMAGES: int = 100000
    PROFESSIONAL_MAX_CLIENTS: int = 10
    ENTERPRISE_MAX_IMAGES: int = -1  # Unlimited
    ENTERPRISE_MAX_CLIENTS: int = -1  # Unlimited

    # Email (for notifications)
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = 587
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = "MedSynth Platform"

    # Monitoring
    SENTRY_DSN: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
