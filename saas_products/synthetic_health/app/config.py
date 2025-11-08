"""
Configuration management for SyntheticHealth.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = "SyntheticHealth"
    app_version: str = "1.0.0"
    debug: bool = False
    secret_key: str

    # Database
    database_url: str = "sqlite+aiosqlite:///./synthetichealth.db"

    # Model Storage
    models_dir: str = "./models"
    generator_model_path: str = "./models/generator.h5"
    output_dir: str = "./generated_images"

    # Image Generation
    default_image_size: int = 128
    latent_dim: int = 200
    max_batch_size: int = 100

    # Storage
    use_s3: bool = False
    s3_bucket: str = "synthetic-health-images"
    s3_endpoint: str = "https://s3.amazonaws.com"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # Pricing
    free_tier_credits: int = 10
    credit_cost_per_image: int = 1

    # Rate Limiting
    rate_limit_per_minute: int = 60

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
