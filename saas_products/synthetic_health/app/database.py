"""
Database connection and models.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.sql import func
from typing import AsyncGenerator

from app.config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug
)

# Create async session maker
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()


class User(Base):
    """User model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True)
    hashed_password = Column(String)
    credits = Column(Integer, default=10)  # Free tier credits
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GeneratedImage(Base):
    """Generated image tracking."""
    __tablename__ = "generated_images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_path = Column(String)
    image_url = Column(String, nullable=True)
    model_type = Column(String, default="dcgan")
    image_size = Column(Integer, default=128)
    generation_time = Column(Float)  # seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Initialize database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
