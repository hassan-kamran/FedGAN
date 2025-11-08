"""REST API endpoints."""
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from app.database import get_db, User, GeneratedImage

router = APIRouter()


class GenerateRequest(BaseModel):
    model_type: str = "dcgan"
    num_images: int = 1
    image_size: int = 128


@router.get("/credits")
async def get_credits(
    x_api_key: str = Header(...),
    db: AsyncSession = Depends(get_db)
):
    """Get user credits."""
    # In production, validate API key
    result = await db.execute(select(User).limit(1))
    user = result.scalar_one_or_none()

    return {
        "credits": user.credits if user else 0,
        "free_tier": settings.free_tier_credits
    }


@router.get("/health")
async def health():
    """API health check."""
    return {"status": "healthy"}
