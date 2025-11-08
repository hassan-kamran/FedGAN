"""Image generation router."""
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import time
from pathlib import Path

from app.database import get_db, User, GeneratedImage
from app.services.generator import ImageGenerator
from app.config import settings

router = APIRouter()
generator = ImageGenerator()


@router.post("/generate")
async def generate_image(
    model_type: str = Form("dcgan"),
    num_images: int = Form(1),
    image_size: int = Form(128),
    user_email: str = Form(...),  # In production, get from auth token
    db: AsyncSession = Depends(get_db)
):
    """Generate synthetic medical images."""

    # Get user
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check credits
    cost = num_images * settings.credit_cost_per_image
    if user.credits < cost:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Need {cost}, have {user.credits}"
        )

    # Generate images
    start_time = time.time()
    image_paths = await generator.generate(
        model_type=model_type,
        num_images=num_images,
        image_size=image_size
    )
    generation_time = time.time() - start_time

    # Deduct credits
    user.credits -= cost
    await db.commit()

    # Save records
    for path in image_paths:
        img_record = GeneratedImage(
            user_id=user.id,
            image_path=str(path),
            model_type=model_type,
            image_size=image_size,
            generation_time=generation_time / num_images
        )
        db.add(img_record)

    await db.commit()

    return {
        "success": True,
        "images_generated": len(image_paths),
        "remaining_credits": user.credits,
        "generation_time": generation_time,
        "image_ids": [img.id for img in db.new]
    }


@router.get("/{image_id}/download")
async def download_image(
    image_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Download generated image."""
    result = await db.execute(select(GeneratedImage).where(GeneratedImage.id == image_id))
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = Path(image.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(
        image_path,
        media_type="image/png",
        filename=f"synthetic_{image_id}.png"
    )
