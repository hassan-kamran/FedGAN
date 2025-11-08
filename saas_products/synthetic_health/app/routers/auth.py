"""Authentication router."""
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
import secrets

from app.database import get_db, User
from app.config import settings

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post("/register")
async def register(
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Register new user."""
    # Check if user exists
    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    hashed_password = pwd_context.hash(password)
    user = User(
        email=email,
        username=username,
        hashed_password=hashed_password,
        credits=settings.free_tier_credits
    )
    db.add(user)
    await db.commit()

    return RedirectResponse(url="/auth/login", status_code=303)


@router.post("/login")
async def login(
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Login user."""
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user or not pwd_context.verify(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create token
    token = jwt.encode(
        {"sub": user.email, "exp": datetime.utcnow() + timedelta(hours=24)},
        settings.secret_key,
        algorithm="HS256"
    )

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie("access_token", f"Bearer {token}", httponly=True)
    return response
