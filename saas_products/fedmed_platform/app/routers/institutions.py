"""
Institutions management router.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path
import secrets
import hashlib

from app.database import get_db
from app.models import Institution, User
from app.routers.auth import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def list_institutions(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all institutions."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    institutions = db.query(Institution).all()

    return templates.TemplateResponse(
        "pages/institutions.html",
        {"request": request, "user": current_user, "institutions": institutions}
    )


@router.post("/create")
async def create_institution(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(""),
    address: str = Form(""),
    institution_type: str = Form("hospital"),
    subscription_tier: str = Form("free"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new institution."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    # Generate API credentials
    api_key = f"fmed_{secrets.token_urlsafe(32)}"
    api_secret = secrets.token_urlsafe(32)
    api_secret_hash = hashlib.sha256(api_secret.encode()).hexdigest()

    new_institution = Institution(
        name=name,
        email=email,
        phone=phone,
        address=address,
        institution_type=institution_type,
        subscription_tier=subscription_tier,
        api_key=api_key,
        api_secret_hash=api_secret_hash,
        is_active=True
    )

    db.add(new_institution)
    db.commit()
    db.refresh(new_institution)

    return {
        "status": "success",
        "institution_id": new_institution.id,
        "api_key": api_key,
        "api_secret": api_secret  # Only shown once!
    }


@router.get("/{institution_id}", response_class=HTMLResponse)
async def view_institution(
    institution_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """View institution details."""
    institution = db.query(Institution).filter(Institution.id == institution_id).first()

    if not institution:
        raise HTTPException(status_code=404, detail="Institution not found")

    # Check permissions
    if current_user.role != "admin" and current_user.institution_id != institution_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    return templates.TemplateResponse(
        "pages/institution_detail.html",
        {"request": request, "user": current_user, "institution": institution}
    )
