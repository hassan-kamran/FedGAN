"""
Dashboard router for main UI.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func
from pathlib import Path

from app.database import get_db
from app.models import User, TrainingJob, Client, Institution
from app.routers.auth import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Main dashboard view."""

    # Get statistics
    total_jobs = db.query(func.count(TrainingJob.id)).scalar()
    running_jobs = db.query(func.count(TrainingJob.id)).filter(
        TrainingJob.status == "running"
    ).scalar()
    completed_jobs = db.query(func.count(TrainingJob.id)).filter(
        TrainingJob.status == "completed"
    ).scalar()
    total_clients = db.query(func.count(Client.id)).scalar()
    active_clients = db.query(func.count(Client.id)).filter(
        Client.status == "active"
    ).scalar()

    # Get recent jobs
    recent_jobs = db.query(TrainingJob).order_by(
        TrainingJob.created_at.desc()
    ).limit(10).all()

    return templates.TemplateResponse(
        "pages/dashboard.html",
        {
            "request": request,
            "user": current_user,
            "stats": {
                "total_jobs": total_jobs,
                "running_jobs": running_jobs,
                "completed_jobs": completed_jobs,
                "total_clients": total_clients,
                "active_clients": active_clients
            },
            "recent_jobs": recent_jobs
        }
    )


@router.get("/dashboard/stats")
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get dashboard statistics (HTMX endpoint)."""
    total_jobs = db.query(func.count(TrainingJob.id)).scalar()
    running_jobs = db.query(func.count(TrainingJob.id)).filter(
        TrainingJob.status == "running"
    ).scalar()
    completed_jobs = db.query(func.count(TrainingJob.id)).filter(
        TrainingJob.status == "completed"
    ).scalar()

    return {
        "total_jobs": total_jobs,
        "running_jobs": running_jobs,
        "completed_jobs": completed_jobs
    }
