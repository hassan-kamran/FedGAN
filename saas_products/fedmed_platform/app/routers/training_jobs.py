"""
Training jobs management router.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.database import get_db
from app.models import TrainingJob, User, Client
from app.routers.auth import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def list_training_jobs(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all training jobs."""
    jobs = db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).all()

    return templates.TemplateResponse(
        "pages/training_jobs.html",
        {"request": request, "user": current_user, "jobs": jobs}
    )


@router.get("/new", response_class=HTMLResponse)
async def new_training_job_form(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Display form for creating new training job."""
    return templates.TemplateResponse(
        "pages/new_training_job.html",
        {"request": request, "user": current_user}
    )


@router.post("/create")
async def create_training_job(
    name: str = Form(...),
    description: str = Form(""),
    model_type: str = Form("dcgan"),
    num_rounds: int = Form(10),
    local_epochs: int = Form(5),
    batch_size: int = Form(16),
    learning_rate: float = Form(0.0002),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new training job."""

    new_job = TrainingJob(
        name=name,
        description=description,
        model_type=model_type,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        created_by=current_user.id,
        status="pending"
    )

    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    return {"status": "success", "job_id": new_job.id, "redirect": f"/training/{new_job.id}"}


@router.get("/{job_id}", response_class=HTMLResponse)
async def view_training_job(
    job_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """View training job details."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Get associated clients
    clients = db.query(Client).filter(Client.training_job_id == job_id).all()

    return templates.TemplateResponse(
        "pages/training_job_detail.html",
        {"request": request, "user": current_user, "job": job, "clients": clients}
    )


@router.post("/{job_id}/start")
async def start_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Start a training job."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status == "running":
        raise HTTPException(status_code=400, detail="Job is already running")

    # Update job status
    job.status = "running"
    job.started_at = datetime.utcnow()
    job.current_round = 0

    db.commit()

    # TODO: Trigger background task to run federated training

    return {"status": "success", "message": "Training job started"}


@router.post("/{job_id}/stop")
async def stop_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Stop a training job."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status != "running":
        raise HTTPException(status_code=400, detail="Job is not running")

    # Update job status
    job.status = "cancelled"
    job.completed_at = datetime.utcnow()

    db.commit()

    return {"status": "success", "message": "Training job stopped"}


@router.get("/{job_id}/progress")
async def get_job_progress(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get training job progress (for HTMX polling)."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    return {
        "status": job.status,
        "current_round": job.current_round,
        "total_rounds": job.num_rounds,
        "progress_percent": job.progress_percent,
        "metrics": job.training_metrics
    }


@router.delete("/{job_id}")
async def delete_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a training job."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Check permissions
    if current_user.role not in ["admin", "institution_admin"] and job.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this job")

    db.delete(job)
    db.commit()

    return {"status": "success", "message": "Training job deleted"}
