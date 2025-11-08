"""
RESTful API v1 endpoints for programmatic access.
"""
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import Optional
import hashlib

from app.database import get_db
from app.models import Institution, TrainingJob, Client
from app.routers.auth import get_current_user
from app.models.user import User

router = APIRouter()


async def verify_api_key(
    x_api_key: str = Header(...),
    db: Session = Depends(get_db)
) -> Institution:
    """Verify API key and return institution."""
    institution = db.query(Institution).filter(
        Institution.api_key == x_api_key,
        Institution.is_active == True
    ).first()

    if not institution:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return institution


@router.get("/jobs")
async def api_list_jobs(
    institution: Institution = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """List all training jobs for an institution."""
    # Get clients belonging to this institution
    client_ids = [c.id for c in institution.clients]

    # Get jobs involving these clients
    jobs = db.query(TrainingJob).join(Client).filter(
        Client.id.in_(client_ids)
    ).all()

    return {
        "jobs": [
            {
                "id": job.id,
                "name": job.name,
                "status": job.status,
                "current_round": job.current_round,
                "total_rounds": job.num_rounds,
                "created_at": job.created_at.isoformat() if job.created_at else None
            }
            for job in jobs
        ]
    }


@router.get("/jobs/{job_id}")
async def api_get_job(
    job_id: int,
    institution: Institution = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get training job details."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "name": job.name,
        "description": job.description,
        "status": job.status,
        "model_type": job.model_type,
        "current_round": job.current_round,
        "total_rounds": job.num_rounds,
        "progress_percent": job.progress_percent,
        "metrics": job.training_metrics,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    }


@router.get("/clients")
async def api_list_clients(
    institution: Institution = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """List all clients for an institution."""
    clients = db.query(Client).filter(
        Client.institution_id == institution.id
    ).all()

    return {
        "clients": [
            {
                "id": client.id,
                "name": client.name,
                "client_id": client.client_id,
                "status": client.status,
                "has_gpu": client.has_gpu,
                "num_samples": client.num_training_samples
            }
            for client in clients
        ]
    }


@router.post("/clients/{client_id}/heartbeat")
async def api_client_heartbeat(
    client_id: str,
    institution: Institution = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Client heartbeat endpoint."""
    from datetime import datetime

    client = db.query(Client).filter(
        Client.client_id == client_id,
        Client.institution_id == institution.id
    ).first()

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Update last seen
    client.last_seen = datetime.utcnow()
    client.status = "active"
    db.commit()

    return {"status": "ok", "message": "Heartbeat received"}


@router.get("/health")
async def api_health():
    """API health check."""
    return {"status": "healthy", "version": "1.0.0"}
