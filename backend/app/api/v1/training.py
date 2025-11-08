"""
MedSynth Platform - Training Jobs API
Endpoints for managing federated GAN training jobs
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.core.config import settings
from app.core.security import get_current_user
from app.models import get_db, User, Dataset, TrainingJob, SyntheticImage, PrivacyMetric
from app.models.training_job import JobStatus, JobType
from app.schemas import (
    TrainingJobCreate, TrainingJobUpdate, TrainingJobResponse,
    TrainingJobListResponse, TrainingProgressResponse,
    SyntheticImageResponse, SyntheticImageListResponse,
    PrivacyMetricResponse
)

router = APIRouter()


@router.post("/", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
def create_training_job(
    job_data: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new training job
    """
    # Verify dataset exists and belongs to user
    dataset = db.query(Dataset).filter(
        Dataset.id == job_data.dataset_id,
        Dataset.owner_id == current_user.id
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    # Check subscription limits
    if current_user.subscription_tier.value == "basic":
        if job_data.num_clients > settings.BASIC_MAX_CLIENTS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Basic tier limited to {settings.BASIC_MAX_CLIENTS} clients"
            )

    # Create training job
    new_job = TrainingJob(
        name=job_data.name,
        description=job_data.description,
        job_type=job_data.job_type,
        dataset_id=job_data.dataset_id,
        num_clients=job_data.num_clients,
        num_rounds=job_data.num_rounds,
        batch_size=job_data.batch_size,
        learning_rate=job_data.learning_rate,
        latent_dim=job_data.latent_dim,
        model_config=job_data.model_config,
        training_config=job_data.training_config,
        owner_id=current_user.id,
        organization_id=current_user.organization_id,
        status=JobStatus.QUEUED
    )

    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Queue the training job (will be handled by Celery)
    # background_tasks.add_task(queue_training_job, new_job.id)

    return new_job


@router.get("/", response_model=TrainingJobListResponse)
def list_training_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[JobStatus] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all training jobs for the current user
    """
    query = db.query(TrainingJob).filter(TrainingJob.owner_id == current_user.id)

    # Apply filters
    if status:
        query = query.filter(TrainingJob.status == status)

    # Order by creation date (newest first)
    query = query.order_by(TrainingJob.created_at.desc())

    # Get total count
    total = query.count()

    # Paginate
    jobs = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "jobs": jobs,
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.get("/{job_id}", response_model=TrainingJobResponse)
def get_training_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific training job by ID
    """
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.owner_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )

    return job


@router.get("/{job_id}/progress", response_model=TrainingProgressResponse)
def get_training_progress(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get real-time progress of a training job
    """
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.owner_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )

    # Calculate estimated time remaining
    estimated_time = None
    if job.status == JobStatus.RUNNING and job.started_at:
        elapsed = (datetime.utcnow() - job.started_at).total_seconds()
        if job.current_round > 0:
            time_per_round = elapsed / job.current_round
            remaining_rounds = job.num_rounds - job.current_round
            estimated_time = int(time_per_round * remaining_rounds)

    return {
        "job_id": job.id,
        "status": job.status,
        "current_round": job.current_round,
        "total_rounds": job.num_rounds,
        "progress_percentage": job.progress_percentage,
        "estimated_time_remaining_seconds": estimated_time,
        "current_metrics": {
            "fid_score": job.final_fid_score,
            "inception_score": job.final_inception_score
        }
    }


@router.put("/{job_id}", response_model=TrainingJobResponse)
def update_training_job(
    job_id: int,
    job_data: TrainingJobUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update training job (mainly for status changes)
    """
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.owner_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )

    # Update fields
    update_data = job_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(job, field, value)

    db.commit()
    db.refresh(job)

    return job


@router.post("/{job_id}/cancel")
def cancel_training_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Cancel a running training job
    """
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.owner_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )

    if job.status not in [JobStatus.QUEUED, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job cannot be cancelled in current status"
        )

    # Update status
    job.status = JobStatus.CANCELLED
    db.commit()

    # TODO: Cancel Celery task if running

    return {"message": "Training job cancelled successfully"}


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_training_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a training job
    """
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.owner_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )

    # Delete job (cascades to synthetic images and privacy metrics)
    db.delete(job)
    db.commit()

    return None


# Synthetic Images Endpoints

@router.get("/{job_id}/images", response_model=SyntheticImageListResponse)
def list_synthetic_images(
    job_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List synthetic images generated by a training job
    """
    # Verify job belongs to user
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.owner_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )

    # Get synthetic images
    query = db.query(SyntheticImage).filter(SyntheticImage.training_job_id == job_id)
    query = query.order_by(SyntheticImage.created_at.desc())

    total = query.count()
    images = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "images": images,
        "total": total,
        "page": page,
        "page_size": page_size
    }


# Privacy Metrics Endpoints

@router.get("/{job_id}/privacy", response_model=PrivacyMetricResponse)
def get_privacy_metrics(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get privacy evaluation metrics for a training job
    """
    # Verify job belongs to user
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.owner_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )

    # Get privacy metrics
    privacy_metric = db.query(PrivacyMetric).filter(
        PrivacyMetric.training_job_id == job_id
    ).first()

    if not privacy_metric:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Privacy metrics not yet available"
        )

    return privacy_metric
