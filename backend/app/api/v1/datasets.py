"""
MedSynth Platform - Datasets API
Endpoints for managing medical imaging datasets
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
from pathlib import Path

from app.core.config import settings
from app.core.security import get_current_user
from app.models import get_db, User, Dataset
from app.models.dataset import DatasetStatus, DatasetType
from app.schemas import (
    DatasetCreate, DatasetUpdate, DatasetResponse,
    DatasetListResponse, DatasetStatsResponse
)

router = APIRouter()


@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(
    dataset_data: DatasetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new dataset entry
    """
    # Create dataset
    new_dataset = Dataset(
        name=dataset_data.name,
        description=dataset_data.description,
        dataset_type=dataset_data.dataset_type,
        contains_phi=dataset_data.contains_phi,
        anonymized=dataset_data.anonymized,
        preprocessing_config=dataset_data.preprocessing_config,
        file_path="",  # Will be updated on file upload
        owner_id=current_user.id,
        organization_id=current_user.organization_id,
    )

    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)

    return new_dataset


@router.post("/{dataset_id}/upload")
async def upload_dataset_file(
    dataset_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload dataset file (TFRecord, images, etc.)
    """
    # Get dataset
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.owner_id == current_user.id
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )

    # Create upload directory
    upload_dir = Path(settings.UPLOAD_DIR) / str(current_user.id) / str(dataset_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get file size
    file_size = os.path.getsize(file_path)

    # Update dataset
    dataset.file_path = str(file_path)
    dataset.file_size = file_size
    dataset.file_format = file_ext
    dataset.status = DatasetStatus.PROCESSING

    db.commit()

    return {
        "message": "File uploaded successfully",
        "file_path": str(file_path),
        "file_size": file_size
    }


@router.get("/", response_model=DatasetListResponse)
def list_datasets(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    dataset_type: Optional[DatasetType] = None,
    status: Optional[DatasetStatus] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all datasets for the current user
    """
    query = db.query(Dataset).filter(Dataset.owner_id == current_user.id)

    # Apply filters
    if dataset_type:
        query = query.filter(Dataset.dataset_type == dataset_type)
    if status:
        query = query.filter(Dataset.status == status)

    # Get total count
    total = query.count()

    # Paginate
    datasets = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "datasets": datasets,
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.get("/stats", response_model=DatasetStatsResponse)
def get_dataset_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get dataset statistics for the current user
    """
    datasets = db.query(Dataset).filter(Dataset.owner_id == current_user.id).all()

    total_datasets = len(datasets)
    total_images = sum(d.num_images or 0 for d in datasets)
    total_size = sum(d.file_size or 0 for d in datasets)

    # Count by type
    datasets_by_type = {}
    for dt in DatasetType:
        count = sum(1 for d in datasets if d.dataset_type == dt)
        if count > 0:
            datasets_by_type[dt.value] = count

    # Count by status
    datasets_by_status = {}
    for ds in DatasetStatus:
        count = sum(1 for d in datasets if d.status == ds)
        if count > 0:
            datasets_by_status[ds.value] = count

    return {
        "total_datasets": total_datasets,
        "total_images": total_images,
        "total_size_bytes": total_size,
        "datasets_by_type": datasets_by_type,
        "datasets_by_status": datasets_by_status
    }


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific dataset by ID
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.owner_id == current_user.id
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    return dataset


@router.put("/{dataset_id}", response_model=DatasetResponse)
def update_dataset(
    dataset_id: int,
    dataset_data: DatasetUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update dataset information
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.owner_id == current_user.id
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    # Update fields
    update_data = dataset_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(dataset, field, value)

    db.commit()
    db.refresh(dataset)

    return dataset


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a dataset
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.owner_id == current_user.id
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    # Delete file if exists
    if dataset.file_path and os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)

    # Delete dataset
    db.delete(dataset)
    db.commit()

    return None
