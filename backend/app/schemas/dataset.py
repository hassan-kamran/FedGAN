"""
MedSynth Platform - Dataset Schemas
Pydantic models for dataset API validation
"""
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any
from datetime import datetime
from app.models.dataset import DatasetStatus, DatasetType


class DatasetBase(BaseModel):
    """Base dataset schema"""
    name: str
    description: Optional[str] = None
    dataset_type: DatasetType


class DatasetCreate(DatasetBase):
    """Schema for creating a new dataset"""
    contains_phi: bool = False
    anonymized: bool = False
    preprocessing_config: Optional[Dict[str, Any]] = None


class DatasetUpdate(BaseModel):
    """Schema for updating dataset"""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[DatasetStatus] = None
    anonymized: Optional[bool] = None
    preprocessing_config: Optional[Dict[str, Any]] = None


class DatasetInDB(DatasetBase):
    """Dataset schema with database fields"""
    id: int
    status: DatasetStatus
    file_path: str
    file_size: Optional[int] = None
    file_format: Optional[str] = None
    num_images: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    num_classes: Optional[int] = None
    contains_phi: bool
    anonymized: bool
    hipaa_compliant: bool
    preprocessing_config: Optional[Dict[str, Any]] = None
    owner_id: int
    organization_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DatasetResponse(DatasetInDB):
    """Dataset response schema"""
    pass


class DatasetListResponse(BaseModel):
    """Response for list of datasets"""
    datasets: list[DatasetResponse]
    total: int
    page: int
    page_size: int


class DatasetStatsResponse(BaseModel):
    """Dataset statistics response"""
    total_datasets: int
    total_images: int
    total_size_bytes: int
    datasets_by_type: Dict[str, int]
    datasets_by_status: Dict[str, int]
