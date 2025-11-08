"""
MedSynth Platform - Training Job Schemas
Pydantic models for training job API validation
"""
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.models.training_job import JobStatus, JobType


class TrainingJobBase(BaseModel):
    """Base training job schema"""
    name: str
    description: Optional[str] = None
    job_type: JobType = JobType.FEDERATED


class TrainingJobCreate(TrainingJobBase):
    """Schema for creating a new training job"""
    dataset_id: int
    num_clients: int = 5
    num_rounds: int = 100
    batch_size: int = 32
    learning_rate: float = 0.0002
    latent_dim: int = 200
    model_config: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None

    @validator('num_clients')
    def validate_num_clients(cls, v):
        if v < 1 or v > 20:
            raise ValueError('Number of clients must be between 1 and 20')
        return v

    @validator('num_rounds')
    def validate_num_rounds(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('Number of rounds must be between 1 and 1000')
        return v

    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v <= 0 or v > 0.1:
            raise ValueError('Learning rate must be between 0 and 0.1')
        return v


class TrainingJobUpdate(BaseModel):
    """Schema for updating training job"""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[JobStatus] = None


class TrainingJobInDB(TrainingJobBase):
    """Training job schema with database fields"""
    id: int
    status: JobStatus
    dataset_id: int
    num_clients: int
    num_rounds: int
    batch_size: int
    learning_rate: float
    latent_dim: int
    current_round: int
    progress_percentage: float
    final_fid_score: Optional[float] = None
    final_inception_score: Optional[float] = None
    training_time_seconds: Optional[int] = None
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    logs_path: Optional[str] = None
    error_message: Optional[str] = None
    celery_task_id: Optional[str] = None
    owner_id: int
    organization_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TrainingJobResponse(TrainingJobInDB):
    """Training job response schema"""
    pass


class TrainingJobListResponse(BaseModel):
    """Response for list of training jobs"""
    jobs: List[TrainingJobResponse]
    total: int
    page: int
    page_size: int


class TrainingProgressResponse(BaseModel):
    """Training progress response"""
    job_id: int
    status: JobStatus
    current_round: int
    total_rounds: int
    progress_percentage: float
    estimated_time_remaining_seconds: Optional[int] = None
    current_metrics: Optional[Dict[str, float]] = None


# Synthetic Image Schemas
class SyntheticImageBase(BaseModel):
    """Base synthetic image schema"""
    filename: str


class SyntheticImageInDB(SyntheticImageBase):
    """Synthetic image schema with database fields"""
    id: int
    file_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    quality_score: Optional[float] = None
    realism_score: Optional[float] = None
    generation_round: Optional[int] = None
    training_job_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class SyntheticImageResponse(SyntheticImageInDB):
    """Synthetic image response schema"""
    pass


class SyntheticImageListResponse(BaseModel):
    """Response for list of synthetic images"""
    images: List[SyntheticImageResponse]
    total: int
    page: int
    page_size: int


# Privacy Metric Schemas
class PrivacyMetricBase(BaseModel):
    """Base privacy metric schema"""
    pass


class PrivacyMetricInDB(PrivacyMetricBase):
    """Privacy metric schema with database fields"""
    id: int
    privacy_risk_score: Optional[float] = None
    membership_inference_risk: Optional[float] = None
    model_inversion_risk: Optional[float] = None
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    reconstruction_error: Optional[float] = None
    attack_success_rate: Optional[float] = None
    detailed_results: Optional[Dict[str, Any]] = None
    training_job_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PrivacyMetricResponse(PrivacyMetricInDB):
    """Privacy metric response schema"""
    pass
