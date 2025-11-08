"""
MedSynth Platform - Celery Application
Celery configuration for async task processing
"""
from celery import Celery
from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "medsynth",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 12,  # 12 hours max
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.tasks"])
