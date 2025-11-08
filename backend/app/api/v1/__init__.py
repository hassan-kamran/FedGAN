"""
MedSynth Platform - API v1 Router
Combines all API v1 endpoints
"""
from fastapi import APIRouter
from app.api.v1 import auth, datasets, training

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
