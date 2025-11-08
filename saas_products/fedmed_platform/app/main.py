"""
FedMed Platform - Enterprise Federated Learning Platform for Healthcare

Main FastAPI application entry point.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging
from pathlib import Path

from app.config import settings
from app.database import init_db
from app.routers import auth, institutions, training_jobs, clients, dashboard, api_v1

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise Federated Learning Platform for Healthcare Institutions",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Setup templates
templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(dashboard.router, prefix="", tags=["Dashboard"])
app.include_router(institutions.router, prefix="/institutions", tags=["Institutions"])
app.include_router(training_jobs.router, prefix="/training", tags=["Training Jobs"])
app.include_router(clients.router, prefix="/clients", tags=["Clients"])
app.include_router(api_v1.router, prefix="/api/v1", tags=["API v1"])


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Create required directories
    for directory in [settings.upload_dir, settings.model_storage_path, settings.checkpoint_dir, "logs"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Required directories created")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info(f"Shutting down {settings.app_name}")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Landing page."""
    return templates.TemplateResponse(
        "pages/landing.html",
        {"request": request, "app_name": settings.app_name}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version
    }


@app.get("/api/info")
async def api_info():
    """API information endpoint."""
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "max_clients": settings.max_clients,
        "default_rounds": settings.default_rounds,
        "endpoints": {
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
