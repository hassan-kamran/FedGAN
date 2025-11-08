"""
SyntheticHealth - On-Demand Synthetic Medical Image Generation API

Main FastAPI application.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
import logging

from app.config import settings
from app.database import init_db
from app.routers import auth, images, api, dashboard

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
    description="On-Demand Synthetic Medical Image Generation API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Templates
templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(dashboard.router, prefix="", tags=["Dashboard"])
app.include_router(images.router, prefix="/images", tags=["Image Generation"])
app.include_router(api.router, prefix="/api/v1", tags=["API"])


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Create required directories
    for directory in [settings.models_dir, settings.output_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Required directories created")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.debug
    )
