"""
PrivacyGuard - Privacy Risk Assessment Dashboard

Main FastAPI application.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI(
    title="PrivacyGuard",
    version="1.0.0",
    description="Privacy Risk Assessment and Compliance Platform"
)

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


@app.get("/")
async def root():
    return {
        "app": "PrivacyGuard",
        "description": "Privacy Risk Assessment Platform",
        "features": [
            "Membership inference attack simulation",
            "Model inversion testing",
            "Differential privacy analysis",
            "HIPAA/GDPR compliance reporting"
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)
