"""
FedTrain Express - Simplified Federated Training Service

Main FastAPI application.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI(
    title="FedTrain Express",
    version="1.0.0",
    description="Simplified Federated Training for Everyone"
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Templates
templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "pages/home.html",
        {
            "request": request,
            "features": [
                {"icon": "🚀", "title": "One-Click Setup", "desc": "Start training in minutes"},
                {"icon": "👥", "title": "Free for 3 Clients", "desc": "Perfect for small teams"},
                {"icon": "🎯", "title": "Pre-built Templates", "desc": "Medical imaging ready"},
                {"icon": "🔒", "title": "Privacy-First", "desc": "Data stays local"}
            ]
        }
    )


@app.get("/templates")
async def list_templates():
    """List available training templates."""
    return {
        "templates": [
            {
                "id": "retinopathy",
                "name": "Diabetic Retinopathy Detection",
                "description": "DCGAN for generating synthetic retinal images",
                "model_type": "dcgan",
                "recommended_clients": 3
            },
            {
                "id": "pneumonia",
                "name": "Pneumonia Detection",
                "description": "CNN for chest X-ray classification",
                "model_type": "cnn",
                "recommended_clients": 5
            },
            {
                "id": "skin_lesion",
                "name": "Skin Lesion Classification",
                "description": "ResNet-based classification",
                "model_type": "resnet",
                "recommended_clients": 3
            }
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8003, reload=True)
