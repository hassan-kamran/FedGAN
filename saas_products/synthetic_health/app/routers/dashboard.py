"""Dashboard router."""
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard page."""
    return templates.TemplateResponse(
        "pages/dashboard.html",
        {"request": request}
    )
