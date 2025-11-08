"""
Clients management router.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path
import secrets

from app.database import get_db
from app.models import Client, User, Institution
from app.routers.auth import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def list_clients(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all clients."""
    if current_user.role == "admin":
        clients = db.query(Client).all()
    elif current_user.institution_id:
        clients = db.query(Client).filter(
            Client.institution_id == current_user.institution_id
        ).all()
    else:
        clients = []

    return templates.TemplateResponse(
        "pages/clients.html",
        {"request": request, "user": current_user, "clients": clients}
    )


@router.post("/create")
async def create_client(
    name: str = Form(...),
    institution_id: int = Form(...),
    has_gpu: bool = Form(False),
    gpu_memory_gb: float = Form(None),
    ram_gb: float = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new client."""

    # Generate unique client ID
    client_id = f"client_{secrets.token_urlsafe(16)}"

    new_client = Client(
        name=name,
        client_id=client_id,
        institution_id=institution_id,
        has_gpu=has_gpu,
        gpu_memory_gb=gpu_memory_gb,
        ram_gb=ram_gb,
        status="inactive"
    )

    db.add(new_client)
    db.commit()
    db.refresh(new_client)

    return {
        "status": "success",
        "client_id": new_client.id,
        "client_api_id": client_id
    }


@router.get("/{client_id}", response_class=HTMLResponse)
async def view_client(
    client_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """View client details."""
    client = db.query(Client).filter(Client.id == client_id).first()

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    return templates.TemplateResponse(
        "pages/client_detail.html",
        {"request": request, "user": current_user, "client": client}
    )


@router.delete("/{client_id}")
async def delete_client(
    client_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a client."""
    client = db.query(Client).filter(Client.id == client_id).first()

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Check permissions
    if current_user.role not in ["admin", "institution_admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    db.delete(client)
    db.commit()

    return {"status": "success", "message": "Client deleted"}
