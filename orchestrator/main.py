"""Distributed LLM Orchestrator - Main entry point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from orchestrator.api.completions import router as completions_router
from orchestrator.api.nodes import router as nodes_router
from orchestrator.config import settings
from orchestrator.node_manager.heartbeat import heartbeat_monitor
from orchestrator.node_manager.registry import NodeRegistry
from orchestrator.scheduler.router import PipelineRouter
from orchestrator.session.manager import SessionManager

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.registry = NodeRegistry()
    app.state.session_manager = SessionManager()
    app.state.router = PipelineRouter(app.state.registry)

    # Start background heartbeat monitor
    heartbeat_task = asyncio.create_task(heartbeat_monitor(app.state.registry))
    logger.info(f"Orchestrator started on {settings.host}:{settings.port}")

    yield

    # Shutdown
    heartbeat_task.cancel()
    logger.info("Orchestrator shutting down")


app = FastAPI(
    title="Distributed LLM Orchestrator",
    description="Distribute LLM inference across browser-based WebGPU compute nodes",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - allow browser compute nodes from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(completions_router)
app.include_router(nodes_router)

# Serve the browser compute page (paths anchored to the repo, not the cwd)
_BASE_DIR = Path(__file__).resolve().parent.parent
app.mount(
    "/static",
    StaticFiles(directory=str(_BASE_DIR / "web_compute" / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(_BASE_DIR / "web_compute" / "templates"))


@app.get("/")
async def dashboard(request: Request):
    """Dashboard showing cluster status."""
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"nodes": app.state.registry.summary()},
    )


@app.get("/compute")
async def compute_page(request: Request):
    """The page contributors open to donate compute via WebGPU."""
    return templates.TemplateResponse(request, "compute.html", {})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "nodes": app.state.registry.summary(),
        "active_sessions": app.state.session_manager.active_count,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "orchestrator.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
