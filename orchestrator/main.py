"""FleetLM orchestrator - the fleet's single coordination point."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from orchestrator.api.batches import router as batches_router
from orchestrator.api.completions import router as completions_router
from orchestrator.api.nodes import router as nodes_router
from orchestrator.batch import BatchStore
from orchestrator.config import settings
from orchestrator.metrics import FleetMetrics
from orchestrator.fleet.heartbeat import heartbeat_monitor
from orchestrator.fleet.registry import NodeRegistry
from orchestrator.fleet.router import Router
from orchestrator.session import SessionManager
from orchestrator.verification import Canary, Verifier

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


def build_verifier() -> Verifier | None:
    """Load canaries from disk, or run unverified and say so.

    Returning None rather than an empty Verifier is deliberate: an operator
    who has configured nothing should see "every result is trusted" in the
    log, not a verification subsystem that silently checks nothing.
    """
    if not settings.canary_file:
        logger.warning(
            "No DLLM_CANARY_FILE set - results from every node are trusted "
            "completely. Fine for machines you control; not for a public fleet."
        )
        return None
    try:
        raw = json.loads(Path(settings.canary_file).read_text(encoding="utf-8"))
        canaries = [
            Canary(prompt=c["prompt"], expected=c["expected"], model=c.get("model"))
            for c in raw
        ]
    except (OSError, ValueError, KeyError, TypeError) as e:
        # Loud, and unverified rather than dead: a malformed canary file should
        # not take the fleet down, but it must never look like it is working.
        logger.error(
            f"Could not load canaries from {settings.canary_file}: {e}. "
            "Running unverified."
        )
        return None
    if not canaries:
        logger.error(f"{settings.canary_file} defines no canaries. Running unverified.")
        return None
    logger.info(
        f"Verification on: {len(canaries)} canaries at {settings.canary_rate:.1%}, "
        f"agreement threshold {settings.canary_agreement_threshold} "
        "(assumed, not yet measured)"
    )
    return Verifier(
        canaries=canaries,
        rate=settings.canary_rate,
        agreement_threshold=settings.canary_agreement_threshold,
    )


async def lease_reaper(store: BatchStore) -> None:
    """Return work units whose lease expired (node hung or vanished) to the queue."""
    while True:
        await asyncio.sleep(settings.lease_reaper_interval_sec)
        try:
            await store.expire_leases()
        except Exception as e:
            logger.error(f"Lease reaper error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.metrics = FleetMetrics()
    app.state.registry = NodeRegistry()
    app.state.session_manager = SessionManager()
    app.state.verifier = build_verifier()
    app.state.batch_store = BatchStore(
        metrics=app.state.metrics, verifier=app.state.verifier
    )
    app.state.router = Router(app.state.registry)

    if not settings.join_token:
        logger.warning(
            "No DLLM_JOIN_TOKEN set - any machine that can reach this "
            "orchestrator may join the fleet. Set one before exposing it publicly."
        )

    # Background tasks: node health, and reclaiming stale work leases
    heartbeat_task = asyncio.create_task(heartbeat_monitor(app.state.registry))
    reaper_task = asyncio.create_task(lease_reaper(app.state.batch_store))
    logger.info(f"FleetLM orchestrator started on {settings.host}:{settings.port}")

    yield

    # Shutdown
    heartbeat_task.cancel()
    reaper_task.cancel()
    logger.info("FleetLM orchestrator shutting down")


app = FastAPI(
    title="FleetLM Orchestrator",
    description="LLM inference served by a fleet of everyday laptops",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: batch clients and the dashboard may call from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(completions_router)
app.include_router(batches_router)
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
    """Live fleet status."""
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"nodes": app.state.registry.summary()},
    )


@app.get("/compute")
async def compute_page(request: Request):
    """How to contribute a machine, plus a WebGPU capability probe."""
    return templates.TemplateResponse(
        request, "compute.html", {"join_url": str(request.base_url).rstrip("/")}
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "nodes": app.state.registry.summary(),
        "active_sessions": app.state.session_manager.active_count,
        "batches": app.state.batch_store.summary(),
    }


@app.get("/metrics")
async def metrics():
    """Fleet counters: throughput, unit success rate, join times, per-node totals."""
    return app.state.metrics.snapshot()


@app.get("/verification")
async def verification():
    """What checking untrusted nodes has turned up, and what it cannot tell you."""
    verifier = app.state.verifier
    if verifier is None:
        return {
            "enabled": False,
            "reason": "no canary file configured - every result is trusted",
        }
    snapshot = verifier.snapshot()
    snapshot["caveats"] = [
        "Canaries prove a node answered known prompts correctly. They cannot "
        "prove it computed honestly on anything else.",
        "The agreement threshold is an assumption. How far two honest backends "
        "drift on identical input has not been measured yet.",
    ]
    return snapshot


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "orchestrator.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
