"""Batch inference API — the fleet's native workload.

Latency-tolerant bulk generation is what a fleet of consumer machines on home
internet is actually good at: work units are small, independent, idempotent,
and nobody is waiting on any individual one. Submit a batch, poll for status,
download results as JSONL.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from orchestrator.config import settings
from orchestrator.protocol import BatchCreateRequest, MessageType

logger = logging.getLogger("orchestrator.batches")

router = APIRouter()


def _batch_status(store, batch) -> dict:
    return {
        "id": batch.id,
        "object": "batch",
        "status": batch.state.value,
        "model": batch.model,
        "created_at": int(batch.created_at),
        "completed_at": int(batch.completed_at) if batch.completed_at else None,
        "request_counts": store.counts(batch.id),
        "usage": store.usage(batch.id),
    }


@router.post("/v1/batches")
async def create_batch(req: BatchCreateRequest, request: Request):
    """Submit a batch of chat requests for the fleet to work through."""
    if not req.requests:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "requests must not be empty",
                               "code": "invalid_request"}},
        )
    if len(req.requests) > settings.max_batch_requests:
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": f"batch exceeds {settings.max_batch_requests} requests",
                "code": "batch_too_large",
            }},
        )

    store = request.app.state.batch_store
    registry = request.app.state.registry

    batch = await store.create_batch(
        [r.model_dump() for r in req.requests], req.model
    )

    # Nudge idle nodes so they ask for work immediately instead of waiting
    # for their next poll.
    for node in registry.get_ready_nodes():
        try:
            await node.ws.send_json({"type": MessageType.WORK_AVAILABLE})
        except Exception as e:
            logger.debug(f"Could not notify node {node.node_id[:8]}: {e}")

    return JSONResponse(status_code=201, content=_batch_status(store, batch))


@router.get("/v1/batches")
async def list_batches(request: Request):
    store = request.app.state.batch_store
    batches = sorted(store.batches.values(), key=lambda b: b.created_at, reverse=True)
    return {
        "object": "list",
        "data": [_batch_status(store, b) for b in batches],
    }


@router.get("/v1/batches/{batch_id}")
async def get_batch(batch_id: str, request: Request):
    store = request.app.state.batch_store
    batch = store.get_batch(batch_id)
    if batch is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "batch not found", "code": "not_found"}},
        )
    return _batch_status(store, batch)


@router.get("/v1/batches/{batch_id}/results")
async def get_results(batch_id: str, request: Request):
    """Results as JSONL, one record per request, in submission order."""
    store = request.app.state.batch_store
    batch = store.get_batch(batch_id)
    if batch is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "batch not found", "code": "not_found"}},
        )
    lines = "\n".join(json.dumps(r) for r in store.results(batch_id))
    return PlainTextResponse(content=lines + ("\n" if lines else ""),
                             media_type="application/x-ndjson")


@router.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str, request: Request):
    store = request.app.state.batch_store
    if not await store.cancel_batch(batch_id):
        return JSONResponse(
            status_code=409,
            content={"error": {"message": "batch is not in progress",
                               "code": "not_cancellable"}},
        )
    return _batch_status(store, store.get_batch(batch_id))
