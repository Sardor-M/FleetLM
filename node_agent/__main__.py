"""FleetLM native compute node agent.

Holds a whole model in memory and serves complete generations to the
orchestrator over a single outbound WebSocket (NAT-friendly: the node never
accepts inbound connections).

Usage:
    python -m node_agent
    ORCHESTRATOR_URL=ws://server:8080/nodes/ws python -m node_agent

Environment:
    ORCHESTRATOR_URL  ws endpoint (default ws://localhost:8080/nodes/ws)
    NODE_ENGINE       auto | mlx | llama_cpp | mock   (default auto)
    NODE_MODEL        model to serve; HF repo id for mlx, GGUF path for
                      llama_cpp (default mlx-community/Llama-3.2-1B-Instruct-4bit)
    NODE_BATCH_SIZE   work units leased and decoded per batch (default 4)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import time
import uuid

import psutil
import websockets

from node_agent.engine import (
    BaseEngine,
    BatchItem,
    BatchOutput,
    EngineError,
    create_engine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("node_agent")

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
IDLE_POLL_SEC = 10  # how often an idle node asks for batch work


def detect_gpu() -> tuple[str, int]:
    """Detect GPU type and approximate usable memory in MB."""
    if platform.system() == "Darwin":
        total_ram = psutil.virtual_memory().total // (1024 * 1024)
        return "apple-silicon", int(total_ram * 0.75)

    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return f"nvidia-{gpu.name.lower().replace(' ', '-')}", int(gpu.memoryTotal)
    except ImportError:
        pass

    return "cpu-only", 0


class NodeAgent:
    def __init__(
        self,
        orchestrator_url: str,
        engine: BaseEngine,
        model_id: str,
        batch_size: int = 4,
        join_token: str = "",
    ):
        self.orchestrator_url = orchestrator_url
        self.engine = engine
        self.model_id = model_id
        self.batch_size = batch_size
        self.join_token = join_token
        self.node_id = uuid.uuid4().hex
        self.model_loaded = False
        self.active_sessions = 0
        self.gpu_name, self.gpu_vram_mb = detect_gpu()
        # All outbound traffic funnels through one queue so generation threads,
        # heartbeats, and handlers never write to the socket concurrently.
        self.outbox: asyncio.Queue[dict] = asyncio.Queue()
        # Batch work units run one at a time, behind the same engine lock as
        # interactive requests; this queue is the node's local backlog.
        self.work_queue: asyncio.Queue[dict] = asyncio.Queue()
        self.work_in_progress = 0

    async def run(self):
        logger.info(f"FleetLM node agent starting (id={self.node_id[:8]})")
        logger.info(f"Hardware: {self.gpu_name}, {self.gpu_vram_mb} MB usable memory")
        logger.info(f"Engine: {self.engine.name}, model: {self.model_id}")

        async for ws in websockets.connect(self.orchestrator_url, max_size=None):
            try:
                logger.info("Connected to orchestrator")
                await ws.send(json.dumps({
                    "type": "register",
                    "node_id": self.node_id,
                    "gpu_name": self.gpu_name,
                    "gpu_vram_mb": self.gpu_vram_mb,
                    "runtime": "native",
                    "model_id": self.model_id,
                    "join_token": self.join_token,
                }))

                sender = asyncio.create_task(self._sender_loop(ws))
                heartbeat = asyncio.create_task(self._heartbeat_loop())
                worker = asyncio.create_task(self._work_loop())
                poller = asyncio.create_task(self._work_poll_loop())
                try:
                    await self._message_loop(ws)
                except EngineError as e:
                    logger.error(f"Fatal: {e} - shutting down")
                    return
                finally:
                    for task in (sender, heartbeat, worker, poller):
                        task.cancel()
                    self._drain_work_queue()

            except websockets.ConnectionClosed as e:
                if e.rcvd is not None and e.rcvd.code == 4401:
                    logger.error("Rejected by the fleet: invalid join token")
                    return
                logger.warning("Connection lost, reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _sender_loop(self, ws):
        while True:
            msg = await self.outbox.get()
            await ws.send(json.dumps(msg))

    async def _heartbeat_loop(self):
        while True:
            self.outbox.put_nowait({
                "type": "heartbeat",
                "node_id": self.node_id,
                "cpu_usage": psutil.cpu_percent(),
                "gpu_usage": 0.0,
                "ram_usage": psutil.virtual_memory().percent,
                "active_sessions": self.active_sessions + self.work_in_progress,
            })
            await asyncio.sleep(5)

    async def _message_loop(self, ws):
        async for raw in ws:
            if not isinstance(raw, str):
                logger.debug(f"Ignoring binary frame ({len(raw)} bytes)")
                continue
            msg = json.loads(raw)
            t = msg.get("type")

            if t == "serve_model":
                await self._load_model(msg.get("model_id") or self.model_id)

            elif t == "generate_request":
                asyncio.create_task(self._generate(msg))

            elif t == "work_assignment":
                units = msg.get("units", [])
                for unit in units:
                    self.work_queue.put_nowait(unit)
                if units:
                    logger.info(f"Leased {len(units)} work units")

            elif t == "work_available":
                self._request_work()

            elif t == "session_end":
                pass  # nothing to clean up: generations are stateless per request

            else:
                logger.warning(f"Unknown message type: {t}")

    async def _load_model(self, model_id: str):
        if self.model_loaded:
            self.outbox.put_nowait({
                "type": "model_loaded", "node_id": self.node_id, "model_id": self.model_id,
            })
            return
        try:
            await asyncio.to_thread(self.engine.load, self.model_id)
        except Exception as e:
            self.outbox.put_nowait({
                "type": "error", "node_id": self.node_id,
                "message": f"model load failed: {e}",
            })
            await asyncio.sleep(0.5)  # let the sender flush the error
            raise EngineError(f"model load failed: {e}") from e
        self.model_loaded = True
        self.outbox.put_nowait({
            "type": "model_loaded", "node_id": self.node_id, "model_id": self.model_id,
        })
        logger.info("Model loaded, ready for inference")
        self._request_work()

    async def _generate(self, msg: dict):
        session_id = msg["session_id"]
        messages = msg.get("messages", [])
        max_tokens = msg.get("max_tokens", 256)
        temperature = msg.get("temperature", 0.7)
        loop = asyncio.get_running_loop()
        self.active_sessions += 1
        logger.info(f"Generate: session={session_id}, {len(messages)} messages")

        def produce():
            emitted = 0
            try:
                for piece in self.engine.generate_stream(messages, max_tokens, temperature):
                    if piece:
                        emitted += 1
                        loop.call_soon_threadsafe(self.outbox.put_nowait, {
                            "type": "generate_chunk",
                            "session_id": session_id,
                            "text": piece,
                        })
                loop.call_soon_threadsafe(self.outbox.put_nowait, {
                    "type": "generate_complete",
                    "session_id": session_id,
                    "finish_reason": "stop",
                    "prompt_tokens": self.engine.last_prompt_tokens,
                    "completion_tokens": self.engine.last_completion_tokens,
                })
            except Exception as e:
                logger.error(f"Generation failed for session {session_id}: {e}")
                loop.call_soon_threadsafe(self.outbox.put_nowait, {
                    "type": "generate_error",
                    "session_id": session_id,
                    "message": str(e),
                })
            return emitted

        try:
            emitted = await asyncio.to_thread(produce)
            logger.info(f"Session {session_id}: {emitted} chunks emitted")
        finally:
            self.active_sessions -= 1


        """Decode leased units as one batch, reporting each result separately."""
        while True:
            units = [await self.work_queue.get()]
            # Everything already leased and waiting joins this batch. Decode is
            # memory-bandwidth bound, so extra batch width is nearly free.
            while len(units) < self.batch_size and not self.work_queue.empty():
                units.append(self.work_queue.get_nowait())
            self.work_in_progress += len(units)
            try:
                outputs, seconds = await asyncio.to_thread(self._run_units, units)
            except Exception as e:
                logger.error(f"Batch of {len(units)} units failed: {e}")
                for unit in units:
                    self.outbox.put_nowait({
                        "type": "work_failed",
                        "node_id": self.node_id,
                        "unit_id": unit["unit_id"],
                        "message": str(e),
                    })
            else:
                self._report_units(units, outputs, seconds)
            finally:
                self.work_in_progress -= len(units)
                self._request_work()

    def _run_units(self, units: list[dict]) -> tuple[list[BatchOutput], float]:
        items = [
            BatchItem(
                messages=u.get("messages", []),
                max_tokens=u.get("max_tokens", 256),
                temperature=u.get("temperature", 0.7),
            )
            for u in units
        ]
        started = time.monotonic()
        outputs = self.engine.generate_batch(items)
        elapsed = time.monotonic() - started
        if len(outputs) != len(units):
            raise EngineError(
                f"engine returned {len(outputs)} outputs for {len(units)} units"
            )
        # Units in a batch run concurrently, so each is credited an equal share
        # of wall clock. Per-node tokens/sec then reports real throughput
        # instead of dividing by the batch's time once per unit.
        return outputs, elapsed / len(units)

    def _report_units(
        self, units: list[dict], outputs: list[BatchOutput], seconds: float
    ) -> None:
        completed = 0
        for unit, out in zip(units, outputs):
            unit_id = unit.get("unit_id", "unknown")
            if out.error:
                logger.error(f"Unit {unit_id} failed: {out.error}")
                self.outbox.put_nowait({
                    "type": "work_failed",
                    "node_id": self.node_id,
                    "unit_id": unit_id,
                    "message": out.error,
                })
                continue
            completed += 1
            self.outbox.put_nowait({
                "type": "work_result",
                "node_id": self.node_id,
                "unit_id": unit_id,
                "text": out.text,
                "prompt_tokens": out.prompt_tokens,
                "completion_tokens": out.completion_tokens,
                "generation_sec": round(seconds, 3),
            })
        tokens = sum(o.completion_tokens for o in outputs if not o.error)
        wall = seconds * len(units)
        rate = tokens / wall if wall > 0 else 0
        logger.info(
            f"Batch of {len(units)} done ({completed} ok, {tokens} tokens, "
            f"{wall:.1f}s, {rate:.1f} tok/s)"
        )

    def _drain_work_queue(self) -> None:
        """Drop local backlog on disconnect: the orchestrator requeues those leases."""
        dropped = 0
        while not self.work_queue.empty():
            self.work_queue.get_nowait()
            dropped += 1
        if dropped:
            logger.warning(f"Dropped {dropped} unstarted work units on disconnect")


async def main():
    url = os.environ.get("ORCHESTRATOR_URL", "ws://localhost:8080/nodes/ws")
    engine = create_engine(os.environ.get("NODE_ENGINE", "auto"))
    model_id = os.environ.get("NODE_MODEL", DEFAULT_MODEL)
    batch_size = int(os.environ.get("NODE_BATCH_SIZE", "4"))
    token = os.environ.get("FLEETLM_JOIN_TOKEN", "")
    agent = NodeAgent(url, engine, model_id, batch_size, token)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
