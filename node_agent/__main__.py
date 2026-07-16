"""Native compute node agent.

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
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import uuid

import psutil
import websockets

from node_agent.engine.whole_model import BaseEngine, EngineError, create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("node_agent")

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


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
    def __init__(self, orchestrator_url: str, engine: BaseEngine, model_id: str):
        self.orchestrator_url = orchestrator_url
        self.engine = engine
        self.model_id = model_id
        self.node_id = uuid.uuid4().hex
        self.model_loaded = False
        self.active_sessions = 0
        self.gpu_name, self.gpu_vram_mb = detect_gpu()
        # All outbound traffic funnels through one queue so generation threads,
        # heartbeats, and handlers never write to the socket concurrently.
        self.outbox: asyncio.Queue[dict] = asyncio.Queue()

    async def run(self):
        logger.info(f"Node agent starting (id={self.node_id[:8]})")
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
                    "mode": "whole_model",
                    "model_id": self.model_id,
                }))

                sender = asyncio.create_task(self._sender_loop(ws))
                heartbeat = asyncio.create_task(self._heartbeat_loop())
                try:
                    await self._message_loop(ws)
                except EngineError as e:
                    logger.error(f"Fatal: {e} — shutting down")
                    return
                finally:
                    sender.cancel()
                    heartbeat.cancel()

            except websockets.ConnectionClosed:
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
                "active_sessions": self.active_sessions,
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


async def main():
    url = os.environ.get("ORCHESTRATOR_URL", "ws://localhost:8080/nodes/ws")
    engine = create_engine(os.environ.get("NODE_ENGINE", "auto"))
    model_id = os.environ.get("NODE_MODEL", DEFAULT_MODEL)
    agent = NodeAgent(url, engine, model_id)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
