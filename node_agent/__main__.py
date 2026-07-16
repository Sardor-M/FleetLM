"""Native compute node agent (Python).

This is the optional native path for power users who want 100% GPU performance
via llama-cpp-python instead of ~80% via browser WebGPU.

Usage:
    python -m node_agent
    ORCHESTRATOR_URL=ws://server:8080/nodes/ws python -m node_agent
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import uuid

import numpy as np
import psutil
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("node_agent")


def detect_gpu() -> tuple[str, int]:
    """Detect GPU type and approximate VRAM."""
    if platform.system() == "Darwin":
        total_ram = psutil.virtual_memory().total // (1024 * 1024)
        vram = int(total_ram * 0.75)
        return "apple-silicon", vram

    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return f"nvidia-{gpu.name.lower().replace(' ', '-')}", int(gpu.memoryTotal)
    except ImportError:
        pass

    return "cpu-only", 0


class NativeNodeAgent:
    def __init__(self, orchestrator_url: str):
        self.orchestrator_url = orchestrator_url
        self.node_id = uuid.uuid4().hex
        self.ws = None
        self.running = True
        self.assigned_layers: tuple[int, int] | None = None

        gpu_name, vram = detect_gpu()
        self.gpu_name = gpu_name
        self.gpu_vram_mb = vram

    async def run(self):
        logger.info(f"Node agent starting (id={self.node_id[:8]})")
        logger.info(f"GPU: {self.gpu_name}, VRAM: {self.gpu_vram_mb} MB")
        logger.info(f"CPU: {psutil.cpu_count()} cores, RAM: {psutil.virtual_memory().total // (1024**3)} GB")

        async for ws in websockets.connect(self.orchestrator_url):
            try:
                self.ws = ws
                logger.info("Connected to orchestrator")

                # Register
                await ws.send(json.dumps({
                    "type": "register",
                    "node_id": self.node_id,
                    "gpu_name": self.gpu_name,
                    "gpu_vram_mb": self.gpu_vram_mb,
                    "runtime": "native",
                }))

                # Run heartbeat + message handler
                await asyncio.gather(
                    self._heartbeat_loop(),
                    self._message_loop(),
                )

            except websockets.ConnectionClosed:
                logger.warning("Connection lost, reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _heartbeat_loop(self):
        while self.running:
            try:
                await self.ws.send(json.dumps({
                    "type": "heartbeat",
                    "node_id": self.node_id,
                    "cpu_usage": psutil.cpu_percent(),
                    "gpu_usage": 0.0,
                    "ram_usage": psutil.virtual_memory().percent,
                    "active_sessions": 0,
                }))
            except Exception:
                break
            await asyncio.sleep(5)

    async def _message_loop(self):
        async for raw in self.ws:
            if isinstance(raw, str):
                msg = json.loads(raw)
                await self._handle(msg)
            else:
                # Binary = activation tensor
                arr = np.frombuffer(raw, dtype=np.float32)
                logger.info(f"Received activation: {arr.shape} ({len(raw)} bytes)")

    async def _handle(self, msg: dict):
        t = msg.get("type")

        if t == "layer_assignment":
            self.assigned_layers = (msg["start_layer"], msg["end_layer"])
            logger.info(f"Assigned layers {self.assigned_layers[0]}-{self.assigned_layers[1]}")

            # TODO: load actual model weights via llama-cpp-python
            # from llama_cpp import Llama
            # self.model = Llama(model_path="...", n_gpu_layers=...)

            # Simulate loading
            await asyncio.sleep(1)

            await self.ws.send(json.dumps({
                "type": "layers_loaded",
                "node_id": self.node_id,
                "start_layer": self.assigned_layers[0],
                "end_layer": self.assigned_layers[1],
            }))
            logger.info("Layers loaded, ready for inference")

        elif t == "prefill_request":
            session_id = msg["session_id"]
            tokens = msg["tokens"]
            logger.info(f"Prefill: session={session_id}, {len(tokens)} tokens")

            # TODO: run actual inference through assigned layers
            hidden = np.random.randn(len(tokens), 4096).astype(np.float32)

            await self.ws.send(json.dumps({
                "type": "activation_result",
                "session_id": session_id,
                "shape": list(hidden.shape),
                "dtype": "float32",
            }))
            await self.ws.send(hidden.tobytes())

        elif t == "decode_request":
            session_id = msg["session_id"]

            # TODO: run single-token inference
            hidden = np.random.randn(1, 4096).astype(np.float32)

            await self.ws.send(json.dumps({
                "type": "activation_result",
                "session_id": session_id,
                "shape": [1, 4096],
                "dtype": "float32",
            }))
            await self.ws.send(hidden.tobytes())


async def main():
    url = os.environ.get("ORCHESTRATOR_URL", "ws://localhost:8080/nodes/ws")
    agent = NativeNodeAgent(url)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
