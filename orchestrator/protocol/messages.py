"""Wire protocol message types shared between orchestrator and compute nodes."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel


# ── Enums ───────────────────────────────────────────────────────────────────

class NodeStatus(str, Enum):
    REGISTERING = "registering"
    DOWNLOADING = "downloading"
    READY = "ready"
    BUSY = "busy"
    OFFLINE = "offline"


class MessageType(str, Enum):
    # Node -> Orchestrator
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    LAYERS_LOADED = "layers_loaded"
    ACTIVATION_RESULT = "activation_result"
    ERROR = "error"

    # Orchestrator -> Node
    LAYER_ASSIGNMENT = "layer_assignment"
    PREFILL_REQUEST = "prefill_request"
    DECODE_REQUEST = "decode_request"
    SESSION_END = "session_end"


# ── Node -> Orchestrator Messages ───────────────────────────────────────────

class RegisterMessage(BaseModel):
    type: str = MessageType.REGISTER
    node_id: str
    gpu_name: str = "unknown"
    gpu_vram_mb: int = 0
    runtime: str = "webgpu"  # "webgpu" | "webnn" | "native"


class HeartbeatMessage(BaseModel):
    type: str = MessageType.HEARTBEAT
    node_id: str
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    ram_usage: float = 0.0
    active_sessions: int = 0


class LayersLoadedMessage(BaseModel):
    type: str = MessageType.LAYERS_LOADED
    node_id: str
    start_layer: int
    end_layer: int


class ActivationResultMessage(BaseModel):
    type: str = MessageType.ACTIVATION_RESULT
    session_id: str
    shape: list[int]  # e.g. [1, 4096]
    dtype: str = "float16"
    # actual tensor data sent as binary WebSocket frame after this JSON frame


# ── Orchestrator -> Node Messages ───────────────────────────────────────────

class LayerAssignment(BaseModel):
    type: str = MessageType.LAYER_ASSIGNMENT
    model_id: str
    start_layer: int
    end_layer: int
    weight_shard_urls: list[str] = []


class PrefillRequest(BaseModel):
    type: str = MessageType.PREFILL_REQUEST
    session_id: str
    tokens: list[int]


class DecodeRequest(BaseModel):
    type: str = MessageType.DECODE_REQUEST
    session_id: str
    token: int


# ── B2B API Models ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llama-3-8b"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Usage()
