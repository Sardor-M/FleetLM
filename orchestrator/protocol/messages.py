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


class NodeMode(str, Enum):
    # The node holds the entire model and serves complete generations.
    WHOLE_MODEL = "whole_model"
    # The node holds a range of transformer layers (pipeline-parallel path).
    LAYER_SHARD = "layer_shard"


class SessionFailureCode(str, Enum):
    NO_CAPACITY = "no_capacity"
    NODE_ERROR = "node_error"


class MessageType(str, Enum):
    # Node -> Orchestrator
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    LAYERS_LOADED = "layers_loaded"
    MODEL_LOADED = "model_loaded"
    ACTIVATION_RESULT = "activation_result"
    GENERATE_CHUNK = "generate_chunk"
    GENERATE_COMPLETE = "generate_complete"
    GENERATE_ERROR = "generate_error"
    ERROR = "error"

    # Orchestrator -> Node
    LAYER_ASSIGNMENT = "layer_assignment"
    SERVE_MODEL = "serve_model"
    GENERATE_REQUEST = "generate_request"
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
    mode: str = NodeMode.LAYER_SHARD  # "whole_model" | "layer_shard"
    model_id: str | None = None  # for whole_model: the model this node serves


class ModelLoadedMessage(BaseModel):
    type: str = MessageType.MODEL_LOADED
    node_id: str
    model_id: str


class GenerateChunkMessage(BaseModel):
    type: str = MessageType.GENERATE_CHUNK
    session_id: str
    text: str


class GenerateCompleteMessage(BaseModel):
    type: str = MessageType.GENERATE_COMPLETE
    session_id: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class GenerateErrorMessage(BaseModel):
    type: str = MessageType.GENERATE_ERROR
    session_id: str
    message: str = ""


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


class ServeModelMessage(BaseModel):
    type: str = MessageType.SERVE_MODEL
    model_id: str


class GenerateRequestMessage(BaseModel):
    type: str = MessageType.GENERATE_REQUEST
    session_id: str
    messages: list[dict]
    max_tokens: int = 256
    temperature: float = 0.7


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
