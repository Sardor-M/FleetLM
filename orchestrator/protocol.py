"""Wire protocol shared by the orchestrator and its compute nodes.

Every node holds a whole model, so the protocol only has to express three
things: who is on the fleet, one interactive generation, and one batch work
unit. There is deliberately no layer-sharding vocabulary here — see README §2.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


# ── Enums ───────────────────────────────────────────────────────────────────

class NodeStatus(str, Enum):
    REGISTERING = "registering"
    READY = "ready"
    OFFLINE = "offline"


class SessionFailureCode(str, Enum):
    NO_CAPACITY = "no_capacity"
    NODE_ERROR = "node_error"


class MessageType(str, Enum):
    # Node -> Orchestrator
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    MODEL_LOADED = "model_loaded"
    GENERATE_CHUNK = "generate_chunk"
    GENERATE_COMPLETE = "generate_complete"
    GENERATE_ERROR = "generate_error"
    WORK_REQUEST = "work_request"
    WORK_RESULT = "work_result"
    WORK_FAILED = "work_failed"
    ERROR = "error"

    # Orchestrator -> Node
    SERVE_MODEL = "serve_model"
    GENERATE_REQUEST = "generate_request"
    WORK_ASSIGNMENT = "work_assignment"
    WORK_AVAILABLE = "work_available"
    SESSION_END = "session_end"


# ── Node -> Orchestrator ────────────────────────────────────────────────────

class RegisterMessage(BaseModel):
    type: str = MessageType.REGISTER
    node_id: str
    gpu_name: str = "unknown"
    gpu_vram_mb: int = 0
    runtime: str = "native"  # "native" | "webgpu"
    model_id: str | None = None
    join_token: str = ""


class HeartbeatMessage(BaseModel):
    type: str = MessageType.HEARTBEAT
    node_id: str
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    ram_usage: float = 0.0
    active_sessions: int = 0


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


class WorkRequestMessage(BaseModel):
    type: str = MessageType.WORK_REQUEST
    node_id: str
    capacity: int = 1


class WorkResultMessage(BaseModel):
    type: str = MessageType.WORK_RESULT
    node_id: str
    unit_id: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_sec: float = 0.0


class WorkFailedMessage(BaseModel):
    type: str = MessageType.WORK_FAILED
    node_id: str
    unit_id: str
    message: str = ""


# ── Orchestrator -> Node ────────────────────────────────────────────────────

class ServeModelMessage(BaseModel):
    type: str = MessageType.SERVE_MODEL
    model_id: str


class GenerateRequestMessage(BaseModel):
    type: str = MessageType.GENERATE_REQUEST
    session_id: str
    messages: list[dict]
    max_tokens: int = 256
    temperature: float = 0.7


class WorkAssignmentMessage(BaseModel):
    type: str = MessageType.WORK_ASSIGNMENT
    units: list[dict] = []


# ── Public API models ───────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
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


class BatchRequestItem(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    max_tokens: int = 256
    temperature: float = 0.7


class BatchCreateRequest(BaseModel):
    requests: list[BatchRequestItem]
    model: str | None = None  # default for items that don't name one
