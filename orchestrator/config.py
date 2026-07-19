from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = True

    # Node management
    heartbeat_timeout_sec: int = 15
    activation_timeout_sec: int = 10

    # Generation (whole-model nodes)
    generation_timeout_sec: int = 300  # overall cap per request
    chunk_timeout_sec: int = 120  # max silence between chunks (covers slow first token)

    # Batch work units
    lease_duration_sec: int = 600  # a lease this old is presumed dead and requeued
    lease_reaper_interval_sec: int = 30
    max_unit_attempts: int = 3  # then the unit is dead-lettered
    max_batch_requests: int = 10_000

    # Model config
    default_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    total_layers: int = 32

    model_config = {"env_prefix": "DLLM_"}


settings = Settings()
