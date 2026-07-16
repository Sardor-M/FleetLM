from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = True

    # Node management
    heartbeat_timeout_sec: int = 15
    activation_timeout_sec: int = 10

    # Model config
    default_model: str = "llama-3-8b"
    total_layers: int = 32

    model_config = {"env_prefix": "DLLM_"}


settings = Settings()
