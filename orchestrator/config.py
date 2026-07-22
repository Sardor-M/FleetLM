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

    # Fleet access. Empty means an open fleet (fine locally, not on a public
    # host): set DLLM_JOIN_TOKEN so only invited machines can contribute.
    join_token: str = ""

    # Verifying work that ran on machines the operator does not control.
    # Off by default: canaries need reference answers recorded from a run the
    # operator trusts, and this cannot invent them. Point
    # DLLM_CANARY_FILE at a JSON list of {prompt, expected, model} to switch
    # it on. The threshold is an assumption, not a measurement - how far two
    # honest backends drift on identical input is still an open question.
    canary_file: str = ""
    canary_rate: float = 0.02  # canaries per real unit
    canary_agreement_threshold: float = 0.85

    # Model config
    default_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"

    model_config = {"env_prefix": "DLLM_"}


settings = Settings()
