"""llama-cpp-python wrapper for native inference.

TODO Phase 2: Integrate actual model loading and layer-by-layer inference.

Usage:
    pip install llama-cpp-python

    # macOS Metal:
    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

    # Linux CUDA:
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("node_agent.engine")


class LlamaEngine:
    """Wrapper around llama-cpp-python for layer-level inference.

    Phase 1: Placeholder with simulated outputs.
    Phase 2: Load actual GGUF model and run specific layers.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = model_path
        self.model = None
        self.loaded_layers: tuple[int, int] | None = None

    def load(self, start_layer: int, end_layer: int, weight_paths: list[str] | None = None):
        """Load model weights for the specified layer range."""
        self.loaded_layers = (start_layer, end_layer)

        if self.model_path:
            try:
                from llama_cpp import Llama

                # TODO: llama-cpp-python doesn't natively support loading
                # a subset of layers. Options:
                # 1. Load full model but only run specific layers
                # 2. Use custom GGUF shards (one per layer range)
                # 3. Use PyTorch with safetensors for more control

                self.model = Llama(
                    model_path=str(self.model_path),
                    n_gpu_layers=-1,  # All layers on GPU
                    n_ctx=2048,
                    verbose=False,
                )
                logger.info(f"Model loaded from {self.model_path}")

            except ImportError:
                logger.warning("llama-cpp-python not installed, using placeholder")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.info(f"No model path, using placeholder for layers {start_layer}-{end_layer}")

    def run_layers(
        self,
        hidden_states: np.ndarray,
        is_prefill: bool = False,
    ) -> np.ndarray:
        """Run the assigned layers on input hidden states.

        Args:
            hidden_states: Input tensor [seq_len, hidden_dim] or [1, hidden_dim]
            is_prefill: True for prefill (full sequence), False for single-token decode

        Returns:
            Output hidden states after running through assigned layers.
        """
        if self.model is not None:
            # TODO: Hook into llama.cpp internals to run specific layers
            # This requires modifying llama-cpp-python or using the C API directly
            pass

        # Placeholder: return random hidden states with correct shape
        seq_len = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1] if len(hidden_states.shape) > 1 else 4096
        return np.random.randn(seq_len, hidden_dim).astype(np.float32)

    def run_lm_head(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run the final LM head to produce logits (only on last pipeline stage).

        Returns:
            Logits array [vocab_size]
        """
        # Placeholder
        vocab_size = 128256  # Llama-3 vocab
        return np.random.randn(vocab_size).astype(np.float32)
