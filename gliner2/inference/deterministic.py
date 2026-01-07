"""
Deterministic Inference Module for GLiNER2

This module provides batch-invariant inference capabilities that guarantee
identical outputs regardless of batch size. Based on research from
Thinking Machines Lab: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Usage:
    from gliner2.inference.deterministic import DeterministicMode

    # Enable globally
    with DeterministicMode():
        results = model.batch_extract(texts, schema)

    # Or per-call
    results = model.batch_extract(texts, schema, deterministic=True)
"""

import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Any
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicLevel(Enum):
    """Level of determinism to enforce."""

    OFF = "off"
    BASIC = "basic"  # PyTorch deterministic algorithms only
    BATCH_INVARIANT = "batch_invariant"  # Full batch invariance with custom kernels


# Global state for deterministic mode
_DETERMINISTIC_MODE: Optional[DeterministicLevel] = None
_ORIGINAL_FUNCTIONS: dict = {}


def _check_triton_available() -> bool:
    """Check if Triton is available for custom kernels."""
    try:
        import triton
        import triton.language as tl

        return True
    except ImportError:
        return False


def _check_cuda_available() -> bool:
    """Check if CUDA is available for GPU kernels."""
    return torch.cuda.is_available()


def _can_use_custom_kernels() -> bool:
    """Check if custom Triton kernels can be used (requires both Triton and CUDA)."""
    return _check_triton_available() and _check_cuda_available()


def _setup_pytorch_determinism():
    """Configure PyTorch for deterministic operations."""
    # Environment variable for cuBLAS determinism (only matters for CUDA)
    if _check_cuda_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Disable cuDNN benchmark (non-deterministic algorithm selection)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms (works for both CPU and CUDA)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _restore_pytorch_defaults():
    """Restore PyTorch default (non-deterministic) settings."""
    if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        del os.environ["CUBLAS_WORKSPACE_CONFIG"]

    torch.use_deterministic_algorithms(False)

    if _check_cuda_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _patch_torch_functions():
    """Patch PyTorch functions with batch-invariant implementations."""
    global _ORIGINAL_FUNCTIONS

    # Determine which implementations to use
    use_gpu_kernels = _can_use_custom_kernels()

    if use_gpu_kernels:
        from gliner2.kernels import (
            batch_invariant_linear,
            batch_invariant_layernorm,
            batch_invariant_softmax,
        )
        from gliner2.kernels.softmax import batch_invariant_log_softmax
    else:
        # Use CPU batch-invariant implementations
        from gliner2.kernels.cpu_batch_invariant import (
            cpu_batch_invariant_linear as batch_invariant_linear,
            cpu_batch_invariant_layernorm as batch_invariant_layernorm,
            cpu_batch_invariant_softmax as batch_invariant_softmax,
        )
        # CPU log_softmax (derive from softmax)
        batch_invariant_log_softmax = lambda x, dim=-1: torch.log(
            cpu_batch_invariant_softmax(x, dim) + 1e-10
        )

        if not _check_cuda_available():
            warnings.warn(
                "CUDA not available. Using CPU batch-invariant kernels. "
                "These are ~3-5x slower than GPU but provide full batch invariance."
            )
        elif not _check_triton_available():
            warnings.warn(
                "Triton not available but CUDA is. Using CPU batch-invariant kernels. "
                "Install Triton for faster GPU kernels: pip install triton"
            )

    # Save originals
    _ORIGINAL_FUNCTIONS["F.linear"] = F.linear
    _ORIGINAL_FUNCTIONS["F.layer_norm"] = F.layer_norm
    _ORIGINAL_FUNCTIONS["F.softmax"] = F.softmax
    _ORIGINAL_FUNCTIONS["F.log_softmax"] = F.log_softmax

    # Patch with batch-invariant versions
    def patched_linear(input, weight, bias=None):
        return batch_invariant_linear(input, weight, bias)

    def patched_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        return batch_invariant_layernorm(input, normalized_shape, weight, bias, eps)

    def patched_softmax(input, dim=None, _stacklevel=3, dtype=None):
        if dtype is not None:
            input = input.to(dtype)
        if dim is None:
            dim = -1
        return batch_invariant_softmax(input, dim)

    def patched_log_softmax(input, dim=None, _stacklevel=3, dtype=None):
        if dtype is not None:
            input = input.to(dtype)
        if dim is None:
            dim = -1
        return batch_invariant_log_softmax(input, dim)

    F.linear = patched_linear
    F.layer_norm = patched_layer_norm
    F.softmax = patched_softmax
    F.log_softmax = patched_log_softmax


def _unpatch_torch_functions():
    """Restore original PyTorch functions."""
    global _ORIGINAL_FUNCTIONS

    if "F.linear" in _ORIGINAL_FUNCTIONS:
        F.linear = _ORIGINAL_FUNCTIONS["F.linear"]
    if "F.layer_norm" in _ORIGINAL_FUNCTIONS:
        F.layer_norm = _ORIGINAL_FUNCTIONS["F.layer_norm"]
    if "F.softmax" in _ORIGINAL_FUNCTIONS:
        F.softmax = _ORIGINAL_FUNCTIONS["F.softmax"]
    if "F.log_softmax" in _ORIGINAL_FUNCTIONS:
        F.log_softmax = _ORIGINAL_FUNCTIONS["F.log_softmax"]

    _ORIGINAL_FUNCTIONS.clear()


@dataclass
class DeterministicConfig:
    """Configuration for deterministic inference."""

    level: DeterministicLevel = DeterministicLevel.BATCH_INVARIANT
    warn_on_fallback: bool = True

    def __post_init__(self):
        if isinstance(self.level, str):
            self.level = DeterministicLevel(self.level)


class DeterministicMode:
    """
    Context manager for enabling deterministic/batch-invariant inference.

    Usage:
        with DeterministicMode():
            # All operations here are batch-invariant
            result = model(input)

        # Or with specific level
        with DeterministicMode(level="basic"):
            result = model(input)
    """

    def __init__(
        self,
        level: DeterministicLevel | str = DeterministicLevel.BATCH_INVARIANT,
        warn_on_fallback: bool = True,
    ):
        if isinstance(level, str):
            level = DeterministicLevel(level)
        self.level = level
        self.warn_on_fallback = warn_on_fallback
        self._previous_level: Optional[DeterministicLevel] = None

    def __enter__(self):
        global _DETERMINISTIC_MODE
        self._previous_level = _DETERMINISTIC_MODE
        _DETERMINISTIC_MODE = self.level

        if self.level == DeterministicLevel.OFF:
            return self

        # Setup PyTorch determinism
        _setup_pytorch_determinism()

        # Setup batch-invariant kernels if requested
        if self.level == DeterministicLevel.BATCH_INVARIANT:
            _patch_torch_functions()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _DETERMINISTIC_MODE

        if self.level != DeterministicLevel.OFF:
            # Restore PyTorch defaults
            _restore_pytorch_defaults()

            # Unpatch functions
            if self.level == DeterministicLevel.BATCH_INVARIANT:
                _unpatch_torch_functions()

        _DETERMINISTIC_MODE = self._previous_level
        return False


def get_deterministic_mode() -> Optional[DeterministicLevel]:
    """Get the current deterministic mode level."""
    return _DETERMINISTIC_MODE


def is_deterministic_mode() -> bool:
    """Check if deterministic mode is currently enabled."""
    return _DETERMINISTIC_MODE is not None and _DETERMINISTIC_MODE != DeterministicLevel.OFF


def is_batch_invariant_mode() -> bool:
    """Check if batch-invariant mode is currently enabled."""
    # Now supports both GPU (Triton) and CPU batch-invariant kernels
    return _DETERMINISTIC_MODE == DeterministicLevel.BATCH_INVARIANT


def can_use_gpu_batch_invariant_kernels() -> bool:
    """Check if GPU batch-invariant kernels are available (requires CUDA + Triton)."""
    return _can_use_custom_kernels()


def can_use_batch_invariant_kernels() -> bool:
    """Check if batch-invariant kernels are available (always True - CPU fallback exists)."""
    return True  # CPU batch-invariant kernels are always available


@contextmanager
def deterministic_inference(level: DeterministicLevel | str = DeterministicLevel.BATCH_INVARIANT):
    """
    Context manager for deterministic inference.

    Alias for DeterministicMode for more explicit naming.
    """
    with DeterministicMode(level=level) as ctx:
        yield ctx


def batch_invariant_einsum(
    equation: str,
    *operands: torch.Tensor,
) -> torch.Tensor:
    """
    Batch-invariant einsum for common patterns.

    Currently supports:
    - 'lkd,bpd->bplk' (span scoring) - GPU and CPU
    - 'bmd,nd->bmn' (batched bilinear) - GPU only

    Falls back to torch.einsum for unsupported patterns.
    """
    # Normalize equation (remove spaces)
    equation = equation.replace(" ", "")

    # Check if tensors are on GPU and we have Triton
    on_gpu = all(op.is_cuda for op in operands)
    use_gpu_kernels = on_gpu and _can_use_custom_kernels()

    if equation == "lkd,bpd->bplk" and len(operands) == 2:
        if use_gpu_kernels:
            from gliner2.kernels import batch_invariant_span_score
            return batch_invariant_span_score(operands[0], operands[1])
        else:
            # Use CPU batch-invariant implementation
            from gliner2.kernels.cpu_batch_invariant import cpu_batch_invariant_span_score_fast
            return cpu_batch_invariant_span_score_fast(operands[0], operands[1])

    elif equation == "bmd,nd->bmn" and len(operands) == 2:
        if use_gpu_kernels:
            from gliner2.kernels.span_score import batch_invariant_einsum_bmd_nd
            return batch_invariant_einsum_bmd_nd(operands[0], operands[1])
        else:
            # CPU fallback - process per batch element for invariance
            a, b = operands  # (B, M, D), (N, D)
            B, M, D = a.shape
            N = b.shape[0]
            output = torch.empty((B, M, N), dtype=a.dtype, device=a.device)
            for batch_idx in range(B):
                # Each batch uses same reduction order
                output[batch_idx] = a[batch_idx].float() @ b.T.float()
            return output.to(a.dtype)

    else:
        # Fallback to standard einsum
        if _DETERMINISTIC_MODE == DeterministicLevel.BATCH_INVARIANT:
            warnings.warn(
                f"No batch-invariant kernel for einsum pattern '{equation}'. "
                "Falling back to torch.einsum which may not be batch-invariant."
            )
        return torch.einsum(equation, *operands)


class BatchInvariantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with batch-invariant computation.

    Usage:
        # Replace existing linear layer
        model.fc = BatchInvariantLinear.from_linear(model.fc)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if is_batch_invariant_mode():
            if input.is_cuda and _can_use_custom_kernels():
                from gliner2.kernels import batch_invariant_linear
                return batch_invariant_linear(input, self.weight, self.bias)
            else:
                # Use CPU batch-invariant implementation
                from gliner2.kernels.cpu_batch_invariant import cpu_batch_invariant_linear
                return cpu_batch_invariant_linear(input, self.weight, self.bias)
        return F.linear(input, self.weight, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "BatchInvariantLinear":
        """Create BatchInvariantLinear from existing nn.Linear."""
        bi_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        bi_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            bi_linear.bias.data.copy_(linear.bias.data)
        return bi_linear


class BatchInvariantLayerNorm(nn.Module):
    """
    Drop-in replacement for nn.LayerNorm with batch-invariant computation.
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=device, dtype=dtype)
            )
            self.bias = nn.Parameter(
                torch.zeros(self.normalized_shape, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if is_batch_invariant_mode():
            if input.is_cuda and _can_use_custom_kernels():
                from gliner2.kernels import batch_invariant_layernorm
                return batch_invariant_layernorm(
                    input, self.normalized_shape, self.weight, self.bias, self.eps
                )
            else:
                # Use CPU batch-invariant implementation
                from gliner2.kernels.cpu_batch_invariant import cpu_batch_invariant_layernorm
                return cpu_batch_invariant_layernorm(
                    input, self.normalized_shape, self.weight, self.bias, self.eps
                )
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    @classmethod
    def from_layer_norm(cls, layer_norm: nn.LayerNorm) -> "BatchInvariantLayerNorm":
        """Create BatchInvariantLayerNorm from existing nn.LayerNorm."""
        bi_ln = cls(
            layer_norm.normalized_shape,
            eps=layer_norm.eps,
            elementwise_affine=layer_norm.elementwise_affine,
            device=layer_norm.weight.device if layer_norm.weight is not None else None,
            dtype=layer_norm.weight.dtype if layer_norm.weight is not None else None,
        )
        if layer_norm.weight is not None:
            bi_ln.weight.data.copy_(layer_norm.weight.data)
        if layer_norm.bias is not None:
            bi_ln.bias.data.copy_(layer_norm.bias.data)
        return bi_ln
