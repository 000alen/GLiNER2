"""
Batch-Invariant LayerNorm Kernel

This kernel ensures identical results regardless of batch size by:
1. Processing each row independently on a single thread block
2. Using fixed-order sequential reduction for mean and variance
3. Never parallelizing reduction across multiple blocks

Based on Thinking Machines Lab's approach where "each batch element is
processed on a single compute core, eliminating inter-core communication
for feature-dimension reductions."
"""

import torch
import triton
import triton.language as tl
from typing import Optional, List, Union


# Block size for processing hidden dimension
# Must be fixed regardless of actual hidden size
BLOCK_SIZE = 1024


@triton.jit
def _batch_invariant_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,  # Hidden dimension (normalized dimension)
    eps,
    stride_x_row,
    stride_out_row,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Batch-invariant LayerNorm kernel.

    Each program (thread block) handles exactly one row.
    Reduction is done with fixed sequential ordering within the row.
    This ensures the same floating-point accumulation order regardless
    of how many rows (batch size) we're processing.
    """
    # Each program handles one row
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x_row

    # =========================================================================
    # Pass 1: Compute mean with fixed-order reduction
    # =========================================================================
    mean_acc = tl.zeros((1,), dtype=tl.float32)

    # Process in fixed-size blocks, sequentially
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Load values
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)

        # Accumulate sum (fixed order within block via tl.sum)
        mean_acc += tl.sum(x_vals.to(tl.float32), axis=0)

    mean = mean_acc / N

    # =========================================================================
    # Pass 2: Compute variance with fixed-order reduction
    # =========================================================================
    var_acc = tl.zeros((1,), dtype=tl.float32)

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)

        # Compute squared differences
        diff = x_vals.to(tl.float32) - mean
        var_acc += tl.sum(diff * diff, axis=0)

    var = var_acc / N
    rstd = tl.math.rsqrt(var + eps)

    # =========================================================================
    # Pass 3: Normalize and apply affine transform
    # =========================================================================
    out_start = row_idx * stride_out_row

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Load input
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)

        # Normalize
        normalized = (x_vals.to(tl.float32) - mean) * rstd

        # Apply weight and bias if provided
        if HAS_WEIGHT:
            weight_vals = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
            normalized = normalized * weight_vals

        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
            normalized = normalized + bias_vals

        # Store output
        tl.store(output_ptr + out_start + offsets, normalized.to(x_ptr.dtype.element_ty), mask=mask)


def batch_invariant_layernorm(
    x: torch.Tensor,
    normalized_shape: Union[int, List[int]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Batch-invariant LayerNorm implementation.

    Args:
        x: Input tensor of shape (..., *normalized_shape)
        normalized_shape: Shape of the normalized dimensions
        weight: Optional weight tensor of shape normalized_shape
        bias: Optional bias tensor of shape normalized_shape
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of the same shape as input

    This function guarantees identical results for each input row
    regardless of how many rows are processed together.
    """
    # Handle normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    else:
        normalized_shape = list(normalized_shape)

    # Compute the size of normalized dimensions
    N = 1
    for s in normalized_shape:
        N *= s

    # Flatten input to 2D: (num_rows, N)
    orig_shape = x.shape
    num_rows = x.numel() // N
    x_2d = x.reshape(num_rows, N).contiguous()

    # Allocate output
    output = torch.empty_like(x_2d)

    # Determine block size (must be power of 2 for Triton)
    block_size = min(BLOCK_SIZE, triton.next_power_of_2(N))

    # Launch one program per row (critical for batch invariance)
    grid = (num_rows,)

    _batch_invariant_layernorm_kernel[grid](
        x_2d,
        weight if weight is not None else x_2d,  # Dummy pointer if no weight
        bias if bias is not None else x_2d,  # Dummy pointer if no bias
        output,
        N,
        eps,
        x_2d.stride(0),
        output.stride(0),
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE=block_size,
    )

    return output.reshape(orig_shape)


@triton.jit
def _batch_invariant_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    N,
    eps,
    stride_x_row,
    stride_out_row,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Batch-invariant RMSNorm kernel (no mean subtraction).

    RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x_row

    # Compute mean of squares with fixed-order reduction
    ms_acc = tl.zeros((1,), dtype=tl.float32)

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        ms_acc += tl.sum(x_vals.to(tl.float32) * x_vals.to(tl.float32), axis=0)

    ms = ms_acc / N
    rstd = tl.math.rsqrt(ms + eps)

    # Normalize and apply weight
    out_start = row_idx * stride_out_row

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        normalized = x_vals.to(tl.float32) * rstd

        if HAS_WEIGHT:
            weight_vals = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
            normalized = normalized * weight_vals

        tl.store(output_ptr + out_start + offsets, normalized.to(x_ptr.dtype.element_ty), mask=mask)


def batch_invariant_rmsnorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Batch-invariant RMSNorm implementation.

    Args:
        x: Input tensor of shape (..., hidden_size)
        weight: Optional weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of the same shape as input
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    num_rows = x.numel() // N

    x_2d = x.reshape(num_rows, N).contiguous()
    output = torch.empty_like(x_2d)

    block_size = min(BLOCK_SIZE, triton.next_power_of_2(N))
    grid = (num_rows,)

    _batch_invariant_rmsnorm_kernel[grid](
        x_2d,
        weight if weight is not None else x_2d,
        output,
        N,
        eps,
        x_2d.stride(0),
        output.stride(0),
        HAS_WEIGHT=weight is not None,
        BLOCK_SIZE=block_size,
    )

    return output.reshape(orig_shape)
