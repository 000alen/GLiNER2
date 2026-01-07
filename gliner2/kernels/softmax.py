"""
Batch-Invariant Softmax Kernel

This kernel ensures identical softmax results regardless of batch size by:
1. Processing each row independently on a single thread block
2. Using fixed-order sequential reduction for max and sum
3. Never parallelizing reduction across multiple blocks

Critical for classification outputs where softmax determines class predictions.
"""

import torch
import triton
import triton.language as tl
from typing import Optional


BLOCK_SIZE = 1024


@triton.jit
def _batch_invariant_softmax_kernel(
    input_ptr,
    output_ptr,
    N,  # Number of elements per row
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Batch-invariant softmax kernel.

    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))

    Each row is processed independently with fixed-order reduction.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_input_row

    # =========================================================================
    # Pass 1: Find max with fixed-order reduction (for numerical stability)
    # =========================================================================
    max_val = tl.full((1,), float("-inf"), dtype=tl.float32)

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x_vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=float("-inf"))
        block_max = tl.max(x_vals.to(tl.float32), axis=0)
        max_val = tl.maximum(max_val, block_max)

    # =========================================================================
    # Pass 2: Compute sum of exp(x - max) with fixed-order reduction
    # =========================================================================
    sum_exp = tl.zeros((1,), dtype=tl.float32)

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x_vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=float("-inf"))
        exp_vals = tl.exp(x_vals.to(tl.float32) - max_val)

        # Mask out invalid positions for sum
        exp_vals = tl.where(mask, exp_vals, 0.0)
        sum_exp += tl.sum(exp_vals, axis=0)

    # =========================================================================
    # Pass 3: Compute and store softmax output
    # =========================================================================
    out_start = row_idx * stride_output_row

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x_vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=float("-inf"))
        exp_vals = tl.exp(x_vals.to(tl.float32) - max_val)
        softmax_vals = exp_vals / sum_exp

        tl.store(output_ptr + out_start + offsets, softmax_vals.to(output_ptr.dtype.element_ty), mask=mask)


def batch_invariant_softmax(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Batch-invariant softmax implementation.

    Args:
        x: Input tensor
        dim: Dimension along which to compute softmax

    Returns:
        Softmax output tensor of the same shape

    Guarantees identical results for each row regardless of total batch size.
    """
    # Handle negative dimensions
    if dim < 0:
        dim = x.dim() + dim

    # Move target dimension to last position for processing
    if dim != x.dim() - 1:
        x = x.transpose(dim, -1)
        transposed = True
    else:
        transposed = False

    # Flatten to 2D
    orig_shape = x.shape
    N = orig_shape[-1]
    num_rows = x.numel() // N
    x_2d = x.reshape(num_rows, N).contiguous()

    # Allocate output
    output = torch.empty_like(x_2d)

    # Determine block size
    block_size = min(BLOCK_SIZE, triton.next_power_of_2(N))

    # Launch one program per row
    grid = (num_rows,)

    _batch_invariant_softmax_kernel[grid](
        x_2d,
        output,
        N,
        x_2d.stride(0),
        output.stride(0),
        BLOCK_SIZE=block_size,
    )

    # Reshape back
    output = output.reshape(orig_shape)

    # Transpose back if needed
    if transposed:
        output = output.transpose(dim, -1)

    return output


@triton.jit
def _batch_invariant_log_softmax_kernel(
    input_ptr,
    output_ptr,
    N,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Batch-invariant log_softmax kernel.

    log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_input_row

    # Pass 1: Find max
    max_val = tl.full((1,), float("-inf"), dtype=tl.float32)

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x_vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=float("-inf"))
        block_max = tl.max(x_vals.to(tl.float32), axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Pass 2: Compute sum of exp(x - max)
    sum_exp = tl.zeros((1,), dtype=tl.float32)

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x_vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=float("-inf"))
        exp_vals = tl.exp(x_vals.to(tl.float32) - max_val)
        exp_vals = tl.where(mask, exp_vals, 0.0)
        sum_exp += tl.sum(exp_vals, axis=0)

    log_sum_exp = tl.log(sum_exp)

    # Pass 3: Compute log_softmax
    out_start = row_idx * stride_output_row

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x_vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=float("-inf"))
        log_softmax_vals = x_vals.to(tl.float32) - max_val - log_sum_exp
        tl.store(output_ptr + out_start + offsets, log_softmax_vals.to(output_ptr.dtype.element_ty), mask=mask)


def batch_invariant_log_softmax(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Batch-invariant log_softmax implementation.

    Args:
        x: Input tensor
        dim: Dimension along which to compute log_softmax

    Returns:
        Log softmax output tensor of the same shape
    """
    if dim < 0:
        dim = x.dim() + dim

    if dim != x.dim() - 1:
        x = x.transpose(dim, -1)
        transposed = True
    else:
        transposed = False

    orig_shape = x.shape
    N = orig_shape[-1]
    num_rows = x.numel() // N
    x_2d = x.reshape(num_rows, N).contiguous()

    output = torch.empty_like(x_2d)
    block_size = min(BLOCK_SIZE, triton.next_power_of_2(N))
    grid = (num_rows,)

    _batch_invariant_log_softmax_kernel[grid](
        x_2d,
        output,
        N,
        x_2d.stride(0),
        output.stride(0),
        BLOCK_SIZE=block_size,
    )

    output = output.reshape(orig_shape)

    if transposed:
        output = output.transpose(dim, -1)

    return output
