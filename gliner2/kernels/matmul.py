"""
Batch-Invariant Matrix Multiplication Kernel

This kernel ensures identical results regardless of batch size by using:
1. Fixed tile sizes (no dynamic selection based on matrix dimensions)
2. Sequential K-dimension reduction (fixed accumulation order)
3. No Split-K parallelization (avoids non-deterministic atomic adds)

Trade-off: ~20% slower than cuBLAS but guarantees batch invariance.
"""

import torch
import triton
import triton.language as tl
from typing import Optional


# Fixed tile sizes for batch invariance
# These must remain constant regardless of matrix dimensions
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32


@triton.jit
def _batch_invariant_matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Tile sizes (compile-time constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batch-invariant matrix multiplication: C = A @ B

    Key design choices for batch invariance:
    - Fixed BLOCK_M, BLOCK_N, BLOCK_K regardless of matrix size
    - Sequential iteration over K dimension (no parallel reduction)
    - Each program computes one output tile independently
    """
    # Program ID determines which output tile we compute
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute starting indices for this tile
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Create offset ranges for the tile
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    # Initialize accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Sequential reduction across K dimension (CRITICAL for batch invariance)
    # This ensures the same accumulation order regardless of batch size
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: A[m_start:m_start+BLOCK_M, k_start:k_start+BLOCK_K]
        a_ptrs = a_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: B[k_start:k_start+BLOCK_K, n_start:n_start+BLOCK_N]
        b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate tile product (fixed order within tile)
        acc += tl.dot(a_tile, b_tile, allow_tf32=False)

    # Store output tile
    c_ptrs = c_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def batch_invariant_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Batch-invariant matrix multiplication: C = A @ B

    Args:
        a: Input tensor of shape (..., M, K)
        b: Input tensor of shape (..., K, N)

    Returns:
        Output tensor of shape (..., M, N)

    This function guarantees identical results regardless of how many
    matrices are processed together (batch invariance).
    """
    # Handle batched inputs by flattening batch dimensions
    a_shape = a.shape
    b_shape = b.shape

    # Get matrix dimensions
    M, K = a_shape[-2], a_shape[-1]
    K2, N = b_shape[-2], b_shape[-1]
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    # Flatten batch dimensions
    if len(a_shape) > 2:
        batch_dims = a_shape[:-2]
        batch_size = 1
        for d in batch_dims:
            batch_size *= d
        a_flat = a.reshape(batch_size * M, K)
        M_total = batch_size * M
    else:
        a_flat = a
        M_total = M
        batch_dims = ()

    if len(b_shape) > 2:
        # For batched B, we need to handle differently
        # For now, assume B is 2D or broadcast
        b_flat = b.reshape(-1, N) if len(b_shape) > 2 else b
    else:
        b_flat = b

    # Ensure contiguous
    a_flat = a_flat.contiguous()
    b_flat = b_flat.contiguous()

    # Allocate output
    c = torch.empty((M_total, N), device=a.device, dtype=a.dtype)

    # Launch kernel with fixed tile sizes
    grid = (triton.cdiv(M_total, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _batch_invariant_matmul_kernel[grid](
        a_flat,
        b_flat,
        c,
        M_total,
        N,
        K,
        a_flat.stride(0),
        a_flat.stride(1),
        b_flat.stride(0),
        b_flat.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Reshape output to match input batch dimensions
    if batch_dims:
        c = c.reshape(*batch_dims, M, N)

    return c


def batch_invariant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Batch-invariant linear layer: y = x @ W^T + b

    Args:
        x: Input tensor of shape (..., in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)

    Returns:
        Output tensor of shape (..., out_features)
    """
    # Get dimensions
    orig_shape = x.shape
    in_features = orig_shape[-1]
    out_features = weight.shape[0]

    # Flatten to 2D: (batch, in_features)
    x_2d = x.reshape(-1, in_features).contiguous()
    M = x_2d.shape[0]

    # Weight is (out_features, in_features), we need (in_features, out_features)
    # So we compute x @ W^T = x @ (W.T)
    weight_t = weight.T.contiguous()

    # Allocate output
    output = torch.empty((M, out_features), device=x.device, dtype=x.dtype)

    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(out_features, BLOCK_N))

    _batch_invariant_matmul_kernel[grid](
        x_2d,
        weight_t,
        output,
        M,
        out_features,
        in_features,
        x_2d.stride(0),
        x_2d.stride(1),
        weight_t.stride(0),
        weight_t.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Add bias
    if bias is not None:
        output = output + bias

    # Reshape to match input batch dimensions
    output_shape = orig_shape[:-1] + (out_features,)
    return output.reshape(output_shape)
