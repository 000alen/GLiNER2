"""
CPU Batch-Invariant Operations

These implementations provide batch invariance on CPU by enforcing fixed
reduction ordering. Based on techniques from:
- https://dtunai.blog/blog/reproducing-batch-invariant-ops-rmsnorm-and-matmul-learning-log-i
- https://arxiv.org/html/2511.00025

Key insight: Standard BLAS libraries (MKL, OpenBLAS) use different reduction
strategies based on matrix dimensions, causing batch-dependent numerical
differences. By using explicit sequential reduction, we ensure identical
results regardless of batch size.

Trade-off: ~3-5x slower than optimized BLAS, but guarantees batch invariance.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Union


# Tile sizes for CPU batch-invariant operations
# Smaller tiles for CPU cache efficiency
CPU_TILE_M = 32
CPU_TILE_N = 32
CPU_TILE_K = 32


def cpu_batch_invariant_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    CPU batch-invariant matrix multiplication using tiled sequential reduction.

    This function guarantees identical results regardless of batch size by:
    1. Using fixed tile sizes
    2. Accumulating K-dimension tiles in strict sequential order
    3. Processing each output tile independently

    Args:
        a: Input tensor of shape (..., M, K)
        b: Input tensor of shape (..., K, N)

    Returns:
        Output tensor of shape (..., M, N)

    Note:
        ~3-5x slower than torch.matmul, but batch-invariant.
    """
    # Get shapes
    *batch_a, M, K = a.shape
    *batch_b, K2, N = b.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    # Handle batched case by flattening
    if batch_a or batch_b:
        # For simplicity, use the sequential approach for batched inputs
        a_2d = a.reshape(-1, K) if batch_a else a
        b_2d = b.reshape(K, -1) if batch_b else b

        if batch_a:
            batch_size = a_2d.shape[0] // M
            a_2d = a.reshape(batch_size, M, K)
            # Process each batch element with the same reduction order
            results = []
            for i in range(batch_size):
                result = _tiled_matmul_sequential(a_2d[i], b)
                results.append(result)
            output = torch.stack(results)
            return output.reshape(*batch_a, M, N)

    return _tiled_matmul_sequential(a, b)


def _tiled_matmul_sequential(
    a: torch.Tensor,  # (M, K)
    b: torch.Tensor,  # (K, N)
) -> torch.Tensor:
    """
    Tiled matrix multiplication with sequential K-reduction.

    Forces a fixed accumulation order: tiles are processed left-to-right
    along K dimension, ensuring identical floating-point results.
    """
    M, K = a.shape
    K2, N = b.shape

    # Pad to tile boundaries if needed
    M_pad = (CPU_TILE_M - M % CPU_TILE_M) % CPU_TILE_M
    N_pad = (CPU_TILE_N - N % CPU_TILE_N) % CPU_TILE_N
    K_pad = (CPU_TILE_K - K % CPU_TILE_K) % CPU_TILE_K

    if M_pad or K_pad:
        a = F.pad(a, (0, K_pad, 0, M_pad))
    if K_pad or N_pad:
        b = F.pad(b, (0, N_pad, 0, K_pad))

    M_padded, K_padded = a.shape
    _, N_padded = b.shape

    # Number of tiles
    n_m_tiles = M_padded // CPU_TILE_M
    n_n_tiles = N_padded // CPU_TILE_N
    n_k_tiles = K_padded // CPU_TILE_K

    # Output accumulator (use float32 for stability)
    output = torch.zeros((M_padded, N_padded), dtype=torch.float32, device=a.device)

    # Sequential tile accumulation (fixed order for batch invariance)
    for mi in range(n_m_tiles):
        m_start = mi * CPU_TILE_M
        m_end = m_start + CPU_TILE_M

        for ni in range(n_n_tiles):
            n_start = ni * CPU_TILE_N
            n_end = n_start + CPU_TILE_N

            # Accumulate across K tiles in FIXED ORDER (critical for batch invariance)
            tile_acc = torch.zeros((CPU_TILE_M, CPU_TILE_N), dtype=torch.float32, device=a.device)

            for ki in range(n_k_tiles):
                k_start = ki * CPU_TILE_K
                k_end = k_start + CPU_TILE_K

                # Extract tiles
                a_tile = a[m_start:m_end, k_start:k_end].float()
                b_tile = b[k_start:k_end, n_start:n_end].float()

                # Accumulate (order within tile is also fixed by torch.mm)
                tile_acc = tile_acc + torch.mm(a_tile, b_tile)

            output[m_start:m_end, n_start:n_end] = tile_acc

    # Remove padding and convert back to original dtype
    return output[:M, :N].to(a.dtype)


def cpu_batch_invariant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    CPU batch-invariant linear layer: y = x @ W^T + b

    Processes each sample with identical reduction order.
    """
    orig_shape = x.shape
    in_features = orig_shape[-1]
    out_features = weight.shape[0]

    # Flatten to (batch, in_features)
    x_2d = x.reshape(-1, in_features)
    batch_size = x_2d.shape[0]

    # Process each sample with same reduction order
    # This ensures batch invariance
    results = []
    weight_t = weight.T.contiguous()  # (in_features, out_features)

    for i in range(batch_size):
        # Single sample matmul with fixed reduction
        result = _sequential_vector_matrix(x_2d[i], weight_t)
        results.append(result)

    output = torch.stack(results)

    if bias is not None:
        output = output + bias

    return output.reshape(*orig_shape[:-1], out_features)


def _sequential_vector_matrix(
    v: torch.Tensor,  # (K,)
    m: torch.Tensor,  # (K, N)
) -> torch.Tensor:
    """
    Vector-matrix multiplication with sequential reduction.

    Computes v @ m with fixed accumulation order across K dimension.
    """
    K, N = m.shape

    # Tile K dimension for sequential accumulation
    n_k_tiles = (K + CPU_TILE_K - 1) // CPU_TILE_K

    output = torch.zeros(N, dtype=torch.float32, device=v.device)

    for ki in range(n_k_tiles):
        k_start = ki * CPU_TILE_K
        k_end = min(k_start + CPU_TILE_K, K)

        # Extract tile
        v_tile = v[k_start:k_end].float()
        m_tile = m[k_start:k_end, :].float()

        # Accumulate (fixed order)
        output = output + v_tile @ m_tile

    return output.to(v.dtype)


def cpu_batch_invariant_layernorm(
    x: torch.Tensor,
    normalized_shape: Union[int, List[int]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    CPU batch-invariant LayerNorm.

    Processes each row with identical sequential reduction for mean and variance.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]

    N = 1
    for s in normalized_shape:
        N *= s

    orig_shape = x.shape
    x_2d = x.reshape(-1, N)
    batch_size = x_2d.shape[0]

    output = torch.empty_like(x_2d)

    for i in range(batch_size):
        row = x_2d[i].float()

        # Sequential mean computation (fixed order)
        mean = _sequential_sum(row) / N

        # Sequential variance computation (fixed order)
        diff = row - mean
        var = _sequential_sum(diff * diff) / N

        # Normalize
        rstd = torch.rsqrt(var + eps)
        normalized = diff * rstd

        # Apply affine transform
        if weight is not None:
            normalized = normalized * weight.float()
        if bias is not None:
            normalized = normalized + bias.float()

        output[i] = normalized.to(x.dtype)

    return output.reshape(orig_shape)


def _sequential_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Sequential sum with fixed accumulation order.

    Uses tiled reduction to ensure identical results regardless of tensor size.
    """
    N = x.numel()
    x_flat = x.flatten()

    # Tile size for sequential accumulation
    tile_size = 64

    acc = torch.tensor(0.0, dtype=torch.float32, device=x.device)

    for start in range(0, N, tile_size):
        end = min(start + tile_size, N)
        # Sum within tile (small enough to be deterministic)
        tile_sum = x_flat[start:end].sum()
        acc = acc + tile_sum

    return acc


def cpu_batch_invariant_softmax(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    CPU batch-invariant softmax.

    Each row processed with identical sequential reduction for max and sum.
    """
    if dim < 0:
        dim = x.dim() + dim

    # Move target dim to last
    if dim != x.dim() - 1:
        x = x.transpose(dim, -1)
        transposed = True
    else:
        transposed = False

    orig_shape = x.shape
    N = orig_shape[-1]
    x_2d = x.reshape(-1, N)
    batch_size = x_2d.shape[0]

    output = torch.empty_like(x_2d)

    for i in range(batch_size):
        row = x_2d[i].float()

        # Sequential max (for numerical stability)
        max_val = _sequential_max(row)

        # Compute exp(x - max)
        exp_x = torch.exp(row - max_val)

        # Sequential sum of exp
        sum_exp = _sequential_sum(exp_x)

        # Normalize
        output[i] = (exp_x / sum_exp).to(x.dtype)

    output = output.reshape(orig_shape)

    if transposed:
        output = output.transpose(dim, -1)

    return output


def _sequential_max(x: torch.Tensor) -> torch.Tensor:
    """Sequential max with fixed comparison order."""
    N = x.numel()
    x_flat = x.flatten()

    tile_size = 64
    max_val = torch.tensor(float('-inf'), dtype=torch.float32, device=x.device)

    for start in range(0, N, tile_size):
        end = min(start + tile_size, N)
        tile_max = x_flat[start:end].max()
        max_val = torch.maximum(max_val, tile_max)

    return max_val


def cpu_batch_invariant_span_score(
    span_rep: torch.Tensor,  # (L, K, D)
    struct_proj: torch.Tensor,  # (B, P, D)
) -> torch.Tensor:
    """
    CPU batch-invariant span scoring (einsum 'lkd,bpd->bplk').

    Each output element computed with identical reduction order.
    """
    L, K, D = span_rep.shape
    B, P, D2 = struct_proj.shape
    assert D == D2

    output = torch.empty((B, P, L, K), dtype=span_rep.dtype, device=span_rep.device)

    # Process each output element with same reduction order
    for b in range(B):
        for p in range(P):
            proj = struct_proj[b, p].float()  # (D,)

            for l in range(L):
                for k in range(K):
                    span = span_rep[l, k].float()  # (D,)

                    # Sequential dot product
                    dot = _sequential_dot(span, proj)
                    output[b, p, l, k] = dot.to(span_rep.dtype)

    return output


def _sequential_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sequential dot product with fixed accumulation order."""
    N = a.numel()

    tile_size = 64
    acc = torch.tensor(0.0, dtype=torch.float32, device=a.device)

    for start in range(0, N, tile_size):
        end = min(start + tile_size, N)
        tile_dot = (a[start:end] * b[start:end]).sum()
        acc = acc + tile_dot

    return acc


# Optimized version using vectorized operations where safe
def cpu_batch_invariant_span_score_fast(
    span_rep: torch.Tensor,  # (L, K, D)
    struct_proj: torch.Tensor,  # (B, P, D)
) -> torch.Tensor:
    """
    Faster CPU batch-invariant span scoring.

    Uses per-element reduction which is batch-invariant since each (b,p,l,k)
    output only depends on fixed inputs span_rep[l,k] and struct_proj[b,p].
    """
    L, K, D = span_rep.shape
    B, P, D2 = struct_proj.shape

    # Reshape for broadcasting
    # span_rep: (L, K, D) -> (1, 1, L, K, D)
    # struct_proj: (B, P, D) -> (B, P, 1, 1, D)
    span_exp = span_rep.unsqueeze(0).unsqueeze(0).float()  # (1, 1, L, K, D)
    proj_exp = struct_proj.unsqueeze(2).unsqueeze(3).float()  # (B, P, 1, 1, D)

    # Element-wise multiply and sum along D
    # Each (b,p,l,k) element computed independently
    # The reduction is along D dimension only, with fixed order
    product = span_exp * proj_exp  # (B, P, L, K, D)

    # Sequential sum along D for batch invariance
    output = torch.zeros((B, P, L, K), dtype=torch.float32, device=span_rep.device)

    tile_size = 64
    for d_start in range(0, D, tile_size):
        d_end = min(d_start + tile_size, D)
        output = output + product[..., d_start:d_end].sum(dim=-1)

    return output.to(span_rep.dtype)
