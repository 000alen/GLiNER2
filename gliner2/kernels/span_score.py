"""
Batch-Invariant Span Scoring Kernel

This kernel implements the critical einsum operation used in GLiNER2
for computing span scores:

    torch.einsum('lkd,bpd->bplk', span_rep, struct_proj)

where:
    - span_rep: (L, K, D) - span representations (L=num_positions, K=max_width, D=hidden)
    - struct_proj: (B, P, D) - projected structure embeddings (B=count, P=num_fields, D=hidden)
    - output: (B, P, L, K) - span scores for each field at each position

The kernel ensures batch invariance by:
1. Computing each output element independently
2. Using fixed-order reduction across the D dimension
3. Never parallelizing the dot product reduction
"""

import torch
import triton
import triton.language as tl


# Block size for D-dimension reduction
BLOCK_D = 128


@triton.jit
def _batch_invariant_span_score_kernel(
    span_rep_ptr,  # (L, K, D)
    struct_proj_ptr,  # (B, P, D)
    output_ptr,  # (B, P, L, K)
    L,  # num positions
    K,  # max width
    D,  # hidden dimension
    B,  # count (batch)
    P,  # num fields
    stride_span_l,
    stride_span_k,
    stride_span_d,
    stride_proj_b,
    stride_proj_p,
    stride_proj_d,
    stride_out_b,
    stride_out_p,
    stride_out_l,
    stride_out_k,
    BLOCK_D: tl.constexpr,
):
    """
    Batch-invariant span scoring kernel.

    Each program computes one output element (b, p, l, k) by computing
    the dot product span_rep[l, k, :] @ struct_proj[b, p, :] with
    fixed-order reduction across D.
    """
    # Linear index to (b, p, l, k)
    pid = tl.program_id(0)

    # Decompose linear index (row-major order for determinism)
    k = pid % K
    l = (pid // K) % L
    p = (pid // (K * L)) % P
    b = pid // (K * L * P)

    # Bounds check
    if b >= B or p >= P or l >= L or k >= K:
        return

    # Compute base pointers for this element
    span_base = l * stride_span_l + k * stride_span_k
    proj_base = b * stride_proj_b + p * stride_proj_p

    # Fixed-order reduction across D dimension
    acc = 0.0

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < D

        # Load span_rep[l, k, d_start:d_start+BLOCK_D]
        span_vals = tl.load(
            span_rep_ptr + span_base + d_offsets * stride_span_d,
            mask=mask,
            other=0.0,
        )

        # Load struct_proj[b, p, d_start:d_start+BLOCK_D]
        proj_vals = tl.load(
            struct_proj_ptr + proj_base + d_offsets * stride_proj_d,
            mask=mask,
            other=0.0,
        )

        # Accumulate dot product (fixed order via sequential blocks + tl.sum)
        acc += tl.sum(span_vals.to(tl.float32) * proj_vals.to(tl.float32))

    # Store result
    out_offset = b * stride_out_b + p * stride_out_p + l * stride_out_l + k * stride_out_k
    tl.store(output_ptr + out_offset, acc.to(output_ptr.dtype.element_ty))


def batch_invariant_span_score(
    span_rep: torch.Tensor,  # (L, K, D)
    struct_proj: torch.Tensor,  # (B, P, D)
) -> torch.Tensor:
    """
    Batch-invariant implementation of:
        torch.einsum('lkd,bpd->bplk', span_rep, struct_proj)

    Args:
        span_rep: Span representations of shape (L, K, D)
                  L = number of positions, K = max width, D = hidden dimension
        struct_proj: Projected structure embeddings of shape (B, P, D)
                     B = count (number of instances), P = number of fields

    Returns:
        Span scores of shape (B, P, L, K)

    This function guarantees identical results regardless of the values of B
    (the batch/count dimension).
    """
    # Get dimensions
    L, K, D = span_rep.shape
    B, P, D2 = struct_proj.shape
    assert D == D2, f"Hidden dimensions must match: {D} vs {D2}"

    # Ensure contiguous
    span_rep = span_rep.contiguous()
    struct_proj = struct_proj.contiguous()

    # Allocate output
    output = torch.empty((B, P, L, K), device=span_rep.device, dtype=span_rep.dtype)

    # Total number of output elements
    total_elements = B * P * L * K

    if total_elements == 0:
        return output

    # Determine block size for D reduction
    block_d = min(BLOCK_D, triton.next_power_of_2(D))

    # Launch one program per output element
    grid = (total_elements,)

    _batch_invariant_span_score_kernel[grid](
        span_rep,
        struct_proj,
        output,
        L,
        K,
        D,
        B,
        P,
        span_rep.stride(0),
        span_rep.stride(1),
        span_rep.stride(2),
        struct_proj.stride(0),
        struct_proj.stride(1),
        struct_proj.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        BLOCK_D=block_d,
    )

    return output


@triton.jit
def _batch_invariant_einsum_bmd_nd_bmn_kernel(
    a_ptr,  # (B, M, D)
    b_ptr,  # (N, D)
    output_ptr,  # (B, M, N)
    B,
    M,
    D,
    N,
    stride_a_b,
    stride_a_m,
    stride_a_d,
    stride_b_n,
    stride_b_d,
    stride_out_b,
    stride_out_m,
    stride_out_n,
    BLOCK_D: tl.constexpr,
):
    """
    Batch-invariant einsum: 'bmd,nd->bmn'

    Useful for batched bilinear operations.
    """
    pid = tl.program_id(0)

    # Decompose to (b, m, n)
    n = pid % N
    m = (pid // N) % M
    b = pid // (N * M)

    if b >= B or m >= M or n >= N:
        return

    a_base = b * stride_a_b + m * stride_a_m
    b_base = n * stride_b_n

    acc = 0.0

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < D

        a_vals = tl.load(a_ptr + a_base + d_offsets * stride_a_d, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + b_base + d_offsets * stride_b_d, mask=mask, other=0.0)

        acc += tl.sum(a_vals.to(tl.float32) * b_vals.to(tl.float32))

    out_offset = b * stride_out_b + m * stride_out_m + n * stride_out_n
    tl.store(output_ptr + out_offset, acc.to(output_ptr.dtype.element_ty))


def batch_invariant_einsum_bmd_nd(
    a: torch.Tensor,  # (B, M, D)
    b: torch.Tensor,  # (N, D)
) -> torch.Tensor:
    """
    Batch-invariant einsum: 'bmd,nd->bmn'

    Computes batched bilinear product where each (b, m) row of a
    is dotted with each row of b.
    """
    B, M, D = a.shape
    N, D2 = b.shape
    assert D == D2

    a = a.contiguous()
    b = b.contiguous()
    output = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    total = B * M * N
    if total == 0:
        return output

    block_d = min(BLOCK_D, triton.next_power_of_2(D))
    grid = (total,)

    _batch_invariant_einsum_bmd_nd_bmn_kernel[grid](
        a,
        b,
        output,
        B,
        M,
        D,
        N,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_D=block_d,
    )

    return output
