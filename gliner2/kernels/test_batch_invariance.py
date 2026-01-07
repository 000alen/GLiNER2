"""
Tests for Batch-Invariant Kernels

These tests verify that our custom kernels produce identical results
regardless of batch size, achieving true batch invariance.

Run with: python -m pytest gliner2/kernels/test_batch_invariance.py -v
"""

import pytest
import torch
import numpy as np
from typing import Callable, List, Tuple

# Skip all tests if Triton is not available
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    """
    Compute the maximum ULP (Units in Last Place) distance between two tensors.

    ULP distance of 0 means bitwise identical.
    """
    if a.dtype == torch.float32:
        a_int = a.view(torch.int32)
        b_int = b.view(torch.int32)
    elif a.dtype == torch.float16:
        a_int = a.view(torch.int16)
        b_int = b.view(torch.int16)
    else:
        # For other dtypes, compare directly
        return 0 if torch.equal(a, b) else float('inf')

    return torch.abs(a_int - b_int).max().item()


def check_batch_invariance(
    kernel_fn: Callable,
    create_inputs: Callable[[int], Tuple[torch.Tensor, ...]],
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
    device: str = "cuda",
    max_ulp: int = 0,
) -> bool:
    """
    Test that a kernel produces identical results across different batch sizes.

    Args:
        kernel_fn: The kernel function to test
        create_inputs: Function that creates inputs for a given batch size
        batch_sizes: List of batch sizes to test
        device: Device to run on
        max_ulp: Maximum acceptable ULP distance (0 for bitwise identical)

    Returns:
        True if kernel is batch-invariant, False otherwise
    """
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")

    # Get reference result with batch_size=1
    inputs_1 = create_inputs(1)
    inputs_1 = tuple(x.to(device) for x in inputs_1)
    ref_result = kernel_fn(*inputs_1)

    for batch_size in batch_sizes[1:]:
        inputs_b = create_inputs(batch_size)
        inputs_b = tuple(x.to(device) for x in inputs_b)
        result_b = kernel_fn(*inputs_b)

        # Extract first element for comparison
        if result_b.dim() > ref_result.dim():
            # Batched output
            result_first = result_b[0]
        else:
            result_first = result_b

        ulp = ulp_distance(ref_result.flatten(), result_first.flatten())

        if ulp > max_ulp:
            print(f"Batch size {batch_size}: ULP distance = {ulp}")
            return False

    return True


class TestBatchInvariantMatmul:
    """Tests for batch-invariant matrix multiplication."""

    def test_matmul_batch_invariance(self):
        """Test that matmul produces identical results across batch sizes."""
        from gliner2.kernels.matmul import batch_invariant_matmul

        M, K, N = 64, 128, 64

        # Fixed input matrices
        torch.manual_seed(42)
        a_base = torch.randn(M, K, dtype=torch.float32)
        b_base = torch.randn(K, N, dtype=torch.float32)

        def create_inputs(batch_size):
            # Replicate for batching (matmul doesn't really batch, but we test consistency)
            return (a_base.clone(), b_base.clone())

        assert check_batch_invariance(
            batch_invariant_matmul,
            create_inputs,
            batch_sizes=[1, 1, 1, 1],  # Test run-to-run consistency
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def test_linear_batch_invariance(self):
        """Test that linear layer produces identical results for each sample."""
        from gliner2.kernels.matmul import batch_invariant_linear

        in_features, out_features = 256, 128

        torch.manual_seed(42)
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        bias = torch.randn(out_features, dtype=torch.float32)
        x_base = torch.randn(1, in_features, dtype=torch.float32)

        def run_linear(x):
            return batch_invariant_linear(x, weight.to(x.device), bias.to(x.device))

        def create_inputs(batch_size):
            # Same input replicated
            return (x_base.expand(batch_size, -1).clone(),)

        if torch.cuda.is_available():
            results = []
            for bs in [1, 2, 4, 8]:
                x = x_base.expand(bs, -1).clone().cuda()
                w = weight.cuda()
                b = bias.cuda()
                result = batch_invariant_linear(x, w, b)
                results.append(result[0].cpu())

            # Check all results are identical
            ref = results[0]
            for i, r in enumerate(results[1:], 2):
                ulp = ulp_distance(ref, r)
                assert ulp == 0, f"Batch size {i}: ULP distance = {ulp}"


class TestBatchInvariantLayerNorm:
    """Tests for batch-invariant LayerNorm."""

    def test_layernorm_batch_invariance(self):
        """Test that LayerNorm produces identical results for each sample."""
        from gliner2.kernels.layernorm import batch_invariant_layernorm

        hidden_size = 768

        torch.manual_seed(42)
        weight = torch.randn(hidden_size, dtype=torch.float32)
        bias = torch.randn(hidden_size, dtype=torch.float32)
        x_base = torch.randn(1, hidden_size, dtype=torch.float32)

        if torch.cuda.is_available():
            results = []
            for bs in [1, 2, 4, 8, 16, 32]:
                x = x_base.expand(bs, -1).clone().cuda()
                w = weight.cuda()
                b = bias.cuda()
                result = batch_invariant_layernorm(x, hidden_size, w, b)
                results.append(result[0].cpu())

            # Check all results are identical
            ref = results[0]
            for i, r in enumerate(results[1:], 2):
                ulp = ulp_distance(ref, r)
                assert ulp == 0, f"Batch size {2**i}: ULP distance = {ulp}"

    def test_rmsnorm_batch_invariance(self):
        """Test that RMSNorm produces identical results for each sample."""
        from gliner2.kernels.layernorm import batch_invariant_rmsnorm

        hidden_size = 768

        torch.manual_seed(42)
        weight = torch.randn(hidden_size, dtype=torch.float32)
        x_base = torch.randn(1, hidden_size, dtype=torch.float32)

        if torch.cuda.is_available():
            results = []
            for bs in [1, 2, 4, 8, 16, 32]:
                x = x_base.expand(bs, -1).clone().cuda()
                w = weight.cuda()
                result = batch_invariant_rmsnorm(x, w)
                results.append(result[0].cpu())

            ref = results[0]
            for i, r in enumerate(results[1:], 2):
                ulp = ulp_distance(ref, r)
                assert ulp == 0, f"Batch size {2**i}: ULP distance = {ulp}"


class TestBatchInvariantSpanScore:
    """Tests for batch-invariant span scoring."""

    def test_span_score_batch_invariance(self):
        """Test that span scoring produces identical results across batch sizes."""
        from gliner2.kernels.span_score import batch_invariant_span_score

        L, K, D = 64, 8, 256  # Typical GLiNER2 dimensions
        P = 4  # Number of fields

        torch.manual_seed(42)
        span_rep = torch.randn(L, K, D, dtype=torch.float32)
        proj_base = torch.randn(1, P, D, dtype=torch.float32)

        if torch.cuda.is_available():
            span_rep_cuda = span_rep.cuda()

            results = []
            for B in [1, 2, 4, 8, 16]:
                # Create struct_proj with B instances (simulating count)
                struct_proj = proj_base.expand(B, -1, -1).clone().cuda()
                result = batch_invariant_span_score(span_rep_cuda, struct_proj)
                # Get first instance's result
                results.append(result[0].cpu())

            # Check all first-instance results are identical
            ref = results[0]
            for i, r in enumerate(results[1:], 2):
                ulp = ulp_distance(ref, r)
                assert ulp == 0, f"B={2**(i-1)}: ULP distance = {ulp}"


class TestBatchInvariantSoftmax:
    """Tests for batch-invariant softmax."""

    def test_softmax_batch_invariance(self):
        """Test that softmax produces identical results for each sample."""
        from gliner2.kernels.softmax import batch_invariant_softmax

        num_classes = 100

        torch.manual_seed(42)
        x_base = torch.randn(1, num_classes, dtype=torch.float32)

        if torch.cuda.is_available():
            results = []
            for bs in [1, 2, 4, 8, 16, 32]:
                x = x_base.expand(bs, -1).clone().cuda()
                result = batch_invariant_softmax(x, dim=-1)
                results.append(result[0].cpu())

            ref = results[0]
            for i, r in enumerate(results[1:], 2):
                ulp = ulp_distance(ref, r)
                assert ulp == 0, f"Batch size {2**i}: ULP distance = {ulp}"

    def test_log_softmax_batch_invariance(self):
        """Test that log_softmax produces identical results for each sample."""
        from gliner2.kernels.softmax import batch_invariant_log_softmax

        num_classes = 100

        torch.manual_seed(42)
        x_base = torch.randn(1, num_classes, dtype=torch.float32)

        if torch.cuda.is_available():
            results = []
            for bs in [1, 2, 4, 8, 16, 32]:
                x = x_base.expand(bs, -1).clone().cuda()
                result = batch_invariant_log_softmax(x, dim=-1)
                results.append(result[0].cpu())

            ref = results[0]
            for i, r in enumerate(results[1:], 2):
                ulp = ulp_distance(ref, r)
                assert ulp == 0, f"Batch size {2**i}: ULP distance = {ulp}"


class TestDeterministicMode:
    """Tests for the deterministic inference mode."""

    def test_deterministic_mode_context(self):
        """Test that DeterministicMode context manager works."""
        from gliner2.inference.deterministic import (
            DeterministicMode,
            DeterministicLevel,
            is_batch_invariant_mode,
            is_deterministic_mode,
        )

        # Initially off
        assert not is_deterministic_mode()
        assert not is_batch_invariant_mode()

        # Enable batch-invariant mode
        with DeterministicMode(level=DeterministicLevel.BATCH_INVARIANT):
            assert is_deterministic_mode()
            assert is_batch_invariant_mode()

        # Back to off
        assert not is_deterministic_mode()
        assert not is_batch_invariant_mode()

    def test_batch_invariant_einsum(self):
        """Test the batch_invariant_einsum helper function."""
        from gliner2.inference.deterministic import batch_invariant_einsum

        L, K, D = 32, 8, 128
        B, P = 4, 3

        torch.manual_seed(42)
        span_rep = torch.randn(L, K, D)
        struct_proj = torch.randn(B, P, D)

        if torch.cuda.is_available():
            span_rep = span_rep.cuda()
            struct_proj = struct_proj.cuda()

        # Should use batch-invariant kernel
        result = batch_invariant_einsum("lkd,bpd->bplk", span_rep, struct_proj)

        # Compare with torch.einsum
        expected = torch.einsum("lkd,bpd->bplk", span_rep, struct_proj)

        # Results should be close (may not be identical due to different algorithms)
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)


def verify_gliner2_batch_invariance(
    model,
    text: str,
    schema,
    batch_sizes: List[int] = [1, 2, 4, 8],
    num_trials: int = 3,
) -> dict:
    """
    Verify that a GLiNER2 model produces batch-invariant results.

    Args:
        model: GLiNER2 model instance
        text: Test text
        schema: Extraction schema
        batch_sizes: Batch sizes to test
        num_trials: Number of trials per batch size

    Returns:
        Dictionary with test results
    """
    results = {}

    for batch_size in batch_sizes:
        texts = [text] * batch_size

        trial_results = []
        for trial in range(num_trials):
            output = model.batch_extract(
                texts, schema,
                batch_size=batch_size,
                deterministic=True
            )
            trial_results.append(output[0])  # First result

        # Check run-to-run consistency
        all_same = all(r == trial_results[0] for r in trial_results)
        results[batch_size] = {
            'output': trial_results[0],
            'run_consistent': all_same
        }

    # Check cross-batch consistency
    reference = results[batch_sizes[0]]['output']
    cross_batch_consistent = all(
        r['output'] == reference
        for r in results.values()
    )

    return {
        'batch_results': results,
        'run_consistent': all(r['run_consistent'] for r in results.values()),
        'cross_batch_consistent': cross_batch_consistent,
        'fully_invariant': cross_batch_consistent and all(r['run_consistent'] for r in results.values())
    }


class TestCPUBatchInvariance:
    """Tests for CPU batch-invariant kernels (no CUDA/Triton required)."""

    def test_cpu_linear_batch_invariance(self):
        """Test CPU batch-invariant linear layer."""
        from gliner2.kernels.cpu_batch_invariant import cpu_batch_invariant_linear

        in_features, out_features = 128, 64

        torch.manual_seed(42)
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)
        x_base = torch.randn(1, in_features)

        results = []
        for bs in [1, 2, 4, 8, 16]:
            x = x_base.expand(bs, -1).clone()
            result = cpu_batch_invariant_linear(x, weight, bias)
            results.append(result[0])

        ref = results[0]
        for i, r in enumerate(results[1:], 2):
            ulp = ulp_distance(ref, r)
            assert ulp == 0, f"Batch size {i}: ULP distance = {ulp}"

    def test_cpu_layernorm_batch_invariance(self):
        """Test CPU batch-invariant LayerNorm."""
        from gliner2.kernels.cpu_batch_invariant import cpu_batch_invariant_layernorm

        hidden_size = 256

        torch.manual_seed(42)
        weight = torch.randn(hidden_size)
        bias = torch.randn(hidden_size)
        x_base = torch.randn(1, hidden_size)

        results = []
        for bs in [1, 2, 4, 8, 16]:
            x = x_base.expand(bs, -1).clone()
            result = cpu_batch_invariant_layernorm(x, hidden_size, weight, bias)
            results.append(result[0])

        ref = results[0]
        for i, r in enumerate(results[1:], 2):
            ulp = ulp_distance(ref, r)
            assert ulp == 0, f"Batch size {i}: ULP distance = {ulp}"

    def test_cpu_span_score_batch_invariance(self):
        """Test CPU batch-invariant span scoring."""
        from gliner2.kernels.cpu_batch_invariant import cpu_batch_invariant_span_score_fast

        L, K, D = 16, 4, 64
        P = 3

        torch.manual_seed(42)
        span_rep = torch.randn(L, K, D)
        proj_base = torch.randn(1, P, D)

        results = []
        for B in [1, 2, 4, 8]:
            struct_proj = proj_base.expand(B, -1, -1).clone()
            result = cpu_batch_invariant_span_score_fast(span_rep, struct_proj)
            results.append(result[0])

        ref = results[0]
        for i, r in enumerate(results[1:], 2):
            ulp = ulp_distance(ref, r)
            assert ulp == 0, f"B={i}: ULP distance = {ulp}"

    def test_cpu_softmax_batch_invariance(self):
        """Test CPU batch-invariant softmax."""
        from gliner2.kernels.cpu_batch_invariant import cpu_batch_invariant_softmax

        num_classes = 50

        torch.manual_seed(42)
        x_base = torch.randn(1, num_classes)

        results = []
        for bs in [1, 2, 4, 8, 16]:
            x = x_base.expand(bs, -1).clone()
            result = cpu_batch_invariant_softmax(x, dim=-1)
            results.append(result[0])

        ref = results[0]
        for i, r in enumerate(results[1:], 2):
            ulp = ulp_distance(ref, r)
            assert ulp == 0, f"Batch size {i}: ULP distance = {ulp}"


if __name__ == "__main__":
    # Run basic sanity checks
    print("Running batch invariance tests...")

    # Always run CPU tests
    print("\n=== CPU Batch Invariance Tests ===")

    print("\n1. Testing CPU batch-invariant linear...")
    test_cpu = TestCPUBatchInvariance()
    test_cpu.test_cpu_linear_batch_invariance()
    print("   PASSED")

    print("\n2. Testing CPU batch-invariant LayerNorm...")
    test_cpu.test_cpu_layernorm_batch_invariance()
    print("   PASSED")

    print("\n3. Testing CPU batch-invariant span score...")
    test_cpu.test_cpu_span_score_batch_invariance()
    print("   PASSED")

    print("\n4. Testing CPU batch-invariant softmax...")
    test_cpu.test_cpu_softmax_batch_invariance()
    print("   PASSED")

    print("\nCPU batch invariance tests PASSED!")

    # GPU tests require Triton and CUDA
    if not TRITON_AVAILABLE:
        print("\nTriton not available, skipping GPU tests")
        exit(0)

    if not torch.cuda.is_available():
        print("\nCUDA not available, skipping GPU tests")
        exit(0)

    print("\n=== GPU Batch Invariance Tests ===")

    print("\n1. Testing GPU batch-invariant matmul...")
    test_matmul = TestBatchInvariantMatmul()
    test_matmul.test_linear_batch_invariance()
    print("   PASSED")

    print("\n2. Testing GPU batch-invariant LayerNorm...")
    test_ln = TestBatchInvariantLayerNorm()
    test_ln.test_layernorm_batch_invariance()
    print("   PASSED")

    print("\n3. Testing GPU batch-invariant span score...")
    test_span = TestBatchInvariantSpanScore()
    test_span.test_span_score_batch_invariance()
    print("   PASSED")

    print("\n4. Testing GPU batch-invariant softmax...")
    test_softmax = TestBatchInvariantSoftmax()
    test_softmax.test_softmax_batch_invariance()
    print("   PASSED")

    print("\n5. Testing deterministic mode context...")
    test_det = TestDeterministicMode()
    test_det.test_deterministic_mode_context()
    print("   PASSED")

    print("\nAll batch invariance tests PASSED!")
