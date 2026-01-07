"""
Modal Script for Running Batch Invariance Tests on GPU

This script runs the batch invariance tests on Modal's GPU infrastructure,
ensuring our custom Triton kernels produce identical results regardless of
batch size.

Setup:
    # Install Modal CLI
    pip install modal

    # Authenticate (first time only)
    modal setup

Usage (run from repository root):
    # Run all GPU batch invariance tests
    modal run tests/modal_batch_invariance.py

    # Run with pytest for detailed output
    modal run tests/modal_batch_invariance.py::run_pytest

    # Run specific function
    modal run tests/modal_batch_invariance.py::run_gpu_tests

Cost:
    - Uses A10G GPU (~$0.000306/sec)
    - Typical run: ~30-60 seconds = ~$0.01-0.02
    - Free tier includes $30/month credits
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("gliner2-batch-invariance-tests")

# Get the repository root (parent of tests directory)
REPO_ROOT = Path(__file__).parent.parent

# Define the image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "triton>=2.1.0",
        "pytest>=7.0.0",
        "numpy>=1.21.0",
        "gliner",  # Required dependency
    )
    # Copy gliner2 package from repo root
    .add_local_dir(str(REPO_ROOT / "gliner2"), "/root/gliner2")
)


@app.function(
    image=image,
    gpu="A10G",  # Cost-efficient GPU with Triton support
    timeout=600,  # 10 minutes
)
def run_gpu_tests() -> dict:
    """
    Run all GPU batch invariance tests.

    Returns a dictionary with test results and any failures.
    """
    import sys
    sys.path.insert(0, "/root")

    import torch
    results = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tests": {},
        "all_passed": True,
    }

    if not torch.cuda.is_available():
        results["error"] = "CUDA not available"
        results["all_passed"] = False
        return results

    # Check Triton
    try:
        import triton
        results["triton_version"] = triton.__version__
    except ImportError:
        results["error"] = "Triton not available"
        results["all_passed"] = False
        return results

    # Run individual tests
    test_functions = [
        ("gpu_linear", test_gpu_linear_batch_invariance),
        ("gpu_layernorm", test_gpu_layernorm_batch_invariance),
        ("gpu_rmsnorm", test_gpu_rmsnorm_batch_invariance),
        ("gpu_span_score", test_gpu_span_score_batch_invariance),
        ("gpu_softmax", test_gpu_softmax_batch_invariance),
        ("gpu_log_softmax", test_gpu_log_softmax_batch_invariance),
        ("deterministic_mode", test_deterministic_mode),
        ("batch_invariant_einsum", test_batch_invariant_einsum),
    ]

    for name, test_fn in test_functions:
        try:
            test_fn()
            results["tests"][name] = {"status": "PASSED"}
        except Exception as e:
            results["tests"][name] = {"status": "FAILED", "error": str(e)}
            results["all_passed"] = False

    return results


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
)
def run_pytest() -> str:
    """
    Run tests using pytest and return the output.
    """
    import subprocess
    import sys

    sys.path.insert(0, "/root")

    result = subprocess.run(
        ["python", "-m", "pytest", "/root/gliner2/kernels/test_batch_invariance.py", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd="/root",
    )

    return f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nReturn code: {result.returncode}"


def ulp_distance(a, b):
    """Compute maximum ULP distance between two tensors."""
    import torch

    if a.dtype == torch.float32:
        a_int = a.view(torch.int32)
        b_int = b.view(torch.int32)
    elif a.dtype == torch.float16:
        a_int = a.view(torch.int16)
        b_int = b.view(torch.int16)
    else:
        return 0 if torch.equal(a, b) else float('inf')

    return torch.abs(a_int - b_int).max().item()


def test_gpu_linear_batch_invariance():
    """Test GPU batch-invariant linear layer."""
    import torch
    from gliner2.kernels.matmul import batch_invariant_linear

    in_features, out_features = 256, 128

    torch.manual_seed(42)
    weight = torch.randn(out_features, in_features, dtype=torch.float32).cuda()
    bias = torch.randn(out_features, dtype=torch.float32).cuda()
    x_base = torch.randn(1, in_features, dtype=torch.float32)

    results = []
    for bs in [1, 2, 4, 8, 16, 32]:
        x = x_base.expand(bs, -1).clone().cuda()
        result = batch_invariant_linear(x, weight, bias)
        results.append(result[0].cpu())

    ref = results[0]
    for i, r in enumerate(results[1:], 2):
        ulp = ulp_distance(ref, r)
        assert ulp == 0, f"Batch size {2**(i-1)}: ULP distance = {ulp}"


def test_gpu_layernorm_batch_invariance():
    """Test GPU batch-invariant LayerNorm."""
    import torch
    from gliner2.kernels.layernorm import batch_invariant_layernorm

    hidden_size = 768

    torch.manual_seed(42)
    weight = torch.randn(hidden_size, dtype=torch.float32).cuda()
    bias = torch.randn(hidden_size, dtype=torch.float32).cuda()
    x_base = torch.randn(1, hidden_size, dtype=torch.float32)

    results = []
    for bs in [1, 2, 4, 8, 16, 32]:
        x = x_base.expand(bs, -1).clone().cuda()
        result = batch_invariant_layernorm(x, hidden_size, weight, bias)
        results.append(result[0].cpu())

    ref = results[0]
    for i, r in enumerate(results[1:], 2):
        ulp = ulp_distance(ref, r)
        assert ulp == 0, f"Batch size {2**(i-1)}: ULP distance = {ulp}"


def test_gpu_rmsnorm_batch_invariance():
    """Test GPU batch-invariant RMSNorm."""
    import torch
    from gliner2.kernels.layernorm import batch_invariant_rmsnorm

    hidden_size = 768

    torch.manual_seed(42)
    weight = torch.randn(hidden_size, dtype=torch.float32).cuda()
    x_base = torch.randn(1, hidden_size, dtype=torch.float32)

    results = []
    for bs in [1, 2, 4, 8, 16, 32]:
        x = x_base.expand(bs, -1).clone().cuda()
        result = batch_invariant_rmsnorm(x, weight)
        results.append(result[0].cpu())

    ref = results[0]
    for i, r in enumerate(results[1:], 2):
        ulp = ulp_distance(ref, r)
        assert ulp == 0, f"Batch size {2**(i-1)}: ULP distance = {ulp}"


def test_gpu_span_score_batch_invariance():
    """Test GPU batch-invariant span scoring."""
    import torch
    from gliner2.kernels.span_score import batch_invariant_span_score

    L, K, D = 64, 8, 256
    P = 4

    torch.manual_seed(42)
    span_rep = torch.randn(L, K, D, dtype=torch.float32).cuda()
    proj_base = torch.randn(1, P, D, dtype=torch.float32)

    results = []
    for B in [1, 2, 4, 8, 16]:
        struct_proj = proj_base.expand(B, -1, -1).clone().cuda()
        result = batch_invariant_span_score(span_rep, struct_proj)
        results.append(result[0].cpu())

    ref = results[0]
    for i, r in enumerate(results[1:], 2):
        ulp = ulp_distance(ref, r)
        assert ulp == 0, f"B={2**(i-1)}: ULP distance = {ulp}"


def test_gpu_softmax_batch_invariance():
    """Test GPU batch-invariant softmax."""
    import torch
    from gliner2.kernels.softmax import batch_invariant_softmax

    num_classes = 100

    torch.manual_seed(42)
    x_base = torch.randn(1, num_classes, dtype=torch.float32)

    results = []
    for bs in [1, 2, 4, 8, 16, 32]:
        x = x_base.expand(bs, -1).clone().cuda()
        result = batch_invariant_softmax(x, dim=-1)
        results.append(result[0].cpu())

    ref = results[0]
    for i, r in enumerate(results[1:], 2):
        ulp = ulp_distance(ref, r)
        assert ulp == 0, f"Batch size {2**(i-1)}: ULP distance = {ulp}"


def test_gpu_log_softmax_batch_invariance():
    """Test GPU batch-invariant log_softmax."""
    import torch
    from gliner2.kernels.softmax import batch_invariant_log_softmax

    num_classes = 100

    torch.manual_seed(42)
    x_base = torch.randn(1, num_classes, dtype=torch.float32)

    results = []
    for bs in [1, 2, 4, 8, 16, 32]:
        x = x_base.expand(bs, -1).clone().cuda()
        result = batch_invariant_log_softmax(x, dim=-1)
        results.append(result[0].cpu())

    ref = results[0]
    for i, r in enumerate(results[1:], 2):
        ulp = ulp_distance(ref, r)
        assert ulp == 0, f"Batch size {2**(i-1)}: ULP distance = {ulp}"


def test_deterministic_mode():
    """Test DeterministicMode context manager."""
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


def test_batch_invariant_einsum():
    """Test the batch_invariant_einsum helper function."""
    import torch
    from gliner2.inference.deterministic import batch_invariant_einsum

    L, K, D = 32, 8, 128
    B, P = 4, 3

    torch.manual_seed(42)
    span_rep = torch.randn(L, K, D).cuda()
    struct_proj = torch.randn(B, P, D).cuda()

    result = batch_invariant_einsum("lkd,bpd->bplk", span_rep, struct_proj)
    expected = torch.einsum("lkd,bpd->bplk", span_rep, struct_proj)

    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)


@app.function(
    image=image,
    gpu="A100",  # More powerful GPU for faster tests
    timeout=600,
)
def run_gpu_tests_a100() -> dict:
    """Run tests on A100 GPU (faster but more expensive)."""
    # Same implementation as run_gpu_tests
    import sys
    sys.path.insert(0, "/root")

    import torch
    results = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tests": {},
        "all_passed": True,
    }

    if not torch.cuda.is_available():
        results["error"] = "CUDA not available"
        results["all_passed"] = False
        return results

    try:
        import triton
        results["triton_version"] = triton.__version__
    except ImportError:
        results["error"] = "Triton not available"
        results["all_passed"] = False
        return results

    test_functions = [
        ("gpu_linear", test_gpu_linear_batch_invariance),
        ("gpu_layernorm", test_gpu_layernorm_batch_invariance),
        ("gpu_rmsnorm", test_gpu_rmsnorm_batch_invariance),
        ("gpu_span_score", test_gpu_span_score_batch_invariance),
        ("gpu_softmax", test_gpu_softmax_batch_invariance),
        ("gpu_log_softmax", test_gpu_log_softmax_batch_invariance),
        ("deterministic_mode", test_deterministic_mode),
        ("batch_invariant_einsum", test_batch_invariant_einsum),
    ]

    for name, test_fn in test_functions:
        try:
            test_fn()
            results["tests"][name] = {"status": "PASSED"}
        except Exception as e:
            results["tests"][name] = {"status": "FAILED", "error": str(e)}
            results["all_passed"] = False

    return results


@app.local_entrypoint()
def main(use_a100: bool = False, pytest_mode: bool = False):
    """
    Run the batch invariance tests on Modal.

    Args:
        use_a100: Use A100 GPU instead of A10G (faster but more expensive)
        pytest_mode: Run with pytest for detailed output
    """
    print("=" * 60)
    print("GLiNER2 Batch Invariance Tests on Modal")
    print("=" * 60)

    if pytest_mode:
        print("\nRunning with pytest (detailed output)...")
        output = run_pytest.remote()
        print(output)
        return

    gpu_type = "A100" if use_a100 else "A10G"
    print(f"\nRunning tests on {gpu_type} GPU...")

    # Run the tests
    if use_a100:
        results = run_gpu_tests_a100.remote()
    else:
        results = run_gpu_tests.remote()

    # Print results
    print(f"\nCUDA Available: {results['cuda_available']}")
    print(f"GPU: {results.get('gpu_name', 'N/A')}")
    print(f"Triton Version: {results.get('triton_version', 'N/A')}")

    print("\n" + "-" * 60)
    print("Test Results:")
    print("-" * 60)

    passed = 0
    failed = 0
    for test_name, test_result in results["tests"].items():
        status = test_result["status"]
        if status == "PASSED":
            print(f"  [PASS] {test_name}")
            passed += 1
        else:
            print(f"  [FAIL] {test_name}")
            print(f"         Error: {test_result.get('error', 'Unknown error')}")
            failed += 1

    print("-" * 60)
    print(f"Summary: {passed} passed, {failed} failed")

    if results["all_passed"]:
        print("\nAll tests PASSED!")
        print("Batch invariance verified: outputs are bitwise identical across batch sizes.")
    else:
        print("\nSome tests FAILED!")
        if "error" in results:
            print(f"Error: {results['error']}")

    return results
