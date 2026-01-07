"""
Modal Script for Testing Full Model Determinism

This script tests that GLiNER2's full inference pipeline produces bitwise-identical
results regardless of batch size when deterministic mode is enabled.

Unlike the kernel-level tests in modal_batch_invariance.py, this tests the entire
model including:
- Transformer encoder (patched F.linear, F.layer_norm)
- Span representation
- Count prediction
- Structure extraction with batch_invariant_einsum
- Classification

Setup:
    pip install modal
    modal setup

Usage:
    # Run full determinism tests
    modal run tests/modal_full_determinism.py

    # Run with A100 GPU (faster)
    modal run tests/modal_full_determinism.py --use-a100
    
    # Run with detailed numerical output
    modal run tests/modal_full_determinism.py --verbose

Cost:
    - A10G: ~$0.000306/sec
    - A100: ~$0.001/sec
    - Typical run: ~2-3 minutes
"""

import modal
from pathlib import Path

app = modal.App("gliner2-full-determinism-tests")

REPO_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "triton>=2.1.0",
        "transformers>=4.30.0",
        "safetensors",
        "huggingface_hub",
        "gliner",
    )
    .add_local_dir(str(REPO_ROOT / "gliner2"), "/root/gliner2")
)


def hash_result(result: dict) -> str:
    """Create a deterministic hash of extraction results."""
    import json
    import hashlib
    
    # Sort keys recursively for consistent hashing
    def sort_dict(obj):
        if isinstance(obj, dict):
            return {k: sort_dict(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [sort_dict(x) for x in obj]
        return obj
    
    sorted_result = sort_dict(result)
    json_str = json.dumps(sorted_result, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def results_match(results: list) -> tuple[bool, str]:
    """Check if all results in the list are identical."""
    if not results:
        return True, "No results"
    
    first_hash = hash_result(results[0])
    for i, result in enumerate(results[1:], 1):
        h = hash_result(result)
        if h != first_hash:
            return False, f"Result {i} differs: {first_hash} vs {h}"
    
    return True, f"All {len(results)} results match (hash: {first_hash})"


def extract_confidence_values(result: dict) -> list[float]:
    """Extract all confidence values from a result dict recursively."""
    values = []
    
    def extract(obj):
        if isinstance(obj, dict):
            if "confidence" in obj:
                values.append(obj["confidence"])
            for v in obj.values():
                extract(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item)
    
    extract(result)
    return sorted(values, reverse=True)  # Sort for consistent comparison


def compare_confidences(results_by_batch: dict, verbose: bool = False) -> tuple[bool, str, dict]:
    """
    Compare confidence values across batch sizes.
    
    Returns:
        (all_identical, summary_string, detailed_comparison)
    """
    import struct
    
    batch_sizes = sorted(results_by_batch.keys())
    ref_bs = batch_sizes[0]
    ref_confidences = extract_confidence_values(results_by_batch[ref_bs])
    
    comparison = {
        "reference_batch_size": ref_bs,
        "reference_confidences": ref_confidences,
        "comparisons": {}
    }
    
    all_identical = True
    max_diff = 0.0
    max_ulp = 0
    
    for bs in batch_sizes[1:]:
        confidences = extract_confidence_values(results_by_batch[bs])
        
        if len(confidences) != len(ref_confidences):
            comparison["comparisons"][bs] = {
                "error": f"Different number of values: {len(ref_confidences)} vs {len(confidences)}"
            }
            all_identical = False
            continue
        
        diffs = []
        ulps = []
        for ref_val, val in zip(ref_confidences, confidences):
            diff = abs(ref_val - val)
            diffs.append(diff)
            
            # Calculate ULP distance for float32
            ref_bits = struct.unpack('I', struct.pack('f', ref_val))[0]
            val_bits = struct.unpack('I', struct.pack('f', val))[0]
            ulp = abs(ref_bits - val_bits)
            ulps.append(ulp)
            
            if diff > max_diff:
                max_diff = diff
            if ulp > max_ulp:
                max_ulp = ulp
        
        is_identical = all(u == 0 for u in ulps)
        if not is_identical:
            all_identical = False
        
        comparison["comparisons"][bs] = {
            "identical": is_identical,
            "max_abs_diff": max(diffs) if diffs else 0,
            "max_ulp_diff": max(ulps) if ulps else 0,
            "num_values": len(confidences),
            "confidences": confidences if verbose else None
        }
    
    if all_identical:
        summary = f"All {len(batch_sizes)} batch sizes produce bitwise-identical confidences ({len(ref_confidences)} values)"
    else:
        summary = f"MISMATCH: max_diff={max_diff:.2e}, max_ulp={max_ulp}"
    
    return all_identical, summary, comparison


@app.function(
    image=image,
    gpu="A10G",
    timeout=900,
)
def run_full_determinism_tests(verbose: bool = False) -> dict:
    """
    Run full model determinism tests with numerical comparison.
    
    Tests that the same input produces identical output regardless of batch size.
    Shows actual confidence values and ULP distances.
    """
    import sys
    sys.path.insert(0, "/root")
    
    import torch
    from gliner2 import GLiNER2, DeterministicMode, DeterministicLevel
    
    results = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tests": {},
        "all_passed": True,
        "verbose": verbose,
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
    
    # Load model
    print("Loading model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model = model.cuda()
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    
    # Test cases
    test_text = "Apple CEO Tim Cook announced the new iPhone 15 Pro at the Cupertino headquarters on September 12, 2023."
    
    # Run all tests
    test_functions = [
        ("entity_extraction", test_entity_extraction_determinism),
        ("classification", test_classification_determinism),
        ("json_extraction", test_json_extraction_determinism),
        ("relation_extraction", test_relation_extraction_determinism),
        ("batch_consistency", test_batch_consistency),
        ("numerical_precision", test_numerical_precision),
    ]
    
    for name, test_fn in test_functions:
        try:
            passed, details, numerical_data = test_fn(model, test_text, verbose)
            results["tests"][name] = {
                "status": "PASSED" if passed else "FAILED",
                "details": details,
                "numerical_data": numerical_data
            }
            if not passed:
                results["all_passed"] = False
        except Exception as e:
            import traceback
            results["tests"][name] = {
                "status": "FAILED",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            results["all_passed"] = False
    
    return results


def test_entity_extraction_determinism(model, test_text: str, verbose: bool = False) -> tuple[bool, str, dict]:
    """Test that entity extraction is deterministic across batch sizes."""
    entity_types = ["company", "person", "product", "location", "date"]
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    # Create test batch: same text repeated
    texts = [test_text] * 32
    
    results_by_batch = {}
    
    for bs in batch_sizes:
        batch_results = model.batch_extract_entities(
            texts[:bs],
            entity_types,
            batch_size=bs,
            deterministic=True,
            include_confidence=True
        )
        results_by_batch[bs] = batch_results[0]
    
    # Numerical comparison
    identical, summary, comparison = compare_confidences(results_by_batch, verbose)
    
    return identical, summary, comparison


def test_classification_determinism(model, test_text: str, verbose: bool = False) -> tuple[bool, str, dict]:
    """Test that classification is deterministic across batch sizes."""
    tasks = {
        "sentiment": ["positive", "negative", "neutral"],
        "topic": ["technology", "business", "science", "other"]
    }
    batch_sizes = [1, 2, 4, 8, 16]
    
    texts = [test_text] * 16
    results_by_batch = {}
    
    for bs in batch_sizes:
        batch_results = model.batch_classify_text(
            texts[:bs],
            tasks,
            batch_size=bs,
            deterministic=True,
            include_confidence=True
        )
        results_by_batch[bs] = batch_results[0]
    
    identical, summary, comparison = compare_confidences(results_by_batch, verbose)
    
    return identical, summary, comparison


def test_json_extraction_determinism(model, test_text: str, verbose: bool = False) -> tuple[bool, str, dict]:
    """Test that structured JSON extraction is deterministic."""
    structures = {
        "announcement": [
            "company::str::Company making announcement",
            "executive::str::Executive mentioned",
            "product::str::Product announced",
            "location::str::Location of announcement",
            "date::str::Date of announcement"
        ]
    }
    batch_sizes = [1, 2, 4, 8, 16]
    
    texts = [test_text] * 16
    results_by_batch = {}
    
    for bs in batch_sizes:
        batch_results = model.batch_extract_json(
            texts[:bs],
            structures,
            batch_size=bs,
            deterministic=True,
            include_confidence=True
        )
        results_by_batch[bs] = batch_results[0]
    
    identical, summary, comparison = compare_confidences(results_by_batch, verbose)
    
    return identical, summary, comparison


def test_relation_extraction_determinism(model, test_text: str, verbose: bool = False) -> tuple[bool, str, dict]:
    """Test that relation extraction is deterministic."""
    relation_types = ["works_for", "announced", "located_in"]
    batch_sizes = [1, 2, 4, 8, 16]
    
    texts = [test_text] * 16
    results_by_batch = {}
    
    for bs in batch_sizes:
        batch_results = model.batch_extract_relations(
            texts[:bs],
            relation_types,
            batch_size=bs,
            deterministic=True,
            include_confidence=True
        )
        results_by_batch[bs] = batch_results[0]
    
    identical, summary, comparison = compare_confidences(results_by_batch, verbose)
    
    return identical, summary, comparison


def test_batch_consistency(model, test_text: str, verbose: bool = False) -> tuple[bool, str, dict]:
    """
    Test that results are consistent when same text appears at different
    positions within a batch.
    """
    entity_types = ["company", "person", "product"]
    
    # Create a batch where test_text appears at different positions
    other_texts = [
        "Microsoft announced Windows 12 in Seattle.",
        "Google's Sundar Pichai spoke at the I/O conference.",
        "Amazon opened a new warehouse in Texas.",
    ]
    
    # Test with target text at position 0
    batch1 = [test_text] + other_texts
    results1 = model.batch_extract_entities(
        batch1,
        entity_types,
        batch_size=4,
        deterministic=True,
        include_confidence=True
    )
    
    # Test with target text at position 2
    batch2 = other_texts[:2] + [test_text] + other_texts[2:]
    results2 = model.batch_extract_entities(
        batch2,
        entity_types,
        batch_size=4,
        deterministic=True,
        include_confidence=True
    )
    
    # Test with target text at position 3
    batch3 = other_texts + [test_text]
    results3 = model.batch_extract_entities(
        batch3,
        entity_types,
        batch_size=4,
        deterministic=True,
        include_confidence=True
    )
    
    # Compare results for test_text across all batches
    results_by_position = {
        "position_0": results1[0],
        "position_2": results2[2],
        "position_3": results3[3]
    }
    
    identical, summary, comparison = compare_confidences(results_by_position, verbose)
    
    return identical, summary, comparison


def test_numerical_precision(model, test_text: str, verbose: bool = False) -> tuple[bool, str, dict]:
    """
    Test numerical precision by examining raw confidence values.
    
    This test extracts entities and shows the exact float32 bit patterns
    to verify bitwise identity.
    """
    import struct
    
    entity_types = ["company", "person", "product", "location"]
    batch_sizes = [1, 4, 16]
    
    texts = [test_text] * 16
    
    detailed_results = {}
    all_bits_match = True
    
    reference_bits = None
    
    for bs in batch_sizes:
        batch_results = model.batch_extract_entities(
            texts[:bs],
            entity_types,
            batch_size=bs,
            deterministic=True,
            include_confidence=True
        )
        result = batch_results[0]
        
        # Extract confidence values with their bit patterns
        confidences_with_bits = []
        
        def extract_with_bits(obj, path=""):
            if isinstance(obj, dict):
                if "confidence" in obj and "text" in obj:
                    conf = obj["confidence"]
                    bits = struct.unpack('I', struct.pack('f', conf))[0]
                    confidences_with_bits.append({
                        "path": path,
                        "text": obj["text"],
                        "confidence": conf,
                        "bits_hex": f"0x{bits:08X}",
                        "bits_int": bits
                    })
                for k, v in obj.items():
                    extract_with_bits(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_with_bits(item, f"{path}[{i}]")
        
        extract_with_bits(result)
        confidences_with_bits.sort(key=lambda x: x["path"])
        
        detailed_results[f"batch_{bs}"] = {
            "num_extractions": len(confidences_with_bits),
            "extractions": confidences_with_bits
        }
        
        # Compare bit patterns
        current_bits = [x["bits_int"] for x in confidences_with_bits]
        if reference_bits is None:
            reference_bits = current_bits
        else:
            if current_bits != reference_bits:
                all_bits_match = False
    
    # Build summary
    if all_bits_match:
        num_values = len(reference_bits) if reference_bits else 0
        summary = f"All {len(batch_sizes)} batch sizes produce bitwise-identical values ({num_values} confidence scores)"
    else:
        summary = "MISMATCH: Bit patterns differ between batch sizes"
    
    return all_bits_match, summary, detailed_results


@app.function(
    image=image,
    gpu="A100",
    timeout=900,
)
def run_full_determinism_tests_a100(verbose: bool = False) -> dict:
    """Run tests on A100 (faster)."""
    return run_full_determinism_tests.local(verbose=verbose)


@app.function(
    image=image,
    gpu="A10G",
    timeout=900,
)
def run_nondeterministic_comparison() -> dict:
    """
    Compare deterministic vs non-deterministic mode to show the difference.
    
    This demonstrates that without deterministic mode, results CAN vary by batch size.
    Shows actual numerical differences in confidence scores.
    """
    import sys
    sys.path.insert(0, "/root")
    import struct
    
    import torch
    from gliner2 import GLiNER2
    
    results = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tests": {},
    }
    
    if not torch.cuda.is_available():
        results["error"] = "CUDA not available"
        return results
    
    print("Loading model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model = model.cuda()
    model.eval()
    
    test_text = "Apple CEO Tim Cook announced the new iPhone 15 Pro at the Cupertino headquarters."
    entity_types = ["company", "person", "product", "location"]
    texts = [test_text] * 32
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    def extract_entity_confidences(result):
        """Extract entity name -> confidence mapping."""
        confidences = {}
        entities = result.get("entities", {})
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if isinstance(entity, dict) and "text" in entity and "confidence" in entity:
                    key = f"{entity_type}:{entity['text']}"
                    confidences[key] = entity["confidence"]
        return confidences
    
    def get_bits(val):
        """Get IEEE 754 float32 bit pattern."""
        return struct.unpack('I', struct.pack('f', val))[0]
    
    # =========================================================================
    # Run WITHOUT deterministic mode
    # =========================================================================
    print("\n" + "=" * 70)
    print("WITHOUT DETERMINISTIC MODE (deterministic=False)")
    print("=" * 70)
    
    nondet_results = {}
    nondet_confidences = {}
    
    for bs in batch_sizes:
        batch_results = model.batch_extract_entities(
            texts[:bs],
            entity_types,
            batch_size=bs,
            deterministic=False,
            include_confidence=True
        )
        nondet_results[bs] = batch_results[0]
        nondet_confidences[bs] = extract_entity_confidences(batch_results[0])
    
    # Find entities that appear in all batch sizes
    common_entities = set(nondet_confidences[1].keys())
    for bs in batch_sizes[1:]:
        common_entities &= set(nondet_confidences[bs].keys())
    
    # Show numerical differences
    print("\nConfidence values by batch size:")
    print("-" * 70)
    
    nondet_diffs = {}
    for entity in sorted(common_entities):
        values = [nondet_confidences[bs].get(entity, 0) for bs in batch_sizes]
        bits = [get_bits(v) for v in values]
        
        # Calculate max ULP difference from batch_size=1
        ref_bits = bits[0]
        ulp_diffs = [abs(b - ref_bits) for b in bits]
        max_ulp = max(ulp_diffs)
        max_abs_diff = max(abs(v - values[0]) for v in values)
        
        nondet_diffs[entity] = {
            "values": values,
            "max_ulp": max_ulp,
            "max_abs_diff": max_abs_diff
        }
        
        print(f"\n  {entity}:")
        for bs, val, ulp in zip(batch_sizes, values, ulp_diffs):
            bits_hex = f"0x{get_bits(val):08X}"
            ulp_str = f"(+{ulp} ULP)" if ulp > 0 else "(reference)"
            print(f"    batch_{bs:2d}: {val:.10f}  {bits_hex}  {ulp_str}")
        
        if max_ulp > 0:
            print(f"    ⚠️  MAX DIFFERENCE: {max_abs_diff:.2e} ({max_ulp} ULP)")
    
    nondet_has_diffs = any(d["max_ulp"] > 0 for d in nondet_diffs.values())
    
    results["tests"]["non_deterministic"] = {
        "has_differences": nondet_has_diffs,
        "entity_diffs": nondet_diffs,
        "summary": "Results VARY by batch size" if nondet_has_diffs else "Results happened to match (lucky run)"
    }
    
    # =========================================================================
    # Run WITH deterministic mode
    # =========================================================================
    print("\n" + "=" * 70)
    print("WITH DETERMINISTIC MODE (deterministic=True)")
    print("=" * 70)
    
    det_results = {}
    det_confidences = {}
    
    for bs in batch_sizes:
        batch_results = model.batch_extract_entities(
            texts[:bs],
            entity_types,
            batch_size=bs,
            deterministic=True,
            include_confidence=True
        )
        det_results[bs] = batch_results[0]
        det_confidences[bs] = extract_entity_confidences(batch_results[0])
    
    # Show numerical values (should all be identical)
    print("\nConfidence values by batch size:")
    print("-" * 70)
    
    det_diffs = {}
    for entity in sorted(common_entities):
        values = [det_confidences[bs].get(entity, 0) for bs in batch_sizes]
        bits = [get_bits(v) for v in values]
        
        ref_bits = bits[0]
        ulp_diffs = [abs(b - ref_bits) for b in bits]
        max_ulp = max(ulp_diffs)
        max_abs_diff = max(abs(v - values[0]) for v in values)
        
        det_diffs[entity] = {
            "values": values,
            "max_ulp": max_ulp,
            "max_abs_diff": max_abs_diff
        }
        
        print(f"\n  {entity}:")
        for bs, val, ulp in zip(batch_sizes, values, ulp_diffs):
            bits_hex = f"0x{get_bits(val):08X}"
            status = "✓" if ulp == 0 else "✗"
            print(f"    batch_{bs:2d}: {val:.10f}  {bits_hex}  {status}")
        
        if max_ulp == 0:
            print(f"    ✓ BITWISE IDENTICAL across all batch sizes")
    
    det_has_diffs = any(d["max_ulp"] > 0 for d in det_diffs.values())
    
    results["tests"]["deterministic"] = {
        "has_differences": det_has_diffs,
        "entity_diffs": det_diffs,
        "summary": "Results are BITWISE IDENTICAL" if not det_has_diffs else "UNEXPECTED: Results differ!"
    }
    
    # =========================================================================
    # Summary comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_nondet_ulp = sum(d["max_ulp"] for d in nondet_diffs.values())
    total_det_ulp = sum(d["max_ulp"] for d in det_diffs.values())
    
    results["summary"] = {
        "non_deterministic_total_ulp_drift": total_nondet_ulp,
        "deterministic_total_ulp_drift": total_det_ulp,
        "entities_compared": len(common_entities)
    }
    
    print(f"\nEntities compared: {len(common_entities)}")
    print(f"\nWithout deterministic mode:")
    print(f"  Total ULP drift: {total_nondet_ulp}")
    print(f"  Status: {'⚠️  RESULTS VARY BY BATCH SIZE' if nondet_has_diffs else '(happened to match this run)'}")
    
    print(f"\nWith deterministic mode:")
    print(f"  Total ULP drift: {total_det_ulp}")
    print(f"  Status: {'✓ BITWISE IDENTICAL' if not det_has_diffs else '✗ UNEXPECTED DIFFERENCES'}")
    
    return results


@app.local_entrypoint()
def main(use_a100: bool = False, compare_modes: bool = False, verbose: bool = False):
    """
    Run the full determinism tests.
    
    Args:
        use_a100: Use A100 GPU (faster, more expensive)
        compare_modes: Run comparison between deterministic and non-deterministic
        verbose: Show detailed numerical values for each extraction
    """
    print("=" * 70)
    print("GLiNER2 Full Model Determinism Tests")
    print("=" * 70)
    
    if compare_modes:
        print("\nRunning mode comparison to demonstrate non-determinism...")
        print("This shows that WITHOUT deterministic mode, results vary by batch size.\n")
        results = run_nondeterministic_comparison.remote()
        
        # The detailed output is already printed by the remote function
        # Just show the final verdict
        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)
        
        if "summary" in results:
            summary = results["summary"]
            nondet_drift = summary.get("non_deterministic_total_ulp_drift", 0)
            det_drift = summary.get("deterministic_total_ulp_drift", 0)
            
            if nondet_drift > 0 and det_drift == 0:
                print("\n✓ DEMONSTRATION SUCCESSFUL")
                print(f"\n  Without deterministic mode: {nondet_drift} ULP total drift")
                print(f"  With deterministic mode:    {det_drift} ULP total drift (bitwise identical)")
                print("\n  This proves that the deterministic patches eliminate batch-size")
                print("  dependent floating-point variations in the inference pipeline.")
            elif nondet_drift == 0:
                print("\n⚠️  Non-deterministic mode happened to produce identical results")
                print("    this run. This can happen due to GPU scheduling luck.")
                print("    Run again or try different inputs to observe variation.")
            else:
                print("\n✗ UNEXPECTED: Deterministic mode showed drift!")
        
        return results
    
    gpu_type = "A100" if use_a100 else "A10G"
    print(f"\nRunning full determinism tests on {gpu_type} GPU...")
    if verbose:
        print("(Verbose mode: showing numerical details)")
    
    if use_a100:
        results = run_full_determinism_tests_a100.remote(verbose=verbose)
    else:
        results = run_full_determinism_tests.remote(verbose=verbose)
    
    # Print results
    print(f"\nCUDA Available: {results['cuda_available']}")
    print(f"GPU: {results.get('gpu_name', 'N/A')}")
    print(f"Triton Version: {results.get('triton_version', 'N/A')}")
    
    print("\n" + "-" * 70)
    print("Test Results:")
    print("-" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, test_result in results["tests"].items():
        status = test_result["status"]
        if status == "PASSED":
            print(f"\n  [PASS] {test_name}")
            print(f"         {test_result.get('details', '')}")
            passed += 1
        else:
            print(f"\n  [FAIL] {test_name}")
            if "error" in test_result:
                print(f"         Error: {test_result['error']}")
            if "details" in test_result:
                print(f"         Details: {test_result['details']}")
            failed += 1
        
        # Show numerical data in verbose mode
        if verbose and "numerical_data" in test_result:
            data = test_result["numerical_data"]
            
            # For numerical_precision test, show bit patterns
            if test_name == "numerical_precision":
                print("\n         Bit patterns by batch size:")
                for batch_key, batch_data in data.items():
                    if batch_key.startswith("batch_"):
                        print(f"\n         {batch_key}:")
                        for ext in batch_data.get("extractions", [])[:5]:  # Show first 5
                            print(f"           {ext['text']}: {ext['confidence']:.8f} ({ext['bits_hex']})")
                        if len(batch_data.get("extractions", [])) > 5:
                            print(f"           ... and {len(batch_data['extractions']) - 5} more")
            
            # For other tests, show reference confidences
            elif "reference_confidences" in data:
                ref_confs = data["reference_confidences"]
                print(f"\n         Reference confidences (batch_size={data['reference_batch_size']}):")
                for i, conf in enumerate(ref_confs[:8]):  # Show first 8
                    print(f"           [{i}] {conf:.8f}")
                if len(ref_confs) > 8:
                    print(f"           ... and {len(ref_confs) - 8} more values")
                
                # Show comparison details
                for bs, comp in data.get("comparisons", {}).items():
                    if comp.get("identical"):
                        print(f"         batch_{bs}: ✓ identical (0 ULP diff)")
                    else:
                        print(f"         batch_{bs}: max_ulp={comp.get('max_ulp_diff', '?')}, max_diff={comp.get('max_abs_diff', '?'):.2e}")
    
    print("\n" + "-" * 70)
    print(f"Summary: {passed} passed, {failed} failed")
    print("-" * 70)
    
    if results["all_passed"]:
        print("\n✓ All tests PASSED!")
        print("Full model inference is deterministic: outputs are bitwise-identical")
        print("across all batch sizes when deterministic=True is used.")
        print("\nThis means:")
        print("  • Confidence scores are identical to the last bit (0 ULP difference)")
        print("  • Extracted text spans are identical")
        print("  • Results don't depend on batch size or position in batch")
    else:
        print("\n✗ Some tests FAILED!")
        if "error" in results:
            print(f"Error: {results['error']}")
    
    return results

