"""
Batch-Invariant Kernels for Deterministic Inference

This module provides custom Triton kernels that guarantee identical outputs
regardless of batch size. Based on research from Thinking Machines Lab:
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

The key insight is that standard GPU kernels use different reduction strategies
for different batch sizes, causing floating-point non-associativity to produce
different results. These kernels use fixed tile sizes and sequential reduction
ordering to ensure batch invariance.
"""

from gliner2.kernels.matmul import batch_invariant_matmul, batch_invariant_linear
from gliner2.kernels.layernorm import batch_invariant_layernorm
from gliner2.kernels.span_score import batch_invariant_span_score
from gliner2.kernels.softmax import batch_invariant_softmax

__all__ = [
    "batch_invariant_matmul",
    "batch_invariant_linear",
    "batch_invariant_layernorm",
    "batch_invariant_span_score",
    "batch_invariant_softmax",
]
