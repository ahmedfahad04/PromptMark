"""
Watermarking Experiments
========================
Collection of watermark embedding strategies for code.

Included Strategies:
- exp_code_only: Base watermark only in code identifiers
- exp_comment_wm: Watermark in both code and comments
- exp_iterative_wm: Iterative refinement approach
- exp_refactoring_wm: Post-hoc identifier refactoring
- exp_static_wm: Static alphabet division
"""

__all__ = [
    "exp_code_only",
    "exp_comment_wm",
    "exp_iterative_wm",
    "exp_refactoring_wm",
    "exp_static_wm",
]
