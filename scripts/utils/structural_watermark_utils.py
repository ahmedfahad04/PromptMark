"""
Structural Watermark Integration Utilities

Provides convenience functions for embedding and detecting structural watermarks
alongside the existing lexical watermark in PromptMark.

Usage:
    embeds = get_structural_embedding_prompt(task_id)
    detector = get_structural_detector(task_id)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import random

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from bigram_analysis.binning_strategy import (
    load_binning_scheme,
    get_green_structural_hints,
)
from metrics.ast_bigram_detector import StructuralWatermarkDetector


# ============================================================================
# STRUCTURAL EMBEDDING
# ============================================================================

def get_task_specific_green_bins(
    task_id: str,
    scheme_path: str = "data/bigram_binning_scheme_v1.json",
) -> List[int]:
    """
    Get green bins for a specific task using deterministic selection.

    Same task_id always yields same green bins, but different tasks get different bins.

    Args:
        task_id: Task identifier (e.g., "HumanEval/0")
        scheme_path: Path to binning scheme JSON

    Returns:
        List of green bin IDs for this task
    """
    with open(scheme_path, "r") as f:
        scheme = json.load(f)

    total_bins = scheme["binning_config"]["total_bins"]

    # Create deterministic seed from task_id
    seed_hash = int(hashlib.md5(task_id.encode()).hexdigest(), 16)
    rng = random.Random(seed_hash)

    # Select 50% of bins
    all_bins = list(range(total_bins))
    rng.shuffle(all_bins)
    num_green = max(1, total_bins // 2)
    green_bins = sorted(all_bins[:num_green])

    return green_bins


def get_structural_embedding_prompt(
    task_id: str,
    scheme_path: str = "data/bigram_binning_scheme_v1.json",
) -> str:
    """
    Generate a prompt component with structural hints for code generation.

    Args:
        task_id: Task identifier
        scheme_path: Path to binning scheme JSON

    Returns:
        Prompt component with structural guidance (natural language)
    """
    with open(scheme_path, "r") as f:
        scheme = json.load(f)

    green_bins = get_task_specific_green_bins(task_id, scheme_path)

    # Get hints for the green bins
    hints = get_green_structural_hints(scheme, green_bins)

    if not hints:
        return ""

    hint_text = "\n".join([f"  • {hint}" for hint in hints[:5]])  # Limit to 5 hints

    prompt = f"""### Structural Code Organization:
When writing code for this problem, consider these structural patterns:
{hint_text}

These patterns help write clear, well-organized code that naturally follows good algorithmic practices.
"""

    return prompt


def get_structural_detector(
    scheme_path: str = "data/bigram_binning_scheme_v1.json",
    z_threshold: float = 1.645,
    verbose: bool = False,
) -> StructuralWatermarkDetector:
    """
    Initialize a structural watermark detector.

    Args:
        scheme_path: Path to binning scheme JSON
        z_threshold: Detection threshold (default 1.645 for p < 0.05)
        verbose: If True, print detection details

    Returns:
        StructuralWatermarkDetector instance
    """
    return StructuralWatermarkDetector(
        scheme_path=scheme_path,
        z_threshold=z_threshold,
        verbose=verbose,
    )


# ============================================================================
# DUAL-CHANNEL WATERMARKING
# ============================================================================

def combine_watermark_signals(
    lexical_z: float,
    structural_z: float,
    lexical_weight: float = 0.5,
) -> Dict[str, float]:
    """
    Combine lexical and structural watermark signals.

    Uses weighted combination of z-scores.

    Args:
        lexical_z: Z-score from lexical watermark (identifier starting letters)
        structural_z: Z-score from structural watermark (AST bigrams)
        lexical_weight: Weight for lexical signal (0-1), structural gets (1-weight)

    Returns:
        Dictionary with combined metrics
    """
    from scipy import stats

    structural_weight = 1.0 - lexical_weight

    # Weighted combination
    combined_z = (lexical_weight * lexical_z) + (structural_weight * structural_z)

    # Combined p-value (approximation)
    combined_p = 1 - stats.norm.cdf(combined_z)

    return {
        "combined_z": float(combined_z),
        "combined_p": float(combined_p),
        "watermark_detected": combined_p < 0.05,
        "lexical_z": float(lexical_z),
        "structural_z": float(structural_z),
    }


def fisher_combine_pvalues(p1: float, p2: float) -> Tuple[float, float]:
    """
    Combine two independent p-values using Fisher's method.

    More powerful than simple AND/OR logic.

    Args:
        p1: First p-value (lexical)
        p2: Second p-value (structural)

    Returns:
        Tuple of (combined_z, combined_p)
    """
    import math
    from scipy import stats

    # Avoid log(0)
    p1 = max(p1, 1e-300)
    p2 = max(p2, 1e-300)

    # Fisher's combined test statistic: X² = -2(ln(p1) + ln(p2))
    chi_sq_stat = -2 * (math.log(p1) + math.log(p2))

    # p-value from chi-squared distribution with 4 degrees of freedom
    combined_p = 1 - stats.chi2.cdf(chi_sq_stat, df=4)

    # Convert to z-score equivalent
    combined_z = stats.norm.ppf(1 - combined_p)

    return combined_z, combined_p


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_both_watermarks(
    code: str,
    task_id: str,
    lexical_result: Dict,
    scheme_path: str = "data/bigram_binning_scheme_v1.json",
) -> Dict:
    """
    Evaluate both lexical and structural watermarks for a code sample.

    Args:
        code: Generated Python code
        task_id: Task identifier
        lexical_result: Result from lexical watermark detector
        scheme_path: Path to binning scheme

    Returns:
        Combined evaluation result
    """
    # Detect structural watermark
    detector = get_structural_detector(scheme_path, verbose=False)
    structural_result = detector.detect(code, task_id=task_id)

    # Extract key metrics
    lexical_z = lexical_result.get("z", 0.0) or 0.0
    structural_z = structural_result.get("z_score", 0.0) or 0.0

    lexical_p = lexical_result.get("p_unified", 1.0) or 1.0
    structural_p = structural_result.get("p_value", 1.0) or 1.0

    # Combine signals (Fisher's method)
    combined_z, combined_p = fisher_combine_pvalues(lexical_p, structural_p)

    return {
        # Lexical channel
        "lexical_detected": lexical_result.get("meets_z", False),
        "lexical_z": float(lexical_z),
        "lexical_p": float(lexical_p),
        "lexical_token_count": lexical_result.get("token_count", 0),
        "lexical_green_count": lexical_result.get("green_count", 0),
        # Structural channel
        "structural_detected": structural_result.get("watermark_detected", False),
        "structural_z": float(structural_z),
        "structural_p": float(structural_p),
        "structural_token_count": structural_result.get("total_recognized", 0),
        "structural_green_count": structural_result.get("green_count", 0),
        # Combined
        "combined_z": float(combined_z),
        "combined_p": float(combined_p),
        "both_detected": lexical_result.get("meets_z", False) and structural_result.get("watermark_detected", False),
        "either_detected": lexical_result.get("meets_z", False) or structural_result.get("watermark_detected", False),
        # Details
        "lexical_details": lexical_result,
        "structural_details": structural_result,
    }


# ============================================================================
# SETUP & VALIDATION
# ============================================================================

def validate_structural_setup(
    scheme_path: str = "data/bigram_binning_scheme_v1.json",
) -> bool:
    """
    Validate that all required resources for structural watermarking are present.

    Args:
        scheme_path: Path to check for binning scheme

    Returns:
        True if all resources are available, False otherwise
    """
    scheme_file = Path(scheme_path)

    if not scheme_file.exists():
        print(f"❌ Missing binning scheme: {scheme_path}")
        return False

    try:
        with open(scheme_path, "r") as f:
            scheme = json.load(f)
        required_keys = ["metadata", "binning_config", "green_bins", "bins"]
        if not all(key in scheme for key in required_keys):
            print(f"⚠️  Invalid scheme format")
            return False
        print(f"✅ Binning scheme valid: {len(scheme['bins'])} bins")
        return True
    except Exception as e:
        print(f"❌ Error loading scheme: {e}")
        return False


def print_setup_summary(
    scheme_path: str = "data/bigram_binning_scheme_v1.json",
):
    """Print summary of structural watermarking setup."""
    print("\n" + "=" * 70)
    print("STRUCTURAL WATERMARKING SETUP SUMMARY")
    print("=" * 70)

    if not validate_structural_setup(scheme_path):
        print("\n⚠️  Structural watermarking not properly configured")
        return

    with open(scheme_path, "r") as f:
        scheme = json.load(f)

    config = scheme["binning_config"]
    green_bins = scheme["green_bins"]

    print(f"\nBinning Scheme:")
    print(f"  Total bigrams:    {config['total_bigrams']}")
    print(f"  Total bins:       {config['total_bins']}")
    print(f"  Bin width:        {config['bin_width']}")
    print(f"  Green bins:       {green_bins['count']} / {config['total_bins']} ({green_bins['proportion']:.0%})")
    print(f"  Gamma (baseline): {config['gamma_baseline']:.3f}")

    print(f"\nDetection Threshold:")
    print(f"  Z-score:  1.645 (α = 0.05)")
    print(f"  P-value:  0.05")

    print(f"\nEmbedding:")
    print(f"  Method: Task-specific green bins (deterministic per task_id)")
    print(f"  Guidance: Natural language structural hints in system prompt")

    print(f"\nCombined Watermarking:")
    print(f"  Lexical + Structural dual-channel")
    print(f"  Fisher's combined p-value test")
    print(f"  Both must detect for strong confidence")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("🔧 Structural Watermark Integration Utility\n")

    # Test imports
    try:
        validate_structural_setup()
        print_setup_summary()

        # Test prompt generation
        prompt = get_structural_embedding_prompt("HumanEval/0")
        print("Sample embedding prompt:")
        print(prompt)

        print("\n✅ Integration utilities ready!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
