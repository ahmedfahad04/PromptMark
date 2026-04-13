"""
AST Structural Watermark Detector

Detects presence of AST bigram watermark in code using statistical testing.
Combines binomial test with z-score for robust detection.

Usage:
    detector = StructuralWatermarkDetector(scheme_path)
    result = detector.detect(generated_code)
"""

import ast
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats


# ============================================================================
# AST BIGRAM EXTRACTION (from extract_bigram_corpus.py)
# ============================================================================

CONTROL_FLOW_NODES = {
    "FunctionDef", "AsyncFunctionDef", "For", "While", "If", "Return",
    "Break", "Continue", "Try", "ExceptHandler", "With", "Assert",
    "Call", "Assign", "AugAssign", "AnnAssign",
    "ListComp", "SetComp", "DictComp", "GeneratorExp",
    "Raise", "Pass",
}

EXCLUDE_NODES = {
    "Import", "ImportFrom", "Global", "Nonlocal", "Expr",
    "Module", "Interactive", "Expression", "Compare", "BinOp",
    "UnaryOp", "BoolOp",
}


def extract_nested_bigrams(code: str) -> List[Tuple[str, str]]:
    """Extract parent-child AST node-type bigrams from code."""
    bigrams = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    def walk_and_extract(node, parent_type: Optional[str] = None):
        node_type = node.__class__.__name__

        if node_type in EXCLUDE_NODES:
            for child in ast.iter_child_nodes(node):
                walk_and_extract(child, parent_type)
            return

        if parent_type and node_type in CONTROL_FLOW_NODES:
            if parent_type in CONTROL_FLOW_NODES:
                bigrams.append((parent_type, node_type))

        visited_children = set()

        if hasattr(node, "body") and isinstance(node.body, list):
            for child in node.body:
                walk_and_extract(
                    child,
                    parent_type=node_type if node_type in CONTROL_FLOW_NODES else parent_type
                )
                visited_children.add(id(child))

        if hasattr(node, "orelse") and isinstance(node.orelse, list):
            for child in node.orelse:
                walk_and_extract(
                    child,
                    parent_type=node_type if node_type in CONTROL_FLOW_NODES else parent_type
                )
                visited_children.add(id(child))

        if hasattr(node, "handlers") and isinstance(node.handlers, list):
            for child in node.handlers:
                walk_and_extract(
                    child,
                    parent_type=node_type if node_type in CONTROL_FLOW_NODES else parent_type
                )
                visited_children.add(id(child))

        if hasattr(node, "finalbody") and isinstance(node.finalbody, list):
            for child in node.finalbody:
                walk_and_extract(
                    child,
                    parent_type=node_type if node_type in CONTROL_FLOW_NODES else parent_type
                )
                visited_children.add(id(child))

        for child in ast.iter_child_nodes(node):
            if id(child) not in visited_children:
                walk_and_extract(
                    child,
                    parent_type=node_type if node_type in CONTROL_FLOW_NODES else parent_type
                )

    walk_and_extract(tree)
    return bigrams


# ============================================================================
# STRUCTURAL WATERMARK DETECTOR
# ============================================================================

class StructuralWatermarkDetector:
    """
    Detects AST bigram-based structural watermark in code.

    Workflow:
    1. Load binning scheme with green bins
    2. Extract bigrams from generated code
    3. Count green vs total bigrams
    4. Compute z-score and p-value using binomial test
    5. Make detection decision based on threshold
    """

    def __init__(
        self,
        scheme_path: str,
        z_threshold: float = 1.645,  # α = 0.05 (one-sided)
        verbose: bool = False,
    ):
        """
        Initialize detector.

        Args:
            scheme_path: Path to binning scheme JSON
            z_threshold: Z-score threshold for detection (default 1.645 → p < 0.05)
            verbose: If True, print detection details
        """
        self.scheme_path = Path(scheme_path)
        self.z_threshold = z_threshold
        self.verbose = verbose

        # Load scheme
        with open(scheme_path, "r") as f:
            self.scheme = json.load(f)

        # Build bigram-to-bin mapping
        self._build_bigram_to_bin_mapping()

        # Extract configuration
        self.total_bins = self.scheme["binning_config"]["total_bins"]
        self.green_bins = self.scheme["green_bins"]["ids"]
        self.gamma = self.scheme["binning_config"]["gamma_baseline"]

    def _build_bigram_to_bin_mapping(self):
        """Create mapping from (parent, child) → bin_id for fast lookup."""
        self.bigram_to_bin = {}

        for bin_name, bin_data in self.scheme["bins"].items():
            bin_id = bin_data["bin_id"]
            for bigram in bin_data["bigrams"]:
                key = (bigram["parent_type"], bigram["child_type"])
                self.bigram_to_bin[key] = bin_id

    def detect(
        self,
        code: str,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect watermark in generated code.

        Args:
            code: Generated Python code
            task_id: Optional task identifier (for logging)

        Returns:
            Dictionary with detection results:
            {
                'task_id': str,
                'status': 'SUCCESS' | 'ERROR',
                'reason': str (if ERROR),
                'total_bigrams_found': int,
                'total_recognized': int,
                'green_count': int,
                'gamma_empirical': float,
                'gamma_expected': float,
                'z_score': float,
                'p_value': float,
                'p_exact': float,
                'watermark_detected': bool,
                'confidence': float,
                'bin_distribution': Dict[int, int],
            }
        """
        result = {
            "task_id": task_id or "unknown",
            "status": "SUCCESS",
            "reason": None,
        }

        # Extract bigrams from code
        try:
            extracted_bigrams = extract_nested_bigrams(code)
        except Exception as e:
            result["status"] = "ERROR"
            result["reason"] = f"Failed to parse code: {str(e)}"
            result["total_bigrams_found"] = 0
            result["watermark_detected"] = False
            return result

        if not extracted_bigrams:
            result["status"] = "ERROR"
            result["reason"] = "No control-flow bigrams found in code"
            result["total_bigrams_found"] = 0
            result["watermark_detected"] = False
            return result

        # Map bigrams to bins and count
        bin_counts = {}
        green_count = 0
        total_recognized = 0

        for parent_type, child_type in extracted_bigrams:
            key = (parent_type, child_type)
            if key in self.bigram_to_bin:
                bin_id = self.bigram_to_bin[key]
                bin_counts[bin_id] = bin_counts.get(bin_id, 0) + 1
                total_recognized += 1

                if bin_id in self.green_bins:
                    green_count += 1

        result["total_bigrams_found"] = len(extracted_bigrams)
        result["total_recognized"] = total_recognized

        if total_recognized == 0:
            result["status"] = "ERROR"
            result["reason"] = "No recognized bigrams in code"
            result["watermark_detected"] = False
            return result

        # Compute statistics
        gamma_empirical = green_count / total_recognized
        gamma_expected = self.gamma

        result["green_count"] = green_count
        result["gamma_empirical"] = gamma_empirical
        result["gamma_expected"] = gamma_expected

        # Compute z-score
        se = math.sqrt(gamma_expected * (1 - gamma_expected) / total_recognized)
        if se > 0:
            z_score = (gamma_empirical - gamma_expected) / se
        else:
            z_score = 0.0

        # Compute p-value (one-sided test: H1: gamma > gamma_expected)
        p_value = 1 - stats.norm.cdf(z_score) if not math.isnan(z_score) else 1.0

        # Compute exact binomial p-value for completeness
        try:
            p_exact = stats.binomtest(
                green_count, total_recognized, gamma_expected, alternative="greater"
            ).pvalue
        except:
            p_exact = p_value

        result["z_score"] = float(z_score)
        result["p_value"] = float(p_value)
        result["p_exact"] = float(p_exact)

        # Detection decision
        watermark_detected = p_value < (1 - stats.norm.cdf(self.z_threshold))
        result["watermark_detected"] = bool(watermark_detected)
        result["confidence"] = float(abs(z_score))

        # Bin distribution
        result["bin_distribution"] = {
            f"bin_{bin_id}": count
            for bin_id, count in sorted(bin_counts.items())
        }

        if self.verbose:
            self._print_detection_summary(result)

        return result

    def _print_detection_summary(self, result: Dict):
        """Print formatted detection summary."""
        print(f"\n{'=' * 70}")
        print(f"STRUCTURAL WATERMARK DETECTION")
        print(f"{'=' * 70}")
        print(f"Task ID:                    {result['task_id']}")
        print(f"Status:                     {result['status']}")

        if result["status"] == "ERROR":
            print(f"Reason:                     {result['reason']}")
            return

        print(f"Total bigrams found:        {result['total_bigrams_found']}")
        print(f"Recognized bigrams:         {result['total_recognized']}")
        print(f"Green bigrams:              {result['green_count']}")
        print(f"\nGamma (empirical):          {result['gamma_empirical']:.4f}")
        print(f"Gamma (expected):           {result['gamma_expected']:.4f}")
        print(f"Z-score:                    {result['z_score']:.4f}")
        print(f"P-value (normal):           {result['p_value']:.6f}")
        print(f"P-value (exact binomial):   {result['p_exact']:.6f}")
        print(f"Confidence (|z|):           {result['confidence']:.4f}")
        print(f"\nWatermark detected:         {'✅ YES' if result['watermark_detected'] else '❌ NO'}")
        print(f"{'=' * 70}\n")

    def detect_batch(
        self,
        code_dict: Dict[str, str],
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Detect watermark in multiple code samples.

        Args:
            code_dict: Dictionary of {task_id: code}
            verbose: If True, print per-sample results

        Returns:
            List of detection results
        """
        results = []
        for task_id, code in code_dict.items():
            result = self.detect(code, task_id=task_id)
            if verbose:
                print(f"[{task_id}] Watermark: {result['watermark_detected']}")
            results.append(result)
        return results

    @staticmethod
    def compute_roc_metrics(
        results: List[Dict],
        synthetic_label: bool = True,
    ) -> Dict:
        """
        Compute ROC metrics (TPR, FPR) for detection evaluation.

        Args:
            results: List of detection results
            synthetic_label: If True, treat as synthetic (watermarked) code

        Returns:
            Dictionary with aggregate metrics
        """
        detected = sum(1 for r in results if r.get("watermark_detected", False))
        total = len(results)

        return {
            "total_samples": total,
            "detected_count": detected,
            "detection_rate": detected / total if total > 0 else 0.0,
            "label": "synthetic" if synthetic_label else "genuine",
        }


# ============================================================================
# UTILITIES
# ============================================================================

def load_and_detect(
    scheme_path: str,
    code_paths: Dict[str, str],
    z_threshold: float = 1.645,
) -> List[Dict]:
    """
    Convenience function to load detector and run batch detection.

    Args:
        scheme_path: Path to binning scheme
        code_paths: Dictionary of {task_id: code_file_path}
        z_threshold: Detection threshold

    Returns:
        List of detection results
    """
    detector = StructuralWatermarkDetector(scheme_path, z_threshold=z_threshold)

    code_dict = {}
    for task_id, code_path in code_paths.items():
        try:
            with open(code_path, "r") as f:
                code_dict[task_id] = f.read()
        except FileNotFoundError:
            print(f"⚠️  File not found: {code_path}")

    return detector.detect_batch(code_dict, verbose=True)


if __name__ == "__main__":
    # Example usage
    print("🔍 Structural Watermark Detector Demo\n")

    # Test code with bigrams
    test_code = """
def solve(n):
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(i)
    return result
"""

    # Load detector
    detector = StructuralWatermarkDetector(
        "data/bigram_binning_scheme_v1.json",
        verbose=True,
    )

    # Detect
    result = detector.detect(test_code, task_id="test_1")

    print("\nDetection result:")
    print(f"  Watermark detected: {result['watermark_detected']}")
    print(f"  P-value: {result['p_value']:.6f}")
    print(f"  Z-score: {result['z_score']:.4f}")
    print(f"  Green count: {result['green_count']} / {result['total_recognized']}")

    print("\n✅ Demo complete!")
