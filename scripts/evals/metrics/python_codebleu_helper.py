#!/usr/bin/env python3
"""
CodeBLEU Evaluation Helper for Python Code

This script provides easy-to-use functions for calculating CodeBLEU scores 
specifically for Python code, with support for both file-based and string-based evaluation.

Usage examples:
1. Evaluate single example:
   score = evaluate_python_code_bleu(reference_code, generated_code)

2. Evaluate from files:
   score = evaluate_python_files(reference_file, hypothesis_file)

3. Batch evaluation:
   scores = batch_evaluate_python([ref1, ref2], [gen1, gen2])
"""

import os
import sys
import tempfile
from typing import List, Dict, Union, Optional

# Add the metrics directory to path for imports
metrics_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, metrics_dir)

# Import with handling for relative imports
try:
    from calc_code_bleu import evaluate_per_example, get_codebleu, SUPPORTED_LANGUAGES
except ImportError:
    # Handle relative import issue by importing individual modules
    import bleu
    import weighted_ngram_match  
    import syntax_match
    import dataflow_match
    
    # Import the main functions manually
    exec(open(os.path.join(metrics_dir, 'calc_code_bleu.py')).read())


def evaluate_python_code_bleu(
    reference: str, 
    generated: str, 
    weights: str = "0.25,0.25,0.25,0.25"
) -> Dict[str, float]:
    """
    Evaluate CodeBLEU score for Python code strings.
    
    Args:
        reference: Reference (ground truth) Python code as string
        generated: Generated Python code as string to evaluate
        weights: Comma-separated weights for alpha,beta,gamma,theta (default: equal weights)
        
    Returns:
        Dictionary containing detailed scores:
        - em: Exact match (1.0 if identical, 0.0 otherwise)
        - bleu: Standard BLEU score
        - wbleu: Weighted BLEU score (considering Python keywords)
        - syntax: Syntax tree match score
        - dataflow: Dataflow match score  
        - codebleu: Overall CodeBLEU score (weighted combination)
    """
    if 'python' not in SUPPORTED_LANGUAGES:
        raise ValueError("Python language not supported in current configuration")
    
    try:
        result = evaluate_per_example(reference, generated, "python", weights)
        return result
    except Exception as e:
        print(f"Error evaluating Python CodeBLEU: {e}")
        raise


def evaluate_python_files(
    reference_file: str,
    hypothesis_file: str, 
    weights: str = "0.25,0.25,0.25,0.25"
) -> float:
    """
    Evaluate CodeBLEU score for Python code from files.
    
    Args:
        reference_file: Path to file containing reference Python code
        hypothesis_file: Path to file containing generated Python code  
        weights: Comma-separated weights for alpha,beta,gamma,theta
        
    Returns:
        CodeBLEU score as float
    """
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"Reference file not found: {reference_file}")
    if not os.path.exists(hypothesis_file):
        raise FileNotFoundError(f"Hypothesis file not found: {hypothesis_file}")
    
    try:
        score = get_codebleu([reference_file], hypothesis_file, "python", weights)
        return score
    except Exception as e:
        print(f"Error evaluating Python CodeBLEU from files: {e}")
        raise


def batch_evaluate_python(
    references: List[str],
    hypotheses: List[str],
    weights: str = "0.25,0.25,0.25,0.25"
) -> List[Dict[str, float]]:
    """
    Batch evaluate CodeBLEU scores for multiple Python code pairs.
    
    Args:
        references: List of reference Python code strings
        hypotheses: List of generated Python code strings
        weights: Comma-separated weights for alpha,beta,gamma,theta
        
    Returns:
        List of score dictionaries, one for each code pair
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Mismatch in number of references ({len(references)}) and hypotheses ({len(hypotheses)})")
    
    results = []
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        try:
            score = evaluate_python_code_bleu(ref, hyp, weights)
            score['example_id'] = i
            results.append(score)
        except Exception as e:
            print(f"Error evaluating example {i}: {e}")
            # Add placeholder result with zeros
            results.append({
                'example_id': i,
                'em': 0.0, 'bleu': 0.0, 'wbleu': 0.0, 
                'syntax': 0.0, 'dataflow': 0.0, 'codebleu': 0.0,
                'error': str(e)
            })
    
    return results


def evaluate_python_files_batch(
    reference_files: List[str],
    hypothesis_files: List[str], 
    weights: str = "0.25,0.25,0.25,0.25"
) -> List[float]:
    """
    Batch evaluate CodeBLEU scores from multiple file pairs.
    
    Args:
        reference_files: List of paths to reference Python code files
        hypothesis_files: List of paths to generated Python code files
        weights: Comma-separated weights for alpha,beta,gamma,theta
        
    Returns:
        List of CodeBLEU scores
    """
    if len(reference_files) != len(hypothesis_files):
        raise ValueError(f"Mismatch in number of reference files ({len(reference_files)}) and hypothesis files ({len(hypothesis_files)})")
    
    scores = []
    for i, (ref_file, hyp_file) in enumerate(zip(reference_files, hypothesis_files)):
        try:
            score = evaluate_python_files(ref_file, hyp_file, weights)
            scores.append(score)
        except Exception as e:
            print(f"Error evaluating file pair {i} ({ref_file}, {hyp_file}): {e}")
            scores.append(0.0)
    
    return scores


def create_temp_files_and_evaluate(
    references: List[str],
    hypotheses: List[str],
    weights: str = "0.25,0.25,0.25,0.25"
) -> float:
    """
    Create temporary files and evaluate CodeBLEU score.
    Useful when you have code strings but want to use the file-based evaluation.
    
    Args:
        references: List of reference Python code strings  
        hypotheses: List of generated Python code strings
        weights: Comma-separated weights
        
    Returns:
        CodeBLEU score
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as ref_file:
        ref_file.write('\n'.join(references))
        ref_file_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as hyp_file:
        hyp_file.write('\n'.join(hypotheses))
        hyp_file_path = hyp_file.name
    
    try:
        score = evaluate_python_files(ref_file_path, hyp_file_path, weights)
        return score
    finally:
        # Clean up temporary files
        os.unlink(ref_file_path)
        os.unlink(hyp_file_path)


def print_detailed_scores(scores: Dict[str, float], title: str = "CodeBLEU Evaluation Results"):
    """
    Pretty print detailed CodeBLEU scores.
    
    Args:
        scores: Score dictionary from evaluate_python_code_bleu
        title: Title for the output
    """
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Exact Match (EM):      {scores.get('em', 0.0):.4f}")
    print(f"BLEU Score:            {scores.get('bleu', 0.0):.4f}")
    print(f"Weighted BLEU:         {scores.get('wbleu', 0.0):.4f}")
    print(f"Syntax Match:          {scores.get('syntax', 0.0):.4f}")
    print(f"Dataflow Match:        {scores.get('dataflow', 0.0):.4f}")
    print(f"CodeBLEU (Overall):    {scores.get('codebleu', 0.0):.4f}")
    if 'error' in scores:
        print(f"Error: {scores['error']}")


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Simple function comparison
    print("Example 1: Evaluating Python function similarity")
    
    reference_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    generated_code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""
    
    scores = evaluate_python_code_bleu(reference_code, generated_code)
    print_detailed_scores(scores, "Fibonacci Function Comparison")
    
    # Example 2: Batch evaluation
    print("\n\nExample 2: Batch evaluation of multiple code pairs")
    
    references = [
        "def add(a, b): return a + b",
        "def multiply(x, y): return x * y",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    ]
    
    hypotheses = [
        "def add(x, y): return x + y",  # Similar but different variable names
        "def multiply(a, b): return a * b",  # Similar but different variable names  
        "def factorial(num): return 1 if num <= 1 else num * factorial(num-1)"  # Similar with different variable name
    ]
    
    batch_scores = batch_evaluate_python(references, hypotheses)
    
    for i, score in enumerate(batch_scores):
        print(f"\nExample {i+1}:")
        print(f"  Reference:  {references[i]}")
        print(f"  Generated:  {hypotheses[i]}")
        print(f"  CodeBLEU:   {score['codebleu']:.4f}")
        print(f"  BLEU:       {score['bleu']:.4f}")
        print(f"  Syntax:     {score['syntax']:.4f}")
    
    # Example 3: Different weight configurations
    print("\n\nExample 3: Testing different weight configurations")
    
    test_ref = "def hello(): print('Hello, World!')"
    test_gen = "def hello(): print('Hello, World!')"  # Identical
    
    weight_configs = [
        ("Equal weights", "0.25,0.25,0.25,0.25"),
        ("BLEU focused", "0.7,0.1,0.1,0.1"), 
        ("Syntax focused", "0.1,0.1,0.7,0.1"),
        ("Dataflow focused", "0.1,0.1,0.1,0.7")
    ]
    
    for name, weights in weight_configs:
        score = evaluate_python_code_bleu(test_ref, test_gen, weights)
        print(f"{name:20s}: CodeBLEU = {score['codebleu']:.4f}")
    
    print("\n✅ Python CodeBLEU evaluation examples completed successfully!")