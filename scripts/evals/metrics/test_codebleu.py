#!/usr/bin/env python3
"""
Simple test script for the enhanced CodeBLEU implementation.
This script tests Python code evaluation without relative import issues.
"""

import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Direct imports
import bleu
import weighted_ngram_match
import syntax_match 
import dataflow_match

# Test the language configuration
LANGUAGE_CONFIG = {
    'python': {
        'keyword_file': 'python.txt',
        'tree_sitter_name': 'python',
        'comment_removal_lang': 'python',
        'wrapper_needed': False,
        'wrapper_template': None
    },
    'java': {
        'keyword_file': 'java.txt',
        'tree_sitter_name': 'java',
        'comment_removal_lang': 'java',
        'wrapper_needed': True,
        'wrapper_template': 'public class Wrapper {{\n{code}\n}}'
    },
    'cpp': {
        'keyword_file': 'cpp.txt',
        'tree_sitter_name': 'cpp', 
        'comment_removal_lang': 'cpp',
        'wrapper_needed': False,
        'wrapper_template': None
    }
}

SUPPORTED_LANGUAGES = list(LANGUAGE_CONFIG.keys())

def get_language_config(lang: str):
    """Get language configuration for the specified language."""
    if lang not in LANGUAGE_CONFIG:
        raise ValueError(f"Unsupported language: {lang}. Supported languages: {SUPPORTED_LANGUAGES}")
    return LANGUAGE_CONFIG[lang]

def load_keywords(lang: str):
    """Load keywords for the specified language."""
    config = get_language_config(lang)
    keyword_file_path = os.path.join(current_dir, "keywords", config['keyword_file'])
    
    if not os.path.exists(keyword_file_path):
        print(f"Warning: Keyword file not found for {lang} at {keyword_file_path}")
        return []
    
    try:
        with open(keyword_file_path, "r", encoding="utf-8") as f:
            keywords = [x.strip() for x in f.readlines() if x.strip()]
        return keywords
    except Exception as e:
        print(f"Warning: Failed to load keywords for {lang}: {e}")
        return []

def simple_evaluate_python_bleu(reference: str, hypothesis: str):
    """Simple Python CodeBLEU evaluation without complex dependencies."""
    
    print("Testing Python CodeBLEU evaluation...")
    
    # Basic tokenization
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    # Load Python keywords
    keywords = load_keywords('python')
    print(f"Loaded {len(keywords)} Python keywords")
    
    # Simple BLEU calculation
    tokenized_refs = [[ref_tokens]]
    tokenized_hyps = [hyp_tokens]
    
    try:
        bleu_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)
        print(f"BLEU Score: {bleu_score:.4f}")
    except Exception as e:
        print(f"BLEU calculation failed: {e}")
        bleu_score = 0.0
    
    # Weighted BLEU with keywords
    def make_weights(tokens, keyword_list):
        return {token: 1 if token in keyword_list else 0.2 for token in tokens}
    
    try:
        tokenized_refs_with_weights = [
            [[ref_tokens, make_weights(ref_tokens, keywords)]]
        ]
        weighted_bleu_score = weighted_ngram_match.corpus_bleu(
            tokenized_refs_with_weights, tokenized_hyps
        )
        print(f"Weighted BLEU Score: {weighted_bleu_score:.4f}")
    except Exception as e:
        print(f"Weighted BLEU calculation failed: {e}")
        weighted_bleu_score = 0.0
    
    # Check exact match
    exact_match = 1.0 if reference.strip() == hypothesis.strip() else 0.0
    print(f"Exact Match: {exact_match:.4f}")
    
    return {
        'bleu': bleu_score,
        'weighted_bleu': weighted_bleu_score,
        'exact_match': exact_match
    }

def test_language_support():
    """Test language support configuration."""
    print("\n=== Testing Language Support ===")
    
    for lang in SUPPORTED_LANGUAGES:
        try:
            config = get_language_config(lang)
            keywords = load_keywords(lang)
            print(f"✅ {lang:10s}: {len(keywords):3d} keywords, wrapper={config['wrapper_needed']}")
        except Exception as e:
            print(f"❌ {lang:10s}: Error - {e}")

def test_python_examples():
    """Test with Python code examples."""
    print("\n=== Testing Python Code Examples ===")
    
    # Example 1: Identical code
    print("\nExample 1: Identical functions")
    ref1 = "def add(a, b): return a + b"
    hyp1 = "def add(a, b): return a + b"
    scores1 = simple_evaluate_python_bleu(ref1, hyp1)
    
    # Example 2: Similar code with different variable names
    print("\nExample 2: Different variable names")
    ref2 = "def add(a, b): return a + b"
    hyp2 = "def add(x, y): return x + y"  
    scores2 = simple_evaluate_python_bleu(ref2, hyp2)
    
    # Example 3: Different logic
    print("\nExample 3: Different logic")
    ref3 = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    hyp3 = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
    scores3 = simple_evaluate_python_bleu(ref3, hyp3)
    
    return [scores1, scores2, scores3]

if __name__ == "__main__":
    print("🚀 Testing Enhanced CodeBLEU Implementation")
    print("=" * 50)
    
    # Test language support
    test_language_support()
    
    # Test Python examples
    test_python_examples()
    
    print("\n✅ Testing completed!")
    print("\nTo use the full implementation:")
    print("1. For command line: python calc_code_bleu.py --refs ref.py --hyp gen.py --lang python")
    print("2. For Python API: Use the functions in calc_code_bleu.py")
    print("3. Check README.md for detailed usage examples")