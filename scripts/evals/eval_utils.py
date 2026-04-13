#!/usr/bin/env python3
"""
Code Evaluation Utilities - README and Examples

This package provides comprehensive evaluation tools for AI-generated code,
calculating multiple metrics including Pass@1, AUROC, TPR@XF, and CodeBLEU.

AVAILABLE SCRIPTS:
==================

1. comprehensive_evaluation.py - Main evaluation script with full functionality
2. quick_eval.py - Simplified script for quick evaluations  
3. batch_eval.py - Batch evaluation for multiple models/experiments
4. eval_utils.py - This utility script with examples and documentation

USAGE EXAMPLES:
===============

# Quick evaluation of a single experiment
python quick_eval.py --experiment gemini_exp1_during_gen_v1_100_mbpp

# Full comprehensive evaluation
python comprehensive_evaluation.py \\
    --csv_file /home/fahad/Documents/MASTERS/CODEBLOCKS/Masters-Thesis-Code/promptMark/results/raw/gemini_exp1_during_gen_v1_100_mbpp.csv \\
    --generated_dir /home/fahad/Documents/MASTERS/CODEBLOCKS/Masters-Thesis-Code/promptMark/output/gemini_exp1_during_gen_v1_100_mbpp \\
    --reference_file /home/fahad/Documents/MASTERS/CODEBLOCKS/Masters-Thesis-Code/promptMark/datasets/core/sanitized-mbpp-sample-100.json

# Batch evaluation of multiple experiments
python batch_eval.py --experiments gemini_exp1_during_gen_v1_100_mbpp qwen_exp1_during_gen_v1_100_mbpp

# List available models/experiments
python batch_eval.py --list_available

METRICS CALCULATED:
===================

1. Pass@1 (Average Pass Rate): Percentage of generated code that passes all tests
2. AUROC: Area Under ROC Curve for watermark detection
3. T@X%F: True Positive Rate at X% False Positive Rate (X=0,1,5,10)
4. CodeBLEU: Code similarity metric combining BLEU, syntax, and dataflow matching

INPUT DATA STRUCTURE:
=====================

CSV File Columns:
- task_id: Unique identifier for each coding task
- pass_rate or all_passed: Test success rate
- original_z_score, generated_z_score: Watermark detection scores
- tests_passed, total_tests: Test execution details

Generated Code Directory:
- Contains .py files named by task_id (e.g., 57.py, 590.py)
- Each file contains the AI-generated solution

Reference File (JSON):
- JSONL format with task_id, code, prompt, and test cases
- Contains ground truth implementations for comparison

OUTPUT FORMAT:
==============

The scripts provide both console output and optional JSON export with:
- Detailed metric scores
- Statistical summaries (mean, std, min, max)
- Error counts and debugging information
- Formatted comparison tables (for batch evaluation)
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

def print_usage_examples():
    """Print comprehensive usage examples."""
    
    examples = [
        {
            "title": "Quick Single Experiment Evaluation",
            "command": "python quick_eval.py --experiment gemini_exp1_during_gen_v1_100_mbpp",
            "description": "Evaluate a single experiment with auto-detected file paths"
        },
        {
            "title": "Manual Path Specification", 
            "command": """python comprehensive_evaluation.py \\
    --csv_file /home/fahad/Documents/MASTERS/CODEBLOCKS/Masters-Thesis-Code/promptMark/results/raw/gemini_exp1_during_gen_v1_100_mbpp.csv \\
    --generated_dir /home/fahad/Documents/MASTERS/CODEBLOCKS/Masters-Thesis-Code/promptMark/output/gemini_exp1_during_gen_v1_100_mbpp \\
    --reference_file /home/fahad/Documents/MASTERS/CODEBLOCKS/Masters-Thesis-Code/promptMark/datasets/core/sanitized-mbpp-sample-100.json \\
    --output results.json""",
            "description": "Full evaluation with manual paths and JSON output"
        },
        {
            "title": "Batch Evaluation",
            "command": "python batch_eval.py --experiments gemini_exp1_during_gen_v1_100_mbpp qwen_exp1_during_gen_v1_100_mbpp",
            "description": "Compare multiple experiments"
        },
        {
            "title": "CodeBLEU Sample Size Control",
            "command": "python quick_eval.py --model codegemma-7b-it --experiment exp1_dgen_v1_100 --codebleu_sample_size 50",
            "description": "Control number of examples for CodeBLEU (for speed)"
        },
        {
            "title": "Full CodeBLEU Evaluation",
            "command": "python quick_eval.py --model codegemma-7b-it --experiment exp1_dgen_v1_100 --full_codebleu",
            "description": "Evaluate CodeBLEU on all examples (slower but complete)"
        }
    ]
    
    print("🚀 CODE EVALUATION TOOLKIT - USAGE EXAMPLES")
    print("=" * 60)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print("-" * len(example['title']))
        print(f"Command:")
        print(f"  {example['command']}")
        print(f"Description: {example['description']}")

def validate_data_structure(csv_file: str, generated_dir: str, reference_file: str) -> Dict[str, List[str]]:
    """Validate data structure and return issues found."""
    
    issues = {
        "errors": [],
        "warnings": [],
        "info": []
    }
    
    # Check CSV file
    if not os.path.exists(csv_file):
        issues["errors"].append(f"CSV file not found: {csv_file}")
    else:
        import pandas as pd
        try:
            df = pd.read_csv(csv_file)
            required_cols = ["task_id"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                issues["errors"].append(f"Missing required CSV columns: {missing_cols}")
            
            # Check for optional columns
            optional_cols = ["pass_rate", "all_passed", "original_z_score", "generated_z_score"]
            available_optional = [col for col in optional_cols if col in df.columns]
            issues["info"].append(f"Available optional columns: {available_optional}")
            
            issues["info"].append(f"CSV contains {len(df)} examples")
            
        except Exception as e:
            issues["errors"].append(f"Error reading CSV: {e}")
    
    # Check generated code directory
    if not os.path.exists(generated_dir):
        issues["errors"].append(f"Generated code directory not found: {generated_dir}")
    else:
        py_files = [f for f in os.listdir(generated_dir) if f.endswith('.py')]
        issues["info"].append(f"Found {len(py_files)} Python files in generated directory")
        
        if len(py_files) == 0:
            issues["warnings"].append("No Python files found in generated directory")
    
    # Check reference file
    if not os.path.exists(reference_file):
        issues["errors"].append(f"Reference file not found: {reference_file}")
    else:
        try:
            with open(reference_file, 'r') as f:
                references = [json.loads(line) for line in f]
            issues["info"].append(f"Reference file contains {len(references)} examples")
            
            # Check structure
            if references:
                sample = references[0]
                required_keys = ["task_id", "code"]
                missing_keys = [key for key in required_keys if key not in sample]
                if missing_keys:
                    issues["warnings"].append(f"Reference entries missing keys: {missing_keys}")
                    
        except Exception as e:
            issues["errors"].append(f"Error reading reference file: {e}")
    
    return issues

def main():
    """Main function for the utility script."""
    
    parser = argparse.ArgumentParser(description='Code Evaluation Utilities')
    parser.add_argument('--examples', action='store_true', help='Show usage examples')
    parser.add_argument('--validate', action='store_true', help='Validate data structure')
    parser.add_argument('--csv_file', type=str, help='CSV file to validate')
    parser.add_argument('--generated_dir', type=str, help='Generated code directory to validate')
    parser.add_argument('--reference_file', type=str, help='Reference file to validate')
    
    args = parser.parse_args()
    
    if args.examples:
        print_usage_examples()
        return 0
    
    if args.validate:
        if not all([args.csv_file, args.generated_dir, args.reference_file]):
            print("❌ For validation, specify --csv_file, --generated_dir, and --reference_file")
            return 1
        
        print("🔍 Validating data structure...")
        issues = validate_data_structure(args.csv_file, args.generated_dir, args.reference_file)
        
        # Print results
        if issues["errors"]:
            print("\n❌ ERRORS FOUND:")
            for error in issues["errors"]:
                print(f"  • {error}")
        
        if issues["warnings"]:
            print("\n⚠️  WARNINGS:")
            for warning in issues["warnings"]:
                print(f"  • {warning}")
        
        if issues["info"]:
            print("\n📋 INFORMATION:")
            for info in issues["info"]:
                print(f"  • {info}")
        
        if not issues["errors"]:
            print("\n✅ Data structure validation passed!")
        else:
            print(f"\n❌ Found {len(issues['errors'])} errors that need to be fixed")
        
        return 0 if not issues["errors"] else 1
    
    # Default: show help and available options
    print("📚 CODE EVALUATION TOOLKIT")
    print("=" * 40)
    print("\nAvailable scripts:")
    print("  • comprehensive_evaluation.py - Full evaluation")
    print("  • quick_eval.py - Quick evaluation") 
    print("  • batch_eval.py - Batch evaluation")
    print("  • eval_utils.py - This utility script")
    
    print("\nOptions:")
    print("  --examples     Show usage examples")
    print("  --validate     Validate data structure")
    print("  --help         Show this help")
    
    print("\nQuick start:")
    print("  python eval_utils.py --examples")
    print("  python batch_eval.py --list_available")
    print("  python quick_eval.py --experiment EXPERIMENT")
    
    return 0

if __name__ == "__main__":
    exit(main())