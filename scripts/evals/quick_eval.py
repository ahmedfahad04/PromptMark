#!/usr/bin/env python3
"""
Quick Evaluation Script

A simplified version of the comprehensive evaluation for quick testing and batch evaluation.

Usage:
    python quick_eval.py --model codegemma-7b-it --experiment exp1_dgen_v1_100
    python quick_eval.py --csv_file path/to/file.csv --generated_dir path/to/generated --reference_file path/to/ref_dataset.json
"""

import os
import sys
import argparse
import glob
from comprehensive_evaluation import ComprehensiveEvaluator

def list_available_experiments():
    """List available experiments by scanning results/raw and output directories."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    
    results_dir = f"{base_dir}/results/raw"
    output_dir = f"{base_dir}/output"
    
    # Find CSV files in results/raw
    csv_files = glob.glob(f"{results_dir}/*.csv")
    experiments = []
    
    for csv_file in csv_files:
        basename = os.path.basename(csv_file).replace('.csv', '')
        # Check if corresponding output directory exists
        output_path = f"{output_dir}/{basename}"
        if os.path.exists(output_path):
            experiments.append(basename)
    
    return sorted(experiments)

def get_default_paths(experiment: str):
    """Get default file paths based on experiment name."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to project root
    
    # Parse experiment name: {model}_{exp}_{gen_type}_{version}_{size}_{dataset}
    # e.g., claude_expA_during_gen_v1_2_humaneval
    parts = experiment.split('_')
    if len(parts) < 6:
        raise ValueError(f"Invalid experiment name format: {experiment}. Expected format: model_exp_gen_type_version_size_dataset")
    
    model = parts[0]
    exp = parts[1]
    gen_type = parts[2]
    version = parts[4] if len(parts) > 4 else 'v1'
    size = parts[5] if len(parts) > 5 else '100'
    dataset = parts[6] if len(parts) > 6 else 'humaneval'
    
    # CSV file
    csv_file = f"{base_dir}/results/raw/{experiment}.csv"
    
    # Generated code directory
    generated_dir = f"{base_dir}/output/{experiment}"
    
    # Reference file - auto-detect based on dataset
    if dataset == 'humaneval':
        reference_file = f"{base_dir}/datasets/human_eval_164.jsonl"
    elif dataset == 'mbpp':
        reference_file = f"{base_dir}/datasets/sanitized-mbpp.json"
    else:
        reference_file = f"{base_dir}/datasets/{dataset}.jsonl"  # fallback
    
    return csv_file, generated_dir, reference_file

def main():
    parser = argparse.ArgumentParser(description='Quick Code Evaluation Script')
    
    # Option 1: Use experiment name (auto-detect paths)
    parser.add_argument('--experiment', type=str,
                       help='Experiment name (e.g., claude_expA_during_gen_v1_2_humaneval)')
    
    # List available experiments
    parser.add_argument('--list_available', action='store_true',
                       help='List available experiments and exit')
    
    # Auto evaluate first available
    parser.add_argument('--auto_evaluate', action='store_true',
                       help='Automatically evaluate the first available experiment')
    
    # Option 2: Specify paths manually
    parser.add_argument('--csv_file', type=str,
                       help='Path to CSV file with evaluation results')
    parser.add_argument('--generated_dir', type=str,
                       help='Directory containing AI-generated code files')
    parser.add_argument('--reference_file', type=str,
                       help='JSON file containing reference code implementations')
    
    # Evaluation options
    parser.add_argument('--codebleu_sample_size', type=int, default=20,
                       help='Number of examples for CodeBLEU evaluation (default: 20)')
    parser.add_argument('--full_codebleu', action='store_true',
                       help='Evaluate CodeBLEU on all examples (slower)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results JSON')
    
    args = parser.parse_args()
    
    # Handle list available
    if args.list_available:
        experiments = list_available_experiments()
        if not experiments:
            print("No available experiments found.")
            return 0
        print("Available experiments:")
        for exp in experiments:
            print(f"  - {exp}")
        return 0
    
    # Handle auto evaluate
    if args.auto_evaluate:
        experiments = list_available_experiments()
        if not experiments:
            print("No available experiments found for auto-evaluation.")
            return 1
        args.experiment = experiments[0]
        print(f"🔄 Auto-selected experiment: {args.experiment}")
    
    # Determine file paths
    if args.experiment:
        csv_file, generated_dir, reference_file = get_default_paths(args.experiment)
        print(f"🔍 Auto-detected paths for {args.experiment}:")
        print(f"   CSV: {csv_file}")
        print(f"   Generated: {generated_dir}")
        print(f"   Reference: {reference_file}")
    elif args.csv_file and args.generated_dir and args.reference_file:
        csv_file = args.csv_file
        generated_dir = args.generated_dir
        reference_file = args.reference_file
    else:
        print("❌ Error: Specify either (--experiment) OR (--csv_file, --generated_dir, --reference_file)")
        return 1
    
    # Validate paths
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return 1
    
    if not os.path.exists(generated_dir):
        print(f"❌ Generated code directory not found: {generated_dir}")
        return 1
    
    if not os.path.exists(reference_file):
        print(f"❌ Reference file not found: {reference_file}")
        return 1
    
    # Set CodeBLEU sample size
    codebleu_sample_size = None if args.full_codebleu else args.codebleu_sample_size
    
    try:
        # Run evaluation
        evaluator = ComprehensiveEvaluator(csv_file, generated_dir, reference_file)
        results = evaluator.run_comprehensive_evaluation(codebleu_sample_size)
        evaluator.print_summary_report(results)
        
        # Save results if requested
        if args.output:
            import json
            import numpy as np
            
            # Convert numpy types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            json_results[key][k] = v.tolist()
                        elif isinstance(v, (np.integer, np.floating)):
                            json_results[key][k] = float(v)
                        else:
                            json_results[key][k] = v
                elif isinstance(value, (np.integer, np.floating)):
                    json_results[key] = float(value)
                else:
                    json_results[key] = value
            
            with open(args.output, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())