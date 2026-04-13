#!/usr/bin/env python3
"""
Calculate CodeBLEU scores for MBPP sample codes against reference implementations.

Usage:
    python calculate_mbpp_codebleu.py --dataset-json datasets/core/sanitized-mbpp.json \
                                      --sample-dir datasets/core/sanitized-mbpp-sample-100-codes \
                                      --output results/codebleu_scores.csv
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add metrics directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metrics'))

from metrics.calc_code_bleu import evaluate_per_example

def load_mbpp_dataset(json_path: str) -> Dict[int, Dict]:
    """Load MBPP dataset and create mapping from task_id to reference code."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mapping: task_id -> reference code
    task_references = {}
    for item in data:
        task_id = item['task_id']
        task_references[task_id] = {
            'code': item['code'],
            'prompt': item['prompt'],
            'task_id': task_id
        }
    
    return task_references

def calculate_codebleu_for_samples(
    sample_dir: str,
    task_references: Dict[int, Dict],
    output_csv: Optional[str] = None
) -> List[Dict]:
    """
    Calculate CodeBLEU for each sample code against its reference.
    
    Args:
        sample_dir: Directory containing sample code files
        task_references: Dictionary mapping task_id to reference code
        output_csv: Optional path to save results as CSV
        
    Returns:
        List of result dictionaries
    """
    results = []
    sample_path = Path(sample_dir)
    
    # Get all Python files in sample directory
    sample_files = sorted(sample_path.glob('*.py'))
    
    print(f"Found {len(sample_files)} sample files")
    print(f"Processing CodeBLEU calculations...")
    print("-" * 80)
    
    for idx, sample_file in enumerate(sample_files, 1):
        try:
            # Extract task_id from filename (e.g., "111.py" -> 111)
            task_id = int(sample_file.stem)
            
            # Get reference code
            if task_id not in task_references:
                print(f"⚠️  Task {task_id} not found in reference dataset")
                continue
            
            reference_code = task_references[task_id]['code']
            
            # Read sample code
            with open(sample_file, 'r', encoding='utf-8') as f:
                sample_code = f.read()
            
            # Calculate CodeBLEU
            scores = evaluate_per_example(
                reference=reference_code,
                hypothesis=sample_code,
                lang='python',
                params='0.25,0.25,0.25,0.25'
            )
            
            # Store result
            result = {
                'task_id': task_id,
                'filename': sample_file.name,
                'em': scores['em'],
                'bleu': scores['bleu'],
                'wbleu': scores['wbleu'],
                'syntax': scores['syntax'],
                'dataflow': scores['dataflow'],
                'codebleu': scores['codebleu']
            }
            results.append(result)
            
            # Print progress
            print(f"[{idx}/{len(sample_files)}] Task {task_id}: CodeBLEU = {scores['codebleu']:.4f}")
            
        except Exception as e:
            print(f"❌ Error processing {sample_file.name}: {e}")
            continue
    
    print("-" * 80)
    
    # Calculate statistics
    if results:
        codebleu_scores = [r['codebleu'] for r in results]
        bleu_scores = [r['bleu'] for r in results]
        syntax_scores = [r['syntax'] for r in results]
        dataflow_scores = [r['dataflow'] for r in results]
        
        print(f"\n📊 Statistics:")
        print(f"  Total evaluated: {len(results)}")
        print(f"  CodeBLEU  - Mean: {sum(codebleu_scores)/len(codebleu_scores):.4f}, "
              f"Min: {min(codebleu_scores):.4f}, Max: {max(codebleu_scores):.4f}")
        print(f"  BLEU      - Mean: {sum(bleu_scores)/len(bleu_scores):.4f}, "
              f"Min: {min(bleu_scores):.4f}, Max: {max(bleu_scores):.4f}")
        print(f"  Syntax    - Mean: {sum(syntax_scores)/len(syntax_scores):.4f}, "
              f"Min: {min(syntax_scores):.4f}, Max: {max(syntax_scores):.4f}")
        print(f"  Dataflow  - Mean: {sum(dataflow_scores)/len(dataflow_scores):.4f}, "
              f"Min: {min(dataflow_scores):.4f}, Max: {max(dataflow_scores):.4f}")
    
    # Save to CSV if requested
    if output_csv and results:
        save_results_to_csv(results, output_csv)
    
    return results

def save_results_to_csv(results: List[Dict], output_path: str):
    """Save results to CSV file."""
    import csv
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['task_id', 'filename', 'em', 'bleu', 'wbleu', 'syntax', 'dataflow', 'codebleu']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Calculate CodeBLEU scores for MBPP sample codes'
    )
    parser.add_argument(
        '--dataset-json',
        type=str,
        default='datasets/core/sanitized-mbpp.json',
        help='Path to MBPP JSON dataset'
    )
    parser.add_argument(
        '--sample-dir',
        type=str,
        default='datasets/core/sanitized-mbpp-sample-100-codes',
        help='Directory containing sample code files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/codebleu_mbpp_sample_100.csv',
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.dataset_json):
        print(f"❌ Dataset JSON not found: {args.dataset_json}")
        sys.exit(1)
    
    if not os.path.exists(args.sample_dir):
        print(f"❌ Sample directory not found: {args.sample_dir}")
        sys.exit(1)
    
    print(f"📂 Loading dataset from: {args.dataset_json}")
    task_references = load_mbpp_dataset(args.dataset_json)
    print(f"✅ Loaded {len(task_references)} reference implementations")
    
    print(f"\n📁 Processing samples from: {args.sample_dir}")
    results = calculate_codebleu_for_samples(
        sample_dir=args.sample_dir,
        task_references=task_references,
        output_csv=args.output
    )
    
    print(f"\n✨ CodeBLEU calculation complete!")

if __name__ == '__main__':
    main()
