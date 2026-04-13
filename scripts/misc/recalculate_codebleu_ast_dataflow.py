#!/usr/bin/env python3
"""
Recalculate CodeBLEU scores with modified weights:
- BLEU weight: 0
- Weighted BLEU weight: 0
- AST/Syntax weight: 0.5
- Dataflow weight: 0.5

This script processes all experiment folders and creates new CSV files with the recalculated scores.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd

# Add metrics directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts/evals/metrics'))

# Import the CodeBLEU calculation functions directly
sys.path.insert(0, 'scripts/evals')
from calculate_mbpp_codebleu import load_mbpp_dataset, save_results_to_csv

# Import the evaluate function with custom weights
from metrics.calc_code_bleu import evaluate_per_example

# Configuration
EXPERIMENT_BASE_DIR = "output/experiment_results"
MBPP_DATASET_JSON = "dataset/sanitized-mbpp.json"
HUMANEVAL_DATASET_PARQUET = "dataset/human_eval_164.parquet"
OUTPUT_BASE_DIR = "results/experiment_codebleu_ast_dataflow"

# New weights: BLEU=0, wBLEU=0, AST=0.5, Dataflow=0.5
CUSTOM_WEIGHTS = "0,0,0.5,0.5"

# Create output directory
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def load_humaneval_dataset(dataset_path: str) -> dict:
    """
    Load HumanEval dataset from parquet file.
    
    Args:
        dataset_path: Path to the HumanEval parquet file
        
    Returns:
        Dictionary mapping task_id (numeric) to reference code
    """
    df = pd.read_parquet(dataset_path)
    task_references = {}
    
    for _, row in df.iterrows():
        # Extract numeric ID from task_id (e.g., "HumanEval/0" -> 0)
        task_id_str = row['task_id']
        if '/' in task_id_str:
            task_id = int(task_id_str.split('/')[-1])
        else:
            task_id = int(task_id_str)
        
        task_references[task_id] = {
            'code': row['canonical_solution'],
            'task_id': task_id_str,
            'entry_point': row['entry_point']
        }
    
    return task_references


def calculate_codebleu_custom_weights(
    sample_dir: str,
    task_references: dict,
    output_csv: str = None,
    weights: str = CUSTOM_WEIGHTS
) -> list:
    """
    Calculate CodeBLEU with custom weights for each sample code against its reference.
    
    Args:
        sample_dir: Directory containing sample code files
        task_references: Dictionary mapping task_id to reference code
        output_csv: Optional path to save results as CSV
        weights: Custom weights as comma-separated string (alpha,beta,gamma,theta)
        
    Returns:
        List of result dictionaries
    """
    results = []
    sample_path = Path(sample_dir)
    
    # Get all Python files in sample directory
    sample_files = sorted(sample_path.glob('*.py'))
    
    if not sample_files:
        return results
    
    for sample_file in sample_files:
        try:
            # Extract task_id from filename (e.g., "111.py" -> 111)
            task_id = int(sample_file.stem)
            
            # Get reference code
            if task_id not in task_references:
                continue
            
            reference_code = task_references[task_id]['code']
            
            # Read sample code
            with open(sample_file, 'r', encoding='utf-8') as f:
                sample_code = f.read()
            
            # Calculate CodeBLEU with custom weights
            scores = evaluate_per_example(
                reference=reference_code,
                hypothesis=sample_code,
                lang='python',
                params=weights
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
            
        except Exception as e:
            print(f"❌ Error processing {sample_file.name}: {e}")
            continue
    
    # Save to CSV if requested
    if output_csv and results:
        save_results_to_csv(results, output_csv)
    
    return results


# Load dataset once
print("=" * 80)
print("🔬 RECALCULATING CODEBLEU WITH AST + DATAFLOW WEIGHTS")
print("=" * 80)
print(f"\n📊 Weight Configuration:")
print(f"   BLEU (alpha):          0.0")
print(f"   Weighted BLEU (beta):  0.0")
print(f"   AST/Syntax (gamma):    0.5")
print(f"   Dataflow (theta):      0.5")
print("=" * 80)

print("\n📂 Loading datasets...")
mbpp_task_references = load_mbpp_dataset(MBPP_DATASET_JSON)
print(f"✅ Loaded {len(mbpp_task_references)} MBPP reference implementations")

humaneval_task_references = load_humaneval_dataset(HUMANEVAL_DATASET_PARQUET)
print(f"✅ Loaded {len(humaneval_task_references)} HumanEval reference implementations\n")

# Get all experiment directories (excluding zip files)
experiment_dirs = []
for item in os.listdir(EXPERIMENT_BASE_DIR):
    item_path = os.path.join(EXPERIMENT_BASE_DIR, item)
    if os.path.isdir(item_path) and not item.endswith('.zip'):
        experiment_dirs.append(item)

experiment_dirs.sort()

print(f"Found {len(experiment_dirs)} experiment directories:")
for exp_dir in experiment_dirs:
    print(f"  - {exp_dir}")

print(f"\n💾 Output directory: {OUTPUT_BASE_DIR}")
print("=" * 80)

# Process each experiment directory
results_summary = []
total_processed = 0

for idx, exp_dir in enumerate(experiment_dirs, 1):
    exp_path = os.path.join(EXPERIMENT_BASE_DIR, exp_dir)
    
    print(f"\n[{idx}/{len(experiment_dirs)}] Processing: {exp_dir}")
    print("-" * 80)
    
    # Check if this directory contains .py files directly (flat structure like HumanEval)
    py_files = list(Path(exp_path).glob('*.py'))
    
    if py_files:
        # Flat structure: this directory itself contains the code files
        print(f"   Detected flat structure (contains {len(py_files)} .py files)")
        
        # Detect if this is HumanEval or MBPP based on directory name
        is_humaneval = 'humaneval' in exp_dir.lower()
        dataset_type = "HumanEval" if is_humaneval else "MBPP"
        task_references = humaneval_task_references if is_humaneval else mbpp_task_references
        
        output_csv = os.path.join(OUTPUT_BASE_DIR, f"{exp_dir}_codebleu_ast_dataflow.csv")
        
        print(f"   [1/1] {exp_dir[:40]:40s} [{dataset_type:9s}]", end=" ")
        
        try:
            # Calculate CodeBLEU with custom weights
            results = calculate_codebleu_custom_weights(
                sample_dir=exp_path,
                task_references=task_references,
                output_csv=output_csv,
                weights=CUSTOM_WEIGHTS
            )
            
            if results:
                avg_codebleu = sum(r['codebleu'] for r in results) / len(results)
                avg_syntax = sum(r['syntax'] for r in results) / len(results)
                avg_dataflow = sum(r['dataflow'] for r in results) / len(results)
                
                print(f"✅ CB:{avg_codebleu:.4f} AST:{avg_syntax:.4f} DF:{avg_dataflow:.4f} ({len(results)} samples)")
                
                results_summary.append({
                    'experiment': exp_dir,
                    'model': exp_dir,
                    'dataset': dataset_type,
                    'avg_codebleu': avg_codebleu,
                    'avg_syntax': avg_syntax,
                    'avg_dataflow': avg_dataflow,
                    'total_samples': len(results),
                    'csv_file': output_csv
                })
                total_processed += 1
            else:
                print(f"⚠️  No results")
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
    else:
        # Nested structure: look for subdirectories (model folders)
        model_dirs = []
        for item in os.listdir(exp_path):
            item_path = os.path.join(exp_path, item)
            if os.path.isdir(item_path):
                model_dirs.append(item)
        
        model_dirs.sort()
        
        if not model_dirs:
            print(f"⚠️  No model directories found in {exp_dir}, skipping...")
            continue
        
        print(f"   Found {len(model_dirs)} model directories")
        
        # Process each model directory
        for model_idx, model_dir in enumerate(model_dirs, 1):
            sample_dir = os.path.join(exp_path, model_dir)
            
            # Detect if this is HumanEval or MBPP based on directory name
            is_humaneval = 'humaneval' in model_dir.lower() or 'humaneval' in exp_dir.lower()
            dataset_type = "HumanEval" if is_humaneval else "MBPP"
            task_references = humaneval_task_references if is_humaneval else mbpp_task_references
            
            output_csv = os.path.join(OUTPUT_BASE_DIR, f"{exp_dir}_{model_dir}_codebleu_ast_dataflow.csv")
            
            print(f"   [{model_idx}/{len(model_dirs)}] {model_dir[:40]:40s} [{dataset_type:9s}]", end=" ")
            
            try:
                # Calculate CodeBLEU with custom weights
                results = calculate_codebleu_custom_weights(
                    sample_dir=sample_dir,
                    task_references=task_references,
                    output_csv=output_csv,
                    weights=CUSTOM_WEIGHTS
                )
                
                if results:
                    avg_codebleu = sum(r['codebleu'] for r in results) / len(results)
                    avg_syntax = sum(r['syntax'] for r in results) / len(results)
                    avg_dataflow = sum(r['dataflow'] for r in results) / len(results)
                    
                    print(f"✅ CB:{avg_codebleu:.4f} AST:{avg_syntax:.4f} DF:{avg_dataflow:.4f} ({len(results)} samples)")
                    
                    results_summary.append({
                        'experiment': exp_dir,
                        'model': model_dir,
                        'dataset': dataset_type,
                        'avg_codebleu': avg_codebleu,
                        'avg_syntax': avg_syntax,
                        'avg_dataflow': avg_dataflow,
                        'total_samples': len(results),
                        'csv_file': output_csv
                    })
                    total_processed += 1
                else:
                    print(f"⚠️  No results")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")

print("\n" + "=" * 80)
print("📊 SUMMARY OF ALL EVALUATIONS")
print("=" * 80)

if results_summary:
    # Save summary to JSON
    summary_json = os.path.join(OUTPUT_BASE_DIR, "all_experiments_summary_ast_dataflow.json")
    with open(summary_json, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Create summary CSV
    import csv
    summary_csv = os.path.join(OUTPUT_BASE_DIR, "all_experiments_summary_ast_dataflow.csv")
    with open(summary_csv, 'w', newline='') as f:
        fieldnames = ['experiment', 'model', 'dataset', 'avg_codebleu', 'avg_syntax', 'avg_dataflow', 'total_samples', 'csv_file']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_summary)
    
    print(f"\n✅ Successfully processed {total_processed} model evaluations")
    
    # Group by experiment
    print(f"\n📊 Results by Experiment:")
    experiments = {}
    for result in results_summary:
        exp = result['experiment']
        if exp not in experiments:
            experiments[exp] = []
        experiments[exp].append(result)
    
    for exp_name in sorted(experiments.keys()):
        exp_results = experiments[exp_name]
        avg_score = sum(r['avg_codebleu'] for r in exp_results) / len(exp_results)
        avg_syntax = sum(r['avg_syntax'] for r in exp_results) / len(exp_results)
        avg_dataflow = sum(r['avg_dataflow'] for r in exp_results) / len(exp_results)
        
        # Count datasets
        datasets = {}
        for r in exp_results:
            ds = r['dataset']
            datasets[ds] = datasets.get(ds, 0) + 1
        dataset_str = ", ".join([f"{k}: {v}" for k, v in datasets.items()])
        
        print(f"\n   {exp_name}:")
        print(f"      Models: {len(exp_results)} ({dataset_str})")
        print(f"      Avg CodeBLEU: {avg_score:.4f} (AST: {avg_syntax:.4f}, Dataflow: {avg_dataflow:.4f})")
        for r in sorted(exp_results, key=lambda x: x['avg_codebleu'], reverse=True)[:3]:
            print(f"         - [{r['dataset']:9s}] {r['model'][:35]:35s}: {r['avg_codebleu']:.4f}")
    
    print(f"\n🏆 Top 10 Models Overall (by AST+Dataflow CodeBLEU):")
    sorted_results = sorted(results_summary, key=lambda x: x['avg_codebleu'], reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"   {i:2d}. [{result['dataset']:9s}] {result['model'][:35]:35s} {result['avg_codebleu']:.4f} "
              f"(AST:{result['avg_syntax']:.3f} DF:{result['avg_dataflow']:.3f}) - {result['experiment']}")
    
    print(f"\n💾 Summary files saved:")
    print(f"   - {summary_json}")
    print(f"   - {summary_csv}")
    print(f"\n📁 Individual CSV files: {total_processed} files in {OUTPUT_BASE_DIR}/")
else:
    print("\n⚠️  No results to summarize")

print("=" * 80)
print("✨ Batch evaluation complete!")
print("=" * 80)
