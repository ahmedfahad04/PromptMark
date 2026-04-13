#!/usr/bin/env python3
"""
Batch process all experiment folders and calculate CodeBLEU scores.
Creates a separate CSV file for each experiment folder.
"""

import os
import sys
import json
from pathlib import Path

# Add metrics directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts/evals/metrics'))

# Import the CodeBLEU calculation functions directly
sys.path.insert(0, 'scripts/evals')
from calculate_mbpp_codebleu import load_mbpp_dataset, calculate_codebleu_for_samples

# Configuration
EXPERIMENT_BASE_DIR = "output/experiment_results"
DATASET_JSON = "dataset/sanitized-mbpp.json"
OUTPUT_BASE_DIR = "results/experiment_codebleu"

# Create output directory
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Load dataset once
print("📂 Loading MBPP dataset...")
task_references = load_mbpp_dataset(DATASET_JSON)
print(f"✅ Loaded {len(task_references)} reference implementations\n")

# Get all experiment directories (excluding zip files)
experiment_dirs = []
for item in os.listdir(EXPERIMENT_BASE_DIR):
    item_path = os.path.join(EXPERIMENT_BASE_DIR, item)
    if os.path.isdir(item_path) and not item.endswith('.zip'):
        experiment_dirs.append(item)

experiment_dirs.sort()

print("=" * 80)
print("🔬 BATCH CODEBLEU EVALUATION FOR ALL EXPERIMENTS")
print("=" * 80)
print(f"\nFound {len(experiment_dirs)} experiment directories:")
for exp_dir in experiment_dirs:
    print(f"  - {exp_dir}")

print(f"\n Output directory: {OUTPUT_BASE_DIR}")
print("=" * 80)

# Process each experiment directory
results_summary = []
total_processed = 0

for idx, exp_dir in enumerate(experiment_dirs, 1):
    exp_path = os.path.join(EXPERIMENT_BASE_DIR, exp_dir)
    
    print(f"\n[{idx}/{len(experiment_dirs)}] Processing: {exp_dir}")
    print("-" * 80)
    
    # Get all subdirectories (model folders) in this experiment
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
        output_csv = os.path.join(OUTPUT_BASE_DIR, f"{exp_dir}_{model_dir}_codebleu.csv")
        
        print(f"   [{model_idx}/{len(model_dirs)}] {model_dir[:50]:50s}", end=" ")
        
        try:
            # Calculate CodeBLEU directly
            results = calculate_codebleu_for_samples(
                sample_dir=sample_dir,
                task_references=task_references,
                output_csv=output_csv
            )
            
            if results:
                avg_codebleu = sum(r['codebleu'] for r in results) / len(results)
                print(f"✅ {avg_codebleu:.4f} ({len(results)} samples)")
                
                results_summary.append({
                    'experiment': exp_dir,
                    'model': model_dir,
                    'avg_codebleu': avg_codebleu,
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
    summary_json = os.path.join(OUTPUT_BASE_DIR, "all_experiments_summary.json")
    with open(summary_json, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Create summary CSV
    import csv
    summary_csv = os.path.join(OUTPUT_BASE_DIR, "all_experiments_summary.csv")
    with open(summary_csv, 'w', newline='') as f:
        fieldnames = ['experiment', 'model', 'avg_codebleu', 'total_samples', 'csv_file']
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
        print(f"\n   {exp_name}:")
        print(f"      Models: {len(exp_results)}, Avg CodeBLEU: {avg_score:.4f}")
        for r in sorted(exp_results, key=lambda x: x['avg_codebleu'], reverse=True)[:3]:
            print(f"         - {r['model'][:45]:45s}: {r['avg_codebleu']:.4f}")
    
    print(f"\n🏆 Top 10 Models Overall:")
    sorted_results = sorted(results_summary, key=lambda x: x['avg_codebleu'], reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"   {i:2d}. {result['model'][:45]:45s} {result['avg_codebleu']:.4f} ({result['experiment']})")
    
    print(f"\n💾 Summary files saved:")
    print(f"   - {summary_json}")
    print(f"   - {summary_csv}")
    print(f"\n📁 Individual CSV files: {total_processed} files in {OUTPUT_BASE_DIR}/")
else:
    print("\n⚠️  No results to summarize")

print("=" * 80)
print("✨ Batch evaluation complete!")
print("=" * 80)
