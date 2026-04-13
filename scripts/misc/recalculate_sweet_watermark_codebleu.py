#!/usr/bin/env python3
"""
Recalculate CodeBLEU scores for sweet-watermark extracted responses with modified weights:
- BLEU weight: 0
- Weighted BLEU weight: 0
- AST/Syntax weight: 0.5
- Dataflow weight: 0.5
"""

import os
import sys
import json
from pathlib import Path

# Add metrics directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts/evals/metrics'))

# Import the CodeBLEU calculation functions directly
sys.path.insert(0, 'scripts/evals')
from calculate_mbpp_codebleu import load_mbpp_dataset, save_results_to_csv

# Import the evaluate function with custom weights
from metrics.calc_code_bleu import evaluate_per_example

# Configuration
SAMPLE_DIR = "extracted_responses"
DATASET_JSON = "dataset/sanitized-mbpp.json"
OUTPUT_DIR = "results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "sweet_watermark_codebleu_ast_dataflow.csv")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "sweet_watermark_codebleu_ast_dataflow.json")

# New weights: BLEU=0, wBLEU=0, AST=0.5, Dataflow=0.5
CUSTOM_WEIGHTS = "0,0,0.5,0.5"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("🔬 RECALCULATING SWEET-WATERMARK CODEBLEU WITH AST + DATAFLOW WEIGHTS")
print("=" * 80)
print(f"\n📊 Weight Configuration:")
print(f"   BLEU (alpha):          0.0")
print(f"   Weighted BLEU (beta):  0.0")
print(f"   AST/Syntax (gamma):    0.5")
print(f"   Dataflow (theta):      0.5")
print("=" * 80)

print("\n📂 Loading MBPP dataset...")
task_references = load_mbpp_dataset(DATASET_JSON)
print(f"✅ Loaded {len(task_references)} reference implementations")

print(f"\n📁 Processing samples from: {SAMPLE_DIR}")
print("-" * 80)

results = []
sample_path = Path(SAMPLE_DIR)

# Get all Python files in sample directory
sample_files = sorted(sample_path.glob('*.py'))

print(f"Found {len(sample_files)} sample files")
print("Processing CodeBLEU calculations...")
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
        
        # Calculate CodeBLEU with custom weights
        scores = evaluate_per_example(
            reference=reference_code,
            hypothesis=sample_code,
            lang='python',
            params=CUSTOM_WEIGHTS
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
        print(f"[{idx}/{len(sample_files)}] Task {task_id}: "
              f"CodeBLEU={scores['codebleu']:.4f} "
              f"(AST={scores['syntax']:.4f}, Dataflow={scores['dataflow']:.4f})")
        
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
    
    # Save to CSV
    save_results_to_csv(results, OUTPUT_CSV)
    
    # Save to JSON with summary
    summary = {
        'total_samples': len(results),
        'weights': {
            'bleu': 0.0,
            'weighted_bleu': 0.0,
            'ast_syntax': 0.5,
            'dataflow': 0.5
        },
        'statistics': {
            'codebleu': {
                'mean': sum(codebleu_scores)/len(codebleu_scores),
                'min': min(codebleu_scores),
                'max': max(codebleu_scores)
            },
            'bleu': {
                'mean': sum(bleu_scores)/len(bleu_scores),
                'min': min(bleu_scores),
                'max': max(bleu_scores)
            },
            'syntax': {
                'mean': sum(syntax_scores)/len(syntax_scores),
                'min': min(syntax_scores),
                'max': max(syntax_scores)
            },
            'dataflow': {
                'mean': sum(dataflow_scores)/len(dataflow_scores),
                'min': min(dataflow_scores),
                'max': max(dataflow_scores)
            }
        },
        'results': results
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   - CSV: {OUTPUT_CSV}")
    print(f"   - JSON: {OUTPUT_JSON}")
    
    print("\n" + "=" * 80)
    print("✨ Sweet-watermark CodeBLEU calculation complete!")
    print("=" * 80)
else:
    print("\n⚠️  No results to save")
