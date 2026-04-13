#!/usr/bin/env python3
"""
Batch Evaluation Script

Evaluate multiple models/experiments and generate a comparison report.

Usage:
    python batch_eval.py --models codegemma-7b-it qwen-2_5-14b --experiments exp1_dgen_v1_100
    python batch_eval.py --config batch_config.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict
from pathlib import Path
from comprehensive_evaluation import ComprehensiveEvaluator

# Get the directory of this script and calculate project root
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
results_dir = os.path.join(root_dir, 'results')

def create_comparison_report(all_results: Dict[str, Dict], output_file: str = None):
    """Create a comprehensive comparison report."""
    
    print("\n" + "="*80)
    print("📊 BATCH EVALUATION COMPARISON REPORT (UNIFIED DETECTION METHOD)")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    
    for model_exp, results in all_results.items():
        row = {
            'Model_Experiment': model_exp,
            'Pass@1': round(results.get('pass_at_1', 0.0)*100, 2),
            'AUROC': round(results.get('auroc', 0.0), 4),
            'T@0%F': round(results.get('tpr_values', {}).get('T@0%F', 0.0), 4),
            'T@1%F': round(results.get('tpr_values', {}).get('T@1%F', 0.0), 4),
            'T@5%F': round(results.get('tpr_values', {}).get('T@5%F', 0.0), 4),
            'T@10%F': round(results.get('tpr_values', {}).get('T@10%F', 0.0), 4),
            'CodeBLEU_Mean': round(results.get('codebleu', {}).get('mean', 0.0)*100, 2),
            'CodeBLEU_Std': round(results.get('codebleu', {}).get('std', 0.0)*100, 2),
            'Examples_Evaluated': results.get('codebleu', {}).get('count', 0)
        }
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Print formatted table
    print("\n📋 Summary Table (All Experiments):")
    print("-" * 120)
    
    # Format and display the table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(df_comparison.round(4).to_string(index=False))
    
    # Create model-level aggregation
    print("\n📊 MODEL-LEVEL AGGREGATION:")
    print("-" * 120)
    
    # Group by model (extract model name from key)
    model_aggregation = {}
    
    for model_exp, results in all_results.items():
        # Extract model name (first part before underscore)
        model_name = model_exp.split('_')[0]
        
        if model_name not in model_aggregation:
            model_aggregation[model_name] = []
        
        model_aggregation[model_name].append({
            'pass_at_1': results.get('pass_at_1', 0.0),
            'auroc': results.get('auroc', 0.0),
            't@0%f': results.get('tpr_values', {}).get('T@0%F', 0.0),
            't@1%f': results.get('tpr_values', {}).get('T@1%F', 0.0),
            't@5%f': results.get('tpr_values', {}).get('T@5%F', 0.0),
            't@10%f': results.get('tpr_values', {}).get('T@10%F', 0.0),
            'codebleu_mean': results.get('codebleu', {}).get('mean', 0.0),
            'codebleu_std': results.get('codebleu', {}).get('std', 0.0),
            'examples_evaluated': results.get('codebleu', {}).get('count', 0)
        })
    
    # Calculate averages for each model
    model_avg_data = []
    for model_name, experiments in model_aggregation.items():
        if len(experiments) > 1:
            # Calculate averages across versions
            avg_pass_at_1 = sum(exp['pass_at_1'] for exp in experiments) / len(experiments)
            avg_auroc = sum(exp['auroc'] for exp in experiments) / len(experiments)
            avg_t0f = sum(exp['t@0%f'] for exp in experiments) / len(experiments)
            avg_t1f = sum(exp['t@1%f'] for exp in experiments) / len(experiments)
            avg_t5f = sum(exp['t@5%f'] for exp in experiments) / len(experiments)
            avg_t10f = sum(exp['t@10%f'] for exp in experiments) / len(experiments)
            avg_codebleu_mean = sum(exp['codebleu_mean'] for exp in experiments) / len(experiments)
            avg_codebleu_std = sum(exp['codebleu_std'] for exp in experiments) / len(experiments)
            total_examples = sum(exp['examples_evaluated'] for exp in experiments)
            
            model_avg_data.append({
                'Model': model_name,
                'Versions_Aggregated': len(experiments),
                'Avg_Pass@1': round(avg_pass_at_1 * 100, 2),
                'Avg_AUROC': round(avg_auroc, 4),
                'Avg_T@0%F': round(avg_t0f, 4),
                'Avg_T@1%F': round(avg_t1f, 4),
                'Avg_T@5%F': round(avg_t5f, 4),
                'Avg_T@10%F': round(avg_t10f, 4),
                'Avg_CodeBLEU_Mean': round(avg_codebleu_mean * 100, 2),
                'Avg_CodeBLEU_Std': round(avg_codebleu_std * 100, 2),
                'Total_Examples': total_examples
            })
    
    if model_avg_data:
        df_model_avg = pd.DataFrame(model_avg_data)
        
        print("\n📋 Model Averages Table:")
        print("-" * 120)
        print(df_model_avg.to_string(index=False))
        
        # Best performer analysis for models
        print("\n🏆 Best Performing Models (Averaged):")
        print("-" * 60)
        
        metrics = ['Avg_Pass@1', 'Avg_AUROC', 'Avg_T@10%F', 'Avg_CodeBLEU_Mean']
        for metric in metrics:
            if metric in df_model_avg.columns:
                best_idx = df_model_avg[metric].idxmax()
                best_model = df_model_avg.iloc[best_idx]['Model']
                best_value = df_model_avg.iloc[best_idx][metric]
                versions = df_model_avg.iloc[best_idx]['Versions_Aggregated']
                print(f"{metric:18s}: {best_model} ({best_value:.4f}, {versions} versions)")
    else:
        print("ℹ️  No models with multiple versions found for aggregation")
    
    # Best performer analysis
    print("\n🏆 Best Performers (By Individual Experiment):")
    print("-" * 60)
    
    metrics = ['Pass@1', 'AUROC', 'T@10%F', 'CodeBLEU_Mean']
    for metric in metrics:
        if metric in df_comparison.columns:
            best_idx = df_comparison[metric].idxmax()
            best_model = df_comparison.iloc[best_idx]['Model_Experiment']
            best_value = df_comparison.iloc[best_idx][metric]
            print(f"{metric:15s}: {best_model:50s} ({best_value:.4f})")

    # Print detection method note
    print("\n" + "="*80)
    print("📌 DETECTION METHOD INFORMATION:")
    print("="*80)
    print("✓ Detection Method:  UNIFIED p-value approach (-log10(p_unified))")
    print("✓ Scoring Metric:    All samples use exact binomial p-value")
    print("✓ ROC Consistency:   Decisions and ranking use SAME metric ✓")
    print("✓ T@X%F Accuracy:    Proper FPR threshold-based TPR calculation ✓")
    print("="*80)

    # update output file path 
    Path(f"{results_dir}/evals").mkdir(parents=True, exist_ok=True)
    output_file =  Path(results_dir) / "evals" / output_file
    print("\n💾 OUTPUT FILE PATH:", output_file)

    # Save detailed results
    if output_file:
        # Save comparison table as CSV
        csv_file = output_file.with_name(output_file.stem + '_comparison.csv')
        df_comparison.to_csv(csv_file, index=False)
        
        # Save model averages as separate CSV if available
        if model_avg_data:
            model_csv_file = output_file.with_name(output_file.stem + '_model_averages.csv')
            df_model_avg.to_csv(model_csv_file, index=False)
            print(f"💾 Model averages saved to: {model_csv_file}")
        
        # Save detailed results as JSON
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"💾 Detailed results saved to: {output_file}")
        print(f"💾 Comparison table saved to: {csv_file}")

def get_available_experiments():
    """Scan for available experiments in the results and output directories."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    
    results_dir = f"{base_dir}/results/raw"
    output_dir = f"{base_dir}/output"
    
    available = []
    
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                # Check if corresponding CSV exists
                csv_path = os.path.join(results_dir, f"{item}.csv")
                if os.path.exists(csv_path):
                    available.append(item)
    
    return available

def main():
    parser = argparse.ArgumentParser(description='Batch Code Evaluation Script')
    
    parser.add_argument('--experiments', nargs='+', type=str, default='all',
                       help='Experiment names to evaluate (use "all" for all available)')
    parser.add_argument('--config', type=str,
                       help='JSON config file with experiment names')
    parser.add_argument('--codebleu_sample_size', type=int, default=20,
                       help='Number of examples for CodeBLEU evaluation')
    parser.add_argument('--output', type=str, default='batch_evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--list_available', action='store_true',
                       help='List available experiments and exit')
    
    args = parser.parse_args()
    
    # List available experiments
    if args.list_available:
        print("🔍 Scanning for available experiments...")
        available = get_available_experiments()
        
        if available:
            print("\n📋 Available Experiments:")
            print("-" * 50)
            for experiment in available:
                print(f"  {experiment}")
            print(f"\nTotal: {len(available)} experiments found")
            print("\nUsage example:")
            print(f"  python batch_eval.py --experiments {available[0]}")
            print(f"  # Or use --experiments all to evaluate everything")
        else:
            print("❌ No matching experiments found")
        
        return 0
    
    # Determine evaluation pairs
    eval_pairs = []

    if args.config:
        # Load from config file
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        eval_pairs = config['experiments']
    
    if 'all' in args.experiments:
        available = get_available_experiments()
        eval_pairs = available
    else:
        eval_pairs = args.experiments
    
    if not eval_pairs:
        print("❌ No evaluation pairs specified")
        return 1
    
    print(f"🚀 Starting batch evaluation for {len(eval_pairs)} experiments...")
    
    all_results = {}
    successful = 0
    failed = 0
    
    for i, experiment in enumerate(eval_pairs):
        print(f"\n{'='*60}")
        print(f"📈 Evaluating {i+1}/{len(eval_pairs)}: {experiment}")
        print(f"{'='*60}")
        
        try:
            # Parse experiment name
            parts = experiment.split('_')
            print("PARTS: ", parts)
            if len(parts) < 7 or parts[3] != 'gen':
                print(f"⚠️  Invalid experiment name format: {experiment}")
                print(f"   Expected format: model_exp-version_generation-type_gen_version_size_dataset")
                print(f"   Example: codegemma_exp1-1_during_gen_v4_100_mbpp")
                failed += 1
                continue
            
            model = parts[0]
            exp_number = parts[1]
            generation_type = parts[2]  # 'during' or 'after'
            version = parts[4]
            size = parts[5]
            dataset = parts[6]
            
            print(f"📋 Parsed experiment: Model={model}, Exp={exp_number}, GenType={generation_type}, Version={version}, Size={size}, Dataset={dataset}")
            
            # Get file paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(script_dir))
            csv_file = f"{base_dir}/results/raw/{experiment}.csv"
            generated_dir = f"{base_dir}/output/{experiment}"
            
            # Reference file - auto-detect based on dataset
            if dataset == 'humaneval':
                reference_file = f"{base_dir}/datasets/human_eval_164.jsonl"
            elif dataset == 'mbpp':
                reference_file = f"{base_dir}/datasets/sanitized-mbpp.json"
            else:
                reference_file = f"{base_dir}/datasets/{dataset}.jsonl"  # fallback
            
            # Validate paths
            if not os.path.exists(csv_file):
                print(f"⚠️  CSV file not found: {csv_file}")
                failed += 1
                continue
            
            if not os.path.exists(generated_dir):
                print(f"⚠️  Generated directory not found: {generated_dir}")
                failed += 1
                continue
            
            if not os.path.exists(reference_file):
                print(f"⚠️  Reference file not found: {reference_file}")
                failed += 1
                continue
            
            # Run evaluation
            evaluator = ComprehensiveEvaluator(csv_file, generated_dir, reference_file)
            results = evaluator.run_comprehensive_evaluation(args.codebleu_sample_size)
            
            # Store results
            model_exp_key = experiment
            all_results[model_exp_key] = results
            successful += 1
            
            print(f"✅ {experiment} evaluation completed")
            
        except Exception as e:
            print(f"❌ Error evaluating {experiment}: {e}")
            failed += 1
            continue
    
    print(f"\n🏁 Batch evaluation completed: {successful} successful, {failed} failed")
    
    if successful > 0:
        # Generate comparison report
        create_comparison_report(all_results, args.output)
    
    return 0 if successful > 0 else 1

if __name__ == "__main__":
    exit(main())