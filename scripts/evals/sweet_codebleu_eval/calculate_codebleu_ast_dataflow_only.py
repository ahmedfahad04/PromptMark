"""
CodeBLEU Calculation with AST+Dataflow Only
============================================

This script calculates CodeBLEU scores using ONLY the AST and Dataflow components,
excluding the BLEU and weighted BLEU components. This focuses on structural similarity
and program-level semantics.

AST+Dataflow CodeBLEU = gamma * AST + delta * Dataflow
where gamma = 0.25 and delta = 0.25 (equal weights)
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add metrics to path
metrics_dir = '/home/fahad/Documents/PROJECTS/promptmark/experiments/evals/metrics'
sys.path.insert(0, metrics_dir)

try:
    from calc_code_bleu import evaluate_per_example
except ImportError as e:
    print(f"Error importing calc_code_bleu: {e}")
    sys.exit(1)


def extract_code_from_generation(generation_text: str) -> str:
    """Extract pure code from generation text that contains docstrings and code."""
    if not generation_text:
        return ""
    
    lines = generation_text.split('\n')
    code_lines = []
    in_docstring = False
    docstring_quote_count = 0
    
    for line in lines:
        if '"""' in line:
            docstring_quote_count += line.count('"""')
            in_docstring = (docstring_quote_count % 2) == 1
            if not in_docstring and line.strip() == '"""':
                continue
        
        if not in_docstring:
            if line.strip() and not line.strip().startswith('#'):
                code_lines.append(line)
    
    code = '\n'.join(code_lines).strip()
    return code


def calculate_codebleu_ast_dataflow_only(
    generated_code: str, 
    reference_code: str,
    ast_weight: float = 0.5,
    dataflow_weight: float = 0.5
) -> Tuple[float, Dict]:
    """
    Calculate CodeBLEU score using ONLY AST and Dataflow components.
    
    Args:
        generated_code: Generated Python code
        reference_code: Reference Python code
        ast_weight: Weight for AST component (default 0.5)
        dataflow_weight: Weight for Dataflow component (default 0.5)
    
    Returns:
        (ast_dataflow_score, detailed_metrics): AST+Dataflow score and all metrics
    """
    try:
        # Use standard weights for all components, but we'll extract only AST and Dataflow
        result = evaluate_per_example(reference_code, generated_code, "python", "0.25,0.25,0.25,0.25")
        
        # Extract AST (syntax) and Dataflow scores
        ast_score = result.get('syntax', 0.0)
        dataflow_score = result.get('dataflow', 0.0)
        
        # Calculate combined AST+Dataflow score
        ast_dataflow_combined = ast_weight * ast_score + dataflow_weight * dataflow_score
        
        # Add the AST+Dataflow score to result
        result['codebleu_ast_dataflow'] = ast_dataflow_combined
        result['ast_weight'] = ast_weight
        result['dataflow_weight'] = dataflow_weight
        
        return float(ast_dataflow_combined), result
        
    except Exception as e:
        return 0.0, {'error': str(e)}


def calculate_std(values: List[float]) -> float:
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def calculate_codebleu_ast_dataflow(
    dataset_name: str,
    generations_path: str,
    dataset_path: str,
    eval_results_path: str,
    ast_weight: float = 0.5,
    dataflow_weight: float = 0.5,
    output_path: str = None
) -> Dict:
    """
    Calculate CodeBLEU using ONLY AST and Dataflow components with proper mapping.
    """
    
    print(f"\n{'='*70}")
    print(f"CALCULATING CodeBLEU (AST+DATAFLOW ONLY) for {dataset_name.upper()}")
    print(f"AST Weight: {ast_weight}, Dataflow Weight: {dataflow_weight}")
    print(f"{'='*70}")
    
    # Load evaluation results
    print("Loading evaluation results...")
    with open(eval_results_path, 'r') as f:
        eval_data = json.load(f)
    
    eval_key = dataset_name.lower() if dataset_name.lower() in eval_data else dataset_name
    if eval_key not in eval_data:
        print(f"ERROR: '{eval_key}' not found in evaluation results")
        return {'status': 'error', 'message': f'Dataset {dataset_name} not found in eval results'}
    
    eval_dataset = eval_data[eval_key]
    pass_info = eval_dataset.get('pass_info', {})
    
    print(f"  Found {len(pass_info)} evaluated task IDs")
    
    # Load generations
    print("Loading generations...")
    with open(generations_path, 'r') as f:
        generations = json.load(f)
    print(f"  Loaded {len(generations)} generations")
    
    # Load reference dataset
    print("Loading reference dataset...")
    if dataset_name.lower() == 'mbpp':
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        problems = {item.get('task_id'): item for item in dataset if 'task_id' in item}
    elif dataset_name.lower() == 'humaneval':
        problems = {}
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    problems[item.get('task_id')] = item
    else:
        return {'status': 'error', 'message': f'Unknown dataset: {dataset_name}'}
    
    print(f"  Loaded {len(problems)} reference problems")
    
    results = {
        'dataset': dataset_name,
        'evaluation_method': 'ast_dataflow_only',
        'ast_weight': ast_weight,
        'dataflow_weight': dataflow_weight,
        'evaluated_task_ids': len(pass_info),
        'total_problems_in_dataset': len(problems),
        'successful_evaluations': 0,
        'failed_evaluations': 0,
        'codebleu_scores': {},
        'summary_stats': {},
        'component_stats': {},
        'errors': defaultdict(list)
    }
    
    scores_ast_dataflow = []
    scores_ast = []
    scores_dataflow = []
    
    # Evaluate each task
    for task_id_str in sorted(pass_info.keys(), key=lambda x: int(x) if x.isdigit() else 9999):
        try:
            task_id = int(task_id_str)
        except ValueError:
            continue
        
        # Check generation exists
        if task_id >= len(generations):
            results['failed_evaluations'] += 1
            results['errors']['generation_not_found'].append(f"Task {task_id}")
            continue
        
        generation_list = generations[task_id]
        if not generation_list or not generation_list[0]:
            results['failed_evaluations'] += 1
            results['errors']['empty_generation'].append(f"Task {task_id}")
            continue
        
        generated_code = generation_list[0]
        
        # Check problem exists
        if task_id not in problems:
            results['failed_evaluations'] += 1
            results['errors']['problem_not_found'].append(f"Task {task_id}")
            continue
        
        problem = problems[task_id]
        
        # Extract reference code
        if dataset_name.lower() == 'mbpp':
            reference_code = problem.get('code', '')
        else:
            reference_code = problem.get('canonical_solution', '')
        
        if not reference_code:
            results['failed_evaluations'] += 1
            results['errors']['missing_reference'].append(f"Task {task_id}")
            continue
        
        # Extract and evaluate
        try:
            extracted_code = extract_code_from_generation(generated_code)
            
            if not extracted_code:
                results['failed_evaluations'] += 1
                results['errors']['extraction_failed'].append(f"Task {task_id}")
                continue
            
            # Calculate AST+Dataflow score
            score_ast_dataflow, metrics = calculate_codebleu_ast_dataflow_only(
                extracted_code,
                reference_code,
                ast_weight=ast_weight,
                dataflow_weight=dataflow_weight
            )
            
            # Extract component scores
            ast_score = metrics.get('syntax', 0.0)
            dataflow_score = metrics.get('dataflow', 0.0)
            
            results['codebleu_scores'][task_id] = {
                'ast_dataflow_combined': score_ast_dataflow,
                'ast_score': ast_score,
                'dataflow_score': dataflow_score,
                'bleu_score': metrics.get('bleu', 0.0),
                'weighted_bleu_score': metrics.get('wbleu', 0.0),
                'full_codebleu': metrics.get('codebleu', 0.0),
                'reference_length': len(reference_code),
                'generated_length': len(extracted_code)
            }
            
            scores_ast_dataflow.append(score_ast_dataflow)
            scores_ast.append(ast_score)
            scores_dataflow.append(dataflow_score)
            results['successful_evaluations'] += 1
            
            if (task_id + 1) % 50 == 0:
                print(f"  Processed {task_id + 1}/{len(pass_info)} samples...")
        
        except Exception as e:
            results['failed_evaluations'] += 1
            results['errors']['evaluation_error'].append(f"Task {task_id}: {str(e)}")
    
    # Calculate statistics
    if scores_ast_dataflow:
        results['summary_stats'] = {
            'ast_dataflow': {
                'mean': float(sum(scores_ast_dataflow) / len(scores_ast_dataflow)),
                'max': float(max(scores_ast_dataflow)),
                'min': float(min(scores_ast_dataflow)),
                'median': float(sorted(scores_ast_dataflow)[len(scores_ast_dataflow)//2]),
                'std': calculate_std(scores_ast_dataflow),
                'count': len(scores_ast_dataflow)
            }
        }
        
        results['component_stats'] = {
            'ast': {
                'mean': float(sum(scores_ast) / len(scores_ast)),
                'max': float(max(scores_ast)),
                'min': float(min(scores_ast)),
                'median': float(sorted(scores_ast)[len(scores_ast)//2]),
                'std': calculate_std(scores_ast)
            },
            'dataflow': {
                'mean': float(sum(scores_dataflow) / len(scores_dataflow)),
                'max': float(max(scores_dataflow)),
                'min': float(min(scores_dataflow)),
                'median': float(sorted(scores_dataflow)[len(scores_dataflow)//2]),
                'std': calculate_std(scores_dataflow)
            }
        }
    
    print(f"\n{dataset_name.upper()} Results (AST+Dataflow Only):")
    print(f"  Successfully evaluated: {results['successful_evaluations']}/{len(pass_info)}")
    print(f"  Failed: {results['failed_evaluations']}/{len(pass_info)}")
    
    if results['summary_stats']:
        stats = results['summary_stats']['ast_dataflow']
        print(f"\n  AST+Dataflow Combined Score:")
        print(f"    Mean:   {stats['mean']:.4f}")
        print(f"    Median: {stats['median']:.4f}")
        print(f"    Min:    {stats['min']:.4f}")
        print(f"    Max:    {stats['max']:.4f}")
        print(f"    Std:    {stats['std']:.4f}")
        
        comp = results['component_stats']
        print(f"\n  Component Breakdown:")
        print(f"    AST Mean:      {comp['ast']['mean']:.4f}")
        print(f"    Dataflow Mean: {comp['dataflow']['mean']:.4f}")
    
    if results['errors']:
        print(f"\n  Errors ({sum(len(v) for v in results['errors'].values())} total):")
        for error_type, error_list in results['errors'].items():
            print(f"    • {error_type}: {len(error_list)}")
    
    return results


def main():
    """Main execution"""
    
    base_path = Path('/home/fahad/Documents/PROJECTS/promptmark')
    
    # MBPP paths
    mbpp_gen_path = base_path / 'output/baseline_results/mbpp/generations.json'
    mbpp_dataset_path = base_path / 'datasets/sanitized-mbpp.json'
    mbpp_eval_path = base_path / 'output/baseline_results/mbpp/evaluation_results.json'
    mbpp_output_path = base_path / 'output/baseline_results/mbpp/codebleu_ast_dataflow_only.json'
    
    # HumanEval paths
    humaneval_gen_path = base_path / 'output/baseline_results/humaneval_gen3/generations.json'
    humaneval_dataset_path = base_path / 'datasets/humaneval_164.jsonl'
    humaneval_eval_path = base_path / 'output/baseline_results/humaneval_gen3/evaluation_results.json'
    humaneval_output_path = base_path / 'output/baseline_results/humaneval_gen3/codebleu_ast_dataflow_only.json'
    
    print("\nVerifying file paths...")
    for path, name in [
        (mbpp_gen_path, 'MBPP generations'),
        (mbpp_dataset_path, 'MBPP dataset'),
        (mbpp_eval_path, 'MBPP eval results'),
        (humaneval_dataset_path, 'HumanEval dataset'),
        (humaneval_eval_path, 'HumanEval eval results')
    ]:
        if path.exists():
            size_mb = path.stat().st_size / (1024*1024)
            print(f"  ✓ {name}: {size_mb:.2f} MB")
        else:
            print(f"  ✗ {name}: NOT FOUND")
    
    # AST+Dataflow weights (equal distribution)
    ast_weight = 0.5
    dataflow_weight = 0.5
    
    all_results = {}
    
    # MBPP
    mbpp_results = calculate_codebleu_ast_dataflow(
        'mbpp',
        str(mbpp_gen_path),
        str(mbpp_dataset_path),
        str(mbpp_eval_path),
        ast_weight=ast_weight,
        dataflow_weight=dataflow_weight,
        output_path=str(mbpp_output_path)
    )
    all_results['mbpp'] = mbpp_results
    
    # HumanEval
    humaneval_results = calculate_codebleu_ast_dataflow(
        'humaneval',
        str(humaneval_gen_path),
        str(humaneval_dataset_path),
        str(humaneval_eval_path),
        ast_weight=ast_weight,
        dataflow_weight=dataflow_weight,
        output_path=str(humaneval_output_path)
    )
    all_results['humaneval'] = humaneval_results
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    results_path = base_path / 'output/codebleu_evaluation_ast_dataflow_only.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Complete results saved to: {results_path}")
    
    mbpp_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mbpp_output_path, 'w') as f:
        json.dump(mbpp_results, f, indent=2)
    print(f"✓ MBPP results saved to: {mbpp_output_path}")
    
    humaneval_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(humaneval_output_path, 'w') as f:
        json.dump(humaneval_results, f, indent=2)
    print(f"✓ HumanEval results saved to: {humaneval_output_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY (AST+DATAFLOW ONLY)")
    print(f"{'='*70}\n")
    
    for dataset_name, data in all_results.items():
        print(f"📊 {dataset_name.upper()}:")
        if 'summary_stats' in data and data['summary_stats']:
            stats = data['summary_stats'].get('ast_dataflow', {})
            print(f"  AST+Dataflow Mean: {stats.get('mean', 0):.4f}")
            print(f"  AST+Dataflow Median: {stats.get('median', 0):.4f}")
            
            comp = data.get('component_stats', {})
            if comp:
                print(f"  AST Mean: {comp.get('ast', {}).get('mean', 0):.4f}")
                print(f"  Dataflow Mean: {comp.get('dataflow', {}).get('mean', 0):.4f}")
        print()


if __name__ == '__main__':
    main()
