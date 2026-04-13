"""
CodeBLEU Calculation with Proper Evaluation Results Mapping
===========================================================

This script uses the evaluation_results.json files to properly map
generated code with reference code and calculate CodeBLEU scores.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re
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
    """
    Extract pure code from generation text that contains docstrings and code.
    """
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


def calculate_codebleu_score(generated_code: str, reference_code: str) -> Tuple[float, Dict]:
    """Calculate CodeBLEU score for generated vs reference code"""
    try:
        result = evaluate_per_example(reference_code, generated_code, "python", "0.25,0.25,0.25,0.25")
        codebleu_score = result.get('codebleu', 0.0)
        return float(codebleu_score), result
    except Exception as e:
        return 0.0, {'error': str(e)}


def calculate_std(values: List[float]) -> float:
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def calculate_codebleu_with_eval_mapping(
    dataset_name: str,
    generations_path: str,
    dataset_path: str,
    eval_results_path: str,
    output_path: str = None
) -> Dict:
    """
    Calculate CodeBLEU using evaluation_results.json for proper mapping.
    
    The evaluation_results.json pass_info contains the actual task IDs that were evaluated,
    so we use that as the source of truth for which samples to evaluate.
    """
    
    print(f"\n{'='*70}")
    print(f"CALCULATING CodeBLEU for {dataset_name.upper()} Dataset (with Eval Mapping)")
    print(f"{'='*70}")
    
    # Load evaluation results to get the actual task mapping
    print("Loading evaluation results...")
    with open(eval_results_path, 'r') as f:
        eval_data = json.load(f)
    
    # Get the dataset key (e.g., 'mbpp' or 'humaneval')
    eval_key = dataset_name.lower() if dataset_name.lower() in eval_data else dataset_name
    if eval_key not in eval_data:
        print(f"ERROR: '{eval_key}' not found in evaluation results")
        return {'status': 'error', 'message': f'Dataset {dataset_name} not found in eval results'}
    
    eval_dataset = eval_data[eval_key]
    pass_info = eval_dataset.get('pass_info', {})
    
    print(f"  Found {len(pass_info)} evaluated task IDs in evaluation results")
    
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
        # Create task_id -> problem mapping
        problems = {item.get('task_id'): item for item in dataset if 'task_id' in item}
    elif dataset_name.lower() == 'humaneval':
        problems = {}
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    problems[item.get('task_id')] = item
    else:
        print(f"Unknown dataset: {dataset_name}")
        return {'status': 'error', 'message': f'Unknown dataset: {dataset_name}'}
    
    print(f"  Loaded {len(problems)} reference problems")
    
    results = {
        'dataset': dataset_name,
        'generation_method': 'with_eval_mapping',
        'evaluated_task_ids': len(pass_info),
        'total_problems_in_dataset': len(problems),
        'total_generations_available': len(generations),
        'successful_evaluations': 0,
        'failed_evaluations': 0,
        'codebleu_scores': {},
        'summary_stats': {},
        'errors': defaultdict(list)
    }
    
    scores = []
    
    # Evaluate each task_id from pass_info
    for task_id_str in sorted(pass_info.keys(), key=lambda x: int(x) if x.isdigit() else 9999):
        try:
            task_id = int(task_id_str)
        except ValueError:
            continue
        
        # Check if we have a generation for this task_id
        if task_id >= len(generations):
            results['failed_evaluations'] += 1
            results['errors']['generation_not_found'].append(f"Task {task_id} not in generations")
            continue
        
        # Get generation
        generation_list = generations[task_id]
        if not generation_list or not generation_list[0]:
            results['failed_evaluations'] += 1
            results['errors']['empty_generation'].append(f"Task {task_id} has empty generation")
            continue
        
        generated_code = generation_list[0]
        
        # Get reference problem - need to find it by task_id
        if task_id not in problems:
            results['failed_evaluations'] += 1
            results['errors']['problem_not_found'].append(f"Task {task_id} not in dataset")
            continue
        
        problem = problems[task_id]
        
        # Extract reference code
        if dataset_name.lower() == 'mbpp':
            reference_code = problem.get('code', '')
        else:  # humaneval
            reference_code = problem.get('canonical_solution', '')
        
        if not reference_code:
            results['failed_evaluations'] += 1
            results['errors']['missing_reference'].append(f"Task {task_id} has no reference")
            continue
        
        # Extract and evaluate
        try:
            extracted_code = extract_code_from_generation(generated_code)
            
            if not extracted_code:
                results['failed_evaluations'] += 1
                results['errors']['extraction_failed'].append(f"Task {task_id} - code extraction failed")
                continue
            
            score, metrics = calculate_codebleu_score(extracted_code, reference_code)
            
            results['codebleu_scores'][task_id] = {
                'score': score,
                'metrics': metrics,
                'reference_length': len(reference_code),
                'generated_length': len(extracted_code)
            }
            
            scores.append(score)
            results['successful_evaluations'] += 1
            
            if (task_id + 1) % 50 == 0:
                print(f"  Processed {task_id + 1}/{len(pass_info)} evaluated samples...")
        
        except Exception as e:
            results['failed_evaluations'] += 1
            results['errors']['evaluation_error'].append(f"Task {task_id}: {str(e)}")
    
    # Calculate statistics
    if scores:
        results['summary_stats'] = {
            'mean_codebleu': float(sum(scores) / len(scores)),
            'max_codebleu': float(max(scores)),
            'min_codebleu': float(min(scores)),
            'median_codebleu': float(sorted(scores)[len(scores)//2]) if len(scores) > 0 else 0.0,
            'std_codebleu': calculate_std(scores) if len(scores) > 1 else 0.0,
            'num_scores': len(scores)
        }
    
    print(f"\n{dataset_name.upper()} Results:")
    print(f"  Successfully evaluated: {results['successful_evaluations']}/{len(pass_info)}")
    print(f"  Failed: {results['failed_evaluations']}/{len(pass_info)}")
    if scores:
        print(f"  Mean CodeBLEU: {results['summary_stats']['mean_codebleu']:.4f}")
        print(f"  Median CodeBLEU: {results['summary_stats']['median_codebleu']:.4f}")
        print(f"  Min CodeBLEU: {results['summary_stats']['min_codebleu']:.4f}")
        print(f"  Max CodeBLEU: {results['summary_stats']['max_codebleu']:.4f}")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error_type, error_list in results['errors'].items():
            print(f"  {error_type}: {len(error_list)} cases")
    
    return results


def main():
    """Main execution function"""
    
    base_path = Path('/home/fahad/Documents/PROJECTS/promptmark')
    
    # MBPP paths
    mbpp_gen_path = base_path / 'output/baseline_results/mbpp/generations.json'
    mbpp_dataset_path = base_path / 'datasets/sanitized-mbpp.json'
    mbpp_eval_path = base_path / 'output/baseline_results/mbpp/evaluation_results.json'
    mbpp_output_path = base_path / 'output/baseline_results/mbpp/codebleu_scores_mapped.json'
    
    # HumanEval paths
    humaneval_gen_path = base_path / 'output/baseline_results/humaneval_gen3/generations.json'
    humaneval_dataset_path = base_path / 'datasets/humaneval_164.jsonl'
    humaneval_eval_path = base_path / 'output/baseline_results/humaneval_gen3/evaluation_results.json'
    humaneval_output_path = base_path / 'output/baseline_results/humaneval_gen3/codebleu_scores_mapped.json'
    
    # Verify files
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
    
    # Calculate CodeBLEU for both datasets using eval_results mapping
    all_results = {}
    
    # MBPP
    mbpp_results = calculate_codebleu_with_eval_mapping(
        'mbpp',
        str(mbpp_gen_path),
        str(mbpp_dataset_path),
        str(mbpp_eval_path),
        str(mbpp_output_path)
    )
    all_results['mbpp'] = mbpp_results
    
    # HumanEval
    humaneval_results = calculate_codebleu_with_eval_mapping(
        'humaneval',
        str(humaneval_gen_path),
        str(humaneval_dataset_path),
        str(humaneval_eval_path),
        str(humaneval_output_path)
    )
    all_results['humaneval'] = humaneval_results
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    results_path = base_path / 'output/codebleu_evaluation_mapped.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Complete results saved to: {results_path}")
    
    # Save individual results
    mbpp_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mbpp_output_path, 'w') as f:
        json.dump(mbpp_results, f, indent=2)
    print(f"✓ MBPP results saved to: {mbpp_output_path}")
    
    humaneval_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(humaneval_output_path, 'w') as f:
        json.dump(humaneval_results, f, indent=2)
    print(f"✓ HumanEval results saved to: {humaneval_output_path}")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY (WITH EVAL RESULTS MAPPING)")
    print(f"{'='*70}")
    
    if 'summary_stats' in mbpp_results and mbpp_results['summary_stats']:
        print("\nMBPP Summary:")
        for key, value in mbpp_results['summary_stats'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    if 'summary_stats' in humaneval_results and humaneval_results['summary_stats']:
        print("\nHumanEval Summary:")
        for key, value in humaneval_results['summary_stats'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
