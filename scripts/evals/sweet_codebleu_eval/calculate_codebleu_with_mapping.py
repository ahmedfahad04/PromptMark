"""
CodeBLEU Calculation with Proper Data Mapping
==============================================

This script:
1. Loads generations.json and respective datasets
2. Maps generated code to reference code using indices/task_ids
3. Extracts code portions from generation.json (which contains docstring + code)
4. Calculates CodeBLEU scores using the evals framework
5. Outputs comprehensive evaluation results
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
    print("Make sure metrics folder is in the path")
    sys.exit(1)


class CodeExtractionError(Exception):
    """Exception for code extraction failures"""
    pass


def extract_code_from_generation(generation_text: str) -> str:
    """
    Extract pure code from generation text that contains docstrings and code.
    
    Format in generation.json contains problem description in triple quotes
    followed by actual code implementation.
    
    We need to extract only the code part (after the closing triple quotes of docstring).
    """
    if not generation_text:
        return ""
    
    # Pattern: Find all triple-quoted docstrings and extract code after them
    # This handles multiple functions in one generation
    
    lines = generation_text.split('\n')
    code_lines = []
    in_docstring = False
    docstring_quote_count = 0
    
    for line in lines:
        # Count triple quotes
        if '"""' in line:
            docstring_quote_count += line.count('"""')
            in_docstring = (docstring_quote_count % 2) == 1
            # If this line only has the closing """, skip this line
            if not in_docstring and line.strip() == '"""':
                continue
        
        # Add line if not in docstring (and it's not an import or empty)
        if not in_docstring:
            if line.strip() and not line.strip().startswith('#'):
                code_lines.append(line)
    
    code = '\n'.join(code_lines).strip()
    return code


def load_mbpp_dataset(dataset_path: str) -> Dict[int, Dict]:
    """Load MBPP dataset and create task_id -> problem mapping"""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Create mapping by task_id
    problems = {}
    for item in dataset:
        task_id = item.get('task_id')
        if task_id is not None:
            problems[task_id] = item
    
    return problems


def load_humaneval_dataset(dataset_path: str) -> Dict[int, Dict]:
    """Load HumanEval dataset from JSONL format"""
    problems = {}
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                task_id = item.get('task_id')
                if task_id is not None:
                    problems[task_id] = item
    
    return problems


def load_generations(gen_path: str) -> List:
    """Load generations.json file"""
    with open(gen_path, 'r') as f:
        generations = json.load(f)
    return generations


def extract_reference_code(problem: Dict, dataset_type: str) -> str:
    """Extract reference code from problem dictionary"""
    if dataset_type == 'mbpp':
        # MBPP has 'code' field with the solution
        return problem.get('code', '')
    elif dataset_type == 'humaneval':
        # HumanEval has 'canonical_solution' field
        return problem.get('canonical_solution', '')
    return ''


def calculate_codebleu_score(generated_code: str, reference_code: str) -> Tuple[float, Dict]:
    """
    Calculate CodeBLEU score for generated vs reference code
    
    Returns:
        (score, metrics_dict): CodeBLEU score and detailed metrics
    """
    try:
        # Use default weights: 0.25 for each component (BLEU, weighted BLEU, AST, Dataflow)
        result = evaluate_per_example(reference_code, generated_code, "python", "0.25,0.25,0.25,0.25")
        
        # Result is a dictionary with keys: em, bleu, wbleu, syntax, dataflow, codebleu
        codebleu_score = result.get('codebleu', 0.0)
        
        return float(codebleu_score), result
        
    except Exception as e:
        print(f"Error calculating CodeBLEU: {e}")
        return 0.0, {'error': str(e)}


def calculate_mbpp_codebleu(
    generations_path: str,
    dataset_path: str,
    output_path: str = None
) -> Dict:
    """
    Calculate CodeBLEU for MBPP dataset with proper mapping
    """
    print(f"\n{'='*70}")
    print("CALCULATING CodeBLEU for MBPP Dataset")
    print(f"{'='*70}")
    
    # Load data
    print("Loading MBPP dataset...")
    problems = load_mbpp_dataset(dataset_path)
    print(f"  Loaded {len(problems)} problems")
    
    print("Loading generations...")
    generations = load_generations(generations_path)
    print(f"  Loaded {len(generations)} generations")
    
    results = {
        'dataset': 'mbpp',
        'total_samples': len(generations),
        'successful_evaluations': 0,
        'failed_evaluations': 0,
        'codebleu_scores': {},
        'summary_stats': {},
        'errors': defaultdict(list)
    }
    
    # Map generations to problems using index
    scores = []
    
    for idx, generation_list in enumerate(generations):
        task_id = idx
        
        # Get corresponding problem
        if task_id not in problems:
            results['failed_evaluations'] += 1
            results['errors']['missing_problem'].append(f"Task {task_id} not in dataset")
            continue
        
        problem = problems[task_id]
        reference_code = extract_reference_code(problem, 'mbpp')
        
        if not reference_code:
            results['failed_evaluations'] += 1
            results['errors']['missing_reference'].append(f"Task {task_id} has no reference code")
            continue
        
        # generation_list contains multiple generations for the same task
        # For now, use the first one (or average all?)
        if not generation_list or not generation_list[0]:
            results['failed_evaluations'] += 1
            results['errors']['empty_generation'].append(f"Task {task_id} has empty generation")
            continue
        
        generated_code = generation_list[0]  # Use first generation
        
        try:
            # Extract pure code from generation
            extracted_code = extract_code_from_generation(generated_code)
            
            if not extracted_code:
                results['failed_evaluations'] += 1
                results['errors']['extraction_failed'].append(f"Task {task_id} - code extraction failed")
                continue
            
            # Calculate CodeBLEU
            score, metrics = calculate_codebleu_score(extracted_code, reference_code)
            
            results['codebleu_scores'][task_id] = {
                'score': score,
                'metrics': metrics,
                'problem_prompt': problem.get('prompt', '')[:100],  # First 100 chars of prompt
                'reference_length': len(reference_code),
                'generated_length': len(extracted_code)
            }
            
            scores.append(score)
            results['successful_evaluations'] += 1
            
            if (task_id + 1) % 20 == 0:
                print(f"  Processed {task_id + 1}/{len(generations)} samples...")
        
        except Exception as e:
            results['failed_evaluations'] += 1
            results['errors']['evaluation_error'].append(f"Task {task_id}: {str(e)}")
    
    # Calculate summary statistics
    if scores:
        results['summary_stats'] = {
            'mean_codebleu': float(sum(scores) / len(scores)),
            'max_codebleu': float(max(scores)),
            'min_codebleu': float(min(scores)),
            'median_codebleu': float(sorted(scores)[len(scores)//2]) if len(scores) > 0 else 0.0,
            'std_codebleu': calculate_std(scores) if len(scores) > 1 else 0.0
        }
    
    print(f"\nMBPP Results:")
    print(f"  Successful: {results['successful_evaluations']}")
    print(f"  Failed: {results['failed_evaluations']}")
    if scores:
        print(f"  Mean CodeBLEU: {results['summary_stats']['mean_codebleu']:.4f}")
        print(f"  Max CodeBLEU: {results['summary_stats']['max_codebleu']:.4f}")
        print(f"  Min CodeBLEU: {results['summary_stats']['min_codebleu']:.4f}")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error_type, error_list in results['errors'].items():
            print(f"  {error_type}: {len(error_list)} cases")
            if len(error_list) <= 3:
                for err in error_list:
                    print(f"    - {err}")
    
    return results


def calculate_humaneval_codebleu(
    generations_path: str,
    dataset_path: str,
    output_path: str = None
) -> Dict:
    """
    Calculate CodeBLEU for HumanEval dataset with proper mapping
    """
    print(f"\n{'='*70}")
    print("CALCULATING CodeBLEU for HumanEval Dataset")
    print(f"{'='*70}")
    
    # Load data
    print("Loading HumanEval dataset...")
    problems = load_humaneval_dataset(dataset_path)
    print(f"  Loaded {len(problems)} problems")
    
    print("Loading generations...")
    if os.path.exists(generations_path):
        generations = load_generations(generations_path)
        print(f"  Loaded {len(generations)} generations")
    else:
        print(f"  WARNING: {generations_path} does not exist or is empty")
        return {
            'dataset': 'humaneval',
            'status': 'file_not_found',
            'message': f'Generations file not found: {generations_path}'
        }
    
    # Check if generations is empty
    if not generations or len(generations) == 0:
        print("  WARNING: Generations file is empty")
        return {
            'dataset': 'humaneval',
            'status': 'empty_file',
            'message': 'Generations file is empty'
        }
    
    results = {
        'dataset': 'humaneval',
        'total_samples': len(generations),
        'successful_evaluations': 0,
        'failed_evaluations': 0,
        'codebleu_scores': {},
        'summary_stats': {},
        'errors': defaultdict(list)
    }
    
    # Map generations to problems using index
    scores = []
    
    for idx, generation_list in enumerate(generations):
        task_id = idx
        
        # Get corresponding problem
        if task_id not in problems:
            results['failed_evaluations'] += 1
            results['errors']['missing_problem'].append(f"Task {task_id} not in dataset")
            continue
        
        problem = problems[task_id]
        reference_code = extract_reference_code(problem, 'humaneval')
        
        if not reference_code:
            results['failed_evaluations'] += 1
            results['errors']['missing_reference'].append(f"Task {task_id} has no reference code")
            continue
        
        # generation_list contains multiple generations for the same task
        if not generation_list or not generation_list[0]:
            results['failed_evaluations'] += 1
            results['errors']['empty_generation'].append(f"Task {task_id} has empty generation")
            continue
        
        generated_code = generation_list[0]  # Use first generation
        
        try:
            # Extract pure code from generation
            extracted_code = extract_code_from_generation(generated_code)
            
            if not extracted_code:
                results['failed_evaluations'] += 1
                results['errors']['extraction_failed'].append(f"Task {task_id} - code extraction failed")
                continue
            
            # Calculate CodeBLEU
            score, metrics = calculate_codebleu_score(extracted_code, reference_code)
            
            results['codebleu_scores'][task_id] = {
                'score': score,
                'metrics': metrics,
                'entry_point': problem.get('entry_point', ''),
                'reference_length': len(reference_code),
                'generated_length': len(extracted_code)
            }
            
            scores.append(score)
            results['successful_evaluations'] += 1
            
            if (task_id + 1) % 20 == 0:
                print(f"  Processed {task_id + 1}/{len(generations)} samples...")
        
        except Exception as e:
            results['failed_evaluations'] += 1
            results['errors']['evaluation_error'].append(f"Task {task_id}: {str(e)}")
    
    # Calculate summary statistics
    if scores:
        results['summary_stats'] = {
            'mean_codebleu': float(sum(scores) / len(scores)),
            'max_codebleu': float(max(scores)),
            'min_codebleu': float(min(scores)),
            'median_codebleu': float(sorted(scores)[len(scores)//2]) if len(scores) > 0 else 0.0,
            'std_codebleu': calculate_std(scores) if len(scores) > 1 else 0.0
        }
    
    print(f"\nHumanEval Results:")
    print(f"  Successful: {results['successful_evaluations']}")
    print(f"  Failed: {results['failed_evaluations']}")
    if scores:
        print(f"  Mean CodeBLEU: {results['summary_stats']['mean_codebleu']:.4f}")
        print(f"  Max CodeBLEU: {results['summary_stats']['max_codebleu']:.4f}")
        print(f"  Min CodeBLEU: {results['summary_stats']['min_codebleu']:.4f}")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error_type, error_list in results['errors'].items():
            print(f"  {error_type}: {len(error_list)} cases")
            if len(error_list) <= 3:
                for err in error_list:
                    print(f"    - {err}")
    
    return results


def calculate_std(values: List[float]) -> float:
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def main():
    """Main execution function"""
    
    base_path = Path('/home/fahad/Documents/PROJECTS/promptmark')
    
    # MBPP paths
    mbpp_gen_path = base_path / 'output/baseline_results/mbpp/generations.json'
    mbpp_dataset_path = base_path / 'datasets/sanitized-mbpp.json'
    mbpp_output_path = base_path / 'output/baseline_results/mbpp/codebleu_scores.json'
    
    # HumanEval paths
    humaneval_gen_path = base_path / 'output/baseline_results/humaneval_gen3/generations.json'
    humaneval_dataset_path = base_path / 'datasets/humaneval_164.jsonl'
    humaneval_output_path = base_path / 'output/baseline_results/humaneval_gen3/codebleu_scores.json'
    
    # Verify files exist
    print("\nVerifying file paths...")
    for path, name in [
        (mbpp_gen_path, 'MBPP generations'),
        (mbpp_dataset_path, 'MBPP dataset'),
        (humaneval_dataset_path, 'HumanEval dataset')
    ]:
        if path.exists():
            size_mb = path.stat().st_size / (1024*1024)
            print(f"  ✓ {name}: {path} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {name}: {path} NOT FOUND")
    
    # Calculate CodeBLEU for both datasets
    all_results = {}
    
    # MBPP
    mbpp_results = calculate_mbpp_codebleu(
        str(mbpp_gen_path),
        str(mbpp_dataset_path),
        str(mbpp_output_path)
    )
    all_results['mbpp'] = mbpp_results
    
    # HumanEval
    humaneval_results = calculate_humaneval_codebleu(
        str(humaneval_gen_path),
        str(humaneval_dataset_path),
        str(humaneval_output_path)
    )
    all_results['humaneval'] = humaneval_results
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    results_path = base_path / 'output/codebleu_evaluation_complete.json'
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
    
    # Print summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    if 'summary_stats' in mbpp_results and mbpp_results['summary_stats']:
        print("\nMBPP Summary:")
        for key, value in mbpp_results['summary_stats'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    
    if 'summary_stats' in humaneval_results and humaneval_results['summary_stats']:
        print("\nHumanEval Summary:")
        for key, value in humaneval_results['summary_stats'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()
