"""
Experiment: Static Watermarking During Generation (expS)
Applies watermarking by encouraging green-letter identifiers during code generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import ast
import re
import math
from datetime import datetime
from scipy.stats import norm
from collections import defaultdict

from shared_utils import (
    get_red_green_sets, extract_code_from_response, generate_response,
    _get_tests_from_record, test_code, fix_method_name, CodeNavigator,
    load_frequency_data, calculate_gamma, detect_watermark
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NUMBER = "expS"
EXP_VERSION = "v1"
GENERATION_TYPE = "during"
DATASET = "humaneval"

# ============================================================================
# CONTROL CONSTANTS
# ============================================================================

COMMENT_ENABLED = False           # No comment watermarking
CHECK_CORRECTNESS = True          # Evaluate correctness
APPLY_WATERMARKING = True         # YES - apply watermarking
ITERATIVE_MODE = False             # One-shot generation
MINIMAL_TOKEN_CHECK = False
TRIVIAL_APPROACH = False

# Watermark parameters
Z_THRESHOLD = 2.12
P_THRESHOLD = norm.sf(Z_THRESHOLD)
SEED_KEY = "exp2025"
SMALL_SAMPLE_THRESHOLD = 30
N_MIN_TOKENS = 5
ITER_CAP = 5


# ============================================================================
# SYSTEM PROMPT AND PROBLEM TEMPLATE
# ============================================================================

SYSTEM_PROMPT = '''
### Green Letter List: {green_words}
### Red Letter List: {red_words}

### Command:
Generate code following the given instructions:
    1. Green Letter Enriched Identifier: When generating identifiers (local variables, function parameters, private helper functions, internal class attributes, and temporary variables) prefer those starting with letters from the 'Green Letter List'. Use them naturally and consistently.
    2. Correct & Relevant: Generate correct code following the problem statement.
    2. Avoiding Instruction: Do not add docstrings. Add brief comments only to clarify complex logic, but do not over-comment. Reduce the use of Red List letters.
    3. Important: Write the method named according to the given test case.
    4. Warning: Do not mention or explain the renaming rules in your output.

### Example Identifier names:
    - Preferred (Green List): answer, count, index, value, sum, key, item, name, word, var, input, output, obj, attr, param, arg, var1, var2, temp_var, helper
    - To Avoid (Red List): result, temp, data, list, flag, ptr, elem, hash, dict, res, tmp, dat, lst, flg, p, el, h, d
'''

PROBLEM_TEMPLATE = (
    "You are a helpful code assistant who can teach a junior developer how to code. Your language of choice is Python. Only generate the Python code for the following task enclosed in ```python```\n\n"
    "##Prompt:\n{prompt}\n\n"
    "##Test Cases:\n{tests}\n\n"
)


# ============================================================================
# CODE GENERATION
# ============================================================================

def generate_code(record, feedback_prompt=""):
    """Generate code using the configured method."""
    task_id = record["task_id"]
    problem_query = record["prompt"]
    
    # Extract test cases
    test_imports, tests = _get_tests_from_record(record)
    testcases = "\n".join(tests) if tests else ""
    full_prompt = PROBLEM_TEMPLATE.format(prompt=problem_query, tests=testcases)

    # Get green/red letters for this experiment
    green_letters, red_letters, _ = get_red_green_sets(secret_key=SEED_KEY, base_dir=".")
    system_instruction = SYSTEM_PROMPT.format(
        green_words=sorted(green_letters), red_words=sorted(red_letters)
    )

    if len(feedback_prompt) > 0:
        full_prompt = full_prompt + "\n\n" + feedback_prompt

    full_prompt_with_system = f"{system_instruction}\n\n{full_prompt}"

    print(f"FULL PROMPT (first 500 chars): {full_prompt_with_system[:500]}...\n")

    result = generate_response(full_prompt_with_system, max_tokens=2048, track_tokens=True)

    full_text = result["text"].strip()
    code_text, explanation_text = extract_code_from_response(full_text)

    return {
        "code": code_text,
        "explanation": explanation_text,
        "full_response": full_text,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "total_tokens": result["total_tokens"],
    }


# ============================================================================
# CODE EVALUATION
# ============================================================================

CURRENT_GREEN_SET_SIZE = 0


def evaluate_candidate(record, generated_code):
    """Evaluate generated code for correctness and watermark fidelity."""
    global CURRENT_GREEN_SET_SIZE

    task_id = record["task_id"]
    
    # Check correctness
    if CHECK_CORRECTNESS:
        test_imports, tests = _get_tests_from_record(record)
        
        if "test_list" in record:
            generated_code = fix_method_name(generated_code, tests)
        
        test_result = test_code(generated_code, test_imports, tests, timeout=2)
        passed, failed = test_result.get("result", (0, 0))
    else:
        passed, failed = 0, 0
        test_result = {"error": "Correctness check disabled"}

    is_correct = (failed == 0) if CHECK_CORRECTNESS else None
    total_tests = passed + failed if CHECK_CORRECTNESS else 0
    pass_rate = (passed / total_tests * 100.0) if (CHECK_CORRECTNESS and total_tests > 0) else 0.0
    all_passed = (failed == 0) if CHECK_CORRECTNESS else None

    eval_res = {
        "task_id": task_id,
        "tests_passed": passed,
        "tests_failed": failed,
        "total_tests": total_tests,
        "pass_rate": pass_rate,
        "all_passed": all_passed if CHECK_CORRECTNESS else False,
        "correctness": is_correct if CHECK_CORRECTNESS else None,
        "error_message": test_result.get("error", ""),
    }

    # Calculate watermark metrics
    green_letters, red_letters, size_g = get_red_green_sets(secret_key=SEED_KEY, base_dir=".")
    CURRENT_GREEN_SET_SIZE = size_g

    original_code = record.get("canonical_solution", "") or record.get("code", "")
    letter_freqs, total_identifiers = load_frequency_data(green_letters, ".")
    GAMMA = calculate_gamma(letter_freqs, total_identifiers, green_letters)

    try:
        tree = ast.parse(generated_code)
        visitor = CodeNavigator()
        visitor.visit(tree)
        estimated_tokens = len(visitor.non_public_vars | visitor.non_public_funcs | visitor.non_public_classes)
    except:
        estimated_tokens = 0

    # Perform watermark detection
    detection_result = detect_watermark(
        original_code, generated_code, green_letters, red_letters, GAMMA, 
        comment_enabled=COMMENT_ENABLED
    )

    eval_res.update(detection_result)
    eval_res["meets_z"] = bool(detection_result.get("generated_is_watermarked", False))

    return eval_res


# ============================================================================
# GENERATION LOOP
# ============================================================================

def run_phase1(record, max_iterations=ITER_CAP, verbose=False):
    """Generate code with watermarking (one-shot for this method)."""
    print(f"\n{'='*60}\n[Generation]\n{'='*60}")

    gen = generate_code(record, feedback_prompt="")
    code = gen["code"]
    reasoning_text = gen.get("explanation", "")

    eval_res = evaluate_candidate(record, code)
    eval_res.update({
        "iteration": 0,
        "reasoning_text": reasoning_text,
        "full_llm_response": gen["full_response"],
        "input_tokens": gen["input_tokens"],
        "output_tokens": gen["output_tokens"],
        "code": code,
        "stopping_condition_met": True,
    })

    print(f"\n📊 EVALUATION METRICS:")
    print(f"   Correctness: {eval_res['correctness']}")
    print(f"   Tests Passed: {eval_res['tests_passed']}/{eval_res['tests_passed'] + eval_res['tests_failed']}")
    print(f"   Generated Token Count: {eval_res['generated_token_count']}")
    print(f"   Generated Green Count: {eval_res['generated_green_count']}")
    print(f"   P-Value (p_exact): {eval_res['generated_p_exact']:.10f}")
    print(f"   Meets Watermark: {eval_res['meets_z']}")
    print(f"   Stopping Condition Met: {eval_res['stopping_condition_met']}\n")

    token_tracking = {
        "total_input_tokens": gen["input_tokens"],
        "total_output_tokens": gen["output_tokens"],
        "total_tokens": gen["total_tokens"],
        "iterations": [{
            "iteration": 0,
            "input_tokens": gen["input_tokens"],
            "output_tokens": gen["output_tokens"],
            "total_tokens": gen["total_tokens"],
        }],
    }

    return (eval_res, token_tracking)


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def process_dataset(df, output_dir, base_dir="."):
    """Process dataset and save results."""
    Path(output_dir).parent.mkdir(exist_ok=True)
    results = []
    all_token_tracking = []
    total_exp_input_tokens = 0
    total_exp_output_tokens = 0

    for idx, record in df.iterrows():
        task_id = record.get("task_id") or record.get("id")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            sel, token_info = run_phase1(record, max_iterations=ITER_CAP, verbose=True)
            code = sel.get("code", "") if sel else ""
            iteration_used = sel.get("iteration") if sel and "iteration" in sel else None

            # Save code
            output_file = out_dir / f"{task_id}.py"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(code or "")
            
            # Track tokens
            total_exp_input_tokens += token_info.get("total_input_tokens", 0)
            total_exp_output_tokens += token_info.get("total_output_tokens", 0)
            
            # Store results
            result_row = {
                "task_id": task_id,
                "iteration_used": iteration_used,
                **sel
            }
            results.append(result_row)
            all_token_tracking.append({
                "task_id": task_id,
                **token_info
            })
            
        except Exception as e:
            print(f"❌ Error processing {task_id}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(dataset_path, output_dir, results_csv, sample_size=100):
    """Main entry point for the experiment."""
    print(f"🚀 Starting {EXPERIMENT_NUMBER} ({DATASET}) Experiment")
    print(f"   Watermarking: {APPLY_WATERMARKING}")
    print(f"   Comments: {COMMENT_ENABLED}")
    print(f"   Iterative: {ITERATIVE_MODE}")
    print(f"   Check Correctness: {CHECK_CORRECTNESS}\n")

    # Load dataset
    df = pd.read_json(dataset_path, lines=True)
    n = min(sample_size, len(df))
    seed = int.from_bytes(__import__('hashlib').sha256(SEED_KEY.encode()).digest()[:4], "big")
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    print(f"Dataset: {len(df)} samples")

    # Process dataset
    results = process_dataset(df, output_dir)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Results saved to {results_csv}")

    return results_df


if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/humaneval_164.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else f"output/claude_{EXPERIMENT_NUMBER}_{GENERATION_TYPE}_gen_{EXP_VERSION}_100_{DATASET}"
    results_csv = sys.argv[3] if len(sys.argv) > 3 else f"results/raw/claude_{EXPERIMENT_NUMBER}_{GENERATION_TYPE}_gen_{EXP_VERSION}_100_{DATASET}.csv"
    sample_size = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    main(dataset_path, output_dir, results_csv, sample_size)
