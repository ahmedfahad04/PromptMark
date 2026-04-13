"""
Experiment: Refactoring-Based Watermarking (Post-Hoc) (expX)
First generates correct code, then refactors it to embed watermarks through identifier renaming.
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

EXPERIMENT_NUMBER = "expX"
EXP_VERSION = "v1"
GENERATION_TYPE = "only-gen"  # Two-phase: generate then refactor
DATASET = "humaneval"

# ============================================================================
# CONTROL CONSTANTS
# ============================================================================

COMMENT_ENABLED = False           # No comments in refactoring
CHECK_CORRECTNESS = True          # Evaluate correctness
APPLY_WATERMARKING = True         # YES - apply watermarking in phase 2
ITERATIVE_MODE = False             # One-shot per phase
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
# SYSTEM PROMPTS AND PROBLEM TEMPLATES
# ============================================================================

# Phase 1: Generate code without watermarking
GENERATION_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Generate clean, correct code that solves the given problem.\n"
    "Write the solution within ```python``` delimiters.\n"
)

GENERATION_PROBLEM_TEMPLATE = (
    "##Prompt:\n{prompt}\n\n"
)

# Phase 2: Refactor code to add watermark
REFACTORING_SYSTEM_PROMPT = '''
### Green Letter List: {green_words}
### Red Letter List: {red_words}

### Command:
Refactor the provided code following these instructions:
    1. Green Letter Enriched Identifier: Rename identifiers (local variables, function parameters, private helper functions, internal class attributes, and temporary variables) to prefer those starting with letters from the 'Green Letter List'. Use them naturally and consistently, avoiding Red List letters where possible.
    2. Correct & Relevant: Do not change the functionality of the code. Ensure the refactored code is correct and passes the given test cases.
    3. Avoiding Instruction: Do not add docstrings. Add brief comments only to clarify complex logic, but do not over-comment.
    4. Important: Keep the method names as per the test cases, but refactor internal identifiers.
    5. Warning: Do not mention or explain the renaming rules in your output.

### Example Identifier names:
    - Preferred (Green List): answer, count, index, value, sum, key, item, name, word, var, input, output, obj, attr, param, arg, var1, var2, temp_var, helper
    - To Avoid (Red List): result, temp, data, list, flag, ptr, elem, hash, dict, res, tmp, dat, lst, flg, p, el, h, d
'''

REFACTORING_PROBLEM_TEMPLATE = (
    "You are an expert Python developer. Refactor the following Python code by modifying identifier names, "
    "ensuring it remains functional and passes ALL the test cases. Only output the refactored Python code enclosed in ```python```\n\n"
    "##Original Code:\n{code}\n\n"
    "##Test Cases:\n{tests}\n\n"
)


# ============================================================================
# CODE GENERATION - PHASE 1
# ============================================================================

def generate_code_phase1(record):
    """Phase 1: Generate code without watermarking constraints."""
    task_id = record["task_id"]
    problem_query = record["prompt"]
    full_prompt = GENERATION_PROBLEM_TEMPLATE.format(prompt=problem_query)
    full_prompt_with_system = f"{GENERATION_SYSTEM_PROMPT}\n\n{full_prompt}"

    print(f"[PHASE 1] Generating code without watermarking...\n")
    
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
# CODE REFACTORING - PHASE 2
# ============================================================================

def refactor_code_phase2(record, generated_code):
    """Phase 2: Refactor generated code to add watermarks."""
    task_id = record["task_id"]
    
    # Get test cases
    test_imports, tests = _get_tests_from_record(record)
    testcases = "\n".join(tests) if tests else ""
    
    # Get green/red letters
    green_letters, red_letters, _ = get_red_green_sets(secret_key=SEED_KEY, base_dir=".")
    system_instruction = REFACTORING_SYSTEM_PROMPT.format(
        green_words=sorted(green_letters), red_words=sorted(red_letters)
    )
    
    full_prompt = REFACTORING_PROBLEM_TEMPLATE.format(
        code=generated_code, tests=testcases
    )
    full_prompt_with_system = f"{system_instruction}\n\n{full_prompt}"

    print(f"[PHASE 2] Refactoring code to add watermarking...\n")
    
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
# TWO-PHASE GENERATION
# ============================================================================

def run_two_phase_generation(record):
    """Run both phases: generate code, then refactor with watermarking."""
    print(f"\n{'='*60}\n[Task: {record.get('task_id')}]\n{'='*60}")

    # Phase 1: Generate correct code
    gen1 = generate_code_phase1(record)
    code1 = gen1["code"]
    
    # Evaluate phase 1 code
    print(f"[PHASE 1] Evaluating generated code...")
    eval1 = evaluate_candidate(record, code1)
    
    if not eval1["correctness"]:
        print(f"⚠️  Phase 1 code is not correct. Using as-is for refactoring.\n")
    else:
        print(f"✅ Phase 1 code is correct.\n")

    # Phase 2: Refactor the code to add watermarks
    gen2 = refactor_code_phase2(record, code1)
    code2 = gen2["code"]
    
    # Evaluate phase 2 code
    print(f"[PHASE 2] Evaluating refactored code...")
    eval2 = evaluate_candidate(record, code2)
    
    if not eval2["correctness"]:
        print(f"⚠️  Phase 2 code lost correctness. Using phase 1 code.\n")
        final_code = code1
        final_eval = eval1
    else:
        print(f"✅ Phase 2 code is correct.\n")
        final_code = code2
        final_eval = eval2

    # Combine token tracking
    total_input = gen1["input_tokens"] + gen2["input_tokens"]
    total_output = gen1["output_tokens"] + gen2["output_tokens"]

    token_tracking = {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "phase1_input": gen1["input_tokens"],
        "phase1_output": gen1["output_tokens"],
        "phase2_input": gen2["input_tokens"],
        "phase2_output": gen2["output_tokens"],
        "iterations": [],
    }

    # Prepare result
    result = {
        **final_eval,
        "code": final_code,
        "phase": "2" if final_code == code2 else "1",
        "input_tokens": final_eval.get("input_tokens", total_input),
        "output_tokens": final_eval.get("output_tokens", total_output),
        "stopping_condition_met": True,
    }

    # Print summary
    print(f"\n📊 FINAL EVALUATION:")
    print(f"   Phase Used: {result['phase']}")
    print(f"   Correctness: {result['correctness']}")
    print(f"   Tests Passed: {result['tests_passed']}/{result['tests_passed'] + result['tests_failed']}")
    print(f"   Generated P-Value: {result['generated_p_exact']:.10f}")
    print(f"   Meets Watermark: {result['meets_z']}\n")

    return (result, token_tracking)


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
            sel, token_info = run_two_phase_generation(record)
            code = sel.get("code", "") if sel else ""

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
                "iteration_used": None,
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
    print(f"🚀 Starting {EXPERIMENT_NUMBER} ({DATASET}) Experiment (Two-Phase)")
    print(f"   Phase 1: Generate code without watermarking")
    print(f"   Phase 2: Refactor with watermarking\n")

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
