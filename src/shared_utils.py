"""
Shared utilities for all watermarking experiments.
Contains common functions, classes, and configuration.
"""

import pandas as pd
import numpy as np
import re
import math
import ast
import os
import sys
import json
import hashlib
import random
import textwrap
import boto3
import signal
import subprocess
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable, Optional, Union
from datetime import datetime
import builtins
import keyword
import multiprocessing

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)
from scipy.stats import binomtest, norm, binom, chi2

# Import LLM providers for flexible model switching
try:
    from llm_providers import LLMProviderFactory, get_llm_provider
except ImportError:
    # Fallback for development - try relative import
    try:
        from .llm_providers import LLMProviderFactory, get_llm_provider
    except ImportError:
        print("WARNING: llm_providers not found. Falls back to direct Bedrock API.")
        LLMProviderFactory = None
        get_llm_provider = None

# ============================================================================
# COMMON IDENTIFIER EXTRACTION
# ============================================================================

COMMON_STD_METHODS = {
    "self", "re", "cls", "append", "join", "dummy_function", "find", "kwargs",
    "args", "range", "print", "len", "dict", "list", "str", "int", "float",
    "set", "tuple", "os", "np", "subprocess", "now", "today", "timedelta",
    "strptime", "date", "time", "datetime", "logging", "log", "info", "debug",
    "error", "warning", "exception", "lower", "upper", "strip", "split",
    "replace", "endswith", "startswith", "extend", "insert", "remove", "pop",
    "sort", "clear", "keys", "values", "items", "get", "update", "copy",
    "format", "count", "index",
}

BUILTIN_NAMES = set(dir(builtins)).union(COMMON_STD_METHODS)

# ============================================================================
# LLM PROVIDER CONFIGURATION
# ============================================================================

# Global LLM provider instance
_llm_provider = None

# Current provider name (default: "claude")
_current_provider_name = os.getenv("LLM_PROVIDER", "claude").lower()

def initialize_llm_provider(provider_name: str = None) -> None:
    """
    Initialize the global LLM provider.
    
    Args:
        provider_name: Name of provider ("claude", "gemini", etc.)
                      If None, uses LLM_PROVIDER env var or defaults to "claude"
    """
    global _llm_provider, _current_provider_name
    
    if provider_name:
        _current_provider_name = provider_name.lower()
    
    if LLMProviderFactory is None:
        print("WARNING: LLM providers not available. Using fallback mode.")
        return
    
    try:
        _llm_provider = LLMProviderFactory.create(_current_provider_name)
        if _llm_provider.validate_connection():
            print(f"✓ LLM provider '{_current_provider_name}' initialized successfully")
        else:
            print(f"⚠ LLM provider '{_current_provider_name}' initialized but connection validation failed")
    except Exception as e:
        print(f"✗ Failed to initialize LLM provider '{_current_provider_name}': {e}")
        _llm_provider = None

def get_current_llm_provider():
    """Get the current global LLM provider instance."""
    global _llm_provider
    if _llm_provider is None:
        initialize_llm_provider()
    return _llm_provider

def set_llm_provider(provider_name: str) -> None:
    """
    Switch to a different LLM provider at runtime.
    
    Args:
        provider_name: Name of provider to switch to
    """
    global _current_provider_name
    print(f"Switching LLM provider from '{_current_provider_name}' to '{provider_name}'...")
    initialize_llm_provider(provider_name)


class CodeNavigator(ast.NodeVisitor):
    """Extract identifiers from Python code."""
    
    def __init__(self):
        self.public_classes = set()
        self.non_public_classes = set()
        self.public_funcs = set()
        self.non_public_funcs = set()
        self.public_vars = set()
        self.non_public_vars = set()
        self.docstrings = []

    def visit_FunctionDef(self, node):
        name = node.name
        if name.startswith("__") and name.endswith("__"):
            pass
        elif name.startswith("_"):
            self.non_public_funcs.add(name)
        else:
            self.public_funcs.add(name)

        for arg in node.args.args:
            if arg.arg not in BUILTIN_NAMES:
                self.non_public_vars.add(arg.arg)

        doc = ast.get_docstring(node)
        if doc:
            self.docstrings.append({
                "type": "function_docstring",
                "name": node.name,
                "line_number": node.lineno,
                "content": doc,
            })
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if node.name.startswith("_"):
            self.non_public_classes.add(node.name)
        else:
            self.public_classes.add(node.name)
        self.non_public_vars.add(node.name)

        doc = ast.get_docstring(node)
        if doc:
            self.docstrings.append({
                "type": "class_docstring",
                "name": node.name,
                "line_number": node.lineno,
                "content": doc,
            })
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id not in BUILTIN_NAMES:
                if target.id.isupper():
                    self.public_vars.add(target.id)
                else:
                    self.non_public_vars.add(target.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr not in BUILTIN_NAMES and node.attr not in COMMON_STD_METHODS:
                self.public_funcs.add(node.attr)
        elif node.attr not in BUILTIN_NAMES and node.attr not in COMMON_STD_METHODS:
            self.non_public_vars.add(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id not in BUILTIN_NAMES:
            self.non_public_vars.add(node.id)
        self.generic_visit(node)

    def visit_Module(self, node):
        doc = ast.get_docstring(node)
        if doc:
            self.docstrings.append({
                "type": "module_docstring",
                "name": "__main__",
                "line_number": getattr(node, "lineno", 1),
                "content": doc,
            })
        self.generic_visit(node)


def get_tokens(code):
    """Extract tokens from code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    visitor = CodeNavigator()
    visitor.visit(tree)

    all_tokens = (
        visitor.public_classes
        | visitor.non_public_funcs
        | visitor.non_public_vars
        | visitor.non_public_classes
    )

    cleaned_tokens = {
        t for t in all_tokens if t not in COMMON_STD_METHODS and t not in BUILTIN_NAMES
    }

    return cleaned_tokens


# ============================================================================
# GREEN/RED SET GENERATION
# ============================================================================

def get_frequent_candidates(
    humaneval_freq_file: str, mbpp_freq_file: str, top_n: int = 18
) -> List[str]:
    """Extract most frequent characters from both datasets combined."""
    freq_sum = defaultdict(int)

    for filepath in [humaneval_freq_file, mbpp_freq_file]:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                for char, count in data["letter_freqs"].items():
                    freq_sum[char] += count
        else:
            print(f"⚠️ Warning: Frequency file not found: {filepath}")

    sorted_chars = sorted(freq_sum.items(), key=lambda x: x[1], reverse=True)
    return [char for char, _ in sorted_chars[:top_n]]


def build_green_set(
    secret_key: str, candidate_letters: List[str], g_min: int = 8, g_max: int = 18
) -> Tuple[set, int]:
    """Construct green-set from secret key and candidate letters."""
    if len(candidate_letters) < g_max:
        raise ValueError(
            f"Candidate set size ({len(candidate_letters)}) must be >= g_max ({g_max})"
        )
    if g_min < 1 or g_max > len(candidate_letters) or g_min > g_max:
        raise ValueError(
            f"Invalid bounds: g_min={g_min}, g_max={g_max}, |C|={len(candidate_letters)}"
        )

    seed_hash = int(hashlib.sha256(secret_key.encode()).hexdigest(), 16)
    seed_hash_mod = seed_hash % 11
    green_set_size = g_min + seed_hash_mod

    rng = random.Random(seed_hash)
    shuffled = list(candidate_letters)
    rng.shuffle(shuffled)

    green_set = set(shuffled[:green_set_size])

    return green_set, green_set_size


def get_red_green_sets(
    secret_key: str,
    base_dir: str = None,
    g_min: int = 8,
    g_max: int = 18
) -> Tuple[set, set, int]:
    """
    Get green and red letter sets from secret key.
    
    Args:
        secret_key: Secret key for deterministic generation
        base_dir: Base directory for frequency files (optional)
        g_min: Minimum green set size
        g_max: Maximum green set size
    
    Returns:
        Tuple of (green_letters, red_letters, green_set_size)
    """
    # Try to use frequency-based candidate selection if base_dir provided
    if base_dir and os.path.exists(base_dir):
        try:
            he_freq_path = os.path.join(base_dir, "results/dataset/humaneval_letter_freqs.json")
            mbpp_freq_path = os.path.join(base_dir, "results/dataset/mbpp_letter_freqs.json")
            candidate_letters = get_frequent_candidates(he_freq_path, mbpp_freq_path, top_n=18)
        except Exception:
            candidate_letters = list("abcdefghijklmnopqrstuvwxyz")
    else:
        candidate_letters = list("abcdefghijklmnopqrstuvwxyz")
    
    green_letters, green_set_size = build_green_set(secret_key, candidate_letters, g_min=g_min, g_max=g_max)
    all_letters = set("abcdefghijklmnopqrstuvwxyz")
    red_letters = all_letters - green_letters

    return green_letters, red_letters, green_set_size


def load_problem_set(
    dataset_path: str,
    num_problems: int = 50,
    file_format: str = None
) -> List[Dict[str, str]]:
    """
    Load problems from a dataset file (JSONL or JSON).
    
    Args:
        dataset_path: Path to dataset file (.jsonl or .json)
        num_problems: Number of problems to load (default: load all)
        file_format: Force format ('jsonl' or 'json'). Auto-detect if None.
    
    Returns:
        List of problem dicts with 'prompt' and 'tests' keys
    """
    problems = []
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Detect format if not specified
    if file_format is None:
        file_format = "jsonl" if dataset_path.endswith(".jsonl") else "json"
    
    if file_format == "jsonl":
        with open(dataset_path) as f:
            for i, line in enumerate(f):
                if i >= num_problems:
                    break
                try:
                    data = json.loads(line.strip())
                    # Normalize to expected format
                    problems.append({
                        "prompt": data.get("prompt", ""),
                        "tests": data.get("test", data.get("tests", "")),
                        **{k: v for k, v in data.items() if k not in ["prompt", "test", "tests"]}
                    })
                except json.JSONDecodeError:
                    continue
    
    elif file_format == "json":
        with open(dataset_path) as f:
            data = json.load(f)
        
        # Handle list format
        if isinstance(data, list):
            for i, item in enumerate(data):
                if i >= num_problems:
                    break
                problems.append({
                    "prompt": item.get("prompt", ""),
                    "tests": item.get("test", item.get("tests", "")),
                    **{k: v for k, v in item.items() if k not in ["prompt", "test", "tests"]}
                })
        
        # Handle dict format (keyed by problem ID)
        elif isinstance(data, dict):
            for i, (key, item) in enumerate(data.items()):
                if i >= num_problems:
                    break
                if isinstance(item, dict):
                    problems.append({
                        "prompt": item.get("prompt", ""),
                        "tests": item.get("test", item.get("tests", "")),
                        **{k: v for k, v in item.items() if k not in ["prompt", "test", "tests"]}
                    })
    
    if not problems:
        raise ValueError(f"No valid problems loaded from {dataset_path}")
    
    return problems[:num_problems] if num_problems > 0 else problems


# ============================================================================
# COMMENT EXTRACTION
# ============================================================================

def extract_comments_from_source(source_code: str) -> list:
    """Extract comments from source code."""
    comments = []
    lines = source_code.split("\n")

    for line_number, line in enumerate(lines, start=1):
        hash_index = line.find("#")
        if hash_index != -1:
            comment_content = line[hash_index + 1:].strip()
            if comment_content:
                code_before_hash = line[:hash_index].strip()
                comment_type = "inline_comment" if code_before_hash else "full_line_comment"
                comments.append({
                    "line_number": line_number,
                    "content": comment_content,
                    "type": comment_type,
                    "code_context": (
                        code_before_hash[:50] + "..."
                        if len(code_before_hash) > 50
                        else code_before_hash
                    ),
                })

    tree = ast.parse(source_code)
    visitor = CodeNavigator()
    visitor.visit(tree)
    comments.extend(visitor.docstrings)

    return comments


def get_comment_starting_letters(source_code: str, gamma: float) -> tuple:
    """Extract starting letters from comments."""
    try:
        comments = extract_comments_from_source(source_code)
        starting_letters = []
        relevant_words = set()

        for comment in comments:
            comment_lines = comment["content"].split("\n")
            for line in comment_lines:
                line = line.strip()
                if not line:
                    continue

                words = re.findall(r"\b[a-zA-Z]+\b", line)
                if words:
                    first_word = words[0].lower()
                    relevant_words.add(first_word)

                    if first_word:
                        first_char = first_word[0].lower()
                        if first_char.isalpha():
                            starting_letters.append(first_char)

        return starting_letters, relevant_words, 0, 0.0, 1.0

    except Exception as e:
        print(f"❌ Error extracting comment letters: {type(e).__name__}: {e}")
        return [], set(), 0, 0.0, 1.0


# ============================================================================
# GAMMA CALCULATION
# ============================================================================

def calculate_gamma(letter_counts: Dict, total_count: int, green_letters: set) -> float:
    """Calculate gamma (proportion of green letters)."""
    if total_count == 0:
        return 0.0

    gamma = 0.0
    for letter in green_letters:
        if letter in letter_counts:
            p_letter = letter_counts[letter] / total_count
            gamma += p_letter

    return gamma


# ============================================================================
# TEST EXTRACTION AND EXECUTION
# ============================================================================

def extract_assertions_from_check_function(test_code_str: str, entry_point_name=None) -> List[str]:
    """Extract assertion statements from HumanEval check() function."""
    assertions = []

    try:
        tree = ast.parse(test_code_str)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "check":
                for stmt in node.body:
                    if isinstance(stmt, ast.Assert):
                        assert_code = ast.unparse(stmt)

                        if entry_point_name:
                            assert_code = assert_code.replace('candidate(', f'{entry_point_name}(')

                        assertions.append(assert_code)
                break

        return assertions
    except SyntaxError:
        return []


def _get_tests_from_record(record) -> Tuple[List[str], List[str]]:
    """Return (test_imports, tests_list) from a record for both MBPP and HumanEval."""

    if "test_list" in record and record.get("test_list"):
        test_imports = record.get("test_imports", []) or record.get("imports", []) or []
        tests_list = record.get("test_list", []) or record.get("tests", []) or []
    elif "test" in record and record.get("test"):
        test_code_str = record.get("test", "")
        test_imports = [
            line.strip()
            for line in test_code_str.split("\n")
            if line.strip().startswith(("import ", "from "))
        ]
        entry_point = record.get("entry_point", None)
        tests_list = extract_assertions_from_check_function(test_code_str, entry_point_name=entry_point)
    else:
        test_imports = []
        tests_list = []

    if isinstance(test_imports, str):
        test_imports = [test_imports]
    if isinstance(tests_list, str):
        tests_list = [tests_list]

    return test_imports, tests_list


def run_code_with_tests(code: str, test_imports: List[str], tests: List[str], return_dict: Dict) -> Dict:
    """Execute code with test assertions and track results."""
    try:
        env = {}

        for imp in test_imports:
            exec(imp, env, env)

        exec(code, env, env)

        passed, failed = 0, 0
        return_dict["error"] = ""

        for t in tests:
            try:
                exec(t, env, env)
                passed += 1
            except AssertionError:
                failed += 1
                return_dict["error"] += f"Assertion Error in: {t}\n"
            except Exception as e:
                failed += 1
                return_dict["error"] += f"Exception Error in: {t} → {e}\n"

        return_dict["result"] = (passed, failed)

    except Exception as e:
        tb = traceback.format_exc()
        return_dict["error"] = f"Runtime Error in user code:\n{tb}"

    return return_dict


def safe_exec_with_tests(code: str, test_imports: List[str], tests: List[str], timeout: int = 2) -> Dict:
    """Execute code with tests using multiprocessing for timeout handling."""
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(
        target=run_code_with_tests, args=(code, test_imports, tests, return_dict)
    )

    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        return_dict["result"] = (0, len(tests))
        return_dict["error"] = "Timeout: possible infinite loop or hanging code"
        return return_dict

    if "error" in return_dict:
        return return_dict

    return return_dict


def test_code(code: str, test_imports: List[str], tests: List[str], timeout: int = 2) -> Dict:
    """Test generated code against test cases."""
    return safe_exec_with_tests(code, test_imports, tests, timeout=timeout)


# ============================================================================
# WATERMARK DETECTION
# ============================================================================

def calculate_z_score(token_count: int, green_count: int, gamma: float) -> float:
    """Calculate z-score for green token count."""
    if token_count == 0 or gamma <= 0 or gamma >= 1:
        return float("nan")
    return (green_count - gamma * token_count) / math.sqrt(gamma * (1 - gamma) * token_count)


def calculate_p_value_exact(green_count: int, token_count: int, gamma: float) -> float:
    """Calculate exact binomial p-value."""
    if token_count == 0:
        return float("nan")
    return binomtest(green_count, token_count, gamma, alternative="greater").pvalue


def get_unified_detection_score(token_count: int, green_count: int, gamma: float) -> Dict:
    """Calculate unified detection score using exact binomial p-value."""
    p_exact = binom.sf(green_count - 1, token_count, gamma)
    score = -np.log10(np.clip(p_exact, 1e-300, 1.0))

    return {
        "p_exact": p_exact,
        "score": score,
        "token_count": token_count,
    }


def extract_function_names_from_code(code: str) -> List[str]:
    """Extract all function names defined in the user code."""
    try:
        tree = ast.parse(code)
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    except:
        return []


def extract_function_name_from_tests(test_list: List[str]) -> Optional[str]:
    """Extract the function name used in assert statements."""
    for test in test_list:
        try:
            tree = ast.parse(test)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for arg in node.args:
                        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                            return arg.func.id
                    if isinstance(node.func, ast.Name):
                        return node.func.id
        except SyntaxError:
            continue
    return None


def replace_function_name(code: str, old_name: str, new_name: str) -> str:
    """Safely rename the function in the code using AST."""

    class RenameTransformer(ast.NodeTransformer):
        def __init__(self):
            self.renamed = False

        def visit_FunctionDef(self, node):
            if node.name == old_name:
                node.name = new_name
                self.renamed = True
            return node

    try:
        tree = ast.parse(code)
        tree = RenameTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except:
        return code


def fix_method_name(user_code: str, test_list: List[str]) -> str:
    """If needed, rename user's function to match test case function name."""
    code_func_names = extract_function_names_from_code(user_code)
    test_func_name = extract_function_name_from_tests(test_list)

    if not test_func_name:
        return user_code

    if test_func_name in code_func_names:
        return user_code

    if code_func_names:
        old_name = code_func_names[0]
        updated_code = replace_function_name(user_code, old_name, test_func_name)
        print(f"🔄 Renamed '{old_name}' → '{test_func_name}'")
        return updated_code

    return user_code


# ============================================================================
# LLM GENERATION
# ============================================================================

def generate_response(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    track_tokens: bool = False,
    system_prompt: str = "",
) -> Union[str, Dict]:
    """
    Generate response from LLM using the configured provider.
    
    Supports multiple providers (Claude, Gemini, etc.) through strategy pattern.
    Uses the global LLM provider, which can be switched at runtime.
    
    Args:
        prompt: User prompt/query
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 - 1.0)
        track_tokens: If True, return dict with text and token counts
        system_prompt: Optional system instruction
        
    Returns:
        str: Generated text (if track_tokens=False)
        Dict: {"text": str, "input_tokens": int, "output_tokens": int, "total_tokens": int}
              (if track_tokens=True)
    """
    
    # Try to use LLM provider strategy pattern if available
    if LLMProviderFactory is not None:
        try:
            provider = get_current_llm_provider()
            if provider is not None:
                response = provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                if track_tokens:
                    return {
                        "text": response.text,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "total_tokens": response.input_tokens + response.output_tokens,
                    }
                else:
                    return response.text
        except Exception as e:
            print(f"WARNING: LLM provider failed ({e}), falling back to direct Bedrock API")
    
    # Fallback to direct Bedrock API (original implementation)
    print("Using fallback Bedrock API (not using provider strategy)")
    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        )

        model_id = os.getenv("DEFAULT_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            request_body["system"] = system_prompt

        response = client.invoke_model(
            modelId=model_id, body=json.dumps(request_body), contentType="application/json"
        )

        response_body = json.loads(response["body"].read())
        text = ""
        if "content" in response_body and len(response_body["content"]) > 0:
            text = response_body["content"][0]["text"]

        if track_tokens:
            usage = response_body.get("usage", {})
            return {
                "text": text,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            }
        else:
            return text
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}")


def extract_code_from_response(full_text: str) -> Tuple[str, str]:
    """Extract code and explanation from LLM response."""
    code_blocks = re.findall(r"```python(.*?)```", full_text, re.DOTALL)
    actual_code_blocks = [block.strip() for block in code_blocks if block.strip()]
    code_text = actual_code_blocks[-1] if actual_code_blocks else ""
    explanation_text = re.sub(r"```python.*?```", "", full_text, flags=re.DOTALL).strip()

    return code_text, explanation_text


# ============================================================================
# FREQUENCY DATA LOADING
# ============================================================================

def load_frequency_data(green_letters: set, base_dir: str) -> Tuple[Dict, int]:
    """Load letter frequency data from datasets."""
    he_freq_path = os.path.join(base_dir, "results/dataset/humaneval_letter_freqs.json")
    mbpp_freq_path = os.path.join(base_dir, "results/dataset/mbpp_letter_freqs.json")

    letter_freqs = defaultdict(int)
    total_identifiers = 0

    if os.path.exists(he_freq_path):
        with open(he_freq_path, "r") as f:
            df_he = json.load(f)
            df_he = df_he["letter_freqs"]
            for letter in green_letters:
                letter_freqs[letter] += df_he.get(letter, 0)
            total_identifiers += sum(df_he.values())

    if os.path.exists(mbpp_freq_path):
        with open(mbpp_freq_path, "r") as f:
            df_mbpp = json.load(f)
            df_mbpp = df_mbpp["letter_freqs"]
            for letter in green_letters:
                letter_freqs[letter] += df_mbpp.get(letter, 0)
            total_identifiers += sum(df_mbpp.values())

    return dict(letter_freqs), total_identifiers


# ============================================================================
# WATERMARK DETECTION WITH DEDUPLICATION
# ============================================================================

def detect_watermark(original_code: str, generated_code: str, green_letters: set, red_letters: set, gamma: float, comment_enabled: bool = False) -> Dict:
    """Detect watermark in code with proper deduplication."""

    def get_starting_letters(code: str) -> Dict:
        code = textwrap.dedent(code)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {
                "id_starting_letters": [],
                "id_unique_tokens": set(),
                "id_green_count": 0,
                "id_token_count": 0,
                "comment_starting_letters": [],
                "comment_unique_tokens": set(),
                "comment_green_count": 0,
                "comment_token_count": 0,
                "total_starting_letters": [],
                "total_green_count": 0,
                "total_token_count": 0,
                "overlap_count": 0,
                "p_exact": 1.0,
                "score": 0.0,
                "z_score": 0.0,
            }

        visitor = CodeNavigator()
        visitor.visit(tree)

        non_public_tokens = (
            visitor.non_public_classes
            | visitor.non_public_funcs
            | visitor.non_public_vars
        )

        relevant_tokens = [
            token for token in non_public_tokens if token not in {"self", "cls"}
        ]

        def get_starting_letter(word):
            if not word:
                return None
            if word[0] == "_":
                if len(word) > 1 and word[1].isalpha():
                    return word[1].lower()
                else:
                    return None
            elif word[0].isalpha():
                return word[0].lower()
            else:
                return None

        id_starting_letters = [
            letter
            for word in relevant_tokens
            if (letter := get_starting_letter(word)) is not None
        ]

        id_green_count = sum(1 for letter in id_starting_letters if letter in green_letters)
        id_token_count = len(id_starting_letters)
        id_unique_tokens = set(relevant_tokens)

        comment_data = {}
        comment_starting_letters = []
        comment_unique_tokens = set()
        comment_green_count = 0

        if comment_enabled:
            cmnt_starting_letters, cmn_relevant_words, _, _, _ = get_comment_starting_letters(code, gamma)
            comment_starting_letters = cmnt_starting_letters
            comment_unique_tokens = set(cmn_relevant_words)
            comment_green_count = sum(1 for letter in comment_starting_letters if letter in green_letters)

        comment_token_count = len(comment_starting_letters)

        overlap_tokens = id_unique_tokens & comment_unique_tokens
        total_starting_letters = id_starting_letters + comment_starting_letters
        total_green_count = id_green_count + comment_green_count
        total_token_count = id_token_count + comment_token_count

        if total_token_count > 0:
            p_exact = binom.sf(total_green_count - 1, total_token_count, gamma)
            score = -np.log10(np.clip(p_exact, 1e-300, 1.0))
            z_score = (total_green_count - gamma * total_token_count) / math.sqrt(gamma * (1 - gamma) * total_token_count)
        else:
            p_exact = 1.0
            score = 0.0
            z_score = 0.0

        unique_starts = ", ".join(sorted(set(total_starting_letters)))

        return {
            "id_starting_letters": id_starting_letters,
            "id_unique_tokens": id_unique_tokens,
            "id_green_count": id_green_count,
            "id_token_count": id_token_count,
            "comment_starting_letters": comment_starting_letters,
            "comment_unique_tokens": comment_unique_tokens,
            "comment_green_count": comment_green_count,
            "comment_token_count": comment_token_count,
            "total_starting_letters": total_starting_letters,
            "total_green_count": total_green_count,
            "total_token_count": total_token_count,
            "overlap_count": len(overlap_tokens),
            "p_exact": p_exact,
            "score": score,
            "z_score": z_score,
            "unique_starts": unique_starts,
        }

    original_data = get_starting_letters(original_code)
    generated_data = get_starting_letters(generated_code)

    P_THRESHOLD = norm.sf(2.12)

    return {
        "original_z_score": original_data["z_score"],
        "original_p_exact": original_data["p_exact"],
        "original_score": original_data["score"],
        "original_token_count": original_data["total_token_count"],
        "original_green_count": original_data["total_green_count"],
        "original_is_watermarked": original_data["p_exact"] < P_THRESHOLD,
        "original_unique_starts": original_data["unique_starts"],
        "generated_z_score": generated_data["z_score"],
        "generated_p_exact": generated_data["p_exact"],
        "generated_score": generated_data["score"],
        "generated_token_count": generated_data["total_token_count"],
        "generated_green_count": generated_data["total_green_count"],
        "generated_is_watermarked": generated_data["p_exact"] < P_THRESHOLD,
        "generated_unique_starts": generated_data["unique_starts"],
    }


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Initialize LLM provider lazily (only when first needed)
# Removed eager initialization to avoid blocking imports
_provider_initialized = False

def _ensure_provider_initialized():
    """Initialize provider on first use (lazy initialization)."""
    global _provider_initialized
    if _provider_initialized:
        return
    
    if LLMProviderFactory is not None:
        try:
            initialize_llm_provider()
            _provider_initialized = True
        except Exception as e:
            print(f"Note: LLM provider initialization skipped ({type(e).__name__})")
            print("      Running in fallback mode (direct API)")
            _provider_initialized = True

