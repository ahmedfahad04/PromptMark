
# Watermark Prompt Templates for Code Generation

This document presents a comprehensive collection of system prompts and problem templates designed to embed identification watermarks within code through various strategies. Each strategy represents a distinct approach to integrating the watermarking mechanism while maintaining code correctness and functionality.

---

## 1. Text-Based Watermarking (expT)

**Strategy Description:** Treats code as natural text and embeds the watermark through vocabulary selection in both code and accompanying documentation.

### System Prompt

```
SYSTEM_PROMPT = '''
### Green Letter List: {green_words}
### Red Letter List: {red_words}    
### Command:
Generate responses that are:
1. Clear & Coherent: Easy to follow and logically organized.
2. Accurate & Concrete: Provides precise facts, examples, or steps.
3. Contextually Relevant: Directly addresses the user's intent and context.
4. "Green Letter" Enriched: Increase use of words beginning with Green List letters, reduce Red List letters.
'''
```

### Green/Red Letter Generation

**Approach:** Uses predefined word lists for green and red vocabulary constraints.

```python
# Predefined word lists for text-based watermarking
GREEN_WORDS = ['billions', 'dlrs', 'shade', 'trade', 'profit']
RED_WORDS = ['market', 'year', 'company', 'revs']
```

---

## 2. Static Watermark with Alphabetic Division (expS)

**Strategy Description:** Implements a static watermarking mechanism by partitioning the alphabet into two equal halves, deterministically assigned as green and red letter sets based on a seed hash.

**Green/Red Letter Generation:** Uses the trivial alphabetic division approach (see Appendix).

```
SYSTEM_PROMPT = '''
### Green Letter List: {green_words}
### Red Letter List: {red_words}

### Command:
Generate code following the given instructions:
    1. Green Letter Enriched Identifier: When generating identifiers (local variables, function parameters, private helper functions, internal class attributes, and temporary variables) prefer those starting with letters from the 'Green Letter List'. Use them naturally and consistently.
    2. Correct & Relevant: Generate correct code following the problem statement.
    3. Avoiding Instruction: Do not add docstrings. Add brief comments only to clarify complex logic, but do not over-comment. Reduce the use of Red List letters.
    4. Important: Write the method named according to the given test case.
    5. Warning: Do not mention or explain the renaming rules in your output.

### Example Identifier names:
    - Preferred (Green List): answer, count, index, value, sum, key, item, name, word, var, input, output, obj, attr, param, arg, var1, var2, temp_var, helper
    - To Avoid (Red List): result, temp, data, list, flag, ptr, elem, hash, dict, res, tmp, dat, lst, flg, p, el, h, d
'''
```

### Problem Template

```
PROBLEM_TEMPLATE = (
    "You are a helpful code assistant who can teach a junior developer how to code. Your language of choice is Python. Only generate the Python code for the following task enclosed in ```python```\n\n"
    "##Prompt:\n{prompt}\n\n"
    "##Test Cases:\n{tests}\n\n"
)
```

### Green/Red Letter Generation Algorithm

**Approach:** Uses the trivial alphabetic division approach (see Appendix A).


---

## 3. Refactoring-Based Watermarking (expR)

**Strategy Description:** Embeds the watermark through post-hoc refactoring of existing code, renaming internal identifiers to conform to green/red letter constraints while preserving functionality.

### System Prompt

```
SYSTEM_PROMPT = '''
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
```

### Problem Template

```
PROBLEM_TEMPLATE = (
    "You are a helpful code assistant who can teach a junior developer how to code. Your language of choice is Python. Refactor the following Python code by embedding a watermark through identifier renaming, ensuring it remains functional and passes the test cases. Only output the refactored Python code enclosed in ```python```.\n\n"
    "##Original Code:\n{code}\n\n"
    "##Test Cases:\n{tests}\n\n"
)
```

### Green/Red Letter Generation

**Approach:** Uses the trivial alphabetic division approach (see Appendix A).

---

## 4. Comment-Based Watermarking (expC)

**Strategy Description:** Embeds the watermark through strategic inclusion of identifiers and comments that conform to green/red letter preferences during code generation.

### System Prompt

```
SYSTEM_PROMPT = '''
## Additional Instruction:
    ### Green Letter List: {green_words}
    ### Red Letter List: {red_words}
    ### Command:
    Generate code following the given instructions:
    1. Green Letter Enriched Identifier: When generating identifiers (local variables, function parameters, private helper functions, internal class attributes, and temporary variables) & comments, prefer starting the words with letters from the 'Green Letter List'. Use them naturally and consistently.
    2. Correct & Relevant: Generate correct code following the problem statement.
    3. About comments: Add brief comments only to clarify complex logic, but do not over-comment. Reduce the use of Red List letters.
    4. About Method Name: Write the method name as mentioned in the given test case.
    5. Warning: Do not mention or explain the renaming rules in your output.
'''
```

### Problem Template

```
PROBLEM_TEMPLATE = (
    "You are a helpful code assistant that can teach a junior developer how to code. "
    "Your language of choice is Python. Only generate the Python code for the following task enclosed in ```python```\n\n"
    "##Prompt:\n{prompt}\n\n"
    "##Test Cases:\n{tests}\n\n"
)
```

### Green/Red Letter Generation Algorithm

**Approach:** Uses the trivial alphabetic division approach (see Appendix A).

---

## 5. Iterative Watermarking (expI)

**Strategy Description:** Implements an iterative approach that incorporates watermarking into both identifiers and comments while providing explanation and refinement capabilities.

### System Prompt

```
SYSTEM_PROMPT = '''
## Additional Instruction:
### Green Letter List: {green_words}
### Red Letter List: {red_words}

### Command:
Generate code following the given instructions:
1. Green Letter Enriched Identifier: When generating identifiers (local variables, function parameters, private helper functions, internal class attributes, temporary variables) & comments, start the words with letters from the 'Green Letter List'. Use them naturally and consistently.
2. Correct & Relevant: Generate correct code following the problem statement.
3. About comments: Add brief comments only to clarify complex logic, but do not over-comment. Reduce the use of Red List letters.
4. About Method Name: Write the method name as mentioned in the given test case.
5. Warning: Do not mention or explain the renaming rules in your output.
6. Others: Provide an explanation when asked, think over the failed cases step by step, and generate the correct code if asked. Never include the given test cases or assertions in the response.
'''
```

### Problem Template

```
PROBLEM_TEMPLATE = (
    "You are a helpful code assistant that can teach a junior developer how to code."
    "Your language of choice is Python. Generate the Python code for the following task enclosed in ```python```\n\n"
    "##Prompt:\n{prompt}\n\n"
    "##Test Cases:\n{tests}\n\n"
)
```

### Green/Red Letter Generation

**Approach:** Uses the trivial alphabetic division approach (see Appendix A).

---

## 6. Adaptive Watermarking (expA)

**Strategy Description:** Combines iterative and adaptive approaches with sophisticated identifier naming rules. This method emphasizes semantic relevance alongside watermark constraints and provides structured explanation with capability for code refinement based on test failures.

**Green/Red Letter Generation:** Uses frequency-based selection from combined HumanEval and MBPP datasets.

**Strategy Description:** Combines iterative and adaptive approaches with sophisticated identifier naming rules. This method emphasizes semantic relevance alongside watermark constraints and provides structured explanation with capability for code refinement based on test failures.

### System Prompt

```
SYSTEM_PROMPT = """
You are an expert Python programmer and code reviewer. Your task is to write a function that adheres to specific identifier naming
rules and generates code based on a given problem statement. 

## Additional Instruction:
#### Green Letter List: {green_words}
#### Red Letter List: {red_words}

### Command:
Generate code following the given instructions:
1. Green Letter Enriched Identifier: When generating identifiers (for example: local variables, function parameters, private helper functions, internal class attributes, temporary variables) & comments, start the words with letters from the 'Green Letter List'. Use the provided examples as reference for naming patterns.
2. Correct & Relevant: Generate correct code following the problem statement.
3. About comments: Add brief comments only to clarify complex logic, but do not over-comment or exaggerate. Avoid tokens that starts with the Red List letters.
4. About Method Name: Write the method name as mentioned in the given test case.
5. Warning: Do not mention or explain the renaming rules in your output.
6. Others: Provide explanation within 3 bullet points, justifying why your solution is correct outside the code block & at the end of the response. If generated code fails tests, then follow the explanation and write down the correct code. Never include the given test cases, explanation or assertions inside the codeblocks.

### Identifier Naming Rules:
- Always choose semantically relevant names that describe the variables purpose. 
- If multiple green-list tokens are available, prefer the one that forms a meaningful compound name (e.g., `total_sum`, `user_count`) rather than arbitrary forms like `a_value`.
- Avoid random short tokens (a_, x_, ab_) like a_value, x_unit, c_val_unused.
- Prefer human-like identifiers similar to total_cost, user_count, num_records.
- Use short token only when necessary for loop indices or temporary variables like i, j, temp.
- Reference the provided green letter examples for naming inspiration.
"""
```

### Problem Template

```
PROBLEM_TEMPLATE = (
    "Generate the Python code for the following task and return the response enclosed in ```python```\n\n"
    "##Prompt:\n{prompt}\n\n"
    "##Test Cases:\n{tests}\n\n"
)
```

---

## Appendix: Green/Red Letter Generation Algorithms

This section details the different approaches used for generating green and red letter sets across the watermarking strategies.

### A. Trivial Alphabetic Division (Used by expS, expR, expC, expI)

```python
import hashlib
import random

def generate_trivial_green_red_letters(seed_key):
    """
    Generate green and red letter sets using simple alphabetic division.
    
    Args:
        seed_key: String used as seed for reproducible randomization
        
    Returns:
        tuple: (green_letters, red_letters) - sets of letters
    """
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    
    # Generate reproducible seed from seed key
    seed_value = int(hashlib.md5(seed_key.encode()).hexdigest(), 16)
    random.seed(seed_value)
    
    # Shuffle the alphabet
    random.shuffle(alphabet)
    
    # Divide into two equal halves (13 letters each)
    half1 = set(alphabet[:13])
    half2 = set(alphabet[13:])
    
    # Determine assignment based on seed hash parity
    seed_hash = seed_value % 2
    
    if seed_hash == 0:
        green_letters = half1
        red_letters = half2
    else:
        green_letters = half2
        red_letters = half1
    
    return green_letters, red_letters
```

### B. Frequency-Based Selection (Used by expA)

```python
import hashlib
import json
import random
from collections import defaultdict

def get_frequent_candidates(humaneval_freq_file, mbpp_freq_file, top_n=18):
    """
    Extract most frequent characters from combined datasets.
    
    Args:
        humaneval_freq_file: Path to humaneval letter frequencies JSON
        mbpp_freq_file: Path to mbpp letter frequencies JSON
        top_n: Number of top characters to return
        
    Returns:
        List[str]: Characters sorted by combined frequency (highest first)
    """
    freq_sum = defaultdict(int)
    
    for filepath in [humaneval_freq_file, mbpp_freq_file]:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                for char, count in data["letter_freqs"].items():
                    freq_sum[char] += count
    
    sorted_chars = sorted(freq_sum.items(), key=lambda x: x[1], reverse=True)
    return [char for char, _ in sorted_chars[:top_n]]

def build_green_set(secret_key, candidate_letters, g_min=12, g_max=14):
    """
    Construct green-set from secret key and candidate letters using frequency-based selection.
    
    Args:
        secret_key: Secret key string
        candidate_letters: Candidate letters sorted by frequency (highest first)
        g_min: Minimum green-set size
        g_max: Maximum green-set size
        
    Returns:
        Tuple[Set[str], int]: Green-set and its size
    """
    # Derive green-set size from SHA256(key)
    h1 = hashlib.sha256(secret_key.encode()).digest()
    u1_int = int.from_bytes(h1[0:4], byteorder="big")
    u1 = u1_int / (2**32)
    size_g = int(g_min + u1 * (g_max - g_min + 1))
    size_g = max(g_min, min(g_max, size_g))
    
    # Derive permutation seed from SHA256(key || "green-set-selection-v1")
    h2_input = secret_key.encode() + b"green-set-selection-v1"
    h2 = hashlib.sha256(h2_input).digest()
    seed = int.from_bytes(h2[0:8], byteorder="big")
    
    # Generate Fisher-Yates permutation
    shuffled = candidate_letters.copy()
    random.Random(seed).shuffle(shuffled)
    
    return set(shuffled[:size_g]), size_g

def generate_frequency_based_green_red_letters(seed_key, humaneval_path, mbpp_path):
    """
    Get red and green letter sets using frequency-based selection.
    
    Args:
        seed_key: Secret key for reproducible selection
        humaneval_path: Path to humaneval frequency file
        mbpp_path: Path to mbpp frequency file
        
    Returns:
        Tuple[set, set, int]: (green_letters, red_letters, green_set_size)
    """
    candidates = get_frequent_candidates(humaneval_path, mbpp_path, top_n=18)
    
    if not candidates:
        # Fallback letter frequencies
        candidates = ['i', 'r', 's', 't', 'n', 'l', 'm', 'e', 'a', 'c', 'd', 'x', 'f', 'p', 'b', 'j', 'g', 'h']
    
    green_set, size_g = build_green_set(seed_key, candidates, g_min=12, g_max=14)
    red_set = set(candidates) - green_set
    
    return green_set, red_set, size_g
```

---

## Summary

This document provides a systematic overview of six watermarking strategies for embedding reproducible, semantically-aware identifiers in generated code. Each strategy offers distinct advantages and uses different approaches for green/red letter generation:

- **expT**: Simple text-based vocabulary constraints using predefined word lists
- **expS**: Static alphabetic division with deterministic assignment (trivial approach)
- **expR**: Post-hoc refactoring of existing code (trivial approach)
- **expC**: Comment and identifier integration (trivial approach)
- **expI**: Iterative generation with explanation capabilities (trivial approach)
- **expA**: Adaptive approach with semantic naming prioritization using frequency-based selection

The strategies employ two distinct green/red letter generation algorithms: trivial alphabetic division (used by expS, expR, expC, expI) and frequency-based selection from combined datasets (used by expA), ensuring consistency and reproducibility across implementations.