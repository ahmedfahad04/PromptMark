# CodeBLEU Metrics - Enhanced Multi-Language Support

This directory contains an enhanced version of the CodeBLEU evaluation metrics with improved flexibility for Python, Java, C++, and other programming languages.

## Overview

CodeBLEU is a comprehensive metric for evaluating code generation quality that combines:
- **BLEU**: Standard n-gram overlap
- **Weighted BLEU**: Keyword-weighted n-gram overlap  
- **Syntax Match**: Abstract syntax tree similarity
- **Dataflow Match**: Data flow graph similarity

## Key Improvements

### ✅ **Flexible Language Support**
- **Before**: Hardcoded language handling, inconsistent mappings
- **After**: Configurable language system supporting Python, Java, C++, C, JavaScript, etc.

### ✅ **Python-First Design**
- Optimized for Python code evaluation
- Proper Python keyword weighting
- Python-specific syntax and dataflow analysis

### ✅ **Better Error Handling**
- Graceful handling of missing keyword files
- Clear error messages for unsupported languages
- Robust file I/O with detailed error reporting

### ✅ **Enhanced API**
- Simple helper functions for Python evaluation
- Batch evaluation support
- Both string-based and file-based evaluation

## Files Structure

```
metrics/
├── calc_code_bleu.py          # Main CodeBLEU calculation (enhanced)
├── python_codebleu_helper.py  # Python-specific helper functions
├── bleu.py                    # Standard BLEU implementation
├── weighted_ngram_match.py    # Keyword-weighted BLEU
├── syntax_match.py            # AST similarity (improved)
├── dataflow_match.py          # Dataflow graph similarity (improved)
├── utils.py                   # Utility functions
├── keywords/                  # Language keyword files
│   ├── python.txt            # Python keywords
│   ├── java.txt              # Java keywords
│   ├── cpp.txt               # C++ keywords (new)
│   ├── c.txt                 # C keywords
│   └── javascript.txt        # JavaScript keywords
└── parser/                   # Tree-sitter parsers
    ├── DFG.py               # Data flow graph extraction
    ├── utils.py             # Parser utilities
    └── languages.so         # Compiled tree-sitter languages
```

## Usage

### 1. Command Line Usage

```bash
# Evaluate Python code
python calc_code_bleu.py --refs reference.py --hyp generated.py --lang python

# Evaluate Java code  
python calc_code_bleu.py --refs reference.java --hyp generated.java --lang java

# Evaluate C++ code
python calc_code_bleu.py --refs reference.cpp --hyp generated.cpp --lang cpp

# Custom weights (alpha, beta, gamma, theta)
python calc_code_bleu.py --refs ref.py --hyp gen.py --lang python --params "0.4,0.3,0.2,0.1"
```

### 2. Python API - Simple Evaluation

```python
from python_codebleu_helper import evaluate_python_code_bleu

# Evaluate two Python code strings
reference = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

generated = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""

scores = evaluate_python_code_bleu(reference, generated)
print(f"CodeBLEU Score: {scores['codebleu']:.4f}")
```

### 3. Python API - Batch Evaluation

```python
from python_codebleu_helper import batch_evaluate_python

references = [
    "def add(a, b): return a + b",
    "def multiply(x, y): return x * y"
]

hypotheses = [
    "def add(x, y): return x + y", 
    "def multiply(a, b): return a * b"
]

scores = batch_evaluate_python(references, hypotheses)
for i, score in enumerate(scores):
    print(f"Example {i+1}: CodeBLEU = {score['codebleu']:.4f}")
```

### 4. Python API - File-based Evaluation

```python
from python_codebleu_helper import evaluate_python_files

score = evaluate_python_files("reference.py", "generated.py")
print(f"CodeBLEU Score: {score:.4f}")
```

### 5. Advanced API - Multi-language Support

```python
from calc_code_bleu import evaluate_per_example

# Python
scores = evaluate_per_example(ref_code, gen_code, "python")

# Java  
scores = evaluate_per_example(ref_code, gen_code, "java")

# C++
scores = evaluate_per_example(ref_code, gen_code, "cpp")
```

## Supported Languages

| Language | Code | Keyword File | Tree-sitter | Status |
|----------|------|--------------|-------------|---------|
| Python | `python` | ✅ python.txt | ✅ python | ✅ Fully Supported |
| Java | `java` | ✅ java.txt | ✅ java | ✅ Fully Supported |  
| C++ | `cpp` | ✅ cpp.txt | ✅ cpp | ✅ Fully Supported |
| C | `c` | ✅ c.txt | ✅ c | ✅ Fully Supported |
| JavaScript | `javascript` | ✅ javascript.txt | ✅ javascript | ✅ Supported |
| C# | `c_sharp` | ⚠️ Fallback to C | ✅ c_sharp | ⚠️ Partial Support |
| PHP | `php` | ⚠️ Fallback to C | ✅ php | ⚠️ Partial Support |
| Go | `go` | ⚠️ Fallback to C | ✅ go | ⚠️ Partial Support |
| Ruby | `ruby` | ⚠️ Fallback to C | ✅ ruby | ⚠️ Partial Support |

## Score Interpretation

CodeBLEU scores range from 0.0 to 1.0:

- **0.8-1.0**: Excellent similarity (nearly identical or very similar code)
- **0.6-0.8**: Good similarity (similar logic, may have different variable names)
- **0.4-0.6**: Moderate similarity (similar structure, some differences)
- **0.2-0.4**: Low similarity (different approach but some overlap)
- **0.0-0.2**: Very low similarity (substantially different code)

### Component Scores

- **EM (Exact Match)**: 1.0 if identical, 0.0 otherwise
- **BLEU**: Token-level n-gram overlap (0.0-1.0)
- **Weighted BLEU**: Keyword-emphasized BLEU (0.0-1.0)  
- **Syntax**: AST structure similarity (0.0-1.0)
- **Dataflow**: Variable dependency similarity (0.0-1.0)

## Weight Configuration

The four components can be weighted differently based on your evaluation needs:

```python
# Equal weights (default)
weights = "0.25,0.25,0.25,0.25"

# Emphasize syntax structure
weights = "0.1,0.1,0.6,0.2" 

# Emphasize dataflow patterns
weights = "0.1,0.1,0.2,0.6"

# Emphasize token overlap
weights = "0.6,0.2,0.1,0.1"
```

## Adding New Languages

To add support for a new language:

1. **Add keyword file**: Create `keywords/{lang}.txt` with language keywords
2. **Update config**: Add entry to `LANGUAGE_CONFIG` in `calc_code_bleu.py`
3. **Verify parser**: Ensure tree-sitter parser exists in `languages.so`

Example for adding Rust support:

```python
# In calc_code_bleu.py LANGUAGE_CONFIG
'rust': {
    'keyword_file': 'rust.txt',
    'tree_sitter_name': 'rust',
    'comment_removal_lang': 'rust',
    'wrapper_needed': False,
    'wrapper_template': None
}
```

## Dependencies

- `tree-sitter` >= 0.20.0
- `numpy`
- Python 3.7+

## Examples and Testing

Run the helper script to see examples:

```bash
python python_codebleu_helper.py
```

This will demonstrate:
- Single code pair evaluation
- Batch evaluation
- Different weight configurations
- Detailed score breakdowns

## Migration from Original CodeBLEU

The enhanced version is backward compatible. Simply update your imports:

```python
# Old way
from calc_code_bleu import get_codebleu

# New way (same function, enhanced)
from calc_code_bleu import get_codebleu

# Or use new Python-specific helpers
from python_codebleu_helper import evaluate_python_code_bleu
```

## Troubleshooting

### Common Issues

1. **Language not supported**: Check `SUPPORTED_LANGUAGES` list
2. **Missing keyword file**: The system will warn and use empty keywords
3. **Parser errors**: Ensure `languages.so` contains the required tree-sitter parser
4. **File not found**: Check file paths and permissions

### Debug Mode

Enable verbose output by setting debug flags in the code or catching exceptions:

```python
try:
    score = evaluate_python_code_bleu(ref, gen)
except Exception as e:
    print(f"Evaluation error: {e}")
    import traceback
    traceback.print_exc()
```

## Performance Notes

- **Single evaluation**: ~0.1-0.5 seconds per code pair
- **Batch evaluation**: More efficient for multiple pairs
- **Memory usage**: Scales with code length and complexity
- **Parser overhead**: Initial tree-sitter parsing takes most time

## License

Based on Microsoft's CodeXGLUE implementation, enhanced for multi-language flexibility.