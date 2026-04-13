# Watermarking Experiments - Modular Python Structure

This directory contains a refactored and modularized version of the watermarking experiments from the notebooks. All experiments have been extracted into individual Python modules with a shared utilities library.

## Directory Structure

```
src/
├── shared_utils.py              # Shared utilities and common functions
├── exp_code_only.py             # Experiment: Code as Text Only (expT)
├── exp_static_wm.py             # Experiment: Static Watermarking (expS)
├── exp_comment_wm.py            # Experiment: Comment-Based Watermarking (expA)
├── exp_iterative_wm.py          # Experiment: Iterative Watermarking (expI)
├── exp_refactoring_wm.py        # Experiment: Refactoring-Based Watermarking (expX)
├── run_experiments.sh           # Bash script to run all experiments
└── README.md                    # This file
```

## Experiment Methods

`TRIVIAL_APPROACH` - If `True`, then it means green-red tokens are 50/50 equally randomly divided.
`TRIVIAL_APPROACH*` - If `True`, then it means green-red tokens are taken from randomly distributed words samples as mentioned in [this](https://arxiv.org/abs/2505.16934) paper.

If `False` then it will use our proposed frequency-based cryptographic token stratification technique to select green/red tokens.

### 1. **expT: Code as Text Only**

- **File**: `exp_code_only.py`
- **Approach**: Generates code without watermarking constraints, treating it as natural text
- **Settings**:
  - `APPLY_WATERMARKING = False`
  - `COMMENT_ENABLED = False`
  - `ITERATIVE_MODE = False`
  - `CHECK_CORRECTNESS = True`
  - `TRIVIAL_APPROACH* = True`

### 2. **expS: Static Watermarking**

- **File**: `exp_static_wm.py`
- **Approach**: Embeds watermarks through identifier naming during code generation
- **Settings**:
  - `APPLY_WATERMARKING = True`
  - `COMMENT_ENABLED = False`
  - `ITERATIVE_MODE = False`
  - `CHECK_CORRECTNESS = True`
  - `TRIVIAL_APPROACH = True`

### 3. **expA: Comment-Based Watermarking**

- **File**: `exp_comment_wm.py`
- **Approach**: Embeds watermarks in both identifiers and comments
- **Settings**:
  - `APPLY_WATERMARKING = True`
  - `COMMENT_ENABLED = True`
  - `ITERATIVE_MODE = False`
  - `CHECK_CORRECTNESS = True`
  - `TRIVIAL_APPROACH = True`

### 4. **expX: Refactoring-Based Watermarking (Post-Hoc)**

- **File**: `exp_refactoring_wm.py`
- **Approach**: Two-phase process:
  1. Provide the actual solution code instead of a prompt to the LLM
  2. Refactor the code to rename identifiers and embed watermarks
- **Settings**:
  - `APPLY_WATERMARKING = True`
  - `COMMENT_ENABLED = False`
  - `ITERATIVE_MODE = False`
  - `CHECK_CORRECTNESS = True`
  - `TRIVIAL_APPROACH = True`

### 5. **expI: Iterative Watermarking**

- **File**: `exp_iterative_wm.py`
- **Approach**: Generates code with iterative refinement based on correctness and watermark fidelity feedback
- **Settings**:
  - `APPLY_WATERMARKING = True`
  - `COMMENT_ENABLED = True`
  - `ITERATIVE_MODE = True`
  - `CHECK_CORRECTNESS = True`
  - `TRIVIAL_APPROACH = False`

### 6. **expA: Iterative + Adaptive Watermarking (Ours)**

- **File**: `exp_refactoring_wm.py`
- **Approach**: Combines iterative feedback loops with adaptive watermarking strategies to optimize both correctness and watermark detectability
- **Settings**:
  - `APPLY_WATERMARKING = True`
  - `COMMENT_ENABLED = True`
  - `ITERATIVE_MODE = True`
  - `CHECK_CORRECTNESS = True`
  - `TRIVIAL_APPROACH = False`

## Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn scipy boto3
# Set up AWS credentials (for Bedrock API)
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export DEFAULT_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0  # Optional
```

### Run All Experiments

```bash
cd src
bash run_experiments.sh all humaneval 100
```

### Run Specific Experiments

```bash
# Single experiment
bash run_experiments.sh expS humaneval 100

# Multiple experiments (comma-separated)
bash run_experiments.sh expS,expA,expI humaneval 100

# Custom dataset
bash run_experiments.sh all /path/to/dataset.json 50
```

### Run Individual Experiment

```bash
python exp_static_wm.py datasets/humaneval_164.json output/exp_s results/exp_s.csv
```

## Command Line Arguments

### `run_experiments.sh`

```bash
run_experiments.sh [METHOD] [DATASET] [SAMPLE_SIZE]
```

- **METHOD**: `all`, `expT`, `expS`, `expA`, `expI`, `expX`, or comma-separated list
- **DATASET**: `humaneval`, `mbpp`, or path to custom dataset
- **SAMPLE_SIZE**: Number of samples to process (default: 100)

### Individual Python Modules

```bash
python exp_code_only.py [DATASET_PATH] [OUTPUT_DIR] [RESULTS_CSV]
```

- **DATASET_PATH**: Path to JSON dataset file
- **OUTPUT_DIR**: Directory to save generated code files
- **RESULTS_CSV**: Path to save results CSV file

## Shared Utilities (`shared_utils.py`)

The `shared_utils.py` module provides common functionality used by all experiments:

### Key Classes

- **`CodeNavigator`**: AST-based extractor for Python identifiers

### Key Functions

- **`get_red_green_sets()`**: Generates green/red letter sets based on secret key
- **`detect_watermark()`**: Performs watermark detection with deduplication
- **`generate_response()`**: Calls Claude via AWS Bedrock
- **`test_code()`**: Executes generated code with test cases
- **`load_frequency_data()`**: Loads letter frequency data from datasets
- **`calculate_gamma()`**: Computes watermark embedding ratio

### Watermark Detection Metrics

- **p_exact**: Exact binomial p-value
- **z_score**: Standard normal z-score
- **score**: -log10(p_exact) for ROC analysis
- **token_count**: Number of identifiers/comments
- **green_count**: Number of green-letter tokens

## Output Structure

### CSV Results Files

```
results/raw/
├── claude_expT_during_gen_v1_100_humaneval.csv
├── claude_expS_during_gen_v1_100_humaneval.csv
├── claude_expA_during_gen_v1_100_humaneval.csv
├── claude_expI_during_gen_v1_100_humaneval.csv
└── claude_expX_only-gen_gen_v1_100_humaneval.csv
```

### Generated Code Files

```
output/
├── claude_expT_during_gen_v1_100_humaneval/
│   ├── HumanEval_0.py
│   ├── HumanEval_1.py
│   └── ...
├── claude_expS_during_gen_v1_100_humaneval/
└── ...
```

## CSV Output Columns

Common columns in results CSVs:

- `task_id`: Problem identifier
- `correctness`: Whether generated code passes all tests
- `tests_passed`: Number of passing test cases
- `tests_failed`: Number of failing test cases
- `total_tests`: Total number of test cases
- `pass_rate`: Percentage of passing tests
- `iteration_used`: Which iteration was selected (for iterative methods)
- `input_tokens`: Input tokens used by LLM
- `output_tokens`: Output tokens generated
- `error_message`: Error details if tests failed

Watermark columns (when APPLY_WATERMARKING=True):

- `generated_p_exact`: Watermark p-value
- `generated_z_score`: Watermark z-score
- `generated_score`: -log10(p-value) score
- `generated_token_count`: Total tokens analyzed
- `generated_green_count`: Green-letter tokens
- `generated_is_watermarked`: Whether watermark detected (p < threshold)
- `meets_z`: Boolean indicating watermark threshold met

## Configuration

Each experiment module has configuration constants that can be modified:

```python
# Experiment identification
EXPERIMENT_NUMBER = "expT"
EXP_VERSION = "v1"
GENERATION_TYPE = "during"
DATASET = "humaneval"

# Control flags
COMMENT_ENABLED = False
CHECK_CORRECTNESS = True
APPLY_WATERMARKING = False
ITERATIVE_MODE = False

# Watermark parameters
Z_THRESHOLD = 2.12
P_THRESHOLD = norm.sf(Z_THRESHOLD)
SEED_KEY = "exp2025"
N_MIN_TOKENS = 5
ITER_CAP = 5
```

## Example Usage

### Run a single experiment with verbose output

```bash
cd /home/fahad/Documents/PROJECTS/promptmark
python src/exp_static_wm.py datasets/humaneval_164.json output/test_expS results/test_expS.csv
```

### Run all experiments with small sample

```bash
cd /home/fahad/Documents/PROJECTS/promptmark
bash src/run_experiments.sh all humaneval 10
```

### Run iterative and refactoring methods only

```bash
bash src/run_experiments.sh expI,expX humaneval 100
```

## Modifying Experiments

To create a new variant:

1. Copy an existing experiment module
2. Modify the constants at the top
3. Adjust `SYSTEM_PROMPT` and `PROBLEM_TEMPLATE` as needed
4. Change `EXPERIMENT_NUMBER` to avoid conflicts
5. Run the modified script

## Troubleshooting

### AWS Credentials Error

Ensure environment variables are set:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export DEFAULT_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
```

### Dataset Not Found

Verify dataset path:

```bash
ls -la datasets/humaneval_164.json
```

### Timeout Errors

Increase the test timeout in `shared_utils.py`:

```python
def test_code(code, test_imports, tests, timeout=2):  # Change timeout value
```

### Memory Issues

Reduce sample size or run experiments sequentially instead of in parallel.

## Performance Notes

- **expT**: Fastest (no watermarking)
- **expS**: Fast (simple watermarking)
- **expA**: Moderate (with comment extraction)
- **expI**: Slowest (iterative feedback loops)
- **expX**: Moderate (two-phase generation)

## Output Example

After running `bash run_experiments.sh all humaneval 10`, you'll see:

```
✅ expT experiment completed!
✅ expS experiment completed!
✅ expA experiment completed!
✅ expI experiment completed!
✅ expX experiment completed!

Generated Files:
✅ results/raw/claude_expT_during_gen_v1_10_humaneval.csv
✅ results/raw/claude_expS_during_gen_v1_10_humaneval.csv
✅ results/raw/claude_expA_during_gen_v1_10_humaneval.csv
✅ results/raw/claude_expI_during_gen_v1_10_humaneval.csv
✅ results/raw/claude_expX_only-gen_gen_v1_10_humaneval.csv
```

## Citation

If you use these experiments, please cite the original watermarking research:

```bibtex
@article{promptmark2024,
  title={Promptmark: Watermarking LLM Code Generation via Adaptive Identifier Renaming},
  year={2024}
}
```

## License

Same as the parent promptmark project.
