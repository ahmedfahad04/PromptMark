# Code Evaluation Toolkit

A comprehensive suite of evaluation tools for AI-generated code, calculating multiple metrics including Pass@1, AUROC, TPR@X%F, and CodeBLEU.

## 📋 Overview

This toolkit provides four main evaluation metrics for generated code:

1. **Pass@1 (Average Pass Rate)** - Percentage of generated code that passes all test cases
2. **AUROC** - Area Under ROC Curve for watermark detection capability
3. **T@X%F** - True Positive Rate at X% False Positive Rate (X=0,1,5,10) for watermark detection
4. **CodeBLEU** - Code similarity metric combining BLEU, syntax matching, and dataflow analysis

## 🚀 Quick Start

```bash
# Quick evaluation of a single model
python quick_eval.py --model codegemma-7b-it --experiment exp1_dgen_v1_100

# List available models/experiments
python batch_eval.py --list_available

# Show usage examples
python eval_utils.py --examples
```

## 📁 Scripts Overview

| Script | Purpose | Best For |
|--------|---------|----------|
| `comprehensive_evaluation.py` | Full-featured evaluation with manual path specification | Detailed analysis, custom setups |
| `quick_eval.py` | Simplified evaluation with auto-detected paths | Quick testing, standard setups |
| `batch_eval.py` | Multi-model comparison and batch processing | Comparing multiple models |
| `eval_utils.py` | Documentation, examples, and validation utilities | Learning, troubleshooting |

## 📊 Input Data Requirements

### CSV File Structure
Required columns:
- `task_id`: Unique identifier for each coding task

Optional columns (for different metrics):
- `pass_rate` or `all_passed`: Test execution results
- `original_z_score`, `generated_z_score`: Watermark detection scores
- `tests_passed`, `total_tests`: Detailed test results

Example:
```csv
task_id,original_z_score,generated_z_score,pass_rate,all_passed
590,-1.508,0.418,0.0,False
776,-2.835,-3.729,33.33,False
```

### Generated Code Directory
- Contains Python files named by task_id (e.g., `57.py`, `590.py`)
- Each file contains the AI-generated code solution

### Reference File (JSONL)
- JSON Lines format with ground truth implementations
- Required fields: `task_id`, `code`
- Optional fields: `prompt`, `test_list`

Example:
```json
{"task_id": 590, "code": "def polar_rect(x,y):\n    import cmath\n    ...", "prompt": "Write a function..."}
```

## 💡 Usage Examples

### 1. Quick Single Model Evaluation
```bash
python quick_eval.py --model codegemma-7b-it --experiment exp1_dgen_v1_100
```

### 2. Full Evaluation with Custom Paths
```bash
python comprehensive_evaluation.py \
    --csv_file ../../results/raw/exp1_codegemma-7b-it_during_gen_v1_100.csv \
    --generated_dir ../../output/codegemma-7b-it_exp1_dgen_v1_100 \
    --reference_file ../../datasets/core/sanitized-mbpp-sample-100.json \
    --output results.json
```

### 3. Batch Evaluation (Multiple Models/Versions)
```bash
# Evaluate all versions of specific models
python batch_eval.py --models codegemma qwen14b --experiments all

# Evaluate all available models and versions
python batch_eval.py --models all --experiments all

# Results include both experiment-level and model-level aggregations
```

### 4. Control CodeBLEU Evaluation Size
```bash
# Fast evaluation (sample 20 examples)
python quick_eval.py --model codegemma-7b-it --experiment exp1_dgen_v1_100 --codebleu_sample_size 20

# Full evaluation (all examples, slower)
python quick_eval.py --model codegemma-7b-it --experiment exp1_dgen_v1_100 --full_codebleu
```

## 📈 Output Format

## 📊 Output Format

### Console Output
```
🎯 Execution Metrics:
   Pass@1 (Avg Pass Rate): 0.6833 (68.33%)

🔍 Watermark Detection Metrics:
   AUROC:                   0.4712
   T@0%F               : 0.0100
   T@1%F               : 0.0100
   T@5%F               : 0.0600
   T@10%F              : 0.1000

📝 Code Quality Metrics:
   CodeBLEU Mean:           0.2808
   CodeBLEU Std:            0.1956
   CodeBLEU Range:          [0.0777, 0.6714]
```

### Model-Level Aggregation (New Feature)
When multiple versions of the same model are evaluated, the script automatically creates model-level averages:

```
📊 MODEL-LEVEL AGGREGATION:
--------------------------------------------------------------------------------

📋 Model Averages Table:
--------------------------------------------------------------------------------
    Model  Versions_Aggregated  Avg_Pass@1  Avg_AUROC  Avg_T@0%F  Avg_T@1%F  Avg_T@5%F  Avg_T@10%F  Avg_CodeBLEU_Mean
  Avg_CodeBLEU_Std  Total_Examples
  codegemma                    2       68.83     0.4712        1.0        1.0        6.0        10.0              22.44
              9.09              10
```

### JSON Output (Optional)
```json
{
  "pass_at_1": 0.6833,
  "auroc": 0.4712,
  "tpr_values": {
    "T@0%F": 0.0100,
    "T@1%F": 0.0100,
    "T@5%F": 0.0600,
    "T@10%F": 0.1000
  },
  "codebleu": {
    "mean": 0.2808,
    "std": 0.1956,
    "min": 0.0777,
    "max": 0.6714,
    "count": 100
  }
}
```

### CSV Output Files
The batch evaluation generates multiple CSV files:

1. **`[output]_comparison.csv`** - Experiment-level results (all individual evaluations)
2. **`[output]_model_averages.csv`** - Model-level averages (when multiple versions exist)

CSV columns include: Model_Experiment, Pass@1, AUROC, T@0%F, T@1%F, T@5%F, T@10%F, CodeBLEU_Mean, CodeBLEU_Std, Examples_Evaluated

## 🔧 Advanced Options

### CodeBLEU Performance Tuning
- Use `--codebleu_sample_size N` for faster evaluation on N examples
- Use `--full_codebleu` for complete evaluation (slower but comprehensive)
- CodeBLEU is the most computationally expensive metric

### Batch Evaluation Features
- Automatically generates comparison tables
- Exports both detailed JSON and summary CSV
- Identifies best performers across metrics

### Validation and Debugging
```bash
# Validate data structure before evaluation
python eval_utils.py --validate \
    --csv_file path/to/file.csv \
    --generated_dir path/to/generated \
    --reference_file path/to/references.json

# Show all usage examples
python eval_utils.py --examples
```

## 🏗️ File Structure Expected

```
your_project/
├── results/raw/
│   └── exp1_[model]_during_gen_v1_100.csv
├── output/
│   └── [model]_[experiment]/
│       ├── 57.py
│       ├── 590.py
│       └── ...
└── datasets/core/
    └── sanitized-mbpp-sample-100.json
```

## 🎯 Metrics Explanation

### Pass@1 (Execution Success)
- Measures functional correctness
- Percentage of generated code passing all test cases
- Higher is better (0.0 to 1.0)

### AUROC (Watermark Detection)
- Measures distinguishability between human and AI-generated code
- Based on z-scores from watermark detection
- Range: 0.0 to 1.0 (0.5 = random, 1.0 = perfect detection)

### T@X%F (True Positive Rate)
- TPR when False Positive Rate ≤ X%
- Critical for watermark detection applications
- Higher values indicate better detection with low false alarms

### CodeBLEU (Code Similarity)
- Combines multiple code similarity measures:
  - BLEU score (token-level similarity)
  - Weighted BLEU (keyword-aware)
  - Syntax tree matching
  - Dataflow analysis
- Range: 0.0 to 1.0 (higher = more similar to reference)

## 🛠️ Dependencies

- pandas
- numpy
- scikit-learn
- tree-sitter (for CodeBLEU syntax analysis)
- Custom CodeBLEU implementation (included)

## 🚨 Troubleshooting

### Common Issues

1. **Missing Z-scores**: AUROC calculation skipped
   - Solution: Ensure CSV has `original_z_score` and `generated_z_score` columns

2. **CodeBLEU errors**: Tree-sitter parser issues
   - Solution: Check if Python tree-sitter parser is properly compiled

3. **File not found**: Path issues
   - Solution: Use `--list_available` to see valid model/experiment combinations

4. **Performance issues**: CodeBLEU slow on large datasets
   - Solution: Use `--codebleu_sample_size` to limit evaluation scope

### Debugging Commands
```bash
# Check what's available
python batch_eval.py --list_available

# Validate your data
python eval_utils.py --validate --csv_file X --generated_dir Y --reference_file Z

# Test with small sample
python quick_eval.py --model MODEL --experiment EXPERIMENT --codebleu_sample_size 5
```

## 📚 Related Files

This toolkit builds upon and integrates with:
- `calculate_auroc_v2.py` - AUROC calculation utilities
- `python_codebleu_helper.py` - CodeBLEU evaluation helpers  
- `calc_code_bleu.py` - Core CodeBLEU implementation
- Enhanced metrics from the `metrics/` directory

## 🤝 Contributing

To extend the toolkit:
1. Add new metrics to `ComprehensiveEvaluator` class
2. Update the output formatting in `print_summary_report`
3. Add corresponding command-line options
4. Update this README with new features

---

For more examples and detailed documentation, run:
```bash
python eval_utils.py --examples
```