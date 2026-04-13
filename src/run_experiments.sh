#!/bin/bash

# ============================================================================
# Watermarking Experiments Runner
# ============================================================================
# This script runs all watermarking experiment methods:
#   - expT: Code as text only (no watermarking)
#   - expS: Static watermarking (during generation)
#   - expA: Comment-based watermarking (during generation)
#   - expI: Iterative watermarking (with feedback loops)
#   - expX: Refactoring-based watermarking (post-hoc)
#
# Usage:
#   ./run_experiments.sh [METHOD] [DATASET] [SAMPLE_SIZE]
#
# Examples:
#   ./run_experiments.sh all humaneval 100      # Run all methods
#   ./run_experiments.sh expS humaneval 100     # Run static watermarking only
#   ./run_experiments.sh expT,expS,expA humaneval 50  # Run multiple methods

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="$BASE_DIR/datasets"
OUTPUT_DIR="$BASE_DIR/output"
RESULTS_DIR="$BASE_DIR/results/raw"

# Active model configuration for experiment runs
ACTIVE_PROVIDER="claude"  # claude, gemini
ACTIVE_MODEL="${DEFAULT_MODEL:-us.anthropic.claude-sonnet-4-20250514-v1:0}"

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║ $1"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

run_experiment() {
    local exp_name=$1
    local module_name=$2
    local dataset_path=$3
    local sample_size=$4
    local dataset_type=$5

    print_section "Running $exp_name Experiment"
    
    local output_dir="$OUTPUT_DIR/claude_${exp_name}_during_gen_v1_${sample_size}_${dataset_type}"
    local results_csv="$RESULTS_DIR/claude_${exp_name}_during_gen_v1_${sample_size}_${dataset_type}.csv"
    
    # Special case for refactoring which uses "only-gen"
    if [ "$exp_name" = "expX" ]; then
        output_dir="$OUTPUT_DIR/claude_${exp_name}_only-gen_gen_v1_${sample_size}_${dataset_type}"
        results_csv="$RESULTS_DIR/claude_${exp_name}_only-gen_gen_v1_${sample_size}_${dataset_type}.csv"
    fi

    echo "📊 Experiment: $exp_name"
    echo "📁 Output Directory: $output_dir"
    echo "📄 Results CSV: $results_csv"
    echo "🔧 Dataset: $dataset_path"
    echo "🤖 Provider: $ACTIVE_PROVIDER"
    echo "🧠 Model: $ACTIVE_MODEL"
    echo ""

    # Run the Python module from src/watermarking and include src on PYTHONPATH
    local module_path="$SCRIPT_DIR/watermarking/${module_name}.py"
    if [ ! -f "$module_path" ]; then
        echo "❌ Error: Experiment module not found at $module_path"
        exit 1
    fi

    (
        cd "$BASE_DIR"
        LLM_PROVIDER="$ACTIVE_PROVIDER" \
        PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            python "$module_path" "$dataset_path" "$output_dir" "$results_csv" "$sample_size"
    )
    
    echo ""
    echo "✅ $exp_name experiment completed!"
    echo ""
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

# Parse command line arguments
METHODS="${1:-all}"
DATASET="${2:-humaneval}"
SAMPLE_SIZE="${3:-100}"

# Validate sample size
if ! [[ "$SAMPLE_SIZE" =~ ^[0-9]+$ ]] || [ "$SAMPLE_SIZE" -le 0 ]; then
    echo "❌ Error: SAMPLE_SIZE must be a positive integer (got '$SAMPLE_SIZE')"
    exit 1
fi

# Resolve dataset path
if [ "$DATASET" = "humaneval" ]; then
    DATASET_PATH="$DATASET_DIR/humaneval_164.jsonl"
    DATASET_TYPE="humaneval"
elif [ "$DATASET" = "mbpp" ]; then
    DATASET_PATH="$DATASET_DIR/sanitized-mbpp-sample-100.json"
    DATASET_TYPE="mbpp"
else
    DATASET_PATH="$DATASET"
    DATASET_TYPE="custom"
fi

# Validate dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found at $DATASET_PATH"
    exit 1
fi

print_header "Watermarking Experiments Runner"
echo "📌 Configuration:"
echo "   Methods: $METHODS"
echo "   Dataset: $DATASET_TYPE ($DATASET_PATH)"
echo "   Sample Size: $SAMPLE_SIZE"
echo "   Provider: $ACTIVE_PROVIDER"
echo "   Model: $ACTIVE_MODEL"
echo "   Base Directory: $BASE_DIR"
echo ""

# Array of all available experiments
declare -A EXPERIMENTS=(
    ["expT"]="exp_code_only"
    ["expS"]="exp_static_wm"
    ["expA"]="exp_comment_wm"
    ["expI"]="exp_iterative_wm"
    ["expX"]="exp_refactoring_wm"
)

# Determine which experiments to run
if [ "$METHODS" = "all" ]; then
    METHODS_TO_RUN=("expT" "expS" "expA" "expI" "expX")
else
    # Split comma-separated methods
    IFS=',' read -ra METHODS_TO_RUN <<< "$METHODS"
fi

# Validate requested methods
for method in "${METHODS_TO_RUN[@]}"; do
    method=$(echo "$method" | xargs)  # Trim whitespace
    if [ -z "${EXPERIMENTS[$method]}" ]; then
        echo "❌ Error: Unknown experiment method '$method'"
        echo "   Available methods: expT, expS, expA, expI, expX, all"
        exit 1
    fi
done

# ============================================================================
# Run Experiments
# ============================================================================

TOTAL_METHODS=${#METHODS_TO_RUN[@]}
CURRENT_METHOD=0

print_header "Starting Experiments ($(date))"

START_TIME=$(date +%s)

for method in "${METHODS_TO_RUN[@]}"; do
    method=$(echo "$method" | xargs)  # Trim whitespace
    CURRENT_METHOD=$((CURRENT_METHOD + 1))
    
    module_name="${EXPERIMENTS[$method]}"
    echo "[$CURRENT_METHOD/$TOTAL_METHODS] Running experiment: $method"
    
    run_experiment "$method" "$module_name" "$DATASET_PATH" "$SAMPLE_SIZE" "$DATASET_TYPE"
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_TIME / 60))
TOTAL_SECONDS=$((TOTAL_TIME % 60))

print_header "Experiments Completed ✅"
echo "📊 Summary:"
echo "   Total Methods: $TOTAL_METHODS"
echo "   Total Time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "   Results Location: $RESULTS_DIR"
echo "   Output Location: $OUTPUT_DIR"
echo ""
echo "📁 Generated Files:"
for method in "${METHODS_TO_RUN[@]}"; do
    method=$(echo "$method" | xargs)
    if [ "$method" = "expX" ]; then
        csv_file="$RESULTS_DIR/claude_${method}_only-gen_gen_v1_${SAMPLE_SIZE}_${DATASET_TYPE}.csv"
    else
        csv_file="$RESULTS_DIR/claude_${method}_during_gen_v1_${SAMPLE_SIZE}_${DATASET_TYPE}.csv"
    fi
    if [ -f "$csv_file" ]; then
        echo "   ✅ $csv_file"
    fi
done
echo ""
echo "🎉 All experiments completed successfully!"
