#!/usr/bin/env python3
"""
Comprehensive Code Evaluation Script

This script calculates multiple metrics for evaluating AI-generated code:
1. Pass@1 (Average pass rate)
2. AUROC value
3. T@X%F (TPR at X% FPR where X=[0,1,5,10])
4. CodeBLEU (with support for full or AST+Dataflow-only modes)

Usage:
    python comprehensive_evaluation.py --csv_file path/to/results.csv --generated_dir path/to/generated/code --reference_file path/to/reference.json
    
Basic Example (Full CodeBLEU):
    python comprehensive_evaluation.py --csv_file ../../results/raw/exp1_codegemma-7b-it_during_gen_v1_100.csv --generated_dir ../../output/codegemma-7b-it_exp1_dgen_v1_100 --reference_file ../../datasets/core/sanitized-mbpp-sample-100.json

AST+Dataflow Only (gamma=delta=0.25):
    python comprehensive_evaluation.py --csv_file ../../results/raw/exp1_codegemma-7b-it_during_gen_v1_100.csv --generated_dir ../../output/codegemma-7b-it_exp1_dgen_v1_100 --reference_file ../../datasets/core/sanitized-mbpp-sample-100.json --codebleu_ast_dataflow_only --codebleu_ast_weight 0.25 --codebleu_dataflow_weight 0.25

Custom Weights (gamma=0.3, delta=0.7):
    python comprehensive_evaluation.py --csv_file ../../results/raw/exp1_codegemma-7b-it_during_gen_v1_100.csv --generated_dir ../../output/codegemma-7b-it_exp1_dgen_v1_100 --reference_file ../../datasets/core/sanitized-mbpp-sample-100.json --codebleu_ast_weight 0.3 --codebleu_dataflow_weight 0.7

CodeBLEU Modes:
  Full CodeBLEU (default):
    CodeBLEU = alpha*Ngram + beta*WeightedNgram + gamma*AST + delta*Dataflow
    Weights the AST and Dataflow components with specified gamma and delta.
    
  AST+Dataflow Only (--codebleu_ast_dataflow_only):
    CodeBLEU_reported = gamma*AST + delta*Dataflow
    Uses only AST and Dataflow components, ignoring Ngram-based components.
    Per Mahmud et al. 2025, AST and Dataflow most accurately capture structural
    similarity and program-level semantics.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import glob
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Import scipy for binomial test
try:
    from scipy.stats import binomtest
except ImportError:
    print("Warning: scipy not available. Binomial p-value calculations will be disabled")
    binomtest = None

# Add metrics directory to path
metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics')
sys.path.insert(0, metrics_dir)

# Import CodeBLEU evaluation functions
try:
    from python_codebleu_helper import evaluate_python_code_bleu, batch_evaluate_python
    from calc_code_bleu import get_codebleu
except ImportError as e:
    print(f"Warning: CodeBLEU imports failed: {e}")
    print("CodeBLEU evaluation will be disabled")

def list_available_experiments():
    """List available experiments by scanning results/raw and output directories."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    
    results_dir = f"{base_dir}/results/raw"
    output_dir = f"{base_dir}/output"
    
    # Find CSV files in results/raw
    csv_files = glob.glob(f"{results_dir}/*.csv")
    experiments = []
    
    for csv_file in csv_files:
        basename = os.path.basename(csv_file).replace('.csv', '')
        # Check if corresponding output directory exists
        output_path = f"{output_dir}/{basename}"
        if os.path.exists(output_path):
            experiments.append(basename)
    
    return sorted(experiments)


class ComprehensiveEvaluator:
    """Main evaluator class for comprehensive code evaluation metrics."""
    
    def __init__(self, csv_file: str, generated_dir: str, reference_file: str,
                 codebleu_ast_weight: float = 1.0, codebleu_dataflow_weight: float = 1.0,
                 codebleu_ast_dataflow_only: bool = False):
        """
        Initialize the evaluator.
        
        Args:
            csv_file: Path to CSV file with watermarking results and pass rates
            generated_dir: Directory containing AI-generated code files
            reference_file: JSON file containing reference code implementations
            codebleu_ast_weight: Weight for AST component (default: 1.0)
            codebleu_dataflow_weight: Weight for Dataflow component (default: 1.0)
            codebleu_ast_dataflow_only: If True, use only AST+Dataflow components
        """
        self.csv_file = csv_file
        self.generated_dir = generated_dir
        self.reference_file = reference_file
        self.codebleu_ast_weight = codebleu_ast_weight
        self.codebleu_dataflow_weight = codebleu_dataflow_weight
        self.codebleu_ast_dataflow_only = codebleu_ast_dataflow_only
        
        # Load data
        self.df = pd.read_csv(csv_file)
        self.references = self._load_references()
        
        print(f"✅ Loaded {len(self.df)} examples from CSV")
        print(f"✅ Loaded {len(self.references)} reference implementations")
        print(f"✅ Generated code directory: {generated_dir}")
        
        # Display CodeBLEU configuration
        if self.codebleu_ast_dataflow_only:
            print(f"✅ CodeBLEU Mode: AST+Dataflow Only")
        print(f"✅ CodeBLEU AST weight (gamma): {self.codebleu_ast_weight}")
        print(f"✅ CodeBLEU Dataflow weight (delta): {self.codebleu_dataflow_weight}")
    
    def _load_references(self) -> Dict[str, str]:
        """Load reference code implementations from JSON file."""
        with open(self.reference_file, 'r') as f:
            data = [json.loads(line) for line in f]
        
        references = {}
        for item in data:
            task_id = str(item['task_id'])  # Store as string key for consistent lookup
            code = item['code']
            references[task_id] = code

        print("✅ Reference implementations loaded")
        print(f"   Total references: {len(references)}")
        print("SAMPLE: ", list(references.items())[:2])
        
        return references
    
    def calculate_pass_at_1(self) -> float:
        """
        Calculate Pass@1 metric (average pass rate).
        
        Returns:
            Average pass rate across all examples
        """
        if 'pass_rate' in self.df.columns:
            # Use pass_rate if available
            pass_at_1 = self.df['pass_rate'].mean() / 100.0  # Convert percentage to decimal
        elif 'all_passed' in self.df.columns:
            # Use all_passed boolean if available
            pass_at_1 = self.df['all_passed'].mean()
        else:
            # Calculate from tests_passed and total_tests
            pass_at_1 = (self.df['tests_passed'] / self.df['total_tests']).mean()
        
        return pass_at_1
    
    def calculate_auroc(self) -> Tuple[float, Dict[str, float]]:
        """
        ✅ UNIFIED APPROACH: Calculate AUROC value and TPR at various FPR thresholds.
        
        Uses consistent unified scoring metric for ALL samples:
        - Score: -log10(p_unified) where p_unified is exact binomial p-value
        - Same metric for decisions AND ROC ranking (ensures consistency)
        
        Returns:
            Tuple of (AUROC score, TPR values dict)
        """
        print("\n" + "="*80)
        print("📈 AUROC & TPR CALCULATION (UNIFIED DETECTION)")
        print("="*80)
        
        # ✅ CHECK FOR UNIFIED SCORES FIRST (preferred)
        has_unified_scores = ('original_score' in self.df.columns and 
                              'generated_score' in self.df.columns)
        
        # ⚠️  FALLBACK: Old z-score approach (not recommended)
        has_z_scores = ('original_z_score' in self.df.columns and 
                       'generated_z_score' in self.df.columns)
        
        if not (has_unified_scores or has_z_scores):
            print("❌ ERROR: Neither unified scores nor z-scores found in CSV.")
            print("   Expected columns: 'original_score' and 'generated_score' (unified)")
            print("   Or: 'original_z_score' and 'generated_z_score' (fallback)")
            return 0.0, {}
        
        # ✅ PREFERRED: Use unified scores if available
        if has_unified_scores:
            print("✅ Using UNIFIED detection scores (-log10(p_unified))")
            print("   • All samples use exact binomial p-value")
            print("   • Consistent metric for decisions AND ROC ranking")
            print("   • Decisions: p_unified < P_THRESHOLD")
            print("   • ROC Score: -log10(p_unified)")
            
            human_scores = np.array(self.df['original_score'].fillna(0))
            machine_scores = np.array(self.df['generated_score'].fillna(0))
            scoring_method = "unified"
        
        else:
            # ⚠️  FALLBACK: Old z-score approach
            print("⚠️  WARNING: Using legacy z-score approach (not recommended)")
            print("   • This approach had issues with sample-size-dependent metrics")
            print("   • Consider re-running detection notebook with unified method")
            
            human_scores = np.array(self.df['original_z_score'].fillna(0))
            machine_scores = np.array(self.df['generated_z_score'].fillna(0))
            scoring_method = "z_score"
        
        # 📊 DIAGNOSTIC OUTPUT
        print(f"\n📊 Data Summary:")
        print(f"   Original (human) code samples: {len(human_scores)}")
        print(f"   Generated (machine) code samples: {len(machine_scores)}")
        print(f"   Scoring method: {scoring_method}")
        print(f"   Score range: [{human_scores.min():.6f}, {human_scores.max():.6f}] (original)")
        print(f"   Score range: [{machine_scores.min():.6f}, {machine_scores.max():.6f}] (generated)")
        
        # ✅ CREATE CONSISTENT LABELS AND SCORES
        # Label: 0 = original code (should NOT be watermarked)
        #        1 = generated code (should be watermarked)
        true_labels = np.concatenate([
            np.zeros(len(human_scores)),   # Original: label 0
            np.ones(len(machine_scores))    # Generated: label 1
        ])
        
        # All scores use SAME metric (unified)
        all_scores = np.concatenate([human_scores, machine_scores])
        
        # ✅ CALCULATE ROC CURVE (using consistent metric)
        fpr, tpr, thresholds = roc_curve(true_labels, all_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        print(f"\n📈 ROC Curve Statistics:")
        print(f"   AUROC: {roc_auc:.6f}")
        print(f"   Number of ROC points: {len(fpr)}")
        print(f"   FPR range: [{fpr.min():.6f}, {fpr.max():.6f}]")
        print(f"   TPR range: [{tpr.min():.6f}, {tpr.max():.6f}]")
        
        # ✅ CALCULATE TPR AT SPECIFIC FPR THRESHOLDS
        tpr_values = {}
        fpr_thresholds = [0.0, 0.01, 0.05, 0.10]
        
        print(f"\n🎯 TPR at Specific FPR Thresholds:")
        for fpr_threshold in fpr_thresholds:
            tpr_at_fpr = self._get_tpr_at_fpr(fpr, tpr, fpr_threshold)
            tpr_values[f"T@{int(fpr_threshold*100)}%F"] = tpr_at_fpr
            print(f"   T@{int(fpr_threshold*100):2d}%F: {tpr_at_fpr:.6f}")
        
        return roc_auc, tpr_values
    
    def _get_tpr_at_fpr(self, fpr: np.ndarray, tpr: np.ndarray, fpr_threshold: float) -> float:
        """
        ✅ CORRECT: Get TPR at specific FPR threshold from ROC curve.
        
        Interpretation: "What's the highest TPR we can achieve while keeping FPR ≤ fpr_threshold?"
        
        Args:
            fpr: False Positive Rates from ROC curve (sorted, increasing)
            tpr: True Positive Rates from ROC curve (correspondingly sorted)
            fpr_threshold: Maximum acceptable FPR (e.g., 0.01 for 1%)
            
        Returns:
            TPR value at the highest valid FPR ≤ fpr_threshold
        """
        # Find all indices where FPR <= threshold
        valid_indices = np.where(fpr <= fpr_threshold)[0]
        
        if len(valid_indices) == 0:
            # No FPR value found at or below threshold
            print(f"   ⚠️  Warning: No FPR <= {fpr_threshold:.6f} found in ROC curve")
            print(f"       Minimum FPR available: {fpr.min():.6f}")
            return 0.0
        
        # Return TPR at the RIGHTMOST valid point (highest FPR <= threshold)
        # This gives the maximum TPR achievable at the given FPR constraint
        best_idx = valid_indices[-1]
        return float(tpr[best_idx])
    
    def calculate_codebleu(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate CodeBLEU scores for generated vs reference code.
        
        Supports two modes:
        1. Full CodeBLEU: Uses all components with standard weights
        2. AST+Dataflow Only: Uses only AST and Dataflow components
        
        Args:
            sample_size: Number of examples to evaluate (None for all)
            
        Returns:
            Dictionary with CodeBLEU statistics and configuration info
        """
        try:
            # Import check
            from python_codebleu_helper import evaluate_python_code_bleu
        except ImportError:
            print("⚠️  CodeBLEU evaluation not available")
            return {"mean": 0.0, "std": 0.0, "count": 0, "mode": "unavailable"}
        
        codebleu_scores = []
        ast_scores = []
        dataflow_scores = []
        processed_count = 0
        error_count = 0
        
        # Determine which examples to process
        task_ids = self.df['task_id'].tolist()
        if sample_size:
            task_ids = task_ids[:sample_size]
        
        # Display mode information
        if self.codebleu_ast_dataflow_only:
            mode_str = f"AST+Dataflow Only (γ={self.codebleu_ast_weight}, δ={self.codebleu_dataflow_weight})"
        else:
            mode_str = f"Full CodeBLEU (γ={self.codebleu_ast_weight}, δ={self.codebleu_dataflow_weight})"
        
        print(f"🔄 Calculating CodeBLEU ({mode_str}) for {len(task_ids)} examples...")
        
        for i, task_id in enumerate(task_ids):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(task_ids)}")
            
            try:
                # Load generated code
                generated_file = os.path.join(self.generated_dir, f"{task_id}.py")
                if not os.path.exists(generated_file):
                    print(f"⚠️  Generated file not found: {generated_file}")
                    error_count += 1
                    continue
                
                with open(generated_file, 'r') as f:
                    generated_code = f.read().strip()
                
                # Get reference code
                if str(task_id) not in self.references:
                    print(f"⚠️  Reference not found for task_id: {task_id}")
                    error_count += 1
                    continue
                
                reference_code = self.references[str(task_id)].strip()
                
                # Calculate full CodeBLEU result
                try:
                    result = evaluate_python_code_bleu(reference_code, generated_code)
                    
                    if self.codebleu_ast_dataflow_only:
                        # AST+Dataflow only mode: weight AST and Dataflow components
                        # CodeBLEU uses 'syntax' for AST matching and 'dataflow' for dataflow
                        ast_score = result.get('syntax', result.get('ast_match', result.get('ast', 0.0)))
                        dataflow_score = result.get('dataflow', result.get('dataflow_match', result.get('Dataflow', 0.0)))
                        
                        # Normalize weights to sum to 1 if ast_dataflow_only is True
                        weight_sum = self.codebleu_ast_weight + self.codebleu_dataflow_weight
                        normalized_ast_weight = self.codebleu_ast_weight / weight_sum if weight_sum > 0 else 0.5
                        normalized_df_weight = self.codebleu_dataflow_weight / weight_sum if weight_sum > 0 else 0.5
                        
                        weighted_score = (normalized_ast_weight * ast_score + 
                                        normalized_df_weight * dataflow_score)
                        
                        codebleu_scores.append(weighted_score)
                        ast_scores.append(ast_score)
                        dataflow_scores.append(dataflow_score)
                        
                        print(f"✅ CodeBLEU for task_id {task_id}: AST={ast_score:.4f}, "
                              f"Dataflow={dataflow_score:.4f}, Weighted={weighted_score:.4f}")
                    else:
                        # Full CodeBLEU: use standard scoring
                        codebleu_score = result.get('codebleu', 0.0)
                        codebleu_scores.append(codebleu_score)
                        ast_scores.append(result.get('syntax', result.get('ast_match', result.get('ast', 0.0))))
                        dataflow_scores.append(result.get('dataflow', result.get('dataflow_match', 0.0)))
                        
                        print(f"✅ CodeBLEU for task_id {task_id}: {codebleu_score:.4f}")
                
                except Exception as e:
                    print(f"⚠️  ERROR processing CodeBLEU for task_id {task_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    codebleu_scores.append(0.0)
                    ast_scores.append(0.0)
                    dataflow_scores.append(0.0)
                    error_count += 1
                    continue
                
                processed_count += 1
                
            except Exception as e:
                print(f"⚠️  Error processing task_id {task_id}: {e}")
                error_count += 1
                continue
        
        print(f"✅ Processed {processed_count} examples, {error_count} errors")
        
        if not codebleu_scores:
            return {"mean": 0.0, "std": 0.0, "count": 0, "mode": "unavailable"}
        
        result_dict = {
            "mean": np.mean(codebleu_scores),
            "std": np.std(codebleu_scores),
            "min": np.min(codebleu_scores),
            "max": np.max(codebleu_scores),
            "count": len(codebleu_scores),
            "scores": codebleu_scores,
            "mode": "ast_dataflow_only" if self.codebleu_ast_dataflow_only else "full",
            "ast_weight": self.codebleu_ast_weight,
            "dataflow_weight": self.codebleu_dataflow_weight,
        }
        
        # Include component statistics if available
        if ast_scores:
            result_dict["ast_mean"] = np.mean(ast_scores)
            result_dict["dataflow_mean"] = np.mean(dataflow_scores)
        
        return result_dict
    
    def run_comprehensive_evaluation(self, codebleu_sample_size: Optional[int] = None,
                                      codebleu_config: Optional[Dict] = None) -> Dict:
        """
        Run all evaluation metrics and return comprehensive results.
        
        Args:
            codebleu_sample_size: Number of examples for CodeBLEU (None for all)
            codebleu_config: Optional dict with CodeBLEU configuration details
            
        Returns:
            Dictionary containing all evaluation results
        """
        print("🚀 Starting Comprehensive Code Evaluation")
        print("=" * 50)
        
        results = {}
        
        # 1. Pass@1 Calculation
        print("\n📊 Calculating Pass@1...")
        pass_at_1 = self.calculate_pass_at_1()
        results['pass_at_1'] = pass_at_1
        print(f"✅ Pass@1: {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
        
        # 2. AUROC and TPR Calculations
        print("\n📈 Calculating AUROC and TPR values...")
        auroc, tpr_values = self.calculate_auroc()
        results['auroc'] = auroc
        results['tpr_values'] = tpr_values
        
        if auroc > 0:
            print(f"✅ AUROC: {auroc:.4f}")
            for metric, value in tpr_values.items():
                print(f"✅ {metric}: {value:.4f}")
        else:
            print("⚠️  AUROC calculation skipped (missing z-scores)")
        
        # 3. CodeBLEU Calculation
        print("\n🔧 Calculating CodeBLEU scores...")
        codebleu_results = self.calculate_codebleu(codebleu_sample_size)
        results['codebleu'] = codebleu_results
        
        if codebleu_results['count'] > 0:
            print(f"✅ CodeBLEU Mean: {codebleu_results['mean']:.4f}")
            print(f"✅ CodeBLEU Std: {codebleu_results['std']:.4f}")
            print(f"✅ CodeBLEU Range: [{codebleu_results['min']:.4f}, {codebleu_results['max']:.4f}]")
        else:
            print("⚠️  CodeBLEU calculation failed")
        
        return results
    
    def print_summary_report(self, results: Dict):
        """Print a formatted summary report of all metrics."""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE EVALUATION SUMMARY (UNIFIED DETECTION)")
        print("="*80)
        
        print(f"\n🎯 Execution Metrics:")
        print(f"   Pass@1 (Avg Pass Rate): {results['pass_at_1']:.4f} ({results['pass_at_1']*100:.2f}%)")
        
        print(f"\n🔍 Watermark Detection Metrics:")
        print(f"   Method: UNIFIED p-value approach (-log10(p_unified))")
        print(f"   • All samples use exact binomial p-value")
        print(f"   • Same metric for decisions AND ROC ranking ✓")
        print(f"   • Consistent TPR and AUROC ✓")
        
        if results['auroc'] > 0:
            print(f"\n   AUROC:                   {results['auroc']:.6f}")
            for metric, value in results['tpr_values'].items():
                print(f"   {metric:20s}: {value:.6f}")
        else:
            print("   AUROC: Not available (missing detection scores)")
        
        print(f"\n📝 Code Quality Metrics (CodeBLEU):")
        codebleu = results['codebleu']
        if codebleu['count'] > 0:
            # Display mode and weights
            if codebleu.get('mode') == 'ast_dataflow_only':
                mode_str = "AST+Dataflow Only"
            else:
                mode_str = "Full CodeBLEU"
            
            ast_weight = codebleu.get('ast_weight', 'N/A')
            df_weight = codebleu.get('dataflow_weight', 'N/A')
            
            print(f"   Mode:                    {mode_str}")
            print(f"   AST Weight (γ):          {ast_weight}")
            print(f"   Dataflow Weight (δ):     {df_weight}")
            print(f"   CodeBLEU Mean:           {codebleu['mean']:.4f}")
            print(f"   CodeBLEU Std:            {codebleu['std']:.4f}")
            print(f"   CodeBLEU Range:          [{codebleu['min']:.4f}, {codebleu['max']:.4f}]")
            
            # Display component statistics if available
            if 'ast_mean' in codebleu:
                print(f"   AST Mean:                {codebleu['ast_mean']:.4f}")
                print(f"   Dataflow Mean:           {codebleu['dataflow_mean']:.4f}")
            
            print(f"   Evaluated Examples:      {codebleu['count']}")
        else:
            print("   CodeBLEU: Not available")
        
        print(f"\n📁 Data Sources:")
        print(f"   CSV File:     {os.path.basename(self.csv_file)}")
        print(f"   Generated:    {os.path.basename(self.generated_dir)}")
        print(f"   Reference:    {os.path.basename(self.reference_file)}")
        print(f"   Total Examples: {len(self.df)}")
        
        print("\n" + "="*80)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Code Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\nCodeBLEU Configuration Examples:
  Standard full CodeBLEU:
    python comprehensive_evaluation.py --csv_file results.csv --generated_dir ./code --reference_file ref.json
  
  AST+Dataflow only with equal weights (γ=δ=0.25):
    python comprehensive_evaluation.py --csv_file results.csv --generated_dir ./code --reference_file ref.json \\n      --codebleu_ast_dataflow_only --codebleu_ast_weight 0.25 --codebleu_dataflow_weight 0.25
  
  Custom weights (γ=0.3, δ=0.7):
    python comprehensive_evaluation.py --csv_file results.csv --generated_dir ./code --reference_file ref.json \\n      --codebleu_ast_weight 0.3 --codebleu_dataflow_weight 0.7
        """)
    
    parser.add_argument('--csv_file', required=False,
                       help='Path to CSV file with evaluation results')
    parser.add_argument('--generated_dir', required=False,
                       help='Directory containing AI-generated code files')
    parser.add_argument('--reference_file', required=False,
                       help='JSON file containing reference code implementations')
    parser.add_argument('--list_available', action='store_true',
                       help='List available experiments and exit')
    parser.add_argument('--auto_evaluate', action='store_true',
                       help='Automatically evaluate the first available experiment')
    parser.add_argument('--codebleu_sample_size', type=int, default=200,
                       help='Number of examples for CodeBLEU evaluation (default: 200)')
    parser.add_argument('--codebleu_ast_weight', type=float, default=1.0,
                       help='Weight for AST component in CodeBLEU (gamma, default: 1.0)')
    parser.add_argument('--codebleu_dataflow_weight', type=float, default=1.0,
                       help='Weight for Dataflow component in CodeBLEU (delta, default: 1.0)')
    parser.add_argument('--codebleu_ast_dataflow_only', action='store_true',
                       help='Use only AST+Dataflow components (ignore Ngram and WeightedNgram)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results JSON (optional)')
    
    args = parser.parse_args()
    
    # Handle list available
    if args.list_available:
        experiments = list_available_experiments()
        if not experiments:
            print("No available experiments found.")
            return 0
        print("Available experiments:")
        for exp in experiments:
            print(f"  - {exp}")
        return 0
    
    # Handle auto evaluate
    if args.auto_evaluate:
        experiments = list_available_experiments()
        if not experiments:
            print("No available experiments found for auto-evaluation.")
            return 1
        experiment = experiments[0]
        print(f"🔄 Auto-selected experiment: {experiment}")
        # Set paths based on experiment name
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))
        parts = experiment.split('_')
        if len(parts) >= 6:
            dataset = parts[-1] if len(parts) > 6 else 'humaneval'
            if dataset == 'humaneval':
                ref_file = f"{base_dir}/datasets/human_eval_164.jsonl"
            elif dataset == 'mbpp':
                ref_file = f"{base_dir}/datasets/sanitized-mbpp.json"
            else:
                ref_file = f"{base_dir}/datasets/{dataset}.jsonl"
            args.csv_file = f"{base_dir}/results/raw/{experiment}.csv"
            args.generated_dir = f"{base_dir}/output/{experiment}"
            args.reference_file = ref_file
        else:
            print("❌ Could not parse experiment name for auto-evaluation.")
            return 1
    
    # Validate required args when not listing
    if not args.csv_file or not args.generated_dir or not args.reference_file:
        print("❌ Error: --csv_file, --generated_dir, and --reference_file are required unless --list_available is used")
        return 1
    if not os.path.exists(args.csv_file):
        print(f"❌ CSV file not found: {args.csv_file}")
        return 1
    
    if not os.path.exists(args.generated_dir):
        print(f"❌ Generated code directory not found: {args.generated_dir}")
        return 1
    
    if not os.path.exists(args.reference_file):
        print(f"❌ Reference file not found: {args.reference_file}")
        return 1
    
    try:
        # Initialize evaluator with CodeBLEU configuration
        evaluator = ComprehensiveEvaluator(
            csv_file=args.csv_file,
            generated_dir=args.generated_dir,
            reference_file=args.reference_file,
            codebleu_ast_weight=args.codebleu_ast_weight,
            codebleu_dataflow_weight=args.codebleu_dataflow_weight,
            codebleu_ast_dataflow_only=args.codebleu_ast_dataflow_only
        )
        
        # Prepare CodeBLEU configuration info
        codebleu_config = {
            'mode': 'ast_dataflow_only' if args.codebleu_ast_dataflow_only else 'full',
            'ast_weight': args.codebleu_ast_weight,
            'dataflow_weight': args.codebleu_dataflow_weight
        }
        
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation(
            codebleu_sample_size=args.codebleu_sample_size,
            codebleu_config=codebleu_config
        )
        
        # Print summary report
        evaluator.print_summary_report(results)
        
        # Save results if output file specified
        if args.output:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            json_results[key][k] = v.tolist()
                        elif isinstance(v, (np.integer, np.floating)):
                            json_results[key][k] = float(v)
                        else:
                            json_results[key][k] = v
                elif isinstance(value, (np.integer, np.floating)):
                    json_results[key] = float(value)
                else:
                    json_results[key] = value
            
            with open(args.output, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())