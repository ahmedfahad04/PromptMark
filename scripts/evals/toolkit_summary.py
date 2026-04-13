#!/usr/bin/env python3
"""
Code Evaluation Toolkit Summary

Quick reference and testing script for the comprehensive code evaluation toolkit.
"""

import os
import sys

def print_toolkit_summary():
    """Print a comprehensive summary of the evaluation toolkit."""
    
    print("🚀 COMPREHENSIVE CODE EVALUATION TOOLKIT")
    print("=" * 60)
    
    print("\n📋 WHAT THIS TOOLKIT PROVIDES:")
    print("-" * 35)
    metrics = [
        ("Pass@1", "Average pass rate of generated code", "68.33%"),
        ("AUROC", "Watermark detection capability", "0.4712"),
        ("T@0%F", "True Positive Rate at 0% False Positive Rate", "0.0100"),
        ("T@1%F", "True Positive Rate at 1% False Positive Rate", "0.0100"), 
        ("T@5%F", "True Positive Rate at 5% False Positive Rate", "0.0600"),
        ("T@10%F", "True Positive Rate at 10% False Positive Rate", "0.1000"),
        ("CodeBLEU", "Code similarity vs reference implementation", "0.2808")
    ]
    
    for metric, description, example in metrics:
        print(f"✅ {metric:12s} - {description:45s} (e.g., {example})")
    
    print("\n🛠️  AVAILABLE SCRIPTS:")
    print("-" * 25)
    scripts = [
        ("comprehensive_evaluation.py", "Full-featured evaluation with manual path specification"),
        ("quick_eval.py", "Simplified evaluation with auto-detected paths"),
        ("batch_eval.py", "Multi-model comparison and batch processing"),
        ("eval_utils.py", "Documentation, examples, and validation utilities"),
        ("README.md", "Complete documentation and usage guide")
    ]
    
    for script, description in scripts:
        print(f"📄 {script:25s} - {description}")
    
    print("\n🎯 QUICK START COMMANDS:")
    print("-" * 25)
    commands = [
        ("List available models", "python batch_eval.py --list_available"),
        ("Quick single evaluation", "python quick_eval.py --experiment gemini_exp1_during_gen_v1_100_mbpp"),
        ("Batch comparison", "python batch_eval.py --experiments gemini_exp1_during_gen_v1_100_mbpp qwen_exp1_during_gen_v1_100_mbpp"),
        ("Show examples", "python eval_utils.py --examples"),
        ("Validate data", "python eval_utils.py --validate --csv_file X --generated_dir Y --reference_file Z")
    ]
    
    for purpose, command in commands:
        print(f"🔧 {purpose:20s}: {command}")
    
    print("\n📊 INPUT DATA REQUIREMENTS:")
    print("-" * 30)
    requirements = [
        ("CSV File", "Contains task_id, pass_rate, z_scores (watermark detection data)"),
        ("Generated Code Dir", "Python files named by task_id (e.g., 57.py, 590.py)"),
        ("Reference File", "JSONL with ground truth implementations for comparison")
    ]
    
    for data_type, description in requirements:
        print(f"📁 {data_type:18s}: {description}")
    
    print("\n✨ EXAMPLE OUTPUT:")
    print("-" * 20)
    print("""🎯 Execution Metrics:
   Pass@1 (Avg Pass Rate): 0.6833 (68.33%)

🔍 Watermark Detection Metrics:
   AUROC:                   0.4712
   T@0%F               : 0.0100
   T@1%F               : 0.0100
   T@5%F               : 0.0600
   T@10%F              : 0.1000

📝 Code Quality Metrics:
   CodeBLEU Mean:           0.2808
   CodeBLEU Std:            0.1956""")
    
    print("\n🚨 PERFORMANCE TIPS:")
    print("-" * 20)
    tips = [
        "Use --codebleu_sample_size N for faster evaluation (e.g., N=20)",
        "Use --full_codebleu only when you need complete CodeBLEU analysis",
        "Batch evaluation generates comparison tables automatically",
        "JSON output available for programmatic analysis"
    ]
    
    for tip in tips:
        print(f"💡 {tip}")
    
    print("\n" + "=" * 60)
    print("📚 For complete documentation: cat README.md")
    print("🔍 For usage examples: python eval_utils.py --examples")
    print("🚀 For quick start: python batch_eval.py --list_available")
    print("=" * 60)

def test_toolkit_availability():
    """Test if all toolkit components are available."""
    
    print("\n🔧 TESTING TOOLKIT AVAILABILITY:")
    print("-" * 35)
    
    scripts = [
        "comprehensive_evaluation.py",
        "quick_eval.py", 
        "batch_eval.py",
        "eval_utils.py",
        "README.md"
    ]
    
    all_available = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - NOT FOUND")
            all_available = False
    
    print("\n🔍 TESTING DEPENDENCIES:")
    print("-" * 25)
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError:
        print("❌ pandas - pip install pandas")
        all_available = False
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError:
        print("❌ numpy - pip install numpy")
        all_available = False
        
    try:
        from sklearn.metrics import roc_curve, auc
        print("✅ scikit-learn")
    except ImportError:
        print("❌ scikit-learn - pip install scikit-learn")
        all_available = False
    
    # Check CodeBLEU availability
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics')
    if os.path.exists(metrics_dir):
        print("✅ CodeBLEU metrics directory")
        
        try:
            sys.path.insert(0, metrics_dir)
            from python_codebleu_helper import evaluate_python_code_bleu
            print("✅ CodeBLEU evaluation functions")
        except ImportError as e:
            print(f"⚠️  CodeBLEU functions - {e}")
    else:
        print("❌ CodeBLEU metrics directory - NOT FOUND")
        all_available = False
    
    if all_available:
        print("\n🎉 ALL COMPONENTS AVAILABLE - READY TO USE!")
    else:
        print("\n⚠️  SOME COMPONENTS MISSING - CHECK INSTALLATION")
    
    return all_available

def main():
    """Main function."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode
        print_toolkit_summary()
        test_toolkit_availability()
    else:
        # Default: show summary
        print_toolkit_summary()
        
        # Quick availability check
        print("\n🔧 Quick availability check:")
        essential_files = ["comprehensive_evaluation.py", "quick_eval.py", "batch_eval.py"]
        available = sum(1 for f in essential_files if os.path.exists(f))
        print(f"   {available}/{len(essential_files)} essential scripts found")
        
        if available == len(essential_files):
            print("   ✅ Toolkit ready to use!")
            print("\n🚀 Try: python batch_eval.py --list_available")
        else:
            print("   ⚠️  Some scripts missing - run with --test for details")

if __name__ == "__main__":
    main()