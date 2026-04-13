#!/usr/bin/env python3
"""
Script to:
1. Sample generation files to understand format
2. Extract all generated code
3. Calculate CodeBLEU scores
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import re

# Add metrics directory to path
metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          'experiments', 'evals', 'metrics')
sys.path.insert(0, metrics_dir)

from python_codebleu_helper import evaluate_python_code_bleu


# ================================================================================
# PART 1: SAMPLE GENERATION FILES
# ================================================================================

def sample_generation_files(generation_files: List[str], sample_size: int = 2):
    """Sample the first N items from generation files to understand format."""
    
    print("\n" + "="*80)
    print(f"SAMPLING FIRST {sample_size} ITEMS FROM GENERATION FILES")
    print("="*80 + "\n")
    
    for gen_file in generation_files:
        if not os.path.exists(gen_file):
            print(f"⚠️  File not found: {gen_file}")
            continue
            
        print(f"\n📄 File: {gen_file}")
        print(f"   Size: {os.path.getsize(gen_file) / (1024*1024):.2f} MB")
        
        try:
            with open(gen_file, 'r') as f:
                data = json.load(f)
            
            print(f"   Total items: {len(data)}")
            print(f"   Type of data: {type(data)}")
            
            if isinstance(data, list) and len(data) > 0:
                print(f"\n   --------- SAMPLE (First {sample_size} items) ---------")
                for idx, item in enumerate(data[:sample_size]):
                    print(f"\n   [Item {idx}]")
                    print(f"   Type: {type(item)}")
                    
                    if isinstance(item, str):
                        # Show first 300 chars of the string
                        preview = item[:300]
                        if len(item) > 300:
                            preview += "...[TRUNCATED]"
                        print(f"   Content preview: {preview}")
                        print(f"   Total length: {len(item)} characters")
                    else:
                        print(f"   Content: {item}")
                        
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON decode error: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


# ================================================================================
# PART 2: EXTRACT GENERATED CODE
# ================================================================================

def extract_code_blocks(text: str) -> List[str]:
    """
    Extract Python code blocks from generated text.
    The text is already code - just clean it up.
    Each text block may contain multiple docstring-separated problems.
    Split by triple quotes to get individual solutions.
    """
    code_blocks = []
    
    # Split by triple quotes to separate problem descriptions from code
    parts = re.split(r'"""', text)
    
    # Process parts - odd indices are usually prose, even are code
    for i, part in enumerate(parts):
        part = part.strip()
        if not part or len(part) < 5:
            continue
            
        # Check if this part looks like code (has function def, imports, etc.)
        if any(keyword in part for keyword in ['def ', 'import ', 'class ', 'return', '=']):
            code_blocks.append(part)
    
    # If no split was successful, treat entire text as code if it looks like code
    if not code_blocks and any(keyword in text for keyword in ['def ', 'import ', 'return']):
        code_blocks.append(text.strip())
    
    # Clean extracted blocks
    cleaned_blocks = []
    for block in code_blocks:
        block = block.strip()
        if block and len(block) > 20:  # Filter out very small blocks
            cleaned_blocks.append(block)
    
    return cleaned_blocks


def extract_all_generated_code(generation_files: List[str]) -> Dict[str, List[str]]:
    """Extract all generated code from files."""
    
    print("\n" + "="*80)
    print("EXTRACTING ALL GENERATED CODE")
    print("="*80 + "\n")
    
    all_code = {}
    total_extracted = 0
    
    for gen_file in generation_files:
        if not os.path.exists(gen_file):
            print(f"⚠️  Skipping {gen_file} (not found)")
            continue
        
        file_key = Path(gen_file).stem
        all_code[file_key] = []
        
        try:
            with open(gen_file, 'r') as f:
                data = json.load(f)
            
            print(f"📥 Processing: {file_key}")
            
            for idx, item in enumerate(data):
                if isinstance(item, list):
                    # Item is a list of strings (multiple code variations)
                    for variant_str in item:
                        if isinstance(variant_str, str):
                            code_blocks = extract_code_blocks(variant_str)
                            all_code[file_key].extend(code_blocks)
                            total_extracted += len(code_blocks)
                elif isinstance(item, str):
                    # Item is a single string
                    code_blocks = extract_code_blocks(item)
                    all_code[file_key].extend(code_blocks)
                    total_extracted += len(code_blocks)
            
            print(f"   ✅ Extracted {len(all_code[file_key])} code blocks")
            
        except Exception as e:
            print(f"   ❌ Error processing {file_key}: {e}")
    
    print(f"\n✅ Total code blocks extracted: {total_extracted}")
    return all_code


# ================================================================================
# PART 3: CALCULATE CodeBLEU
# ================================================================================

def calculate_codebleu_scores(reference_file: str, generated_code: List[str], 
                              sample_size: int = 50) -> Dict[str, Any]:
    """
    Calculate CodeBLEU scores for generated code.
    
    Args:
        reference_file: Path to reference code file (JSONL or JSON)
        generated_code: List of generated code strings
        sample_size: Number of samples to evaluate (for speed)
    
    Returns:
        Dictionary with CodeBLEU metrics
    """
    
    print("\n" + "="*80)
    print("CALCULATING CodeBLEU SCORES")
    print("="*80 + "\n")
    
    if not os.path.exists(reference_file):
        print(f"❌ Reference file not found: {reference_file}")
        return {}
    
    # Load reference code
    references = []
    try:
        if reference_file.endswith('.jsonl'):
            with open(reference_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'code' in data:
                        references.append(data['code'])
                    elif 'entry_point' in data and 'canonical_solution' in data:
                        references.append(data['canonical_solution'])
        else:
            with open(reference_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'code' in item:
                            references.append(item['code'])
                        elif isinstance(item, dict) and 'canonical_solution' in item:
                            references.append(item['canonical_solution'])
    except Exception as e:
        print(f"❌ Error loading references: {e}")
        return {}
    
    print(f"✅ Loaded {len(references)} reference implementations")
    
    # Limit sample size for speed
    gen_sample = generated_code[:sample_size]
    ref_sample = references[:len(gen_sample)]
    
    print(f"📊 Evaluating {len(gen_sample)} generated samples against {len(ref_sample)} references")
    
    codebleu_scores = []
    errors = 0
    
    for idx, (gen_code, ref_code) in enumerate(zip(gen_sample, ref_sample)):
        try:
            result = evaluate_python_code_bleu(ref_code, gen_code)
            
            # Extract CodeBLEU score from result dict
            if isinstance(result, dict):
                codebleu = result.get('codebleu', 0)
            else:
                codebleu = result
            
            codebleu_scores.append(codebleu)
            
            if (idx + 1) % 10 == 0:
                print(f"   ✅ Processed {idx + 1}/{len(gen_sample)} samples")
                
        except Exception as e:
            errors += 1
            if errors <= 5:  # Print first 5 errors
                print(f"   ⚠️  Error evaluating sample {idx}: {str(e)[:100]}")
    
    # Calculate statistics
    if codebleu_scores:
        import numpy as np
        stats = {
            'total_evaluated': len(codebleu_scores),
            'errors': errors,
            'mean_codebleu': float(np.mean(codebleu_scores)),
            'std_codebleu': float(np.std(codebleu_scores)),
            'min_codebleu': float(np.min(codebleu_scores)),
            'max_codebleu': float(np.max(codebleu_scores)),
            'median_codebleu': float(np.median(codebleu_scores)),
        }
    else:
        stats = {'error': 'No valid scores calculated'}
    
    return stats


# ================================================================================
# MAIN
# ================================================================================

def main():
    """Main execution."""
    
    # Define paths
    base_dir = "/home/fahad/Documents/PROJECTS/promptmark"
    output_dir = os.path.join(base_dir, "output", "baseline_results")
    
    # Find ALL generation JSON files recursively
    generation_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file == "generations.json":
                generation_files.append(os.path.join(root, file))
    
    # Reference file for CodeBLEU
    reference_file = os.path.join(base_dir, "datasets", "sanitized-mbpp-sample-100.jsonl")
    
    print("🚀 Starting generation file analysis and CodeBLEU calculation...")
    print(f"Base directory: {base_dir}")
    print(f"Found {len(generation_files)} generation files")
    
    # STEP 1: Sample files
    sample_generation_files(generation_files, sample_size=2)
    
    # STEP 2: Extract all code
    all_code_dict = extract_all_generated_code(generation_files)
    
    # STEP 3: Calculate CodeBLEU with more samples
    combined_code = []
    for code_list in all_code_dict.values():
        combined_code.extend(code_list)
    
    print(f"\n✅ Total code blocks available for evaluation: {len(combined_code)}")
    
    # Evaluate with progressively larger samples
    sample_sizes = [50, 100, 200, 500]
    
    for sample_size in sample_sizes:
        if len(combined_code) < sample_size:
            continue
            
        print(f"\n{'='*80}")
        print(f"CODEBLEU EVALUATION WITH {sample_size} SAMPLES")
        print(f"{'='*80}")
        codebleu_stats = calculate_codebleu_scores(reference_file, combined_code, sample_size=sample_size)
        
        for key, value in codebleu_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
