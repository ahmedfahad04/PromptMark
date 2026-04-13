# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU

# -*- coding:utf-8 -*-
import os
import argparse
from typing import Dict, List, Optional, Tuple

try:
    from . import bleu, weighted_ngram_match, syntax_match, dataflow_match
except ImportError:
    import bleu
    import weighted_ngram_match
    import syntax_match
    import dataflow_match

# Language configuration mapping
LANGUAGE_CONFIG = {
    'python': {
        'keyword_file': 'python.txt',
        'tree_sitter_name': 'python',
        'comment_removal_lang': 'python',
        'wrapper_needed': False,
        'wrapper_template': None
    },
    'java': {
        'keyword_file': 'java.txt',
        'tree_sitter_name': 'java',
        'comment_removal_lang': 'java',
        'wrapper_needed': True,
        'wrapper_template': 'public class Wrapper {{\n{code}\n}}'
    },
    'cpp': {
        'keyword_file': 'cpp.txt',
        'tree_sitter_name': 'cpp',
        'comment_removal_lang': 'cpp',
        'wrapper_needed': False,
        'wrapper_template': None
    },
    'c': {
        'keyword_file': 'c.txt',
        'tree_sitter_name': 'c',
        'comment_removal_lang': 'c',
        'wrapper_needed': False,
        'wrapper_template': None
    },
    'javascript': {
        'keyword_file': 'javascript.txt',
        'tree_sitter_name': 'javascript',
        'comment_removal_lang': 'javascript',
        'wrapper_needed': False,
        'wrapper_template': None
    },
    'c_sharp': {
        'keyword_file': 'c.txt',  # Fallback to C keywords
        'tree_sitter_name': 'c_sharp',
        'comment_removal_lang': 'c_sharp',
        'wrapper_needed': False,
        'wrapper_template': None
    },
    'php': {
        'keyword_file': 'c.txt',  # Fallback to C keywords
        'tree_sitter_name': 'php',
        'comment_removal_lang': 'php',
        'wrapper_needed': False,
        'wrapper_template': None
    },
    'go': {
        'keyword_file': 'c.txt',  # Fallback to C keywords
        'tree_sitter_name': 'go',
        'comment_removal_lang': 'go',
        'wrapper_needed': False,
        'wrapper_template': None
    },
    'ruby': {
        'keyword_file': 'c.txt',  # Fallback to C keywords
        'tree_sitter_name': 'ruby',
        'comment_removal_lang': 'ruby',
        'wrapper_needed': False,
        'wrapper_template': None
    }
}

# Supported languages
SUPPORTED_LANGUAGES = list(LANGUAGE_CONFIG.keys())

def get_language_config(lang: str) -> Dict:
    """Get language configuration for the specified language."""
    if lang not in LANGUAGE_CONFIG:
        raise ValueError(f"Unsupported language: {lang}. Supported languages: {SUPPORTED_LANGUAGES}")
    return LANGUAGE_CONFIG[lang]

def apply_code_wrapper(code: str, lang: str) -> str:
    """Apply language-specific code wrapper if needed."""
    config = get_language_config(lang)
    if config['wrapper_needed'] and config['wrapper_template']:
        return config['wrapper_template'].format(code=code)
    return code

def load_keywords(lang: str) -> List[str]:
    """Load keywords for the specified language."""
    config = get_language_config(lang)
    root_dir = os.path.dirname(__file__)
    keyword_file_path = os.path.join(root_dir, "keywords", config['keyword_file'])
    
    if not os.path.exists(keyword_file_path):
        print(f"Warning: Keyword file not found for {lang} at {keyword_file_path}. Using empty keyword list.")
        return []
    
    try:
        with open(keyword_file_path, "r", encoding="utf-8") as f:
            keywords = [x.strip() for x in f.readlines() if x.strip()]
        return keywords
    except Exception as e:
        print(f"Warning: Failed to load keywords for {lang}: {e}. Using empty keyword list.")
        return []


def evaluate_per_example(
    reference: str, hypothesis: str, lang: str, params="0.25,0.25,0.25,0.25"
):
    """Evaluate CodeBLEU score for a single example.
    
    Args:
        reference: Reference (ground truth) code
        hypothesis: Generated code to evaluate
        lang: Programming language (python, java, cpp, c, etc.)
        params: Comma-separated weights for alpha,beta,gamma,theta components
        
    Returns:
        Dictionary with detailed scores including EM, BLEU, weighted BLEU, syntax, dataflow, and CodeBLEU
    """
    # Validate language
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang}. Supported: {SUPPORTED_LANGUAGES}")
    
    alpha, beta, gamma, theta = [float(x) for x in params.split(",")]

    # Apply language-specific code wrappers
    hypothesis = apply_code_wrapper(hypothesis, lang)
    reference = apply_code_wrapper(reference, lang)

    hypothesis = [hypothesis]
    pre_references = [[reference]]
    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])
    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references) * len(hypothesis)
    
    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]
    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)
    
    # calculate weighted ngram match
    keywords = load_keywords(lang)

    def make_weights(reference_tokens, key_word_list):
        return {
            token: 1 if token in key_word_list else 0.2 for token in reference_tokens
        }

    tokenized_refs_with_weights = [
        [
            [reference_tokens, make_weights(reference_tokens, keywords)]
            for reference_tokens in reference
        ]
        for reference in tokenized_refs
    ]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
        tokenized_refs_with_weights, tokenized_hyps
    )
    # calculate syntax match
    config = get_language_config(lang)
    tree_sitter_lang = config['tree_sitter_name']
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, tree_sitter_lang)
    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(
        references, hypothesis, tree_sitter_lang
    )
    # dataflow_match_score = dataflow_match.my_dataflow_match(references, hypothesis, lang)
    # print(
    #     'ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'
    #     .format(ngram_match_score, weighted_ngram_match_score, syntax_match_score,
    #             dataflow_match_score))
    codebleu = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * syntax_match_score
        + theta * dataflow_match_score
    )
    return {
        "em": 1.0 if reference.strip() == hypothesis[0].strip() else 0.0,
        "bleu": ngram_match_score,
        "wbleu": weighted_ngram_match_score,
        "syntax": syntax_match_score,
        "dataflow": dataflow_match_score,
        "codebleu": codebleu,
    }


def get_codebleu(refs, hyp, lang, params="0.25,0.25,0.25,0.25"):
    """Calculate CodeBLEU score for files containing reference and hypothesis code.
    
    Args:
        refs: List of reference file paths or single file path
        hyp: Hypothesis file path
        lang: Programming language (python, java, cpp, c, etc.)
        params: Comma-separated weights for alpha,beta,gamma,theta components
        
    Returns:
        CodeBLEU score (float)
    """
    # Validate language
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang}. Supported: {SUPPORTED_LANGUAGES}")
        
    if not isinstance(refs, list):
        refs = [refs]
    alpha, beta, gamma, theta = [float(x) for x in params.split(",")]

    # preprocess inputs
    try:
        # Read entire files as single code samples (not line-by-line)
        pre_references = [
            [open(file, "r", encoding="utf-8").read().strip()]
            for file in refs
        ]
        hypothesis = [open(hyp, "r", encoding="utf-8").read().strip()]

        print(f"Reference files: {len(pre_references)}")
        print(f"Hypothesis samples: {len(hypothesis)}")
        print(f"Reference samples per file: {[len(refs) for refs in pre_references]}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not read input files: {e}")
    except Exception as e:
        raise Exception(f"Error reading input files: {e}")

    # For single file evaluation, we expect 1 hypothesis sample
    # For multiple reference files, we still expect 1 hypothesis sample (comparing against multiple references)
    if len(pre_references) > 1:
        # Multiple reference files - combine them as multiple references for one hypothesis
        combined_references = []
        for ref_list in pre_references:
            combined_references.extend(ref_list)
        pre_references = [combined_references]
        print(f"Combined references for comparison: {len(combined_references)} total reference samples")

    # Validate that we have exactly one hypothesis sample and one reference set
    assert len(pre_references) == 1, f"Expected exactly 1 reference set, got {len(pre_references)}"
    assert len(hypothesis) == 1, f"Expected exactly 1 hypothesis sample, got {len(hypothesis)}"
    assert len(pre_references[0]) >= 1, f"Expected at least 1 reference sample, got {len(pre_references[0])}"

    # Apply language-specific wrappers to all code samples
    if get_language_config(lang)['wrapper_needed']:
        for ref_set in pre_references:
            for i, ref in enumerate(ref_set):
                ref_set[i] = apply_code_wrapper(ref, lang)
        for i, hyp in enumerate(hypothesis):
            hypothesis[i] = apply_code_wrapper(hyp, lang)

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references) * len(hypothesis)

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    keywords = load_keywords(lang)

    def make_weights(reference_tokens, key_word_list):
        return {
            token: 1 if token in key_word_list else 0.2 for token in reference_tokens
        }

    tokenized_refs_with_weights = [
        [
            [reference_tokens, make_weights(reference_tokens, keywords)]
            for reference_tokens in reference
        ]
        for reference in tokenized_refs
    ]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
        tokenized_refs_with_weights, tokenized_hyps
    )

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(
        references, hypothesis, lang
    )
    # dataflow_match_score = dataflow_match.my_dataflow_match(references, hypothesis, lang)

    print(
        "ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}".format(
            ngram_match_score,
            weighted_ngram_match_score,
            syntax_match_score,
            dataflow_match_score,
        )
    )

    codebleu = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * syntax_match_score
        + theta * dataflow_match_score
    )

    return codebleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refs", type=str, nargs="+", required=True, help="reference files"
    )
    parser.add_argument("--hyp", type=str, required=True, help="hypothesis file")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=SUPPORTED_LANGUAGES,
        help=f"programming language. Supported: {', '.join(SUPPORTED_LANGUAGES)}",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="0.25,0.25,0.25,0.25",
        help="alpha, beta and gamma",
    )

    args = parser.parse_args()
    
    try:
        print(f"Calculating CodeBLEU for language: {args.lang}")
        print(f"Reference files: {args.refs}")
        print(f"Hypothesis file: {args.hyp}")
        print(f"Weights (alpha,beta,gamma,theta): {args.params}")
        print("-" * 50)
        
        code_bleu_score = get_codebleu(args.refs, args.hyp, args.lang, args.params)
        print(f"CodeBLEU score: {code_bleu_score:.4f}")
        
    except Exception as e:
        print(f"Error calculating CodeBLEU: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
