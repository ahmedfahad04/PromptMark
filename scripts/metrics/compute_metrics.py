#!/usr/bin/env python3
"""Compute code metrics across a codebase (Python files).

Outputs a JSON file with per-file metrics and aggregated metrics.
"""
import ast
import argparse
import io
import json
import os
import re
import tokenize
from collections import Counter, defaultdict


def node_type_counts(tree):
    c = Counter()
    for node in ast.walk(tree):
        c[type(node).__name__] += 1
    return c


def max_ast_depth(node):
    def _depth(n):
        child_depths = [_depth(c) for c in ast.iter_child_nodes(n)]
        return 1 + max(child_depths) if child_depths else 1
    return _depth(node)


def cyclomatic_complexity(tree):
    # simple approximation
    decision_nodes = (
        ast.If,
        ast.For,
        ast.While,
        ast.AsyncFor,
        ast.With,
        ast.Try,
        ast.IfExp,
    )
    cc = 1
    for node in ast.walk(tree):
        if isinstance(node, decision_nodes):
            cc += 1
        elif isinstance(node, ast.BoolOp):
            # boolean ops increase branching
            cc += max(0, len(node.values) - 1)
    return cc


def token_type_ngrams(code, n=3, topk=20):
    types = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            types.append(tokenize.tok_name[tok.type])
    except Exception:
        pass
    grams = Counter()
    for i in range(len(types) - n + 1):
        grams[' '.join(types[i : i + n])] += 1
    return grams.most_common(topk)


def identifier_stats(tree):
    ids = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            ids.append(node.id)
        elif isinstance(node, ast.FunctionDef):
            ids.append(node.name)
        elif isinstance(node, ast.ClassDef):
            ids.append(node.name)
        elif isinstance(node, ast.arg):
            ids.append(node.arg)
    first_letter = Counter()
    length_hist = Counter()
    char_grams = Counter()
    for ident in ids:
        if not ident:
            continue
        first_letter[ident[0]] += 1
        length_hist[len(ident)] += 1
        for i in range(len(ident) - 2):
            char_grams[ident[i : i + 3]] += 1
    return {
        'total_identifiers': len(ids),
        'first_letter': first_letter.most_common(30),
        'length_hist': sorted(length_hist.items()),
        'char_trigrams': char_grams.most_common(30),
    }


def literal_stats(tree):
    str_count = 0
    num_count = 0
    str_lengths = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, str):
                str_count += 1
                str_lengths.append(len(val))
            elif isinstance(val, (int, float, complex)):
                num_count += 1
    return {
        'string_literals': str_count,
        'numeric_literals': num_count,
        'avg_string_length': sum(str_lengths) / len(str_lengths) if str_lengths else 0,
    }


def comment_stats(code):
    comments = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tokenize.tok_name[tok.type] == 'COMMENT':
                comments.append(tok.string.lstrip('#').strip())
    except Exception:
        pass
    tags = []
    tag_re = re.compile(r'\b(wm|wm:|watermark|sid|id)[:=]?\s*([A-Za-z0-9_-]+)', re.I)
    for c in comments:
        for m in tag_re.finditer(c):
            tags.append(m.group(2))
    # compute average comment length (characters)
    comment_lengths = [len(c) for c in comments]
    avg_comment_length = sum(comment_lengths) / len(comment_lengths) if comment_lengths else 0
    return {
        'comment_count': len(comments),
        'tags': tags,
        'avg_comment_length': avg_comment_length,
        'sample_comments': comments[:5],
    }


def whitespace_stats(code):
    lines = code.splitlines()
    blank = 0
    indents = Counter()
    trailing_spaces = 0
    for L in lines:
        if not L.strip():
            blank += 1
        m = re.match(r'^(\s+)', L)
        if m:
            indents[len(m.group(1))] += 1
        if L.endswith(' '):
            trailing_spaces += 1
    return {'blank_lines': blank, 'indent_hist': indents.most_common(20), 'trailing_spaces': trailing_spaces}


def analyze_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception:
        return {'path': path, 'error': 'could not read file'}
    try:
        tree = ast.parse(code)
    except Exception:
        tree = None
    res = {'path': path}
    if tree is not None:
        res['ast_nodes'] = dict(node_type_counts(tree))
        res['max_ast_depth'] = max_ast_depth(tree)
        res['cyclomatic'] = cyclomatic_complexity(tree)
        res['identifiers'] = identifier_stats(tree)
        res['literals'] = literal_stats(tree)
    else:
        res['parse_error'] = True
    res['token_trigrams'] = token_type_ngrams(code, n=3, topk=30)
    res['comments'] = comment_stats(code)
    res['whitespace'] = whitespace_stats(code)
    return res


def walk_and_analyze(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.py'):
                files.append(os.path.join(dirpath, fn))
    results = []
    agg = defaultdict(int)
    for p in sorted(files):
        r = analyze_file(p)
        results.append(r)
        if 'ast_nodes' in r:
            for k, v in r['ast_nodes'].items():
                agg[k] += v
    aggregated = {'total_files': len(results), 'ast_node_totals': dict(agg)}
    return {'files': results, 'aggregated': aggregated}


def _flatten_row(fmetrics, group, root):
    # produce a flat CSV row from file metrics
    path = fmetrics.get('path', '')
    rel = os.path.relpath(path, root) if root and path.startswith(root) else os.path.basename(path)
    ast_nodes = fmetrics.get('ast_nodes', {})
    total_ast_nodes = sum(ast_nodes.values()) if ast_nodes else 0
    ids = fmetrics.get('identifiers', {})
    first_letter = ids.get('first_letter', [])
    most_common_initial = first_letter[0][0] if first_letter else ''
    most_common_initial_count = first_letter[0][1] if first_letter else 0
    trigrams = fmetrics.get('token_trigrams', [])
    top_trigram, top_trigram_count = ('', 0)
    if trigrams:
        top_trigram, top_trigram_count = trigrams[0]
    comments = fmetrics.get('comments', {})
    whitespace = fmetrics.get('whitespace', {})
    literals = fmetrics.get('literals', {})
    return {
        'group': group,
        'file': path,
        'relpath': rel,
        'total_ast_nodes': total_ast_nodes,
        'cyclomatic': fmetrics.get('cyclomatic', 0),
        'max_ast_depth': fmetrics.get('max_ast_depth', 0),
        'total_identifiers': ids.get('total_identifiers', 0),
        'most_common_initial': most_common_initial,
        'most_common_initial_count': most_common_initial_count,
        'string_literals': literals.get('string_literals', 0),
        'numeric_literals': literals.get('numeric_literals', 0),
        'avg_string_length': literals.get('avg_string_length', 0),
        'avg_comment_length': comments.get('avg_comment_length', 0),
        'comment_count': comments.get('comment_count', 0),
        'blank_lines': whitespace.get('blank_lines', 0),
        'trailing_spaces': whitespace.get('trailing_spaces', 0),
        'top_token_trigram': top_trigram,
        'top_token_trigram_count': top_trigram_count,
    }


def _analyze_path_group(root):
    if not root:
        return []
    if os.path.isfile(root) and root.endswith('.py'):
        return [analyze_file(root)]
    return walk_and_analyze(root)['files']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', help='Path to raw code folder or file')
    ap.add_argument('--watermarked', help='Path to watermarked code folder or file')
    ap.add_argument('--refactored', help='Path to refactored/paraphrased code folder or file')
    ap.add_argument('--output', '-o', required=False, default='metrics_output.json', help='Output JSON file (also write CSV alongside)')
    args = ap.parse_args()

    groups = [('raw', args.raw), ('watermarked', args.watermarked), ('refactored', args.refactored)]
    out = {'groups': {}, 'summary_rows': []}
    for name, path in groups:
        if not path:
            continue
        files_metrics = _analyze_path_group(path)
        out['groups'][name] = {'root': path, 'files': files_metrics}
        for fm in files_metrics:
            out['summary_rows'].append(_flatten_row(fm, name, path))

    # write JSON and CSV into a dedicated subfolder inside `output/`
    base_name = os.path.splitext(os.path.basename(args.output))[0]
    dest_dir = os.path.join('output', f"{base_name}_analysis")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    json_path = os.path.join(dest_dir, f"{base_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('Wrote', json_path)

    # write CSV summary for easy analysis
    try:
        import csv

        csv_path = os.path.join(dest_dir, f"{base_name}_summary.csv")
        if out['summary_rows']:
            keys = list(out['summary_rows'][0].keys())
            with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                writer = csv.DictWriter(cf, fieldnames=keys)
                writer.writeheader()
                for r in out['summary_rows']:
                    writer.writerow(r)
            print('Wrote', csv_path)
    except Exception:
        pass


if __name__ == '__main__':
    main()
