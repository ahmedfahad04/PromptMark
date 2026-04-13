# Metrics Output — Brief Guide

This document explains the fields produced by `compute_metrics.py` (JSON per-file groups and a CSV `*_summary.csv`). Use the CSV for quick analysis and the JSON for full detail.

Per-file JSON fields (group → files list):
- `path`: full file path.
- `ast_nodes`: map of AST node type → count (useful to compare structural changes).
- `max_ast_depth`: maximum AST tree depth (higher → deeper nesting).
- `cyclomatic`: approximate cyclomatic complexity (higher → more branches/logic).
- `identifiers`: object with:
  - `total_identifiers`: count of identifier occurrences.
  - `first_letter`: list of (letter, count) for identifier initial letters.
  - `length_hist`: histogram of identifier lengths.
  - `char_trigrams`: frequent character trigrams from identifiers.
- `literals`: counts of string and numeric literals and `avg_string_length`.
- `token_trigrams`: most common token-type trigrams (token class sequences).
- `comments`: object with `comment_count`, `avg_comment_length` (chars), `tags` (detected watermark-like tags), and sample comments.
- `whitespace`: `blank_lines`, `indent_hist` (indent-run histogram), `trailing_spaces`.

CSV `summary` columns (one row per file):
- `group`: which folder (raw/watermarked/refactored).
- `file`, `relpath`: file identifiers.
- `total_ast_nodes`, `cyclomatic`, `max_ast_depth`.
- `total_identifiers`, `most_common_initial`, `most_common_initial_count` (quick id-name signal).
- `string_literals`, `numeric_literals`, `avg_string_length`.
- `avg_comment_length`, `comment_count`.
- `blank_lines`, `trailing_spaces`.
- `top_token_trigram`, `top_token_trigram_count`.

How to use these metrics for watermark analysis (short):
- Identifier channels: compare `most_common_initial` and `first_letter` distributions across groups to detect intentional initial-letter encoding.
- Structural channels: compare `ast_nodes`, `total_ast_nodes`, `cyclomatic`, and `max_ast_depth` to find preserved or altered control-flow structure after paraphrase.
- Literal/comment channels: `string_literals`, `avg_string_length`, and `avg_comment_length` are good carriers (comments are easy to prompt into generated code; paraphrasers may delete or rewrite them — use redundancy).
- Token-type and whitespace channels: token trigrams and `indent_hist` provide stylistic fingerprints; use as supplementary signals.

Practical tips:
- Canonicalize before detection: parse to AST and normalize identifiers when comparing structural features.
- Use HMAC/ECC over multiple channels to tolerate noise (embedding + fuzzy thresholds help when exact recovery fails).
- Train a classifier on paraphrased examples to improve detection when signature recovery is noisy.

Location of outputs created by the script:
- JSON: `output/<base>_analysis/<base>.json`
- CSV summary: `output/<base>_analysis/<base>_summary.csv`

If you want, I can add: docstring detection, comment word-count, or a small notebook that visualizes differences across groups.
