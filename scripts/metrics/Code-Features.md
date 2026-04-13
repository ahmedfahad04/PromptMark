I'll create a short TODO plan for this task, then propose robust watermarking ideas and next steps.

Created 5 todos

Progress: TODO plan created. Next: concise robust watermarking ideas.

**High-level approach**
- **Redundancy + ECC:** Encode the watermark across many independent, small signals and protect it with an error-correcting code so parts can be lost/changed but the signature remains recoverable.
- **Secret-key signing:** Compute a signature H = HMAC(secret, robust_features) and embed it indirectly; detector verifies H against extracted features.
- **Adversarially-trained detector:** Train a classifier on many paraphrases/transformations so detection is robust to LLM rewrites.

**Robust, measurable feature candidates**
- **AST Shape:** counts/distribution of AST node types and small subtree signatures (structure-level features survive renames/formatting).
- **Control-flow metrics:** cyclomatic complexity, number of branches/loops, nesting depth — semantic structure often preserved.
- **Token-type n-grams:** sequences of token types (e.g., NAME, OP, NUMBER) rather than exact names.
- **Identifier character-level stats:** frequency histogram of initial letters, character n-grams, length distribution — less sensitive than full names but still touchable.
- **Identifier usage graph:** variable-use adjacency / read/write pattern — semantic relationships resilient to renaming.
- **Literal/constant patterns:** counts and relative magnitudes of numeric/string literal forms (optionally tolerant to small perturbations).
- **Comment fingerprint:** controlled short comment tokens/phrases injected in multiple places; use synonyms mapping to bits so paraphrasing may change wording but retain signal probability.
- **Whitespace/formatting patterns:** run-length distributions (indentation, blank-line placement) — LLMs sometimes reformat, so use only as secondary signal.
- **Subtle semantic-preserving transforms:** e.g., reorder independent statements, expand/compress expressions, add no-op expressions or identity functions with unique names — but prefer transforms that survive LLM rewriting.
- **Embedding-based signature:** compute a robust embedding of the canonicalized code (AST + token-type info) and sign/quantize it; compare with tolerance (cosine threshold).

**Embedding watermark across channels (concrete encoding ideas)**
- **Multichannel bit mapping:** map bits of H to (a) identifier initial letters, (b) presence/absence of small no-op functions, (c) controlled comment tokens, (d) rounding pattern for floats. Each channel is noisy; redundancy + ECC recovers bits.
- **Semantic parity markers:** modify semantics in ways that preserve externally visible behavior but change internal traceable metrics (e.g., canonical order of commutative ops, deterministic random seeds). Only use if safe.
- **Token-order micro-patterns:** insert tiny, semantically-neutral token choices (e.g., prefer `if x: return` vs `if x == True:`) per a secret mapping.

**Detection strategy**
- **Canonicalize then extract:** parse to AST, normalize (rename all identifiers generically, but preserve structural/use patterns), then compute features.
- **Fuzzy matching & thresholds:** use HMAC + ECC to allow a noisy feature-vector -> signature recovery; or allow an embedding distance threshold for match.
- **Classifier fallback:** when signature recovery fails, run an ML classifier trained to spot likely-watermarked distributions (use adversarial paraphrases in training).

**Evaluation plan**
- Generate paraphrases/transformations with multiple LLMs and programmatic transformations (renaming, formatting, AST-preserving obfuscation).
- Measure AUROC and bit-recovery rate under each transformation type.
- Tune channel weights, ECC redundancy, and feature set to maximize recovery vs false positives.

**Trade-offs & cautions**
- **Stealth vs robustness:** more redundancy and many small inserts increase robustness but also visibility.
- **Functionality safety:** never change external behavior; unit tests / property-based tests should be part of pipeline.
- **Legal/ethical:** ensure watermarking complies with policy and consent where required.

Would you like me to (recommended) prototype a small pipeline that:
1) extracts a compact robust feature vector (AST + token-type n-grams + identifier-initial histogram),
2) computes HMAC + ECC,
3) shows how to encode bits into identifier initials + comment tokens,
and 4) evaluate resilience against GPT-based paraphrases?

---

**AST Shape**  
- **What:** counts/distribution of node types (If, For, Call), depth, common small subtree signatures.  
- **Example metric:** {If:4, For:2, Call:12, max_depth:6, subtree_hashes:[h1,h2...]}.  
- **How to extract:** parse with `ast` and compute histograms/subtree hashes.  
- **Prompting:** partially promptable — e.g., "Keep function/loop structure and don't rename variables; preserve control-flow." Example:  
  "Keep the same control-flow structure (if/for/while blocks) and do not rename variables."  
- **Reliability:** medium-high (AST shape resists renames/formatting), but LLM paraphrase can change structure.

**Control-flow Metrics (Cyclomatic Complexity)**  
- **What:** cyclomatic complexity, branch counts, nesting depths.  
- **Example metric:** `cyclomatic=5, branches=7, max_nesting=3`.  
- **How to extract:** compute from CFG or AST.  
- **Prompting:** limited — you can ask for similar structure ("Keep branching structure intact"), but LLMs may unintentionally refactor. Mark: partially promptable.

**Token-type n-grams**  
- **What:** sequences of token classes (NAME, OP, NUMBER) frequency (e.g., NAME-OP-NAME).  
- **Example metric:** top-10 token-type trigrams + counts.  
- **How to extract:** tokenize with `tokenize` module and map to token types.  
- **Prompting:** not directly forceable to exact sequences; you can ask "prefer idiomatic expressions X" but fragile. Mark: not reliably promptable.

**Identifier Character-level Stats**  
- **What:** histogram of first letters, length distribution, char n-gram frequencies.  
- **Example metric:** first_letter_freq = {'a':5,'m':3,...}, length_hist=[1,4,6,...].  
- **How to extract:** scan identifiers in AST.  
- **Prompting:** highly promptable — e.g., "Start variable names with these initials in this mapping (a/b/c → bit patterns)." Example:  
  "Map watermark bits to variable initials: use names starting with `m_`, `n_`, `p_` as indicated."  
- **Reliability:** good if the generator obeys prompt; vulnerable to later automatic renaming/paraphrase.

**Identifier Usage Graph (Data-flow/Use-patterns)**  
- **What:** adjacency of definition→use, read/write roles, scope-level degree.  
- **Example metric:** graph signature hashes or degree distributions.  
- **How to extract:** build use-def graph from AST.  
- **Prompting:** not direct; you can request "preserve variable usage style" but LLM may change. Mark: not reliably promptable.

**Literal/Constant Patterns**  
- **What:** counts/types of literals (numeric, string, unique formats), relative magnitudes/patterns.  
- **Example metric:** string_literal_hist={'id':'uuid-like', 'fmt':3}, numeric_precision_profile.  
- **How to extract:** inspect AST literal nodes; normalize numbers (rounding).  
- **Prompting:** highly promptable — e.g., "Preserve comments and these sentinel string tokens: /*SID:XYZ*/" or "Use these numeric precisions."  
- **Reliability:** good if you inject controlled literals; paraphrasing often preserves literals.

**Comment Fingerprint**  
- **What:** positions and tokens in short comments, tag tokens, synonyms mapping.  
- **Example metric:** tags present: ['TODO:zx1','WM:ab'] and comment-token histogram.  
- **How to extract:** parse comments (tokenize) and canonicalize synonyms.  
- **Prompting:** highly promptable — "Insert short comments with these tokens" (example: `# wm: 1101`).  
- **Reliability:** medium — paraphrasing can reword or remove comments; use redundancy and synonym sets to tolerate paraphrase.

**Whitespace / Formatting Patterns**  
- **What:** indent-run-lengths, blank-line positions, trailing spaces distribution.  
- **Example metric:** indent_hist=[0,2,4,8], blank_line_positions=[10,25,...].  
- **How to extract:** raw text analysis.  
- **Prompting:** promptable ("Keep formatting style X"), but LLM often reformat — mark: low reliability.

**Embedding-based Signature**  
- **What:** compute embedding of canonicalized AST/token-type representation and quantize/sign it.  
- **Example metric:** quantized vector or hash signed with HMAC.  
- **How to extract:** canonicalize then run an embedding model (or handcrafted featurizer).  
- **Prompting:** not applicable — it's a detection-side operation. Useable as robust fuzzy-match.

**No-op / Identity Function Patterns (Semantic-preserving markers)**  
- **What:** insert small no-op helpers or identity wrappers with secret-named functions; measure presence/positions.  
- **Example metric:** list of helper-function-names and call-sites signature.  
- **How to extract:** AST search for known patterns.  
- **Prompting:** fully promptable — "Add a helper `def _id123(x): return x` and call it in three places."  
- **Reliability:** good if generator obeys; paraphrases may remove or rename — mitigate with ECC and distributing across many locations.

Practical encoding recommendations (short)  
- Use multi-channel encoding: split watermark bits across several features above (ident-initials, comment tags, literals, no-op functions).  
- Protect with ECC and HMAC(secret, features). Store HMAC in comment tokens or external registry.  
- Canonicalize source on detection: normalize whitespace, rename identifiers to placeholders (but keep structural/use info), compute features, then try ECC/HMAC recovery.  
- Train an adversarial detector/classifier on model-paraphrases to raise AUROC where signature recovery fails.

Would you like a small prototype that (1) extracts `identifier-initial` histograms, `AST node` counts, and `comment tags`, (2) creates an HMAC+Reed-Solomon style ECC, and (3) shows encoding into variable initials + comments? I can implement and run it on the files in `test/` next.