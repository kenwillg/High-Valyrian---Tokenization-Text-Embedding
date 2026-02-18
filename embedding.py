"""
Word Embedding for High Valyrian Corpus
========================================
Builds word embeddings from scratch using:
  1. Co-occurrence matrix (context window)
  2. PPMI (Positive Pointwise Mutual Information) weighting
  3. Truncated SVD for dimensionality reduction

No pre-built NLP libraries — only NumPy for linear algebra.

Outputs:
  - embedding.npy       : (vocab_size, embed_dim) matrix
  - embedding_meta.json : metadata + token-to-index mapping
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import Counter

# --- Config ---
EMBED_DIM = 50          # embedding dimensions
WINDOW_SIZE = 4         # context window (each side)
MIN_FREQ = 1            # minimum token frequency to include

BASE_DIR = Path(__file__).parent
CLEANED_CORPUS = BASE_DIR / "cleaned_corpus.txt"
VOCAB_FILE = BASE_DIR / "vocab.json"
EMBED_FILE = BASE_DIR / "embedding.npy"
META_FILE = BASE_DIR / "embedding_meta.json"


def tokenize(text: str) -> list[str]:
    """Same tokenizer as tokenizer.py for consistency."""
    text = text.lower()
    text = re.sub(r'([.!?,;:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()


def build_cooccurrence(corpus_tokens: list[str], vocab: dict, window: int) -> np.ndarray:
    """
    Build a co-occurrence matrix from the corpus.

    For each token, count how many times each other token appears
    within `window` positions (left and right).
    Distance-weighted: closer words get higher counts (1/distance).
    """
    V = len(vocab)
    cooc = np.zeros((V, V), dtype=np.float64)

    for i, token in enumerate(corpus_tokens):
        if token not in vocab:
            continue
        idx_i = vocab[token]

        # Look at context window
        start = max(0, i - window)
        end = min(len(corpus_tokens), i + window + 1)

        for j in range(start, end):
            if j == i:
                continue
            ctx_token = corpus_tokens[j]
            if ctx_token not in vocab:
                continue
            idx_j = vocab[ctx_token]
            distance = abs(i - j)
            cooc[idx_i][idx_j] += 1.0 / distance  # distance weighting

    return cooc


def apply_ppmi(cooc: np.ndarray) -> np.ndarray:
    """
    Apply Positive Pointwise Mutual Information (PPMI).

    PMI(w, c) = log2(P(w,c) / (P(w) * P(c)))
    PPMI = max(0, PMI)

    This reduces the bias toward very frequent words.
    """
    total = cooc.sum()
    if total == 0:
        return cooc

    # Row and column marginals
    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)

    # Avoid division by zero
    row_sum[row_sum == 0] = 1
    col_sum[col_sum == 0] = 1

    # PMI = log2(P(w,c) / (P(w) * P(c)))
    #     = log2((cooc / total) / ((row_sum / total) * (col_sum / total)))
    #     = log2(cooc * total / (row_sum * col_sum))
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log2(cooc * total / (row_sum * col_sum))

    # Replace -inf and NaN with 0
    pmi[~np.isfinite(pmi)] = 0

    # PPMI: keep only positive values
    ppmi = np.maximum(pmi, 0)

    return ppmi


def truncated_svd(matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Reduce matrix to k dimensions using SVD.

    M ≈ U_k * S_k * V_k^T
    Embedding = U_k * sqrt(S_k) for better results.
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Take top-k components
    k = min(k, len(S))
    U_k = U[:, :k]
    S_k = S[:k]

    # Weight by sqrt of singular values
    embeddings = U_k * np.sqrt(S_k)

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_similar(token: str, embeddings: np.ndarray, vocab: dict, top_k: int = 10) -> list:
    """Find the top-k most similar tokens to the given token."""
    if token not in vocab:
        return []
    idx = vocab[token]
    vec = embeddings[idx]

    similarities = []
    for other_token, other_idx in vocab.items():
        if other_token == token:
            continue
        sim = cosine_similarity(vec, embeddings[other_idx])
        similarities.append((other_token, other_idx, sim))

    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]


def main():
    # --- Load vocab ---
    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab_data = json.load(f)
    token_to_id = vocab_data["token_to_id"]

    # --- Read & tokenize corpus ---
    text = CLEANED_CORPUS.read_text(encoding="utf-8")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    corpus_tokens = []
    for line in lines:
        tokens = tokenize(line)
        corpus_tokens.extend(tokens)

    print("=" * 60)
    print("WORD EMBEDDING PIPELINE")
    print("=" * 60)
    print(f"Corpus tokens:    {len(corpus_tokens)}")
    print(f"Vocab size:       {len(token_to_id)}")
    print(f"Embedding dim:    {EMBED_DIM}")
    print(f"Context window:   {WINDOW_SIZE}")

    # --- Step 1: Co-occurrence matrix ---
    print("\n[1/3] Building co-occurrence matrix...")
    cooc = build_cooccurrence(corpus_tokens, token_to_id, WINDOW_SIZE)
    non_zero = np.count_nonzero(cooc)
    sparsity = 1.0 - non_zero / (cooc.shape[0] * cooc.shape[1])
    print(f"  Matrix shape:   {cooc.shape}")
    print(f"  Non-zero cells: {non_zero}")
    print(f"  Sparsity:       {sparsity:.1%}")

    # --- Step 2: PPMI weighting ---
    print("\n[2/3] Applying PPMI weighting...")
    ppmi = apply_ppmi(cooc)
    print(f"  PPMI max:       {ppmi.max():.4f}")
    print(f"  PPMI mean:      {ppmi.mean():.4f}")

    # --- Step 3: SVD dimensionality reduction ---
    actual_dim = min(EMBED_DIM, len(token_to_id) - 1)
    print(f"\n[3/3] SVD → {actual_dim} dimensions...")
    embeddings = truncated_svd(ppmi, actual_dim)
    print(f"  Embeddings shape: {embeddings.shape}")

    # --- Save outputs ---
    np.save(EMBED_FILE, embeddings)
    print(f"\nSaved: {EMBED_FILE.name} ({embeddings.nbytes / 1024:.1f} KB)")

    # Save metadata
    id_to_token = {v: k for k, v in token_to_id.items()}
    meta = {
        "embed_dim": actual_dim,
        "vocab_size": len(token_to_id),
        "window_size": WINDOW_SIZE,
        "corpus_tokens": len(corpus_tokens),
        "method": "PPMI + SVD",
        "id_to_token": id_to_token,
        "token_to_id": token_to_id,
    }
    META_FILE.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {META_FILE.name}")

    # --- Demo: show similar words ---
    print("\n" + "=" * 60)
    print("SIMILARITY DEMO")
    print("=" * 60)

    demo_words = ["se", "zaldrīzoti", "dārys", "jorrāelagon", "perzys", "henujagon"]
    for word in demo_words:
        if word not in token_to_id:
            continue
        similar = find_similar(word, embeddings, token_to_id, top_k=5)
        print(f"\n  '{word}' →")
        for tok, tid, sim in similar:
            print(f"    {tok:<22} (ID {tid:>3})  sim={sim:+.4f}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
