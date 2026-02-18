"""
Custom Tokenizer for High Valyrian Corpus
==========================================
Implements a word-level tokenizer from scratch (no NLTK/SpaCy/HuggingFace).

Features:
- Word-level tokenization with punctuation separation
- Vocabulary building with frequency counts
- Token-to-ID mapping (vocab.json)
- Corpus encoding (encoded_tokens.csv)
- Special tokens: <PAD>, <UNK>, <BOS>, <EOS>
"""

import re
import json
import csv
from pathlib import Path
from collections import Counter


# --- Paths ---
BASE_DIR = Path(__file__).parent
CLEANED_CORPUS = BASE_DIR / "cleaned_corpus.txt"
VOCAB_FILE = BASE_DIR / "vocab.json"
ENCODED_FILE = BASE_DIR / "encoded_tokens.csv"


# ===========================================================
# TOKENIZER
# ===========================================================

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def tokenize(text: str) -> list[str]:
    """
    Tokenize a string into a list of tokens.

    Rules:
    1. Lowercase (corpus already lowered, but just in case)
    2. Separate punctuation from words (.,!?;:) as own tokens
    3. Split on whitespace
    4. Remove empty tokens
    """
    text = text.lower()

    # Insert space before/after punctuation so they become separate tokens
    text = re.sub(r'([.!?,;:])', r' \1 ', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    return tokens


def tokenize_line(line: str) -> list[str]:
    """Tokenize a single line, wrapping with <BOS> and <EOS>."""
    tokens = tokenize(line)
    if tokens:
        return ["<BOS>"] + tokens + ["<EOS>"]
    return []


# ===========================================================
# VOCABULARY
# ===========================================================

def build_vocab(all_tokens: list[str], min_freq: int = 1) -> dict:
    """
    Build vocabulary from token list.

    Returns dict: {token: id}
    Sorted by frequency (descending), with special tokens at the start.
    """
    freq = Counter(all_tokens)

    # Filter by min frequency (exclude special tokens from filtering)
    filtered = {
        tok: count for tok, count in freq.items()
        if count >= min_freq and tok not in SPECIAL_TOKENS
    }

    # Sort by frequency descending, then alphabetically for ties
    sorted_tokens = sorted(filtered.keys(), key=lambda t: (-filtered[t], t))

    # Build vocab: special tokens first, then sorted tokens
    vocab = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        vocab[tok] = i

    offset = len(SPECIAL_TOKENS)
    for i, tok in enumerate(sorted_tokens):
        vocab[tok] = offset + i

    return vocab, freq


def encode(tokens: list[str], vocab: dict) -> list[int]:
    """Convert token list to ID list using vocab. Unknown tokens get <UNK> id."""
    unk_id = vocab["<UNK>"]
    return [vocab.get(tok, unk_id) for tok in tokens]


def decode(ids: list[int], vocab: dict) -> list[str]:
    """Convert ID list back to token list."""
    id_to_token = {v: k for k, v in vocab.items()}
    return [id_to_token.get(i, "<UNK>") for i in ids]


# ===========================================================
# MAIN PIPELINE
# ===========================================================

def main():
    # --- Read cleaned corpus ---
    text = CLEANED_CORPUS.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # --- Tokenize all lines ---
    all_tokenized_lines = []
    all_tokens_flat = []

    for line in lines:
        tokens = tokenize_line(line)
        if tokens:
            all_tokenized_lines.append(tokens)
            all_tokens_flat.extend(tokens)

    # --- Build vocabulary ---
    vocab, freq = build_vocab(all_tokens_flat)

    # --- Save vocab.json ---
    vocab_export = {
        "_meta": {
            "total_tokens": len(all_tokens_flat),
            "vocab_size": len(vocab),
            "special_tokens": SPECIAL_TOKENS,
        },
        "token_to_id": vocab,
    }
    VOCAB_FILE.write_text(
        json.dumps(vocab_export, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- Encode all lines & save encoded_tokens.csv ---
    with open(ENCODED_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["line_num", "tokens", "token_ids"])

        for i, tokens in enumerate(all_tokenized_lines):
            ids = encode(tokens, vocab)
            writer.writerow([
                i + 1,
                " ".join(tokens),
                " ".join(str(x) for x in ids),
            ])

    # --- Print stats ---
    print("=" * 55)
    print("TOKENIZER & VOCABULARY REPORT")
    print("=" * 55)
    print(f"Lines tokenized:      {len(all_tokenized_lines)}")
    print(f"Total tokens:         {len(all_tokens_flat)}")
    print(f"Vocab size:           {len(vocab)}")
    print(f"  - Special tokens:   {len(SPECIAL_TOKENS)}")
    print(f"  - Unique words:     {len(vocab) - len(SPECIAL_TOKENS)}")
    print("-" * 55)
    print("Top 20 most frequent tokens:")
    for tok, count in freq.most_common(20):
        tok_id = vocab.get(tok, "?")
        print(f"  [{tok_id:>4}] {tok:<25} {count:>5}x")
    print("-" * 55)
    print(f"Saved: {VOCAB_FILE.name}")
    print(f"Saved: {ENCODED_FILE.name}")
    print("=" * 55)

    # --- Quick sanity check: encode & decode first line ---
    sample = all_tokenized_lines[0]
    encoded_sample = encode(sample, vocab)
    decoded_sample = decode(encoded_sample, vocab)
    print("\n--- Sanity Check (Line 1) ---")
    print(f"Tokens:  {sample[:10]}...")
    print(f"IDs:     {encoded_sample[:10]}...")
    print(f"Decoded: {decoded_sample[:10]}...")
    print(f"Match:   {sample == decoded_sample}")


if __name__ == "__main__":
    main()
