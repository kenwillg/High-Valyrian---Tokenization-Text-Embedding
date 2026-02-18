"""Inject vocab and frequency data into vocab_explorer.html"""
import json
import re
from pathlib import Path
from collections import Counter

base = Path(__file__).parent

# Load vocab
with open(base / "vocab.json", encoding="utf-8") as f:
    data = json.load(f)
vocab = data["token_to_id"]

# Read cleaned corpus & compute token frequencies
text = (base / "cleaned_corpus.txt").read_text(encoding="utf-8")
lines = [l.strip() for l in text.split("\n") if l.strip()]

all_tokens = []
for line in lines:
    line = line.lower()
    line = re.sub(r'([.!?,;:])', r' \1 ', line)
    line = re.sub(r'\s+', ' ', line).strip()
    tokens = ["<BOS>"] + line.split() + ["<EOS>"]
    all_tokens.extend(tokens)

freq = dict(Counter(all_tokens))

# Read HTML template and replace placeholders
html = (base / "vocab_explorer.html").read_text(encoding="utf-8")
html = html.replace("VOCAB_PLACEHOLDER", json.dumps(vocab, ensure_ascii=False))
html = html.replace("FREQ_PLACEHOLDER", json.dumps(freq, ensure_ascii=False))

(base / "vocab_explorer.html").write_text(html, encoding="utf-8")
print(f"Done! Vocab: {len(vocab)} tokens, Freq: {len(freq)} entries")
