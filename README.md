# Working with Text Data – Tokenisasi & Text Embedding

Tugas individu NLP Week 3: membangun pipeline pemrosesan teks dari awal, mulai dari pengumpulan korpus hingga analisis embedding.

## Corpus

- **Bahasa**: High Valyrian (constructed language dari universe Game of Thrones)
- **Sumber**: [High Valyrian Corpus from Game of Thrones – Kaggle](https://www.kaggle.com/datasets/viceriomarinowski/high-valyrian-corpus-from-game-of-thrones?resource=download)
- **Ukuran**: ~14.000+ kata, 497 unique words

## Pipeline

### 1. Corpus Collection ✅
- File: `raw_corpus.txt`
- Korpus teks mentah sebelum cleaning

### 2. Text Cleaning ✅
- Script: `clean_corpus.py`
- Output: `cleaned_corpus.txt`
- Proses:
  - Normalize line endings
  - Strip whitespace
  - Hapus dialog markers & tanda kutip
  - Normalize punctuation
  - Collapse blank lines & deduplikasi baris berurutan
  - Lowercase semua teks

### 3. Custom Tokenizer ✅
- File: `tokenizer.py`
- Word-level tokenizer (custom, tanpa library siap pakai)
- Punctuation dipisah sebagai token tersendiri
- Setiap baris dibungkus `<BOS>` dan `<EOS>`
- Fungsi: `tokenize()`, `encode()`, `decode()`

### 4. Vocabulary & Token ID Mapping ✅
- File: `vocab.json` — 299 tokens (4 special + 295 unique words)
- File: `encoded_tokens.csv` — 666 baris, 18.125 total tokens
- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- Vocab diurutkan berdasarkan frekuensi (descending)

### 5. Embedding ✅
- Script: `embedding.py`
- Output: `embedding.npy` (299×50, 116.8 KB), `embedding_meta.json`
- Metode: Co-occurrence matrix → PPMI → SVD (50 dimensi)
- Context window: 4

### 6. Analysis ✅
- File: `analysis.ipynb`
- Pipeline terpadu: cleaning → tokenizer → vocab → embedding → analisis
- Visualisasi: Top-20 chart, Zipf's Law, SVD decay, similarity heatmap, PCA 2D, t-SNE 2D
- Ringkasan statistik lengkap

## Deliverables

| File | Status |
|---|---|
| `raw_corpus.txt` | ✅ |
| `cleaned_corpus.txt` | ✅ |
| `tokenizer.py` | ✅ |
| `vocab.json` | ✅ |
| `encoded_tokens.csv` | ✅ |
| `embedding.py` | ✅ |
| `embedding.npy` | ✅ |
| `embedding_meta.json` | ✅ |
| `analysis.ipynb` | ✅ |
| `README.md` | ✅ |

## How to Run

```bash
# Text cleaning
python clean_corpus.py

# Tokenizer + vocab + encoding
python tokenizer.py

# Word embedding
python embedding.py
```
