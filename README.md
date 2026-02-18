# Working with Text Data – Tokenisasi & Text Embedding

Tugas individu NLP Week 3: membangun pipeline pemrosesan teks dari awal, mulai dari pengumpulan korpus hingga analisis embedding.

## Corpus

- **Bahasa**: High Valyrian (constructed language dari universe Game of Thrones)
- **Sumber**: Teks naratif, dialog, puisi, dan prosa yang dikumpulkan secara mandiri
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

### 3. Custom Tokenizer
- File: `tokenizer.py`
- _TODO_

### 4. Vocabulary & Token ID Mapping
- File: `vocab.json`
- File: `encoded_tokens.csv`
- _TODO_

### 5. Embedding
- File: `embedding.npy`
- _TODO_

### 6. Analysis
- File: `analysis.ipynb`
- _TODO_

## Deliverables

| File | Status |
|---|---|
| `raw_corpus.txt` | ✅ |
| `cleaned_corpus.txt` | ✅ |
| `tokenizer.py` | ⬜ |
| `vocab.json` | ⬜ |
| `encoded_tokens.csv` | ⬜ |
| `embedding.npy` | ⬜ |
| `analysis.ipynb` | ⬜ |
| `README.md` | ✅ |

## How to Run

```bash
# Text cleaning
python clean_corpus.py
```
