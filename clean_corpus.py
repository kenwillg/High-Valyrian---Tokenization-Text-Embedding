"""
Text Cleaning Script for High Valyrian Corpus
==============================================
Reads raw_corpus.txt and produces cleaned_corpus.txt

Cleaning steps performed:
1. Normalize line endings (\r\n -> \n)
2. Strip leading/trailing whitespace per line
3. Remove dialog markers (e.g., "Dāria: " -> "")  
4. Remove quotation marks (straight and curly)
5. Normalize punctuation: remove stray/excessive punctuation
6. Collapse multiple blank lines into single blank line
7. Remove exact duplicate consecutive lines
8. Lowercase all text
9. Remove lines that are completely empty after cleaning
10. Final trim and ensure single newline at end
"""

import re
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "raw_corpus.txt"
OUTPUT_FILE = Path(__file__).parent / "cleaned_corpus.txt"


def clean_corpus():
    # --- Read raw ---
    raw_text = INPUT_FILE.read_text(encoding="utf-8")
    
    # --- Step 1: Normalize line endings ---
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    
    # --- Step 1b: Remove any remaining \r ---
    text = text.replace("\r", "")
    
    # --- Split into lines for per-line processing ---
    lines = text.split("\n")
    
    cleaned_lines = []
    for line in lines:
        # Step 2: Strip whitespace
        line = line.strip()
        
        # Step 3: Remove dialog markers (Name: at the start of a line)
        # Pattern: word(s) followed by colon at start of line (unicode-aware)
        line = re.sub(r'^\w+(\s+\w+)?\s*:\s*', '', line, flags=re.UNICODE)
        
        # Step 4: Remove quotation marks (straight and curly/smart quotes)
        line = line.replace('"', '').replace('"', '').replace('"', '')
        line = line.replace("'", "").replace("'", "").replace("'", "")
        
        # Step 5: Normalize punctuation
        # Remove multiple consecutive punctuation (e.g., ".." -> ".")
        line = re.sub(r'([.!?])\1+', r'\1', line)
        # Remove spaces before punctuation
        line = re.sub(r'\s+([.!?,;:])', r'\1', line)
        # Ensure single space after punctuation (if followed by a letter)
        line = re.sub(r'([.!?,;:])\s*([A-Za-zĀ-ž])', r'\1 \2', line)
        
        # Step 6: Collapse multiple spaces into one
        line = re.sub(r'\s+', ' ', line)
        
        # Step 7: Strip again after all processing
        line = line.strip()
        
        cleaned_lines.append(line)
    
    # --- Step 8: Remove exact duplicate consecutive lines ---
    deduped_lines = []
    prev_line = None
    for line in cleaned_lines:
        if line == "" and prev_line == "":
            continue  # Collapse multiple blank lines
        if line != "" and line == prev_line:
            continue  # Skip exact duplicate consecutive lines
        deduped_lines.append(line)
        prev_line = line
    
    # --- Step 9: Lowercase all text ---
    deduped_lines = [line.lower() for line in deduped_lines]
    
    # --- Step 10: Remove leading/trailing blank lines ---
    while deduped_lines and deduped_lines[0] == "":
        deduped_lines.pop(0)
    while deduped_lines and deduped_lines[-1] == "":
        deduped_lines.pop()
    
    # --- Join and write output ---
    cleaned_text = "\n".join(deduped_lines) + "\n"
    OUTPUT_FILE.write_text(cleaned_text, encoding="utf-8")
    
    # --- Print stats ---
    raw_lines = raw_text.count("\n") + 1
    raw_words = len(raw_text.split())
    clean_words = len(cleaned_text.split())
    clean_lines_count = cleaned_text.count("\n")
    
    # Count unique words
    unique_raw = len(set(raw_text.lower().split()))
    unique_clean = len(set(cleaned_text.split()))
    
    print("=" * 50)
    print("CORPUS CLEANING REPORT")
    print("=" * 50)
    print(f"Raw file:     {INPUT_FILE.name}")
    print(f"Cleaned file: {OUTPUT_FILE.name}")
    print("-" * 50)
    print(f"Raw lines:         {raw_lines}")
    print(f"Cleaned lines:     {clean_lines_count}")
    print(f"Raw word count:    {raw_words}")
    print(f"Cleaned word count:{clean_words}")
    print(f"Raw unique words:  {unique_raw}")
    print(f"Clean unique words:{unique_clean}")
    print(f"Lines removed:     {raw_lines - clean_lines_count}")
    print(f"Words removed:     {raw_words - clean_words}")
    print("=" * 50)
    print(f"\nCleaned corpus saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_corpus()
