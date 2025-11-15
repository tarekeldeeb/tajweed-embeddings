
# Tajweed-Model – Tajwīd-Aware Embedding Engine for Quranic Recitation AI

This project provides a **complete embedding engine** for Qur'ān text that encodes:

- Arabic letter identity (one-hot)
- Harakāt (fatḥah, kasrah, ḍammah, sukoon, shaddah)
- Ṣifāt al-ḥurūf (12 phonetic properties)
- Tajwīd rule flags based on structured JSON rule spans  
  (idghām, ikhfa’, iqlāb, madd types, qalqalah, ghunnah…)
- Automatic reconstruction from embedding → text
- Similarity scoring (cosine)
- pytest-based test suite

It is designed as the **core feature extractor** for a full Tajwīd Teaching AI:
- STT → phoneme alignment  
- Tajwīd error detection  
- Recitation scoring  
- Feedback generation  

This repository implements the **embedding layer**, not the full pipeline.

---

# 🚀 Features

### ✔ **Tajweed-aware embeddings**
Every character in the Qur’ān is transformed into a numeric vector containing:

1. **Letter one-hot**
2. **Harakah one-hot**
3. **Ṣifāt 12-dimensional vector**
4. **Tajwīd rule flags (n rules)**

### ✔ **JSON-based Tajwīd rule spans**
Rules are not guessed — they come from curated JSON files.

### ✔ **Embedding → text reconstruction**
Allows round-trip conversion for testing and diagnostics.

### ✔ **Scoring and similarity**
Cosine similarity over embedding sequences.

### ✔ **Full pytest test suite**
Ensures correct behavior across:
- Harakāt
- Shaddah
- Unknown letters
- Empty input
- Long sequences
- Reconstruction stability

---

# 📁 Project Structure

```
tajweed-model/
│
├── tajweed_embedder.py
├── sifat.json
├── tajweed.hafs.uthmani-pause-sajdah.json
│
├── test/
│   └── test_tajweed_embedder.py
│
├── README.md
└── venv/
```

---

# 🔧 Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pytest
```

---

# 📦 Usage

## Load files

```python
import json
from tajweed_embedder import TajweedEmbedder

sifat = json.load(open("sifat.json"))
rules = json.load(open("tajweed.hafs.uthmani-pause-sajdah.json"))

emb = TajweedEmbedder(sifat, rules)
```

## Convert text → embedding

```python
vecs = emb.text_to_embedding("بِسْمِ", "1", "1")
```

## Convert embedding → text

```python
emb.embedding_to_text(vecs)
```

## Compare two recitations

```python
emb.compare(e1, e2)
```

## Score recitation

```python
emb.score(e1, e2)
```

---

# 🧪 Running Tests

```bash
pytest -q
```

---

# License

Please contact author
