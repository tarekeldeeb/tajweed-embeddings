# Quran Tajweed Embeddings – Tajwīd-Aware Embedding Engine for Quranic Recitation AI
  
<!-- GitHub Actions Tests -->
[![Test Status](https://github.com/tarekeldeeb/tajweed-embeddings/actions/workflows/tests.yml/badge.svg)](https://github.com/tarekeldeeb/tajweed-embeddings/actions/workflows/tests.yml)

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

## 🚀 Features

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

### 🧩 Embedding Vector Layout

Each phoneme (1+ character) in the text → one vector:

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                                EMBEDDING VECTOR                               │
└───────────────────────────────────────────────────────────────────────────────┘
[ LETTER (one-hot) | HARAKA (one-hot) | SIFAT (12 floats) | RULE FLAGS (N bits) ]
        ^                   ^                   ^                    ^
        |                   |                   |                    |
   0..L-1           L..H-1            H..H+12-1            (rest of vector)
```

---

## 🔧 Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pytest
```

---

## 📦 Quick Setup

```python
from tajweed_embedder import TajweedEmbedder

emb = TajweedEmbedder()
```

## Usage Examples

### 1️⃣ Embedding a full āyah

```python
vecs = emb.text_to_embedding(1, 1)
print(len(vecs))
```

Expected:

```text
38
```

### 2️⃣ Embedding a sub-string

```python
emb.text_to_embedding(1, 1, "بِسْ")
```

Expected: `3 vectors`

### 3️⃣ Embedding a full surah

```python
full = emb.text_to_embedding(1)
len(full)
```

Expected: `112`

### 4️⃣ Embedding → Text (Reversible)

```python
txt = emb.embedding_to_text(emb.text_to_embedding(1, 1, "بِسْمِ"))
print(txt)
```

Expected: `بِسْمِ`

### 5️⃣ Cosine Similarity

```python
e1 = emb.text_to_embedding(1, 1, "بِسْ")
e2 = emb.text_to_embedding(1, 1, "بَسْ")
emb.compare(e1, e2)
```

Expected: `~0.95`

### 6️⃣ Per-character score

```python
emb.score(e1, e2)
```

Expected: `~0.95`

### 7️⃣ Arabic non-Quranic text

```python
emb.text_to_embedding(1, 1, "سلام عليكم")
```

Expected: length preserved.

### 8️⃣ Special Quran symbols

```python
emb.text_to_embedding(1, 1, "بِسْمِ ۩ اللَّهِ")
```

Symbols produce zero vectors.

### 9️⃣ Cross-ayah concatenation

```python
q = emb.quran["1"]
combined = q["1"] + " " + q["2"]
emb.text_to_embedding(1, subtext=combined)
```

Expected length: 76

### 🔟 Random fuzzing

```python
seq = "".join(random.choice(list(emb.letters)+list(emb.harakat)) for _ in range(50))
emb.text_to_embedding(1, 1, seq)
```

Expected: 50

---

## 🧪 Running Tests

```bash
pytest -q
```

---

## License

Please contact author: Tarek Eldeeb
