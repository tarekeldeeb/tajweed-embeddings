# Quran Tajweed Embeddings – Tajwīd-Aware Embedding Engine for Quranic Recitation AI

<!-- GitHub Actions Tests -->
[![Test Status](https://github.com/tarekeldeeb/tajweed-embeddings/actions/workflows/tests.yml/badge.svg)](https://github.com/tarekeldeeb/tajweed-embeddings/actions/workflows/tests.yml)

Tajwīd-aware embedding engine for Qur'ān (Uthmānī script). Encodes letters, harakāt, pause marks, ṣifāt, and tajwīd rules from curated spans. Ships with packaged Quran/rule data, a CLI for inspection, and a full pytest suite.

---

## What You Get

- Tajwīd embeddings for the full corpus (114 sūrahs / 6236 āyāt), one vector per letter/marker.
- JSON-backed rule spans (`tajweed.rules.json`) plus inline markers (iqlab, tas-heel, imala, ishmam, optional seen).
- Compact 6-bit ṣifāt encoding and explicit haraka states (tanwīn, shadda combos, madd, alternate sukūn).
- Pretty-printing and reconstruction via `encoding_to_string(style="short"|"long")` and `embedding_to_text`.
- Similarity helpers (`compare`, `score`) for alignment/scoring workflows.
- Auto-bootstrap for missing data files (downloads Tanzil text and regenerates spans when absent).
- CLI (`tajweed_embedder`) and pytest coverage.

## Embedding Layout (dim 91)

```
[ letters | haraka | pause | sifat | rules ]
    47       12        3       6       23
```

- **Letters:** Uthmānī glyph set; pause glyphs live in the pause slice, not the letter one-hot.
- **Haraka:** Explicit states including shadda combos, tanwīn, madd, sukūn, and zero-sukūn.
- **Pause:** 3-bit stop categories:
  - 0: do_not_stop
  - 1: word_boundary_emergency (default at word ends without explicit marks)
  - 2: seli (↦)
  - 3: jaiz (≈)
  - 4: taanoq (⋀)
  - 5: qeli_or_ayah_end (⏹)
  - 6: sakta (˽)
  - 7: lazem (⛔)
- **Ṣifāt:** 6-bit compact vector (jahr/hams; rikhwah–tawassut–shiddah; isti'la/istifal; infitah/itbaq; idhlaq/ismat).
- **Rules:** 23 flags (19 from `tajweed.rules.json` spans + 4 inline marker rules: tas_heel, imala, ishmam, optional_seen).

## Install

Runtime dependency is `numpy`; `requests`/`tqdm` are optional for regenerating data.

```bash
python3 -m pip install .
# or for development/testing
python3 -m pip install -e .[test]
```

## Quickstart (Python)

```python
from tajweed_embeddings import TajweedEmbedder

emb = TajweedEmbedder()

vecs = emb.text_to_embedding(1, 1)              # sura 1, āyah 1
sub = emb.text_to_embedding(1, 1, "بِسْمِ")     # custom text (rules skipped)

print(emb.embedding_dim)                        # 90
print(emb.encoding_to_string(sub, style="short"))

round_trip = emb.embedding_to_text(sub)
score = emb.score(sub, emb.text_to_embedding(1, 1, "بَسْمِ"))
```

Notes:
- `subtext` embeds arbitrary strings; diacritics/pause marks attach to the previous letter and do not increase vector count.
- `count` embeds consecutive āyāt starting at `ayah`.
- `encoding_to_string(style="long")` produces labeled, multi-field output; `"short"` is tabular.

## CLI

Inspect embeddings without writing code:

```bash
tajweed_embedder --sura 1 --aya 1 --style short
tajweed_embedder --sura 2 --aya 1 --count 3 --style long
```

Outputs a human-readable view of the vectors (for inspection; not the raw numeric arrays).

## Data + Regeneration

Packaged data lives in `src/tajweed_embeddings/data/` (`quran.json`, `sifat.json`, `tajweed.rules.json`). If any file is missing or empty, `TajweedEmbedder` will download the Tanzil Uthmani text and regenerate spans via `rules_gen/tajweed_classifier.py` (requires `requests` and `tqdm`). Corpus coverage: 114 sūrahs / 6236 āyāt.

## Tests

```bash
python3 -m pip install -e .[test]
pytest -q
```

## License

Dual-licensed: Waqf Public License 2.0 for non-commercial use; commercial or other uses require permission. See `LICENSE`.
