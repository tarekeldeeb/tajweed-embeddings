# Quran Tajweed Embeddings – Tajwīd-Aware Embedding Engine for Quranic Recitation AI

<!-- GitHub Actions Tests -->
[![Test Status](https://github.com/tarekeldeeb/tajweed-embeddings/actions/workflows/tests.yml/badge.svg)](https://github.com/tarekeldeeb/tajweed-embeddings/actions/workflows/tests.yml)

Tajwīd-aware embedding engine for Qur'ān (Uthmānī script). Encodes letters, harakāt, pause marks, ṣifāt, and tajwīd rules from curated spans. Ships with packaged Quran/rule data, a CLI for inspection, and a full pytest suite.

---

## What You Get

- Tajwīd embeddings for the full corpus (114 sūrahs / 6236 āyāt), one vector per phoneme/letter. Quran string is recoverable from the embeddings.
- JSON-backed rule spans (`tajweed.rules.json`) plus inline markers (iqlab, tas-heel, imala, ishmam, optional seen). Rules source: https://github.com/cpfair/quran-tajweed
- Compact 6-bit ṣifāt encoding and explicit haraka states (tanwīn, shadda combos, madd, alternate sukūn).
- Pretty-printing and reconstruction via `encoding_to_string(style="short"|"long")` and `embedding_to_text`.
- Similarity helpers (`compare`, `score`) for alignment/scoring workflows.
- Auto-bootstrap for missing data files (downloads Tanzil text and regenerates spans when absent).
- CLI (`tajweed-embeddings`) and pytest coverage.

## Embedding Layout (dim 90)

```
[ letters | haraka | pause | sifat | rules ]
    46       12        3       6       23
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
pip install tajweed-embeddings
# or for development/testing
python3 -m pip install -e .[test]
# or install from GitHub
python3 -m pip install "git+https://github.com/tarekeldeeb/tajweed-embeddings.git"
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
tajweed-embeddings --sura 1 --aya 1 --style short
tajweed-embeddings --sura 2 --aya 1 --count 3 --style long
```

Outputs a human-readable view of the vectors (for inspection; not the raw numeric arrays).

Example output (full, collapsible):

<details>
<summary>tajweed-embeddings --sura 2 --aya 1 --count 3</summary>

```text
% tajweed-embeddings --sura 2 --aya 1 --count 3
╔══════════════════════════════════════════════════════════╗
║ TajweedEmbedder CLI                                      ║
║   For inspection only — use programmatically for models. ║
║   String output is a human view, NOT the numeric vectors.║
║                                                          ║
║ ┌ Index: row number                                      ║
║ │  ┌ Letter: glyph                                       ║
║ │  │   ┌ Tashkeel: Kasra ‿ , Fatha ^ , .. etc            ║
║ │  │   │   ┌ Pause: stop mark (0/4/6 etc)                ║
║ │  │   │   │   ┌ Jahr 🔊 , Hams 🤫                       ║
║ │  │   │   │   │  ┌ Rikhw 💨 , Tawasot ➖ , Shidda 🚫    ║
║ │  │   │   │   │  │  ┌ Isti'la 🔼 , Istifal 🔻           ║
║ │  │   │   │   │  │  │  ┌ Infitah ▲ , Itbaq ⟂            ║
║ │  │   │   │   │  │  │  │  ┌ Idhlaq 😮 , Ismat 😐        ║
║ │  │   │   │   │  │  │  │  │    ┌ Rules: Tajweed flags   ║
║ │  │   │   │   │  │  │  │  │    │                        ║
╚═╪══╪═══╪═══╪═══╪══╪══╪══╪══╪════╪════════════════════════╝
[ 0] ب | ‿  | - | 🔊 🚫 🔻 ⟂ 😮 |                           
[ 1] س | °  | - | 🤫 💨 🔻 ⟂ 😐 |                           
[ 2] م | ‿  | ! | 🔊 ➖ 🔻 ⟂ 😮 |                           
[ 3] ٱ |    | - | 🔊 💨 🔻 ⟂ 😐 |                           
[ 4] ل |    | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[ 5] ل | ώ  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[ 6] ه | ‿  | ! | 🤫 💨 🔻 ⟂ 😐 |                           
[ 7] ٱ |    | - | 🔊 💨 🔻 ⟂ 😐 |                           
[ 8] ل |    | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[ 9] ر | ώ  | - | 🔊 ➖ 🔼 ⟂ 😮 |                           
[10] ح | °  | - | 🤫 💨 🔻 ⟂ 😐 |                           
[11] م | ^  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[12] ـٰ |    | - | 🔊 💨 🔻 ⟂ 😐 |                           
[13] ن | ‿  | ! | 🔊 ➖ 🔻 ⟂ 😮 |                           
[14] ٱ |    | - | 🔊 💨 🔻 ⟂ 😐 |                           
[15] ل |    | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[16] ر | ώ  | - | 🔊 ➖ 🔼 ⟂ 😮 |                           
[17] ح | ‿  | - | 🤫 💨 🔻 ⟂ 😐 |                           
[18] ي |    | - | 🔊 💨 🔻 ⟂ 😐 |                           
[19] م | ‿  | ! | 🔊 ➖ 🔻 ⟂ 😮 |                           
[20] ا |    | - | 🔊 💨 🔻 ⟂ 😐 |                           
[21] ل | ~  | - | 🔊 ➖ 🔻 ⟂ 😮 | madd_6                    
[22] م | ~  | ⏹ | 🔊 ➖ 🔻 ⟂ 😮 | madd_6                    
[23] ذ | ^  | - | 🔊 ➖ 🔻 ⟂ 😐 |                           
[24] ـٰ |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[25] ل | ‿  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[26] ك | ^  | ! | 🤫 🚫 🔻 ⟂ 😐 |                           
[27] ٱ |    | - | 🔊 💨 🔻 ⟂ 😐 | hamzat_wasl               
[28] ل | °  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[29] ك | ‿  | - | 🤫 🚫 🔻 ⟂ 😐 |                           
[30] ت | ^  | - | 🤫 🚫 🔻 ⟂ 😐 |                           
[31] ـٰ |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[32] ب | و  | ! | 🔊 🚫 🔻 ⟂ 😮 |                           
[33] ل | ^  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[34] ا |    | ! | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[35] ر | ^  | - | 🔊 ➖ 🔼 ⟂ 😮 |                           
[36] ي | °  | - | 🔊 💨 🔻 ⟂ 😐 |                           
[37] ب | ^  | ⋀ | 🔊 🚫 🔻 ⟂ 😮 |                           
[38] ف | ‿  | - | 🤫 💨 🔻 ⟂ 😮 |                           
[39] ي |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[40] ه | ‿  | ⋀ | 🤫 💨 🔻 ⟂ 😐 |                           
[41] ه | و  | - | 🤫 💨 🔻 ⟂ 😐 |                           
[42] د | ^^ | - | 🔊 🚫 🔻 ⟂ 😐 | idghaam_no_ghunnah        
[43] ى |    | ! | 🔊 💨 🔻 ⟂ 😐 | idghaam_no_ghunnah, madd_2
[44] ل | ῳ  | - | 🔊 ➖ 🔻 ⟂ 😮 | idghaam_no_ghunnah        
[45] ل | °  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[46] م | و  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[47] ت | ώ  | - | 🤫 🚫 🔻 ⟂ 😐 |                           
[48] ق | ‿  | - | 🔊 🚫 🔼 ⟂ 😐 |                           
[49] ي |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_246                  
[50] ن | ^  | ⏹ | 🔊 ➖ 🔻 ⟂ 😮 |                           
[51] ٱ |    | - | 🔊 💨 🔻 ⟂ 😐 |                           
[52] ل | ώ  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[53] ذ | ‿  | - | 🔊 ➖ 🔻 ⟂ 😐 |                           
[54] ي |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[55] ن | ^  | ! | 🔊 ➖ 🔻 ⟂ 😮 |                           
[56] ي | و  | - | 🔊 💨 🔻 ⟂ 😐 |                           
[57] ؤ | °  | - | 🤫 🚫 🔻 ⟂ 😐 |                           
[58] م | ‿  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[59] ن | و  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[60] و |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[61] ن | ^  | ! | 🔊 ➖ 🔻 ⟂ 😮 |                           
[62] ب | ‿  | - | 🔊 🚫 🔻 ⟂ 😮 |                           
[63] ٱ |    | - | 🔊 💨 🔻 ⟂ 😐 | hamzat_wasl               
[64] ل | °  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[65] غ | ^  | - | 🔊 💨 🔼 ⟂ 😐 |                           
[66] ي | °  | - | 🔊 💨 🔻 ⟂ 😐 |                           
[67] ب | ‿  | ! | 🔊 🚫 🔻 ⟂ 😮 |                           
[68] و | ^  | - | 🔊 💨 🔻 ⟂ 😐 |                           
[69] ي | و  | - | 🔊 💨 🔻 ⟂ 😐 |                           
[70] ق | ‿  | - | 🔊 🚫 🔼 ⟂ 😐 |                           
[71] ي |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[72] م | و  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[73] و |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[74] ن | ^  | ! | 🔊 ➖ 🔻 ⟂ 😮 |                           
[75] ٱ |    | - | 🔊 💨 🔻 ⟂ 😐 | hamzat_wasl               
[76] ل |    | - | 🔊 ➖ 🔻 ⟂ 😮 | lam_shamsiyyah            
[77] ص | ώ  | - | 🤫 💨 🔼 ▲ 😐 |                           
[78] ل | ^  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[79] و |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2, silent            
[80] ـٰ |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[81] ة | ^  | ! | 🤫 🚫 🔻 ⟂ 😐 |                           
[82] و | ^  | - | 🔊 💨 🔻 ⟂ 😐 |                           
[83] م | ‿  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[84] م | ώ  | - | 🔊 ➖ 🔻 ⟂ 😮 | ghunnah                   
[85] ا |    | ! | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[86] ر | ^  | - | 🔊 ➖ 🔼 ⟂ 😮 |                           
[87] ز | ^  | - | 🔊 💨 🔻 ⟂ 😐 |                           
[88] ق | °  | - | 🔊 🚫 🔼 ⟂ 😐 | qalqalah                  
[89] ن | ^  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[90] ـٰ |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[91] ه | و  | - | 🤫 💨 🔻 ⟂ 😐 |                           
[92] م | °  | ! | 🔊 ➖ 🔻 ⟂ 😮 |                           
[93] ي | و  | - | 🔊 💨 🔻 ⟂ 😐 |                           
[94] ن |    | - | 🔊 ➖ 🔻 ⟂ 😮 | ikhfa                     
[95] ف | ‿  | - | 🤫 💨 🔻 ⟂ 😮 |                           
[96] ق | و  | - | 🔊 🚫 🔼 ⟂ 😐 |                           
[97] و |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_246                  
[98] ن | ^  | ⏹ | 🔊 ➖ 🔻 ⟂ 😮 |     
```
</details>

## Data + Regeneration

Packaged data includes `sifat.json` and the rule trees under `rules_gen/rule_trees/`. If `quran.json` or `tajweed.rules.json` are missing or empty, `TajweedEmbedder` will download the Tanzil Uthmani text and regenerate spans via `rules_gen/tajweed_classifier.py` (requires `requests` and `tqdm`); this needs internet on first run only. Corpus coverage: 114 sūrahs / 6236 āyāt.

## Tests

```bash
python3 -m pip install -e .[test]
pytest -q
```

## License

Dual-licensed: Waqf Public License 2.0 for non-commercial use; commercial or other uses require permission. See `LICENSE`.
