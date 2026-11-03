# Quran Tajweed Embeddings – Tajwīd-Aware Embedding Engine for Quranic Recitation AI

<!-- GitHub Actions Tests -->
[![Test Status](https://github.com/tarekeldeeb/tajweed-embeddings/actions/workflows/tests.yml/badge.svg)](https://github.com/tarekeldeeb/tajweed-embeddings/actions/workflows/tests.yml)

Tajwīd-aware embedding engine for the Qur'ān (Uthmānī script, riwāyat Ḥafṣ). Every letter becomes an 86-dimensional vector encoding the letter, its haraka, pause mark, ṣifāt and tajwīd rules. Primary vectors describe continuous recitation (waṣl), and every word also carries pre-computed stop (waqf) and start (ibtidāʾ) variants. Includes a CLI for inspection and a pytest suite.

---

## What You Get

- Tajwīd embeddings for the full corpus (114 sūrahs / 6236 āyāt), one vector per letter. Diacritics and pause marks fold into the letter they follow.
- Waqf/ibtidāʾ variants for every word boundary (`stop_variants` / `start_variants`, `realize()`), so an aligner can accept a pause at any word without re-embedding.
- Rule spans labeled in waṣl across āyah (and sūrah) boundaries by the bundled decision-tree classifier (`rules_gen/`, derived from https://github.com/cpfair/quran-tajweed), plus inline marker rules (iqlab, tas-heel, imala, ishmam, optional seen).
- Compact 6-bit ṣifāt with contextual tafkhīm/tarqīq for rāʾ and the lām of the Divine Name, and explicit haraka states (tanwīn, shadda combos, madd, alternate sukūn).
- Human-readable views via `encoding_to_string(style="short"|"long", secondary=...)` and approximate text reconstruction via `embedding_to_text`.
- Similarity helpers (`compare`, `score`) for alignment/scoring workflows.
- Self-bootstrapping data: downloads the Tanzil text and (re)generates the rule spans when they are missing or stale.
- CLI (`tajweed-embeddings`) and a pytest suite.

## Embedding Layout (dim 86)

```
[ letters | haraka | pause | sifat | rules ]
    41       13        3       6       23
   0–40     41–53   54–56   57–62   63–85
```

Offsets are exposed as `emb.idx_haraka_start`, `emb.idx_pause_start`, `emb.idx_sifat_start`, `emb.idx_rule_start`; slice widths as `emb.n_letters`, `emb.n_harakat`, `emb.n_pause`, `emb.n_sifat`, `emb.n_rules`; the total as `emb.embedding_dim`.

- **Letters (one-hot, 41):** the keys of `sifat.json` minus pause/diacritic/marker glyphs, sorted by codepoint (`emb.letters`). This includes the hamza forms, `ة`, `ٱ`, the dagger alif (`ٰ`), the small wāw/yāʾ (`ۥ`/`ۦ`) and hamza above (`ٔ`). `آ` is encoded as `ا` + `madd`. Stylistic glyphs are aliased: `ۨ`→`ن`, `ۧ`→`ۦ`, `ۭ`→`ۢ`. Slot 32 (`هـ`, a two-codepoint key) is never set, because text is read one codepoint at a time.
- **Haraka (one-hot, or all-zero, 13):** `fatha`, `fatha_shadda`, `kasra`, `kasra_shadda`, `damma`, `damma_shadda`, `sukun`, `sukun_zero` (`۠`), `fathatan`, `kasratan`, `dammatan`, `madd` (explicit maddah `ٓ` or `آ`), `sukun_shadda`. Letters without a written vowel (bare madd letters, silent letters, `ٱ`, assimilated lām) have an all-zero haraka slice. The rounded zero `۟` folds into `sukun`. In the corpus, `sukun_shadda` only arises in stop variants.
- **Pause (3-bit code, least-significant bit first):** pause glyphs set the code of the letter they follow and never get a vector of their own.

  | Code | Name | CLI | Set by |
  |---|---|---|---|
  | 0 | `do_not_stop` | `-` | letters inside a word; `ۙ` |
  | 1 | `word_boundary_emergency` | `!` | word end without a pause mark |
  | 2 | `seli` | `↦` | `ۖ` |
  | 3 | `jaiz` | `≈` | `ۚ` |
  | 4 | `taanoq` | `⋀` | `ۛ` |
  | 5 | `qeli_or_ayah_end` | `⏹` | `ۗ`; last letter of each embedded āyah |
  | 6 | `sakta` | `˽` | `ۜ` |
  | 7 | `lazem` | `⛔` | `ۘ` |

- **Ṣifāt (6 bits):**
  - bit 0: jahr (1) / hams (0)
  - bits 1–2: rikhwah (both 0) / tawassuṭ (bit 1) / shiddah (bit 2)
  - bit 3: istiʿlāʾ (1) / istifāl (0). This bit is contextual for rāʾ (tafkhīm/tarqīq from its own vowel, or the preceding one when sākin) and for the lām of the Divine Name (heavy after fatḥa/ḍamma, light after kasra).
  - bit 4: infitāḥ (1) / iṭbāq (0)
  - bit 5: idhlāq (1) / iṣmāt (0)
- **Rules (multi-hot, 23)**, in alphabetical order (`emb.rule_names`): `ghunnah`, `ghunnah_tafkheem`, `hamzat_wasl`, `idghaam_ghunnah`, `idghaam_mutajanisayn`, `idghaam_mutaqaribayn`, `idghaam_no_ghunnah`, `idghaam_shafawi`, `ikhfa`, `ikhfa_shafawi`, `imala`, `iqlab`, `ishmam`, `lam_shamsiyyah`, `madd_2`, `madd_246`, `madd_6`, `madd_munfasil`, `madd_muttasil`, `optional_seen`, `qalqalah`, `silent`, `tas_heel`.
  - 19 are span rules from `tajweed.rules.json`. `iqlab` is also set by the `ۢ` marker. The other 4 come only from inline markers: `tas_heel` (`۬`), `imala` (`۪`), `ishmam` (`۫`, applied to the following letter) and `optional_seen` (`ۣ`).
  - `madd_246` (madd ʿāriḍ) is waqf-only: it is set in stop variants, never in primary (waṣl) vectors.

## Install

Requires Python ≥ 3.10. The only runtime dependency is `numpy`. `requests`/`tqdm` are used for data regeneration when installed, but are optional.

```bash
pip install tajweed-embeddings
# or for development/testing
python3 -m pip install -e ".[test]"
# or install from GitHub
python3 -m pip install "git+https://github.com/tarekeldeeb/tajweed-embeddings.git"
```

## Quickstart (Python)

```python
from tajweed_embeddings import TajweedEmbedder

emb = TajweedEmbedder()          # first run downloads/generates data (see "Data + Regeneration")

vecs = emb.text_to_embedding(1, 1)              # sūrah 1, āyah 1 → TajweedEmbedding
whole = emb.text_to_embedding(112)              # a whole sūrah
span = emb.text_to_embedding(2, 1, count=3)     # āyāt 2:1–2:3
sub = emb.text_to_embedding(1, 1, "بِسْمِ")     # located inside 1:1, keeps its rule flags
free = emb.text_to_embedding(1, None, "بِسْمِ") # free text: no rule spans

print(emb.embedding_dim)                        # 86
print(emb.encoding_to_string(sub, style="short"))

round_trip = emb.embedding_to_text(sub)         # "بِسْمِ"
score = emb.score(sub, emb.text_to_embedding(1, 1, "بَسْمِ"))   # 93.75
```

Notes:
- `TajweedEmbedding` is a `list` of `np.ndarray` vectors (shape `(86,)`), so it works anywhere a list does.
- With both `ayah` and `subtext`, the subtext is matched inside that āyah, ignoring diacritics and the `ٱ`/`ا` difference. A match inherits the āyah's rule flags and cross-word context. With no match, or with `ayah=None`, rule spans are skipped (inline marker rules still apply). `count` is ignored when `subtext` is given.
- Diacritics and pause marks attach to the previous letter and never add vectors.
- `count` embeds consecutive āyāt of the same sūrah starting at `ayah`. Embedding by āyah marks each āyah's last letter with pause code 5. Subtext embeddings don't get this mark.
- Primary vectors encode continuous recitation (waṣl) everywhere, āyah ends included. For example, the ي of الرَّحِيمِ is `madd_2`, and idghām/ikhfāʾ/iqlāb/munfaṣil carry across āyah boundaries. Stop (waqf) and start (ibtidāʾ) realizations, such as madd ʿāriḍ (`madd_246`), qalqalah kubrā and iẓhār, live in `stop_variants` / `start_variants`. Use `vecs.realize(stop_words=[...])` to materialize a recitation with stops.
- `encoding_to_string(style="long")` produces labeled, multi-field output, while `"short"` is tabular. Pass `secondary=True` to interleave the variant rows.
- `embedding_to_text` decodes letters and harakāt only. Pause glyphs, inline markers and maddah are not re-emitted, `۟` comes back as `ْ`, and a space is inserted only after pause codes 1 and 5. A word ending in an explicit pause mark is therefore joined to the next word.
- `compare` is the cosine similarity of the flattened sequences, truncated to the shorter one. `score` is that value scaled to 0–100.

### API at a glance

| `TajweedEmbedder` | |
|---|---|
| `text_to_embedding(sura, ayah=None, subtext=None, count=1)` | embed a sūrah, āyah(s) or text → `TajweedEmbedding` |
| `get_text(sura, ayah=None, subtext=None)` | normalized āyah / sūrah text |
| `encoding_to_string(encoding, style="short", secondary=False)` | human-readable view of one vector or a sequence |
| `embedding_to_text(vectors)` | approximate text reconstruction |
| `compare(e1, e2)` / `score(e1, e2)` | cosine similarity / 0–100 score |

| `TajweedEmbedding` | |
|---|---|
| `words` | `[(first_idx, last_idx), ...]`, one inclusive range per word |
| `stop_variants` / `start_variants` | `{idx: vector}` replacements for waqf / ibtidāʾ |
| `realize(stop_words=(), start_words=None)` | plain vector list for one concrete recitation |
| `word_index_of(idx)` | word number containing a vector index |
| `vectors` / `extend_segment(other)` | plain-list copy / append another segment, offsetting its indices |

## CLI

Inspect embeddings without writing code:

```bash
tajweed-embeddings --sura 1 --aya 1 --style short
tajweed-embeddings --sura 2 --aya 1 --count 3 --style long
tajweed-embeddings --sura 1 --aya 1 --secondary   # also show waqf/ibtidāʾ variant rows
tajweed-embeddings --sura 112 -q                  # whole sūrah, no banner
tajweed-embeddings --sura 1 --aya 1 --subtext "ٱلرَّحْمَٰنِ"
```

| Flag | Meaning |
|---|---|
| `--sura N` | Sūrah number (required). |
| `--aya N` | Āyah number. Omit it to embed the whole sūrah. |
| `--count N` | Number of consecutive āyāt starting at `--aya` (default 1). |
| `--style short\|long` | Tabular (default) or labeled output. |
| `--subtext TEXT` | Embed this text instead. With `--aya`, it is located inside that āyah to keep its rule flags. |
| `-s`, `--secondary` | Also print the `⏹` stop / `▶` start variant rows. |
| `-q`, `--quiet` | Skip the banner. |

Outputs a human-readable view of the vectors (for inspection; not the raw numeric arrays). Each row is `[index] letter | haraka | pause | ṣifāt | rules`:

- **Haraka:** `^` fatḥa, `‿` kasra, `و` ḍamma, `°` sukūn, `0` sukūn (`۠`), `^^` / `__` / `oo` tanwīn fatḥ / kasr / ḍamm, `~` madd, and `ώ` / `ῳ` / `ὠ` / `ῶ` for fatḥa / kasra / ḍamma / sukūn + shadda. Blank means no written vowel.
- **Pause:** see the pause table above (`-` inside a word, `!` plain word end, `⏹` qeli / āyah end, …).
- **Ṣifāt:** `🔊`/`🤫` jahr/hams, `💨`/`➖`/`🚫` rikhwah/tawassuṭ/shiddah, `🔼`/`🔻` istiʿlāʾ/istifāl, `⟂`/`▲` infitāḥ/iṭbāq, `😮`/`😐` idhlāq/iṣmāt.
- Rows for letters flagged `silent`, `hamzat_wasl` or `lam_shamsiyyah` are dimmed in the terminal.

Example output (full, collapsible):

<details>
<summary>tajweed-embeddings --sura 2 --aya 1 --count 3</summary>

```text
% tajweed-embeddings --sura 2 --aya 1 --count 3
╔══════════════════════════════════════════════════════════╗
║ Tajweed Embeddings CLI                                   ║
║   For inspection only — use programmatically for models. ║
║   String output is a human view, NOT the numeric vectors.║
║                                                          ║
║ ┌ Index: row number                                      ║
║ │  ┌ Letter: glyph                                       ║
║ │  │   ┌ Tashkeel: Kasra ‿ , Fatha ^ , .. etc            ║
║ │  │   │    ┌ Pause: stop mark (0/4/6 etc)               ║
║ │  │   │    │   ┌ Jahr 🔊 , Hams 🤫                      ║
║ │  │   │    │   │  ┌ Rikhw 💨 , Tawasot ➖ , Shidda 🚫   ║
║ │  │   │    │   │  │  ┌ Isti'la 🔼 , Istifal 🔻          ║
║ │  │   │    │   │  │  │  ┌ Infitah ⟂ , Itbaq ▲           ║
║ │  │   │    │   │  │  │  │  ┌ Idhlaq 😮 , Ismat 😐       ║
║ │  │   │    │   │  │  │  │  │    ┌ Rules: Tajweed flags  ║
║ │  │   │    │   │  │  │  │  │    │                       ║
╚═╪══╪═══╪════╪═══╪══╪══╪══╪══╪════╪═══════════════════════╝
[ 0] ب | ‿  | - | 🔊 🚫 🔻 ⟂ 😮 |                           
[ 1] س | °  | - | 🤫 💨 🔻 ⟂ 😐 |                           
[ 2] م | ‿  | ! | 🔊 ➖ 🔻 ⟂ 😮 |                           
[ 3] ٱ |    | - | 🔊 💨 🔻 ⟂ 😐 | hamzat_wasl               
[ 4] ل |    | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[ 5] ل | ώ  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[ 6] ه | ‿  | ! | 🤫 💨 🔻 ⟂ 😐 |                           
[ 7] ٱ |    | - | 🔊 💨 🔻 ⟂ 😐 | hamzat_wasl               
[ 8] ل |    | - | 🔊 ➖ 🔻 ⟂ 😮 | lam_shamsiyyah            
[ 9] ر | ώ  | - | 🔊 ➖ 🔼 ⟂ 😮 |                           
[10] ح | °  | - | 🤫 💨 🔻 ⟂ 😐 |                           
[11] م | ^  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[12] ـٰ |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[13] ن | ‿  | ! | 🔊 ➖ 🔻 ⟂ 😮 |                           
[14] ٱ |    | - | 🔊 💨 🔻 ⟂ 😐 | hamzat_wasl               
[15] ل |    | - | 🔊 ➖ 🔻 ⟂ 😮 | lam_shamsiyyah            
[16] ر | ώ  | - | 🔊 ➖ 🔼 ⟂ 😮 |                           
[17] ح | ‿  | - | 🤫 💨 🔻 ⟂ 😐 |                           
[18] ي |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
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
[43] ى |    | ! | 🔊 💨 🔻 ⟂ 😐 | idghaam_no_ghunnah, silent
[44] ل | ῳ  | - | 🔊 ➖ 🔻 ⟂ 😮 | idghaam_no_ghunnah        
[45] ل | °  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[46] م | و  | - | 🔊 ➖ 🔻 ⟂ 😮 |                           
[47] ت | ώ  | - | 🤫 🚫 🔻 ⟂ 😐 |                           
[48] ق | ‿  | - | 🔊 🚫 🔼 ⟂ 😐 |                           
[49] ي |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
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
[97] و |    | - | 🔊 💨 🔻 ⟂ 😐 | madd_2                    
[98] ن | ^  | ⏹ | 🔊 ➖ 🔻 ⟂ 😮 |                           
```
</details>

### Secondary (waqf / ibtidāʾ) rules

Each row is the waṣl (continuous) pronunciation. Every word also gets pre-computed variants for
stopping on it and for starting at it. `--secondary` (or `encoding_to_string(..., secondary=True)`)
prints them beneath the phoneme they replace. Only phonemes that actually change get a variant row.

- `[⏹]` **stop (waqf)** variant, used when the reciter stops on that word:
  - sukūn ʿāriḍ on the final letter (shadda + vowel → `sukun_shadda`);
  - madd ʿāriḍ lil-sukūn: `madd_2` → `madd_246` on the madd (or līn) letter before the stopped final;
  - madd al-ʿiwaḍ: a final fatḥatān becomes fatḥa and its seat ا/ى is voiced as `madd_2`. With no
    seat (سَمَآءً), the final letter itself carries the madd;
  - tāʾ marbūṭa → hāʾ sākinah;
  - qalqalah kubrā on a final ق ط ب ج د;
  - hāʾ ṣilah: the small ۥ/ۦ after a pronoun hāʾ is silenced and the hāʾ takes sukūn;
  - a final ا/ى silenced only by the next word's hamzat waṣl is voiced again (`madd_2`);
  - ikhfāʾ/idghām/iqlāb that continue into the next word (or āyah) are undone (iẓhār), and a
    detached `madd_munfasil` reverts to `madd_2`;
  - rāʾ/lām tafkhīm is re-evaluated for the stopped ending.
- `[▶]` **start (ibtidāʾ)** variant, used when the reciter starts at that word:
  - hamzat waṣl `ٱ` is voiced as `أ` with fatḥa (definite article), ḍamma (permanent ḍamma on the
    third letter) or kasra, including the classical exceptions. A following sākin hamza becomes the
    matching madd letter (ibdāl: ٱئْتُوا → إِيتُوا);
  - rule flags from spans that began in the previous word are cleared, and a written idghām shadda
    on the first letter is dropped (هُدًى لِّلْمُتَّقِينَ → لِلْمُتَّقِينَ).

Not modeled: the rawm and ishmām stop styles. Stop variants assume the default sukūn ʿāriḍ.

<details>
<summary>tajweed-embeddings --sura 1 --aya 1 --secondary</summary>

```text
% tajweed-embeddings --sura 1 --aya 1 --secondary
╔══════════════════════════════════════════════════════════╗
║ Tajweed Embeddings CLI                                   ║
║   For inspection only — use programmatically for models. ║
║   String output is a human view, NOT the numeric vectors.║
║                                                          ║
║ ┌ Index: row number                                      ║
║ │  ┌ Letter: glyph                                       ║
║ │  │   ┌ Tashkeel: Kasra ‿ , Fatha ^ , .. etc            ║
║ │  │   │    ┌ Pause: stop mark (0/4/6 etc)               ║
║ │  │   │    │   ┌ Jahr 🔊 , Hams 🤫                      ║
║ │  │   │    │   │  ┌ Rikhw 💨 , Tawasot ➖ , Shidda 🚫   ║
║ │  │   │    │   │  │  ┌ Isti'la 🔼 , Istifal 🔻          ║
║ │  │   │    │   │  │  │  ┌ Infitah ⟂ , Itbaq ▲           ║
║ │  │   │    │   │  │  │  │  ┌ Idhlaq 😮 , Ismat 😐       ║
║ │  │   │    │   │  │  │  │  │    ┌ Rules: Tajweed flags  ║
║ │  │   │    │   │  │  │  │  │    │                       ║
╚═╪══╪═══╪════╪═══╪══╪══╪══╪══╪════╪═══════════════════════╝
[ 0] ب | ‿ | - | 🔊 🚫 🔻 ⟂ 😮 |               
[ 1] س | ° | - | 🤫 💨 🔻 ⟂ 😐 |               
[ 2] م | ‿ | ! | 🔊 ➖ 🔻 ⟂ 😮 |               
[ ⏹] م | ° | ! | 🔊 ➖ 🔻 ⟂ 😮 |               
[ 3] ٱ |   | - | 🔊 💨 🔻 ⟂ 😐 | hamzat_wasl   
[ ▶] أ | ^ | - | 🤫 🚫 🔻 ⟂ 😐 |               
[ 4] ل |   | - | 🔊 ➖ 🔻 ⟂ 😮 |               
[ 5] ل | ώ | - | 🔊 ➖ 🔻 ⟂ 😮 |               
[ 6] ه | ‿ | ! | 🤫 💨 🔻 ⟂ 😐 |               
[ ⏹] ه | ° | ! | 🤫 💨 🔻 ⟂ 😐 |               
[ 7] ٱ |   | - | 🔊 💨 🔻 ⟂ 😐 | hamzat_wasl   
[ ▶] أ | ^ | - | 🤫 🚫 🔻 ⟂ 😐 |               
[ 8] ل |   | - | 🔊 ➖ 🔻 ⟂ 😮 | lam_shamsiyyah
[ 9] ر | ώ | - | 🔊 ➖ 🔼 ⟂ 😮 |               
[10] ح | ° | - | 🤫 💨 🔻 ⟂ 😐 |               
[11] م | ^ | - | 🔊 ➖ 🔻 ⟂ 😮 |               
[12] ـٰ |   | - | 🔊 💨 🔻 ⟂ 😐 | madd_2        
[ ⏹] ـٰ |   | - | 🔊 💨 🔻 ⟂ 😐 | madd_246      
[13] ن | ‿ | ! | 🔊 ➖ 🔻 ⟂ 😮 |               
[ ⏹] ن | ° | ! | 🔊 ➖ 🔻 ⟂ 😮 |               
[14] ٱ |   | - | 🔊 💨 🔻 ⟂ 😐 | hamzat_wasl   
[ ▶] أ | ^ | - | 🤫 🚫 🔻 ⟂ 😐 |               
[15] ل |   | - | 🔊 ➖ 🔻 ⟂ 😮 | lam_shamsiyyah
[16] ر | ώ | - | 🔊 ➖ 🔼 ⟂ 😮 |               
[17] ح | ‿ | - | 🤫 💨 🔻 ⟂ 😐 |               
[18] ي |   | - | 🔊 💨 🔻 ⟂ 😐 | madd_2        
[ ⏹] ي |   | - | 🔊 💨 🔻 ⟂ 😐 | madd_246      
[19] م | ‿ | ⏹ | 🔊 ➖ 🔻 ⟂ 😮 |               
[ ⏹] م | ° | ⏹ | 🔊 ➖ 🔻 ⟂ 😮 |               
```

</details>

The same variants are available programmatically:

```python
vecs = emb.text_to_embedding(1, 1)
print(vecs.words)                       # [(0, 2), (3, 6), (7, 13), (14, 19)]
print(sorted(vecs.stop_variants))       # [2, 6, 12, 13, 18, 19]
print(sorted(vecs.start_variants))      # [3, 7, 14]

# Stop on ٱلرَّحْمَٰنِ (word 2), then resume at ٱلرَّحِيمِ (word 3, implied):
recited = vecs.realize(stop_words=[2])
print(emb.encoding_to_string(recited))

# Begin mid-āyah at ٱللَّهِ (word 1) without a preceding stop:
started = vecs.realize(start_words=[1])
print(vecs.word_index_of(10))           # 2: vector 10 belongs to word 2
```

## Data + Regeneration

| File | Origin |
|---|---|
| `data/sifat.json` | Packaged. Per-letter ṣifāt; its keys also define the letter slice. |
| `rules_gen/rule_trees/*.json` | Packaged. Start/end decision trees, one pair per span rule. |
| `rules_gen/output/quran-uthmani.txt` | Downloaded from Tanzil (Uthmānī text with marks). |
| `data/quran.json` | Generated from the Tanzil text (`util/tanzil_to_json.py`). |
| `data/tajweed.rules.json`, `data/tajweed.rules.version` | Generated by `rules_gen/tajweed_classifier.py`. |

`TajweedEmbedder()` prepares its data on construction. It regenerates `tajweed.rules.json` when the file is missing or invalid, when its `.version` stamp differs from `TajweedEmbedder.RULES_DATA_VERSION` (currently `wasl-2`), or when the cached Tanzil text is missing (the text is downloaded first). It rebuilds `quran.json` from the cached text when that file is missing or invalid. Upgrading to a release with new rule logic therefore rebuilds stale spans automatically. To force a rebuild, delete `data/tajweed.rules.json` or its `.version` stamp.

- All of these files live inside the installed package, so that directory must be writable. Network access is only needed when the Tanzil text has to be downloaded (normally the first run). The classifier then runs over all āyāt, using every CPU core.
- `requests` and `tqdm` are used when installed. Without them, the download falls back to `urllib` and the classifier runs without a progress bar.
- Spans are labeled in continuous recitation. Each āyah is classified with the next āyah appended (across sūrah boundaries, and the last āyah wraps to the first), and the result is clipped to the āyah. A span that crosses into the next āyah is marked `"continues": true`, and its carried-over tail in the next āyah is marked `"continued": true`. The embedder uses these flags to build cross-āyah stop/start variants.
- After classification, spans are post-processed: madd is removed from muqaṭṭaʿāt, munfaṣil → muttaṣil when the hamza is in the same word, competing madd rules on the same letters are resolved by priority, tanwīn-fatḥ seats are marked silent, and ikhfāʾ / ghunnah_tafkheem / ikhfāʾ shafawī are filtered by their trigger letter.
- Schema: `[{"surah": 2, "ayah": 8, "annotations": [{"rule": "idghaam_ghunnah", "start": 0, "end": 2, "continued": true}, ...]}, ...]`. `start`/`end` are codepoint offsets into the āyah text.
- As in the Tanzil text, āyah 1 of every sūrah except 9 begins with the basmala, so the basmala's vectors are part of that āyah.
- Corpus coverage: 114 sūrahs / 6236 āyāt.

## Tests

```bash
python3 -m pip install -e ".[test]"
pytest -q
```

The suite runs against the generated data, so the first run downloads and generates it. CI (GitHub Actions) runs Pylint (`--fail-under=8.5`) and pytest on Python 3.10–3.12.

## Project Layout

```
src/tajweed_embeddings/
  embedder/    TajweedEmbedder, TajweedEmbedding, the CLI, and the letter/haraka/ṣifāt/rule/secondary helpers
  rules_gen/   tajweed_classifier.py and rule_trees/ (rule-span generator)
  util/        Tanzil download, text normalization, Tanzil → quran.json
  data/        sifat.json (+ generated quran.json, tajweed.rules.json)
scripts/ctc_train_sketch.py   CTC training sketch (Whisper encoder + CTC head; needs torch, transformers, torchaudio)
test/                         pytest suite
ROADMAP.md                    planned work
```

## License

Dual-licensed: Waqf Public License 2.0 for non-commercial use; commercial or other uses require permission. See `LICENSE`.
