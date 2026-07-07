# ROADMAP — tajweed-embeddings

Post-v2.0.0 plan (secondary waqf/ibtidāʾ phonemes shipped). Priorities: **P0** = next release,
**P1** = mid-term, **P2** = long-term. Effort: **S** < 1 day, **M** = days, **L** = weeks+.

Guiding rule: the embeddings are the product — anything that can silently corrupt
letter/rule alignment outranks new features.

---

## Near-term (v2.1.x)

### P0 — Packaging & data integrity (quick win, S–M)
- `quran.json` (1.4 MB) and `tajweed.rules.json` (8.9 MB) are gitignored and excluded from
  `[tool.setuptools.package-data]`; every fresh install triggers
  `TajweedEmbedder._ensure_data_ready()` → tanzil.net download + `subprocess.run()` of
  `rules_gen/tajweed_classifier.py`, writing into the installed package dir (breaks on
  read-only site-packages, offline CI, and is a supply-chain risk).
- Fix: commit the generated JSONs (or ship as wheel data), record sha256 in a
  `data/MANIFEST.json`, verify in `_load_json()`. Remove network + subprocess from the
  constructor entirely; keep regeneration as an explicit command
  (`python -m tajweed_embeddings.rules_gen`) that imports the classifier instead of
  shelling out. Add `TajweedEmbedder(data_dir=...)` for overrides (also unlocks riwāyah packs).

### P0 — Rule-span validation vs. authoritative sources (deep work, M)
- Fold in TODO.md's known doubts (Sūrah 2 first ~50 indices): clustered `madd_6` at ~20–24,
  inconsistent `silent`/`madd_2` on alif before hamzat waṣl (~20, 81), offset
  `hamzat_wasl`/`lam_shamsiyyah` rows (~29–30, 77–79), letter-count vs embedding-length drift.
- Prime suspect: the `orig_to_norm` remap for decomposed آ in
  `TajweedRulesEmbedder.apply_rule_spans_ex()` — unit-test it in isolation on every āyah
  containing آ.
- Add a corpus-wide **invariant audit** (CI job): `hamzat_wasl` only on ٱ, `qalqalah` only on
  ق ط ب ج د, madd flags only on madd carriers, `ghunnah` only on ن/م, span indices within
  āyah length. Cross-check span counts against upstream cpfair/quran-tajweed and QUL/Tarteel
  tajweed layers where available.

### P0 — Golden-corpus regression snapshots (quick win, S)
- Snapshot a compact fingerprint (per-āyah hash of each slice: letters/haraka/pause/sifat/rules,
  plus `stop_variants`/`start_variants` counts) for all 6236 āyāt into `test/golden/`.
  CI fails on any unreviewed diff. This converts every future refactor of
  `_embed_segment()` / `SecondaryPhonemeEmbedder.compute()` from "hope" to "diff review".

### P0 — Property-based tests & subtext fuzzing (quick win, S–M)
- `hypothesis` strategies over the Arabic block + random Unicode into
  `text_to_embedding(1, 1, subtext)`: must never raise, vector count == kept-letter count,
  letter slice is one-hot, haraka slice ≤ one-hot, pause bits ∈ {0,1}.
- Properties for `TajweedEmbedding.realize()` (idempotent, length-preserving, only indices in
  variant maps change) and `extend_segment()` (offset correctness).
- Round-trip guarantee: define and CI-enforce
  `strip_marks(normalize(text)) == embedding_to_text(text_to_embedding(sura, ayah))`
  corpus-wide; `embedding_to_text()` is currently "rough" — make its contract explicit.

### P0 — CI & typing hygiene (quick win, S)
- `.github/workflows/tests.yml`: add Python 3.13, one macOS + Windows leg, `pip install -e .[test]`
  (not requirements.txt), cache pip; fix the lint job's `MIN_SCORE` echo (says 7.0, enforces 8.5).
- Add mypy (or pyright) step — strict on the new v2 modules first (`embedding.py`,
  `secondary_embedder.py`), permissive elsewhere; ratchet up.
- Semver discipline: `CHANGELOG.md`; expose `LAYOUT_VERSION` alongside `embedding_dim`
  (any change to slice order/width = major bump); tag releases; publish wheels from CI.

### P1 — Scoring API honesty (S)
- `TajweedEmbedder.compare()` silently truncates to min length — a 10-vector vs 100-vector
  comparison can score high. Add `strict=` (raise on mismatch) and document; DTW-based
  alignment scoring later.

---

## Mid-term (v2.2 → v3)

### P1 — Rawm & ishmām stop variants (deep work, M–L)
- Extend `SecondaryPhonemeEmbedder` (currently documents these as out of scope) with a
  stop-style axis: `sukun` (today's default) | `rawm` (kasra/damma-final only, vowel hinted,
  **no** madd ʿāriḍ extension — madd stays 2) | `ishmam` (damma-final only, lip gesture,
  reuses the existing `ishmam` rule flag).
- API decision to make **before** consumers freeze on the v2 shape: either
  `stop_variants: Dict[int, Dict[str, np.ndarray]]` keyed by style (breaking → v3.0) or
  additive `stop_variants_rawm` / `stop_variants_ishmam` maps (non-breaking). Prefer additive.
- `realize(stop_words=..., stop_style=...)` selects per-stop style.

### P1 — Data-driven rulings (M, prerequisite for riwāyāt)
- Move hardcoded class constants out of `secondary_embedder.py` into a packaged
  `data/rulings.json` (with source citations per entry): `WASL_KASRA_NOUN_PREFIXES`,
  `WASL_DAMMA_EXCEPTIONS`, `WASL_LAM_VERB_PREFIXES`, `NASAL_ASSIMILATION_RULES`,
  `QALQALAH_LETTERS`, plus the silent-alif-before-hamzat-waṣl fallback heuristic at the
  bottom of `apply_rule_spans_ex()`. Schema supports per-riwāyah override layers.

### P1 — Sakt positions (S–M)
- ۜ already maps to pause category 6 in `TajweedRulesEmbedder.pause_char_category`; verify the
  four mandatory Ḥafṣ sakt (18:1, 36:52, 75:27, 83:14) survive normalization end-to-end
  (tests), and add the optional sakt at the 8→9 sūrah juncture as data, since it has no glyph.

### P1 — Madd length policy per riwāyah/ṭarīq (M)
- Today madd is baked as `madd_2/madd_6/madd_246/muttasil/munfasil` flags. Add a `MaddPolicy`
  mapping rule class → concrete counts at `realize()` time (e.g. Shāṭibiyyah ṭarīq: muttaṣil
  4–5, munfaṣil 4–5), keeping the embedding layout stable.

### P1 — Basmala / istiʿādha junctures (M)
- Build on commit `4224d00` ("Unfilter Basmala rules"): explicit API
  `text_to_embedding(..., isti_adha=True, basmala="wasl"|"waqf"|"omit")` composing the
  permitted juncture forms at sūrah starts (and the 9:1 exception).

### P1 — Benchmarks & docs (S + M)
- pytest-benchmark job: full-corpus embed time, secondary-variant overhead, peak memory
  (`_embed_segment` is pure-Python per-char loops — establish a baseline before optimizing).
- mkdocs-material site: embedding layout spec (the dim-90 table from README), rule glossary
  with Arabic terms, secondary-phoneme semantics, API reference; 1–2 example notebooks.

---

## Long-term (v3+)

### P2 — Riwāyāt beyond Ḥafṣ ʿan ʿĀṣim (L, multi-month)
- `TajweedEmbedder(riwayah="hafs")` with pluggable data packs: per-riwāyah muṣḥaf text
  (Warsh/Qālūn from Tanzil or QUL), re-derived rule trees (`rules_gen/rule_trees/*.json` are
  Ḥafṣ-specific), rulings overlays (imāla, taqlīl, naql, ṣilat mīm al-jamʿ, madd policy).
  Depends on: data-driven rulings, madd policy, packaging rework. Start with Warsh.

### P2 — Scholarly validation program (L, ongoing)
- Review of generated spans + secondary variants by ijāzah holders; sample-based sign-off per
  juzʾ; publish a validation report and per-rule confidence levels in the data manifest.

### P2 — Training/consumer ecosystem (L)
- Graduate `scripts/ctc_train_sketch.py` into `tajweed_embeddings.train` extra or a separate
  `tajweed-ctc` repo: drop the `sys.path.insert` hack, replace the O(T·S) Python Viterbi in
  `ctc_forced_align()` with `torchaudio.functional.forced_align`, add eval + checkpoints.
- Export reference embeddings: precomputed corpus (primary + variants) as npz/parquet →
  HuggingFace dataset, versioned by `LAYOUT_VERSION`; ONNX/torch export of the lookup +
  `realize()` logic for on-device aligners.

---

## Sequencing summary

| Order | Item | Why first |
|---|---|---|
| 1 | Packaging/data integrity | Every install & CI run depends on it |
| 2 | Span validation + invariant audit | Known TODO.md misalignment doubts poison training labels |
| 3 | Golden snapshots + property tests | Locks correctness before feature work |
| 4 | CI/typing/semver | Cheap now, expensive after v2 adoption |
| 5 | Rawm/ishmām + data-driven rulings | Completes waqf model; unblocks riwāyāt |
