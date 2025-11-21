# Tajweed Embeddings TODO

1. **Support Qur'ān-specific glyph variants**  
   ✅ Done. `sifat.json` / `TajweedEmbedder.letters` now include Qur’ān glyphs (hamzat‑waṣl, dagger alif, maddah forms, small waw/ya, etc.) and tajwīd markers. Aliases added for glyph variants.

2. **Realign tajwīd rule spans for normalized text**  
   🔄 Still pending. `_apply_rule_spans` indexes raw text; offsets can drift when glyphs are skipped/aliased. Normalize/align spans to the filtered sequence.

3. **Preserve multiple harakāt (shadda + vowel)**  
   ✅ Done. Haraka slice expanded; shadda+vowel combos are explicit states; tanwīn and alternate sukūn handled; pause slice added.

4. **Represent shadda explicitly in embeddings**  
   ✅ Done via expanded haraka states that keep shadda; decoding reflects combined states.
