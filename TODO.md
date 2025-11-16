# Tajweed Embeddings TODO

1. **Support Qur'ān-specific glyph variants**  
   Extend `sifat.json` / `TajweedEmbedder.letters` to include characters such as hamzat‑waṣl (`ٱ`), dagger alif (`ٰ`), maddah forms, etc. Currently these characters are skipped entirely by `text_to_embedding` (`src/tajweed_embeddings/embedder/tajweed_embedder.py:217-226`), so they never produce vectors.

2. **Realign tajwīd rule spans for normalized text**  
   `_apply_rule_spans` indexes the raw Qur’an text, but the embedding loop drops unsupported glyphs, causing offsets to drift. Rules like `lam_shamsiyyah` end up attached to the wrong letters (`sura 1:1`). Normalize the text in the same way when computing spans, or adjust annotations to match the filtered sequence.

3. **Preserve multiple harakāt (shadda + vowel)**  
   The current haraka handling overwrites the slice each time a diacritic is seen (`src/tajweed_embeddings/embedder/tajweed_embedder.py:217-224`). Letters with shadda plus a vowel lose the shadda information, so their embeddings look identical to plain letters. Track shadda separately or allow multiple active haraka flags.

4. **Represent shadda explicitly in embeddings**  
   Beyond the overwrite bug, there is no dedicated feature to indicate a shadda, so doubled consonants cannot be distinguished. Add a boolean/shadda channel (or reuse the haraka vector with multiple bits) and make `encoding_to_string` aware of it.
