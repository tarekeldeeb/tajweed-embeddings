"""Long and stress-test cases for embeddings."""

import pytest
import numpy as np

# Special Quran symbols to ensure they do not break the encoder
SPECIAL_SYMBOLS = [
    "۞", "۩", "۝", "ۜ", "۟", "۠", "ۢ", "ۣ",
    "\u06dd",  # decorative symbol
    "\ufd3f", "\ufd3e",  # ornate parentheses
    "\ufb50", "\ufb51", "\ufb52", "\ufb53"  # Arabic ligatures
]


def count_letters(emb, text):
    """Utility: count only letters (not diacritics, not spaces, not symbols)."""
    normalized = emb._normalize_text(text)  # type: ignore[attr-defined]
    normalized = normalized.replace("آ", "آ")
    count = 0
    for ch in normalized:
        norm_ch = emb.char_aliases.get(ch, ch) if hasattr(emb, "char_aliases") else ch
        if norm_ch == "آ":
            norm_ch = "ا"
        if norm_ch in emb.letters:
            count += 1
    return count


# -------------------------------------------------------------------------
# 1. Long multi-word sequences
# -------------------------------------------------------------------------

def test_long_multiword_sequence(emb):
    """Embedding of a long multi-word phrase inside one ayah."""
    text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    out = emb.text_to_embedding(1, 1, text)

    expected = count_letters(emb, text)

    assert isinstance(out, list)
    assert len(out) == expected


# -------------------------------------------------------------------------
# 2. Long sequences with heavy diacritics
# -------------------------------------------------------------------------

def test_long_sequence_with_diacritics(emb):
    txt = "قَدْ جَآءَكُمْ رَسُولٌ مِّنَ ٱللَّهِ"
    out = emb.text_to_embedding(9, 128, txt)  # random ayah ref

    expected = count_letters(emb, txt)
    assert len(out) == expected


# -------------------------------------------------------------------------
# 3. Cross-ayah concatenation
# -------------------------------------------------------------------------

def test_cross_ayah_concatenation(emb):
    """Full surah 1 concatenation — ensures sequence continuity."""
    sura_text = " ".join(emb.quran["1"][str(i)] for i in range(1, 8))
    out = emb.text_to_embedding(1)

    expected = count_letters(emb, sura_text)

    assert len(out) == expected


def test_explicit_cross_ayah_joining(emb):
    """Manually join ayat 1 and 2 and ensure encoder handles boundary cleanly."""
    q = emb.quran["1"]
    text = q["1"] + " " + q["2"]

    out = emb.text_to_embedding(1, subtext=text)

    expected = count_letters(emb, text)
    assert len(out) == expected


# -------------------------------------------------------------------------
# 4. Special Quran symbols handling
# -------------------------------------------------------------------------

@pytest.mark.parametrize("sym", SPECIAL_SYMBOLS)
def test_symbols_handling(emb, sym):
    """Special Quran symbols should NOT crash processing or alter result length improperly."""
    txt = "بِسْمِ " + sym + " اللَّهِ"
    out = emb.text_to_embedding(1, 1, txt)

    expected = count_letters(emb, txt)
    assert len(out) == expected


# -------------------------------------------------------------------------
# 5. Multi-ayah long combination tests
# -------------------------------------------------------------------------

def test_very_long_multiverse_sequence(emb):
    """Combine several ayāt from Surah Baqarah — long stress test."""
    q = emb.quran["2"]

    combined = q["1"] + " " + q["2"] + " " + q["3"] + " " + q["4"]
    out = emb.text_to_embedding(2, subtext=combined)

    expected = count_letters(emb, combined)
    assert len(out) == expected


# -------------------------------------------------------------------------
# 6. Random fuzz testing
# -------------------------------------------------------------------------

def test_randomized_arabic_fuzzing(emb):
    """Stress test using arbitrary Arabic letters + diacritics."""
    letters = list(emb.letters)
    harakat = list(emb.harakat.keys())

    import random
    random.seed(42)

    seq = ""
    for _ in range(120):
        seq += random.choice(letters + harakat)

    out = emb.text_to_embedding(1, 1, seq)

    expected = count_letters(emb, seq)
    assert len(out) == expected


def test_all_rules_present_in_embeddings(emb):
    """Ensure every tajwīd rule annotation for an ayah appears in its embeddings."""
    sura = "1"
    ayah = "1"
    annotations = emb.tajweed_rules.rules_index.get((sura, ayah), [])
    if not annotations:
        pytest.skip("No rules available for sura 1 ayah 1 in current data")

    vecs = emb.text_to_embedding(int(sura), int(ayah))
    n_rules = emb.n_rules
    active = set()
    for v in vecs:
        rules_slice = v[emb.idx_rule_start : emb.idx_rule_start + n_rules]
        active.update(i for i, val in enumerate(rules_slice) if val > 0)

    expected = set(emb.rule_to_index[a["rule"]] for a in annotations if a.get("rule") in emb.rule_to_index)
    assert expected.issubset(active)
"""Stress tests for long sequences and repeated embeddings."""
