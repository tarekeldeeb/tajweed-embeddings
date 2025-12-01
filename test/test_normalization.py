"""Normalization utility tests."""
from tajweed_embeddings.util.normalization import normalize_superscript_alef


def test_tatweel_before_dagger_alif_removed():
    assert normalize_superscript_alef("حـٰ") == "حٰ"
    assert normalize_superscript_alef("ـٰ") == "ٰ"


def test_unrelated_tatweel_preserved():
    assert normalize_superscript_alef("بــت") == "بــت"


def test_maddah_is_preserved():
    """Combining maddah should remain after normalization."""
    src = "جَآءَ"
    assert normalize_superscript_alef(src) == src


def test_dagger_alif_is_preserved():
    """Dagger alif should remain after normalization."""
    src = "حٰقّ"
    assert normalize_superscript_alef(src) == src


def test_tatweel_is_stripped():
    """Tatweel should be removed when normalizing."""
    src = "حـٰ"
    assert normalize_superscript_alef(src) == "حٰ"
