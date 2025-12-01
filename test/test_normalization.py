"""Normalization utility tests."""
from tajweed_embeddings.util.normalization import normalize_superscript_alef


def test_tatweel_before_dagger_alif_removed():
    assert normalize_superscript_alef("حـٰ") == "حٰ"
    assert normalize_superscript_alef("ـٰ") == "ٰ"


def test_unrelated_tatweel_preserved():
    assert normalize_superscript_alef("بــت") == "بــت"
