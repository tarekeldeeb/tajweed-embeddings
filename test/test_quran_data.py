"""Basic integrity checks for packaged Quran data."""

import pytest

def test_quran_has_expected_counts(emb):
    """Ensure packaged Quran data includes full corpus."""
    quran = emb.quran
    assert isinstance(quran, dict)
    assert len(quran) == 114
    total_ayat = sum(len(ayat_map) for ayat_map in quran.values())
    assert total_ayat == 6236


def test_sura_numbering_is_contiguous(emb):
    """Sura keys should cover 1..114 without gaps."""
    sura_numbers = sorted(int(k) for k in emb.quran.keys())
    assert sura_numbers == list(range(1, 115))


def test_ayah_numbering_is_contiguous_per_sura(emb):
    """Each sura should have 1..N ayat with non-empty text."""
    for sura_key, ayat_map in emb.quran.items():
        keys = sorted(int(k) for k in ayat_map.keys())
        assert keys == list(range(1, len(keys) + 1))
        assert all(isinstance(v, str) and v.strip() for v in ayat_map.values())


def test_known_basmala_text_matches_reference(emb):
    """Spot-check a canonical ayah value for integrity."""
    expected = "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"
    assert emb.quran["1"]["1"] == expected
