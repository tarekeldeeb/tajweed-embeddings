import pytest


def _pause_category(vec, emb):
    pause_slice = vec[emb.idx_pause_start : emb.idx_pause_start + emb.n_pause]
    return int(pause_slice[0] + (pause_slice[1] * 2) + (pause_slice[2] * 4))


def test_end_of_ayah_has_pause(emb):
    """Last letter of an āyah should carry a mandatory pause."""
    vecs = emb.text_to_embedding(1, 1)
    last = vecs[-1]
    assert _pause_category(last, emb) == 5


def test_non_final_word_letters_have_no_pause(emb):
    """Internal word letters should carry 'do_not_stop' pause code."""
    vecs = emb.text_to_embedding(1, 1, "بِسْمِ")
    for vec in vecs[:-1]:
        assert _pause_category(vec, emb) == 0


def test_word_boundary_defaults_to_emergency_pause(emb):
    """Word-ending letters without explicit marks carry the word-boundary pause."""
    vecs = emb.text_to_embedding(1, 1, "بِسْمِ اللَّهِ")
    categories = [_pause_category(v, emb) for v in vecs]
    # First word "بِسْمِ" ends before the space; ensure we see category 1 somewhere.
    assert 1 in categories


@pytest.mark.parametrize(
    "mark,expected",
    [
        ("ۖ", 2),  # Seli
        ("ۚ", 3),  # Jaiz
        ("ۛ", 4),  # Taanoq
        ("ۗ", 5),  # Qeli
        ("ۜ", 6),  # Sakta
        ("ۘ", 7),  # Lazem
        ("ۙ", 0),  # Do not stop
    ],
)
def test_pause_glyph_categories(emb, mark, expected):
    """Pause glyphs map to the correct 3-bit category."""
    vecs = emb.text_to_embedding(1, 1, f"بِ{mark}")
    assert _pause_category(vecs[-1], emb) == expected


def test_full_surah_end_of_ayah_pauses(emb):
    """Full-sūrah embeddings should carry end-of-ayah pauses on each boundary."""
    vecs = emb.text_to_embedding(1)
    categories = [_pause_category(v, emb) for v in vecs]
    assert categories.count(5) == len(emb.quran["1"])
"""Pause category encoding tests."""
