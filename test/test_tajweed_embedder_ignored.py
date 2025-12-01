def test_ignored_symbols_do_not_create_embeddings(emb):
    """Spaces, tatweel, and sajda symbols should be skipped."""
    txt = "ب ـ \u06e9 ت"  # includes space, tatweel, sajda between letters
    out = emb.text_to_embedding(1, 1, txt)
    # Only the two letters should produce embeddings
    assert len(out) == 2
    letters = [emb.index_to_letter[i] for i in (out[0][:emb.n_letters].argmax(), out[1][:emb.n_letters].argmax())]
    assert letters == ["ب", "ت"]


def test_all_ignored_text_returns_zero_vector(emb):
    """If input is only ignored symbols, return fallback zero vector."""
    txt = "   ـــ \u06e9"
    out = emb.text_to_embedding(1, 1, txt)
    assert len(out) == 1
    assert out[0].sum() == 0


"""Tests for symbols that should be ignored or skipped in embeddings."""
