import numpy as np
import pytest

# -------------------------------------------------------------------
# BASIC FUNCTIONALITY TESTS
# -------------------------------------------------------------------

def test_embed_full_ayah(emb):
    """Embed: sura=1, aya=1 (full āyah)."""
    out = emb.text_to_embedding(1, 1)
    assert isinstance(out, list)
    assert len(out) > 0
    for vec in out:
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape[0] == emb.embedding_dim


def test_embed_subtext(emb):
    """Embed fragment inside āyah."""
    out = emb.text_to_embedding(1, 1, "بِسْمِ")
    assert len(out) > 0
    # Ensure rule flags slice properly
    assert all(isinstance(v, np.ndarray) for v in out)


def test_embed_subtext_not_found(emb):
    """If subtext not in āyah → zero rule flags."""
    out = emb.text_to_embedding(1, 1, "ZZZNOTEXISTING")
    assert len(out) > 0
    assert all(vec[emb.idx_rule_start:].sum() == 0 for vec in out)


def test_embed_full_sura(emb):
    """Embedding entire surah (sura=1)."""
    out = emb.text_to_embedding(1)
    assert len(out) > 0
    assert isinstance(out[0], np.ndarray)


def test_reject_invalid_sura(emb):
    """Non-existent sura raises error."""
    with pytest.raises(ValueError):
        emb.text_to_embedding(999)


def test_reject_invalid_ayah(emb):
    """Non-existent aya raises error."""
    with pytest.raises(ValueError):
        emb.text_to_embedding(1, 999)


def test_haraka_detection(emb):
    """Ensure harakah is set correctly for typical words."""
    out = emb.text_to_embedding(1, 1, "بِ")
    assert len(out) == 1
    vec = out[0]

    haraka_slice = vec[emb.idx_haraka_start:emb.idx_haraka_start + emb.n_harakat]
    assert np.argmax(haraka_slice) == 1  # kasra = index 1


def test_vector_length(emb):
    """Embedding dimension must match design."""
    out = emb.text_to_embedding(1, 1, "ب")
    assert out[0].shape[0] == emb.embedding_dim


# -------------------------------------------------------------------
# RECONSTRUCTION TESTS
# -------------------------------------------------------------------

def test_embedding_to_text_basic(emb):
    vecs = emb.text_to_embedding(1, 1, "بِسْ")
    txt = emb.embedding_to_text(vecs)
    assert isinstance(txt, str)
    assert "ب" in txt
    assert "س" in txt
    assert any(h in txt for h in ["َ","ِ","ُ","ْ","ّ"])


def test_embedding_to_text_reversible(emb):
    sample = "بِسْمِ"
    emb_list = emb.text_to_embedding(1, 1, sample)
    reconstructed = emb.embedding_to_text(emb_list)
    # Letters appear in expected order (harakat might differ slightly)
    for ch in ["ب", "س", "م"]:
        assert ch in reconstructed


def test_embedding_to_text_zero_vector(emb):
    zero = np.zeros(emb.embedding_dim)
    txt = emb.embedding_to_text([zero])
    assert isinstance(txt, str)
    assert len(txt) > 0


# -------------------------------------------------------------------
# SIMILARITY + SCORING TESTS
# -------------------------------------------------------------------

def test_compare_identical(emb):
    e1 = emb.text_to_embedding(1, 1, "بِسْ")
    e2 = emb.text_to_embedding(1, 1, "بِسْ")
    sim = emb.compare(e1, e2)
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_compare_different(emb):
    e1 = emb.text_to_embedding(1, 1, "بِ")
    e2 = emb.text_to_embedding(1, 1, "سْ")
    sim = emb.compare(e1, e2)
    assert 0 <= sim < 1


def test_compare_length_mismatch(emb):
    e1 = emb.text_to_embedding(1, 1, "بِسْمِ")
    e2 = emb.text_to_embedding(1, 1, "بِ")
    sim = emb.compare(e1, e2)
    assert isinstance(sim, float)


def test_score_identical(emb):
    e = emb.text_to_embedding(1, 1, "بِسْ")
    s = emb.score(e, e)
    assert s == 100.0


def test_score_scaled(emb):
    e1 = emb.text_to_embedding(1, 1, "بِ")
    e2 = emb.text_to_embedding(1, 1, "سْ")
    s = emb.score(e1, e2)
    assert 0 <= s <= 100


def test_score_length_mismatch(emb):
    e1 = emb.text_to_embedding(1, 1, "بِسْمِ")
    e2 = emb.text_to_embedding(1, 1, "بِ")
    s = emb.score(e1, e2)
    assert isinstance(s, float)


# -------------------------------------------------------------------
# LARGE / CLAUSE TESTS
# -------------------------------------------------------------------

def test_long_subtext(emb):
    txt = "بِسْمِ اللَّهِ " * 3
    e = emb.text_to_embedding(1, 1, txt)
    assert len(e) > 0


def test_full_surah_sequence_long(emb):
    out = emb.text_to_embedding(1)  # Surah Al-Fātiḥa (short but multiple ayat)
    assert len(out) > 10


def test_rule_flags_alignment(emb):
    """Rule spans should align properly for known ayah."""
    out = emb.text_to_embedding(1, 1)
    # rule flag slice must be correct length
    for v in out:
        rules = v[emb.idx_rule_start:]
        assert len(rules) == emb.n_rules
        assert rules.ndim == 1
    