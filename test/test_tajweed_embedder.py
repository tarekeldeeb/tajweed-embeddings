"""Core tests for TajweedEmbedder embeddings and reconstruction."""

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
    idx = np.argmax(haraka_slice)
    assert emb.index_to_haraka_state[idx] == "kasra"


def test_haraka_shadda_combo(emb):
    """Shadda plus vowel should map to combined haraka state."""
    out = emb.text_to_embedding(1, 1, "بَّ")  # shadda + fatha on ba
    assert len(out) == 1
    vec = out[0]
    haraka_slice = vec[emb.idx_haraka_start:emb.idx_haraka_start + emb.n_harakat]
    idx = np.argmax(haraka_slice)
    assert emb.index_to_haraka_state[idx] == "fatha_shadda"


def test_vector_length(emb):
    """Embedding dimension must match design."""
    out = emb.text_to_embedding(1, 1, "ب")
    assert out[0].shape[0] == emb.embedding_dim


# -------------------------------------------------------------------
# RECONSTRUCTION TESTS
# -------------------------------------------------------------------

def test_embedding_to_text_basic(emb):
    """Round-trip produces readable Arabic with harakat."""
    vecs = emb.text_to_embedding(1, 1, "بِسْ")
    txt = emb.embedding_to_text(vecs)
    assert isinstance(txt, str)
    assert "ب" in txt
    assert "س" in txt
    assert any(h in txt for h in ["َ","ِ","ُ","ْ","ّ"])


def test_embedding_to_text_reversible(emb):
    """Reconstruct preserves base letters even if vowels vary."""
    sample = "بِسْمِ"
    emb_list = emb.text_to_embedding(1, 1, sample)
    reconstructed = emb.embedding_to_text(emb_list)
    # Letters appear in expected order (harakat might differ slightly)
    for ch in ["ب", "س", "م"]:
        assert ch in reconstructed


def test_embedding_to_text_zero_vector(emb):
    """Zero vector still returns a printable string."""
    zero = np.zeros(emb.embedding_dim)
    txt = emb.embedding_to_text([zero])
    assert isinstance(txt, str)
    assert len(txt) > 0


# -------------------------------------------------------------------
# SIMILARITY + SCORING TESTS
# -------------------------------------------------------------------

def test_compare_identical(emb):
    """Identical embeddings have cosine 1.0."""
    e1 = emb.text_to_embedding(1, 1, "بِسْ")
    e2 = emb.text_to_embedding(1, 1, "بِسْ")
    sim = emb.compare(e1, e2)
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_compare_different(emb):
    """Different embeddings have similarity below 1."""
    e1 = emb.text_to_embedding(1, 1, "بِ")
    e2 = emb.text_to_embedding(1, 1, "سْ")
    sim = emb.compare(e1, e2)
    assert 0 <= sim < 1


def test_compare_length_mismatch(emb):
    """Length mismatch still returns a float similarity."""
    e1 = emb.text_to_embedding(1, 1, "بِسْمِ")
    e2 = emb.text_to_embedding(1, 1, "بِ")
    sim = emb.compare(e1, e2)
    assert isinstance(sim, float)


def test_score_identical(emb):
    """Score identical embeddings == 100."""
    e = emb.text_to_embedding(1, 1, "بِسْ")
    s = emb.score(e, e)
    assert s == 100.0


def test_score_scaled(emb):
    """Score different embeddings within [0,100]."""
    e1 = emb.text_to_embedding(1, 1, "بِ")
    e2 = emb.text_to_embedding(1, 1, "سْ")
    s = emb.score(e1, e2)
    assert 0 <= s <= 100


def test_score_length_mismatch(emb):
    """Score still returns float when lengths differ."""
    e1 = emb.text_to_embedding(1, 1, "بِسْمِ")
    e2 = emb.text_to_embedding(1, 1, "بِ")
    s = emb.score(e1, e2)
    assert isinstance(s, float)


# -------------------------------------------------------------------
# LARGE / CLAUSE TESTS
# -------------------------------------------------------------------

def test_long_subtext(emb):
    """Long custom subtext still embeds correctly."""
    txt = "بِسْمِ اللَّهِ " * 3
    e = emb.text_to_embedding(1, 1, txt)
    assert len(e) > 0


def test_full_surah_sequence_long(emb):
    """Full surah embeddings produce many vectors."""
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


def test_full_surah_has_rule_flags(emb):
    """Full-sūrah embedding should include tajwīd rules (not all zeros)."""
    out = emb.text_to_embedding(1)
    assert len(out) > 0
    assert any(vec[emb.idx_rule_start:].sum() > 0 for vec in out)


def test_maddah_above_attaches_to_alif(emb):
    """Decomposed alif+maddah should produce one letter with madd haraka."""
    out = emb.text_to_embedding(1, 1, "آ")
    assert len(out) == 1
    vec = out[0]
    haraka_slice = vec[emb.idx_haraka_start : emb.idx_haraka_start + emb.n_harakat]
    idx = int(haraka_slice.argmax())
    assert emb.index_to_haraka_state.get(idx) == "madd"


# -------------------------------------------------------------------
# COVERAGE / CONSISTENCY TESTS
# -------------------------------------------------------------------

def test_first_mid_last_ayah_embeddings_per_sura(emb):
    """Embed first, middle, last āyah of every sūrah and validate rule spans."""
    for sura_str, ayat_map in emb.quran.items():
        sura = int(sura_str)
        ayah_numbers = sorted(int(k) for k in ayat_map.keys())
        mid_idx = len(ayah_numbers) // 2
        targets = {
            ayah_numbers[0],
            ayah_numbers[mid_idx],
            ayah_numbers[-1],
        }
        for ayah in sorted(targets):
            vecs = emb.text_to_embedding(sura, ayah)
            assert vecs, f"Empty embedding for {sura}:{ayah}"
            assert all(vec.shape[0] == emb.embedding_dim for vec in vecs)
            assert all(vec[emb.idx_rule_start:].shape[0] == emb.n_rules for vec in vecs)
            key = (str(sura), str(ayah))
            anns = emb.tajweed_rules.rules_index.get(key, [])
            has_rule_annotations = any(
                ann.get("rule") in emb.rule_to_index for ann in anns
            )
            if has_rule_annotations:
                assert any(
                    vec[emb.idx_rule_start:].sum() > 0 for vec in vecs
                ), f"Expected rule flags for {sura}:{ayah}"


def test_all_rules_present_in_corpus_embeddings(emb):
    """All tajwīd rules from tajweed.rules.json must appear in embeddings somewhere."""
    assert emb.n_rules == len(emb.rule_names)
    seen_rules = set()
    for sura in sorted(int(k) for k in emb.quran.keys()):
        vecs = emb.text_to_embedding(sura)
        for vec in vecs:
            rules_slice = vec[emb.idx_rule_start:]
            seen_rules.update(np.nonzero(rules_slice > 0)[0].tolist())
        if len(seen_rules) == emb.n_rules:
            break
    missing = sorted(set(range(emb.n_rules)) - seen_rules)
    assert not missing, f"Missing rules: {[emb.rule_names[i] for i in missing]}"
