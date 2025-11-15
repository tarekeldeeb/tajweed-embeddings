import numpy as np
import pytest
import json
import os
from tajweed_embedder import TajweedEmbedder

# -------------------------------------------------------------------
# FIXTURES
# -------------------------------------------------------------------

@pytest.fixture
def sifat_json():
    """Minimal sifat JSON for testing."""
    return {
        "ب": {"sifat": {"jahr":1,"hams":0,"shiddah":1,"tawassut":0,"rikhwah":0,
                        "isti'la":0,"istifal":1,"itbaq":0,"infitah":1,
                        "qalqalah":1,"ghunnah":0,"tafkhim":0}},
        "س": {"sifat": {"jahr":0,"hams":1,"shiddah":0,"tawassut":0,"rikhwah":1,
                        "isti'la":0,"istifal":1,"itbaq":0,"infitah":1,
                        "qalqalah":0,"ghunnah":0,"tafkhim":0}},
        "م": {"sifat": {"jahr":1,"hams":0,"shiddah":0,"tawassut":1,"rikhwah":0,
                        "isti'la":0,"istifal":1,"itbaq":0,"infitah":1,
                        "qalqalah":0,"ghunnah":1,"tafkhim":0}},
        "ل": {"sifat": {"jahr":1,"hams":0,"shiddah":0,"tawassut":1,"rikhwah":0,
                        "isti'la":0,"istifal":1,"itbaq":0,"infitah":1,
                        "qalqalah":0,"ghunnah":0,"tafkhim":0}}
    }


@pytest.fixture
def emb():
    # load real JSON files from your project root
    base = os.path.dirname(os.path.dirname(__file__))

    sifat_path = os.path.join(base, "sifat.json")
    rules_path = os.path.join(base, "tajweed.hafs.uthmani-pause-sajdah.json")

    sifat = json.load(open(sifat_path))
    rules = json.load(open(rules_path))

    return TajweedEmbedder(sifat, rules)


# -------------------------------------------------------------------
# TEST: text_to_embedding
# -------------------------------------------------------------------

def test_text_to_embedding_basic(emb):
    out = emb.text_to_embedding("بِسْمِ", "1", "1")
    assert len(out) == 3
    for vec in out:
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1


def test_text_to_embedding_unknown_letter(emb):
    out = emb.text_to_embedding("بِXسْ", "1", "1")
    assert len(out) == 2  # skip X


def test_text_to_embedding_with_spaces(emb):
    out = emb.text_to_embedding("بِ سْ مِ", "1", "1")
    assert len(out) == 3


def test_text_to_embedding_haraka_association(emb):
    out = emb.text_to_embedding("بِ", "1", "1")
    vec = out[0]
    haraka_slice = vec[len(emb.letters):len(emb.letters)+5]
    assert np.argmax(haraka_slice) == 1  # kasrah


def test_text_to_embedding_vector_length(emb):
    out = emb.text_to_embedding("ب", "1", "1")
    vec = out[0]
    expected_len = len(emb.letters) + 5 + 12 + emb.n_rules
    assert vec.shape[0] == expected_len


# -------------------------------------------------------------------
# TEST: embedding_to_text
# -------------------------------------------------------------------

def test_embedding_to_text_basic(emb):
    emb_list = emb.text_to_embedding("بِسْ", "1", "1")
    txt = emb.embedding_to_text(emb_list)
    assert "ب" in txt
    assert "س" in txt
    assert any(h in txt for h in ["َ","ِ","ُ","ْ","ّ"])


def test_embedding_to_text_reversible(emb):
    original = "بِسْمِ"
    emb_list = emb.text_to_embedding(original, "1", "1")
    reconstructed = emb.embedding_to_text(emb_list)
    for ch in "بسم":
        assert ch in reconstructed


def test_embedding_to_text_zero_vector(emb):
    zero = np.zeros(len(emb.letters) + 5 + 12 + emb.n_rules)
    txt = emb.embedding_to_text([zero])
    assert isinstance(txt, str)


# -------------------------------------------------------------------
# TEST: compare
# -------------------------------------------------------------------

def test_compare_identical(emb):
    e1 = emb.text_to_embedding("بِسْ", "1", "1")
    e2 = emb.text_to_embedding("بِسْ", "1", "1")
    score = emb.compare(e1, e2)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_compare_completely_different(emb):
    e1 = emb.text_to_embedding("بِ", "1", "1")
    e2 = emb.text_to_embedding("سْ", "1", "1")
    score = emb.compare(e1, e2)
    assert 0 <= score < 1


def test_compare_length_mismatch(emb):
    e1 = emb.text_to_embedding("بِسْمِ", "1", "1")
    e2 = emb.text_to_embedding("بِسْ", "1", "1")
    score = emb.compare(e1, e2)
    assert isinstance(score, float)


def test_compare_zero_vectors(emb):
    e1 = [np.zeros(10)]
    e2 = [np.zeros(10)]
    sim = emb.compare(e1, e2)
    assert sim == 0


# -------------------------------------------------------------------
# TEST: score()
# -------------------------------------------------------------------

def test_score_identical(emb):
    txt = "بِسْ"
    e = emb.text_to_embedding(txt, "1", "1")
    s = emb.score(e, e)
    assert s == 100.0


def test_score_scaled(emb):
    e1 = emb.text_to_embedding("بِ", "1", "1")
    e2 = emb.text_to_embedding("سْ", "1", "1")
    s = emb.score(e1, e2)
    assert 0 <= s <= 100


def test_score_length_mismatch(emb):
    e1 = emb.text_to_embedding("بِسْمِ", "1", "1")
    e2 = emb.text_to_embedding("بِسْ", "1", "1")
    s = emb.score(e1, e2)
    assert isinstance(s, float)


# -------------------------------------------------------------------
# STRESS & CORNER CASES
# -------------------------------------------------------------------

def test_empty_string(emb):
    out = emb.text_to_embedding("", "1", "1")
    assert out == []


def test_only_harakat(emb):
    e = emb.text_to_embedding("َُِّْ", "1", "1")
    assert e == []


def test_repeated_shaddah(emb):
    out = emb.text_to_embedding("بّ", "1", "1")
    assert len(out) == 1


def test_long_sequence(emb):
    txt = "بِسْمِلّٰهِ" * 5
    e = emb.text_to_embedding(txt, "1", "1")
    assert len(e) > 10