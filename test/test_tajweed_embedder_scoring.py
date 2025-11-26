# test/test_tajweed_embedder_scoring.py
import copy
import numpy as np
import pytest


def _clone_sequence(seq):
    return [v.copy() for v in seq]


# --- SCORE TESTS (0–100 range) ----------------------------------------------------

def test_score_identical_vs_haraka_change(emb):
    base_text = "بِسْ"
    haraka_text = "بَسْ"

    e_base = emb.text_to_embedding(1, 1, base_text)
    e_haraka = emb.text_to_embedding(1, 1, haraka_text)

    s_ident = emb.score(e_base, e_base)   # expect approx 100
    s_haraka = emb.score(e_base, e_haraka)

    assert 0 <= s_ident <= 100
    assert s_ident == pytest.approx(100.0, rel=1e-3)

    assert 0 <= s_haraka <= 100
    assert s_haraka < s_ident


def test_score_letter_change_vs_haraka_change(emb):
    base_text = "بِسْ"
    haraka_text = "بَسْ"
    letter_text = "تِسْ"

    e_base = emb.text_to_embedding(1, 1, base_text)
    e_haraka = emb.text_to_embedding(1, 1, haraka_text)
    e_letter = emb.text_to_embedding(1, 1, letter_text)

    s_ident = emb.score(e_base, e_base)
    s_haraka = emb.score(e_base, e_haraka)
    s_letter = emb.score(e_base, e_letter)

    assert s_ident == pytest.approx(100.0, rel=1e-3)

    assert s_letter <= s_haraka  # letter mismatch <= haraka mismatch
    assert s_haraka < s_ident    # both less than perfect


def test_score_single_rule_bit_flip(emb):
    text = "بِسْمِ اللَّهِ"
    e_base = emb.text_to_embedding(1, 1, text)
    e_mod = _clone_sequence(e_base)

    if emb.idx_rule_start < emb.embedding_dim:
        idx = emb.idx_rule_start
        e_mod[0][idx] = 1.0 - e_mod[0][idx]

    s_ident = emb.score(e_base, e_base)
    s_rule = emb.score(e_base, e_mod)

    assert s_ident == pytest.approx(100.0, rel=1e-3)
    assert 0 <= s_rule <= 100
    assert s_rule < s_ident
    assert s_rule > 0


# --- COMPARE TESTS (normalized 0–1, slight float overshoot ok) --------------------

def assert_norm(v):
    """Allow compare() to return up to 1.000001 due to float ops."""
    assert -0.000001 <= v <= 1.000001


def test_compare_consistent_with_score_for_identical(emb):
    text = "بِسْمِ"
    e = emb.text_to_embedding(1, 1, text)

    s = emb.score(e, e)      # 100
    c = emb.compare(e, e)    # ~1

    assert 0 <= s <= 100
    assert_norm(c)

    assert s == pytest.approx(100.0, rel=1e-3)
    assert c == pytest.approx(1.0, rel=1e-6)


def test_compare_orders_slight_variations(emb):
    base_text = "بِسْ"
    haraka_text = "بَسْ"
    letter_text = "تِسْ"

    e_base = emb.text_to_embedding(1, 1, base_text)
    e_haraka = emb.text_to_embedding(1, 1, haraka_text)
    e_letter = emb.text_to_embedding(1, 1, letter_text)

    c_ident = emb.compare(e_base, e_base)
    c_haraka = emb.compare(e_base, e_haraka)
    c_letter = emb.compare(e_base, e_letter)

    assert_norm(c_ident)
    assert_norm(c_haraka)
    assert_norm(c_letter)

    assert c_ident == pytest.approx(1.0, rel=1e-6)
    assert c_haraka < c_ident
    assert c_letter <= c_haraka + 1e-6


def test_score_symmetry(emb):
    t1 = "بِسْمِ"
    t2 = "بِسْمَ"

    e1 = emb.text_to_embedding(1, 1, t1)
    e2 = emb.text_to_embedding(1, 1, t2)

    s12 = emb.score(e1, e2)
    s21 = emb.score(e2, e1)

    assert_norm(s12 / 100.0)  # scaling to check within expected numeric bounds
    assert s12 == pytest.approx(s21, rel=1e-6)


def test_compare_symmetry(emb):
    t1 = "مَالِكِ"
    t2 = "مَلِكِ"

    e1 = emb.text_to_embedding(1, 1, t1)
    e2 = emb.text_to_embedding(1, 1, t2)

    c12 = emb.compare(e1, e2)
    c21 = emb.compare(e2, e1)

    assert_norm(c12)
    assert_norm(c21)

    assert c12 == pytest.approx(c21, rel=1e-6)


# --- EMPTY SEQUENCE BEHAVIOR ------------------------------------------------------

def test_score_handles_empty_sequences(emb):
    s = emb.score([], [])
    assert isinstance(s, float)
    assert np.isfinite(s)
    assert 0 <= s <= 100


def test_compare_handles_empty_sequences(emb):
    c = emb.compare([], [])
    assert isinstance(c, float)
    assert np.isfinite(c)
    assert_norm(c)
"""Similarity and scoring behavior tests."""
