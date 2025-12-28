"""Sifat editor tests for contextual isti'la/istifal updates."""

import numpy as np


def _isti_la_bit(emb, vec):
    return vec[emb.idx_sifat_start + 3] > 0


def _letter_at(emb, vec):
    return emb.index_to_letter[int(np.argmax(vec[: emb.n_letters]))]


def _haraka_slice(emb, vec):
    return vec[emb.idx_haraka_start : emb.idx_haraka_start + emb.n_harakat]


def _find_ra(emb, vecs):
    for v in vecs:
        if _letter_at(emb, v) == "ر":
            return v
    return None


def _find_allah_lams(emb, vecs):
    indices = []
    for i, v in enumerate(vecs):
        if _letter_at(emb, v) != "ل":
            continue
        base, has_shadda = emb.haraka_helper.decode_haraka(_haraka_slice(emb, v))
        if not has_shadda:
            continue
        if i + 1 < len(vecs) and _letter_at(emb, vecs[i + 1]) == "ه":
            indices.append(i)
    return indices


def test_raa_tafkhim_with_fatha_damma(emb):
    for txt in ("رَ", "رُ"):
        vecs = emb.text_to_embedding(1, 1, txt)
        raa = _find_ra(emb, vecs)
        assert raa is not None
        assert _isti_la_bit(emb, raa)


def test_raa_tarqiq_with_kasra(emb):
    vecs = emb.text_to_embedding(1, 1, "رِ")
    raa = _find_ra(emb, vecs)
    assert raa is not None
    assert not _isti_la_bit(emb, raa)


def test_raa_sukun_prev_fatha(emb):
    vecs = emb.text_to_embedding(1, 1, "فَرْ")
    raa = _find_ra(emb, vecs)
    assert raa is not None
    assert _isti_la_bit(emb, raa)


def test_raa_sukun_prev_kasra_next_isti_la(emb):
    vecs = emb.text_to_embedding(1, 1, "مِرْصَادًا")
    raa = _find_ra(emb, vecs)
    assert raa is not None
    assert _isti_la_bit(emb, raa)


def test_raa_sukun_after_yaa_sukun(emb):
    vecs = emb.text_to_embedding(1, 1, "خَيْرْ")
    raa = _find_ra(emb, vecs)
    assert raa is not None
    assert not _isti_la_bit(emb, raa)


def test_lam_allah_tafkhim_after_fatha_damma(emb):
    vecs = emb.text_to_embedding(59, 22)
    indices = _find_allah_lams(emb, vecs)
    assert indices, "Expected Allah lam with shadda in 59:22"
    assert any(_isti_la_bit(emb, vecs[i]) for i in indices)


def test_lam_allah_tarqiq_after_kasra(emb):
    vecs = emb.text_to_embedding(1, 1)
    indices = _find_allah_lams(emb, vecs)
    assert indices, "Expected Allah lam with shadda in 1:1"
    assert any(not _isti_la_bit(emb, vecs[i]) for i in indices)
