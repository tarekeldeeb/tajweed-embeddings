"""Secondary phoneme (waqf/ibtidāʾ) variant tests."""
import numpy as np
import pytest


def _haraka(emb, vec):
    h = vec[emb.idx_haraka_start : emb.idx_haraka_start + emb.n_harakat]
    if h.max() <= 0:
        return None
    return emb.index_to_haraka_state[int(np.argmax(h))]


def _letter(emb, vec):
    return emb.index_to_letter[int(np.argmax(vec[: emb.n_letters]))]


def _rules(emb, vec):
    return {
        emb.rule_names[i]
        for i, v in enumerate(vec[emb.idx_rule_start :])
        if v > 0
    }


def test_result_type_and_dims(emb):
    """text_to_embedding returns a list-compatible TajweedEmbedding."""
    out = emb.text_to_embedding(1, 1)
    assert isinstance(out, list)
    assert hasattr(out, "stop_variants") and hasattr(out, "start_variants")
    assert out.words, "expected word spans"
    for vec in list(out.stop_variants.values()) + list(out.start_variants.values()):
        assert vec.shape[0] == emb.embedding_dim


def test_sukun_arid_on_final_vowel(emb):
    """Stopping on بِسْمِ turns the final kasra into sukūn."""
    out = emb.text_to_embedding(1, 1, "بِسْمِ")
    last = len(out) - 1
    assert last in out.stop_variants
    assert _haraka(emb, out.stop_variants[last]) == "sukun"


def test_sukun_shadda_state(emb):
    """Stopping on وَتَبَّ keeps the shadda: sukūn+shadda state + qalqalah."""
    out = emb.text_to_embedding(1, 1, "وَتَبَّ")
    last = len(out) - 1
    var = out.stop_variants[last]
    assert _haraka(emb, var) == "sukun_shadda"
    assert "qalqalah" in _rules(emb, var)


def test_qalqalah_kubra_at_stop(emb):
    """ريب in 2:2 gains qalqalah on the ب once sākin."""
    out = emb.text_to_embedding(2, 2)
    hits = [
        i
        for i, var in out.stop_variants.items()
        if _letter(emb, var) == "ب"
        and _haraka(emb, var) == "sukun"
        and "qalqalah" in _rules(emb, var)
    ]
    assert hits, "expected qalqalah kubrā stop variants in 2:2"


def test_madd_arid_lil_sukun(emb):
    """فِيهِ in 2:2: the ي flips madd_2 → madd_246 when the ه takes sukūn."""
    out = emb.text_to_embedding(2, 2)
    flipped = [
        i
        for i, var in out.stop_variants.items()
        if "madd_246" in _rules(emb, var) and "madd_2" in _rules(emb, out[i])
    ]
    assert flipped, "expected madd ʿāriḍ promotions in 2:2"


def test_madd_iwad_with_seat(emb):
    """هُدًى in 2:2: fathatan → fatha and the ى voiced as madd_2."""
    out = emb.text_to_embedding(2, 2)
    seat = next(
        i
        for i, var in out.stop_variants.items()
        if _letter(emb, var) == "ى" and "madd_2" in _rules(emb, var)
    )
    assert _haraka(emb, out.stop_variants[seat - 1]) == "fatha"
    assert _haraka(emb, out[seat - 1]) == "fathatan"
    # Cross-word idghām into لِّلْمُتَّقِينَ is dropped at stop.
    assert "idghaam_no_ghunnah" not in _rules(emb, out.stop_variants[seat])
    assert "idghaam_no_ghunnah" in _rules(emb, out[seat])


def test_madd_iwad_without_seat_samaa_rule(emb):
    """مَآءً-type endings (no seat alif) carry the ʿiwaḍ madd themselves."""
    out = emb.text_to_embedding(1, 1, "مَآءً")
    last = len(out) - 1
    var = out.stop_variants[last]
    assert _letter(emb, var) == "ء"
    assert _haraka(emb, var) == "madd"
    assert "madd_2" in _rules(emb, var)


def test_ta_marbuta_to_ha(emb):
    """Stopping on ة pronounces hāʾ sākinah (letter substitution)."""
    out = emb.text_to_embedding(1, 1, "ٱلصَّلَوٰةَ")
    last = len(out) - 1
    var = out.stop_variants[last]
    assert _letter(emb, var) == "ه"
    assert _haraka(emb, var) == "sukun"
    # Dagger alif before it becomes madd ʿāriḍ.
    assert "madd_246" in _rules(emb, out.stop_variants[last - 1])


def test_ha_silah_dropped_at_stop(emb):
    """بِهِۦ at stop: the small yāʾ silenced, pronoun hāʾ takes sukūn."""
    out = emb.text_to_embedding(1, 1, "بِهِۦ")
    ha, sila = len(out) - 2, len(out) - 1
    assert _haraka(emb, out.stop_variants[ha]) == "sukun"
    assert "silent" in _rules(emb, out.stop_variants[sila])


def test_hamzat_wasl_definite_article_fatha(emb):
    """Starting at ٱلْحَمْدُ pronounces the hamza with fatha."""
    out = emb.text_to_embedding(1, 2)
    first_variants = {
        i: var for i, var in out.start_variants.items() if _letter(emb, out[i]) == "ٱ"
    }
    assert first_variants
    var = first_variants[min(first_variants)]
    assert _letter(emb, var) == "أ"
    assert _haraka(emb, var) == "fatha"
    assert "hamzat_wasl" not in _rules(emb, var)


def test_hamzat_wasl_verb_kasra(emb):
    """ٱهْدِنَا starts with kasra (third letter not damma)."""
    out = emb.text_to_embedding(1, 1, "ٱهْدِنَا")
    assert _haraka(emb, out.start_variants[0]) == "kasra"


def test_hamzat_wasl_verb_damma(emb):
    """ٱنظُرْ starts with damma (third letter carries permanent damma)."""
    out = emb.text_to_embedding(1, 1, "ٱنظُرْ")
    assert _haraka(emb, out.start_variants[0]) == "damma"


def test_hamzat_wasl_damma_exception(emb):
    """ٱئْتُوا۟ starts with kasra (ʿāriḍah damma) and ibdāl ئ → ي madd."""
    out = emb.text_to_embedding(1, 1, "ٱئْتُوا۟")
    assert _haraka(emb, out.start_variants[0]) == "kasra"
    ibdal = out.start_variants[1]
    assert _letter(emb, ibdal) == "ي"
    assert "madd_2" in _rules(emb, ibdal)


def test_hamzat_wasl_samai_noun_kasra(emb):
    """ٱسْمُ starts with kasra despite the damma on its third letter."""
    out = emb.text_to_embedding(1, 1, "ٱسْمُ")
    assert _haraka(emb, out.start_variants[0]) == "kasra"


def test_incoming_idgham_shadda_dropped(emb):
    """Starting at لِّلْمُتَّقِينَ (2:2) drops the written idghām shadda."""
    out = emb.text_to_embedding(2, 2)
    hits = [
        i
        for i, var in out.start_variants.items()
        if _haraka(emb, out[i]) == "kasra_shadda" and _haraka(emb, var) == "kasra"
    ]
    assert hits
    for i in hits:
        assert "idghaam_no_ghunnah" not in _rules(emb, var := out.start_variants[i])


def test_no_variants_for_muqattaa(emb):
    """The muqaṭṭaʿa word in 2:1 (after the Basmala) needs no variants."""
    out = emb.text_to_embedding(2, 1)
    first, last = out.words[-1]  # الٓمٓ
    span = set(range(first, last + 1))
    assert not span & set(out.start_variants)
    assert not span & set(out.stop_variants)


def test_realize_identity_and_pause(emb):
    """realize() returns primaries by default and swaps only at the pause."""
    out = emb.text_to_embedding(1, 1)
    plain = out.realize()
    assert all(np.array_equal(a, b) for a, b in zip(plain, out))

    paused = out.realize(stop_words={0})
    diff = {
        i for i, (a, b) in enumerate(zip(paused, out)) if not np.array_equal(a, b)
    }
    w0_first, w0_last = out.words[0]
    w1_first, w1_last = out.words[1]
    assert diff, "a pause must alter at least one vector"
    assert all(w0_first <= i <= w1_last for i in diff)
    assert not any(w1_last < i for i in diff)


def test_multi_ayah_variant_offsets(emb):
    """Variant indices stay aligned when segments are concatenated."""
    single = emb.text_to_embedding(1, 2)
    combined = emb.text_to_embedding(1, 1, count=2)
    offset = len(emb.text_to_embedding(1, 1))
    for i, var in single.stop_variants.items():
        assert np.array_equal(combined.stop_variants[i + offset], var)
    for i, var in single.start_variants.items():
        assert np.array_equal(combined.start_variants[i + offset], var)


def test_variants_do_not_touch_pause_slice(emb):
    """Stop variants leave the pause slice (metadata) unchanged."""
    out = emb.text_to_embedding(1, 1)
    sl = slice(emb.idx_pause_start, emb.idx_pause_start + emb.n_pause)
    for i, var in out.stop_variants.items():
        assert np.array_equal(var[sl], out[i][sl])
