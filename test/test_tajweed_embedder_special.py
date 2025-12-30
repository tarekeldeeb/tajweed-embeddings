import numpy as np


def test_tanween_detection(emb):
    """Tanween marks should produce dedicated haraka states (no shadda combo)."""
    for mark, state in [("ً", "fathatan"), ("ٍ", "kasratan"), ("ٌ", "dammatan")]:
        out = emb.text_to_embedding(1, 1, "ب" + mark)
        assert len(out) == 1
        vec = out[0]
        haraka_slice = vec[emb.idx_haraka_start:emb.idx_haraka_start + emb.n_harakat]
        idx = np.argmax(haraka_slice)
        assert emb.index_to_haraka_state[idx] == state


def test_tanween_aliases(emb):
    """Compatibility tanween forms should map to the same states."""
    cases = [
        ("ﹰ", "fathatan"),
        ("ﹱ", "fathatan"),
        ("ﹲ", "dammatan"),
        ("ﹴ", "kasratan"),
    ]
    for mark, state in cases:
        out = emb.text_to_embedding(1, 1, "ب" + mark)
        assert len(out) == 1
        vec = out[0]
        haraka_slice = vec[emb.idx_haraka_start:emb.idx_haraka_start + emb.n_harakat]
        idx = np.argmax(haraka_slice)
        assert emb.index_to_haraka_state[idx] == state


def test_alt_sukun_detection(emb):
    """Alternate sukun glyph should map to dedicated sukun_zero state."""
    out = emb.text_to_embedding(1, 1, "ب۠")
    assert len(out) == 1
    vec = out[0]
    haraka_slice = vec[emb.idx_haraka_start:emb.idx_haraka_start + emb.n_harakat]
    idx = np.argmax(haraka_slice)
    assert emb.index_to_haraka_state[idx] == "sukun_zero"


def test_sukun_rounded_zero_alias(emb):
    """Rounded zero sukun mark should behave like plain sukun."""
    out = emb.text_to_embedding(1, 1, "ب۟")
    assert len(out) == 1
    vec = out[0]
    haraka_slice = vec[emb.idx_haraka_start:emb.idx_haraka_start + emb.n_harakat]
    idx = np.argmax(haraka_slice)
    assert emb.index_to_haraka_state[idx] == "sukun"


def test_small_glyph_mapping(emb):
    """Small glyphs should embed as their own letters except noon stays alias."""
    cases = [
        ("ۥ", "ۥ"),  # small waw is its own letter
        ("ۦ", "ۦ"),  # canonical small yeh
        ("ۧ", "ۦ"),  # variant maps to canonical small yeh
        ("ۨ", "ن"),  # small noon stays alias
    ]
    for ch, target in cases:
        out = emb.text_to_embedding(1, 1, ch)
        assert len(out) == 1
        vec = out[0]
        letter_idx = np.argmax(vec[:emb.n_letters])
        assert emb.index_to_letter[letter_idx] == target


def test_iqlab_marker_applies_to_previous_letter(emb):
    """Iqlab marker (ۢ) should set a rule on the previous letter."""
    iqlab_vec = emb.text_to_embedding(1, 1, "بۢ")[0]
    lazem_vec = emb.text_to_embedding(1, 1, "بۘ")[0]

    iqlab_letter = emb.index_to_letter[np.argmax(iqlab_vec[:emb.n_letters])]
    lazem_letter = emb.index_to_letter[np.argmax(lazem_vec[:emb.n_letters])]
    iqlab_idx = emb.rule_to_index["iqlab"]

    assert iqlab_letter == "ب"
    assert lazem_letter == "ب"  # lazim pause applies to previous letter
    assert iqlab_vec[emb.idx_rule_start + iqlab_idx] == 1.0


def test_pause_bits_apply_to_previous_letter(emb):
    """Pause marks should not create new vectors but set pause bits on the previous letter."""
    out = emb.text_to_embedding(1, 1, "بۘ")  # Lazem (mandatory stop)
    assert len(out) == 1
    vec = out[0]
    pause_slice = vec[emb.idx_pause_start:emb.idx_pause_start + emb.n_pause]
    category = int(pause_slice[0] + (pause_slice[1] * 2) + (pause_slice[2] * 4))
    assert category == 7  # lazim code
    assert pause_slice[2] == 1  # mandatory


def test_rule_markers_attach_to_previous_letter(emb):
    """Tajweed rule markers (non-pauses) should not create extra vectors."""
    markers = [
        ("ۢ", "rule-iqlab"),
        ("۬", "rule-tas-heel"),
        ("۪", "rule-imala"),
        ("۫", "rule-ishmam"),
        ("ۣ", "rule-optional-seen"),
    ]
    for ch, _label in markers:
        out = emb.text_to_embedding(1, 1, f"ب{ch}")
        assert len(out) == 1
        vec = out[0]
        letter_idx = np.argmax(vec[:emb.n_letters])
        assert emb.index_to_letter[letter_idx] == "ب"


def test_rule_markers_set_rule_flags(emb):
    """Ensure tajweed rule markers raise their corresponding rule bits."""
    markers = {
        "ۢ": "iqlab",
        "۬": "tas_heel",
        "۪": "imala",
        "۫": "ishmam",
        "ۣ": "optional_seen",
    }
    for ch, rule in markers.items():
        out = emb.text_to_embedding(1, 1, f"ب{ch}")
        vec = out[0]
        ri = emb.rule_to_index[rule]
        assert vec[emb.idx_rule_start + ri] == 1.0


def test_ishmam_marker_attaches_to_next_letter_in_12_11(emb):
    """Ishmam marker in 12:11 should apply to the following letter."""
    ishmam_idx = emb.rule_to_index.get("ishmam")
    if ishmam_idx is None:
        return
    vecs = emb.text_to_embedding(12, 11)
    assert len(vecs) > 21

    prev_vec = vecs[20]
    next_vec = vecs[21]
    prev_letter = emb.index_to_letter[int(np.argmax(prev_vec[:emb.n_letters]))]
    next_letter = emb.index_to_letter[int(np.argmax(next_vec[:emb.n_letters]))]

    assert prev_letter == "م"
    assert next_letter == "ن"
    assert prev_vec[emb.idx_rule_start + ishmam_idx] == 0.0
    assert next_vec[emb.idx_rule_start + ishmam_idx] > 0.0
"""Special-case behaviors (pause bits, unknowns, etc.)."""
