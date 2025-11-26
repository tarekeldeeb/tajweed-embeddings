def test_encoding_to_string_single_line_has_sections(emb):
    """encoding_to_string returns a readable, single-line description for one vector."""
    vec = emb.text_to_embedding(1, 1, "بَ")[0]
    out = emb.encoding_to_string(vec)

    assert isinstance(out, str)
    assert "Letter:" in out
    assert "Haraka:" in out
    assert "Pause:" in out


def test_encoding_to_string_sequence_has_one_line_per_vector(emb):
    """Sequence input should yield one formatted line per embedding with indices."""
    seq = emb.text_to_embedding(1, 1, "بِس")
    out = emb.encoding_to_string(seq)

    lines = out.splitlines()
    assert len(lines) == len(seq)
    assert lines[0].startswith("[0] ")
    assert lines[-1].startswith(f"[{len(seq) - 1}] ")


def test_encoding_to_string_includes_rules_when_active(emb):
    """Active rule bits should be listed in the formatted output."""
    vec = emb.text_to_embedding(1, 1, "بَ")[0].copy()

    # Manually activate a known rule slot to avoid relying on specific spans.
    rule_name = emb.rule_names[0]
    rule_idx = emb.rule_to_index[rule_name]
    vec[emb.idx_rule_start + rule_idx] = 1.0

    out = emb.encoding_to_string(vec)
    assert f"Rules: {rule_name}" in out


def test_encoding_to_string_silent_stops_after_haraka(emb):
    """When haraka is absent, later vector slices are omitted."""
    vec = emb.text_to_embedding(1, 1, "ب")[0]
    out = emb.encoding_to_string(vec)

    assert "Letter:" in out
    assert "Haraka: (none)" in out
    assert "Pause:" not in out
    assert "Sifat:" not in out
    assert "Rules:" not in out


def test_encoding_to_string_long_vowel_not_silent(emb):
    """Long vowels without marks should still be treated as voiced."""
    vecs = emb.text_to_embedding(1, 1, "با")  # alif typically unmarked
    alif_vec = vecs[1]
    out = emb.encoding_to_string(alif_vec)

    assert "Haraka: madd" in out
    assert "Pause:" in out  # downstream sections are still printed


def test_encoding_to_string_dagger_alif_is_madd(emb):
    """Dagger alif mark should surface as madd."""
    vec = emb.text_to_embedding(1, 1, "ٰ")[0]
    out = emb.encoding_to_string(vec)

    assert "Haraka: madd" in out
