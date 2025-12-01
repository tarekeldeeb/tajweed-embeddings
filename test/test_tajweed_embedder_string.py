import re

def test_encoding_to_string_single_line_has_sections(emb):
    """encoding_to_string returns a readable, single-line description for one vector."""
    vec = emb.text_to_embedding(1, 1, "بَ")[0]
    out = emb.encoding_to_string(vec, style="long")

    assert isinstance(out, str)
    assert "Letter:" in out
    assert "Haraka:" in out
    assert "Pause:" in out


def test_encoding_to_string_sequence_has_one_line_per_vector(emb):
    """Sequence input should yield one formatted line per embedding with indices."""
    seq = emb.text_to_embedding(1, 1, "بِس")
    out = emb.encoding_to_string(seq)

    lines = out.splitlines()
    # Strip ANSI dim codes for prefix checks
    stripped = [re.sub(r"\x1b\[[0-9;]*m", "", ln) for ln in lines]
    assert len(lines) == len(seq)
    assert stripped[0].startswith("[0] ")
    assert stripped[-1].startswith(f"[{len(seq) - 1}] ")


def test_encoding_to_string_includes_rules_when_active(emb):
    """Active rule bits should be listed in the formatted output."""
    vec = emb.text_to_embedding(1, 1, "بَ")[0].copy()

    # Manually activate a known rule slot to avoid relying on specific spans.
    rule_name = emb.rule_names[0]
    rule_idx = emb.rule_to_index[rule_name]
    vec[emb.idx_rule_start + rule_idx] = 1.0

    out = emb.encoding_to_string(vec, style="long")
    assert f"Rules: {rule_name}" in out


def test_encoding_to_string_silent_stops_after_haraka(emb):
    """When haraka is absent, later vector slices are omitted."""
    vec = emb.text_to_embedding(1, 1, "ب")[0]
    out = emb.encoding_to_string(vec, style="long")

    assert "Letter:" in out
    assert "Haraka:" in out
    assert "(none)" in out
    assert "Pause:" not in out
    assert "Sifat:" not in out
    assert "Rules:" not in out


def test_encoding_to_string_dims_when_silent_rule(emb):
    """Silent rule should dim the rendered line even if haraka is present."""
    vec = emb.text_to_embedding(1, 1, "بَ")[0].copy()
    silent_idx = emb.rule_to_index["silent"]
    vec[emb.idx_rule_start + silent_idx] = 1.0

    out = emb.encoding_to_string(vec)
    assert "\x1b[90m" in out  # dim applied to full line


def test_encoding_to_string_dimming_rules_trigger(emb):
    """Lines with silent/hamzat_wasl/lam_shamsiyyah should be dimmed."""
    vec = emb.text_to_embedding(1, 1, "بَ")[0].copy()
    dim_rules = [r for r in ("silent", "hamzat_wasl", "lam_shamsiyyah") if r in emb.rule_to_index]
    if not dim_rules:
        pytest.skip("No dimming rules present in rule set")
    for rule in dim_rules:
        vec[emb.idx_rule_start + emb.rule_to_index[rule]] = 1.0
        out = emb.encoding_to_string([vec])
        assert "\x1b[90m" in out
        vec[emb.idx_rule_start + emb.rule_to_index[rule]] = 0.0


def test_encoding_to_string_no_dim_without_rule(emb):
    """Lines without dimming rules should not be dimmed."""
    vec = emb.text_to_embedding(1, 1, "بَ")[0]
    out = emb.encoding_to_string([vec])
    assert "\x1b[90m" not in out
"""String formatting tests for encoding_to_string."""
