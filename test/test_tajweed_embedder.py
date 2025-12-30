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


def test_embedding_to_text_includes_word_spaces(emb):
    """Word-boundary pause should reinsert spaces."""
    sample = "بِسْمِ ٱللَّهِ"
    vecs = emb.text_to_embedding(1, 1, sample)
    reconstructed = emb.embedding_to_text(vecs)
    assert " " in reconstructed


def test_embedding_to_text_long_ayah_round_trip(emb):
    """Long ayah round-trip preserves base letters order."""
    text = emb.quran["2"]["282"]
    vecs = emb.text_to_embedding(2, 282)
    reconstructed = emb.embedding_to_text(vecs)

    def _letters_only(src: str) -> list[str]:
        normalized = emb._normalize_text(src)  # type: ignore[attr-defined]
        normalized = normalized.replace("آ", "آ")
        letters = []
        for ch in normalized:
            norm_ch = emb.char_aliases.get(ch, ch)
            if norm_ch == "آ":
                norm_ch = "ا"
            if norm_ch in emb.letters:
                letters.append(norm_ch)
        return letters

    assert _letters_only(text) == _letters_only(reconstructed)


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


def test_madd_6_present_in_muqattaat(emb):
    """All muqatta'a ayat should include madd_6 rule flags."""
    madd_6_idx = emb.rule_to_index.get("madd_6")
    assert madd_6_idx is not None, "madd_6 rule missing from rule index"
    targets = [
        (2, 1),
        (3, 1),
        (7, 1),
        (10, 1),
        (11, 1),
        (12, 1),
        (13, 1),
        (14, 1),
        (15, 1),
        (19, 1),
        (26, 1),
        (27, 1),
        (28, 1),
        (29, 1),
        (30, 1),
        (31, 1),
        (32, 1),
        (36, 1),
        (38, 1),
        (40, 1),
        (41, 1),
        (42, 1),
        (42, 2),
        (43, 1),
        (44, 1),
        (45, 1),
        (46, 1),
        (50, 1),
        (68, 1),
    ]
    for sura, ayah in targets:
        vecs = emb.text_to_embedding(sura, ayah)
        assert any(vec[emb.idx_rule_start + madd_6_idx] > 0 for vec in vecs), (
            f"madd_6 missing for {sura}:{ayah}"
        )


def test_ikhfa_span_across_pause(emb):
    """Ikhfa on tanween should persist across pause marks/spaces (10:68 “وَلَدًا ۗ”)."""
    embeddings = emb.text_to_embedding(10, 68)
    ikhfa_idx = emb.rule_to_index.get("ikhfa")
    assert ikhfa_idx is not None

    def letter_and_ikhfa(vec):
        letter_slice = vec[: emb.n_letters]
        letter = emb.index_to_letter[int(np.argmax(letter_slice))]
        has_ikhfa = vec[emb.idx_rule_start + ikhfa_idx] > 0
        return letter, has_ikhfa

    tagged = [letter_and_ikhfa(v) for v in embeddings]

    # Find the tanween on د in “وَلَدًا” and ensure ikhfa is active there
    target_idx = None
    for i, (ltr, has) in enumerate(tagged):
        if ltr == "د" and has:
            target_idx = i
            break
    assert target_idx is not None, "Ikhfa not tagged on tanween of د in 10:68"

    # Span should continue to at least one of the following letters (across pause/space)
    assert any(
        j < len(tagged) and tagged[j][1] for j in (target_idx + 1, target_idx + 2)
    ), "Ikhfa span did not cross the pause mark after tanween in 10:68"


@pytest.mark.parametrize("subtext", ["فِرْقٍ", "فرق"])
def test_subtext_ikhfa_on_last_char(emb, subtext):
    """Subtext matching should preserve ikhfa on the final letter."""
    ikhfa_idx = emb.rule_to_index.get("ikhfa")
    if ikhfa_idx is None:
        pytest.skip("ikhfa rule not present")

    vecs = emb.text_to_embedding(26, 63, subtext)
    assert vecs, "Empty embeddings for subtext"

    last_vec = vecs[-1]
    letter_idx = int(np.argmax(last_vec[: emb.n_letters]))
    assert emb.index_to_letter[letter_idx] == "ق"
    assert last_vec[emb.idx_rule_start + ikhfa_idx] > 0


def test_maddah_above_attaches_to_alif(emb):
    """Decomposed alif+maddah should produce one letter with madd haraka."""
    out = emb.text_to_embedding(1, 7, "آ")
    assert len(out) == 1
    vec = out[0]
    haraka_slice = vec[emb.idx_haraka_start : emb.idx_haraka_start + emb.n_harakat]
    idx = int(haraka_slice.argmax())
    assert emb.index_to_haraka_state.get(idx) == "madd"


def test_no_implicit_madd_for_bare_waw(emb):
    """Unmarked waw in basmala should not get implicit madd haraka."""
    vecs = emb.text_to_embedding(1)
    # Waw before the final ba (index observed in debug traces)
    vec = vecs[128]
    haraka_slice = vec[emb.idx_haraka_start : emb.idx_haraka_start + emb.n_harakat]
    idx = int(haraka_slice.argmax())
    assert emb.index_to_haraka_state.get(idx) != "madd"


def test_multi_ayah_count_embeds_consecutively(emb):
    """count>1 should embed consecutive ayāt."""
    vecs = emb.text_to_embedding(1, 1, count=2)
    expected = len(emb.text_to_embedding(1, 1)) + len(emb.text_to_embedding(1, 2))
    assert len(vecs) == expected


def test_count_must_be_positive(emb):
    """count must be > 0."""
    with pytest.raises(ValueError):
        emb.text_to_embedding(1, 1, count=0)


def test_silent_on_alif_before_hamzat_wasl(emb):
    """Alif with no haraka before a spaced hamzat wasl should be marked silent."""
    if "silent" not in emb.rule_to_index:
        pytest.skip("silent rule not present")
    for sura_str, ayat_map in emb.quran.items():
        for ayah_str, text in ayat_map.items():
            chars = list(text)
            target_raw = None
            for i in range(len(chars) - 2):
                if chars[i] in ("ا", "ى") and chars[i + 1].isspace() and chars[i + 2] == "ٱ":
                    target_raw = i
                    break
            if target_raw is None:
                continue

            vecs = emb.text_to_embedding(int(sura_str), int(ayah_str))
            filtered_idx = -1
            current_filtered = -1
            for raw_idx, ch in enumerate(chars):
                norm_ch = emb.char_aliases.get(ch, ch) if hasattr(emb, "char_aliases") else ch
                if norm_ch in emb.letters:
                    current_filtered += 1
                    if raw_idx == target_raw:
                        filtered_idx = current_filtered
                        break
            if filtered_idx < 0 or filtered_idx >= len(vecs):
                continue
            vec = vecs[filtered_idx]
            silent_idx = emb.rule_to_index["silent"]
            assert vec[emb.idx_rule_start + silent_idx] > 0
            return
    pytest.fail("Did not find alif before hamzat wasl to validate silent rule")


def test_composed_alif_madd_produces_single_vector(emb):
    """Composed alif+maddah should map to one vector with madd haraka."""
    vecs = emb.text_to_embedding(1, 7, "آل")
    assert len(vecs) == 2
    haraka_slice = vecs[0][emb.idx_haraka_start : emb.idx_haraka_start + emb.n_harakat]
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


def test_no_duplicate_ikhfa_spans(emb):
    """Ikhfa spans should not overlap within an āyah."""
    if "ikhfa" not in emb.rule_to_index:
        pytest.skip("ikhfa not present in rules JSON")
    for (sura, ayah), anns in emb.tajweed_rules.rules_index.items():
        covered = set()
        for ann in anns:
            if ann.get("rule") != "ikhfa":
                continue
            start = int(ann.get("start", 0))
            end = int(ann.get("end", 0))
            for idx in range(start, end):
                assert idx not in covered, f"Duplicate ikhfa coverage at {sura}:{ayah} index {idx}"
                covered.add(idx)


def test_madd_munfasil_and_muttasil_present(emb):
    """Ensure derived madd spans surface somewhere in the corpus."""
    required = [r for r in ("madd_munfasil", "madd_muttasil") if r in emb.rule_to_index]
    if not required:
        pytest.skip("Derived madd rules not present in rules JSON")
    seen = {r: False for r in required}
    for anns in emb.tajweed_rules.rules_index.values():
        for ann in anns:
            r = ann.get("rule")
            if r in seen:
                seen[r] = True
        if all(seen.values()):
            break
    missing = [r for r, v in seen.items() if not v]
    assert not missing, f"Missing derived madd spans: {missing}"


def test_madd_priority_strips_weaker_flags(emb):
    """Ensure madd priority removes weaker rules when spans overlap."""
    priority = ["madd_6", "madd_muttasil", "madd_munfasil", "madd_246", "madd_2"]
    present = [r for r in priority if r in emb.rule_to_index]
    if len(present) < 2:
        pytest.skip("Not enough madd rules present to test priority")

    def strongest(active_rules):
        for r in priority:
            if r in active_rules:
                return r
        return None

    for vecs in (
        emb.text_to_embedding(1, 1),
        emb.text_to_embedding(2, 4),
        emb.text_to_embedding(2, 5),
    ):
        for v in vecs:
            active = [
                r
                for r in present
                if v[emb.idx_rule_start + emb.rule_to_index[r]] > 0
            ]
            if len(active) <= 1:
                continue
            top = strongest(active)
            assert top is not None
            assert active == [top], f"Weaker madd present alongside {top}: {active}"


def test_no_duplicate_ikhfa_spans(emb):
    """Ikhfa spans should not overlap within an āyah."""
    if "ikhfa" not in emb.rule_to_index:
        pytest.skip("ikhfa not present in rules JSON")
    for (sura, ayah), anns in emb.tajweed_rules.rules_index.items():
        covered = set()
        for ann in anns:
            if ann.get("rule") != "ikhfa":
                continue
            start = int(ann.get("start", 0))
            end = int(ann.get("end", 0))
            for idx in range(start, end):
                assert idx not in covered, f"Duplicate ikhfa coverage at {sura}:{ayah} index {idx}"
                covered.add(idx)


def test_madd_munfasil_and_muttasil_present(emb):
    """Ensure derived madd spans surface somewhere in the corpus."""
    required = [r for r in ("madd_munfasil", "madd_muttasil") if r in emb.rule_to_index]
    if not required:
        pytest.skip("Derived madd rules not present in rules JSON")
    seen = {r: False for r in required}
    for anns in emb.tajweed_rules.rules_index.values():
        for ann in anns:
            r = ann.get("rule")
            if r in seen:
                seen[r] = True
        if all(seen.values()):
            break
    missing = [r for r, v in seen.items() if not v]
    assert not missing, f"Missing derived madd spans: {missing}"
