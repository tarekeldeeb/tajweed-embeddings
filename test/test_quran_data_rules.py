"""Regression test for tajweed rule counts in the packaged JSON and variants."""

import json
from collections import Counter
from pathlib import Path


def _rule_counts(emb, vectors):
    counts = Counter()
    for vec in vectors:
        for i in (vec[emb.idx_rule_start :] > 0).nonzero()[0]:
            counts[emb.rule_names[i]] += 1
    return dict(counts)


def test_rule_counts_are_stable():
    """
    Ensure tajweed.rules.json retains expected rule occurrence totals.

    Spans are waṣl (continuous recitation): madd_246 (madd ʿāriḍ) is a
    waqf-only rule and never appears; it is produced by stop variants.
    """
    data_path = Path("src/tajweed_embeddings/data/tajweed.rules.json")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    counts = {}
    for entry in data:
        for ann in entry.get("annotations", []):
            rule = ann.get("rule")
            if not rule:
                continue
            counts[rule] = counts.get(rule, 0) + 1

    expected = {
        "ghunnah": 4948,
        "ghunnah_tafkheem": 612,
        "hamzat_wasl": 13599,
        "idghaam_ghunnah": 5397,
        "idghaam_mutajanisayn": 57,
        "idghaam_mutaqaribayn": 13,
        "idghaam_no_ghunnah": 1231,
        "idghaam_shafawi": 824,
        "ikhfa": 5606,
        "ikhfa_shafawi": 487,
        "iqlab": 656,
        "lam_shamsiyyah": 2957,
        "madd_2": 49887,
        "madd_6": 149,
        "madd_munfasil": 3054,
        "madd_muttasil": 2000,
        "qalqalah": 3415,
        "silent": 7347,
    }

    assert counts == expected

    # Spans crossing an āyah boundary are split into a "continues" head and a
    # "continued" tail; the extra continues-only spans are āyah-final silent
    # madd letters before the next āyah's hamzat waṣl.
    continues = sum(
        1 for e in data for a in e["annotations"] if a.get("continues")
    )
    continued = sum(
        1 for e in data for a in e["annotations"] if a.get("continued")
    )
    assert (continues, continued) == (1112, 1100)
    assert sum(counts.values()) == 102239


def test_secondary_rule_counts_are_stable(emb):
    """
    Rule occurrences in the waqf (stop) and ibtidāʾ (start) variant vectors
    over the full corpus. Variant vectors are complete, so unchanged flags of
    a variant letter are counted too.
    """
    stop_vectors = []
    start_vectors = []
    for sura in sorted(int(k) for k in emb.quran.keys()):
        out = emb.text_to_embedding(sura)
        stop_vectors.extend(out.stop_variants.values())
        start_vectors.extend(out.start_variants.values())

    expected_stop = {
        "ghunnah": 1857,
        "idghaam_no_ghunnah": 1,
        "madd_2": 7886,
        "madd_246": 21961,
        "madd_6": 23,
        "madd_muttasil": 925,
        "qalqalah": 4098,
        "silent": 2164,
    }
    expected_start = {
        "madd_2": 16,
    }

    assert (len(stop_vectors), len(start_vectors)) == (82027, 18528)
    assert _rule_counts(emb, stop_vectors) == expected_stop
    assert _rule_counts(emb, start_vectors) == expected_start
