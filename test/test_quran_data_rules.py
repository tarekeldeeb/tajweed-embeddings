"""Regression test for tajweed rule counts in the packaged JSON."""

import json
from pathlib import Path


def test_rule_counts_are_stable():
    """Ensure tajweed.rules.json retains expected rule occurrence totals."""
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
        "ghunnah_tafkheem": 1016,
        "hamzat_wasl": 13599,
        "idghaam_ghunnah": 3527,
        "idghaam_mutajanisayn": 57,
        "idghaam_mutaqaribayn": 13,
        "idghaam_no_ghunnah": 1001,
        "idghaam_shafawi": 822,
        "ikhfa": 5493,
        "ikhfa_shafawi": 483,
        "iqlab": 562,
        "lam_shamsiyyah": 2957,
        "madd_2": 48511,
        "madd_246": 4543,
        "madd_6": 149,
        "madd_munfasil": 2999,
        "madd_muttasil": 1997,
        "qalqalah": 3834,
        "silent": 4179,
    }

    assert counts == expected
    assert sum(counts.values()) == 100690
