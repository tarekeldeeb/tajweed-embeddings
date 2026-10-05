"""Regression test for tajweed rule counts in the packaged JSON."""

import json
from pathlib import Path


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
        "ghunnah_tafkheem": 1016,
        "hamzat_wasl": 13599,
        "idghaam_ghunnah": 5397,
        "idghaam_mutajanisayn": 57,
        "idghaam_mutaqaribayn": 13,
        "idghaam_no_ghunnah": 1231,
        "idghaam_shafawi": 824,
        "ikhfa": 6134,
        "ikhfa_shafawi": 487,
        "iqlab": 656,
        "lam_shamsiyyah": 2957,
        "madd_2": 53043,
        "madd_6": 149,
        "madd_munfasil": 3057,
        "madd_muttasil": 1997,
        "qalqalah": 3415,
        "silent": 4215,
    }

    assert counts == expected

    # Spans crossing an āyah boundary are split into a "continues" head and a
    # "continued" tail; the extra continues-only spans are āyah-final silent
    # alifs before the next āyah's hamzat waṣl.
    continues = sum(
        1 for e in data for a in e["annotations"] if a.get("continues")
    )
    continued = sum(
        1 for e in data for a in e["annotations"] if a.get("continued")
    )
    assert (continues, continued) == (1136, 1100)
    assert sum(counts.values()) == 103195
