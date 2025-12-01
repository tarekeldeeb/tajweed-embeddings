"""Shared text normalizations for Quranic input."""
from __future__ import annotations

import re

# Strip tatweel placeholders that now precede superscript alef or maddah above
# in the refreshed Tanzil text, which otherwise yields duplicate alef glyphs.
_TATWEEL_BEFORE_DAGGER = re.compile("\u0640(?=\u0670)")
_TATWEEL_BEFORE_MADDAH = re.compile("\u0640(?=\u0653)")
# Collapse decomposed alif+maddah to precomposed alif maddah.
_ALIF_MADDAH_DECOMP = re.compile("\u0627\u0653")
_MADDAH = "\u0653"
_PLACEHOLDER = "\u200d"  # zero-width joiner to preserve indices without adding letters


def normalize_superscript_alef(text: str) -> str:
    """
    Remove tatweel when it only serves as a carrier for dagger alif.
    """
    if not text:
        return text
    text = _TATWEEL_BEFORE_DAGGER.sub("", text)
    text = _TATWEEL_BEFORE_MADDAH.sub("", text)
    # Preserve maddah mark; keep decomposed alif+maddah as-is.
    return text


__all__ = ["normalize_superscript_alef"]
