"""Haraka state definitions, symbols, and encode/decode helpers."""
# pylint: disable=too-many-instance-attributes,duplicate-code
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class HarakatEmbedder:
    """Haraka states, vectors, and helpers."""

    def __init__(self):
        self.shadda_char: str = "ّ"
        self.alt_sukun_char: str = "۠"

        # Harakāt: explicit states incl. shadda+vowel combos, tanwīn, and alt sukūn
        self.harakat_states: List[str] = [
            "fatha",
            "fatha_shadda",
            "kasra",
            "kasra_shadda",
            "damma",
            "damma_shadda",
            "sukun",
            "sukun_zero",
            "fathatan",
            "kasratan",
            "dammatan",
            "madd",
        ]
        self.harakat_state_labels: List[str] = [
            "fatha",
            "fatha+shadda",
            "kasra",
            "kasra+shadda",
            "damma",
            "damma+shadda",
            "sukun",
            "sukun_zero",
            "fathatan",
            "kasratan",
            "dammatan",
            "madd",
        ]
        self.n_harakat: int = len(self.harakat_states)
        self.harakat_vectors: Dict[str, np.ndarray] = {
            state: np.eye(self.n_harakat, dtype=float)[i]
            for i, state in enumerate(self.harakat_states)
        }
        self.index_to_haraka_state: Dict[int, str] = dict(
            enumerate(self.harakat_states)
        )
        self.default_haraka: np.ndarray = np.zeros(self.n_harakat, dtype=float)
        self.haraka_base_map: Dict[str, str] = {
            "َ": "fatha",
            "ِ": "kasra",
            "ُ": "damma",
            "ْ": "sukun",
            self.alt_sukun_char: "sukun_zero",
            "۟": "sukun",  # rounded zero mark treated as plain sukun
            "ً": "fathatan",
            "ٍ": "kasratan",
            "ٌ": "dammatan",
            "ٓ": "madd",
            "ﹰ": "fathatan",  # Arabic Fathatan isolated form
            "ﹱ": "fathatan",  # Tatweel with Fathatan above
            "ﹲ": "dammatan",  # Arabic Dammatan isolated form
            "ﹴ": "kasratan",  # Arabic Kasratan isolated form
        }
        self.diacritic_chars = set(self.haraka_base_map.keys()) | {self.shadda_char}
        self.haraka_state_to_chars: Dict[str, List[str]] = {
            "fatha": ["َ"],
            "fatha_shadda": ["ّ", "َ"],
            "kasra": ["ِ"],
            "kasra_shadda": ["ّ", "ِ"],
            "damma": ["ُ"],
            "damma_shadda": ["ّ", "ُ"],
            "sukun": ["ْ"],
            "sukun_zero": [self.alt_sukun_char],
            "fathatan": ["ً"],
            "kasratan": ["ٍ"],
            "dammatan": ["ٌ"],
            "madd": [],
        }
        # Backwards compatibility: list of raw diacritic characters
        self.harakat_chars: List[str] = list(self.diacritic_chars)
        self.harakat: Dict[str, np.ndarray] = {
            ch: self.harakat_vectors.get(
                self.compose_haraka_state(
                    self.haraka_base_map.get(ch), ch == self.shadda_char
                ),
                self.default_haraka,
            )
            for ch in self.harakat_chars
        }
        # Long vowel letters often lack explicit haraka marks but should be voiced
        self.long_vowel_defaults: Dict[str, str] = {
            "ا": "madd",  # alif (including madd ā)
            "آ": "madd",
            "ى": "madd",  # alif maqsurah
            "و": "madd",  # long uu/o
            "ي": "madd",  # long ii/ee
            "ٰ": "madd",  # dagger alif mark treated as madd carrier
        }
        # Concise symbols for display (encoding_to_string)
        self.haraka_symbol: Dict[str, str] = {
            "fatha": "^",
            "fatha_shadda": "ώ",
            "kasra": "‿",
            "kasra_shadda": "ῳ",
            "damma": "و",
            "damma_shadda": "ὠ",
            "sukun": "°",
            "sukun_zero": "0",
            "fathatan": "^^",
            "kasratan": "__",
            "dammatan": "oo",
            "madd": "~",
        }

    # ------------------------------------------------------------------
    # HARAKA HELPERS
    # ------------------------------------------------------------------
    def compose_haraka_state(
        self, base: Optional[str], has_shadda: bool
    ) -> Optional[str]:
        """Map base haraka + shadda flag to the explicit state key."""
        if base in ("fatha", "kasra", "damma"):
            return base + ("_shadda" if has_shadda else "")
        if base in ("sukun", "sukun_zero"):
            return base
        if base in ("fathatan", "kasratan", "dammatan"):
            return base
        return None

    def decode_haraka(self, vec: np.ndarray) -> tuple[Optional[str], bool]:
        """Return (base, has_shadda) from an embedding haraka slice."""
        if vec.size != self.n_harakat or np.max(vec) <= 0:
            return None, False

        idx = int(np.argmax(vec))
        state = self.index_to_haraka_state.get(idx)
        if not state:
            return None, False
        if state.endswith("_shadda"):
            return state.replace("_shadda", ""), True
        if state == "sukun":
            return "sukun", False
        return state, False
