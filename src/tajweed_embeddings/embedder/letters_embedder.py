"""Letter inventory and indexing derived from sifat definitions."""
# pylint: disable=too-few-public-methods
from __future__ import annotations

from typing import Dict, List, Set


class LettersEmbedder:
    """Manage letters/indices derived from sifat definitions."""

    def __init__(
        self,
        sifat: Dict,
        pause_chars: Set[str],
        diacritic_chars: Set[str] | None = None,
    ):
        diacritic_chars = diacritic_chars or set()
        self.letters: List[str] = sorted(
            ch
            for ch in sifat.keys()
            if ch not in pause_chars and ch not in diacritic_chars
        )
        self.letter_to_index: Dict[str, int] = {
            ch: i for i, ch in enumerate(self.letters)
        }
        self.index_to_letter: Dict[int, str] = {
            i: ch for ch, i in self.letter_to_index.items()
        }
        self.n_letters: int = len(self.letters)
