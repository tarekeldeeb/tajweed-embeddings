"""Contextual edits for sifat slices (e.g., lam/raa tafkhim)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Set

import numpy as np

from .harakat_embedder import HarakatEmbedder


class SifatEditor:
    """Apply context-dependent edits to encoded sifat vectors."""

    _ISTI_LA_BIT = 3

    def __init__(self, haraka_helper: HarakatEmbedder, isti_la_letters: Set[str]):
        self.haraka_helper = haraka_helper
        self.isti_la_letters = isti_la_letters

    def apply(
        self,
        embeddings: List[np.ndarray],
        letters: Sequence[str],
        idx_haraka_start: int,
        n_harakat: int,
        idx_sifat_start: int,
        word_last_indices: Set[int],
        explicit_pause_indices: Set[int],
    ) -> None:
        """Mutate sifat slices in-place based on contextual rules."""
        if not embeddings:
            return

        def haraka_at(idx: int) -> tuple[Optional[str], bool]:
            vec = embeddings[idx][idx_haraka_start : idx_haraka_start + n_harakat]
            return self.haraka_helper.decode_haraka(vec)

        def is_fatha(base: Optional[str]) -> bool:
            return base in ("fatha", "fathatan")

        def is_damma(base: Optional[str]) -> bool:
            return base in ("damma", "dammatan")

        def is_kasra(base: Optional[str]) -> bool:
            return base in ("kasra", "kasratan")

        def is_sukun(base: Optional[str]) -> bool:
            return base in ("sukun", "sukun_zero")

        def set_isti_la(idx: int, enabled: bool) -> None:
            embeddings[idx][idx_sifat_start + self._ISTI_LA_BIT] = 1.0 if enabled else 0.0

        def prev_haraka_value(start_idx: int) -> Optional[str]:
            for j in range(start_idx, -1, -1):
                base, _ = haraka_at(j)
                if base in (
                    "fatha",
                    "fathatan",
                    "damma",
                    "dammatan",
                    "kasra",
                    "kasratan",
                ):
                    return base
            return None

        for i, letter in enumerate(letters):
            # Lam of "Allah": tafkhim after fatha/damma, tarqiq after kasra.
            if letter == "ل":
                base, has_shadda = haraka_at(i)
                if has_shadda and i not in word_last_indices and letters[i + 1] == "ه":
                    prev_base = prev_haraka_value(i - 1)
                    if is_fatha(prev_base) or is_damma(prev_base):
                        set_isti_la(i, True)
                    elif is_kasra(prev_base):
                        set_isti_la(i, False)
                continue

            if letter != "ر":
                continue

            base, _ = haraka_at(i)
            if is_fatha(base) or is_damma(base):
                set_isti_la(i, True)
                continue
            if is_kasra(base):
                set_isti_la(i, False)
                continue

            if not is_sukun(base):
                continue

            prev_base, _ = haraka_at(i - 1) if i > 0 else (None, False)
            prev_letter = letters[i - 1] if i > 0 else None

            if is_fatha(prev_base) or is_damma(prev_base):
                set_isti_la(i, True)
                continue

            if prev_letter == "ي" and is_sukun(prev_base):
                    set_isti_la(i, False)
                    continue

            next_in_word = None
            if i not in word_last_indices:
                next_in_word = i + 1

            if is_kasra(prev_base):
                if next_in_word is not None:
                    next_letter = letters[next_in_word]
                    next_base, _ = haraka_at(next_in_word)
                    if next_letter in self.isti_la_letters and not is_kasra(next_base):
                        set_isti_la(i, True)
                        continue
                if (i in word_last_indices) and (
                    i in explicit_pause_indices or i == len(embeddings) - 1
                ):
                    set_isti_la(i, True)
                    continue
                set_isti_la(i, False)
                continue

            # Default: keep existing istifal unless a stronger rule matched.
            set_isti_la(i, False)
