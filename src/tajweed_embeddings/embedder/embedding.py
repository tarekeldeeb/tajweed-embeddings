"""Embedding result container carrying secondary (waqf/ibtidāʾ) variants."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


class TajweedEmbedding(list):
    """
    Sequence of per-phoneme embedding vectors plus pre-calculated secondary
    variants for every word boundary.

    Behaves exactly like a list of the primary vectors, which encode fully
    continuous recitation (waṣl). Secondary realizations are always
    pre-computed so a runtime aligner can hypothesize a pause/resume at any
    word boundary without re-embedding:

    - ``words``          : per word ``(first_idx, last_idx)`` inclusive indices.
    - ``stop_variants``  : ``{index: vector}`` replacing the primary vector when
      the reciter STOPS (waqf) on the word containing that index. A stop may
      alter more than one trailing vector (e.g. sukūn ʿāriḍ on the final letter
      plus madd ʿāriḍ on the madd letter before it).
    - ``start_variants`` : ``{index: vector}`` replacing the primary vector when
      the reciter STARTS/resumes (ibtidāʾ) at the word containing that index
      (e.g. hamzat waṣl vocalization, dropped cross-word idghām).

    Indices absent from a variant map are pronounced identically in both
    realizations. Variant vectors share the primary embedding dimension.

    Out of scope (documented limitation): rawm and ishmām stop styles are not
    modeled; stop variants encode the default sukūn-ʿāriḍ realization only.
    """

    def __init__(
        self,
        vectors: Iterable[np.ndarray] = (),
        words: Optional[Iterable[Tuple[int, int]]] = None,
        stop_variants: Optional[Dict[int, np.ndarray]] = None,
        start_variants: Optional[Dict[int, np.ndarray]] = None,
    ):
        super().__init__(vectors)
        self.words: List[Tuple[int, int]] = list(words or [])
        self.stop_variants: Dict[int, np.ndarray] = dict(stop_variants or {})
        self.start_variants: Dict[int, np.ndarray] = dict(start_variants or {})

    @property
    def vectors(self) -> List[np.ndarray]:
        """Primary vectors as a plain list."""
        return list(self)

    def extend_segment(self, other: "TajweedEmbedding") -> None:
        """Append another embedding segment, offsetting its indices."""
        offset = len(self)
        self.words.extend((a + offset, b + offset) for a, b in other.words)
        self.stop_variants.update(
            {i + offset: v for i, v in other.stop_variants.items()}
        )
        self.start_variants.update(
            {i + offset: v for i, v in other.start_variants.items()}
        )
        super().extend(other)

    def word_index_of(self, index: int) -> Optional[int]:
        """Return the word number containing a phoneme index, if any."""
        for w, (first, last) in enumerate(self.words):
            if first <= index <= last:
                return w
        return None

    def realize(
        self,
        stop_words: Iterable[int] = (),
        start_words: Optional[Iterable[int]] = None,
    ) -> List[np.ndarray]:
        """
        Materialize one concrete recitation as a plain vector list.

        ``stop_words`` are word numbers the reciter stops on; ``start_words``
        defaults to the word following each stop (pause-then-resume). Pass
        ``start_words`` explicitly for an ibtidāʾ without a preceding stop
        (e.g. the reciter begins mid-āyah).
        """
        stop_set = set(stop_words)
        if start_words is None:
            start_set = {w + 1 for w in stop_set if w + 1 < len(self.words)}
        else:
            start_set = set(start_words)

        out = list(self)
        for w in stop_set:
            first, last = self.words[w]
            for i in range(first, last + 1):
                if i in self.stop_variants:
                    out[i] = self.stop_variants[i]
        for w in start_set:
            first, last = self.words[w]
            for i in range(first, last + 1):
                if i in self.start_variants:
                    out[i] = self.start_variants[i]
        return out
