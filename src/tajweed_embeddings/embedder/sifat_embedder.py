"""Compact ṣifāt encoder/decoder (6-bit representation)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class SifatEmbedder:
    """Encode/decode ṣifāt using compact bits."""

    def __init__(self):
        # jahr/hams (1), strength trio (2), isti'la/istifal (1), infitah/itbaq (1), idhlaq/ismat (1)
        self.n_sifat: int = 6
        self.short_label_map = {
            "jahr": "🔊",
            "hams": "🤫",
            "rikhwah": "💨",
            "tawassut": "➖",
            "shiddah": "🚫",
            "isti'la": "🔼",
            "istifal": "🔻",
            "infitah": "⟂",
            "itbaq": "▲",
            "idhlaq": "😮",
            "ismat": "😐",
        }

    @staticmethod
    def _safe_float(v) -> float:
        """Convert to float, defaulting to 0.0 on type/parse errors."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def encode(self, s_dict: Dict[str, float]) -> np.ndarray:
        """Encode raw sifat dict into a 6-bit vector."""
        vec = np.zeros(self.n_sifat, dtype=float)
        pos = 0

        # jahr vs hams
        jahr = self._safe_float(s_dict.get("jahr", 0))
        vec[pos] = 1.0 if jahr > 0 else 0.0
        pos += 1

        # strength: rikhwah / tawassut / shiddah
        shiddah = self._safe_float(s_dict.get("shiddah", 0))
        tawassut = self._safe_float(s_dict.get("tawassut", 0))
        rikhwah = self._safe_float(s_dict.get("rikhwah", 0))
        state = 0
        if shiddah > 0:
            state = 2
        elif tawassut > 0:
            state = 1
        elif rikhwah > 0:
            state = 0
        vec[pos] = state & 1
        vec[pos + 1] = (state >> 1) & 1
        pos += 2

        # isti'la vs istifal
        isti = self._safe_float(s_dict.get("isti'la", 0))
        vec[pos] = 1.0 if isti > 0 else 0.0
        pos += 1

        # infitah vs itbaq
        inf = self._safe_float(s_dict.get("infitah", 0))
        vec[pos] = 1.0 if inf > 0 else 0.0
        pos += 1

        # idhlaq vs ismat
        idh = self._safe_float(s_dict.get("idhlaq", 0))
        vec[pos] = 1.0 if idh > 0 else 0.0

        return vec

    def decode(self, slice_vec: np.ndarray) -> List[str]:
        """Decode a 6-bit sifat vector into human-readable labels."""
        if slice_vec.size < self.n_sifat:
            return []
        pos = 0
        labels: List[str] = []

        # jahr / hams
        jahr_bit = slice_vec[pos] > 0
        labels.append("jahr" if jahr_bit else "hams")
        pos += 1

        # strength
        state = int(slice_vec[pos]) | (int(slice_vec[pos + 1]) << 1)
        strength_states = ["rikhwah", "tawassut", "shiddah"]
        if 0 <= state < len(strength_states):
            labels.append(strength_states[state])
        pos += 2

        # isti'la / istifal
        isti_bit = slice_vec[pos] > 0
        labels.append("isti'la" if isti_bit else "istifal")
        pos += 1

        # infitah / itbaq
        inf_bit = slice_vec[pos] > 0
        labels.append("infitah" if inf_bit else "itbaq")
        pos += 1

        # idhlaq / ismat
        idh_bit = slice_vec[pos] > 0
        labels.append("idhlaq" if idh_bit else "ismat")

        return labels

    def short_label(self, label: str) -> str:
        """Return a short, disambiguated label for display."""
        return self.short_label_map.get(label, label[:3])
