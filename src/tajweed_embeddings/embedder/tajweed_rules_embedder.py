from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class TajweedRulesEmbedder:
    """Rule spans, pause categories, and helpers."""

    def __init__(
        self,
        rules: List[Dict],
        marker_rule_map: Dict[str, str],
        char_aliases: Dict[str, str],
        letters: List[str],
        diacritic_chars: set[str],
        pause_chars: set[str],
    ):
        self.char_aliases = char_aliases
        self.letters = set(letters)
        self.diacritic_chars = diacritic_chars
        self.pause_chars = pause_chars

        rule_names_set = set()
        for entry in rules:
            anns = entry.get("annotations", [])
            for ann in anns:
                rn = ann.get("rule")
                if rn:
                    rule_names_set.add(rn)
        rule_names_set.update(marker_rule_map.values())
        self.rule_names: List[str] = sorted(rule_names_set)
        self.n_rules: int = len(self.rule_names)
        self.rule_to_index: Dict[str, int] = {
            name: i for i, name in enumerate(self.rule_names)
        }

        self.rules_index: Dict[Tuple[str, str], List[Dict]] = {}
        for entry in rules:
            sura = str(entry.get("surah") or entry.get("sura"))
            ayah = str(entry.get("ayah"))
            key = (sura, ayah)
            anns = entry.get("annotations", [])
            if not isinstance(anns, list):
                continue
            self.rules_index.setdefault(key, []).extend(anns)

        # Pause slice: 3-bit code for stop categories
        self.n_pause: int = 3
        self.pause_default: np.ndarray = np.zeros(self.n_pause, dtype=float)
        self.pause_categories: Dict[int, str] = {
            0: "do_not_stop",
            1: "seli",
            2: "jaiz",
            3: "taanoq",
            4: "qeli_or_ayah_end",
            5: "sakta",
            6: "lazem",
        }
        self.pause_category_symbol: Dict[int, str] = {
            0: "-",
            1: "↦",
            2: "≈",
            3: "⋀",
            4: "⏹",
            5: "˽",
            6: "⛔",
        }
        self.pause_char_category: Dict[str, int] = {
            "ۖ": 1,  # Seli (continue preferred)
            "ۗ": 4,  # Qeli (stop preferred)
            "ۚ": 2,  # Jaiz (optional)
            "ۛ": 3,  # Taanoq (paired dots)
            "ۘ": 6,  # Lazem (mandatory)
            "ۜ": 5,  # Sakta (brief mandatory pause)
            "ۙ": 0,  # Mamnoo (do not stop)
        }

    # ------------------------------------------------------------------
    def encode_pause_bits(self, category: int) -> np.ndarray:
        """Encode pause category (0-7) into 3-bit binary vector."""
        category = max(0, min(7, int(category)))
        return np.array(
            [(category >> 0) & 1, (category >> 1) & 1, (category >> 2) & 1],
            dtype=float,
        )

    def pause_vector(self, ch: str) -> np.ndarray:
        """Return pause slice 3-bit code for a pause glyph."""
        category = self.pause_char_category.get(ch, 0)
        return self.encode_pause_bits(category)

    # ------------------------------------------------------------------
    def apply_rule_spans(self, sura, ayah, chars: List[str]) -> List[np.ndarray]:
        """
        Returns rule flags aligned to the filtered character sequence (letters only).
        Each element is a rule one-hot vector of length `n_rules` for that filtered index.
        """
        raw_len = len(chars)

        raw_to_filtered: List[int] = [-1] * raw_len
        filtered_len = 0
        for i, ch in enumerate(chars):
            norm_ch = self.char_aliases.get(ch, ch)
            if norm_ch in self.letters:
                raw_to_filtered[i] = filtered_len
                filtered_len += 1
            elif i > 0 and (
                norm_ch in self.diacritic_chars or norm_ch in self.pause_chars
            ):
                raw_to_filtered[i] = raw_to_filtered[i - 1]

        flags = [np.zeros(self.n_rules, dtype=float) for _ in range(filtered_len)]

        if self.n_rules == 0 or filtered_len == 0:
            return flags

        key = (str(sura), str(ayah))
        anns = self.rules_index.get(key, [])

        for ann in anns:
            rule_name = ann.get("rule")
            if rule_name not in self.rule_to_index:
                continue
            idx_rule = self.rule_to_index[rule_name]

            start = int(ann.get("start", 0))
            end = int(ann.get("end", 0))

            start = max(0, start)
            end = min(raw_len, end)

            for raw_idx in range(start, end):
                f_idx = raw_to_filtered[raw_idx]
                if f_idx < 0 or f_idx >= filtered_len:
                    continue
                flags[f_idx][idx_rule] = 1.0

        return flags
"""Tajwīd rule spans, pause encoding, and helpers."""
