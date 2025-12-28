"""Tajwīd rule spans, pause encoding, and helpers."""
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals,duplicate-code
from __future__ import annotations

from typing import Dict, List, Tuple

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
        marker_chars: set[str],
        pause_chars: set[str],
    ):
        self.char_aliases = char_aliases
        self.letters = set(letters)
        self.diacritic_chars = diacritic_chars
        self.marker_chars = marker_chars
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
        self.pause_categories: Dict[int, str] = {
            0: "do_not_stop",
            1: "word_boundary_emergency",
            2: "seli",
            3: "jaiz",
            4: "taanoq",
            5: "qeli_or_ayah_end",
            6: "sakta",
            7: "lazem",
        }
        self.pause_category_symbol: Dict[int, str] = {
            0: "-",
            1: "!",
            2: "↦",
            3: "≈",
            4: "⋀",
            5: "⏹",
            6: "˽",
            7: "⛔",
        }
        self.pause_char_category: Dict[str, int] = {
            "ۖ": 2,  # Seli (continue preferred)
            "ۗ": 5,  # Qeli (stop preferred)
            "ۚ": 3,  # Jaiz (optional)
            "ۛ": 4,  # Taanoq (paired dots)
            "ۘ": 7,  # Lazem (mandatory)
            "ۜ": 6,  # Sakta (brief mandatory pause)
            "ۙ": 0,  # Mamnoo (do not stop)
        }
        # Default pause (inside a word): do not stop.
        self.pause_default: np.ndarray = self.encode_pause_bits(0)
        # Word boundary / emergency stop when no explicit pause mark is present.
        self.word_boundary_pause: np.ndarray = self.encode_pause_bits(1)

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
        norm_len = len(chars)

        # First pass: map normalized text indices → filtered indices.
        norm_to_filtered: List[int] = [-1] * norm_len
        filtered_len = 0
        for i, ch in enumerate(chars):
            norm_ch = self.char_aliases.get(ch, ch)
            if norm_ch == "آ":
                # Match TajweedEmbedder behavior: treat alif maddah as a
                # regular alif for indexing, so rule flags stay aligned
                # with the embedding sequence.
                norm_ch = "ا"
            if norm_ch in self.letters:
                norm_to_filtered[i] = filtered_len
                filtered_len += 1
            elif i > 0 and (
                norm_ch in self.diacritic_chars
                or norm_ch in self.marker_chars
                or norm_ch in self.pause_chars
            ):
                norm_to_filtered[i] = norm_to_filtered[i - 1]

        # Build a mapping for the original classifier indices (which were produced
        # before collapsing the decomposed maddah sequence \"آ\" into \"آ\").
        # Each \"آ\" contributes two original codepoints, both of which should map
        # to the same filtered index.
        orig_to_norm: List[int] = []
        for norm_idx, ch in enumerate(chars):
            orig_to_norm.append(norm_idx)
            if ch == "آ":
                orig_to_norm.append(norm_idx)

        raw_to_filtered: List[int] = [
            norm_to_filtered[norm_idx] if 0 <= norm_idx < norm_len else -1
            for norm_idx in orig_to_norm
        ]
        raw_len = len(raw_to_filtered)

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

        # Fallback: mark silent when an alif/ya is followed (optionally via spaces) by hamzat wasl.
        silent_idx = self.rule_to_index.get("silent")
        if silent_idx is not None:
            for i, ch in enumerate(chars):
                if ch not in {"ا", "ى"}:
                    continue
                j = i + 1
                while j < norm_len and chars[j].isspace():
                    j += 1
                if j < norm_len and chars[j] == "ٱ":
                    f_idx_norm = norm_to_filtered[i]
                    if 0 <= f_idx_norm < filtered_len:
                        flags[f_idx_norm][silent_idx] = 1.0

        return flags
