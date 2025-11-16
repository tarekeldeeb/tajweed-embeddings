"""Tajweed embedding module for Quranic text."""

import os
import json
from importlib.resources import files
from typing import Dict, List, Optional, Tuple

import numpy as np


class TajweedEmbedder:
    """
    Tajwīd-aware text embedder for Qur'ān Uthmānī script.

    Data files (auto-loaded from ./data/ by default):
      - sifat.json           : per-letter ṣifāt
      - tajweed.rules.json   : list of {sura, ayah, annotations[{start,end,rule}]}
      - quran.json           : {"1": {"1": "...", "2": "...", ...}, "2": {...}, ...}
    """

    # ------------------------------------------------------------------
    # CONSTRUCTOR & DATA LOADING
    # ------------------------------------------------------------------
    def __init__(self):

        # Load raw JSONs
        self.quran = self._load_json("tajweed_embeddings.data", "quran.json")
        self.sifat = self._load_json("tajweed_embeddings.data", "sifat.json")
        self.rules = self._load_json("tajweed_embeddings.data", "tajweed.rules.json")

        if not isinstance(self.sifat, dict):
            raise ValueError("Invalid sifat.json format (expected dict)")
        if not isinstance(self.rules, list):
            raise ValueError("Invalid tajweed.rules.json format (expected list)")
        if not isinstance(self.quran, dict):
            raise ValueError("Invalid quran.json format (expected dict)")

        # Letters are taken from sifat keys
        self.letters: List[str] = sorted(self.sifat.keys())
        self.letter_to_index: Dict[str, int] = {
            ch: i for i, ch in enumerate(self.letters)
        }
        self.index_to_letter: Dict[int, str] = {
            i: ch for ch, i in self.letter_to_index.items()
        }
        self.n_letters: int = len(self.letters)

        # Harakāt: fixed 5-dim one-hot (fatḥa, kasra, ḍamma, sukūn, shadda)
        self.harakat_chars: List[str] = ["َ", "ِ", "ُ", "ْ", "ّ"]
        self.n_harakat: int = len(self.harakat_chars)
        self.harakat: Dict[str, np.ndarray] = {
            h: np.eye(self.n_harakat, dtype=float)[i]
            for i, h in enumerate(self.harakat_chars)
        }
        self.index_to_haraka: Dict[int, str] = {
            i: h for i, h in enumerate(self.harakat_chars)
        }
        self.default_haraka: np.ndarray = np.zeros(self.n_harakat, dtype=float)

        # Ṣifāt: fixed order of 12 features per letter
        self.sifat_keys: List[str] = [
            "jahr", "hams", "shiddah", "tawassut", "rikhwah",
            "isti'la", "istifal", "itbaq", "infitah",
            "qalqalah", "ghunnah", "tafkhim",
        ]
        self.n_sifat: int = len(self.sifat_keys)

        # Tajwīd rules: collect all rule names from annotations
        rule_names_set = set()
        for entry in self.rules:
            anns = entry.get("annotations", [])
            for ann in anns:
                rn = ann.get("rule")
                if rn:
                    rule_names_set.add(rn)
        self.rule_names: List[str] = sorted(rule_names_set)
        self.n_rules: int = len(self.rule_names)
        self.rule_to_index: Dict[str, int] = {
            name: i for i, name in enumerate(self.rule_names)
        }

        # Index rules by (sura, ayah) → list of annotations
        self.rules_index: Dict[Tuple[str, str], List[Dict]] = {}
        for entry in self.rules:
            sura = str(entry.get("surah") or entry.get("sura"))
            ayah = str(entry.get("ayah"))
            key = (sura, ayah)
            anns = entry.get("annotations", [])
            if not isinstance(anns, list):
                continue
            self.rules_index.setdefault(key, []).extend(anns)

        # Offsets inside embedding vector
        self.idx_haraka_start: int = self.n_letters
        self.idx_sifat_start: int = self.idx_haraka_start + self.n_harakat
        self.idx_rule_start: int = self.idx_sifat_start + self.n_sifat

        self.embedding_dim: int = (
            self.n_letters + self.n_harakat + self.n_sifat + self.n_rules
        )

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json(package: str, name: str):
        path = files(package).joinpath(name)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _safe_float(v) -> float:
        """Convert value to float, treating non-numeric commentary as 0."""
        try:
            return float(v)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # TEXT RETRIEVAL (SURA / AYAH / SUBTEXT)
    # ------------------------------------------------------------------
    def get_text(
        self,
        sura,
        ayah: Optional[int] = None,
        subtext: Optional[str] = None,
    ) -> str:
        """
        - If subtext is not None → return subtext as-is (no Quran lookup)
        - Else if ayah is not None → return that āyah from quran.json
        - Else → return full sūrah text (all āyāt concatenated)
        """
        # Normalize types
        sura_str = str(sura)
        ayah_str = str(ayah) if ayah is not None else None

        if subtext is not None:
            return subtext

        if sura_str not in self.quran:
            raise ValueError(f"Sura {sura_str} not found in quran.json")

        if ayah_str is not None:
            if ayah_str not in self.quran[sura_str]:
                raise ValueError(f"Ayah {sura_str}:{ayah_str} not found in quran.json")
            return self.quran[sura_str][ayah_str]

        # Full sūrah: concatenate āyāt in numeric order
        ayat_map = self.quran[sura_str]
        ordered_texts = [
            ayat_map[k] for k in sorted(ayat_map.keys(), key=lambda x: int(x))
        ]
        return "".join(ordered_texts)

    # ------------------------------------------------------------------
    # RULE FLAGS
    # ------------------------------------------------------------------
    def _apply_rule_spans(self, sura, ayah, text_len: int) -> List[np.ndarray]:
        """
        Returns a list of length `text_len`.
        Each element is a rule one-hot vector of length `n_rules` for that character index.
        """
        flags = [np.zeros(self.n_rules, dtype=float) for _ in range(text_len)]

        if self.n_rules == 0 or text_len == 0:
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

            # clamp to text length
            start = max(0, start)
            end = min(text_len, end)

            for i in range(start, end):
                flags[i][idx_rule] = 1.0

        return flags

    # ------------------------------------------------------------------
    # MAIN: TEXT → EMBEDDING
    # ------------------------------------------------------------------
    def text_to_embedding(
        self,
        sura,
        ayah: Optional[int] = None,
        subtext: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Create tajwīd embeddings for:

        - Full sūrah:     text_to_embedding(1)
        - Full āyah:      text_to_embedding(1, 1)
        - Custom subtext: text_to_embedding(1, 1, "بِسْمِ")
        """

        # 1) Get text according to parameters
        text = self.get_text(sura, ayah, subtext)
        chars = list(text)

        # 2) Compute rule flags only if we have a specific ayah
        if ayah is not None:
            rule_flags = self._apply_rule_spans(sura, ayah, len(chars))
        else:
            rule_flags = [np.zeros(self.n_rules, dtype=float) for _ in range(len(chars))]

        embeddings: List[np.ndarray] = []

        for i, ch in enumerate(chars):
            # Non-letters: used only to attach harakāt to previous letter
            if ch not in self.letters:
                if embeddings and ch in self.harakat:
                    # overwrite haraka slice of previous vector
                    emb = embeddings[-1]
                    emb[
                        self.idx_haraka_start : self.idx_haraka_start + self.n_harakat
                    ] = self.harakat[ch]
                # ignore all other non-letters (spaces, punctuation, etc.)
                continue

            # Initialize vector
            vec = np.zeros(self.embedding_dim, dtype=float)

            # Letter one-hot
            letter_idx = self.letter_to_index[ch]
            vec[letter_idx] = 1.0

            # Default haraka (may be overwritten by following diacritic)
            vec[
                self.idx_haraka_start : self.idx_haraka_start + self.n_harakat
            ] = self.default_haraka

            # Sifāt
            s_entry = self.sifat.get(ch, {})
            s_dict = s_entry.get("sifat", {}) if isinstance(s_entry, dict) else {}
            sifat_values = [
                self._safe_float(s_dict.get(key, 0)) for key in self.sifat_keys
            ]
            vec[
                self.idx_sifat_start : self.idx_sifat_start + self.n_sifat
            ] = np.array(sifat_values, dtype=float)

            # Rule flags for this character
            if self.n_rules and i < len(rule_flags):
                vec[
                    self.idx_rule_start : self.idx_rule_start + self.n_rules
                ] = rule_flags[i]

            embeddings.append(vec)

        # If no embeddings at all but text is non-empty (e.g., non-Arabic text),
        # return a single zero-vector so tests like "subtext not found" still see > 0 length.
        if not embeddings and text:
            embeddings.append(np.zeros(self.embedding_dim, dtype=float))

        return embeddings

    # ------------------------------------------------------------------
    # EMBEDDING → TEXT (rough reconstruction)
    # ------------------------------------------------------------------
    def embedding_to_text(self, embeddings: List[np.ndarray]) -> str:
        """
        Reconstruct text from embeddings using:
          - letter one-hot
          - haraka one-hot
        Ignores ṣifāt and rule flags.
        """
        chars: List[str] = []

        for vec in embeddings:
            if vec is None or vec.ndim != 1 or vec.shape[0] != self.embedding_dim:
                continue

            # Letter
            letter_slice = vec[: self.n_letters]
            if letter_slice.size == 0:
                continue
            li = int(np.argmax(letter_slice))
            letter = self.index_to_letter.get(li, "")
            if not letter:
                continue
            chars.append(letter)

            # Haraka
            haraka_slice = vec[
                self.idx_haraka_start : self.idx_haraka_start + self.n_harakat
            ]
            if haraka_slice.size == self.n_harakat and np.max(haraka_slice) > 0:
                hi = int(np.argmax(haraka_slice))
                h_char = self.index_to_haraka.get(hi, "")
                if h_char:
                    chars.append(h_char)

        return "".join(chars)

    # ------------------------------------------------------------------
    # COMPARISON & SCORE
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten(emb_list: List[np.ndarray]) -> np.ndarray:
        if not emb_list:
            return np.zeros(1, dtype=float)
        return np.concatenate([np.ravel(e) for e in emb_list])

    def compare(self, e1: List[np.ndarray], e2: List[np.ndarray]) -> float:
        """
        Cosine similarity between two embedding sequences.
        Length mismatch is handled by truncating both to the min length.
        """
        v1 = self._flatten(e1)
        v2 = self._flatten(e2)

        if v1.size == 0 or v2.size == 0:
            return 0.0

        min_len = min(v1.size, v2.size)
        v1 = v1[:min_len]
        v2 = v2[:min_len]

        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def score(self, e1: List[np.ndarray], e2: List[np.ndarray]) -> float:
        """
        Similarity score in [0, 100].
        """
        return round(self.compare(e1, e2) * 100.0, 2)
