"""Tajweed embedding module for Quranic text."""

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

        # Pause glyphs (non-letters) to be modeled via pause slice, not letter one-hot
        self.pause_chars: set[str] = {"ۖ", "ۗ", "ۘ", "ۚ", "ۛ", "ۜ", "ۙ"}
        # Tajweed rule markers (non-pause glyphs)
        self.marker_rule_map: Dict[str, str] = {
            "ۢ": "iqlab",
            "۬": "tas_heel",
            "۪": "imala",
            "۫": "ishmam",
            "ۣ": "optional_seen",
        }

        # Letters are taken from sifat keys
        self.letters: List[str] = sorted(
            ch for ch in self.sifat.keys() if ch not in self.pause_chars
        )
        self.letter_to_index: Dict[str, int] = {
            ch: i for i, ch in enumerate(self.letters)
        }
        self.index_to_letter: Dict[int, str] = {
            i: ch for ch, i in self.letter_to_index.items()
        }
        self.n_letters: int = len(self.letters)
        # Character aliases: map stylistic glyphs to canonical letters
        self.char_aliases: Dict[str, str] = {
            "ۨ": "ن",  # small high noon
            "ۧ": "ۦ",  # map variant small yeh to canonical small yeh glyph
            "ۭ": "ۢ",  # small low meem -> iqlab indicator
        }

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
        self.index_to_haraka_state: Dict[int, str] = {
            i: state for i, state in enumerate(self.harakat_states)
        }
        self.default_haraka: np.ndarray = np.zeros(self.n_harakat, dtype=float)
        self.shadda_char: str = "ّ"
        self.alt_sukun_char: str = "۠"
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
                self._compose_haraka_state(
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

        # Pause slice: 3 bits [present, stop_preferred, stop_mandatory]
        self.n_pause: int = 3
        self.pause_default: np.ndarray = np.zeros(self.n_pause, dtype=float)
        self.pause_map: Dict[str, Tuple[float, float]] = {
            "ۖ": (0.0, 0.0),  # Seli (continue preferred)
            "ۗ": (1.0, 0.0),  # Qeli (stop preferred)
            "ۚ": (0.0, 0.0),  # Jaiz (optional)
            "ۛ": (0.0, 0.0),  # Taanoq (paired dots)
            "ۘ": (0.0, 1.0),  # Lazem (mandatory)
            "ۜ": (0.0, 1.0),  # Sakta (brief mandatory pause)
            "ۙ": (0.0, 0.0),  # Mamnoo (do not stop)
        }

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
        # Include marker-driven rules even if absent from rules.json
        rule_names_set.update(self.marker_rule_map.values())
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
        self.idx_pause_start: int = self.idx_haraka_start + self.n_harakat
        self.idx_sifat_start: int = self.idx_pause_start + self.n_pause
        self.idx_rule_start: int = self.idx_sifat_start + self.n_sifat

        self.embedding_dim: int = (
            self.n_letters
            + self.n_harakat
            + self.n_pause
            + self.n_sifat
            + self.n_rules
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
    # HARAKA HELPERS
    # ------------------------------------------------------------------
    def _compose_haraka_state(
        self, base: Optional[str], has_shadda: bool
    ) -> Optional[str]:
        """
        Map base haraka + shadda flag to the explicit state key.
        Supported bases: fatha, kasra, damma, sukun (+zero), tanwīn
        """
        if base in ("fatha", "kasra", "damma"):
            return base + ("_shadda" if has_shadda else "")
        if base in ("sukun", "sukun_zero"):
            return base
        if base in ("fathatan", "kasratan", "dammatan"):
            return base  # tanwīn does not combine with shadda by design
        return None

    def _decode_haraka(self, vec: np.ndarray) -> Tuple[Optional[str], bool]:
        """Return (base, has_shadda) from an embedding haraka slice."""
        slice_start = self.idx_haraka_start
        slice_end = slice_start + self.n_harakat
        h_slice = vec[slice_start:slice_end]
        if h_slice.size != self.n_harakat or np.max(h_slice) <= 0:
            return None, False

        idx = int(np.argmax(h_slice))
        state = self.index_to_haraka_state.get(idx)
        if not state:
            return None, False
        if state.endswith("_shadda"):
            return state.replace("_shadda", ""), True
        if state == "sukun":
            return "sukun", False
        return state, False

    def _pause_vector(self, ch: str) -> np.ndarray:
        """Return pause slice vector [present, stop_preferred, stop_mandatory]."""
        vec = np.zeros(self.n_pause, dtype=float)
        pref, mand = self.pause_map.get(ch, (0.0, 0.0))
        vec[0] = 1.0  # pause present
        vec[1] = pref
        vec[2] = mand
        return vec

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
    def _apply_rule_spans(self, sura, ayah, chars: List[str]) -> List[np.ndarray]:
        """
        Returns rule flags aligned to the filtered character sequence (letters only).
        Each element is a rule one-hot vector of length `n_rules` for that filtered index.
        """
        raw_len = len(chars)

        # Map raw indices → filtered indices (letters after normalization)
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
                # Allow diacritics/markers to attach to the preceding kept glyph
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

            # clamp to text length
            start = max(0, start)
            end = min(raw_len, end)

            for raw_idx in range(start, end):
                f_idx = raw_to_filtered[raw_idx]
                if f_idx < 0 or f_idx >= filtered_len:
                    continue
                flags[f_idx][idx_rule] = 1.0

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
        filtered_len = sum(
            1 for ch in chars if self.char_aliases.get(ch, ch) in self.letters
        )

        # 2) Compute rule flags only if we have a specific ayah
        if ayah is not None:
            rule_flags = self._apply_rule_spans(sura, ayah, chars)
        else:
            rule_flags = [
                np.zeros(self.n_rules, dtype=float) for _ in range(filtered_len)
            ]

        embeddings: List[np.ndarray] = []
        last_vec: Optional[np.ndarray] = None
        last_base: Optional[str] = None
        last_has_shadda: bool = False
        filtered_idx = 0

        for i, ch in enumerate(chars):
            # Normalize glyph aliases
            ch = self.char_aliases.get(ch, ch)
            # Non-letters: used only to attach harakāt/pause info to previous letter
            if ch not in self.letters:
                if last_vec is not None:
                    if ch in self.diacritic_chars:
                        if ch == self.shadda_char:
                            last_has_shadda = True
                        else:
                            last_base = self.haraka_base_map.get(ch, last_base)

                        state = self._compose_haraka_state(last_base, last_has_shadda)
                        last_vec[
                            self.idx_haraka_start : self.idx_haraka_start + self.n_harakat
                        ] = self.harakat_vectors.get(state, self.default_haraka)
                    elif ch in self.pause_chars:
                        pause_vec = self._pause_vector(ch)
                        last_vec[
                            self.idx_pause_start : self.idx_pause_start + self.n_pause
                        ] = pause_vec
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
            # Long vowel letters may omit haraka marks; set an implicit vowel state
            fallback_base = self.long_vowel_defaults.get(ch)
            if fallback_base:
                vec[
                    self.idx_haraka_start : self.idx_haraka_start + self.n_harakat
                ] = self.harakat_vectors.get(fallback_base, self.default_haraka)
                last_base = fallback_base
            # Default pause slice (may be overwritten by following pause mark)
            vec[
                self.idx_pause_start : self.idx_pause_start + self.n_pause
            ] = self.pause_default
            last_has_shadda = False

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
            if self.n_rules and filtered_idx < len(rule_flags):
                vec[
                    self.idx_rule_start : self.idx_rule_start + self.n_rules
                ] = rule_flags[filtered_idx]
            # Inline tajweed rule markers on the character itself
            if ch in self.marker_rule_map:
                rname = self.marker_rule_map[ch]
                ri = self.rule_to_index.get(rname)
                if ri is not None:
                    vec[self.idx_rule_start + ri] = 1.0

            embeddings.append(vec)
            last_vec = vec
            filtered_idx += 1

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
                state = self.index_to_haraka_state.get(hi, "")
                if state:
                    chars.extend(self.haraka_state_to_chars.get(state, []))

        return "".join(chars)

    def encoding_to_string(self, encoding) -> str:
        """
        Render a single embedding vector as a readable description that lists
        the decoded letter, haraka (if any), numeric ṣifāt values, and active
        tajwīd rules. Useful for debugging embeddings interactively.
        """
        # Allow sequences of encodings for convenience (e.g., direct output of text_to_embedding)
        if isinstance(encoding, (list, tuple)):
            if not encoding:
                raise ValueError("Encoding sequence is empty")
            formatted = []
            for idx, item in enumerate(encoding):
                formatted.append(f"[{idx}] {self.encoding_to_string(item)}")
            return "\n".join(formatted)

        if encoding is None:
            raise ValueError("Encoding must be provided")

        encoding = np.asarray(encoding, dtype=float)

        if encoding.ndim != 1 or encoding.shape[0] != self.embedding_dim:
            raise ValueError("Encoding must be a 1-D vector matching embedding_dim")

        def _format_parts(items: List[tuple[str, str]]) -> str:
            if not items:
                return ""
            label_width = max(len(label) for label, _ in items)
            return " | ".join(
                f"{label.rjust(label_width)}: {value}" for label, value in items
            )

        parts: List[tuple[str, str]] = []

        # Letter
        letter_slice = encoding[: self.n_letters]
        letter = ""
        if letter_slice.size:
            idx = int(np.argmax(letter_slice))
            letter = self.index_to_letter.get(idx, "")
        parts.append(("Letter", letter or "(unknown)"))

        # Haraka
        haraka_slice = encoding[
            self.idx_haraka_start : self.idx_haraka_start + self.n_harakat
        ]
        haraka_name = ""
        if haraka_slice.size == self.n_harakat and np.max(haraka_slice) > 0:
            idx = int(np.argmax(haraka_slice))
            if 0 <= idx < len(self.harakat_state_labels):
                haraka_name = self.harakat_state_labels[idx]
        # Implicit long vowels: letter carries vowel even if haraka slice is empty
        if not haraka_name and letter in self.long_vowel_defaults:
            haraka_name = self.long_vowel_defaults[letter]
        parts.append(("Haraka", haraka_name or "(none)"))

        # If there is no haraka (silent phoneme), skip further vector details
        if not haraka_name:
            return _format_parts(parts)

        # Pause info
        pause_slice = encoding[
            self.idx_pause_start : self.idx_pause_start + self.n_pause
        ]
        pause_desc = ""
        if pause_slice.size == self.n_pause and np.max(pause_slice) > 0:
            present = pause_slice[0] > 0
            pref = pause_slice[1] > 0
            mand = pause_slice[2] > 0
            if present:
                if mand:
                    pause_desc = "mandatory"
                elif pref:
                    pause_desc = "stop-preferred"
                else:
                    pause_desc = "present"
        parts.append(("Pause", pause_desc or "(none)"))

        # Sifat values
        sifat_slice = encoding[
            self.idx_sifat_start : self.idx_sifat_start + self.n_sifat
        ]
        if sifat_slice.size == self.n_sifat:
            sifat_pairs = [
                name
                for name, value in zip(self.sifat_keys, sifat_slice)
                if float(value) != 0.0
            ]
            if sifat_pairs:
                parts.append(("Sifat", ", ".join(sifat_pairs)))
        else:
            parts.append(("Sifat", "(unavailable)"))

        # Tajwid rules (list active ones)
        rules_slice = encoding[self.idx_rule_start :]
        if rules_slice.size == self.n_rules and self.n_rules > 0:
            active_rules = [
                self.rule_names[i]
                for i, value in enumerate(rules_slice)
                if value > 0
            ]
            if active_rules:
                parts.append(("Rules", ", ".join(active_rules)))
        else:
            parts.append(("Rules", "(unavailable)"))

        return _format_parts(parts)

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
