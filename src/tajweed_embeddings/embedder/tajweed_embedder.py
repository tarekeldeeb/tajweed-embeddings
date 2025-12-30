"""Tajweed embedding module for Quranic text."""
# pylint: disable=too-many-instance-attributes,too-many-statements,too-many-branches,too-many-locals,duplicate-code

import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tajweed_embeddings.util import download_quran_txt
from tajweed_embeddings.util.normalization import normalize_superscript_alef

from .harakat_embedder import HarakatEmbedder
from .letters_embedder import LettersEmbedder
from .sifat_embedder import SifatEmbedder
from .sifat_editor import SifatEditor
from .tajweed_rules_embedder import TajweedRulesEmbedder


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

        # Auto-bootstrap missing data files if possible.
        self._ensure_data_ready()

        # Load raw JSONs
        self.quran = self._load_json("quran.json")
        self.sifat = self._load_json("sifat.json")
        self.rules = self._load_json("tajweed.rules.json")

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
        self.marker_chars: set[str] = set(self.marker_rule_map.keys())
        # Character aliases: map stylistic glyphs to canonical letters
        self.char_aliases: Dict[str, str] = {
            "ۨ": "ن",  # small high noon
            "ۧ": "ۦ",  # map variant small yeh to canonical small yeh glyph
            "ۭ": "ۢ",  # small low meem -> iqlab indicator
        }

        # Component helpers
        self.haraka_helper = HarakatEmbedder()
        diacritic_like = self.haraka_helper.diacritic_chars | self.marker_chars
        self.letters_helper = LettersEmbedder(
            self.sifat, self.pause_chars, diacritic_like
        )
        self.sifat_embedder = SifatEmbedder()
        self.isti_la_letters = {
            ch
            for ch, entry in self.sifat.items()
            if isinstance(entry, dict)
            and entry.get("sifat", {}).get("isti'la", 0)
        }
        self.sifat_editor = SifatEditor(self.haraka_helper, self.isti_la_letters)
        self.tajweed_rules = TajweedRulesEmbedder(
            self.rules,
            self.marker_rule_map,
            self.char_aliases,
            self.letters_helper.letters,
            self.haraka_helper.diacritic_chars,
            self.marker_chars,
            self.pause_chars,
        )

        # Expose commonly used attributes for backward compatibility
        self.letters = self.letters_helper.letters
        self.letter_to_index = self.letters_helper.letter_to_index
        self.index_to_letter = self.letters_helper.index_to_letter
        self.n_letters = self.letters_helper.n_letters

        self.harakat_states = self.haraka_helper.harakat_states
        self.harakat_state_labels = self.haraka_helper.harakat_state_labels
        self.n_harakat = self.haraka_helper.n_harakat
        self.harakat_vectors = self.haraka_helper.harakat_vectors
        self.index_to_haraka_state = self.haraka_helper.index_to_haraka_state
        self.default_haraka = self.haraka_helper.default_haraka
        self.shadda_char = self.haraka_helper.shadda_char
        self.alt_sukun_char = self.haraka_helper.alt_sukun_char
        self.haraka_base_map = self.haraka_helper.haraka_base_map
        self.diacritic_chars = self.haraka_helper.diacritic_chars
        self.haraka_state_to_chars = self.haraka_helper.haraka_state_to_chars
        self.harakat_chars = self.haraka_helper.harakat_chars
        self.harakat = self.haraka_helper.harakat
        self.long_vowel_defaults = self.haraka_helper.long_vowel_defaults
        self.haraka_symbol = self.haraka_helper.haraka_symbol

        self.n_pause = self.tajweed_rules.n_pause
        self.pause_default = self.tajweed_rules.pause_default
        self.pause_categories = self.tajweed_rules.pause_categories
        self.pause_category_symbol = self.tajweed_rules.pause_category_symbol
        self.pause_char_category = self.tajweed_rules.pause_char_category

        self.n_sifat = self.sifat_embedder.n_sifat

        self.rule_names = self.tajweed_rules.rule_names
        self.n_rules = self.tajweed_rules.n_rules
        self.rule_to_index = self.tajweed_rules.rule_to_index
        self.rules_index = self.tajweed_rules.rules_index

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
    def _load_json(self, name: str):
        """
        Load JSON directly from the on-disk data directory. This avoids stale
        package caches and ensures freshly generated files are picked up.
        """
        path = self._data_dir() / name
        return json.loads(path.read_text(encoding="utf-8"))

    def _ensure_data_ready(self) -> None:
        """
        Ensure packaged data files exist. If missing, regenerate from the Tanzil
        text and classifier outputs on-the-fly.
        """
        # Resolve repo root and key paths.
        data_dir = self._data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        package_root = data_dir.parent  # .../tajweed_embeddings
        rules_gen_dir = package_root / "rules_gen"
        rules_gen_output = rules_gen_dir / "output"
        quran_txt = rules_gen_output / "quran-uthmani.txt"
        quran_json = data_dir / "quran.json"
        rules_json = data_dir / "tajweed.rules.json"

        # Rebuild rules JSON or source text if missing or invalid.
        need_rules = (
            (not rules_json.exists())
            or rules_json.stat().st_size == 0
            or not self._json_type_matches(rules_json, list)
        )
        need_quran_txt = (not quran_txt.exists()) or quran_txt.stat().st_size == 0

        if need_rules or need_quran_txt:
            if not quran_txt.exists() or quran_txt.stat().st_size == 0:
                rules_gen_output.mkdir(parents=True, exist_ok=True)
                download_quran_txt(quran_txt)
            if not quran_txt.exists() or quran_txt.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Missing Quran text at {quran_txt}. "
                "Run src/tajweed_embeddings/rules_gen/tajweed_classifier.py after installing its "
                "dependencies (pip install -r src/tajweed_embeddings/rules_gen/requirements.txt)."
            )
            cmd = [
                sys.executable,
                str(rules_gen_dir / "tajweed_classifier.py"),
                "--json",
                "--output",
                str(rules_json),
            ]
            try:
                # Feed the existing text via stdin.
                with quran_txt.open("rb") as fh:
                    subprocess.run(cmd, cwd=package_root, check=True, stdin=fh)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                raise RuntimeError(
                    "Failed to generate tajweed.rules.json; ensure rules_gen dependencies "
                    "are installed (pip install -r src/tajweed_embeddings/rules_gen/requirements.txt)."
                ) from exc

        # Build quran.json if missing or invalid.
        need_quran_json = (
            (not quran_json.exists())
            or quran_json.stat().st_size == 0
            or not self._json_type_matches(quran_json, dict)
        )
        if need_quran_json:
            if not quran_txt.exists():
                raise FileNotFoundError(
                    f"Missing Quran text at {quran_txt}; cannot build quran.json."
                )
            cmd = [
                sys.executable,
                str(package_root / "util" / "tanzil_to_json.py"),
                "--input",
                str(quran_txt),
                "--output-dir",
                str(data_dir),
                "--output-filename",
                "quran.json",
            ]
            subprocess.run(cmd, cwd=package_root, check=True)

        # Final sanity: raise if still missing/invalid so failures are explicit.
        if not self._json_type_matches(rules_json, list):
            raise FileNotFoundError(f"Could not prepare tajweed rules JSON at {rules_json}")
        if not self._json_type_matches(quran_json, dict):
            raise FileNotFoundError(f"Could not prepare Quran JSON at {quran_json}")

    def _data_dir(self) -> Path:
        """Return path to the local data directory."""
        return Path(__file__).resolve().parent.parent / "data"

    @staticmethod
    def _json_type_matches(path: Path, expected_type: type) -> bool:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return isinstance(data, expected_type)
        except Exception:  # pylint: disable=broad-exception-caught
            return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Apply shared Quran text normalizations."""
        return normalize_superscript_alef(text or "")

    def _normalize_for_match(self, text: str) -> str:
        """Normalize text for substring matching (ignore tashkeel)."""
        text = self._normalize_text(text).replace("آ", "آ")
        text = text.replace("ٱ", "ا")
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def _raw_to_filtered_map(self, chars: List[str]) -> Tuple[List[int], int]:
        """Map raw indices to filtered letter indices for a normalized char list."""
        filtered_len = 0
        raw_to_filtered: List[int] = []
        prev_filtered = -1
        for ch in chars:
            norm_ch = self.char_aliases.get(ch, ch)
            if norm_ch == "آ":
                norm_ch = "ا"
            if norm_ch in self.letters:
                raw_to_filtered.append(filtered_len)
                prev_filtered = filtered_len
                filtered_len += 1
            elif (
                norm_ch in self.diacritic_chars
                or norm_ch in self.marker_chars
                or norm_ch in self.pause_chars
            ):
                raw_to_filtered.append(prev_filtered)
            else:
                raw_to_filtered.append(-1)
                prev_filtered = -1
        return raw_to_filtered, filtered_len

    def _compact_for_match(
        self, chars: List[str], strip_diacritics: bool
    ) -> Tuple[str, List[int]]:
        """Return compacted text and a mapping to raw indices for matching."""
        compact: List[str] = []
        mapping: List[int] = []
        for idx, ch in enumerate(chars):
            if ch.isspace() or ch == "ـ" or ch in self.pause_chars:
                continue
            norm_ch = self.char_aliases.get(ch, ch)
            if strip_diacritics and (
                unicodedata.category(ch) == "Mn"
                or unicodedata.category(norm_ch) == "Mn"
                or ch in self.diacritic_chars
                or norm_ch in self.diacritic_chars
            ):
                continue
            if norm_ch == "ٱ":
                norm_ch = "ا"
            compact.append(norm_ch)
            mapping.append(idx)
        return "".join(compact), mapping

    # ------------------------------------------------------------------
    # HARAKA HELPERS
    # ------------------------------------------------------------------
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

    def _encode_pause_bits(self, category: int) -> np.ndarray:
        """Backwards-compatible wrapper for pause encoding."""
        return self.tajweed_rules.encode_pause_bits(category)

    def _pause_vector(self, ch: str) -> np.ndarray:
        """Backwards-compatible wrapper for pause vector."""
        return self.tajweed_rules.pause_vector(ch)

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
            # Preserve explicit maddah marks if present in subtext
            if "ٓ" in subtext:
                return subtext
            return self._normalize_text(subtext)

        if sura_str not in self.quran:
            raise ValueError(f"Sura {sura_str} not found in quran.json")

        if ayah_str is not None:
            if ayah_str not in self.quran[sura_str]:
                raise ValueError(f"Ayah {sura_str}:{ayah_str} not found in quran.json")
            return self._normalize_text(self.quran[sura_str][ayah_str])

        # Full sūrah: concatenate āyāt in numeric order
        ayat_map = self.quran[sura_str]
        ordered_texts = [ayat_map[k] for k in sorted(ayat_map.keys(), key=int)]
        return self._normalize_text("".join(ordered_texts))

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
                norm_ch in self.diacritic_chars
                or norm_ch in self.marker_chars
                or norm_ch in self.pause_chars
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
        count: int = 1,
    ) -> List[np.ndarray]:
        """
        Create tajwīd embeddings for:

        - Full sūrah:     text_to_embedding(1)
        - Full āyah:      text_to_embedding(1, 1)
        - Custom subtext: text_to_embedding(1, 1, "بِسْمِ")
        - Multiple āyāt:  text_to_embedding(1, 1, count=3)  → āyāt 1-3
        """
        if count <= 0:
            raise ValueError("count must be a positive integer")
        # Fast-path for decomposed alif + maddah
        if subtext == "آ":
            vec = np.zeros(self.embedding_dim, dtype=float)
            vec[self.letter_to_index.get("آ", 0)] = 1.0
            madd_idx = self.harakat_state_labels.index("madd")
            vec[
                self.idx_haraka_start : self.idx_haraka_start + self.n_harakat
            ] = 0.0
            vec[self.idx_haraka_start + madd_idx] = 1.0
            vec[
                self.idx_pause_start : self.idx_pause_start + self.n_pause
            ] = self.pause_default
            vec[
                self.idx_sifat_start : self.idx_sifat_start + self.n_sifat
            ] = self.sifat_embedder.encode(self.sifat.get("آ", {}).get("sifat", {}))
            return [vec]

        def _embed_segment(
            text: str,
            sura_val,
            ayah_val: Optional[int],
            apply_rules: bool,
            add_end_pause: bool,
            rule_flags_override: Optional[List[np.ndarray]] = None,
        ) -> List[np.ndarray]:
            original_text = text
            text = self._normalize_text(text)
            # Collapse decomposed alif+maddah into precomposed glyph to avoid extra vector.
            text = text.replace("آ", "آ")
            chars = list(text)
            filtered_len = 0
            word_last_indices: set[int] = set()
            current_word: List[int] = []
            raw_to_filtered: List[int] = []
            words_info: List[Tuple[List[int], bool, bool]] = []
            word_letter_raws: List[int] = []
            word_has_madd = False
            word_has_non_madd_diacritic = False
            prev_filtered = -1
            for idx, ch in enumerate(chars):
                norm_ch = self.char_aliases.get(ch, ch)
                if norm_ch == "آ":
                    norm_ch = "ا"
                if norm_ch in self.letters:
                    raw_to_filtered.append(filtered_len)
                    prev_filtered = filtered_len
                    current_word.append(filtered_len)
                    word_letter_raws.append(idx)
                    filtered_len += 1
                elif (
                    norm_ch in self.diacritic_chars
                    or norm_ch in self.marker_chars
                    or norm_ch in self.pause_chars
                ):
                    raw_to_filtered.append(prev_filtered)
                    if norm_ch in self.diacritic_chars:
                        if norm_ch == self.shadda_char:
                            word_has_non_madd_diacritic = True
                        else:
                            base = self.haraka_base_map.get(norm_ch)
                            if base == "madd":
                                word_has_madd = True
                            elif base is not None:
                                word_has_non_madd_diacritic = True
                    continue
                else:
                    raw_to_filtered.append(-1)
                    if current_word:
                        word_last_indices.add(current_word[-1])
                        words_info.append(
                            (
                                word_letter_raws,
                                word_has_madd,
                                word_has_non_madd_diacritic,
                            )
                        )
                        current_word = []
                        word_letter_raws = []
                        word_has_madd = False
                        word_has_non_madd_diacritic = False
                    prev_filtered = -1
            if current_word:
                word_last_indices.add(current_word[-1])
                words_info.append(
                    (word_letter_raws, word_has_madd, word_has_non_madd_diacritic)
                )

            # Rule flags
            if rule_flags_override is not None:
                rule_flags = rule_flags_override
                if len(rule_flags) != filtered_len:
                    raise ValueError(
                        "Rule flags length does not match filtered text length "
                        f"({len(rule_flags)} != {filtered_len})."
                    )
            elif apply_rules and ayah_val is not None:
                rule_flags = self.tajweed_rules.apply_rule_spans(
                    sura_val, ayah_val, chars
                )
            else:
                rule_flags = [
                    np.zeros(self.n_rules, dtype=float) for _ in range(filtered_len)
                ]

            embeddings: List[np.ndarray] = []
            last_vec: Optional[np.ndarray] = None
            last_base: Optional[str] = None
            last_has_shadda: bool = False
            filtered_idx = 0
            explicit_pause_indices: set[int] = set()
            filtered_letters: List[str] = []

            for idx_char, ch in enumerate(chars):
                madd_on_letter = False
                if ch == "آ":
                    ch = "ا"
                    madd_on_letter = True
                ch = self.char_aliases.get(ch, ch)
                if ch in self.marker_rule_map:
                    if last_vec is not None:
                        rname = self.marker_rule_map[ch]
                        ri = self.rule_to_index.get(rname)
                        if ri is not None:
                            last_vec[self.idx_rule_start + ri] = 1.0
                    continue
                if ch not in self.letters:
                    if last_vec is not None:
                        if ch in self.diacritic_chars:
                            if ch == self.shadda_char:
                                last_has_shadda = True
                            elif self.haraka_base_map.get(ch) == "madd":
                                # Explicit maddah diacritic sets madd haraka
                                state = "madd"
                                last_base = None
                                last_vec[
                                    self.idx_haraka_start : self.idx_haraka_start
                                    + self.n_harakat
                                ] = self.harakat_vectors.get(
                                    state, self.default_haraka
                                )
                                continue
                            else:
                                last_base = self.haraka_base_map.get(ch, last_base)

                            state = self.haraka_helper.compose_haraka_state(
                                last_base, last_has_shadda
                            )
                            last_vec[
                                self.idx_haraka_start : self.idx_haraka_start
                                + self.n_harakat
                            ] = self.harakat_vectors.get(
                                state, self.default_haraka
                            )
                        elif ch in self.pause_chars:
                            pause_vec = self._pause_vector(ch)
                            last_vec[
                                self.idx_pause_start : self.idx_pause_start
                                + self.n_pause
                            ] = pause_vec
                            # Track that this letter received an explicit pause glyph.
                            if filtered_idx > 0:
                                explicit_pause_indices.add(filtered_idx - 1)
                    continue

                vec = np.zeros(self.embedding_dim, dtype=float)

                letter_idx = self.letter_to_index[ch]
                vec[letter_idx] = 1.0
                last_base = None
                haraka_set = False

                vec[
                    self.idx_haraka_start : self.idx_haraka_start + self.n_harakat
                ] = self.default_haraka
                next_ch = chars[idx_char + 1] if idx_char + 1 < len(chars) else None
                if madd_on_letter or (next_ch and self.haraka_base_map.get(next_ch) == "madd"):
                    vec[
                        self.idx_haraka_start : self.idx_haraka_start
                        + self.n_harakat
                    ] = self.harakat_vectors.get("madd", self.default_haraka)
                    last_base = "madd"
                    haraka_set = True
                vec[
                    self.idx_pause_start : self.idx_pause_start + self.n_pause
                ] = self.pause_default
                last_has_shadda = False

                s_entry = self.sifat.get(ch, {})
                s_dict = s_entry.get("sifat", {}) if isinstance(s_entry, dict) else {}
                vec[
                    self.idx_sifat_start : self.idx_sifat_start + self.n_sifat
                ] = self.sifat_embedder.encode(s_dict)

                if self.n_rules and filtered_idx < len(rule_flags):
                    vec[
                        self.idx_rule_start : self.idx_rule_start + self.n_rules
                    ] = rule_flags[filtered_idx]
                if ch in self.marker_rule_map:
                    rname = self.marker_rule_map[ch]
                    ri = self.rule_to_index.get(rname)
                    if ri is not None:
                        vec[self.idx_rule_start + ri] = 1.0

                embeddings.append(vec)
                last_vec = vec
                filtered_letters.append(ch)
                filtered_idx += 1

            self.sifat_editor.apply(
                embeddings,
                filtered_letters,
                self.idx_haraka_start,
                self.n_harakat,
                self.idx_sifat_start,
                word_last_indices,
                explicit_pause_indices,
            )

            pause_slice = slice(
                self.idx_pause_start, self.idx_pause_start + self.n_pause
            )
            word_boundary_pause = self.tajweed_rules.word_boundary_pause
            for idx, vec in enumerate(embeddings):
                if idx in explicit_pause_indices:
                    continue
                if idx in word_last_indices:
                    # Only promote to word-boundary pause when nothing explicit was set.
                    if np.array_equal(vec[pause_slice], self.pause_default):
                        vec[pause_slice] = word_boundary_pause
                else:
                    vec[pause_slice] = self.pause_default

            if add_end_pause and embeddings:
                embeddings[-1][
                    self.idx_pause_start : self.idx_pause_start + self.n_pause
                ] = self._encode_pause_bits(5)

            # Ensure explicit maddah sequences retain madd haraka even if diacritic was stripped.
            if not embeddings and text:
                embeddings.append(np.zeros(self.embedding_dim, dtype=float))

            return embeddings

        # Determine control path
        if ayah is None and subtext is None:
            sura_key = str(sura)
            if sura_key not in self.quran:
                raise ValueError(f"Sura {sura_key} not found in quran.json")
            ayat_map = self.quran[sura_key]
            embeddings: List[np.ndarray] = []
            for a_num in sorted(ayat_map.keys(), key=int):
                text = ayat_map[a_num]
                embeddings.extend(
                    _embed_segment(text, sura, int(a_num), True, True)
                )
            return embeddings

        # If no ayah provided but custom text, embed as subtext
        if ayah is None:
            text = self.get_text(sura, ayah, subtext)
            return _embed_segment(
                text,
                sura,
                ayah,
                apply_rules=False,
                add_end_pause=False,
            )

        # Embed one or more consecutive ayāt starting at `ayah`.
        if subtext is not None:
            full_text = self.get_text(sura, ayah, None)
            norm_full = self._normalize_text(full_text).replace("آ", "آ")
            norm_sub = self._normalize_for_match(subtext)
            full_chars = list(norm_full)
            sub_chars = list(norm_sub)
            start_raw = end_raw = -1
            full_compact, full_map = self._compact_for_match(
                full_chars, strip_diacritics=True
            )
            sub_compact, _ = self._compact_for_match(
                sub_chars, strip_diacritics=True
            )
            if sub_compact:
                start_compact = full_compact.find(sub_compact)
                if start_compact >= 0:
                    end_compact = start_compact + len(sub_compact)
                    start_raw = full_map[start_compact]
                    end_raw = full_map[end_compact - 1] + 1
            if start_raw < 0:
                # Fall back to embedding without rule alignment when no match is found.
                return _embed_segment(
                    self.get_text(sura, ayah, subtext),
                    sura,
                    ayah,
                    apply_rules=False,
                    add_end_pause=False,
                )
            raw_to_filtered, _ = self._raw_to_filtered_map(full_chars)
            rule_flags_full = self.tajweed_rules.apply_rule_spans(
                sura, ayah, full_chars
            )
            sub_filtered_indices: List[int] = []
            last_idx = None
            for raw_idx in range(start_raw, end_raw):
                ch = full_chars[raw_idx]
                norm_ch = self.char_aliases.get(ch, ch)
                if norm_ch == "آ":
                    norm_ch = "ا"
                if norm_ch in self.letters:
                    f_idx = raw_to_filtered[raw_idx]
                    if f_idx >= 0 and f_idx != last_idx:
                        sub_filtered_indices.append(f_idx)
                        last_idx = f_idx
            rule_flags_sub = [rule_flags_full[i] for i in sub_filtered_indices]
            return _embed_segment(
                self.get_text(sura, ayah, subtext),
                sura,
                ayah,
                apply_rules=False,
                add_end_pause=False,
                rule_flags_override=rule_flags_sub,
            )

        sura_key = str(sura)
        if sura_key not in self.quran:
            raise ValueError(f"Sura {sura_key} not found in quran.json")
        ayat_map = self.quran[sura_key]
        start = int(ayah)
        end = start + count - 1
        for a_num in range(start, end + 1):
            if str(a_num) not in ayat_map:
                raise ValueError(f"Ayah {sura}:{a_num} not found in quran.json")

        embeddings: List[np.ndarray] = []
        for a_num in range(start, end + 1):
            current_text = self.get_text(sura, a_num, subtext if a_num == start else None)
            embeddings.extend(
                _embed_segment(
                    current_text,
                    sura,
                    a_num,
                    apply_rules=True,
                    add_end_pause=(subtext is None),
                )
            )
            if subtext is not None:
                break
        return embeddings

    # ------------------------------------------------------------------
    # EMBEDDING → TEXT (rough reconstruction)
    # ------------------------------------------------------------------
    def embedding_to_text(self, embeddings: List[np.ndarray]) -> str:
        """
        Reconstruct text from embeddings using:
          - letter one-hot
          - haraka one-hot
        Ignores ṣifāt and rule flags. Uses pause bits for word boundaries.
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

            # Pause: insert space for word boundary or ayah end.
            pause_slice = vec[
                self.idx_pause_start : self.idx_pause_start + self.n_pause
            ]
            if pause_slice.size == self.n_pause:
                bits = [int(b) for b in pause_slice[: self.n_pause]]
                pause_category = (bits[0]) | (bits[1] << 1) | (bits[2] << 2)
                if pause_category in (1, 5):
                    chars.append(" ")

        return "".join(chars).rstrip()

    def encoding_to_string(self, encoding, style: str = "short") -> str:
        """
        Render embeddings in either compact ("short") or descriptive ("long") form.

        - short: values only, aligned with fixed column widths when rendering sequences.
        - long : labeled, descriptive output (previous default).
        """
        if style not in ("short", "long"):
            raise ValueError("style must be 'short' or 'long'")

        def _format_parts(items: List[tuple[str, str]]) -> str:
            if not items:
                return ""
            label_width = max(len(label) for label, _ in items)
            return " | ".join(
                f"{label.rjust(label_width)}: {value}" for label, value in items
            )

        def _extract_values(enc) -> dict:
            """Return decoded components for formatting."""
            enc = np.asarray(enc, dtype=float)
            if enc.ndim != 1 or enc.shape[0] != self.embedding_dim:
                raise ValueError("Encoding must be a 1-D vector matching embedding_dim")

            # Letter
            letter_slice = enc[: self.n_letters]
            letter = ""
            if letter_slice.size:
                idx = int(np.argmax(letter_slice))
                letter = self.index_to_letter.get(idx, "")

            # Haraka
            haraka_slice = enc[
                self.idx_haraka_start : self.idx_haraka_start + self.n_harakat
            ]
            haraka_state = None
            haraka_name = ""
            if haraka_slice.size == self.n_harakat and np.max(haraka_slice) > 0:
                idx = int(np.argmax(haraka_slice))
                haraka_state = self.index_to_haraka_state.get(idx)
                if 0 <= idx < len(self.harakat_state_labels):
                    haraka_name = self.harakat_state_labels[idx]
                else:
                    haraka_name = haraka_state or ""

            # Pause
            pause_slice = enc[
                self.idx_pause_start : self.idx_pause_start + self.n_pause
            ]
            pause_category = None
            pause_value = ""
            if pause_slice.size == self.n_pause:
                bits = [int(b) for b in pause_slice[: self.n_pause]]
                pause_category = (bits[0]) | (bits[1] << 1) | (bits[2] << 2)
                if pause_category in self.pause_categories:
                    pause_value = self.pause_category_symbol.get(
                        pause_category, str(pause_category)
                    )
                    if style != "short":
                        pause_value = self.pause_categories[pause_category]
                elif pause_category > 0:
                    pause_value = f"pause_{pause_category}"

            # Sifat
            sifat_slice = enc[
                self.idx_sifat_start : self.idx_sifat_start + self.n_sifat
            ]
            sifat_values: List[str] = []
            if sifat_slice.size == self.n_sifat:
                sifat_values = self.sifat_embedder.decode(sifat_slice)

            # Rules
            rules_slice = enc[self.idx_rule_start :]
            active_rules: List[str] = []
            if rules_slice.size == self.n_rules and self.n_rules > 0:
                active_rules = [
                    self.rule_names[i]
                    for i, value in enumerate(rules_slice)
                    if value > 0
                ]

            return {
                "letter": letter,
                "haraka_state": haraka_state,
                "haraka_name": haraka_name,
                "pause_category": pause_category,
                "pause_value": pause_value,
                "sifat_values": sifat_values,
                "rules": active_rules,
            }

        def _disp_width(txt: str) -> int:
            """Approximate display width treating combining chars as zero-width."""
            if not txt:
                return 0
            return sum(0 if unicodedata.combining(ch) else 1 for ch in txt)

        ltr_start = "\u2066"  # LTR isolate for bidi-stable terminal output.
        ltr_end = "\u2069"
        lrm = "\u200e"

        def _ljust_disp(txt: str, width: int) -> str:
            pad = max(0, width - _disp_width(txt))
            return txt + (" " * pad)

        def _dim(txt: str) -> str:
            return f"\x1b[90m{txt}\x1b[0m" if txt else txt

        def _isolate_cell(txt: str) -> str:
            # Isolate each cell to prevent RTL runs from reordering columns.
            return f"{ltr_start}{txt}{ltr_end}" if txt else ""

        def _render_short(enc) -> tuple[list[str], bool, bool]:
            decoded = _extract_values(enc)
            letter_val = decoded["letter"] or "-"
            if letter_val and unicodedata.combining(letter_val):
                # Prefix a tatweel to give combining marks display width
                letter_val = f"ـ{letter_val}"
            haraka_state = decoded["haraka_state"]
            haraka_name = decoded["haraka_name"]
            if haraka_state or haraka_name:
                key = haraka_state or haraka_name
                haraka_val = self.haraka_symbol.get(key, key)
            else:
                haraka_val = ""
            pause_val = decoded["pause_value"] if decoded["pause_value"] else ""
            sifat_vals = decoded["sifat_values"]
            if sifat_vals:
                sifat_val = " ".join(self.sifat_embedder.short_label(s) for s in sifat_vals)
            else:
                sifat_val = ""
            rules = decoded["rules"]
            rules_val = ", ".join(rules) if rules else ""
            # Always return fixed columns: Letter, Haraka, Pause, Sifat, Rules
            haraka_missing = not (haraka_state or haraka_name)
            has_dim_rule = any(
                r in {"silent", "hamzat_wasl", "lam_shamsiyyah"} for r in rules
            )
            return [letter_val, haraka_val, pause_val, sifat_val, rules_val], haraka_missing, has_dim_rule

        def _render_long(enc) -> str:
            decoded = _extract_values(enc)
            parts: List[tuple[str, str]] = []
            parts.append(("Letter", decoded["letter"] or "(unknown)"))

            haraka_display = decoded["haraka_name"] or decoded["haraka_state"] or "(none)"
            parts.append(("Haraka", haraka_display))

            # If silent, skip remaining details for long style
            if not decoded["haraka_state"] and not decoded["haraka_name"]:
                return _format_parts(parts)

            pause_desc = decoded["pause_value"] or "(none)"
            parts.append(("Pause", pause_desc))

            sifat_vals = decoded["sifat_values"]
            if sifat_vals:
                parts.append(("Sifat", ", ".join(sifat_vals)))

            rules = decoded["rules"]
            if rules:
                parts.append(("Rules", ", ".join(rules)))

            if not decoded["haraka_state"] and not decoded["haraka_name"]:
                parts = [
                    (
                        label,
                        value
                        if label not in ("Pause", "Sifat", "Rules")
                        else f"\x1b[90m{value}\x1b[0m",
                    )
                    for label, value in parts
                ]

            return _format_parts(parts)

        # Sequence handling with alignment for short style
        if isinstance(encoding, (list, tuple)):
            if not encoding:
                raise ValueError("Encoding sequence is empty")
            if style == "short":
                rows_data = [_render_short(item) for item in encoding]
                rows = [r for r, _, _ in rows_data]
                col_widths = [
                    max(1, max(_disp_width(row[i]) for row in rows))
                    for i in range(len(rows[0]))
                ]
                idx_width = len(str(len(rows) - 1))
                lines = []
                for idx, (row, haraka_missing, has_dim_rule) in enumerate(rows_data):
                    padded = [
                        _ljust_disp(val, col_widths[i]) for i, val in enumerate(row)
                    ]
                    padded = [_isolate_cell(val) for val in padded]
                    sep = f" {lrm}|{lrm} "
                    line = f"[{str(idx).rjust(idx_width)}] {lrm}" + sep.join(padded)
                    if has_dim_rule:
                        line = _dim(line)
                    lines.append(line)
                return "\n".join(lines)

            formatted = []
            for idx, item in enumerate(encoding):
                formatted.append(f"[{idx}] {self.encoding_to_string(item, style=style)}")
            return "\n".join(formatted)

        # Single encoding
        if style == "short":
            row, haraka_missing, has_dim_rule = _render_short(encoding)
            rows = [row]
            col_widths = [
                max(1, max(_disp_width(row[i]) for row in rows))
                for i in range(len(rows[0]))
            ]
            idx_width = len(str(len(rows) - 1))
            padded = [_ljust_disp(val, col_widths[i]) for i, val in enumerate(row)]
            padded = [_isolate_cell(val) for val in padded]
            sep = f" {lrm}|{lrm} "
            line = f"[{str(0).rjust(idx_width)}] {lrm}" + sep.join(padded)
            if has_dim_rule:
                line = _dim(line)
            return line
        line = _render_long(encoding)
        decoded = _extract_values(encoding)
        if any(r in {"silent", "hamzat_wasl", "lam_shamsiyyah"} for r in decoded["rules"]):
            line = _dim(line)
        return line

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
