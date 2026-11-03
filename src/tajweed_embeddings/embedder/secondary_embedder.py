"""
Secondary phoneme variants for waqf (stop) and ibtidāʾ (start).

Every word's boundary phonemes get pre-calculated alternative vectors so a
runtime aligner can accept a pause/resume at any word boundary.

Stop (waqf) rules implemented:
  1. Sukūn ʿāriḍ — final short vowel / kasratan / dammatan → sukūn;
     shadda+vowel → sukūn+shadda.
  2. Madd al-ʿiwaḍ — final fathatan → fatha with the seat alif/ʾalif maqṣūra
     voiced as madd_2; with no seat (e.g. سَمَآءً) the final letter itself
     carries the 2-count madd (haraka=madd + madd_2).
  3. Tāʾ marbūṭa → hāʾ sākinah (letter substitution, tanwīn dropped).
  4. Qalqalah kubrā — final ق ط ب ج د flagged qalqalah once sākin.
  5. Madd ʿāriḍ lil-sukūn — madd (or līn) letter before the now-sākin final:
     madd_2 → madd_246. Primary vectors are pure waṣl, so this is the only
     place madd_246 is produced.
  6. Hāʾ ṣilah — small ۥ/ۦ after pronoun hāʾ silenced; hāʾ → sukūn.
  7. Un-silencing — word-final ا/ى silenced only by the next word's hamzat
     waṣl is restored to madd_2 at stop.
  8. Cross-word assimilation undo — ikhfāʾ/idghām/iqlāb flags from spans that
     continue into the next word (or next āyah) are cleared (iẓhār at stop),
     as are word-final ikhfāʾ/iqlāb/munfaṣil flags; a cleared madd_munfasil
     reverts to madd_2.
  9. Rāʾ/lām tafkhīm re-evaluated under the stopped ending.

Start (ibtidāʾ) rules implemented:
  1. Hamzat waṣl vocalization — ٱ → pronounced hamza (أ) with fatha for the
     definite article, otherwise kasra, or damma when the third letter carries
     a permanent damma (with the classical exception lists; a doubled second
     letter is itself the third letter, as in ٱتَّقُوا). A following sākin
     hamza is replaced by the matching madd letter (ibdāl, e.g. ٱئْتُوا → إيتوا).
  2. Cross-word assimilation undo — rule flags from spans that began in the
     previous word are cleared, and the written idghām shadda on the first
     letter is dropped.

Out of scope: rawm and ishmām stop styles (stop variants assume the default
sukūn-ʿāriḍ realization). Ḥafṣ-specific single-word ishmām/tas-hīl markers are
inline in the text and unaffected by pause position.
"""
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-branches,too-many-locals,too-many-statements
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .harakat_embedder import HarakatEmbedder
from .sifat_editor import SifatEditor
from .sifat_embedder import SifatEmbedder


class SecondaryPhonemeEmbedder:
    """Compute waqf/ibtidāʾ variant vectors for word-boundary phonemes."""

    QALQALAH_LETTERS = {"ق", "ط", "ب", "ج", "د"}
    HAMZA_LETTERS = {"ء", "أ", "إ", "ؤ", "ئ"}
    MADD_SEAT_LETTERS = {"ا", "ى"}
    SMALL_SILA_LETTERS = {"ۥ", "ۦ"}
    MADD_LETTERS = {"ا", "ى", "ي", "و", "ٰ", "ۥ", "ۦ"}
    MADD_RULE_NAMES = ("madd_2", "madd_6", "madd_246", "madd_muttasil", "madd_munfasil")
    # Rules whose cross-word clearing leaves a bare nasal that must take sukūn.
    NASAL_ASSIMILATION_RULES = {
        "ikhfa",
        "ikhfa_shafawi",
        "iqlab",
        "idghaam_ghunnah",
        "idghaam_no_ghunnah",
        "idghaam_shafawi",
        "idghaam_mutajanisayn",
        "idghaam_mutaqaribayn",
    }
    # Nasal rules a word-final nūn/tanwīn only gets from the next word.
    WORD_FINAL_NASAL_RULES = ("ikhfa", "ikhfa_shafawi", "ghunnah_tafkheem", "iqlab")
    # Hamzat waṣl: classical samāʿī nouns start with kasra despite letter shape.
    WASL_KASRA_NOUN_PREFIXES = ("ٱسم", "ٱبن", "ٱمرؤ", "ٱمرأ", "ٱثن", "ٱيم")
    # Verbs whose third-letter damma is ʿāriḍah (temporary) → start with kasra.
    # Matched as stems so suffixed forms are covered too (ٱئْتُونِى).
    WASL_DAMMA_EXCEPTIONS = ("ٱمشو", "ٱبنو", "ٱقضو", "ٱئتو")
    # iftaʿala verbs that begin with ٱل but are not the definite article.
    WASL_LAM_VERB_PREFIXES = ("ٱلتق", "ٱلتم")
    # Stopped rāʾ read with tarqīq for the deleted yāʾ (وَٱلَّيْلِ إِذَا يَسْرِ,
    # 89:4), although the vowel before it would give tafkhīm.
    WAQF_TARQEEQ_RAA_WORDS = {"يسر"}

    def __init__(
        self,
        haraka_helper: HarakatEmbedder,
        sifat_embedder: SifatEmbedder,
        sifat_editor: SifatEditor,
        sifat_data: Dict,
        letter_to_index: Dict[str, int],
        rule_to_index: Dict[str, int],
        rule_names: List[str],
        n_letters: int,
        idx_haraka_start: int,
        n_harakat: int,
        idx_sifat_start: int,
        n_sifat: int,
        idx_rule_start: int,
        n_rules: int,
    ):
        self.haraka_helper = haraka_helper
        self.sifat_embedder = sifat_embedder
        self.sifat_editor = sifat_editor
        self.sifat_data = sifat_data
        self.letter_to_index = letter_to_index
        self.rule_to_index = rule_to_index
        self.rule_names = rule_names
        self.n_letters = n_letters
        self.idx_haraka_start = idx_haraka_start
        self.n_harakat = n_harakat
        self.idx_sifat_start = idx_sifat_start
        self.n_sifat = n_sifat
        self.idx_rule_start = idx_rule_start
        self.n_rules = n_rules

    # ------------------------------------------------------------------
    # SLICE HELPERS
    # ------------------------------------------------------------------
    def _haraka_state(self, vec: np.ndarray) -> Optional[str]:
        h_slice = vec[self.idx_haraka_start : self.idx_haraka_start + self.n_harakat]
        if h_slice.size != self.n_harakat or np.max(h_slice) <= 0:
            return None
        return self.haraka_helper.index_to_haraka_state.get(int(np.argmax(h_slice)))

    def _set_haraka(self, vec: np.ndarray, state: Optional[str]) -> None:
        h_slice = slice(self.idx_haraka_start, self.idx_haraka_start + self.n_harakat)
        if state is None:
            vec[h_slice] = self.haraka_helper.default_haraka
        else:
            vec[h_slice] = self.haraka_helper.harakat_vectors[state]

    def _set_letter(self, vec: np.ndarray, letter: str) -> None:
        """Substitute the letter one-hot and re-encode its ṣifāt."""
        vec[: self.n_letters] = 0.0
        vec[self.letter_to_index[letter]] = 1.0
        entry = self.sifat_data.get(letter, {})
        s_dict = entry.get("sifat", {}) if isinstance(entry, dict) else {}
        vec[
            self.idx_sifat_start : self.idx_sifat_start + self.n_sifat
        ] = self.sifat_embedder.encode(s_dict)

    def _is_madd_carrier(
        self,
        vec: np.ndarray,
        letter: str,
        preceding_state: Optional[str],
    ) -> bool:
        """True when a letter is voiced as a natural madd (flagged or implicit)."""
        if self._flag_on(vec, "madd_2") or self._haraka_state(vec) == "madd":
            return True
        if self._haraka_state(vec) is not None:
            return False
        # Bare long-vowel letters (no rule flags available, e.g. subtext mode).
        if letter in {"ا", "ى", "ٰ"}:
            return True
        if letter == "و":
            return preceding_state in ("damma", "damma_shadda")
        if letter == "ي":
            return preceding_state in ("kasra", "kasra_shadda")
        return False

    def _flag_on(self, vec: np.ndarray, rule: str) -> bool:
        idx = self.rule_to_index.get(rule)
        return idx is not None and vec[self.idx_rule_start + idx] > 0

    def _set_flag(self, vec: np.ndarray, rule: str, enabled: bool) -> None:
        idx = self.rule_to_index.get(rule)
        if idx is not None:
            vec[self.idx_rule_start + idx] = 1.0 if enabled else 0.0

    # ------------------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------------------
    def compute(
        self,
        embeddings: Sequence[np.ndarray],
        letters: Sequence[str],
        words: Sequence[Sequence[int]],
        cross_stop: Dict[int, Set[int]],
        cross_start: Dict[int, Set[int]],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Return (stop_variants, start_variants) as {filtered idx: vector}."""
        stop_variants: Dict[int, np.ndarray] = {}
        start_variants: Dict[int, np.ndarray] = {}
        for w_num, idxs in enumerate(words):
            if not idxs:
                continue
            next_word = words[w_num + 1] if w_num + 1 < len(words) else None
            next_first_letter = letters[next_word[0]] if next_word else None
            self._collect(
                stop_variants,
                self._stop_word(embeddings, letters, list(idxs), cross_stop, next_first_letter),
                embeddings,
            )
            self._collect(
                start_variants,
                self._start_word(embeddings, letters, list(idxs), cross_start),
                embeddings,
            )
        return stop_variants, start_variants

    @staticmethod
    def _collect(
        target: Dict[int, np.ndarray],
        pool: Dict[int, np.ndarray],
        embeddings: Sequence[np.ndarray],
    ) -> None:
        for i, vec in pool.items():
            if not np.array_equal(vec, embeddings[i]):
                target[i] = vec

    # ------------------------------------------------------------------
    # STOP (WAQF) VARIANTS
    # ------------------------------------------------------------------
    def _stop_word(
        self,
        embeddings: Sequence[np.ndarray],
        letters: Sequence[str],
        idxs: List[int],
        cross_stop: Dict[int, Set[int]],
        next_first_letter: Optional[str],
    ) -> Dict[int, np.ndarray]:
        pool: Dict[int, np.ndarray] = {}

        def cur(i: int) -> np.ndarray:
            return pool.get(i, embeddings[i])

        def mut(i: int) -> np.ndarray:
            if i not in pool:
                pool[i] = embeddings[i].copy()
            return pool[i]

        # (8) Clear rules whose spans continue into the next word.
        for i in idxs:
            rule_ids = cross_stop.get(i)
            if not rule_ids:
                continue
            for ri in sorted(rule_ids):
                if cur(i)[self.idx_rule_start + ri] <= 0:
                    continue
                vec = mut(i)
                vec[self.idx_rule_start + ri] = 0.0
                name = self.rule_names[ri]
                if name == "madd_munfasil":
                    # Detached madd loses its trigger at stop → natural madd.
                    self._set_flag(vec, "madd_2", True)
                elif (
                    name in self.NASAL_ASSIMILATION_RULES
                    and letters[i] in {"ن", "م"}
                    and self._haraka_state(vec) is None
                ):
                    # Bare assimilated nasal reverts to iẓhār with sukūn.
                    self._set_haraka(vec, "sukun")

        # (8) Word-final ikhfāʾ/iqlāb spans cover only this word's
        # letters, never the next word that triggers them, so clear them on
        # the final letters directly (the letter before a tanwīn-fatḥ seat
        # alif included).
        final_letters = [idxs[-1]]
        if len(idxs) >= 2 and letters[idxs[-1]] in self.MADD_SEAT_LETTERS:
            final_letters.append(idxs[-2])
        for i in final_letters:
            for rule in self.WORD_FINAL_NASAL_RULES:
                if self._flag_on(cur(i), rule):
                    self._set_flag(mut(i), rule, False)
        # Munfaṣil sits on the trailing madd letters (e.g. ـىٰٓ); an inner one
        # (يَٰٓأَيُّهَا) belongs to the word itself and is kept.
        for i in reversed(idxs):
            if letters[i] not in self.MADD_LETTERS:
                break
            if self._flag_on(cur(i), "madd_munfasil"):
                vec = mut(i)
                self._set_flag(vec, "madd_munfasil", False)
                self._set_flag(vec, "madd_2", True)

        last = idxs[-1]
        letter_last = letters[last]
        effective_final = last
        final_letter = letter_last
        state = self._haraka_state(cur(last))

        # (6) Hāʾ ṣilah: small waw/yeh silenced, pronoun hāʾ takes sukūn.
        if (
            letter_last in self.SMALL_SILA_LETTERS
            and len(idxs) >= 2
            and letters[idxs[-2]] == "ه"
        ):
            vec = mut(last)
            self._set_haraka(vec, None)
            for madd_rule in self.MADD_RULE_NAMES:
                self._set_flag(vec, madd_rule, False)
            self._set_flag(vec, "silent", True)
            self._set_haraka(mut(idxs[-2]), "sukun")
            effective_final = idxs[-2]
            final_letter = "ه"
            ended_sakin = True
        # (3) Tāʾ marbūṭa → hāʾ sākinah (any vowel or tanwīn drops).
        elif letter_last == "ة":
            vec = mut(last)
            self._set_letter(vec, "ه")
            self._set_haraka(vec, "sukun")
            final_letter = "ه"
            ended_sakin = True
        # (2) Madd al-ʿiwaḍ with a written seat (عَلِيمًا / هُدًى).
        elif (
            letter_last in self.MADD_SEAT_LETTERS
            and len(idxs) >= 2
            and self._haraka_state(cur(idxs[-2])) == "fathatan"
        ):
            self._set_haraka(mut(idxs[-2]), "fatha")
            vec = mut(last)
            self._set_flag(vec, "silent", False)
            self._set_flag(vec, "madd_2", True)
            ended_sakin = False
        # (2) Madd al-ʿiwaḍ with no seat (سَمَآءً): madd carried by the letter.
        elif state == "fathatan":
            vec = mut(last)
            self._set_haraka(vec, "madd")
            self._set_flag(vec, "madd_2", True)
            ended_sakin = False
        # (7) Un-silence a final ا/ى muted only by the next word's hamzat waṣl.
        elif (
            letter_last in self.MADD_SEAT_LETTERS
            and state is None
            and next_first_letter == "ٱ"
            and self._flag_on(cur(last), "silent")
        ):
            vec = mut(last)
            self._set_flag(vec, "silent", False)
            self._set_flag(vec, "madd_2", True)
            ended_sakin = False
        # (1) Sukūn ʿāriḍ.
        elif state in {"fatha", "kasra", "damma", "kasratan", "dammatan"}:
            self._set_haraka(mut(last), "sukun")
            ended_sakin = True
        elif state in {"fatha_shadda", "kasra_shadda", "damma_shadda"}:
            self._set_haraka(mut(last), "sukun_shadda")
            ended_sakin = True
        else:
            # Already sākin in waṣl, or ends with a madd letter (no change).
            ended_sakin = state in {"sukun", "sukun_zero", "sukun_shadda"}

        if ended_sakin:
            # (4) Qalqalah kubrā on the audible final letter.
            if final_letter in self.QALQALAH_LETTERS and not self._flag_on(
                cur(effective_final), "qalqalah"
            ):
                self._set_flag(mut(effective_final), "qalqalah", True)
            # (5) Madd ʿāriḍ lil-sukūn on the madd letter before the final.
            pos = idxs.index(effective_final)
            if pos > 0:
                prev_i = idxs[pos - 1]
                prev_vec = cur(prev_i)
                before_state = (
                    self._haraka_state(cur(idxs[pos - 2])) if pos > 1 else None
                )
                is_leen = (
                    letters[prev_i] in {"و", "ي"}
                    and self._haraka_state(prev_vec) in ("sukun", "sukun_zero")
                    and before_state == "fatha"
                )
                if (
                    is_leen
                    or self._is_madd_carrier(prev_vec, letters[prev_i], before_state)
                ) and not self._flag_on(prev_vec, "silent"):
                    vec = mut(prev_i)
                    self._set_flag(vec, "madd_2", False)
                    self._set_flag(vec, "madd_246", True)

        # (9) Re-evaluate rāʾ/lām tafkhīm inside the word under the stop.
        if pool:
            self._reapply_sifat(embeddings, letters, idxs, pool)
            skeleton = "".join(letters[i] for i in idxs)
            if letter_last == "ر" and skeleton in self.WAQF_TARQEEQ_RAA_WORDS:
                mut(last)[self.idx_sifat_start + SifatEditor.ISTI_LA_BIT] = 0.0
        return pool

    def _reapply_sifat(
        self,
        embeddings: Sequence[np.ndarray],
        letters: Sequence[str],
        idxs: List[int],
        pool: Dict[int, np.ndarray],
    ) -> None:
        """Run the contextual ṣifāt editor over the word with variants applied."""
        first, last = idxs[0], idxs[-1]
        window = [np.array(pool.get(i, embeddings[i]), copy=True) for i in range(first, last + 1)]
        win_letters = [letters[i] for i in range(first, last + 1)]
        win_last = {len(window) - 1}
        self.sifat_editor.apply(
            window,
            win_letters,
            self.idx_haraka_start,
            self.n_harakat,
            self.idx_sifat_start,
            win_last,
            wasl_embeddings=[embeddings[i] for i in range(first, last + 1)],
        )
        for offset, vec in enumerate(window):
            i = first + offset
            if not np.array_equal(vec, pool.get(i, embeddings[i])):
                pool[i] = vec

    # ------------------------------------------------------------------
    # START (IBTIDĀʾ) VARIANTS
    # ------------------------------------------------------------------
    def _start_word(
        self,
        embeddings: Sequence[np.ndarray],
        letters: Sequence[str],
        idxs: List[int],
        cross_start: Dict[int, Set[int]],
    ) -> Dict[int, np.ndarray]:
        pool: Dict[int, np.ndarray] = {}

        def cur(i: int) -> np.ndarray:
            return pool.get(i, embeddings[i])

        def mut(i: int) -> np.ndarray:
            if i not in pool:
                pool[i] = embeddings[i].copy()
            return pool[i]

        first = idxs[0]

        # (2) Drop rules whose spans began in the previous word.
        for i in idxs:
            rule_ids = cross_start.get(i)
            if not rule_ids:
                continue
            for ri in sorted(rule_ids):
                if cur(i)[self.idx_rule_start + ri] <= 0:
                    continue
                mut(i)[self.idx_rule_start + ri] = 0.0
        # The written idghām shadda on the first letter is not pronounced
        # when starting at the word (هُدًى لِّلْمُتَّقِينَ → لِلْمُتَّقِينَ).
        if first in cross_start:
            state = self._haraka_state(cur(first))
            if state and state.endswith("_shadda"):
                self._set_haraka(mut(first), state[: -len("_shadda")])

        # (1) Hamzat waṣl vocalization.
        if letters[first] == "ٱ":
            haraka = self._hamzat_wasl_haraka(embeddings, letters, idxs)
            vec = mut(first)
            self._set_letter(vec, "أ")
            self._set_haraka(vec, haraka)
            self._set_flag(vec, "hamzat_wasl", False)
            self._set_flag(vec, "silent", False)

            # Ibdāl: a following sākin hamza becomes the matching madd letter.
            if len(idxs) >= 2 and letters[idxs[1]] in self.HAMZA_LETTERS:
                second_state = self._haraka_state(cur(idxs[1]))
                if second_state in (None, "sukun", "sukun_zero"):
                    madd_letter = {"kasra": "ي", "damma": "و", "fatha": "ا"}[haraka]
                    vec2 = mut(idxs[1])
                    self._set_letter(vec2, madd_letter)
                    self._set_haraka(vec2, None)
                    self._set_flag(vec2, "madd_2", True)

        return pool

    def _hamzat_wasl_haraka(
        self,
        embeddings: Sequence[np.ndarray],
        letters: Sequence[str],
        idxs: List[int],
    ) -> str:
        """Choose the vowel for a pronounced hamzat waṣl (fatha/kasra/damma)."""
        skeleton = "".join(letters[i] for i in idxs)
        if skeleton.startswith(self.WASL_KASRA_NOUN_PREFIXES):
            return "kasra"
        if skeleton.startswith(self.WASL_DAMMA_EXCEPTIONS):
            return "kasra"
        if (
            len(idxs) >= 2
            and letters[idxs[1]] == "ل"
            and not skeleton.startswith(self.WASL_LAM_VERB_PREFIXES)
        ):
            return "fatha"
        if len(idxs) >= 2:
            # A doubled second letter (ٱتَّقُوا) is itself the third letter:
            # its second half carries the shadda's vowel.
            second_state = self._haraka_state(embeddings[idxs[1]])
            if second_state and second_state.endswith("_shadda"):
                return "damma" if second_state == "damma_shadda" else "kasra"
        if len(idxs) >= 3:
            third_state = self._haraka_state(embeddings[idxs[2]])
            if third_state in ("damma", "damma_shadda", "dammatan"):
                return "damma"
        return "kasra"
