import numpy as np
from typing import List, Dict


class TajweedEmbedder:
    """
    Unified Tajwīd embedder.

    Loads:
       - sifat_json: phonetic features per letter
       - rules_json: Tajwīd rule spans per surah/ayah

    Embedding per character:
       [ letter_one_hot
       || harakah_one_hot (5)
       || sifat_vec (12)
       || rule_flags_one_hot (len(rule_names)) ]
    """

    # The canonical rule list (same as your project, stable order)
    RULE_NAMES = [
        "ghunnah",
        "hamzat_wasl",
        "idghaam_ghunnah",
        "idghaam_mutajanisayn",
        "idghaam_mutaqaribayn",
        "idghaam_no_ghunnah",
        "idghaam_shafawi",
        "ikhfa",
        "ikhfa_shafawi",
        "iqlab",
        "lam_shamsiyyah",
        "madd_2",
        "madd_246",
        "madd_6",
        "madd_munfasil",
        "madd_muttasil",
        "qalqalah",
        "silent"
    ]

    def __init__(self, sifat_json: Dict, rules_json: Dict):
        """
        :param sifat_json: per-letter sifat definitions
        :param rules_json: tajweed rules (per surah -> per ayah -> rule -> spans)
        """
        self.sifat = sifat_json
        self.rules = rules_json

        self.letters = list(sifat_json.keys())
        self.letter_to_index = {l: i for i, l in enumerate(self.letters)}

        # Harakat
        self.harakat = {
            "َ": [1, 0, 0, 0, 0],
            "ِ": [0, 1, 0, 0, 0],
            "ُ": [0, 0, 1, 0, 0],
            "ْ": [0, 0, 0, 1, 0],
            "ّ": [0, 0, 0, 0, 1]
        }
        self.default_haraka = [0, 0, 0, 0, 0]

        # Dimensional layout
        self.n_letters = len(self.letters)
        self.n_harakat = 5
        self.n_sifat = 12
        self.n_rules = len(self.RULE_NAMES)

        self.idx_haraka_start = self.n_letters
        self.idx_sifat_start = self.idx_haraka_start + self.n_harakat
        self.idx_rule_start = self.idx_sifat_start + self.n_sifat
        self.embedding_dim = self.idx_rule_start + self.n_rules

    # ---------------------------------------------------------
    # INTERNAL: build rule flags per position
    # ---------------------------------------------------------
    def _compute_rule_flags(self, surah: str, ayah: str, text_len: int):
        """Create a [text_len x n_rules] matrix of flags from JSON spans."""
        flags = np.zeros((text_len, self.n_rules), dtype=float)

        if surah not in self.rules:
            return flags
        if ayah not in self.rules[surah]:
            return flags

        ayah_rules = self.rules[surah][ayah]

        # Example:
        # "ghunnah": [[3,4]]
        for rule_i, rule_name in enumerate(self.RULE_NAMES):
            if rule_name not in ayah_rules:
                continue
            spans = ayah_rules[rule_name]
            for start, end in spans:
                # clamp indices
                start = max(0, start)
                end = min(text_len - 1, end)
                flags[start:end+1, rule_i] = 1.0

        return flags

    # ---------------------------------------------------------
    # 1) TEXT -> EMBEDDING
    # ---------------------------------------------------------
    def text_to_embedding(self, text: str, surah: str, ayah: str) -> List[np.ndarray]:
        """
        Convert Uthmani Qur’an text (NO rule markers) into a sequence of
        Tajwīd embeddings by applying JSON rule spans.
        """
        chars = list(text)
        rule_flags = self._compute_rule_flags(surah, ayah, len(chars))

        embeddings = []

        i = 0
        while i < len(chars):
            ch = chars[i]

            # Base letter or diacritic handling
            if ch not in self.sifat:
                # It may be a harakah
                if embeddings and ch in self.harakat:
                    vec = embeddings[-1]
                    vec[self.idx_haraka_start:self.idx_haraka_start + self.n_harakat] = \
                        np.array(self.harakat[ch])
                # else skip unknown token
                i += 1
                continue

            # Create empty embedding vector
            vec = np.zeros(self.embedding_dim, dtype=float)

            # Letter one-hot
            letter_index = self.letter_to_index[ch]
            vec[letter_index] = 1.0

            # Default harakah (will update if next char is diacritic)
            vec[self.idx_haraka_start:self.idx_haraka_start + self.n_harakat] = \
                np.array(self.default_haraka)

            # Sifat vector
            sifat_vec = []
            s = self.sifat[ch]["sifat"]
            for key in [
                "jahr", "hams", "shiddah", "tawassut", "rikhwah",
                "isti'la", "istifal", "itbaq", "infitah",
                "qalqalah", "ghunnah", "tafkhim"
            ]:
                val = s.get(key, 0)
                if isinstance(val, str):
                    # Some sifat entries in real data are descriptive strings (e.g. conditional tafkhim)
                    val = 0.0
                sifat_vec.append(float(val))
            vec[self.idx_sifat_start:self.idx_sifat_start + self.n_sifat] = \
                np.array(sifat_vec)

            # RULE FLAGS (from JSON)
            vec[self.idx_rule_start:self.idx_rule_start + self.n_rules] = \
                rule_flags[i]

            embeddings.append(vec)
            i += 1

        return embeddings

    # ---------------------------------------------------------
    # 2) EMBEDDING -> TEXT
    # ---------------------------------------------------------
    def embedding_to_text(self, embeddings: List[np.ndarray]) -> str:
        text = ""

        for emb in embeddings:
            letter_slice = emb[:self.n_letters]
            haraka_slice = emb[self.idx_haraka_start:self.idx_haraka_start + self.n_harakat]

            letter = self.letters[int(np.argmax(letter_slice))]
            idx_h = int(np.argmax(haraka_slice))

            haraka = ""
            if idx_h == 0: haraka = "َ"
            elif idx_h == 1: haraka = "ِ"
            elif idx_h == 2: haraka = "ُ"
            elif idx_h == 3: haraka = "ْ"
            elif idx_h == 4: haraka = "ّ"

            text += letter + haraka

        return text

    # ---------------------------------------------------------
    # 3) COMPARE
    # ---------------------------------------------------------
    @staticmethod
    def _cosine(a, b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 1))

    def compare(self, e1, e2):
        if not e1 and not e2:
            return 1.0

        max_len = max(len(e1), len(e2))
        pad = lambda arr, target: arr + [np.zeros_like(arr[0])]*(target - len(arr))

        if len(e1) < max_len: e1 = pad(e1, max_len)
        if len(e2) < max_len: e2 = pad(e2, max_len)

        sims = [self._cosine(a, b) for a, b in zip(e1, e2)]
        return float(sum(sims) / len(sims))

    # ---------------------------------------------------------
    # 4) SCORE
    # ---------------------------------------------------------
    def score(self, ref, user):
        return round(self.compare(ref, user) * 100.0, 2)