#!/bin/python3

from collections import deque, namedtuple
import json
import multiprocessing
try:
    import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    class _TqdmFallback:
        @staticmethod
        def tqdm(iterable, total=None):
            return iterable
    tqdm = _TqdmFallback()
import os
import sys
import unicodedata
import argparse
import importlib.util
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = BASE_DIR.parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

RULE_TREES_DIR = BASE_DIR / "rule_trees"

try:
    from .tree import Exemplar, json2tree
except ImportError:  # pragma: no cover - fallback for direct script invocation
    from tree import Exemplar, json2tree

try:
    from tajweed_embeddings.util.normalization import normalize_superscript_alef
    from tajweed_embeddings.util.quran_download import (
        DEFAULT_TANZIL_URLS,
        download_quran_txt,
    )
except Exception:  # pragma: no cover - avoid heavy imports in minimal envs
    def _load_util_module(module_name: str):
        module_path = SRC_ROOT / "tajweed_embeddings" / "util" / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(
            f"tajweed_embeddings.util.{module_name}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    _norm = _load_util_module("normalization")
    normalize_superscript_alef = _norm.normalize_superscript_alef
    _qd = _load_util_module("quran_download")
    DEFAULT_TANZIL_URLS = _qd.DEFAULT_TANZIL_URLS
    download_quran_txt = _qd.download_quran_txt

RangeAttributes = namedtuple("Attributes", "start end attributes")

RULE2E = {
    'ghunnah'               : 'g',
    'hamzat_wasl'           : 'h',
    'idghaam_ghunnah'       : 'd',
    'idghaam_mutajanisayn'  : 'j',
    'idghaam_mutaqaribayn'  : 'b',
    'idghaam_no_ghunnah'    : 'n',
    'idghaam_shafawi'       : 'f',
    'ikhfa'                 : 'k',
    'ikhfa_shafawi'         : 'w',
    'iqlab'                 : 'i',
    'lam_shamsiyyah'        : 'l',
    'madd_2'                : 'o',
    'madd_246'              : 'm',
    'madd_6'                : 'x',
    'madd_munfasil'         : 's',
    'madd_muttasil'         : 't',
    'qalqalah'              : 'q',
    'silent'                : 'e',
    'ghunnah_tafkheem'      : 'p'
}

class embedding:
    pass

def embed(o):
    etext = o.text
    extra = 0
    rules = set([])
    for a in o.annotations:
        etext = etext[ :a["start"]+1+extra ] + RULE2E[a["rule"]] +  etext[ a["start"]+1+extra: ]
        extra += 1
    return f'{o.surah}|{o.ayah}|{etext}'

def attributes_for(rule, txt, i, include_this=True, auxiliary_stream=None):
    # Determine bounds of this letter.
    start_i = i
    while start_i and unicodedata.category(txt[start_i]) == "Mn" and (txt[start_i] != "ٰ" or txt[start_i - 1] == "ـ"):
        start_i -= 1
    end_i = start_i + 1
    while end_i < len(txt) and unicodedata.category(txt[end_i]) == "Mn" and txt[end_i] != "ٰ":
        end_i += 1

    c = txt[i]
    c_ext = txt[start_i:end_i]
    c_base = txt[start_i]

    # Build attributes dict.
    res = {}
    if auxiliary_stream:
        res.update(auxiliary_stream[i])

    if rule == "ghunnah":
        if not include_this:
            res.update({
                "base_is_heavy": c_base in "هءحعخغ",
                "base_is_noon_or_meem": c_base == "ن" or c_base == "م",
                "has_shaddah": "ّ" in c_ext,
                "has_tanween": any(s in c_ext for s in "ًٌٍ"),
            })
        if include_this:
            res.update({
                "is_noon_or_meem": c == "ن" or c == "م",
                "is_initial": i - 1 < 0 or txt[i - 1] == " ",
            })
    elif rule == "hamzat_wasl":
        if include_this:
            res.update({
                "is_alif_waslah": c == "ٱ",
            })
    elif rule == "idghaam_ghunnah":
        if not include_this:
            res.update({
                "base_is_idghaam_ghunna_set": c_base in "يمون",
                "has_implicit_sukoon": not any(s in c_ext for s in "ًٌٍَُِْ"),
            })
        if include_this:
            res.update({
                "is_noon": c == "ن",
                "is_tanween": any(s == c for s in "ًٌٍ"),
                "is_final": end_i >= len(txt) or txt[end_i] == " ",
            })
    elif rule == "idghaam_mutajanisayn":
        if not include_this:
            res.update({
                "base_is_nateeyah_a": c_base in "تط",
                "base_is_nateeyah_b": c_base in "تد",
                "base_is_lathaweeyah_a": c_base in "ثذ",
                "base_is_lathaweeyah_b": c_base in "ظذ",
                "base_is_meem": c_base == "م",
                "base_is_noon": c_base == "ب",
                "has_implicit_sukoon": not any(s in c_ext for s in "ًٌٍَُِْ"),
            })
    elif rule == "idghaam_mutaqaribayn":
        if not include_this:
            res.update({
                "has_implicit_sukoon": not any(s in c_ext for s in "ًٌٍَُِْ"),
                "base_is_qaf_kaf": c_base in "كق",
                "base_is_lam": c_base == "ل",
                "base_is_rah": c_base == "ر",
            })
    elif rule == "idghaam_no_ghunnah":
        if not include_this:
            res.update({
                "has_implicit_sukoon": not any(s in c_ext for s in "ًٌٍَُِْ"),
                "base_is_noon_rah": c_base in "لر",
            })
        if include_this:
            res.update({
                "is_noon": c == "ن",
                "is_tanween": any(s == c for s in "ًٌٍ"),
                "is_final": end_i >= len(txt) or txt[end_i] == " ",
            })
    elif rule == "idghaam_shafawi":
        if not include_this:
            res.update({
                "has_implicit_sukoon": not any(s in c_ext for s in "ًٌٍَُِْ"),
                "base_is_meem": c_base == "م",
            })
    elif rule == "ikhfa":
        if not include_this:
            res.update({
                "has_implicit_sukoon": not any(s in c_ext for s in "ًٌٍَُِْ"),
                "base_is_ikhfa_set": c_base in "تثجدذزسشصضطظفقك",
                "base_is_idhar_set": c_base in "ءأؤئإٱآهعحغخ",
            })
        if include_this:
            res.update({
                "is_noon": c == "ن",
                "is_high_noon": c == "ۨ",
                "is_tanween": any(s == c for s in "ًٌٍ"),
                "is_final": end_i >= len(txt) or txt[end_i] == " ",
            })
    elif rule == "ghunnah_tafkheem":
        # Same structure as "ikhfa", but adds a feature for the subset
        # of ikhfa letters that are always heavy (tafkhim) and actually belong to ikhfa:
        # {ص, ض, ط, ق, ظ}
        if not include_this:
            res.update({
                "has_implicit_sukoon": not any(s in c_ext for s in "ًٌٍَُِْ"),
                # keep ikhfa-set feature (optional but useful if you want to reuse ikhfa logic)
                "base_is_ikhfa_set": c_base in "تثجدذزسشصضطظفقك",
                # NEW: the correct intersection subset for your rule
                "base_is_ikhfa_tafkheem_set": c_base in "صضطقظق",
            })
        if include_this:
            res.update({
                "is_noon": c == "ن",
                "is_high_noon": c == "ۨ",
                "is_tanween": any(s == c for s in "ًٌٍ"),
                "is_final": end_i >= len(txt) or txt[end_i] == " ",
            })
    elif rule == "ghunnah_tafkheem":
        # Same structure as "ikhfa", but adds a feature for the subset
        # of ikhfa letters that are always heavy (tafkhim) and actually belong to ikhfa:
        # {ص, ض, ط, ق, ظ}
        if not include_this:
            res.update({
                "has_implicit_sukoon": not any(s in c_ext for s in "ًٌٍَُِْ"),
                # keep ikhfa-set feature (optional but useful if you want to reuse ikhfa logic)
                "base_is_ikhfa_set": c_base in "تثجدذزسشصضطظفقك",
                # NEW: the correct intersection subset for your rule
                "base_is_ikhfa_tafkheem_set": c_base in "صضطقظق",
            })
        if include_this:
            res.update({
                "is_noon": c == "ن",
                "is_high_noon": c == "ۨ",
                "is_tanween": any(s == c for s in "ًٌٍ"),
                "is_final": end_i >= len(txt) or txt[end_i] == " ",
            })        
    elif rule == "ikhfa_shafawi":
        if not include_this:
            res.update({
                "has_implicit_sukoon": not any(s in c_ext for s in "ًٌٍَُِْ"),
                "base_is_meem": c_base == "م",
            })
    elif rule == "iqlab":
        if not include_this:
            res.update({
                "has_tanween": any(s in c_ext for s in "ًٌٍ"),
                "has_small_meem": "ۢ" in c_ext or "ۭ" in c_ext,
            })
        if include_this:
            res.update({
                "is_tanween": c in "ًٌٍ",
                "is_base": (unicodedata.category(c) != "Mn" and c != "ـ") or c == "ٰ",
            })      
    elif rule == "lam_shamsiyyah":
        if not include_this:
            res.update({
                "has_vowel_incl_tanween": any(s in c_ext for s in "ًٌٍَُِْ"),
                "has_shaddah": "ّ" in c_ext,
            })
        if include_this:
            res.update({
                "is_alif_waslah": c == "ٱ",
                "is_lam": c == "ل",
                "is_allah_word_start": txt[start_i:start_i + 7] in ("للَّهِ ", "للَّهُ ", "للَّهَ "),
            })
    elif rule == "madd_2":
        if not include_this:
            res.update({
                "has_maddah": "ٓ" in c_ext,
                "has_hamza": any(s in c_ext for s in "ؤئٕإأٔ"),
                "has_vowel_incl_tanween": any(s in c_ext for s in "ًٌٍَُِْ"),
                "has_proc_sukoon": "۟" in c_ext or "ْ" in c_ext or not any(s in c_ext for s in "ًٌٍَُِْ"),
                "is_final_letter_in_ayah": end_i >= len(txt),
            })
        if include_this:
            res.update({
                "is_dagger_alif": c == "ٰ",
                "is_small_yeh": c == "ۦ",
                "is_small_waw": c == "ۥ",
                "is_long_vowel": c in ("ا", "آ", "ى", "ي", "و", "ٰ"),
                "is_final": end_i >= len(txt) or txt[end_i] == " ",
                })
    elif rule == "madd_246":
        if not include_this:
            res.update({
                "has_maddah": "ٓ" in c_ext,
                "has_fathah": "َ" in c_ext,
                "has_dammah": "ُ" in c_ext,
                "has_kasrah": "ِ" in c_ext,
                "has_vowel_no_tanween": any(s in c_ext for s in "َُِْ"),
                "has_tanween": any(s in c_ext for s in "ًٌٍ"),
            })
        if include_this:
            res.update({
                "is_alif": c == "ا",
                "is_yeh": c == "ي",
                "is_waw": c == "و",
            })
    elif rule == "madd_6":
        if not include_this:
            res.update({
                "has_maddah": "ٓ" in c_ext,
                "has_explicit_sukoon": "۟" in c_ext or "ْ" in c_ext,
                "has_vowel_incl_tanween": any(s in c_ext for s in "ًٌٍَُِْ"),
                "has_shaddah": "ّ" in c_ext,
                "has_hamza": any(s in c_ext for s in "ؤئٕإأٔ"),
                "base_is_alif_maksura": c_base == "ى",
            })
        if include_this:
            res.update({
                "is_hamza": c == "ء",
                "is_base": (unicodedata.category(c) != "Mn" and c != "ـ") or c == "ٰ",
                "is_final": end_i >= len(txt) or txt[end_i] == " ",
            })

    elif rule in ("madd_munfasil", "madd_muttasil"):
        if not include_this:
            res.update({
                "has_maddah": "ٓ" in c_ext,
                "has_explicit_sukoon": "۟" in c_ext or "ْ" in c_ext,
                "has_non_initial_hamza": any(s in c_ext for s in "ؤئٕٔ"),
                "base_is_isolated_hamza": c_base == "ء",
                "has_initial_hamza": any(s in c_ext for s in "ٕإأ"),
                # The following attributes permit this to work without inspecting for maddah(?):
                # "has_implicit_sukoon": not any(s in c_ext for s in "ًٌٍَُِْ"),
                # "has_explicit_sukoon_mod": "۟" in c_ext or "ْ" in c_ext or "۠" in c_ext,
                # "has_fathah": "َ" in c_ext,
                # "has_dammah": "ُ" in c_ext,
                # "has_kasrah": "ِ" in c_ext,
            })
        if include_this:
            res.update({
                "is_base": (unicodedata.category(c) != "Mn" and c != "ـ") or c == "ٰ",
                "is_alif": c == "ا",
                "is_dagger_alif": c == "ٰ",
                "is_alif_maksura": c == "ى",
                "is_final": end_i >= len(txt) or txt[end_i] == " ",
                "is_space": c == " ",
            })
    elif rule == "qalqalah":
        if not include_this:
            res.update({
                "has_explicit_sukoon": "۟" in c_ext or "ْ" in c_ext,
                "has_maddah": "ٓ" in c_ext,
            })
        if include_this:
            res.update({
                "is_muqalqalah": c in "بدجطق",
            })
    elif rule == "silent":
        if not include_this:
            res.update({
                "has_silent_circle": "۟" in c_ext,
                "has_vowel_incl_tanween": any(s in c_ext for s in "ًٌٍَُِْ"),
                "base_is_dagger_alif": c_base == "ٰ",
                "is_hamzat_wasl": c_base == "ٱ",
                "is_alif": c_base == "ا",
                "is_alif_maksura": c_base == "ى",
                "is_waw": c_base == "و",
                "is_yeh": c_base == "ي",
            })
        if include_this:
            res.update({
                "precedes_high_seen": i + 1 < len(txt) and txt[i + 1] == "ۜ",
                "is_alif": c == "ا",
                "is_alif_maksura": c == "ى",
                "is_waw": c == "و",
                "is_yeh": c == "ي",
            })
    elif rule == "END":
        if not include_this:
            res.update({
                "base_is_space": c_base == " ",
                "has_no_diacritics": start_i + 1 == end_i,
                "has_high_noon": "ۨ" in c_ext,
                "has_explicit_sukoon": "۟" in c_ext or "ْ" in c_ext,
            })
        if include_this:
            res.update({
                "is_base": (unicodedata.category(c) != "Mn" and c != "ـ") or c == "ٰ",
                "is_final_codepoint_in_letter": i + 1 == end_i,
                "is_final_letter_in_ayah": end_i >= len(txt),
            })
    else:
        raise RuntimeError("Unknown rule %s" % rule)

    return RangeAttributes(start_i, end_i, res)


def exemplars_for(rule, txt, auxiliary_stream=None):
    context_size_map = {
        "ghunnah": (3, 1),
        "hamzat_wasl": (1, 0),
        "idghaam_ghunnah": (1, 3),
        "idghaam_mutajanisayn": (0, 2),
        "idghaam_mutaqaribayn": (1, 2),
        "idghaam_no_ghunnah": (0, 3),
        "idghaam_shafawi": (0, 2),
        # Extend lookahead for ikhfa variants to skip pause glyphs/spaces between
        # tanween/noon and the actual ikhfa letter (e.g., \"ۗ \" before the next word).
        "ikhfa": (0, 5),
        "ghunnah_tafkheem": (0, 5),
        "ikhfa_shafawi": (0, 2),
        "iqlab": (0, 2),
        "lam_shamsiyyah": (1, 1),
        "madd_2": (0, 2),
        "madd_246": (1, 2),
        "madd_6": (1, 1),
        "madd_munfasil": (1, 2),
        "madd_muttasil": (0, 3),
        "qalqalah": (1, 1),
        "silent": (0, 1),
        "END": (1, 0)
    }
    lookbehind, lookahead = context_size_map[rule]

    # Use a circular buffer to store the letter attributes.
    # We calculate the codepoint attributes - which are slightly different - within the main loop.

    # Pre-fill the buffer with empty data representing the initial lookbehind.
    letter_attr_buffer = deque([RangeAttributes(-1, 0, None) for x in range(lookbehind)],
                               maxlen=lookbehind + 1 + lookahead)
    # Prime with real present-letter & lookahead data.
    for x in range(lookahead + 1):
        start_idx = letter_attr_buffer[-1].end if letter_attr_buffer else 0
        range_attrs = attributes_for(rule,
                                     txt,
                                     start_idx,
                                     include_this=False,
                                     auxiliary_stream=auxiliary_stream)
        letter_attr_buffer.append(range_attrs)
        if range_attrs.end == len(txt):
            break

    # If we ran out of letters before filling the lookahead, top it off.
    for x in range(lookbehind + 1 + lookahead - len(letter_attr_buffer)):
        letter_attr_buffer.append(RangeAttributes(len(txt), len(txt), None))

    for i in range(len(txt)):
        # Advance letter buffer if required.
        if i >= letter_attr_buffer[lookbehind].end:
            if letter_attr_buffer[-1].end == len(txt):
                letter_attr_buffer.append(RangeAttributes(len(txt), len(txt), None))
            else:
                letter_attr_buffer.append(attributes_for(rule,
                                                         txt,
                                                         letter_attr_buffer[-1].end,
                                                         include_this=False,
                                                         auxiliary_stream=auxiliary_stream))
            assert i < letter_attr_buffer[lookbehind].end, "Next letter did not advance"

        # Build final attribute dictionary.
        attr_full = {}
        for off in range(lookbehind + 1 + lookahead):
            if letter_attr_buffer[off].attributes is None:
                attr_full.update({"%d_exists" % (off - lookbehind): False})
            else:
                attr_full.update({"%d_%s" % (off - lookbehind, k): v
                                  for k, v in letter_attr_buffer[off].attributes.items()})
                attr_full.update({"%d_exists" % (off - lookbehind): True})
        attr_full.update({"0_%s" % k: v
                          for k, v in attributes_for(rule,
                                                     txt,
                                                     i,
                                                     include_this=True,
                                                     auxiliary_stream=auxiliary_stream).attributes.items()})

        yield Exemplar(None, attr_full, 1)


def run_tree(tree, exemplar):
    while not hasattr(tree, "label"):
        if exemplar.attributes.get(tree.attribute, -1) >= tree.value:
            tree = tree.gt
        else:
            tree = tree.lt
    return tree.label


def label_ayah(params):
    # Support older 4-tuple call signature by defaulting return_json to False.
    if len(params) == 4:
        surah, ayah, text, rule_trees = params  # Multiprocessing...
        return_json = False
    else:
        surah, ayah, text, rule_trees, return_json = params  # Multiprocessing...
    # Keep basmala in place so annotations align with the packaged Quran text.
    offset = 0

    # Initialize exemplar generators.
    rules_start_exemplars = {
        k: exemplars_for(k, text) for k in rule_trees
    }
    # All the rules use the same exemplars for making end decisions.
    end_exemplars = exemplars_for("END", text)

    annotations = []
    annotations_run = {k: deque() for k in rule_trees}
    letter_start = 0
    last_letter_start = 0
    for i in range(len(text)):
        end_e = next(end_exemplars)
        # We need some bookkeeping in here for the end-rule trees.
        # I made a one-size-fits-all solution in exemplars_for's auxiliary stream
        # that fits exactly one case - and it's not this one.
        # Also, note that "-1_in_rule" refers to the first character in the letter group.
        # If the rule starts with some harakat in the letter, it will still be false!
        # There is 1 case where this happens: ikhfa on 21:88
        if unicodedata.category(text[i]) != "Mn" or text[i] == "ٰ":
            last_letter_start = letter_start
            letter_start = i
        for k, trees in rule_trees.items():
            e = next(rules_start_exemplars[k])
            if run_tree(trees["start"], e):
                annotations_run[k].append(i)

            # Hax - the exemplars_for auxiliary stream parameter must be random access.
            # So just paste the value we need in here.
            end_e.attributes.update({
                "0_in_rule": len(annotations_run[k]) > 0,
                "-1_in_rule": any(x <= last_letter_start for x in annotations_run[k])
            })

            if run_tree(trees["end"], end_e):
                annotations.append({
                    "rule": k,
                    "start": annotations_run[k].popleft() + offset,
                    "end": i + 1 + offset
                })

    # Remove madd rules from the first letter of muqatta'at-style words
    # (words containing only maddah diacritics and no other harakat).
    muqattaat_starts: set[int] = set()
    has_maddah = False
    has_other_diacritic = False
    first_base = None
    for idx, ch in enumerate(text):
        if ch == " ":
            if has_maddah and not has_other_diacritic and first_base is not None:
                muqattaat_starts.add(first_base + offset)
            has_maddah = False
            has_other_diacritic = False
            first_base = None
            continue
        if unicodedata.category(ch) == "Mn":
            if ch == "ٓ":
                has_maddah = True
            else:
                has_other_diacritic = True
        elif ch == "ٰ":
            has_other_diacritic = True
        if (unicodedata.category(ch) != "Mn" and ch != "ـ") or ch == "ٰ":
            if first_base is None:
                first_base = idx
    if has_maddah and not has_other_diacritic and first_base is not None:
        muqattaat_starts.add(first_base + offset)
    if muqattaat_starts:
        madd_rules = {
            "madd_2",
            "madd_246",
            "madd_munfasil",
            "madd_muttasil",
        }
        cleaned = []
        for ann in annotations:
            rule = ann.get("rule")
            if rule not in madd_rules:
                cleaned.append(ann)
                continue
            start = int(ann.get("start", 0))
            end = int(ann.get("end", 0))
            if any(start <= pos < end for pos in muqattaat_starts):
                continue
            cleaned.append(ann)
        annotations = cleaned

    # Enforce madd priority: strongest kept, weaker removed when spans overlap exactly.
    madd_priority = {
        "madd_6": 0,
        "madd_muttasil": 1,
        "madd_munfasil": 2,
        "madd_246": 3,
        "madd_2": 4,
    }
    spans: dict[tuple[int, int], list[dict]] = {}
    for ann in annotations:
        rule = ann.get("rule")
        if rule not in madd_priority:
            continue
        key = (int(ann.get("start", 0)), int(ann.get("end", 0)))
        spans.setdefault(key, []).append(ann)
    to_drop: set[int] = set()
    for span, anns in spans.items():
        strongest = min(anns, key=lambda a: madd_priority.get(a.get("rule"), 99))
        keep_rule = strongest.get("rule")
        for ann in anns:
            if ann is strongest or ann.get("rule") == keep_rule:
                continue
            to_drop.add(id(ann))
    if to_drop:
        annotations = [a for a in annotations if id(a) not in to_drop]

    assert all(len(q) == 0 for q in annotations_run.values()), \
        "Some rules left hanging at end of ayah @ %d: %s (%d:%d) %s" % \
        (len(text), annotations_run, surah, ayah, annotations)
    json = {
        "surah": surah,
        "ayah": ayah,
        "annotations": sorted(annotations, key=lambda x: x["start"])
    }
    if return_json:
        return json
    else:
        e = embedding()
        e.__dict__ = json
        e.text = text
        return embed(e)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def delete_last_line(num=1):
    for x in range(num):
        eprint('\x1b[1A', end="")    #cursor up one line
        eprint('\x1b[2K', end="")    #delete last line

def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

if __name__ == "__main__":
    base_dir = str(BASE_DIR)
    output_dir = os.path.join(base_dir, "output")

    parser = argparse.ArgumentParser(description='Generate Quran Tajweed Label Embeddings.')
    parser.add_argument('--json', action='store_true', help='Generate a JSON file, instead of embeddings')
    parser.add_argument('--dictionary', action='store_true', help='Generate a dictionary file. Does not work with --json')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to write output file. Defaults to STDOUT.')
    global args
    global dict
    args = parser.parse_args()
    dict = {'h'}
    # Load rules from incredibly high-tech datastore.
    rule_trees = {}
    rule_start_files = sorted(RULE_TREES_DIR.glob("*.start.json"))
    eprint("Loading rules..", end=" ")
    for start_file in rule_start_files:
        rule_name = start_file.name.partition(".")[0]
        end_file = start_file.with_name(start_file.name.replace(".start.", ".end."))
        rule_trees[rule_name] = {
            "start": json2tree(json.load(open(start_file))),
            "end": json2tree(json.load(open(end_file))),
        }
    if sys.stdin.isatty():
                  
        fname = os.path.join(output_dir, "quran-uthmani.txt")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        if os.path.exists(fname) and os.path.getsize(fname) > 0:
            eprint(f'\nSTDIN is empty! Reading Quran Text from:\n\t{fname}', end="")
        else:
            eprint(f'\nSTDIN is empty! Downloading Quran Text from:\n\t{DEFAULT_TANZIL_URLS[0]}', end="")
            ok = download_quran_txt(Path(fname), urls=DEFAULT_TANZIL_URLS)
            if not ok or not os.path.exists(fname) or os.path.getsize(fname) == 0:
                raise RuntimeError(f"Failed to download Quran text to {fname}")
        file = open(fname, 'r', encoding="utf-8").readlines()
    else:
        file = sys.stdin
    # Read in text to classify
    tasks = []
    expected_counts = {}
    spinner = spinning_cursor()
    eprint("\nReading Text..", end=" ")
    for line in file:
        eprint(next(spinner), end="")
        line = line.split("|")
        if len(line) != 3:
            eprint('\b', end="")
            continue
        eprint('\b', end="")
        sys.stderr.flush()
        surah = int(line[0])
        ayah = int(line[1])
        text = normalize_superscript_alef(line[2].strip())
        expected_counts[surah] = expected_counts.get(surah, 0) + 1
        tasks.append((surah, ayah, text, rule_trees, args.json))

    # Perform classification.
    eprint("\nPerforming classification..")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = []
    for result in tqdm.tqdm(pool.imap(label_ayah, tasks), total=len(tasks)):
        results.append(result)

    # Basic sanity check: we should emit as many ayat per surah as we read in.
    if args.json:
        produced = {}
        for r in results:
            produced[r["surah"]] = produced.get(r["surah"], 0) + 1
    else:
        produced = {}
        for r in results:
            surah_num = int(r.split("|", 2)[0])
            produced[surah_num] = produced.get(surah_num, 0) + 1

    missing = {
        s: expected_counts[s] - produced.get(s, 0)
        for s in expected_counts
        if produced.get(s, 0) != expected_counts[s]
    }
    if missing:
        raise RuntimeError(f"Mismatch in ayah counts per surah: {missing}")

    # Pretty-print output because disk space is cheap.
    delete_last_line(6)
    if args.json:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, sort_keys=True)
        else:
            json.dump(results, sys.stdout, indent=2, sort_keys=True)
    else:
        out_text = "\n".join(results)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(out_text)
                fh.write("\n")
        else:
            print(out_text)
    print("")

    # Print the optional Dictionary
    if args.dictionary:
        for i in results:
            dict = dict.union(set(i.split('|')[2]))
        d = open('dictionary.txt', 'w')
        d.write('# Quran Alphabets')
        d.write('\n'.join(sorted(dict)))
        d.write('\n#New Line')
        d.close()
