"""Basic CLI argument wiring tests."""

import sys

from tajweed_embeddings.embedder.cli import main, parse_args
from tajweed_embeddings.embedder.sifat_embedder import SifatEmbedder


def test_banner_sifat_symbols_match_rows(monkeypatch, capsys):
    """The banner legend uses the same ṣifāt symbols as the rendered rows."""
    labels = SifatEmbedder().short_label_map
    istila = labels["isti'la"]
    monkeypatch.setattr(sys, "argv", ["tajweed-embeddings", "--sura", "1", "--aya", "1"])
    main()
    out = capsys.readouterr().out
    for legend in (
        f"Jahr {labels['jahr']} , Hams {labels['hams']}",
        f"Rikhw {labels['rikhwah']} , Tawasot {labels['tawassut']} , Shidda {labels['shiddah']}",
        f"Isti'la {istila} , Istifal {labels['istifal']}",
        f"Infitah {labels['infitah']} , Itbaq {labels['itbaq']}",
        f"Idhlaq {labels['idhlaq']} , Ismat {labels['ismat']}",
    ):
        assert legend in out


def test_cli_accepts_count_argument(monkeypatch):
    """Ensure --count is parsed and passed through without errors."""
    args = parse_args(
        [
            "--sura",
            "1",
            "--aya",
            "1",
            "--count",
            "2",
            "--subtext",
            "بِسْمِ",
        ]
    )
    assert args.sura == 1
    assert args.aya == 1
    assert args.count == 2
    assert args.subtext == "بِسْمِ"


def test_cli_sura_only_defaults(monkeypatch):
    """Sura only should embed full sura with default count=1 and no aya/subtext."""
    args = parse_args(["--sura", "1"])
    assert args.sura == 1
    assert args.aya is None
    assert args.count == 1
    assert args.subtext is None


def test_cli_sura_and_aya(monkeypatch):
    """Sura+aya should set aya and keep default count/subtext."""
    args = parse_args(["--sura", "2", "--aya", "5"])
    assert args.sura == 2
    assert args.aya == 5
    assert args.count == 1
    assert args.subtext is None


def test_cli_subtext_only(monkeypatch):
    """Subtext with sura/aya should be parsed."""
    args = parse_args(["--sura", "3", "--aya", "1", "--subtext", "ABC"])
    assert args.sura == 3
    assert args.aya == 1
    assert args.subtext == "ABC"
    assert args.count == 1
