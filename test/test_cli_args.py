"""Basic CLI argument wiring tests."""

from tajweed_embeddings.embedder.cli import parse_args


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
