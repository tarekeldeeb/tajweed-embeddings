"""Command-line entrypoint for TajweedEmbedder."""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from .tajweed_embedder import TajweedEmbedder
except ImportError:  # pragma: no cover - direct script execution fallback
    repo_src = Path(__file__).resolve().parents[2]  # .../src
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from tajweed_embeddings.embedder.tajweed_embedder import TajweedEmbedder


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for tajweed embeddings."""
    parser = argparse.ArgumentParser(
        description="Render tajweed embeddings for a given sura/ayah."
    )
    parser.add_argument(
        "--sura", type=int, required=True, help="Sura number (1-114)."
    )
    parser.add_argument(
        "--aya",
        type=int,
        default=None,
        help="Ayah number (1-based). Optional; if omitted, embeds full sura.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of consecutive ayat to embed starting at --aya (default: 1).",
    )
    parser.add_argument(
        "--style",
        choices=["short", "long"],
        default="short",
        help="Display style for encoding_to_string output (default: short).",
    )
    parser.add_argument(
        "--subtext",
        type=str,
        default=None,
        help="Optional custom text instead of the ayah from quran.json.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode: skip banner/warnings.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for CLI execution."""
    args = parse_args()
    emb = TajweedEmbedder()
    embeddings = emb.text_to_embedding(args.sura, args.aya, args.subtext, count=args.count)

    if not args.quiet:
        banner = (
            "╔══════════════════════════════════════════════════════════╗\n"
            "║ Tajweed Embeddings CLI                                   ║\n"
            "║   For inspection only — use programmatically for models. ║\n"
            "║   String output is a human view, NOT the numeric vectors.║\n"
            "║                                                          ║\n"
            "║ ┌ Index: row number                                      ║\n"
            "║ │  ┌ Letter: glyph                                       ║\n"
            "║ │  │   ┌ Tashkeel: Kasra ‿ , Fatha ^ , .. etc            ║\n"
            "║ │  │   │    ┌ Pause: stop mark (0/4/6 etc)               ║\n"
            "║ │  │   │    │   ┌ Jahr 🔊 , Hams 🤫                      ║\n"
            "║ │  │   │    │   │  ┌ Rikhw 💨 , Tawasot ➖ , Shidda 🚫   ║\n"
            "║ │  │   │    │   │  │  ┌ Isti'la 🔼 , Istifal 🔻          ║\n"
            "║ │  │   │    │   │  │  │  ┌ Infitah ▲ , Itbaq ⟂           ║\n"
            "║ │  │   │    │   │  │  │  │  ┌ Idhlaq 😮 , Ismat 😐       ║\n"
            "║ │  │   │    │   │  │  │  │  │    ┌ Rules: Tajweed flags  ║\n"
            "║ │  │   │    │   │  │  │  │  │    │                       ║\n"
            "╚═╪══╪═══╪════╪═══╪══╪══╪══╪══╪════╪═══════════════════════╝"
        )
        print(banner)

    print(emb.encoding_to_string(embeddings, style=args.style))


if __name__ == "__main__":  # pragma: no cover
    main()
