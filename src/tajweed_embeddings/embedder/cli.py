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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tajweed embeddings for a given sura/ayah.")
    parser.add_argument("--sura", type=int, required=True, help="Sura number (1-114).")
    parser.add_argument("--aya", type=int, default=None, help="Ayah number (1-based). Optional; if omitted, embeds full sura.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emb = TajweedEmbedder()

    if not args.quiet:
        banner = (
            "╔══════════════════════════════════════════════════════════╗\n"
            "║ TajweedEmbedder CLI                                      ║\n"
            "║   For inspection only — use programmatically for models. ║\n"
            "║   String output is a human view, NOT the numeric vectors.║\n"
            "╚══════════════════════════════════════════════════════════╝"
        )
        print(banner)

    embeddings = emb.text_to_embedding(args.sura, args.aya, args.subtext)
    print(emb.encoding_to_string(embeddings, style=args.style))


if __name__ == "__main__":  # pragma: no cover
    main()
