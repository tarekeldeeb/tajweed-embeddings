"""Helpers for fetching the Tanzil Uthmani Quran text used by the pipeline."""

from pathlib import Path
from typing import Iterable


DEFAULT_TANZIL_URLS = [
    # Matches rules_gen/tajweed_classifier.py default URL (marks preserved).
    "https://tanzil.net/pub/download/index.php?marks=true&alif=false&quranType=uthmani&outType=txt-2&agree=true",
    # Fallback plain text URL.
    "https://tanzil.net/download/quran-uthmani.txt",
]


def download_quran_txt(target: Path, urls: Iterable[str] | None = None, timeout: int = 30) -> bool:
    """
    Download the Tanzil Uthmani text to `target`. Returns True on success.

    Uses requests if available; falls back to urllib.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    url_list = list(urls) if urls is not None else DEFAULT_TANZIL_URLS

    try:
        import requests  # type: ignore
    except Exception:
        requests = None

    for url in url_list:
        try:
            if requests:
                resp = requests.get(url, timeout=timeout, allow_redirects=True)
                resp.raise_for_status()
                data = resp.content
            else:
                from urllib.request import urlopen

                with urlopen(url, timeout=timeout) as fh:  # type: ignore
                    data = fh.read()
            if data:
                target.write_bytes(data)
            if target.exists() and target.stat().st_size > 0:
                return True
        except Exception:
            continue
    return target.exists() and target.stat().st_size > 0
