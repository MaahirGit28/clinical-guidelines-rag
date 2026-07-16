"""Download AACAP Clinical Practice Guidelines and Practice Parameters.

Edit GUIDELINE_URLS below with verified PDF links from aacap.org. Files are
saved to data/raw/ and skipped if they already exist.

If a download fails (paywall, JS redirect, etc.), grab the PDF manually
from your browser and drop it in data/raw/ — the ingest step is agnostic
to source.
"""
import logging
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Allow running as `python scripts/download_guidelines.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# (filename, url) — verify each URL resolves to a PDF before running.
GUIDELINE_URLS: list[tuple[str, str]] = [
    # ("adhd_practice_parameter.pdf", "https://www.aacap.org/..."),
    # ("anxiety_practice_parameter.pdf", "https://www.aacap.org/..."),
    # ("depression_cpg.pdf", "https://www.aacap.org/..."),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def download_one(filename: str, url: str) -> bool:
    target = RAW_DIR / filename
    if target.exists():
        log.info(f"SKIP {filename} (already present)")
        return True

    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(target, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=filename
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        log.info(f"OK   {filename}")
        return True
    except Exception as e:
        log.error(f"FAIL {filename}: {e}")
        if target.exists():
            target.unlink()
        return False


def main():
    if not GUIDELINE_URLS:
        log.warning(
            "GUIDELINE_URLS is empty. Populate it with verified AACAP PDF "
            "URLs, or drop PDFs directly into data/raw/."
        )
        return

    log.info(f"Downloading {len(GUIDELINE_URLS)} guideline(s) → {RAW_DIR}")
    results = [download_one(name, url) for name, url in GUIDELINE_URLS]
    log.info(f"Done: {sum(results)}/{len(results)} succeeded")


if __name__ == "__main__":
    main()