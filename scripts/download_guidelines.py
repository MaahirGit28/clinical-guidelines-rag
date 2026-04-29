"""Download AACAP clinical practice parameters into data/raw/.

The list below is a *starter*. Curate based on your scope and verify URLs against
https://www.aacap.org/AACAP/Resources_for_Primary_Care/Practice_Parameters_and_Resource_Centers/Practice_Parameters.aspx
since some links rotate. AACAP-published guidelines are intended for clinician use
and are publicly available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_PDF_DIR  # noqa: E402

# Curated starter set. Add ~10–15 more for a real evaluation corpus.
GUIDELINES: dict[str, str] = {
    "depressive_disorders.pdf": (
        "https://www.aacap.org/App_Themes/AACAP/docs/practice_parameters/"
        "depressive_disorders_practice_parameter.pdf"
    ),
    "psychiatric_assessment.pdf": (
        "https://www.aacap.org/App_Themes/AACAP/docs/practice_parameters/"
        "psychiatric_assessment_practice_parameter.pdf"
    ),
    # TODO: add ADHD, autism, anxiety, OCD, bipolar, eating disorders,
    # substance use, suicidal behavior, PTSD, intellectual disability, etc.
}

HEADERS = {"User-Agent": "clinical-guidelines-rag/0.1 (research prototype)"}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        tqdm.write(f"  ✓ {dest.name} (cached)")
        return
    resp = requests.get(url, stream=True, timeout=30, headers=HEADERS)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    tqdm.write(f"  ✓ {dest.name} ({dest.stat().st_size // 1024} KB)")


def main() -> None:
    print(f"Downloading {len(GUIDELINES)} guidelines to {RAW_PDF_DIR}")
    failures: list[str] = []
    for filename, url in tqdm(GUIDELINES.items(), desc="Fetching"):
        try:
            download(url, RAW_PDF_DIR / filename)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"  ✗ {filename}: {exc}")
            failures.append(filename)
    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        sys.exit(1)
    print("\nDone. Next: python -m src.ingest")


if __name__ == "__main__":
    main()
