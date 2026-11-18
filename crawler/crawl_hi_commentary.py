"""Download one JSON file per verse (Sanskrit, 8 translations, Hindi and English commentary)
from github.com/vedicscriptures/bhagavad-gita-data into data/raw/hi_commentary/.

Already downloaded verses are skipped, so it can be re-run. Run: python crawler/crawl_hi_commentary.py
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HI_COMMENTARY_DIR  # noqa: E402
from crawler.base_crawler import fetch, save_raw  # noqa: E402

BASE_URL = ("https://raw.githubusercontent.com/vedicscriptures/bhagavad-gita-data/"
            "main/slok/bhagavadgita_chapter_{ch}_slok_{v}.json")
MAX_RETRIES = 2


def fetch_verse(ch: int, v: int) -> str | None:
    """Raw JSON text for one verse, or None when the chapter has ended (404)."""
    url = BASE_URL.format(ch=ch, v=v)
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return fetch(url)
        except RuntimeError as err:
            if "404" in str(err):
                return None
            if attempt > MAX_RETRIES:
                raise
            print(f"  retrying {ch}.{v} after error: {err}")


def crawl_chapter(ch: int) -> tuple[int, int]:
    """Download every verse of one chapter; returns (downloaded, skipped)."""
    downloaded = skipped = 0
    v = 1
    while True:
        path = HI_COMMENTARY_DIR / f"{ch}_{v}.json"
        if path.exists():
            skipped += 1
        else:
            text = fetch_verse(ch, v)
            if text is None:
                break
            save_raw(text, path)
            downloaded += 1
        v += 1
    return downloaded, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", type=int, nargs="*", default=list(range(1, 19)))
    args = parser.parse_args()

    total = 0
    for ch in args.chapters:
        downloaded, skipped = crawl_chapter(ch)
        total += downloaded + skipped
        print(f"Chapter {ch:2}: {downloaded} downloaded, {skipped} already there")

    print(f"Done: {total} files in {HI_COMMENTARY_DIR}")
    print("Next: python crawler/merge.py")


if __name__ == "__main__":
    main()
