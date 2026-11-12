"""
Crawl the Hindi translations + commentaries source (Person 3: Anvi).

Source: the VedicScriptures Bhagavad Gita data (the same data that powers
https://vedicscriptures.github.io), one JSON file per verse in
https://github.com/vedicscriptures/bhagavad-gita-data
Each file holds the Sanskrit, transliteration, 6 English + 3 Hindi translations,
and commentaries (Swami Ramsukhdas's Hindi commentary is where most verse
cross-references come from).

The crawler uses Antara's base_crawler (robots.txt check, 1.5 s delay, clear
user agent). It walks each chapter verse by verse until the source returns
404, so it needs no hard-coded verse counts. Files already downloaded are
skipped, so an interrupted crawl can simply be run again.

Output: data/raw/hi_commentary/<chapter>_<verse>.json  (raw, unedited)
Then run:  python crawler/merge.py

Run from the repo root:
    python crawler/crawl_hi_commentary.py
    python crawler/crawl_hi_commentary.py --chapters 2 12     (only some chapters, for testing)
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
