"""
Make data/sample_verses.json - a 50-verse sample for us to test with
until the full crawled data/verses.json is ready.

Source: the OEvortex/Bhagavad_Gita dataset on Hugging Face
(700 verses with Sanskrit, Hindi and English; listed as MIT licence).

Run from the repo root:
    pip install datasets
    python crawler/make_sample.py
"""

import json
import os
import re
import sys

from datasets import load_dataset

# make "from schemas import ..." work when this file lives in crawler/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import validate_verse  # noqa: E402

DATASET_NAME = "OEvortex/Bhagavad_Gita"
OUTPUT_PATH = os.path.join("data", "sample_verses.json")
SAMPLE_SIZE = 50


def get_field(row, *names):
    """Return the first column that exists (the dataset spells 'English' as 'Enlgish')."""
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name]).strip()
    return ""


def parse_chapter_verse(row):
    """Turn 'Chapter 2' + 'Verse 2.47' into (2, 47)."""
    verse_text = get_field(row, "Verse")
    match = re.search(r"(\d+)\.(\d+)", verse_text)
    if match:
        return int(match.group(1)), int(match.group(2))

    chapter_match = re.search(r"\d+", get_field(row, "Chapter"))
    verse_match = re.search(r"\d+", verse_text)
    if chapter_match and verse_match:
        return int(chapter_match.group()), int(verse_match.group())
    return None


def to_verse_record(row):
    """Convert one dataset row into our Verse format (see schemas.py)."""
    parsed = parse_chapter_verse(row)
    if parsed is None:
        return None
    chapter, verse = parsed

    return {
        "id": f"{chapter}.{verse}",
        "chapter": chapter,
        "verse": verse,
        "sanskrit": get_field(row, "Sanskrit Anuvad"),
        "transliteration": "",  # not in this dataset; the crawler will add it
        "translations": {
            "en_source1": get_field(row, "Enlgish Translation", "English Translation"),
            "hi_source1": get_field(row, "Hindi Anuvad"),
        },
        "commentary": "",  # not in this dataset; the crawler will add it
        "source_urls": [f"https://huggingface.co/datasets/{DATASET_NAME}"],
    }


def main():
    print(f"Loading {DATASET_NAME} ...")
    rows = load_dataset(DATASET_NAME, split="train")
    print(f"Loaded {len(rows)} rows")

    records = [r for r in (to_verse_record(row) for row in rows) if r is not None]

    # spread the sample across the whole Gita instead of taking chapter 1 only
    step = max(1, len(records) // SAMPLE_SIZE)
    sample = records[::step][:SAMPLE_SIZE]

    bad = 0
    for record in sample:
        problems = validate_verse(record)
        if problems:
            bad += 1
            print(f"  {record['id']}: {problems}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    chapters = sorted({r["chapter"] for r in sample})
    print(f"Saved {len(sample)} verses to {OUTPUT_PATH}")
    print(f"Chapters covered: {chapters}")
    print(f"Records with problems: {bad}")


if __name__ == "__main__":
    main()
