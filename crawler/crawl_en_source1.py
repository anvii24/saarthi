
import json
from pathlib import Path

from config import EN_SOURCE1_DIR
from crawler.base_crawler import fetch, save_raw
from schemas import Verse, validate_verse

BASE_URL = "https://raw.githubusercontent.com/AdiyogiArts/gita-json/main"
BASE_SOURCE_URL = f"{BASE_URL}/base.json"
EN_SOURCE_URL = f"{BASE_URL}/en.json"


def crawl_en_source1():
    EN_SOURCE1_DIR.mkdir(parents=True, exist_ok=True)

    base_path = EN_SOURCE1_DIR / "base.json"
    english_path = EN_SOURCE1_DIR / "en.json"

    if not base_path.exists():
        save_raw(fetch(BASE_SOURCE_URL), base_path)

    if not english_path.exists():
        save_raw(fetch(EN_SOURCE_URL), english_path)

    with base_path.open("r", encoding="utf-8") as file:
        base_data = json.load(file)

    with english_path.open("r", encoding="utf-8") as file:
        english_data = json.load(file)

    verses = []

    for verse_key, base_verse in base_data.items():
        chapter = base_verse["chapter"]
        verse_number = base_verse["verse"]
        english_verse = english_data.get(verse_key, {})

        record: Verse = {
            "id": f"{chapter}.{verse_number}",
            "chapter": chapter,
            "verse": verse_number,
            "sanskrit": base_verse["sanskrit"],
            "transliteration": base_verse["transliteration"],
            "translations": {
                "en_source1": english_verse.get("translation", "")
            },
            "commentary": english_verse.get("explanation", ""),
            "source_urls": [
                BASE_SOURCE_URL,
                EN_SOURCE_URL
            ]
        }

        problems = validate_verse(record)

        if problems:
            raise ValueError(f"Invalid verse {record['id']}: {problems}")

        verses.append(record)

    verses.sort(key=lambda verse: (verse["chapter"], verse["verse"]))

    output_path = EN_SOURCE1_DIR / "verses.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(verses, file, ensure_ascii=False, indent=2)

    print(f"Downloaded and processed {len(verses)} verses")
    print(f"Saved to {output_path}")

    return verses


if __name__ == "__main__":
    crawl_en_source1()
