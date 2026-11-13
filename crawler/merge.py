"""
Merge raw VedicScriptures responses into data/verses.json and data/links.json
(Person 3: Anvi).

Input:  1. data/raw/hi_commentary/  — one raw VedicScriptures JSON per verse
           (from crawler/crawl_hi_commentary.py): Sanskrit, transliteration,
           6 English + 2 Hindi translations, English + Hindi commentary
        2. data/raw/en_source1/verses.json  — Antara's AdiyogiArts source
           (crawler/crawl_en_source1.py). Its English translation is ~98% the same
           as Gambhirananda's, which we already have, so we take only its short
           plain-English explanation of each verse, as "explanation_en".
        3. data/raw/en_source2/verses.json  — Kashvi's Swami Swarupananda (1909)
           translation (crawler/crawl_en_source2.py), added as "en_swarupananda".
           Its edition counts 34 verses in chapter 13, so chapters are aligned by
           comparing texts (see add_aligned_translation), not just by id.
Output: data/verses.json  — one clean Verse record per verse (see schemas.py)
        data/links.json   — verse-to-verse cross-references found in the commentaries

Run from the repo root (after both crawlers):
    python crawler/merge.py
    python crawler/merge.py --raw some/other/folder     (optional)

Notes for the report:
- The source has one extra entry per chapter: the closing colophon
  ("ॐ तत्सदिति ... नाम ... अध्यायः"). It is not a verse, so it is dropped.
- This edition has 35 verses in chapter 13, so the total is 701, not 700.
- Links come from Swami Ramsukhdas's Hindi commentary ("(गीता 6। 30)" or "(6। 30)")
  and Prabhupada's English purport ("Bg. 18.69"). References to other texts
  (मानस, श्रीमद्भागवत, Upaniṣads ...) and three-part numbers are ignored.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import validate_verse  # noqa: E402

RAW_DIR = os.path.join("data", "raw", "hi_commentary")
EN_SOURCE1 = os.path.join("data", "raw", "en_source1", "verses.json")
EN_SOURCE2 = os.path.join("data", "raw", "en_source2", "verses.json")
VERSES_OUT = os.path.join("data", "verses.json")
LINKS_OUT = os.path.join("data", "links.json")
SOURCE_URL = ("https://raw.githubusercontent.com/vedicscriptures/bhagavad-gita-data/"
              "main/slok/bhagavadgita_chapter_{ch}_slok_{v}.json")

# (source key in the API response, field, our translation key)
ENGLISH_TRANSLATIONS = [
    ("siva", "et", "en_sivananda"),
    ("purohit", "et", "en_purohit"),
    ("gambir", "et", "en_gambirananda"),
    ("adi", "et", "en_adidevananda"),
    ("san", "et", "en_sankaranarayan"),
    ("prabhu", "et", "en_prabhupada"),
]
HINDI_TRANSLATIONS = [
    ("tej", "ht", "hi_tejomayananda"),
    ("rams", "ht", "hi_ramsukhdas"),
]
COMMENTARY_EN = ("prabhu", "ec")   # English purport
COMMENTARY_HI = ("rams", "hc")     # Hindi commentary (richest in cross-references)

EN_SOURCE1_URL = "https://raw.githubusercontent.com/AdiyogiArts/gita-json/main/en.json"
EN_SOURCE2_URL = "https://sacred-texts.com/book/srimad-bhagavad-gita/contents"

COLOPHON_START = "ॐ तत्सदिति"
NO_COMMENT = "did not comment"

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

# "2.47 ", "2.47. ", "।।2.47।। ", "2.47।। " at the start of a translation
LEADING_NUMBER = re.compile(r"^\s*(?:।।)?\s*\d{1,2}\.\d{1,3}\.?\s*(?:।।)?\s*")
# "||२-४७||" or "||2-47||" at the end of a verse
TRAILING_MARK = re.compile(r"\|\|\s*[\d०-९]+\s*-\s*[\d०-९]+\s*\|\|\s*$")


def clean(text) -> str:
    """Strip verse-number prefixes, odd spaces and 'did not comment' placeholders."""
    if not text or not isinstance(text, str):
        return ""
    if NO_COMMENT in text:
        return ""
    text = text.replace("\xa0", " ")
    text = LEADING_NUMBER.sub("", text, count=1)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_verse_text(text: str) -> str:
    """Clean the Sanskrit / transliteration text and drop the closing ||2-47|| mark."""
    text = (text or "").replace("\xa0", " ")
    text = TRAILING_MARK.sub("", text.strip())
    return re.sub(r"[ \t]+", " ", text).strip()


def get(raw: dict, key: str, field: str) -> str:
    part = raw.get(key)
    return clean(part.get(field)) if isinstance(part, dict) else ""


# ---------------------------------------------------------------------------
# Cross-reference extraction
# ---------------------------------------------------------------------------

HI_PAREN = re.compile(r"\(([^()]{0,120})\)")
HI_PAIR = re.compile(r"(?<![\d।])\s*(\d{1,2})\s*।\s*(\d{1,3})(?!\s*।\s*\d)")
HI_THREE_PART = re.compile(r"\d+\s*।\s*\d+\s*।\s*\d+")
EN_REF = re.compile(r"(?:Bg\.|Bhagavad-g[iī]t[aā])\s*\(?\s*(\d{1,2})\.(\d{1,3})(?!\.\d)")


def hindi_refs(text: str) -> list[tuple[int, int]]:
    """Gita references in Ramsukhdas's commentary: '(गीता 6। 30)' or '(5। 25 12। 4)'."""
    refs = []
    for match in HI_PAREN.finditer(text or ""):
        inside = match.group(1).strip()
        if HI_THREE_PART.search(inside):
            continue  # e.g. (मानस 1। 140। 5) — another text
        prefix = re.split(r"\d", inside, maxsplit=1)[0].strip()
        if prefix not in ("", "गीता", "गाता"):
            continue  # another text, e.g. मानस, श्रीमद्भा0, योगदर्शन
        refs += [(int(c), int(v)) for c, v in HI_PAIR.findall(inside)]
    return refs


def english_refs(text: str) -> list[tuple[int, int]]:
    """Gita references in Prabhupada's purport: 'Bg. 18.69' or 'Bhagavad-gītā (4.9)'."""
    return [(int(c), int(v)) for c, v in EN_REF.findall(text or "")]


# ---------------------------------------------------------------------------
# Aligning a source that numbers verses differently
# ---------------------------------------------------------------------------
# Some editions count 34 verses in chapter 13 (our edition has 35, with Arjuna's
# question as 13.1), so their 13.1 is our 13.2. Matching ids alone would attach
# every chapter 13 translation to the wrong verse. For each chapter we try
# shifting the source by -1, 0 or +1 and keep the shift whose texts overlap
# most with our existing English translations.

WORD = re.compile(r"[a-z]+")
STOP = {"the", "and", "of", "to", "in", "is", "a", "that", "he", "who", "this", "his",
        "o", "for", "with", "by", "not", "be", "as", "are", "all", "which", "me", "my",
        "i", "you", "thou", "thy", "thee", "it", "from", "one", "these", "they", "so"}


def _words(text: str) -> set[str]:
    return {w for w in WORD.findall((text or "").lower()) if w not in STOP and len(w) > 2}


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def add_aligned_translation(verses: list[dict], source: list[dict], src_key: str, our_key: str) -> int:
    """Add source[...]['translations'][src_key] to our verses as our_key, per-chapter aligned."""
    ours = {v["id"]: _words(" ".join(t for k, t in v["translations"].items() if k.startswith("en_")))
            for v in verses}
    by_chapter: dict[int, list[tuple[int, str]]] = {}
    for record in source:
        text = clean(record.get("translations", {}).get(src_key, ""))
        if text:
            by_chapter.setdefault(int(record["chapter"]), []).append((int(record["verse"]), text))

    by_id = {v["id"]: v for v in verses}
    added = 0
    for ch, items in sorted(by_chapter.items()):
        def score(shift: int) -> float:
            sims = [_overlap(_words(t), ours.get(f"{ch}.{v + shift}", set())) for v, t in items]
            return sum(sims) / len(sims)
        best = max((0, 1, -1), key=score)  # 0 first, so ties keep the ids as they are
        if best != 0:
            print(f"  chapter {ch}: source numbering is shifted, mapping verse n -> n{best:+d} "
                  f"(text overlap {score(0):.2f} -> {score(best):.2f})")
        for v, text in items:
            target = by_id.get(f"{ch}.{v + best}")
            if target:
                target["translations"][our_key] = text
                if EN_SOURCE2_URL not in target["source_urls"]:
                    target["source_urls"].append(EN_SOURCE2_URL)
                added += 1
    return added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def to_verse(raw: dict) -> dict:
    ch, v = int(raw["chapter"]), int(raw["verse"])
    translations = {}
    for key, field, name in ENGLISH_TRANSLATIONS + HINDI_TRANSLATIONS:
        text = get(raw, key, field)
        if text:
            translations[name] = text

    return {
        "id": f"{ch}.{v}",
        "chapter": ch,
        "verse": v,
        "speaker": clean(raw.get("speaker", "")),
        "sanskrit": clean_verse_text(raw.get("slok", "")),
        "transliteration": clean_verse_text(raw.get("transliteration", "")),
        "translations": translations,
        "commentary": get(raw, *COMMENTARY_EN),
        "commentary_hi": get(raw, *COMMENTARY_HI),
        "source_urls": [SOURCE_URL.format(ch=ch, v=v)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=RAW_DIR, help="folder with one raw JSON per verse")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.raw, "*.json"))
    if not files:
        sys.exit(f"No JSON files found in {args.raw}")

    raws = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            raws.append(json.load(f))

    # 1. drop the closing colophon of each chapter
    real = [r for r in raws if not str(r.get("slok", "")).strip().startswith(COLOPHON_START)]
    print(f"Read {len(raws)} files, dropped {len(raws) - len(real)} chapter colophons")

    # 2. convert and validate
    verses = sorted((to_verse(r) for r in real), key=lambda x: (x["chapter"], x["verse"]))
    bad = 0
    for verse in verses:
        problems = validate_verse(verse)
        if problems:
            bad += 1
            print(f"  {verse['id']}: {problems}")
    valid_ids = {v["id"] for v in verses}

    # 2b. add Antara's plain-English explanations (en_source1), matched by verse id
    if os.path.exists(EN_SOURCE1):
        with open(EN_SOURCE1, encoding="utf-8") as f:
            explanations = {r["id"]: clean(r.get("commentary", "")) for r in json.load(f)}
        added = 0
        for verse in verses:
            text = explanations.get(verse["id"], "")
            if text:
                verse["explanation_en"] = text
                verse["source_urls"].append(EN_SOURCE1_URL)
                added += 1
        print(f"Added {added} plain-English explanations from {EN_SOURCE1}")
    else:
        print(f"(No {EN_SOURCE1} yet: run crawler/crawl_en_source1.py to add explanations)")

    # 2c. add Kashvi's Swarupananda (1909) translation (en_source2), aligning chapters
    if os.path.exists(EN_SOURCE2):
        with open(EN_SOURCE2, encoding="utf-8") as f:
            source2 = json.load(f)
        added = add_aligned_translation(verses, source2, "en_source2", "en_swarupananda")
        print(f"Added {added} Swarupananda translations from {EN_SOURCE2}")
    else:
        print(f"(No {EN_SOURCE2} yet: run crawler/crawl_en_source2.py to add it)")

    # 3. extract links (only to verses that exist, never to itself, no duplicates)
    links, seen = [], set()
    for raw, verse in zip(sorted(real, key=lambda r: (int(r["chapter"]), int(r["verse"]))), verses):
        found = [("rams_hi", ref) for ref in hindi_refs(get(raw, *COMMENTARY_HI))]
        found += [("prabhu_en", ref) for ref in english_refs(get(raw, *COMMENTARY_EN))]
        for source, (c, v) in found:
            target = f"{c}.{v}"
            key = (verse["id"], target)
            if target in valid_ids and target != verse["id"] and key not in seen:
                seen.add(key)
                links.append({"from": verse["id"], "to": target, "source": source})

    # 4. save
    os.makedirs("data", exist_ok=True)
    with open(VERSES_OUT, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)
    with open(LINKS_OUT, "w", encoding="utf-8") as f:
        json.dump(links, f, ensure_ascii=False, indent=2)

    per_chapter = Counter(v["chapter"] for v in verses)
    linked = len({l["from"] for l in links} | {l["to"] for l in links})
    print(f"Saved {len(verses)} verses to {VERSES_OUT} ({bad} with problems)")
    print(f"Verses per chapter: {dict(sorted(per_chapter.items()))}")
    print(f"Saved {len(links)} links to {LINKS_OUT}, touching {linked} verses")
    print(f"Links by source: {dict(Counter(l['source'] for l in links))}")


if __name__ == "__main__":
    main()
