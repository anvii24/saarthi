"""
Shared data shapes for Saarthi.

Everyone imports from this file so that crawlers, indexes, search,
ranking and the app all agree on what a verse, a link and a search
result look like. Do not change a field name without telling the team.

Usage:
    from schemas import Verse, Link, SearchResult, THEMES, validate_verse
"""

from typing import TypedDict

# Theme labels 

THEMES = [
    "duty",
    "action_and_results",
    "fear_and_anxiety",
    "anger_and_desire",
    "grief_and_loss",
    "self_control",
    "purpose_and_meaning",
    "devotion",
    "knowledge_and_wisdom",
    "equanimity",
    "relationships",
]

# 1. Verse — one record per verse in data/verses.json (700 in total)

class Verse(TypedDict):
    id: str                       # "2.47"  (chapter.verse)
    chapter: int                  # 2
    verse: int                    # 47
    sanskrit: str                 # Devanagari text
    transliteration: str          # Roman transliteration (IAST)
    translations: dict[str, str]  # {"en_source1": "...", "en_source2": "...", "hi_source1": "..."}
    commentary: str               # commentary text, "" if none
    source_urls: list[str]        # pages this record was built from

# 2. Link — one cross-reference between verses in data/links.json

class Link(TypedDict):
    from_id: str   # "2.47"  (stored in JSON as "from")
    to_id: str     # "3.19"  (stored in JSON as "to")
    source: str    # where the reference was found, e.g. "hi_commentary"


def link_from_json(record: dict) -> Link:
    """Convert a JSON link {"from", "to", "source"} into a Link.

    'from' is a reserved word in Python, so the JSON keys are renamed here.
    """
    return Link(from_id=record["from"], to_id=record["to"], source=record["source"])


def link_to_json(link: Link) -> dict:
    """Convert a Link back into the JSON format used in data/links.json."""
    return {"from": link["from_id"], "to": link["to_id"], "source": link["source"]}

# 3. SearchResult — what every search function returns (a list, best first)

class SearchResult(TypedDict):
    id: str       # verse id, e.g. "2.47"
    score: float  # higher = more relevant
    method: str   # "tfidf", "bm25", "dense", "rrf", "rerank", ...

# Validation

REQUIRED_VERSE_FIELDS = {
    "id": str,
    "chapter": int,
    "verse": int,
    "sanskrit": str,
    "transliteration": str,
    "translations": dict,
    "commentary": str,
    "source_urls": list,
}


def validate_verse(record: dict) -> list[str]:
    """Check one verse record and return a list of problems.

    An empty list means the record is valid. Used by crawler/merge.py
    before writing data/verses.json.
    """
    problems = []

    for field, expected_type in REQUIRED_VERSE_FIELDS.items():
        if field not in record:
            problems.append(f"missing field '{field}'")
        elif not isinstance(record[field], expected_type):
            problems.append(
                f"'{field}' should be {expected_type.__name__}, "
                f"got {type(record[field]).__name__}"
            )

    if problems:
        return problems

    # id must match chapter and verse, e.g. chapter 2 + verse 47 -> "2.47"
    expected_id = f"{record['chapter']}.{record['verse']}"
    if record["id"] != expected_id:
        problems.append(f"id '{record['id']}' does not match chapter.verse '{expected_id}'")

    # the Gita has 18 chapters
    if not 1 <= record["chapter"] <= 18:
        problems.append(f"chapter {record['chapter']} is outside 1-18")

    # at least one translation must be present and non-empty
    if not any(text.strip() for text in record["translations"].values()):
        problems.append("no non-empty translation")

    return problems


def make_result(verse_id: str, score: float, method: str) -> SearchResult:
    """Small helper so every search function builds results the same way."""
    return SearchResult(id=verse_id, score=float(score), method=method)