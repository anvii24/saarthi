"""Parse locally saved chapters of Swami Swarupananda's 1909 Gita edition.

The source archive's terms restrict automated copying. For that reason this
module does not download pages. Save chapter HTML pages manually in
``data/raw/en_source2/source_pages`` as ``chapter_01.html`` ...
``chapter_18.html`` and run this module. It checks each record against the
canonical verse IDs in ``data/verses.json`` when that file is available.

Unambiguously numbered passages are written to ``verses.json``. Passages
covering multiple verses (for example, 42-44) are preserved separately in
``grouped_passages.json`` and are never duplicated across individual verses.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from config import EN_SOURCE2_DIR, VERSES_FILE

SOURCE_INDEX_URL = "https://sacred-texts.com/book/srimad-bhagavad-gita/contents"
SOURCE_NAME = "Swami Swarupananda, Srimad-Bhagavad-Gita (1909)"

# Examples: "1. Text", "42-44. Text", "42–44. Text".
PASSAGE_START = re.compile(
    r"^\s*(\d{1,3})(?:\s*[-\u2013\u2014]\s*(\d{1,3}))?\s*[.)]\s*(.*?)\s*$"
)
PAGE_MARKER = re.compile(r"^\s*p\.?\s*\d+\s*$", re.IGNORECASE)


def _normalise_space(value: str) -> str:
    """Collapse whitespace and remove non-breaking spaces."""
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()



def _chapter_text(html: str) -> str:
    """Extract chapter text, excluding advertisements and footnotes."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements.
    for element in soup([
        "script", "style", "nav", "header", "footer", "aside"
    ]):
        element.decompose()

    # Remove advertisement elements.
    for element in soup.select(".ista-ad-frame-bar"):
        element.decompose()

    content = soup.select_one("div.contents") or soup.body or soup

    # Remove footnote references.
    for footnote in content.select("a[data-footnote-ref]"):
        footnote.decompose()

    # Remove continuation labels, but preserve the actual verse text.
    for marker in content.select("span"):
        if marker.get_text(strip=True) == "[paragraph continues]":
            marker.decompose()

    return content.get_text("\n", strip=True)



def parse_chapter_html(html: str, chapter: int, source_url: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse one chapter into individual and explicitly grouped passages."""
    if not 1 <= chapter <= 18:
        raise ValueError("chapter must be between 1 and 18")

    lines = _chapter_text(html).splitlines()
    individual: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    current_start: int | None = None
    current_end: int | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_end, current_parts
        if current_start is None:
            return
        text = _normalise_space(" ".join(current_parts))
        if text:
            record = {
                "chapter": chapter,
                "start_verse": current_start,
                "end_verse": current_end,
                "text": text,
                "source": SOURCE_NAME,
                "source_url": source_url,
            }
            if current_end == current_start:
                individual.append({
                    "id": f"{chapter}.{current_start}",
                    "chapter": chapter,
                    "verse": current_start,
                    "translations": {"en_source2": text},
                    "source_urls": [source_url] if source_url else [],
                })
            else:
                grouped.append(record)
        current_start = current_end = None
        current_parts = []

    for raw_line in lines:
        line = _normalise_space(raw_line)
        line = line.replace("[paragraph continues]", "").strip()
        if not line:
            continue
        if line.lower().startswith("the end of the") and "chapter" in line.lower():
            flush()
            break
        if PAGE_MARKER.match(line):
            continue

        match = PASSAGE_START.match(line)
        if match:
            start = int(match.group(1))
            end = int(match.group(2) or start)
            # The source combines verses 1.28 and 1.29 in one paragraph
            # labelled "29". Preserve the complete passage as a grouped range.
            if chapter == 1 and start == 29 and end == 29:
             start = 28
            # Verse labels in the main chapter are monotonically increasing.
            if current_start is not None and start <= (current_end or current_start):
                # Repeated numbering after the chapter body is likely front matter/notes.
                continue
            flush()
            current_start, current_end = start, end
            if match.group(3):
                current_parts.append(match.group(3))
            continue

        # Ignore chapter headings and speaker labels before the first numbered text.
        if current_start is not None:
            current_parts.append(line)

    flush()
    return individual, grouped


def _load_expected_ids(path: Path) -> set[str] | None:
    """Load canonical verse IDs when the merged dataset exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = data.get("verses", [])
    return {str(row["id"]) for row in data if isinstance(row, dict) and "id" in row}


def process_source_pages(source_dir: Path, output_dir: Path, expected_file: Path = VERSES_FILE) -> dict[str, Any]:
    """Parse all locally saved chapter pages and write outputs plus coverage."""
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    individual: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    missing_chapter_files: list[int] = []

    for chapter in range(1, 19):
        path = source_dir / f"chapter_{chapter:02d}.html"
        if not path.exists():
            missing_chapter_files.append(chapter)
            continue
        html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        canonical = soup.find("link", rel="canonical")
        chapter_url = canonical.get("href", "") if canonical else SOURCE_INDEX_URL
        one, many = parse_chapter_html(html, chapter, chapter_url)
        individual.extend(one)
        grouped.extend(many)

    individual.sort(key=lambda row: (row["chapter"], row["verse"]))
    grouped.sort(key=lambda row: (row["chapter"], row["start_verse"]))

    verses_path = output_dir / "verses.json"
    groups_path = output_dir / "grouped_passages.json"
    report_path = output_dir / "coverage_report.json"
    verses_path.write_text(json.dumps(individual, ensure_ascii=False, indent=2), encoding="utf-8")
    groups_path.write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")

    expected = _load_expected_ids(expected_file)

    found = {row["id"] for row in individual}

    # Include every verse covered by a grouped passage.
    grouped_found = {
     f"{row['chapter']}.{verse}"
     for row in grouped
     for verse in range(row["start_verse"], row["end_verse"] + 1)
}

    covered = found | grouped_found
    report = {
        "source": SOURCE_NAME,
        "individual_passages": len(individual),
        "grouped_passages": len(grouped),
        "missing_chapter_files": missing_chapter_files,
        "expected_verse_ids_checked": expected is not None,
        "missing_verse_ids": sorted(expected - covered) if expected is not None else [],
        "unexpected_verse_ids": sorted(found - expected) if expected is not None else [],
        "individual_verse_ids": len(found),
        "grouped_verse_ids": len(grouped_found),
        "total_covered_verse_ids": len(covered),
        "grouped_ranges": [
            f"{row['chapter']}.{row['start_verse']}-{row['end_verse']}" for row in grouped
        ],
        "note": "Grouped passages count toward verse coverage but remain separate from individual verse records.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report



def main() -> None:
    """Run the local HTML parser; no network requests are made."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=EN_SOURCE2_DIR / "source_pages")
    parser.add_argument("--output-dir", type=Path, default=EN_SOURCE2_DIR)
    args = parser.parse_args()

    report = process_source_pages(args.source_dir, args.output_dir)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["missing_chapter_files"]:
        print("Some chapter HTML files are missing. Add the missing local files and rerun.")
    else:
        print("All 18 local chapter HTML files were processed successfully.")



if __name__ == "__main__":
    main()
