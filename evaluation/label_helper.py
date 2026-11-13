"""
Labelling helper: suggest candidate verses for test questions (Person 3: Anvi).

For each question it runs three searches and pools the results:
  1. meaning search on the English question
  2. meaning search on the English question + its Gita concepts (rewrite.py)
  3. meaning search on the Hindi question
Then it writes a readable file listing, per question, up to 12 candidate verses
with a translation and the plain-English explanation, so you can pick the
2-6 verses that really answer it and grade them (2 = directly answers,
1 = related) in the sheet's "relevant verses" column, e.g.  2.47:2, 2.48:2, 2.38:1

Important: these are only SUGGESTIONS from our own system. Always read them and
judge for yourself, and add good verses that are missing (the commentaries and
explanations help). Labels copied blindly from the system would make our
evaluation meaningless, since it would just be grading itself.

Questions are read straight from the shared Google Sheet (it must be shared as
"anyone with the link"), or from a CSV you downloaded with --csv.

Run from the repo root:
    python evaluation/label_helper.py --from q067 --to q100
    python evaluation/label_helper.py --from q067 --to q100 --csv questions.csv
Output: data/test_set/label_candidates_<from>-<to>.md
"""

import argparse
import csv
import io
import json
import os
import sys

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indexing.embeddings import search_dense  # noqa: E402
from query.rewrite import rewrite  # noqa: E402

SHEET_ID = "175rEAvNylU5zIEZLepAJdzlGIc45UZeSS84czhJ-FUc"
SHEET_CSV = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid=0"
VERSES_PATH = os.path.join("data", "verses.json")
OUT_DIR = os.path.join("data", "test_set")
PER_SEARCH = 10
MAX_CANDIDATES = 12


def load_questions(csv_path: str | None) -> list[dict]:
    if csv_path:
        with open(csv_path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    response = requests.get(SHEET_CSV, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    return list(csv.DictReader(io.StringIO(response.text)))


def candidates_for(q: dict) -> list[tuple[str, float, set[str]]]:
    """Pool the three searches; returns (verse_id, best_score, which_searches) best first."""
    en, hi = q.get("en", "").strip(), q.get("hi", "").strip()
    searches = {}
    if en:
        searches["en"] = search_dense(en, k=PER_SEARCH)
        concepts = rewrite(en)
        if concepts:
            searches["en+concepts"] = search_dense(f"{en} {' '.join(concepts)}", k=PER_SEARCH)
    if hi:
        searches["hi"] = search_dense(hi, k=PER_SEARCH)

    pooled: dict[str, tuple[float, set[str]]] = {}
    for name, results in searches.items():
        for r in results:
            score, found_by = pooled.get(r["id"], (0.0, set()))
            pooled[r["id"]] = (max(score, r["score"]), found_by | {name})

    ranked = sorted(pooled.items(), key=lambda item: (len(item[1][1]), item[1][0]), reverse=True)
    return [(vid, score, found_by) for vid, (score, found_by) in ranked[:MAX_CANDIDATES]]


def short(text: str, limit: int = 260) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", required=True, help="first id, e.g. q067")
    parser.add_argument("--to", dest="end", required=True, help="last id, e.g. q100")
    parser.add_argument("--csv", help="use a downloaded CSV instead of the live sheet")
    args = parser.parse_args()

    with open(VERSES_PATH, encoding="utf-8") as f:
        verses = {v["id"]: v for v in json.load(f)}

    questions = [q for q in load_questions(args.csv)
                 if args.start <= q.get("id", "") <= args.end and q.get("en", "").strip()]
    if not questions:
        sys.exit(f"No questions with English text found between {args.start} and {args.end}")

    lines = [f"# Candidate verses for {args.start}-{args.end}",
             "",
             "Pick the 2-6 verses that really answer each question and write them in the sheet's",
             "'relevant verses' column as `2.47:2, 2.48:1` (2 = directly answers, 1 = related).",
             "These are suggestions only: judge them yourself and add good verses that are missing.",
             ""]

    for n, q in enumerate(questions, start=1):
        print(f"[{n}/{len(questions)}] {q['id']}")
        lines += [f"## {q['id']} · {q.get('theme', '')}", "", f"**{q['en'].strip()}**", ""]
        for vid, score, found_by in candidates_for(q):
            v = verses.get(vid, {})
            translation = v.get("translations", {}).get("en_gambirananda") or next(
                (t for k, t in v.get("translations", {}).items() if k.startswith("en_")), "")
            lines.append(f"- **{vid}** (score {score:.2f}, found by: {', '.join(sorted(found_by))})")
            lines.append(f"  - {short(translation)}")
            if v.get("explanation_en"):
                lines.append(f"  - *Explanation:* {short(v['explanation_en'], 220)}")
        lines += ["", "Your labels: ", "", "---", ""]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"label_candidates_{args.start}-{args.end}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {out}  (open it in VS Code; Ctrl+Shift+V shows it formatted)")


if __name__ == "__main__":
    main()
