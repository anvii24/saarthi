import argparse
import csv
import io
import json
import os
import re
import sys
from collections import Counter

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import THEMES  # noqa: E402

SHEET_ID = "175rEAvNylU5zIEZLepAJdzlGIc45UZeSS84czhJ-FUc"
SHEET_CSV = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid=0"
VERSES_PATH = os.path.join("data", "verses.json")
QUESTIONS_OUT = os.path.join("data", "test_set", "questions.json")
QRELS_OUT = os.path.join("data", "test_set", "qrels.json")

LABEL = re.compile(r"^\s*(\d{1,2}\.\d{1,3})\s*:\s*(\d)\s*$")


def load_rows(csv_path: str | None) -> list[dict]:
    if csv_path:
        with open(csv_path, encoding="utf-8") as f:
            text = f.read()
    else:
        response = requests.get(SHEET_CSV, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        text = response.text
    rows = list(csv.DictReader(io.StringIO(text)))
    # normalise header names: "relevant verses" / "relevant_verses" / "Relevant Verses"
    return [{k.strip().lower().replace(" ", "_"): (v or "").strip() for k, v in r.items() if k} for r in rows]


def parse_labels(cell: str, valid_ids: set[str]) -> tuple[dict[str, int], list[str]]:
    labels, problems = {}, []
    for part in filter(None, (p.strip() for p in cell.replace(";", ",").split(","))):
        match = LABEL.match(part)
        if not match:
            problems.append(f"can't read '{part}' (expected like 2.47:2)")
            continue
        vid, grade = match.group(1), int(match.group(2))
        if vid not in valid_ids:
            problems.append(f"verse {vid} does not exist")
        elif grade not in (1, 2):
            problems.append(f"{vid} has grade {grade} (use 1 or 2)")
        else:
            labels[vid] = grade
    return labels, problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="use a downloaded CSV instead of the live sheet")
    args = parser.parse_args()

    with open(VERSES_PATH, encoding="utf-8") as f:
        valid_ids = {v["id"] for v in json.load(f)}

    questions, qrels, problems = {}, {}, []
    for row in load_rows(args.csv):
        qid = row.get("id", "")
        if not qid or not row.get("en"):
            continue
        issues = []
        if row.get("theme") not in THEMES:
            issues.append(f"theme '{row.get('theme')}' is not one of the {len(THEMES)} themes")
        for lang in ("hi", "hinglish"):
            if not row.get(lang):
                issues.append(f"no {lang} version")
        labels, label_issues = parse_labels(row.get("relevant_verses", ""), valid_ids)
        issues += label_issues

        questions[qid] = {k: row.get(k, "") for k in ("theme", "en", "hi", "hinglish", "author")}
        if labels:
            qrels[qid] = labels
        problems += [f"{qid}: {issue}" for issue in issues]

    os.makedirs(os.path.dirname(QUESTIONS_OUT), exist_ok=True)
    with open(QUESTIONS_OUT, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(questions.items())), f, ensure_ascii=False, indent=2)
    with open(QRELS_OUT, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(qrels.items())), f, ensure_ascii=False, indent=2)

    authors = Counter(q["author"] or "?" for q in questions.values())
    themes = Counter(q["theme"] for q in questions.values())
    print(f"Saved {len(questions)} questions to {QUESTIONS_OUT}  (by author: {dict(authors)})")
    print(f"Saved labels for {len(qrels)} of them to {QRELS_OUT}  "
          f"({sum(len(l) for l in qrels.values())} relevant verses)")
    print(f"Questions per theme: {dict(sorted(themes.items()))}")
    if problems:
        print(f"\n{len(problems)} problem(s) to fix in the sheet:")
        for p in problems:
            print("  -", p)
    else:
        print("No problems found.")


if __name__ == "__main__":
    main()
