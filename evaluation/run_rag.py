"""
RAG + fact-check evaluation

For every labelled test question (English version):
  1. retrieve   top 5 verses: meaning search on the question + its Gita concepts
                (rewrite.py). Swap in the full fused pipeline once BM25 exists.
  2. explain    generation/rag.py writes an answer citing [chapter.verse]
                (--prompt v1 or v2: see the top of generation/rag.py)
  3. verify     generation/fact_check.py checks every cited sentence against its verse

"""

import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.fact_check import CITATION, verify  # noqa: E402
from generation.rag import explain  # noqa: E402
from indexing.embeddings import search_dense  # noqa: E402
from query.rewrite import rewrite  # noqa: E402

QUESTIONS = os.path.join("data", "test_set", "questions.json")
QRELS = os.path.join("data", "test_set", "qrels.json")
VERSES = os.path.join("data", "verses.json")
OUT_DIR = os.path.join("evaluation", "results")
TOP_N = 5
THRESHOLDS = [0.3, 0.5, 0.7]


def paths(version: str) -> dict:
    suffix = "" if version == "v1" else f"_{version}"
    base = os.path.join(OUT_DIR, f"rag{suffix}")
    return {"summary": base + ".csv", "answers": base + "_answers.json",
            "sentences": base + "_sentences.csv", "review": base + "_checker_review.csv"}


def retrieve(question: str) -> list[str]:
    concepts = rewrite(question)
    query = f"{question} {' '.join(concepts)}" if concepts else question
    return [r["id"] for r in search_dense(query, k=TOP_N)]


def pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


def existing_checks(path: str) -> dict:
    """(id, sentence) -> yes/no already typed into human_supported, so reruns keep them."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {(r["id"], r["sentence"]): r["human_supported"].strip()
                for r in csv.DictReader(f) if r.get("human_supported", "").strip()}


def run(args):
    with open(QUESTIONS, encoding="utf-8") as f:
        questions = json.load(f)
    with open(QRELS, encoding="utf-8") as f:
        qrels = json.load(f)
    with open(VERSES, encoding="utf-8") as f:
        verses = {v["id"]: v for v in json.load(f)}

    ids = [q for q in sorted(qrels) if args.start <= q <= args.end and q in questions]
    if not ids:
        sys.exit("No labelled questions found: run python evaluation/import_test_set.py first")
    files = paths(args.prompt)
    kept = existing_checks(files["sentences"])
    print(f"Running retrieve -> explain ({args.prompt}) -> verify on {len(ids)} questions")

    answers, sentence_rows = [], []
    for n, qid in enumerate(ids, start=1):
        print(f"[{n}/{len(ids)}] {qid}")
        question = questions[qid]["en"]
        retrieved = retrieve(question)
        given = [verses[v] for v in retrieved]
        answer = explain(question, given, prompt_version=args.prompt)
        checks = verify(answer["text"], given)
        answers.append({"id": qid, "question": question, "retrieved": retrieved,
                        "answer": answer["text"], "safety_note": answer["safety_note"], "checks": checks})
        for c in checks:
            if c["verse_ids"]:
                verse_text = " / ".join(
                    verses[v]["translations"].get("en_gambirananda", "") for v in c["verse_ids"] if v in verses)
                sentence_rows.append({"id": qid, "sentence": c["sentence"], "verse_ids": " ".join(c["verse_ids"]),
                                      "verse_text": verse_text, "checker_score": c["score"],
                                      "checker_supported": "yes" if c["supported"] else "no",
                                      "human_supported": kept.get((qid, c["sentence"]), "")})

    # summary
    all_checks = [c for a in answers for c in a["checks"]]
    cited = [c for c in all_checks if c["verse_ids"]]
    supported = [c for c in cited if c["supported"]]
    cited_verses = [(a["id"], v) for a in answers for v in dict.fromkeys(CITATION.findall(a["answer"]))]
    relevant_cited = [1 for qid, v in cited_verses if v in qrels[qid]]
    retrieval_p5 = sum(len(set(a["retrieved"]) & set(qrels[a["id"]])) / TOP_N for a in answers) / len(answers)

    summary = {
        "prompt": args.prompt,
        "questions": len(answers),
        "sentences": len(all_checks),
        "cited_sentences": len(cited),
        "uncited_pct": pct(len(all_checks) - len(cited), len(all_checks)),
        "supported_pct": pct(len(supported), len(cited)),
        "removed_total": len(cited) - len(supported),
        "removed_per_answer": round((len(cited) - len(supported)) / len(answers), 2),
        "answers_fully_supported_pct": pct(
            sum(1 for a in answers if all(c["supported"] is not False for c in a["checks"])), len(answers)),
        "citation_relevance_pct": pct(len(relevant_cited), len(cited_verses)),
        "retrieval_P@5": round(retrieval_p5, 3),
        "safety_notes_added": sum(1 for a in answers if a["safety_note"]),
    }
    for t in THRESHOLDS:
        summary[f"supported_pct@{t}"] = pct(sum(1 for c in cited if (c["score"] or 0) >= t), len(cited))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(files["summary"], "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())
    with open(files["answers"], "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    if sentence_rows:
        with open(files["sentences"], "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sentence_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sentence_rows)

    print()
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    still = sum(1 for r in sentence_rows if r["human_supported"])
    print(f"\nSaved {files['summary']}, {files['answers']} and {files['sentences']}")
    print(f"({still} hand checks kept from before)" if still else
          "Next: hand-check 20+ rows of the sentences file, then run with --score-review")


def score_review(args):
    files = paths(args.prompt)
    if not os.path.exists(files["sentences"]):
        sys.exit(f"Run python evaluation/run_rag.py --prompt {args.prompt} first")
    with open(files["sentences"], encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("human_supported", "").strip().lower() in ("yes", "no")]
    if not rows:
        sys.exit("No rows have yes/no in human_supported yet")

    tp = sum(1 for r in rows if r["checker_supported"] == "yes" and r["human_supported"].lower() == "yes")
    tn = sum(1 for r in rows if r["checker_supported"] == "no" and r["human_supported"].lower() == "no")
    fp = sum(1 for r in rows if r["checker_supported"] == "yes" and r["human_supported"].lower() == "no")
    fn = sum(1 for r in rows if r["checker_supported"] == "no" and r["human_supported"].lower() == "yes")
    n = len(rows)
    # "positive" = the checker flags a sentence as UNSUPPORTED (that is its job: catching bad claims)
    caught_precision = tn / (tn + fn) if tn + fn else 0.0
    caught_recall = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * caught_precision * caught_recall / (caught_precision + caught_recall) if caught_precision + caught_recall else 0.0

    print(f"Hand-checked sentences ({args.prompt}): {n}")
    print(f"  checker agrees with you:          {pct(tp + tn, n)}%  (accuracy)")
    print(f"  wrongly kept (you said no):       {fp}")
    print(f"  wrongly removed (you said yes):   {fn}")
    print(f"  catching unsupported claims:      precision {caught_precision:.2f}, recall {caught_recall:.2f}, F1 {f1:.2f}")
    with open(files["review"], "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows([("hand_checked", n), ("accuracy_pct", pct(tp + tn, n)), ("wrongly_kept", fp),
                          ("wrongly_removed", fn), ("unsupported_precision", round(caught_precision, 3)),
                          ("unsupported_recall", round(caught_recall, 3)), ("unsupported_f1", round(f1, 3))])
    print(f"Saved {files['review']}")


def compare():
    tables = {}
    for version in ("v1", "v2"):
        path = paths(version)["summary"]
        if not os.path.exists(path):
            sys.exit(f"Missing {path}: run python evaluation/run_rag.py --prompt {version} first")
        with open(path, encoding="utf-8") as f:
            tables[version] = {r["metric"]: r["value"] for r in csv.DictReader(f)}
    rows = [m for m in tables["v1"] if m != "prompt"]
    print(f"{'metric':<30}{'v1':>10}{'v2':>10}")
    for m in rows:
        print(f"{m:<30}{tables['v1'].get(m, ''):>10}{tables['v2'].get(m, ''):>10}")
    with open(os.path.join(OUT_DIR, "rag_compare.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "v1", "v2"])
        writer.writerows([(m, tables["v1"].get(m, ""), tables["v2"].get(m, "")) for m in rows])
    print(f"Saved {os.path.join(OUT_DIR, 'rag_compare.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v1")
    parser.add_argument("--from", dest="start", default="q000")
    parser.add_argument("--to", dest="end", default="q999")
    parser.add_argument("--score-review", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare()
    elif args.score_review:
        score_review(args)
    else:
        run(args)


if __name__ == "__main__":
    main()