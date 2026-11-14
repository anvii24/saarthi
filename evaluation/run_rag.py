import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.fact_check import CITATION, verify  
from generation.rag import explain  
from indexing.embeddings import search_dense  
from query.rewrite import rewrite 

QUESTIONS = os.path.join("data", "test_set", "questions.json")
QRELS = os.path.join("data", "test_set", "qrels.json")
VERSES = os.path.join("data", "verses.json")
OUT_DIR = os.path.join("evaluation", "results")
SENTENCES_CSV = os.path.join(OUT_DIR, "rag_sentences.csv")
TOP_N = 5
THRESHOLDS = [0.3, 0.5, 0.7]


def retrieve(question: str) -> list[str]:
    concepts = rewrite(question)
    query = f"{question} {' '.join(concepts)}" if concepts else question
    return [r["id"] for r in search_dense(query, k=TOP_N)]


def pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


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
    print(f"Running retrieve -> explain -> verify on {len(ids)} questions")

    answers, sentence_rows = [], []
    for n, qid in enumerate(ids, start=1):
        print(f"[{n}/{len(ids)}] {qid}")
        question = questions[qid]["en"]
        retrieved = retrieve(question)
        given = [verses[v] for v in retrieved]
        answer = explain(question, given)
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
                                      "human_supported": ""})

    all_checks = [c for a in answers for c in a["checks"]]
    cited = [c for c in all_checks if c["verse_ids"]]
    supported = [c for c in cited if c["supported"]]
    cited_verses = [(a["id"], v) for a in answers for v in dict.fromkeys(CITATION.findall(a["answer"]))]
    relevant_cited = [1 for qid, v in cited_verses if v in qrels[qid]]
    retrieval_p5 = sum(len(set(a["retrieved"]) & set(qrels[a["id"]])) / TOP_N for a in answers) / len(answers)

    summary = {
        "questions": len(answers),
        "sentences": len(all_checks),
        "cited_sentences": len(cited),
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
    with open(os.path.join(OUT_DIR, "rag.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())
    with open(os.path.join(OUT_DIR, "rag_answers.json"), "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    if sentence_rows:
        with open(SENTENCES_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sentence_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sentence_rows)

    print()
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print(f"\nSaved {OUT_DIR}/rag.csv, rag_answers.json and rag_sentences.csv")
    print("Next: hand-check 20+ rows of rag_sentences.csv, then run with --score-review")


def score_review():
    if not os.path.exists(SENTENCES_CSV):
        sys.exit("Run python evaluation/run_rag.py first")
    with open(SENTENCES_CSV, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("human_supported", "").strip().lower() in ("yes", "no")]
    if not rows:
        sys.exit("No rows have yes/no in human_supported yet")

    tp = sum(1 for r in rows if r["checker_supported"] == "yes" and r["human_supported"].lower() == "yes")
    tn = sum(1 for r in rows if r["checker_supported"] == "no" and r["human_supported"].lower() == "no")
    fp = sum(1 for r in rows if r["checker_supported"] == "yes" and r["human_supported"].lower() == "no")
    fn = sum(1 for r in rows if r["checker_supported"] == "no" and r["human_supported"].lower() == "yes")
    n = len(rows)
    # "positive" = the checker flags a sentence as UNSUPPORTED 
    caught_precision = tn / (tn + fn) if tn + fn else 0.0
    caught_recall = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * caught_precision * caught_recall / (caught_precision + caught_recall) if caught_precision + caught_recall else 0.0

    print(f"Hand-checked sentences: {n}")
    print(f"  checker agrees with you:          {pct(tp + tn, n)}%  (accuracy)")
    print(f"  wrongly kept (you said no):       {fp}")
    print(f"  wrongly removed (you said yes):   {fn}")
    print(f"  catching unsupported claims:      precision {caught_precision:.2f}, recall {caught_recall:.2f}, F1 {f1:.2f}")
    with open(os.path.join(OUT_DIR, "rag_checker_review.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows([("hand_checked", n), ("accuracy_pct", pct(tp + tn, n)), ("wrongly_kept", fp),
                          ("wrongly_removed", fn), ("unsupported_precision", round(caught_precision, 3)),
                          ("unsupported_recall", round(caught_recall, 3)), ("unsupported_f1", round(f1, 3))])
    print(f"Saved {OUT_DIR}/rag_checker_review.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", default="q000")
    parser.add_argument("--to", dest="end", default="q999")
    parser.add_argument("--score-review", action="store_true")
    args = parser.parse_args()
    score_review() if args.score_review else run(args)


if __name__ == "__main__":
    main()