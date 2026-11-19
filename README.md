# Saarthi

Ask an everyday question in **English, Hindi or Hinglish** and get an answer from the
Bhagavad Gita, built only from real verses, with every claim about a verse checked
against that verse.

> *"exam mein fail ho gaya, ab kya karun?"* → Krishna says that one has a right to the
> work, but never to its fruits **[2.47]** ✓ …

| Team member | Part |
|---|---|
| Antara | Data collection, inverted index, link analysis (PageRank, HITS), Explore page |
| Kashvi | BM25 / TF-IDF / Boolean search, spelling, theme classifier, retrieval evaluation, Advanced Search and Compare pages |
| Anvi | Multilingual queries, LLM query rewriting, meaning search, answers and fact-checking, pipeline, app design |

## How it works

```
question (en / hi / hinglish)
  → detect language, translate to English          query/language.py
  → predict theme (11 themes)                       query/classifier.py
  → rewrite into Gita concepts with an LLM          query/rewrite.py
  → BM25 keyword search  +  multilingual meaning search
                                                    retrieval/bm25.py, indexing/embeddings.py
  → Reciprocal Rank Fusion                          retrieval/fusion.py
  → re-rank with PageRank + topic PageRank          retrieval/rerank.py, indexing/link_analysis.py
  → LLM answer from the top 5 verses, citing them   generation/rag.py
  → NLI fact-checker drops unsupported sentences    generation/fact_check.py
  → shown in English, Hindi or Hinglish             generation/translate.py, app/
```

All of it is joined in `pipeline.py` (`ask(question)`).

## Results (80 holdout questions, never used for tuning)

**Each part of the search adds something** (nDCG@10, `evaluation/results/retrieval.csv`)

| Setup | nDCG@10 |
|---|---|
| TF-IDF, question only | 0.054 |
| BM25, question only | 0.074 |
| BM25 + Gita concepts | 0.146 |
| Meaning search + Gita concepts | 0.206 |
| + BM25, fused with RRF | 0.198 |
| + PageRank | 0.216 |
| **+ topic PageRank (full Saarthi)** | **0.235** |

**Saarthi vs plain RAG** (`evaluation/results/baseline.csv`)

| | Plain RAG | Saarthi |
|---|---|---|
| P@5 of the verses given to the LLM | 0.105 | **0.150** |
| Cited sentences supported by their verse | 62.7% | **94.6%** |
| Unsupported sentences shown per answer | 2.14 | **0** |

**Hindi and Hinglish questions**: translating to English first brings them level with
English (nDCG@10: Hinglish 0.047 → 0.153, Hindi 0.077 → 0.167; English 0.159).

**Fact-checker**: agreed with our hand checks on 97.1% of 34 sentences, and never kept an
unsupported one (`rag_checker_review.csv`). The stricter v2 answer prompt raised supported
sentences from 60.7% to 93.6% (`rag_compare.csv`).

## Setup

Python 3.10+. From the repo root:

```
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"
```

Create a `.env` file (copy `.env.example`) with a free Gemini API key from
aistudio.google.com. `.env` is ignored by git, so keys are never pushed.

```
LLM_API_KEY=your-key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.1-flash-lite
```

Build the indexes once (the meaning-search index takes a few minutes; the models,
about 2 GB in total, download on first use):

```
python build_indexes.py
python indexing/embeddings.py
```

## Run

```
streamlit run app/Home.py                                  # the app
python pipeline.py "exam mein fail ho gaya, kya karun?"    # one question in the terminal
python -m pytest -q                                        # tests
```

The models load in the background when the app opens, so give it about 30 seconds before
the first question.

| Page | What it does |
|---|---|
| Home | Pick a feeling or ask your own question; a verse to read; how Saarthi works |
| Ask | Answer with cited verses, shown in English, Hindi or Hinglish; every verse found |
| Explore | Read any verse with its Sanskrit, translations and commentary |
| Advanced Search | Boolean and phrase search over the index, with spelling suggestions |
| Compare | One question, four rankings side by side, to see what each search step changes |

## Reproduce the results

```
python evaluation/import_test_set.py       # test set from the shared sheet
python evaluation/tune_bm25.py             # BM25 k1, b (tuning questions)
python evaluation/run_retrieval.py         # search ablation + re-rank weights
python evaluation/run_classifiers.py       # theme classifier
python evaluation/run_graph_alpha.py       # graph alpha (tuning questions)
python evaluation/run_graph.py             # PageRank / topic PageRank / HITS
python evaluation/run_multilingual.py      # English vs Hindi vs Hinglish
python evaluation/run_rag.py --prompt v1   # then --prompt v2, --compare, --score-review
python evaluation/baseline.py              # plain RAG vs Saarthi
```

Results are saved in `evaluation/results/`. LLM replies are cached, so re-runs are quick.

## Data

- `data/verses.json`: 701 verses, each with Sanskrit, transliteration, up to 9 translations
  (7 English, 2 Hindi), English and Hindi commentary, and a plain-English explanation
  (652 verses).
- `data/links.json`: 1,716 verse-to-verse references found in the commentaries (the graph).
- `data/test_set/`: 100 questions in English, Hindi and Hinglish with graded relevance
  labels; 20 for tuning (q035–q054) and 80 held out. Edited only in the shared sheet.

Sources: [vedicscriptures/bhagavad-gita-data](https://github.com/vedicscriptures/bhagavad-gita-data)
(GPL-3.0), [AdiyogiArts/gita-json](https://github.com/AdiyogiArts/gita-json) (MIT), and
Swami Swarupananda's 1909 translation (parsed from pages saved by hand). Details and
licence notes: `data/SOURCES.md`.

## Folders

| Folder | What is in it |
|---|---|
| `crawler/` | download the sources and merge them into `data/` |
| `preprocessing/` | English and Hindi text cleaning |
| `indexing/` | inverted index, meaning-search index, link graph |
| `query/` | language, spelling, theme, query rewriting |
| `retrieval/` | BM25, TF-IDF, Boolean search, fusion, re-ranking |
| `generation/` | LLM calls, answers, fact-checking, translation |
| `app/` | Streamlit app: `Home.py`, `pages/`, `components/` (style, decorations, answer panel), `static/decor/` (images from the team's mood board) |
| `evaluation/` | every experiment and its results |
| `tests/` | unit tests |
