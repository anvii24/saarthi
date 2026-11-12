# Data Sources and Licences

Maintained by Kashvi (P2). Last updated: 12 Nov 2026.
Status key: **[verified]** = checked against the original page by a team member; **[verify]** = not yet checked, do not cite in the report until confirmed.

## 1. Where our data comes from

Sources A and B below are downloaded as static JSON files from public GitHub repositories (via raw.githubusercontent.com), using `crawler/base_crawler.py`. The additional source C (`en_source2`) is handled separately as locally saved HTML because the archive's automated-access terms have not been cleared.

| # | Repository | What we use | Repo licence | Crawled by |
|---|---|---|---|---|
| A | github.com/vedicscriptures/bhagavad-gita-data | Sanskrit, transliteration, 6 English + 2 Hindi translations, English + Hindi commentary (all 701 verses) | GNU General Public License v3.0 (29 June 2007). Verified from the repo's LICENSE file on 12 Nov 2026. [verified] | `crawl_hi_commentary.py` + `merge.py` |
| B | github.com/AdiyogiArts/gita-json | Sanskrit, transliteration, short plain-English explanation (used as `explanation_en`) | MIT License, Copyright (c) 2024 Adiyogi Arts. Verified from the repo's LICENSE file on 12 Nov 2026. [verified]| `crawl_en_source1.py` |

Note: this dataset has 701 verses (35 in chapter 13), not 700. See `merge.py`.

## 2. robots.txt check

| Host | URL checked | Result | Date |
|---|---|---|---|
| raw.githubusercontent.com | https://raw.githubusercontent.com/robots.txt | HTTP 404 reported by the earlier check; re-check before any new crawl | 12 Nov 2026 |
| github.com | https://github.com/robots.txt | Not used by the existing raw-file crawlers; check before fetching GitHub HTML pages | Not checked |
| sacred-texts.com | https://sacred-texts.com/robots.txt | **Not verified. Do not automate requests until checked manually.** | Pending |

The Sacred Texts archive is subject to separate site terms. Its text page says the 1909 book is public domain in the United States, but that statement is not an India-specific legal opinion. The archive's terms have also been reported to restrict automated copying. Therefore `crawl_en_source2.py` currently parses **locally saved HTML only** and makes no network requests. Do not bypass this restriction or use `base_crawler.fetch()` against the archive without confirming current robots.txt and obtaining permission for the intended request volume.

The GitHub crawlers use user-agent `SaarthiBot/1.0 (educational research project)` and a 1.5 s delay between requests; `base_crawler.can_fetch()` checks robots.txt before every fetch. The Sacred Texts parser makes no requests and therefore does not rely on the GitHub crawler's robots.txt check.

## 3. Underlying translators and commentators (dataset A)

The repo being freely downloadable does not by itself mean the underlying books are public domain. Years and publishers below are from memory and are **not verified**.

| Key in verses.json | Author | Work / publisher (verify) | Year (verify) | Public-domain status |
|---|---|---|---|---|
| en_sivananda | Swami Sivananda | Divine Life Society | 1939 (first edition) | Likely public domain |
| en_purohit | Shri Purohit Swami | Faber & Faber (The Geeta) | 1935 | Likely public domain in India |
| en_gambirananda | Swami Gambhirananda | Advaita Ashrama (Shankara's commentary) | 1984 (first edition) | In copyright |
| en_adidevananda | Swami Adidevananda | Sri Ramakrishna Math (Ramanuja's commentary) | 2001 (documented edition) | In copyright |
| en_sankaranarayan | Dr. S. Sankaranarayan | Sri Venkateswara University, Tirupati | 1985 | Undetermined |
| en_prabhupada | A.C. Bhaktivedanta Swami Prabhupada | Bhaktivedanta Book Trust (As It Is) | 1968 (original edition) | In copyright |
| hi_tejomayananda | Swami Tejomayananda | Chinmaya Mission | 1993 | In copyright |
| hi_ramsukhdas | Swami Ramsukhdas | Gita Press, Gorakhpur | 1995 (documented edition) | In copyright |

The original Sanskrit verses are ancient text and in the public domain.

## 4. Limitations to state in the report

- Several translations and commentaries may still be in copyright. The repository owners distribute them openly, but we have not obtained separate permission from the original publishers.
- `en_source2` is attributed to Swami Swarupananda by the 1909 edition, but the online archive notes an authorship debate. The archive's US public-domain statement is not an India-specific legal opinion. Do not redistribute the extracted text until the team/instructor has reviewed the rights position and archive terms.
- Our use is educational and non-commercial (a university course project). We do not redistribute the dataset as our own work, and we credit every source here and in the report references.
- If the course or an examiner objects, the fallback is to restrict the system to the Sanskrit text and the translations confirmed as public domain.



## 5. Additional English translation (Person 2: `en_source2`)

| Field | Details |
|---|---|
| Work | *Srimad-Bhagavad-Gita* |
| Byline / compiler | Swami Swarupananda; the first-edition preface says he compiled the edition with collaborating sannyasins and Western disciples of Swami Vivekananda |
| Publisher / imprint | Advaita Ashrama, Mayavati, Almora, Himalayas |
| Publication | 18 October 1909 (date stated in the first-edition preface) |
| Primary online text | https://sacred-texts.com/book/srimad-bhagavad-gita/start-reading |
| Contents / chapter links | https://sacred-texts.com/book/srimad-bhagavad-gita/contents |
| Rights status | **Provisional / verify before redistribution.** The archive describes the text as public domain in the US because it predates 1923. That does not by itself establish status in every jurisdiction. The archive's index also notes debate over attribution. The first-edition preface itself attributes compilation to Swami Swarupananda and collaborators. |
| Crawling method | Manual/local HTML capture only. The current `crawler/crawl_en_source2.py` does not download pages. Confirm current robots.txt and archive terms, and obtain permission if automated collection is desired. |
| Data handling | Individual numbered passages are mapped to `chapter.verse`. Combined passages such as `42-44` are stored separately in `grouped_passages.json`; they are not copied into three verse records. |

**Important limitation:** the edition is a close prose rendering, but not every numbered passage necessarily corresponds one-to-one with a Sanskrit shloka. The parser preserves explicit ranges and the coverage report flags missing IDs. A human must review grouped passages and numbering before the translation is merged into `data/verses.json`.

**Authorship note:** the archive's index mentions a debate about whether Sister Nivedita (Margaret Noble) may have contributed to or authored the translation. The first-edition preface describes Swami Swarupananda as compiler and notes collaboration. We retain the archive's byline for attribution but do not claim the attribution debate is settled.
