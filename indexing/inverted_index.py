
from gettext import translation
import re
import json
from collections import defaultdict
from pathlib import Path

from preprocessing.clean_en import clean_en
from preprocessing.clean_hi import clean_hi
from preprocessing.normalise import normalise_sanskrit


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.document_lengths = defaultdict(dict)
        self.document_frequencies = defaultdict(dict)

    def build_index(self, verses, lang):
        self.index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.document_lengths = defaultdict(dict)
        self.document_frequencies = defaultdict(dict)

        for verse in verses:
            verse_id = verse["id"]

            if lang == "en":
                translation = verse.get("translations", {}).get("en_prabhupada", "")
                translation_tokens = clean_en(translation)
            elif lang == "hi":
                translation = verse.get("translations", {}).get("hi_tejomayananda", "")
                translation_tokens = clean_hi(translation)
            else:
                raise ValueError("Language must be 'en' or 'hi'")

            if lang == "en":
                commentary = verse.get("commentary", "")
                commentary_tokens = clean_en(commentary)
            else:
                commentary = verse.get("commentary_hi", "")
                commentary_tokens = clean_hi(commentary)

            transliteration = verse.get("transliteration", "")
            sanskrit_tokens = [
                normalise_sanskrit(token.lower())
                for token in re.findall(r"[a-zA-Zāīūṛṝḷṃḥṅñṭḍṇśṣ]+", transliteration.lower())
            ]

            zones = {
                "translation": translation_tokens,
                "commentary": commentary_tokens,
                "sanskrit": sanskrit_tokens
            }

            for zone, tokens in zones.items():
                self.document_lengths[zone][verse_id] = len(tokens)

                for position, token in enumerate(tokens):
                    self.index[zone][token][verse_id].append(position)

        for zone, terms in self.index.items():
            self.document_frequencies[zone] = {
                term: len(postings)
                for term, postings in terms.items()
            }

    def get_postings(self, term, zone=None):
        if zone:
            return dict(self.index[zone].get(term, {}))

        result = {}
        for current_zone, terms in self.index.items():
            if term in terms:
                result[current_zone] = dict(terms[term])

        return result

    def doc_length(self, verse_id, zone="translation"):
        return self.document_lengths[zone].get(verse_id, 0)

    def doc_freq(self, term, zone="translation"):
        return self.document_frequencies[zone].get(term, 0)

    def phrase_search(self, phrase, zone="translation", lang="en"):
        tokens = clean_en(phrase) if lang == "en" else clean_hi(phrase)

        if not tokens:
            return []

        first_postings = self.index[zone].get(tokens[0], {})
        matches = []

        for verse_id, positions in first_postings.items():
            for position in positions:
                if all(
                    position + offset in self.index[zone].get(token, {}).get(verse_id, [])
                    for offset, token in enumerate(tokens[1:], start=1)
                ):
                    matches.append(verse_id)
                    break

        return matches

    def save(self, path="data/processed/index.json"):
        data = {
            "index": {
                zone: {
                    term: dict(postings)
                    for term, postings in terms.items()
                }
                for zone, terms in self.index.items()
            },
            "document_lengths": {
                zone: dict(lengths)
                for zone, lengths in self.document_lengths.items()
            },
            "document_frequencies": {
                zone: dict(frequencies)
                for zone, frequencies in self.document_frequencies.items()
            }
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

    def load(self, path="data/processed/index.json"):
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.document_lengths = defaultdict(dict)
        self.document_frequencies = defaultdict(dict)

        for zone, terms in data["index"].items():
            for term, postings in terms.items():
                self.index[zone][term] = defaultdict(
                    list,
                    {verse_id: positions for verse_id, positions in postings.items()}
                )

        for zone, lengths in data["document_lengths"].items():
            self.document_lengths[zone] = lengths

        for zone, frequencies in data["document_frequencies"].items():
            self.document_frequencies[zone] = frequencies
