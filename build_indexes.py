
import json
from indexing.inverted_index import InvertedIndex

with open("data/verses.json", "r", encoding="utf-8") as file:
    verses = json.load(file)

en_index = InvertedIndex()
en_index.build_index(verses, "en")
en_index.save("data/processed/index_en.json")

hi_index = InvertedIndex()
hi_index.build_index(verses, "hi")
hi_index.save("data/processed/index_hi.json")

print("English index built")
print("English terms:", len(en_index.index["translation"]))
print("Hindi index built")
print("Hindi terms:", len(hi_index.index["translation"]))
print("Total verses indexed:", len(verses))
