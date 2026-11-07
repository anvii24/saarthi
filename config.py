
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw data sources
EN_SOURCE1_DIR = RAW_DIR / "en_source1"
EN_SOURCE2_DIR = RAW_DIR / "en_source2"
HI_COMMENTARY_DIR = RAW_DIR / "hi_commentary"

# Dataset files
VERSES_FILE = DATA_DIR / "verses.json"
LINKS_FILE = DATA_DIR / "links.json"
THEMES_FILE = DATA_DIR / "verse_themes.json"

# Test dataset
TEST_SET_DIR = DATA_DIR / "test_set"
QUESTIONS_FILE = TEST_SET_DIR / "questions.json"
QRELS_FILE = TEST_SET_DIR / "qrels.json"

# Search configuration
TOP_K = 20
