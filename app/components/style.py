"""Saarthi's look: warm Indian colours, fonts and small shared pieces (brand header, chips).

Call apply_style() at the top of every page, right after st.set_page_config().
"""

import html

import streamlit as st

SPEAKERS = {"श्रीभगवान्": "Krishna", "अर्जुन": "Arjuna", "सञ्जय": "Sanjaya", "धृतराष्ट्र": "Dhritarashtra"}
LANG_NAMES = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}

FONTS = ("https://fonts.googleapis.com/css2?family=Yatra+One&family=Cormorant+Garamond:wght@500;600;700"
         "&family=Mukta:wght@400;500;600&family=Tiro+Devanagari+Sanskrit&display=swap")

CSS = f"""
<style>
@import url('{FONTS}');

/* body text (icons keep their own font, so only text elements are listed) */
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button p, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {{
    font-family: 'Mukta', 'Segoe UI', sans-serif;
}}
.stApp {{ background-color: #FBF4E6; }}
[data-testid="stHeader"] {{ background: transparent; }}

h1, h2, h3, h4 {{
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    color: #6E2A1A !important; letter-spacing: .2px;
}}
h1 {{ font-weight: 700 !important; }}
a {{ color: #B5462C; }}

/* sidebar: deep maroon with cream text */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #7A2E1F 0%, #561B12 100%);
    border-right: 3px solid #C8902A;
}}
[data-testid="stSidebar"] * {{ color: #FBEBD0 !important; }}
[data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: rgba(200, 144, 42, .22) !important; border-radius: 8px;
}}

/* cards, expanders, inputs, buttons */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] details {{
    border-color: #E2C38F !important; border-radius: 14px !important;
    background: rgba(255, 251, 243, .78);
}}
[data-testid="stExpander"] summary:hover {{ color: #B5462C; }}
.stTextInput input {{
    border-radius: 12px; border: 1.5px solid #D9B77C; background: #FFFBF3; font-size: 1.05rem;
}}
.stButton button, .stFormSubmitButton button {{
    border-radius: 999px; border: 1.5px solid #C8902A; background: #FFF8EC; color: #6E2A1A;
    font-weight: 600; transition: all .15s ease;
}}
.stButton button:hover, .stFormSubmitButton button:hover {{ border-color: #B5462C; color: #B5462C; }}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primaryFormSubmit"] {{
    background: linear-gradient(135deg, #C2532F 0%, #E08A2E 100%); color: #FFF8EC; border: none;
}}

/* brand header */
.saarthi-brand {{ text-align: center; margin: .2rem 0 1.4rem 0; }}
.saarthi-mark {{ font-family: 'Tiro Devanagari Sanskrit', serif; color: #C8902A; font-size: 1.25rem; }}
.saarthi-name {{
    font-family: 'Yatra One', 'Cormorant Garamond', serif; font-size: 3.4rem; line-height: 1.1;
    background: linear-gradient(135deg, #8E2B1C 0%, #C2532F 55%, #E08A2E 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.saarthi-tag {{ font-family: 'Cormorant Garamond', serif; font-size: 1.35rem; color: #5A3A28; font-style: italic; }}

/* small labels */
.saarthi-chip {{
    display: inline-block; padding: .12rem .7rem; margin: 0 .35rem .35rem 0; border-radius: 999px;
    background: #F4E4C8; border: 1px solid #E2C38F; color: #5A3A28; font-size: .9rem;
}}
.saarthi-chip b {{ color: #8E2B1C; font-weight: 600; }}

/* verses */
.saarthi-sanskrit {{
    font-family: 'Tiro Devanagari Sanskrit', serif; font-size: 1.18rem; line-height: 1.9;
    color: #5C1F14; text-align: center; margin: .3rem 0 .4rem 0;
}}
.saarthi-translit {{
    font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 1.05rem;
    color: #7A5A44; text-align: center; margin-bottom: .6rem;
}}
.saarthi-verse-title {{ font-family: 'Cormorant Garamond', serif; font-size: 1.25rem; font-weight: 700; color: #6E2A1A; }}
.saarthi-speaker {{ font-size: .95rem; color: #B5462C; margin-left: .5rem; font-weight: 500; }}
.saarthi-quote {{
    border-left: 4px solid #E08A2E; padding: .35rem .9rem; margin: .4rem 0 1rem 0;
    background: rgba(244, 228, 200, .55); border-radius: 0 10px 10px 0; font-size: 1.05rem;
}}
.saarthi-footer {{ text-align: center; color: #8A6A54; font-size: .88rem; margin-top: 2.5rem; }}
</style>
"""


def apply_style(decor: bool = True) -> None:
    """Load the fonts and colours, plus the tile frame and hanging lanterns."""
    st.markdown(CSS, unsafe_allow_html=True)
    if decor:
        from app.components.decor import decorations
        decorations()


def brand(tagline: str = "Guidance from the Bhagavad Gita, in your own language") -> None:
    """The Saarthi name in large decorative type, with a tagline."""
    st.markdown(
        "<div class='saarthi-brand'><div class='saarthi-mark'>सारथी</div>"
        "<div class='saarthi-name'>Saarthi</div>"
        f"<div class='saarthi-tag'>{html.escape(tagline)}</div></div>",
        unsafe_allow_html=True,
    )


def chips(label: str, items: list[str]) -> str:
    """HTML for one label followed by small rounded tags."""
    tags = "".join(f"<span class='saarthi-chip'>{html.escape(i)}</span>" for i in items)
    return f"<span class='saarthi-chip'><b>{html.escape(label)}</b></span>{tags}"


def speaker_name(verse: dict) -> str:
    """Who speaks the verse, in English (Krishna, Arjuna, ...)."""
    return SPEAKERS.get(verse.get("speaker", ""), verse.get("speaker", ""))


def nice_theme(theme: str) -> str:
    """'action_and_results' -> 'Action and results'; 'unclassified' -> 'No clear theme'."""
    if not theme or theme == "unclassified":
        return "No clear theme"
    return theme.replace("_", " ").capitalize()


def footer() -> None:
    from app.components.decor import footer_art
    st.markdown(
        f"<div class='saarthi-footer'>{footer_art()}Saarthi shares what the Gita says; it is not a substitute for "
        "professional help.<br>Made by Antara, Kashvi and Anvi · CSD358 Information Retrieval · "
        "Shiv Nadar Institution of Eminence</div>",
        unsafe_allow_html=True,
    )
