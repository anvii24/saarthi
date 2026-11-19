"""Decorations from the team's mood board (images in app/static/decor/):
a blue-tile frame around the whole page, hanging lanterns, the flute,
lotuses and a row of small illustrations in the footer.

Images are served by Streamlit's static folder (enableStaticServing in .streamlit/config.toml).
Animations stop for people who turn on "reduce motion" in their system settings.
"""

import streamlit as st

IMG = "app/static/decor/{}.webp"
TILE_BLUE = "#1D3B7E"
GOLD = "#C8902A"
CORNER = 150  # px width of each tile corner


def img(name: str, cls: str = "", style: str = "") -> str:
    """An <img> for one decoration (alt="" because they are purely decorative)."""
    return f"<img src='{IMG.format(name)}' class='{cls}' style='{style}' alt=''>"


def _url(name: str) -> str:
    return f'url("{IMG.format(name)}")'


DECOR_CSS = f"""
<style>
/* the blue-tile frame: four corners joined by a blue and gold border, around the whole page */
[data-testid="stMain"] {{
    background-image:
        {_url("tile_tl")}, {_url("tile_tr")}, {_url("tile_bl")}, {_url("tile_br")},
        linear-gradient({GOLD}, {GOLD}), linear-gradient({GOLD}, {GOLD}),
        linear-gradient({GOLD}, {GOLD}), linear-gradient({GOLD}, {GOLD}),
        linear-gradient({TILE_BLUE}, {TILE_BLUE}), linear-gradient({TILE_BLUE}, {TILE_BLUE}),
        linear-gradient({TILE_BLUE}, {TILE_BLUE}), linear-gradient({TILE_BLUE}, {TILE_BLUE});
    background-position:
        left top, right top, left bottom, right bottom,
        left 0 top 12px, left 0 bottom 12px, left 12px top 0, right 12px top 0,
        left top, left bottom, left top, right top;
    background-size:
        {CORNER}px auto, {CORNER}px auto, {CORNER}px auto, {CORNER}px auto,
        100% 2px, 100% 2px, 2px 100%, 2px 100%,
        100% 9px, 100% 9px, 9px 100%, 9px 100%;
    background-repeat: no-repeat;
    background-attachment: local;
}}
[data-testid="stMain"] {{ position: relative; }}
[data-testid="stMain"] .block-container {{ padding-top: 5rem; padding-bottom: 6rem; }}

/* lanterns hang from the top of the page, just inside the right corner (they scroll away with the page) */
[data-testid="stMain"]::after {{
    content: ""; position: absolute; top: 0; right: {CORNER + 14}px; width: 78px; height: 155px;
    background: {_url("lanterns")} no-repeat top / contain; pointer-events: none; z-index: 5;
    transform-origin: 50% 0; animation: swing 5.5s ease-in-out infinite;
    filter: drop-shadow(0 0 14px rgba(255, 196, 80, .55));
}}
@keyframes swing {{ 0%, 100% {{ transform: rotate(2.5deg); }} 50% {{ transform: rotate(-2.5deg); }} }}
@keyframes sway {{ 0%, 100% {{ transform: rotate(-2deg); }} 50% {{ transform: rotate(2deg); }} }}

/* home header */
.saarthi-hero {{ text-align: center; margin: 0 auto 1.2rem auto; }}
.saarthi-hero .flute {{ width: 170px; margin-bottom: -.3rem; animation: sway 5s ease-in-out infinite; }}

/* page titles with a small illustration on the right */
.saarthi-page-header {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; }}
.saarthi-side-art {{ width: 104px; margin-right: 110px; }}  /* leaves room for the lanterns */

.saarthi-lotus-divider {{ text-align: center; margin: 1.2rem 0 .4rem 0; }}
.saarthi-lotus-divider img {{ width: 72px; }}

/* a lotus that breathes while Saarthi is thinking */
.saarthi-loader {{ text-align: center; padding: 1rem 0; color: #8E2B1C; font-family: 'Cormorant Garamond', serif;
                   font-size: 1.25rem; font-style: italic; }}
.saarthi-loader img {{ width: 110px; animation: breathe 2.2s ease-in-out infinite; }}
@keyframes breathe {{ 0%, 100% {{ transform: scale(.88); opacity: .7; }} 50% {{ transform: scale(1.05); opacity: 1; }} }}

/* answer sentences appear one after another */
.saarthi-reveal {{ opacity: 0; animation: reveal .7s ease forwards; }}
@keyframes reveal {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}

/* cards lift a little on hover */
[data-testid="stVerticalBlockBorderWrapper"] {{ transition: transform .18s ease, box-shadow .18s ease; }}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{ transform: translateY(-2px); box-shadow: 0 8px 22px rgba(110, 42, 26, .12); }}

.saarthi-footer-art {{ margin-bottom: .4rem; }}
.saarthi-footer-art img {{ height: 50px; margin: 0 .5rem; vertical-align: bottom; }}

@media (max-width: 900px) {{
    [data-testid="stMain"]::after {{ display: none; }}
    .saarthi-side-art {{ margin-right: 0; }}
    [data-testid="stMain"] {{ background-size: 80px auto, 80px auto, 80px auto, 80px auto,
                                             100% 2px, 100% 2px, 2px 100%, 2px 100%, 100% 6px, 100% 6px, 6px 100%, 6px 100%; }}
}}
@media (prefers-reduced-motion: reduce) {{
    [data-testid="stMain"]::after, .saarthi-hero .flute, .saarthi-loader img, .saarthi-reveal {{
        animation: none !important; opacity: 1; }}
}}
</style>
"""


def decorations() -> None:
    """The tile frame and lanterns (all pages)."""
    st.markdown(DECOR_CSS, unsafe_allow_html=True)


def hero(tagline: str) -> None:
    """Home-page header: the flute, then the name and tagline."""
    st.markdown(
        "<div class='saarthi-hero'>" + img("flute_hands", "flute")
        + "<div class='saarthi-mark'>सारथी</div><div class='saarthi-name'>Saarthi</div>"
        f"<div class='saarthi-tag'>{tagline}</div></div>",
        unsafe_allow_html=True,
    )


def page_header(title: str, art: str) -> None:
    """A page title with a small illustration on its right."""
    st.markdown(
        f"<div class='saarthi-page-header'><h1>{title}</h1>{img(art, 'saarthi-side-art')}</div>",
        unsafe_allow_html=True,
    )


def lotus_divider() -> None:
    st.markdown(f"<div class='saarthi-lotus-divider'>{img('lotus_water')}</div>", unsafe_allow_html=True)


def lotus_loader(text: str) -> str:
    """HTML for a softly breathing lotus with a line of text (put it in st.empty() while waiting)."""
    return f"<div class='saarthi-loader'>{img('lotus_pink')}<div>{text}</div></div>"


def footer_art() -> str:
    return ("<div class='saarthi-footer-art'>" + img("hibiscus") + img("jasmine_ball")
            + img("candle_jar") + img("swan") + "</div>")
