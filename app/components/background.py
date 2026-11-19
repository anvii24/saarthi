"""Run slow work in a background thread while the page shows a loader.

Streamlit can only switch pages between its own calls, so a page that waits inside a
long call (models, Gemini) freezes the whole app. Here the page waits in short steps
instead: the loader can show what is happening, and clicking another page works at once.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from app.components.decor import lotus_loader

_pool = ThreadPoolExecutor(max_workers=2)


@st.cache_resource(show_spinner=False)
def start_warm_up() -> threading.Thread:
    """Load the models once in the background when the app starts (shared by every visitor)."""
    def warm():
        try:
            from pipeline import warm_up
            warm_up()
        except Exception as err:  # the first question will load them instead
            print(f"Warm-up skipped: {err}")

    thread = threading.Thread(target=warm, daemon=True)
    thread.start()
    return thread


def run_with_loader(work, first_text: str, note: str = ""):
    """Run work(on_step) in the background and show a breathing lotus with its latest step.

    work gets one argument, on_step(text), to report progress. Returns work's result
    (or raises its error).
    """
    status = {"text": first_text}
    future = _pool.submit(work, lambda text: status.update(text=text))
    box = st.empty()
    shown, started = None, time.time()
    while not future.done():
        seconds = int(time.time() - started)
        text = f"{status['text']}…" + (f"<br><small>{note}</small>" if note else "")
        if seconds >= 5:
            text += f"<br><small>{seconds} s</small>"
        if text != shown:
            box.markdown(lotus_loader(text), unsafe_allow_html=True)
            shown = text
        time.sleep(0.25)
    box.empty()
    return future.result()
