import hashlib
import json
import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CACHE_PATH = os.path.join("data", "processed", "llm_cache.json")

_client = None
_cache = None


def _get_client():
    """Create the OpenAI client the first time it is needed."""
    global _client
    if _client is None:
        from openai import OpenAI

        key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("No API key: add LLM_API_KEY=sk-... to your .env file")
        base_url = os.getenv("LLM_BASE_URL") or None  # None = OpenAI itself
        _client = OpenAI(api_key=key, base_url=base_url)
    return _client


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False, indent=1)


def ask(prompt: str, system: str = "", json_mode: bool = False, temperature: float = 0.2) -> str:
    """Send one prompt to the LLM and return its reply as text (cached)."""
    cache = _load_cache()
    key = hashlib.sha256(json.dumps([MODEL, system, prompt, json_mode, temperature]).encode()).hexdigest()
    if key in cache:
        return cache[key]

    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    kwargs = {"model": MODEL, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    reply = _get_client().chat.completions.create(**kwargs).choices[0].message.content or ""
    cache[key] = reply
    _save_cache()
    return reply


def ask_json(prompt: str, system: str = "", temperature: float = 0.2) -> dict:
    """Like ask(), but the reply must be a JSON object; returns it as a dict."""
    reply = ask(prompt, system=system, json_mode=True, temperature=temperature)
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        return {}