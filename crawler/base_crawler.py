
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests


# Identify our crawler clearly
USER_AGENT = "SaarthiBot/1.0 (educational research project)"

# Minimum delay between requests
REQUEST_DELAY = 1.5

# Store last request time for each domain
_last_request_time = {}

# Cache robots.txt parsers
_robots_cache = {}


def can_fetch(url):
    """Check whether our crawler is allowed to fetch the given URL."""

    parsed_url = urlparse(url)
    domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

    if domain not in _robots_cache:
        robots_url = f"{domain}/robots.txt"

        parser = RobotFileParser()
        parser.set_url(robots_url)

        try:
            response = requests.get(
                robots_url,
                headers={"User-Agent": USER_AGENT},
                timeout=10
            )

            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            elif response.status_code in (401, 403):
                # If access to robots.txt is denied,
                # conservatively disallow crawling.
                _robots_cache[domain] = None
                return False
            else:
                # If robots.txt is unavailable,
                # assume no specific restrictions.
                parser.parse([])

            _robots_cache[domain] = parser

        except requests.RequestException:
            # If robots.txt cannot be checked,
            # do not crawl.
            return False

    parser = _robots_cache[domain]

    if parser is None:
        return False

    return parser.can_fetch(USER_AGENT, url)


def fetch(url):
    """Download a webpage after checking robots.txt and respecting request delays."""

    if not can_fetch(url):
        raise PermissionError(
            f"Crawling is not allowed for this URL: {url}"
        )

    domain = urlparse(url).netloc

    # Enforce delay between requests to the same domain
    last_request = _last_request_time.get(domain, 0)
    elapsed = time.monotonic() - last_request

    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=20
        )

        _last_request_time[domain] = time.monotonic()

        response.raise_for_status()

        # Detect encoding before decoding HTML
        response.encoding = response.apparent_encoding

        return response.text

    except requests.RequestException as error:
        raise RuntimeError(f"Failed to fetch {url}: {error}") from error


def save_raw(html, path):
    """Save raw HTML to disk, creating parent directories if necessary."""

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(html, encoding="utf-8")

    return path
