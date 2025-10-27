import random
from urllib3 import Retry
from fastapi import requests
from requests.adapters import HTTPAdapter


def get_request_session(proxies: dict = None):
    """Return a requests session with retries, headers, and optional proxy"""
    session = requests.Session()

    retries = Retry(
        total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 429]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/118.0",
    ]
    session.headers.update({"User-Agent": random.choice(user_agents)})

    if proxies:
        session.proxies.update(proxies)

    return session