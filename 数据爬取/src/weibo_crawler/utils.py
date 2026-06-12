from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlparse


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
INVALID_FILENAME_RE = re.compile(r'[\\/:*?"<>|\s]+')


def clean_weibo_text(value: str | None) -> str:
    """Convert Weibo HTML snippets to compact plain text."""
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    return SPACE_RE.sub(" ", text).strip()


def sanitize_filename(value: str, max_len: int = 80) -> str:
    value = INVALID_FILENAME_RE.sub("_", value).strip("._")
    if len(value) > max_len:
        value = value[:max_len].rstrip("._")
    return value or "untitled"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def topic_query(word: str, word_scheme: str = "") -> str:
    query = word_scheme or word
    return query.strip()


def search_container_id(query: str) -> str:
    return "100103type=1&q=" + quote(query, safe="")


def s_weibo_search_url(query: str, band_rank: int | None = None, page: int = 1) -> str:
    url = "https://s.weibo.com/weibo?q=" + quote(query, safe="")
    params = ["t=31", "Refer=top"]
    if band_rank:
        params.append(f"band_rank={band_rank}")
    if page > 1:
        params.append(f"page={page}")
    return url + "&" + "&".join(params)


def post_url(user_id: str, mblogid: str, post_id: str) -> str:
    if user_id and mblogid:
        return f"https://weibo.com/{user_id}/{mblogid}"
    if post_id:
        return f"https://weibo.com/detail/{post_id}"
    return ""


def filename_from_url(url: str, fallback_prefix: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if not suffix or len(suffix) > 8:
        suffix = ".bin"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{fallback_prefix}_{digest}{suffix}"
