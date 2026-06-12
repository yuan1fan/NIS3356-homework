from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


class WeiboClient:
    def __init__(
        self,
        cookie: str = "",
        timeout: float = 20,
        delay: float = 1.0,
        retries: int = 2,
    ) -> None:
        self.timeout = timeout
        self.delay = delay
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        if cookie:
            self.session.headers["Cookie"] = cookie

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        referer: str | None = None,
    ) -> dict[str, Any]:
        headers = {"Referer": referer} if referer else None
        text = self.get_text(url, params=params, headers=headers)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            snippet = text[:300].replace("\n", " ")
            raise RuntimeError(f"Response is not JSON: {snippet}") from exc

    def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        final_url = url if not params else f"{url}?{urlencode(params)}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.delay * attempt)
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                response.raise_for_status()
                time.sleep(self.delay)
                return response.text
            except requests.RequestException as exc:
                last_error = exc
        raise RuntimeError(f"GET failed after retries: {final_url}") from last_error

    def download(
        self,
        url: str,
        output_path: Path,
        referer: str | None = None,
        max_bytes: int = 80 * 1024 * 1024,
    ) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = {"Referer": referer} if referer else None
        with self.session.get(
            url,
            headers=headers,
            timeout=self.timeout,
            stream=True,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            total = 0
            with output_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        output_path.unlink(missing_ok=True)
                        return False
                    file.write(chunk)
        time.sleep(self.delay)
        return True

