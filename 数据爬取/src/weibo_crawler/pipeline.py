from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .client import WeiboClient
from .models import HotTopic, MediaItem
from .parser import parse_hot_topics, parse_pc_search_posts, parse_search_posts
from .selector import SelectionPolicy, select_representative_post
from .utils import filename_from_url, now_stamp, s_weibo_search_url, sanitize_filename, search_container_id, topic_query


HOT_SEARCH_URL = "https://weibo.com/ajax/side/hotSearch"
MOBILE_SEARCH_URL = "https://m.weibo.cn/api/container/getIndex"


class WeiboHotSearchCrawler:
    def __init__(
        self,
        client: WeiboClient,
        output_dir: Path,
        policy: SelectionPolicy | None = None,
        max_topics: int = 50,
        pages_per_topic: int = 2,
        download_media: bool = True,
        max_media_per_post: int = 6,
        include_raw: bool = False,
    ) -> None:
        self.client = client
        self.output_dir = output_dir
        self.policy = policy or SelectionPolicy()
        self.max_topics = max_topics
        self.pages_per_topic = pages_per_topic
        self.download_media = download_media
        self.max_media_per_post = max_media_per_post
        self.include_raw = include_raw

    def crawl(self) -> Path:
        run_dir = self.output_dir / now_stamp()
        run_dir.mkdir(parents=True, exist_ok=True)
        topics = self.snapshot_hot_topics()
        self._write_jsonl(run_dir / "hot_topics_snapshot.jsonl", [t.to_dict() for t in topics])

        results: list[dict[str, Any]] = []
        for topic in topics:
            result = self.crawl_topic(topic, run_dir)
            results.append(result)
            self._append_jsonl(run_dir / "representative_posts.jsonl", result)

        summary = {
            "hot_topic_count": len(topics),
            "selected_post_count": sum(1 for item in results if item["selection"]["selected_post"]),
            "policy": asdict(self.policy),
            "files": {
                "hot_topics_snapshot": "hot_topics_snapshot.jsonl",
                "representative_posts": "representative_posts.jsonl",
            },
        }
        self._write_json(run_dir / "summary.json", summary)
        return run_dir

    def snapshot_hot_topics(self) -> list[HotTopic]:
        payload = self.client.get_json(
            HOT_SEARCH_URL,
            referer="https://weibo.com/",
        )
        return parse_hot_topics(payload, limit=self.max_topics)

    def crawl_topic(self, topic: HotTopic, run_dir: Path) -> dict[str, Any]:
        candidates = []
        errors: list[str] = []
        query = topic_query(topic.word, topic.word_scheme)
        for page in range(1, self.pages_per_topic + 1):
            try:
                html_text = self.client.get_text(
                    s_weibo_search_url(query, band_rank=topic.rank, page=page),
                    headers={"Referer": "https://s.weibo.com/top/summary?cate=realtimehot"},
                )
                candidates.extend(parse_pc_search_posts(html_text))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"s.weibo.com page {page}: {exc}")

        if not candidates:
            for page in range(1, self.pages_per_topic + 1):
                try:
                    payload = self.client.get_json(
                        MOBILE_SEARCH_URL,
                        params={
                            "containerid": search_container_id(query),
                            "page_type": "searchall",
                            "page": page,
                        },
                        referer="https://m.weibo.cn/",
                    )
                    candidates.extend(parse_search_posts(payload))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"m.weibo.cn page {page}: {exc}")

        seen: set[str] = set()
        deduped = []
        for post in candidates:
            key = post.post_id or post.mblogid or post.source_url
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(post)

        selection = select_representative_post(deduped, self.policy)
        if selection.post and self.download_media:
            topic_dir = run_dir / "media" / f"{topic.rank:02d}_{sanitize_filename(topic.word)}"
            self._download_post_media(selection.post.images, topic_dir / "images", "image", selection.post.source_url, run_dir)
            self._download_post_media(selection.post.videos, topic_dir / "videos", "video", selection.post.source_url, run_dir)
            self._download_post_media(selection.post.audios, topic_dir / "audios", "audio", selection.post.source_url, run_dir)

        return {
            "topic": topic.to_dict(),
            "selection": selection.to_dict(include_raw=self.include_raw),
            "errors": errors,
        }

    def _download_post_media(
        self,
        items: list[MediaItem],
        output_dir: Path,
        prefix: str,
        referer: str,
        run_dir: Path,
    ) -> None:
        for item in items[: self.max_media_per_post]:
            filename = filename_from_url(item.url, prefix)
            path = output_dir / filename
            try:
                ok = self.client.download(item.url, path, referer=referer)
            except Exception as exc:  # noqa: BLE001
                item.raw["download_error"] = str(exc)
                continue
            if ok:
                item.local_path = str(path.relative_to(run_dir))
            else:
                item.raw["download_error"] = "file_exceeded_max_bytes"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
