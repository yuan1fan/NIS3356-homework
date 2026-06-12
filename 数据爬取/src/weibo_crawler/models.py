from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HotTopic:
    rank: int
    word: str
    note: str = ""
    word_scheme: str = ""
    raw_hot_score: int | None = None
    label: str = ""
    detail_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "word": self.word,
            "note": self.note,
            "word_scheme": self.word_scheme,
            "raw_hot_score": self.raw_hot_score,
            "label": self.label,
            "detail_url": self.detail_url,
        }


@dataclass(slots=True)
class MediaItem:
    type: str
    url: str
    local_path: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "url": self.url,
            "local_path": self.local_path,
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
        }
        return {key: value for key, value in data.items() if value not in (None, "", [])}


@dataclass(slots=True)
class CandidatePost:
    post_id: str
    mblogid: str = ""
    user_id: str = ""
    user_name: str = ""
    created_at: str = ""
    text: str = ""
    reposts_count: int = 0
    comments_count: int = 0
    attitudes_count: int = 0
    source_url: str = ""
    images: list[MediaItem] = field(default_factory=list)
    videos: list[MediaItem] = field(default_factory=list)
    audios: list[MediaItem] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text_len(self) -> int:
        return len(self.text.strip())

    @property
    def media_count(self) -> int:
        return len(self.images) + len(self.videos) + len(self.audios)

    @property
    def max_video_duration(self) -> float | None:
        durations = [
            item.duration_seconds
            for item in self.videos
            if item.duration_seconds is not None
        ]
        return max(durations) if durations else None

    @property
    def engagement_score(self) -> int:
        return self.reposts_count * 3 + self.comments_count * 2 + self.attitudes_count

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "post_id": self.post_id,
            "mblogid": self.mblogid,
            "user": {
                "id": self.user_id,
                "name": self.user_name,
            },
            "created_at": self.created_at,
            "text": self.text,
            "metrics": {
                "reposts_count": self.reposts_count,
                "comments_count": self.comments_count,
                "attitudes_count": self.attitudes_count,
                "engagement_score": self.engagement_score,
            },
            "source_url": self.source_url,
            "media": {
                "images": [item.to_dict() for item in self.images],
                "videos": [item.to_dict() for item in self.videos],
                "audios": [item.to_dict() for item in self.audios],
            },
        }
        if include_raw:
            data["raw"] = self.raw
        return data


@dataclass(slots=True)
class SelectionResult:
    post: CandidatePost | None
    reason: str
    candidate_count: int
    rejected: list[dict[str, Any]]

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        return {
            "selected_post": self.post.to_dict(include_raw=include_raw) if self.post else None,
            "reason": self.reason,
            "candidate_count": self.candidate_count,
            "rejected": self.rejected,
        }

