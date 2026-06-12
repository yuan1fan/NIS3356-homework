from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import unquote

from .models import CandidatePost, HotTopic, MediaItem
from .utils import clean_weibo_text, post_url, s_weibo_search_url


def parse_hot_topics(payload: dict[str, Any], limit: int = 50) -> list[HotTopic]:
    realtime = payload.get("data", {}).get("realtime", [])
    topics: list[HotTopic] = []
    seen: set[str] = set()
    for item in realtime:
        word = str(item.get("word") or item.get("note") or "").strip()
        if not word or word in seen:
            continue
        seen.add(word)
        rank = int(item.get("realpos") or len(topics) + 1)
        word_scheme = str(item.get("word_scheme") or item.get("note") or word).strip()
        detail_url = _detail_url(word_scheme, word, rank)
        topics.append(
            HotTopic(
                rank=rank,
                word=word,
                note=str(item.get("note") or ""),
                word_scheme=word_scheme,
                raw_hot_score=_safe_int(item.get("num")),
                label=str(item.get("label_name") or item.get("icon_desc") or ""),
                detail_url=detail_url,
            )
        )
        if len(topics) >= limit:
            break
    return topics


def parse_search_posts(payload: dict[str, Any]) -> list[CandidatePost]:
    cards = payload.get("data", {}).get("cards", [])
    posts: list[CandidatePost] = []
    for card in cards:
        posts.extend(_posts_from_card(card))
    return posts


def parse_pc_search_posts(html_text: str) -> list[CandidatePost]:
    posts: list[CandidatePost] = []
    for card in _split_pc_cards(html_text):
        post = _parse_pc_card(card)
        if post:
            posts.append(post)
    posts.extend(_parse_media_module_posts(html_text))
    return _dedupe_posts(posts)


def _posts_from_card(card: dict[str, Any]) -> list[CandidatePost]:
    posts: list[CandidatePost] = []
    if isinstance(card.get("mblog"), dict):
        posts.append(parse_mblog(card["mblog"]))
    for group in card.get("card_group") or []:
        if isinstance(group, dict):
            if isinstance(group.get("mblog"), dict):
                posts.append(parse_mblog(group["mblog"]))
            elif isinstance(group.get("card_group"), list):
                for child in group["card_group"]:
                    if isinstance(child, dict) and isinstance(child.get("mblog"), dict):
                        posts.append(parse_mblog(child["mblog"]))
    return posts


def parse_mblog(mblog: dict[str, Any]) -> CandidatePost:
    user = mblog.get("user") or {}
    post_id = str(mblog.get("id") or mblog.get("mid") or "")
    mblogid = str(mblog.get("mblogid") or "")
    user_id = str(user.get("id") or "")
    text = clean_weibo_text(mblog.get("text") or mblog.get("text_raw"))
    return CandidatePost(
        post_id=post_id,
        mblogid=mblogid,
        user_id=user_id,
        user_name=str(user.get("screen_name") or ""),
        created_at=str(mblog.get("created_at") or ""),
        text=text,
        reposts_count=_safe_int(mblog.get("reposts_count")) or 0,
        comments_count=_safe_int(mblog.get("comments_count")) or 0,
        attitudes_count=_safe_int(mblog.get("attitudes_count")) or 0,
        source_url=post_url(user_id, mblogid, post_id),
        images=_extract_images(mblog),
        videos=_extract_videos(mblog),
        audios=_extract_audios(mblog),
        raw=mblog,
    )


def _split_pc_cards(html_text: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r'<!--card-wrap-->\s*<div class="card-wrap"', html_text)]
    cards: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(html_text)
        block = html_text[start:end]
        if 'action-type="feed_list_item"' in block and ' mid="' in block:
            cards.append(block)
    return cards


def _parse_pc_card(card: str) -> CandidatePost | None:
    mid = _first_match(r'\bmid="(\d+)"', card)
    if not mid:
        return None
    name_match = re.search(r'<a[^>]+class="name"[^>]*nick-name="([^"]*)"[^>]*>(.*?)</a>', card, re.S)
    user_name = clean_weibo_text(name_match.group(1) or name_match.group(2)) if name_match else ""
    user_id = _first_match(r'href="//weibo\.com/(\d+)\?', card)
    mblogid = _first_match(r'href="//weibo\.com/\d+/([A-Za-z0-9]+)\?', card)
    from_block = _first_match(r'<div class="from"[^>]*>(.*?)</div>', card, flags=re.S)
    created_at = clean_weibo_text(_first_match(r'<a[^>]*>(.*?)</a>', from_block, flags=re.S))
    text_html = _first_match(
        r'<p class="txt"[^>]+node-type="feed_list_content_full"[^>]*>(.*?)</p>',
        card,
        flags=re.S,
    )
    if not text_html:
        text_html = _first_match(
            r'<p class="txt"[^>]+node-type="feed_list_content"[^>]*>(.*?)</p>',
            card,
            flags=re.S,
        )
    text = clean_weibo_text(text_html)
    images = _extract_pc_images(card)
    videos = _extract_pc_videos(card)
    reposts, comments, attitudes = _extract_pc_metrics(card)
    return CandidatePost(
        post_id=mid,
        mblogid=mblogid,
        user_id=user_id,
        user_name=user_name,
        created_at=created_at,
        text=text,
        reposts_count=reposts,
        comments_count=comments,
        attitudes_count=attitudes,
        source_url=post_url(user_id, mblogid, mid),
        images=images,
        videos=videos,
        audios=[],
        raw={"source": "s.weibo.com"},
    )


def _extract_pc_images(card: str) -> list[MediaItem]:
    urls: list[str] = []
    media_blocks = re.findall(
        r'<div node-type="feed_list_media_prev">(.*?)<div node-type="feed_list_media_disp">',
        card,
        re.S,
    )
    scan_text = "\n".join(media_blocks) if media_blocks else card
    for url in re.findall(r'<img[^>]+src="([^"]+)"', scan_text):
        normalized = _normalize_url(url)
        if not re.search(r'//w[wx]\d?\.sinaimg\.cn/', normalized):
            continue
        if any(token in normalized for token in ("tvax", "tva", "crop.")):
            continue
        urls.append(_prefer_large_image(normalized))
    return [
        MediaItem(type="image", url=url, raw={"source": "s.weibo.com"})
        for url in _unique(urls)
    ]


def _extract_pc_videos(card: str) -> list[MediaItem]:
    urls = []
    patterns = [
        r'video_url["\']?\s*[:=]\s*["\']([^"\']+)',
        r'\bsrc:\s*["\']([^"\']+\.mp4[^"\']*)',
        r'\bvalue["\']?\s*:\s*["\']([^"\']+\.mp4[^"\']*)',
        r'(https?://video\.weibo\.com/[^"\'<\\]+)',
        r'(https?://[^"\'<\\]+\.mp4[^"\'<\\]*)',
    ]
    for pattern in patterns:
        urls.extend(re.findall(pattern, card))
    duration = _extract_pc_video_duration(card)
    return [
        MediaItem(type="video", url=_normalize_url(url), duration_seconds=duration, raw={"source": "s.weibo.com"})
        for url in _unique(urls)
    ]


def _extract_pc_video_duration(card: str) -> float | None:
    duration = _first_match(r'\bduration:\s*(\d+(?:\.\d+)?)', card)
    if not duration:
        duration = _first_match(r'"duration"\s*:\s*(\d+(?:\.\d+)?)', card)
    try:
        value = float(duration) if duration else None
        return value if value and value > 0 else None
    except ValueError:
        return None


def _extract_pc_metrics(card: str) -> tuple[int, int, int]:
    values = {
        "reposts": _metric_after_label(card, "转发"),
        "comments": _metric_after_label(card, "评论"),
        "attitudes": _metric_after_label(card, "赞"),
    }
    return values["reposts"], values["comments"], values["attitudes"]


def _metric_after_label(card: str, label: str) -> int:
    pattern = rf'{label}\s*(?:</a>|</span>|<[^>]+>)*\s*(\d+|万)?'
    match = re.search(pattern, clean_weibo_text(card))
    if not match:
        return 0
    raw = match.group(1) or "0"
    if raw.endswith("万"):
        return int(float(raw[:-1] or 0) * 10000)
    return _safe_int(raw) or 0


def _parse_media_module_posts(html_text: str) -> list[CandidatePost]:
    posts: list[CandidatePost] = []
    for encoded in re.findall(r'<media-module[^>]+data-params="([^"]+)"', html_text):
        try:
            payload = json.loads(unquote(encoded))
        except json.JSONDecodeError:
            continue
        items = payload.get("card_multimodal", {}).get("data", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("cur_mid") or "")
            if not mid:
                continue
            image_url = item.get("img") or ""
            video_url = item.get("video_url") or ""
            images = [MediaItem(type="image", url=_normalize_url(image_url), raw={"source": "media-module"})] if image_url else []
            videos = [MediaItem(type="video", url=_normalize_url(video_url), raw={"source": "media-module"})] if video_url else []
            posts.append(
                CandidatePost(
                    post_id=mid,
                    user_name=str(item.get("user_name") or ""),
                    text=clean_weibo_text(item.get("text") or item.get("text_n") or ""),
                    source_url=f"https://weibo.com/detail/{mid}",
                    images=images,
                    videos=videos,
                    raw={"source": "media-module"},
                )
            )
    return posts


def _dedupe_posts(posts: list[CandidatePost]) -> list[CandidatePost]:
    merged: dict[str, CandidatePost] = {}
    for post in posts:
        key = post.post_id or post.source_url
        if key not in merged:
            merged[key] = post
            continue
        current = merged[key]
        if not current.text and post.text:
            current.text = post.text
        if not current.images and post.images:
            current.images = post.images
        if not current.videos and post.videos:
            current.videos = post.videos
        if not current.user_name and post.user_name:
            current.user_name = post.user_name
    return list(merged.values())


def _first_match(pattern: str, text: str, flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else ""


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _prefer_large_image(url: str) -> str:
    for marker in ("/orj360/", "/thumb150/", "/thumbnail/", "/small/"):
        url = url.replace(marker, "/large/")
    return url


def _extract_images(mblog: dict[str, Any]) -> list[MediaItem]:
    images: list[MediaItem] = []
    pic_infos = mblog.get("pic_infos") or {}
    pic_ids = mblog.get("pic_ids") or list(pic_infos.keys())
    for pic_id in pic_ids:
        info = pic_infos.get(pic_id, {}) if isinstance(pic_infos, dict) else {}
        largest = (
            info.get("largest")
            or info.get("mw2000")
            or info.get("large")
            or info.get("original")
            or info.get("thumbnail")
            or {}
        )
        url = largest.get("url") or info.get("url")
        if not url:
            continue
        images.append(
            MediaItem(
                type="image",
                url=_normalize_url(url),
                width=_safe_int(largest.get("width")),
                height=_safe_int(largest.get("height")),
                raw=info,
            )
        )
    return images


def _extract_videos(mblog: dict[str, Any]) -> list[MediaItem]:
    videos: list[MediaItem] = []
    page_info = mblog.get("page_info") or {}
    media_info = page_info.get("media_info") or mblog.get("media_info") or {}
    if not isinstance(media_info, dict):
        return videos
    urls = [
        media_info.get("stream_url_hd"),
        media_info.get("stream_url"),
        media_info.get("mp4_hd_url"),
        media_info.get("mp4_sd_url"),
    ]
    page_url = page_info.get("page_url")
    for url in urls + [page_url]:
        if not url:
            continue
        videos.append(
            MediaItem(
                type="video",
                url=_normalize_url(str(url)),
                duration_seconds=_safe_duration(media_info),
                raw=media_info,
            )
        )
        break
    return videos


def _extract_audios(mblog: dict[str, Any]) -> list[MediaItem]:
    audios: list[MediaItem] = []
    page_info = mblog.get("page_info") or {}
    if page_info.get("object_type") != "audio":
        return audios
    media_info = page_info.get("media_info") or {}
    url = (
        media_info.get("stream_url")
        or media_info.get("audio_url")
        or page_info.get("page_url")
        or ""
    )
    if url:
        audios.append(
            MediaItem(
                type="audio",
                url=_normalize_url(str(url)),
                duration_seconds=_safe_duration(media_info),
                raw=media_info,
            )
        )
    return audios


def _safe_duration(media_info: dict[str, Any]) -> float | None:
    for key in ("duration", "duration_seconds", "play_time"):
        value = media_info.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 100000:
            number = number / 1000
        return number
    return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_url(url: str) -> str:
    url = unquote(url)
    if url.startswith("//"):
        return "https:" + url
    return url


def _detail_url(word_scheme: str, word: str, rank: int) -> str:
    query = word_scheme or word
    return s_weibo_search_url(query, band_rank=rank)
