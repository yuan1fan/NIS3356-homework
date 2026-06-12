from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.weibo_crawler.client import WeiboClient
from src.weibo_crawler.pipeline import WeiboHotSearchCrawler
from src.weibo_crawler.selector import SelectionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl representative posts from Weibo hot-search topics.")
    parser.add_argument("--cookie", default="", help="Weibo login cookie string. Prefer WEIBO_COOKIE env var.")
    parser.add_argument("--cookie-file", default="", help="Path to a text file containing the Weibo cookie.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for crawl outputs.")
    parser.add_argument("--max-topics", type=int, default=50, help="Number of hot-search topics to snapshot.")
    parser.add_argument("--pages-per-topic", type=int, default=2, help="Mobile search pages to inspect per topic.")
    parser.add_argument("--min-text-len", type=int, default=12, help="Reject posts with shorter plain text.")
    parser.add_argument("--max-video-minutes", type=float, default=20, help="Reject posts containing videos longer than this.")
    parser.add_argument("--allow-text-only", action="store_true", help="Allow selected posts without image/video/audio.")
    parser.add_argument("--no-download-media", action="store_true", help="Keep media URLs only; do not download files.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds.")
    parser.add_argument("--include-raw", action="store_true", help="Include raw Weibo JSON in output.")
    return parser.parse_args()


def read_cookie(args: argparse.Namespace) -> str:
    if args.cookie:
        return args.cookie.strip()
    if args.cookie_file:
        return Path(args.cookie_file).read_text(encoding="utf-8").strip()
    return os.environ.get("WEIBO_COOKIE", "").strip()


def main() -> None:
    args = parse_args()
    cookie = read_cookie(args)
    policy = SelectionPolicy(
        min_text_len=args.min_text_len,
        require_media=not args.allow_text_only,
        max_video_duration_seconds=int(args.max_video_minutes * 60),
    )
    client = WeiboClient(cookie=cookie, delay=args.delay)
    crawler = WeiboHotSearchCrawler(
        client=client,
        output_dir=Path(args.output_dir),
        policy=policy,
        max_topics=args.max_topics,
        pages_per_topic=args.pages_per_topic,
        download_media=not args.no_download_media,
        include_raw=args.include_raw,
    )
    run_dir = crawler.crawl()
    print(f"Crawl finished: {run_dir}")


if __name__ == "__main__":
    main()

