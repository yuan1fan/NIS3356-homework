from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "real_raw"


SOURCES = [
    {
        "name": "baidu_realtime_hotlist",
        "url": "https://top.baidu.com/board?tab=realtime",
        "platform": "baidu_hotlist",
    },
    {
        "name": "thepaper_news",
        "url": "https://www.thepaper.cn/",
        "platform": "thepaper",
    },
    {
        "name": "people_news",
        "url": "http://www.people.com.cn/",
        "platform": "people",
    },
    {
        "name": "cctv_news",
        "url": "https://news.cctv.com/",
        "platform": "cctv",
    },
    {
        "name": "qq_news",
        "url": "https://news.qq.com/",
        "platform": "qq_news",
    },
    {
        "name": "ithome_news",
        "url": "https://www.ithome.com/",
        "platform": "ithome",
    },
]


def safe_name(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)
    return text.strip("_").lower() or "sample"


def collect_screenshots(
    output_dir: Path,
    target_count: int = 40,
    viewport_width: int = 1280,
    viewport_height: int = 900,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    records: list[dict] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            device_scale_factor=1,
            locale="zh-CN",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        for source in SOURCES:
            if len(records) >= target_count:
                break
            source_records = _capture_source(
                page=page,
                source=source,
                output_dir=output_dir,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                remaining=target_count - len(records),
            )
            records.extend(source_records)

        context.close()
        browser.close()

    with metadata_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def _capture_source(
    page,
    source: dict,
    output_dir: Path,
    viewport_width: int,
    viewport_height: int,
    remaining: int,
) -> list[dict]:
    records: list[dict] = []
    url = source["url"]
    name = source["name"]
    platform = source["platform"]

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(3000)
        _dismiss_common_overlays(page)
    except PlaywrightTimeoutError:
        return records
    except Exception:
        return records

    page_title = ""
    try:
        page_title = page.title()
    except Exception:
        page_title = ""

    scroll_height = _get_scroll_height(page)
    max_scrolls = min(max(3, scroll_height // viewport_height + 1), 12)
    source_limit = min(remaining, 8)
    timestamp = datetime.now().isoformat(timespec="seconds")

    for index in range(source_limit):
        y = int(index * viewport_height * 0.72)
        if y > max(0, scroll_height - viewport_height):
            y = max(0, scroll_height - viewport_height)
        if index >= max_scrolls:
            break

        try:
            page.evaluate("(y) => window.scrollTo(0, y)", y)
            page.wait_for_timeout(1000)
            screenshot_path = output_dir / f"real_{safe_name(name)}_{index + 1:02d}.png"
            page.screenshot(path=str(screenshot_path), full_page=False)
        except Exception:
            continue

        if not _has_enough_visual_content(screenshot_path):
            screenshot_path.unlink(missing_ok=True)
            continue

        records.append(
            {
                "image_path": str(screenshot_path),
                "source_name": name,
                "platform": platform,
                "url": url,
                "domain": urlparse(url).netloc,
                "page_title": page_title,
                "captured_at": timestamp,
                "viewport": {
                    "width": viewport_width,
                    "height": viewport_height,
                    "scroll_y": y,
                },
                "collection_method": "playwright_viewport_screenshot",
            }
        )

    return records


def _dismiss_common_overlays(page) -> None:
    selectors = [
        "text=同意",
        "text=我知道了",
        "text=知道了",
        "text=关闭",
        "button:has-text('同意')",
        "button:has-text('关闭')",
        ".close",
        ".modal-close",
        ".popup-close",
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.is_visible(timeout=500):
                locator.click(timeout=500)
        except Exception:
            pass


def _get_scroll_height(page) -> int:
    try:
        return int(
            page.evaluate(
                "() => Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"
            )
        )
    except Exception:
        return 3000


def _has_enough_visual_content(path: Path) -> bool:
    try:
        image = Image.open(path).convert("L")
        width, height = image.size
        if width < 300 or height < 300:
            return False
        histogram = image.histogram()
        total = width * height
        whiteish = sum(histogram[245:])
        blackish = sum(histogram[:10])
        blank_ratio = max(whiteish, blackish) / max(1, total)
        return blank_ratio < 0.95
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real web screenshots for OCR testing.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--count", type=int, default=40, help="Target screenshot count.")
    args = parser.parse_args()

    records = collect_screenshots(Path(args.output_dir), target_count=args.count)
    print(f"Collected {len(records)} screenshots in {args.output_dir}")
    for record in records:
        print(f"- {record['source_name']} {record['viewport']['scroll_y']} -> {record['image_path']}")


if __name__ == "__main__":
    main()

