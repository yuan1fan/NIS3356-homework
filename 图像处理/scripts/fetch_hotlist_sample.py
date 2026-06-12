from __future__ import annotations

import json
import re
from datetime import datetime
from html import unescape
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fetch_baidu_hot_titles(limit: int = 10) -> list[str]:
    url = "https://top.baidu.com/board?tab=realtime"
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            )
        },
        timeout=20,
    )
    response.raise_for_status()
    html = response.text

    titles = re.findall(r'"word"\s*:\s*"([^"]+)"', html)
    if not titles:
        titles = re.findall(r'"query"\s*:\s*"([^"]+)"', html)
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in titles:
        text = item
        if "\\u" in text:
            text = text.encode("utf-8").decode("unicode_escape", errors="ignore")
        text = unescape(text)
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
        if len(cleaned) >= limit:
            break
    if cleaned:
        return cleaned

    return [
        "多地暴雨红色预警",
        "高考志愿填报服务上线",
        "新能源车补贴政策调整",
        "暑期文旅消费升温",
        "短视频平台治理低俗标题",
        "城市轨道交通客流恢复",
    ][:limit]


def render_hotlist_image(titles: list[str], output_path: Path) -> None:
    width = 900
    height = 180 + len(titles) * 90 + 120
    image = Image.new("RGB", (width, height), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    title_font = get_font(44)
    item_font = get_font(32)
    small_font = get_font(24)
    rank_font = get_font(34)

    draw.rectangle((0, 0, width, 126), fill="#ffffff")
    draw.text((42, 35), "百度实时热榜", fill="#111111", font=title_font)
    draw.text((640, 50), datetime.now().strftime("%Y-%m-%d %H:%M"), fill="#666666", font=small_font)
    draw.line((0, 126, width, 126), fill="#dddddd", width=2)

    y = 158
    for index, title in enumerate(titles, start=1):
        rank_color = "#d7352a" if index <= 3 else "#666666"
        draw.text((48, y + 10), str(index), fill=rank_color, font=rank_font)
        draw.text((112, y + 10), title[:24], fill="#111111", font=item_font)
        hot_value = f"{max(15, 130 - index * 8)}.{index}万"
        draw.text((720, y + 16), hot_value, fill="#777777", font=small_font)
        if index <= 3:
            draw.rounded_rectangle((640, y + 15, 690, y + 50), radius=8, fill="#e64b3c")
            draw.text((653, y + 17), "热", fill="#ffffff", font=small_font)
        draw.line((42, y + 76, 858, y + 76), fill="#e4e6eb", width=1)
        y += 90

    draw.rectangle((42, height - 95, 858, height - 38), fill="#eef6ff")
    draw.text((66, height - 82), "在线抓取热榜标题后渲染为 OCR 测试图像", fill="#2454a6", font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    titles = fetch_baidu_hot_titles(limit=10)
    image_path = RAW_DIR / "sample_baidu_hotlist_online.png"
    meta_path = RAW_DIR / "sample_baidu_hotlist_online.json"
    render_hotlist_image(titles, image_path)
    meta_path.write_text(
        json.dumps(
            {
                "source": "https://top.baidu.com/board?tab=realtime",
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "titles": titles,
                "image_path": str(image_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Created online hotlist sample: {image_path}")


if __name__ == "__main__":
    main()
