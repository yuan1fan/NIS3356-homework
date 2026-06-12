from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


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


def draw_tag(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str) -> None:
    x, y = xy
    font = get_font(24)
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle(
        (x - 8, y - 4, bbox[2] + 8, bbox[3] + 4),
        radius=8,
        fill=color,
    )
    draw.text((x, y), text, fill="white", font=font)


def create_weibo_hotsearch(path: Path) -> None:
    image = Image.new("RGB", (900, 1300), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    title_font = get_font(44)
    body_font = get_font(34)
    small_font = get_font(26)
    rank_font = get_font(32)

    draw.rectangle((0, 0, 900, 120), fill="#ffffff")
    draw.text((42, 34), "微博热搜", fill="#111111", font=title_font)
    draw.text((680, 48), "20:30 更新", fill="#666666", font=small_font)
    draw.line((0, 120, 900, 120), fill="#dddddd", width=2)

    rows = [
        ("1", "多地暴雨红色预警", "爆", "248.6万"),
        ("2", "高考志愿填报服务上线", "热", "193.2万"),
        ("3", "新能源车补贴政策调整", "新", "128.9万"),
        ("4", "博物馆暑期预约量上升", "", "96.5万"),
        ("5", "短视频平台治理低俗标题", "热", "82.1万"),
        ("6", "地铁早高峰客流恢复", "", "75.4万"),
        ("7", "电影暑期档预售开启", "新", "64.8万"),
    ]

    y = 155
    for rank, topic, tag, hot_value in rows:
        draw.text((46, y + 12), rank, fill="#d33b30", font=rank_font)
        draw.text((105, y + 8), topic, fill="#111111", font=body_font)
        if tag:
            color = "#e64b3c" if tag == "爆" else "#ff8a00" if tag == "热" else "#3f7ee8"
            draw_tag(draw, (610, y + 11), tag, color)
        draw.text((720, y + 14), hot_value, fill="#777777", font=small_font)
        draw.line((42, y + 86, 858, y + 86), fill="#e4e6eb", width=1)
        y += 98

    draw.rectangle((42, 990, 858, 1170), fill="#fff4e7")
    draw.text((70, 1022), "热点提示", fill="#d65a00", font=body_font)
    draw.text((70, 1080), "榜单文字、排名、热度值可作为趋势预测特征", fill="#333333", font=small_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def create_xiaohongshu_cover(path: Path) -> None:
    image = Image.new("RGB", (900, 1200), "#fefefe")
    draw = ImageDraw.Draw(image)
    title_font = get_font(58)
    subtitle_font = get_font(38)
    body_font = get_font(28)
    small_font = get_font(24)

    draw.rectangle((0, 0, 900, 520), fill="#cce6ff")
    draw.rectangle((0, 520, 900, 1200), fill="#ffffff")
    draw.ellipse((600, 90, 820, 310), fill="#3b82f6")
    draw.rectangle((80, 120, 535, 380), fill="#ffffff")
    draw.text((118, 165), "暴雨天气", fill="#1d4ed8", font=title_font)
    draw.text((118, 250), "出行安全提醒", fill="#111827", font=subtitle_font)

    draw.text((72, 570), "小红书热点笔记", fill="#111111", font=subtitle_font)
    draw.text((72, 635), "多地发布强降雨预警，通勤和校园安全成为讨论焦点", fill="#222222", font=body_font)
    draw.text((72, 700), "#暴雨预警 #城市交通 #热点新闻", fill="#b91c1c", font=body_font)
    draw.text((72, 780), "评论区高频词：停课、积水、地铁延误、应急通知", fill="#333333", font=body_font)
    draw.text((72, 860), "账号：城市观察员  发布时间：今天 19:20", fill="#666666", font=small_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def create_degraded_sample(source: Path, target: Path) -> None:
    image = Image.open(source).convert("RGB")
    image = image.resize((int(image.width * 0.7), int(image.height * 0.7)))
    image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
    image.save(target, quality=45)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    weibo = RAW_DIR / "sample_weibo_hotsearch.png"
    xhs = RAW_DIR / "sample_xiaohongshu_cover.png"
    create_weibo_hotsearch(weibo)
    create_xiaohongshu_cover(xhs)
    create_degraded_sample(weibo, RAW_DIR / "sample_weibo_hotsearch_compressed.jpg")
    print(f"Created samples in {RAW_DIR}")


if __name__ == "__main__":
    main()

