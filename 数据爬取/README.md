# 微博热搜帖子数据爬取模块

本模块用于爬取微博热搜 Top50 中每个条目对应的代表帖子，输出帖子文本、图片、视频、音频链接以及可选本地媒体文件。爬取流程会先保存当前时刻热搜快照，再按快照顺序逐条处理，避免微博热搜实时刷新导致重复或漏抓。

## 模块流程

1. 请求 `https://weibo.com/ajax/side/hotSearch`，一次性保存当前热搜 Top50 到 `hot_topics_snapshot.jsonl`。
2. 对每个热搜词使用移动端搜索接口抓取前若干页候选帖子。
3. 清洗微博 HTML 文本，提取图片、视频、音频 URL 和互动指标。
4. 按规则选择代表帖：
   - 默认文本长度至少 12 个字符；
   - 默认必须包含图片、视频或音频；
   - 默认跳过超过 20 分钟的视频；
   - 在候选中优先选择媒体更完整、文本更充足、互动量更高的帖子。
5. 输出 `representative_posts.jsonl`，并按需下载媒体到 `media/`。

## 安全注意

不要把微博登录 cookie 写入代码或提交到 Git。建议通过环境变量或本地文件传入：

```powershell
$env:WEIBO_COOKIE = "你的 cookie"
python run_weibo_hotsearch.py --max-topics 50 --pages-per-topic 2
```

或者：

```powershell
python run_weibo_hotsearch.py --cookie-file .weibo_cookie.txt
```

`.weibo_cookie.txt` 和 `outputs/` 已在 `.gitignore` 中排除。

## 常用参数

```powershell
python run_weibo_hotsearch.py `
  --max-topics 50 `
  --pages-per-topic 2 `
  --min-text-len 12 `
  --max-video-minutes 20
```

如果只需要 URL，不下载媒体：

```powershell
python run_weibo_hotsearch.py --no-download-media
```

如果课程测试时部分热搜没有媒体，允许纯文本代表帖：

```powershell
python run_weibo_hotsearch.py --allow-text-only
```

## 输出结构

每次运行会新建一个时间戳目录，例如：

```text
outputs/20260612_183000/
  hot_topics_snapshot.jsonl
  representative_posts.jsonl
  summary.json
  media/
```

`representative_posts.jsonl` 每行对应一个热搜条目，核心字段包括：

- `topic`：热搜排名、词条、热度值、详情链接；
- `selection.selected_post.text`：代表帖正文；
- `selection.selected_post.media.images/videos/audios`：媒体 URL 和可选本地路径；
- `selection.rejected`：未被选中的候选帖子及过滤原因。

## 测试

```powershell
python -m pytest tests -q
```

