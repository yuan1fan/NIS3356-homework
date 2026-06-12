from src.weibo_crawler.parser import parse_hot_topics, parse_pc_search_posts, parse_search_posts
from src.weibo_crawler.selector import SelectionPolicy, select_representative_post


def test_parse_hot_topics_deduplicates_and_limits():
    payload = {
        "data": {
            "realtime": [
                {"word": "话题A", "realpos": 1, "num": 100, "word_scheme": "#话题A#"},
                {"word": "话题A", "realpos": 2, "num": 90},
                {"word": "话题B", "realpos": 3, "num": 80},
            ]
        }
    }

    topics = parse_hot_topics(payload, limit=2)

    assert [topic.word for topic in topics] == ["话题A", "话题B"]
    assert topics[0].rank == 1
    assert topics[0].raw_hot_score == 100


def test_parse_search_posts_extracts_text_and_media():
    payload = {
        "data": {
            "cards": [
                {
                    "mblog": {
                        "id": "1",
                        "mblogid": "Abc",
                        "created_at": "now",
                        "text": "<a href='x'>#话题#</a> 这是一条包含图片和视频的微博<br/>第二行",
                        "user": {"id": 42, "screen_name": "用户"},
                        "reposts_count": 2,
                        "comments_count": 3,
                        "attitudes_count": 4,
                        "pic_ids": ["p1"],
                        "pic_infos": {
                            "p1": {
                                "largest": {
                                    "url": "//wx1.sinaimg.cn/large/demo.jpg",
                                    "width": 640,
                                    "height": 480,
                                }
                            }
                        },
                        "page_info": {
                            "media_info": {
                                "stream_url": "https://video.example/demo.mp4",
                                "duration": 60,
                            }
                        },
                    }
                }
            ]
        }
    }

    posts = parse_search_posts(payload)

    assert len(posts) == 1
    assert posts[0].text == "#话题# 这是一条包含图片和视频的微博 第二行"
    assert posts[0].images[0].url == "https://wx1.sinaimg.cn/large/demo.jpg"
    assert posts[0].videos[0].duration_seconds == 60
    assert posts[0].source_url == "https://weibo.com/42/Abc"


def test_selector_rejects_short_text_no_media_and_long_video():
    payload = {
        "data": {
            "cards": [
                {"mblog": {"id": "short", "text": "太短", "user": {}}},
                {"mblog": {"id": "nomedia", "text": "这是一条足够长但是没有媒体的微博", "user": {}}},
                {
                    "mblog": {
                        "id": "longvideo",
                        "text": "这是一条包含超长视频的微博文本",
                        "user": {},
                        "page_info": {
                            "media_info": {
                                "stream_url": "https://video.example/long.mp4",
                                "duration": 3600,
                            }
                        },
                    }
                },
                {
                    "mblog": {
                        "id": "ok",
                        "text": "这是一条合适的代表微博，包含图片且文字足够。",
                        "user": {},
                        "attitudes_count": 10,
                        "pic_ids": ["p1"],
                        "pic_infos": {"p1": {"largest": {"url": "https://img.example/a.jpg"}}},
                    }
                },
            ]
        }
    }
    posts = parse_search_posts(payload)

    result = select_representative_post(
        posts,
        SelectionPolicy(min_text_len=12, require_media=True, max_video_duration_seconds=1200),
    )

    assert result.post is not None
    assert result.post.post_id == "ok"
    assert {item["reason"] for item in result.rejected} == {
        "text_too_short",
        "no_media",
        "video_too_long",
    }


def test_parse_pc_search_posts_extracts_real_card_fields():
    html = """
    <!--card-wrap-->
    <div class="card-wrap" action-type="feed_list_item" mid="5308916377321523" >
      <div class="card">
        <div class="card-feed">
          <a href="//weibo.com/2803301701?refer_flag=1001030103_" class="name" nick-name="人民日报">人民日报</a>
          <div class="from"><a href="//weibo.com/2803301701/R3JXfuIF5?refer_flag=1001030103_">今天09:29</a></div>
          <p class="txt" node-type="feed_list_content" nick-name="人民日报">
            【标题】<a href="/weibo?q=x">#话题#</a> 正文摘要 <a action-type="fl_unfold">展开</a>
          </p>
          <p class="txt" node-type="feed_list_content_full" nick-name="人民日报" style="display:none">
            【标题】<a href="/weibo?q=x">#话题#</a> 完整正文内容
          </p>
          <div node-type="feed_list_media_prev">
            <div class="media media-piclist" action-data="uid=2803301701&mid=5308916377321523">
              <img src="https://wx4.sinaimg.cn/orj360/demo.jpg">
            </div>
          <div node-type="feed_list_media_disp"></div>
        </div>
      </div>
    </div>
    """

    posts = parse_pc_search_posts(html)

    assert len(posts) == 1
    assert posts[0].post_id == "5308916377321523"
    assert posts[0].user_name == "人民日报"
    assert posts[0].created_at == "今天09:29"
    assert posts[0].text == "【标题】#话题# 完整正文内容"
    assert posts[0].images[0].url == "https://wx4.sinaimg.cn/large/demo.jpg"
    assert posts[0].source_url == "https://weibo.com/2803301701/R3JXfuIF5"
