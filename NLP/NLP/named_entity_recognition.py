"""
NLP 命名实体识别模块 — 基于词典 + 规则的传统方法

从零实现，使用词典匹配和正则规则识别中文文本中的实体：
- 人名（姓氏+名字模式）
- 地名（省市县/国家/常见地名）
- 机构名（公司/大学/局/处等后缀）
- 时间（日期/时间表达式）
- 金额（数字+单位）
- 产品名（常见产品后缀）

Usage:
    from named_entity_recognition import NERExtractor
    ner = NERExtractor()
    entities = ner.recognize("2026年6月，某公司在上海发布新款手机。")
    # [{"text": "2026年6月", "type": "时间", "start": 0, "end": 7}, ...]
"""

import re
from typing import Dict, List, Optional, Tuple

from preprocessing import TextPreprocessor


# ═══════════════════════════════════════════════════════════
# 1. 内置词典与规则
# ═══════════════════════════════════════════════════════════

class NERDict:
    """NER 所需的内置词典与规则模式。"""

    # ── 百家姓（常见） ──
    SURNAMES = frozenset({
        "赵","钱","孙","李","周","吴","郑","王","冯","陈","褚","卫",
        "蒋","沈","韩","杨","朱","秦","尤","许","何","吕","施","张",
        "孔","曹","严","华","金","魏","陶","姜","戚","谢","邹","喻",
        "柏","水","窦","章","云","苏","潘","葛","范","彭","郎","鲁",
        "韦","昌","马","苗","凤","花","方","俞","任","袁","柳","酆",
        "鲍","史","唐","费","廉","岑","薛","雷","贺","倪","汤","滕",
        "殷","罗","毕","郝","邬","安","常","乐","于","时","傅","皮",
        "卞","齐","康","伍","余","元","卜","顾","孟","平","黄","和",
        "穆","萧","尹","姚","邵","湛","汪","祁","毛","禹","狄","米",
        "贝","明","臧","计","伏","成","戴","谈","宋","茅","庞","熊",
        "纪","舒","屈","项","祝","董","梁","杜","阮","蓝","闵","席",
        "季","麻","强","贾","路","娄","危","江","童","颜","郭","梅",
        "盛","林","刁","钟","徐","邱","骆","高","夏","蔡","田","樊",
        "胡","凌","霍","虞","万","支","柯","昝","管","卢","莫","经",
        "房","裘","缪","干","解","应","宗","丁","宣","贲","邓","郁",
        "单","杭","洪","包","诸","左","石","崔","吉","钮","龚","程",
        "嵇","邢","滑","裴","陆","荣","翁","荀","羊","於","惠","甄",
        "曲","家","封","芮","羿","储","靳","汲","邴","糜","松","井",
        "段","富","巫","乌","焦","巴","弓","牧","隗","山","谷","车",
        "侯","宓","蓬","全","郗","班","仰","秋","仲","伊","宫","宁",
        "仇","栾","暴","甘","钭","厉","戎","祖","武","符","刘","景",
        "詹","束","龙","叶","幸","司","韶","郜","黎","蓟","薄","印",
        "宿","白","怀","蒲","邰","从","鄂","索","咸","籍","赖","卓",
        "蔺","屠","蒙","池","乔","阴","郁","胥","能","苍","双","闻",
        "莘","党","翟","谭","贡","劳","逄","姬","申","扶","堵","冉",
        "宰","郦","雍","郤","璩","桑","桂","濮","牛","寿","通","边",
        "扈","燕","冀","郏","浦","尚","农","温","别","庄","晏","柴",
        "瞿","阎","充","慕","连","茹","习","宦","艾","鱼","容","向",
        "古","易","慎","戈","廖","庾","终","暨","居","衡","步","都",
        "耿","满","弘","匡","国","文","寇","广","禄","阙","东","欧",
        "殳","沃","利","蔚","越","夔","隆","师","巩","厍","聂","晁",
        "勾","敖","融","冷","訾","辛","阚","那","简","饶","空","曾",
        "毋","沙","乜","养","鞠","须","丰","巢","关","蒯","相","查",
        "后","荆","红","游","竺","权","逯","盖","益","桓","公","仉",
        "岳","帅","缑","亢","况","后","有","琴","商","牟","佘","佴",
        "伯","赏","墨","哈","谯","笪","年","爱","阳","佟","言","福",
        "欧阳","太史","端木","上官","司马","东方","独孤","南宫",
        "夏侯","诸葛","尉迟","公羊","赫连","澹台","皇甫","宗政",
        "濮阳","公孙","慕容","仲孙","钟离","长孙","宇文","闾丘",
        "司空","鲜于","司寇","子车","颛孙","端木","巫马","公西",
        "漆雕","乐正","壤驷","公良","拓跋","夹谷","宰父","谷梁",
        "段干","百里","东郭","南门","呼延","羊舌","微生","梁丘",
        "左丘","东门","西门","第五",
    })

    # ── 中国省份 ──
    PROVINCES = frozenset({
        "北京","天津","上海","重庆","河北","山西","辽宁","吉林",
        "黑龙江","江苏","浙江","安徽","福建","江西","山东","河南",
        "湖北","湖南","广东","海南","四川","贵州","云南","陕西",
        "甘肃","青海","台湾","内蒙古","广西","西藏","宁夏","新疆",
        "香港","澳门",
    })

    # ── 常见城市 ──
    CITIES = frozenset({
        "广州","深圳","珠海","汕头","佛山","东莞","中山","惠州",
        "杭州","宁波","温州","嘉兴","湖州","绍兴","金华","苏州",
        "无锡","常州","南京","徐州","南通","扬州","镇江","盐城",
        "成都","绵阳","德阳","宜宾","南充","武汉","黄石","宜昌",
        "襄阳","荆州","郑州","洛阳","开封","新乡","安阳","许昌",
        "济南","青岛","淄博","烟台","潍坊","泰安","威海","日照",
        "长沙","株洲","湘潭","衡阳","岳阳","合肥","芜湖","蚌埠",
        "福州","厦门","泉州","漳州","南昌","赣州","九江","沈阳",
        "大连","鞍山","哈尔滨","齐齐哈尔","牡丹江","长春","吉林",
        "昆明","大理","丽江","贵阳","遵义","南宁","桂林","海口",
        "三亚","兰州","西宁","西安","宝鸡","咸阳","太原","大同",
        "石家庄","唐山","呼和浩特","包头","乌鲁木齐","拉萨",
        "银川","台北","高雄","台中","台南",
    })

    # ── 国家名 ──
    COUNTRIES = frozenset({
        "中国","美国","日本","韩国","朝鲜","英国","法国","德国",
        "意大利","西班牙","葡萄牙","荷兰","比利时","瑞士","瑞典",
        "挪威","丹麦","芬兰","俄罗斯","乌克兰","波兰","捷克",
        "匈牙利","罗马尼亚","保加利亚","希腊","土耳其","伊朗",
        "伊拉克","沙特","阿联酋","印度","巴基斯坦","孟加拉",
        "泰国","越南","缅甸","柬埔寨","老挝","菲律宾","马来西亚",
        "新加坡","印度尼西亚","澳大利亚","新西兰","加拿大","墨西哥",
        "巴西","阿根廷","智利","埃及","南非","肯尼亚","尼日利亚",
    })

    # ── 机构后缀 ──
    ORG_SUFFIXES = frozenset({
        "公司","集团","银行","医院","学校","大学","学院","中学",
        "小学","幼儿园","研究院","研究所","中心","委员会","协会",
        "基金会","俱乐部","报社","出版社","电视台","广播台",
        "局","处","部","委","办","厅","司","署","院",
        "公安局","派出所","分局",  # 公安系统
        "工作室","学会","商会","工会","联社","总队","支队",
        "品牌","系列","厂商","商行","公社",
    })

    # ── 产品后缀 ──
    PRODUCT_SUFFIXES = frozenset({
        "手机","电脑","平板","笔记本","电视","冰箱","洗衣机",
        "空调","耳机","音箱","相机","镜头","手表","手环",
        "汽车","电动车","摩托车","自行车","飞机","轮船",
        "系统","软件","APP","应用","游戏","平台","服务",
        "基金","股票","债券","保险","理财",
        "面膜","精华","乳液","面霜","眼霜","口红","香水",
    })

    # ── 时间单位 ──
    TIME_UNITS = frozenset({
        "年","月","日","号","时","分","秒","周","星期",
        "季度","半年度","年度","月份","年份",
    })

    # ── 星期 ──
    WEEKDAYS = frozenset({
        "星期一","星期二","星期三","星期四","星期五","星期六","星期日",
        "周一","周二","周三","周四","周五","周六","周日",
        "礼拜一","礼拜二","礼拜三","礼拜四","礼拜五","礼拜六","礼拜日",
    })

    # ── 相对时间 ──
    RELATIVE_TIME = frozenset({
        "今天","明天","昨天","前天","后天",
        "今年","明年","去年","前年","后年",
        "本月","上月","下月","本周","上周","下周",
        "今日","明日","昨日",
        "刚刚","刚才","现在","目前","当前"
    })


# ═══════════════════════════════════════════════════════════
# 2. NER 提取器
# ═══════════════════════════════════════════════════════════



# ── BERT-NER (可选) ──
_USE_BERT = False
_bert_ner = None
try:
    import os as _os
    _cache = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "_model_cache")
    _md = _os.path.join(_cache, "models--ckiplab--bert-base-chinese-ner")
    _sd = _os.path.join(_md, "snapshots")
    _mp = None
    if _os.path.exists(_sd):
        for _d in _os.listdir(_sd):
            _p = _os.path.join(_sd, _d)
            if _os.path.exists(_os.path.join(_p, "config.json")):
                _mp = _p
                break
    if _mp:
        from transformers import AutoModelForTokenClassification as _AMFTC
        from transformers import AutoTokenizer as _AT
        from transformers import pipeline as _PL
        _m = _AMFTC.from_pretrained(_mp)
        _t = _AT.from_pretrained(_mp)
        _bert_ner = _PL("token-classification", model=_m, tokenizer=_t, aggregation_strategy="simple")
        _USE_BERT = True
except Exception:
    pass

class NERExtractor:
    """基于词典 + 规则的命名实体识别器。"""

    def __init__(self):
        self._dict = NERDict()
        self._pp = TextPreprocessor()
        # 预编译正则
        self._compile_patterns()

    def _compile_patterns(self):
        """预编译所有实体识别正则。"""

        # ── 日期时间 ──
        dt_patterns = []
        # 年月日: 2026年6月12日
        dt_patterns.append(
            r"(?:\d{2,4}\s*[年月](?:\s*\d{1,2}\s*[月日号])?)"
        )
        # 时分秒: 12:30, 12时30分
        dt_patterns.append(
            r"(?:\d{1,2}\s*[：:]\s*\d{2}(?:\s*[：:]\s*\d{2})?)"
        )
        # 年月日时分秒合一
        dt_patterns.append(
            r"(?:\d{4}\s*[年\-/]\s*\d{1,2}\s*[月\-/]\s*\d{1,2}"
            r"(?:\s*[日号])?(?:\s*\d{1,2}\s*[：:]\s*\d{2}(?:\s*[：:]\s*\d{2})?)?)"
        )
        self._datetime_re = re.compile("|".join(dt_patterns))

        # ── 星期 ──
        weekday_pat = "|".join(re.escape(w) for w in self._dict.WEEKDAYS)
        self._weekday_re = re.compile(weekday_pat)

        # ── 相对时间 ──
        rel_pat = "|".join(re.escape(w) for w in self._dict.RELATIVE_TIME)
        self._reltime_re = re.compile(rel_pat)

        # ── 金额 ──
        self._money_re = re.compile(
            r"(?:(?:\d+(?:\.\d+)?)\s*(?:元|块|美元|欧元|英镑|日元|万|亿|"
            r"万元|亿元|美元|港币|韩元|卢布|澳元|加元|法郎|马克))"
        )

        # ── 百分比 ──
        self._percent_re = re.compile(
            r"(?:\d+(?:\.\d+)?\s*%)"
        )

    # ── 主识别方法 ──

    def recognize(self, text: str) -> List[Dict]:
        """识别文本中的所有实体。使用 BERT-NER 如可用，否则回退到 jieba。"""

        entities = []
        try:
            global _USE_BERT, _bert_ner
            if _USE_BERT and _bert_ner is not None:
                results = _bert_ner(text)
                be = []
                tm = {"PER":"人名","LOC":"地名","ORG":"机构名"}
                for r in results:
                    et = r.get("entity_group","")
                    if et in tm:
                        w = "".join(r["word"].split())
                        if r.get("score",0) < 0.5:
                            continue
                        be.append({"text":w,"type":tm[et],"start":r.get("start",0),"end":r.get("end",0)})
                if be:
                    entities.extend(be)
        except Exception:
            pass

        if not text or not text.strip():
            return []

        # 依次识别各类实体
        entities.extend(self._extract_time(text))
        entities.extend(self._extract_money(text))
        entities.extend(self._extract_person(text))
        entities.extend(self._extract_place(text))
        entities.extend(self._extract_org(text))
        entities.extend(self._extract_product(text))

        # 去重 + 合并重叠实体（保留最长）
        entities = self._deduplicate(entities)

        # 按 start 排序
        entities.sort(key=lambda e: e["start"])

        return entities

    # ── 各实体识别 ──

    def _extract_time(self, text: str) -> List[Dict]:
        """识别时间实体。"""
        entities = []

        for match in self._datetime_re.finditer(text):
            entities.append({
                "text": match.group().strip(),
                "type": "时间",
                "start": match.start(),
                "end": match.end(),
            })

        for match in self._weekday_re.finditer(text):
            entities.append({
                "text": match.group(),
                "type": "时间",
                "start": match.start(),
                "end": match.end(),
            })

        for match in self._reltime_re.finditer(text):
            entities.append({
                "text": match.group(),
                "type": "时间",
                "start": match.start(),
                "end": match.end(),
            })

        return entities

    def _extract_money(self, text: str) -> List[Dict]:
        """识别金额实体。"""
        entities = []
        for match in self._money_re.finditer(text):
            entities.append({
                "text": match.group(),
                "type": "金额",
                "start": match.start(),
                "end": match.end(),
            })
        for match in self._percent_re.finditer(text):
            entities.append({
                "text": match.group(),
                "type": "金额",
                "start": match.start(),
                "end": match.end(),
            })
        return entities
    def _extract_with_posseg(self, text):
        import jieba.posseg as pseg
        tm = {"nr":"人名","ns":"地名","nt":"机构名"}
        ents = []
        for _tok in pseg.cut(text):
            w, f = _tok.word, _tok.flag
            if f in tm:
                pos = text.find(w)
                if pos >= 0:
                    ents.append({"text":w,"type":tm[f],"start":pos,"end":pos+len(w)})
        return ents

    def _extract_person(self, text):
        return [e for e in self._extract_with_posseg(text) if e["type"]=="人名"]

    def _extract_place(self, text):
        return [e for e in self._extract_with_posseg(text) if e["type"]=="地名"]

    def _extract_org(self, text):
        return [e for e in self._extract_with_posseg(text) if e["type"]=="机构名"]

    def _extract_product(self, text: str) -> List[Dict]:
        """识别产品名：前缀(可选) + 产品后缀。"""
        entities = []
        suffixes = sorted(self._dict.PRODUCT_SUFFIXES, key=len, reverse=True)
        escaped = [re.escape(s) for s in suffixes]
        # 产品名可以有 1~8 个前缀汉字
        pattern = r"(?:[\u4e00-\u9fff]{1,3}(?:" + "|".join(escaped) + r"))"
        for match in re.finditer(pattern, text):
            t = match.group()
            if len(t) >= 2:  # 至少2个字
                entities.append({
                    "text": t,
                    "type": "产品名",
                    "start": match.start(),
                    "end": match.end(),
                })
        return entities

    # ── 工具方法 ──

    @staticmethod
    def _deduplicate(entities: List[Dict]) -> List[Dict]:
        """去重并合并重叠实体（保留最长的）。"""
        if not entities:
            return []

        # 按 start 升序，end 降序排序
        sorted_ents = sorted(entities, key=lambda e: (e["start"], -e["end"]))

        result = []
        i = 0
        while i < len(sorted_ents):
            current = sorted_ents[i]
            j = i + 1
            # 合并所有与当前重叠的实体，保留最长的
            while j < len(sorted_ents):
                next_ent = sorted_ents[j]
                if next_ent["start"] >= current["end"]:
                    break  # 不再重叠
                # 重叠了，保留长的
                if (next_ent["end"] - next_ent["start"]
                        > current["end"] - current["start"]):
                    current = next_ent
                j += 1
            # 检查是否已存在于结果中（完全相同）
            if not any(e["text"] == current["text"]
                       and e["type"] == current["type"]
                       for e in result):
                result.append(current)
            i = j

        return result


# ═══════════════════════════════════════════════════════════
# 3. 展示工具
# ═══════════════════════════════════════════════════════════

def format_entities(entities: List[Dict], text: str = "") -> str:
    """格式化显示实体识别结果。"""
    if not entities:
        return "(无实体)"

    lines = []
    for e in entities:
        # 在原文中的上下文
        ctx_start = max(0, e["start"] - 5)
        ctx_end = min(len(text), e["end"] + 5)
        ctx = text[ctx_start:e["start"]] + "【" + e["text"] + "】" + text[e["end"]:ctx_end]
        if text:
            lines.append(f"  [{e['type']}] {e['text']}  ...{ctx}...")
        else:
            lines.append(f"  [{e['type']}] {e['text']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 4. Demo
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("命名实体识别 Demo")
    print("=" * 60)

    ner = NERExtractor()

    test_cases = [
        "#突发#！！某地暴雨太大了！！！详情见 http://xxx.com",
        "2026年6月，某公司在上海发布新款手机。",
        "今天天气真好，白鹿工作室发表声明",
        "北京市公安局刑侦总队政治处主任李小燕公布了案例",
        "太让人愤怒了，这纯属欺骗消费者行为！",
        "央视曝养生馆围猎老年人，涉案金额高达3000万余元",
        "真的太喜欢了，超级好用，强烈推荐！",
    ]

    for text in test_cases:
        print(f"\n原文: {text[:60]}")
        entities = ner.recognize(text)
        print(format_entities(entities, text))

    # 真实数据
    print("\n" + "-" * 60)
    print("真实微博数据")
    print("-" * 60)
    jsonl_path = (
        "E:\\6+7\\SJTU\\大三下\\信息内容安全\\大作业\\数据"
        "\\20260612_141245\\20260612_141245\\representative_posts.jsonl"
    )
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            data = json.loads(line)
            topic = data.get("topic", {}).get("word", "未知")
            text = data.get("selection", {}).get("selected_post", {}).get("text", "")
            if text:
                entities = ner.recognize(text)
                print(f"\n[{topic}]")
                print(f"  文本: {text[:50]}...")
                for e in entities[:5]:
                    print(f"  [{e['type']}] {e['text']}")
