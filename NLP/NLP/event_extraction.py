"""
NLP 事件抽取模块 — 基于触发词 + 规则 + NER 的传统方法

从文本中识别"发生了什么"，输出结构化事件信息。
事件要素通过触发词检测 + NER 槽位填充完成。

Usage:
    from event_extraction import EventExtractor
    ee = EventExtractor()
    events = ee.extract("某品牌因质量问题发布召回公告")
    # [{"type": "产品召回", "trigger": "发布召回公告",
    #   "slots": {"主体": "某品牌", "原因": "质量问题", "动作": "发布召回公告"}}]
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from named_entity_recognition import NERExtractor


# ═══════════════════════════════════════════════════════════
# 1. 事件类型定义
# ═══════════════════════════════════════════════════════════

EVENT_DEFINITIONS = {
    # ==== 一、社会事件类 ====
    "自然灾害": {
        "triggers": ["地震","洪水","台风","暴雨","火灾","山体滑坡","泥石流","海啸","旱灾","雪灾","冰雹","沙尘暴"],
        "slots": ["时间","地点","类型","结果"],
        "desc": "自然灾害发生及灾情报告",
    },
    "事故灾难": {
        "triggers": ["车祸","坠机","爆炸","坍塌","踩踏","火灾","矿难","沉船","追尾","连环撞","泄露"],
        "slots": ["时间","地点","类型","结果"],
        "desc": "生产安全或交通事故灾难",
    },
    "公共卫生": {
        "triggers": ["疫情","病毒","流感","食物中毒","疫苗","确诊病例","无症状","核酸检测","封控","隔离"],
        "slots": ["时间","地点","类型","结果"],
        "desc": "公共卫生事件或疫情防控",
    },
    "社会治安": {
        "triggers": ["抢劫","凶杀","绑架","诈骗","斗殴","枪击","砍人","遇害","杀人","强奸","盗窃"],
        "slots": ["时间","地点","主体","结果"],
        "desc": "刑事犯罪或社会治安案件",
    },
    "执法司法": {
        "triggers": ["逮捕","判刑","起诉","通缉","判决","拘留","刑事诉讼","公诉","定罪"],
        "slots": ["时间","主体","事由","结果"],
        "desc": "执法机关执法或司法审判",
    },
    "维权抗议": {
        "triggers": ["维权","抗议","罢工","示威","上访","举报","投诉","集体维权","讨薪"],
        "slots": ["时间","地点","主体","事由"],
        "desc": "群众维权抗议或罢工事件",
    },
    "救援救助": {
        "triggers": ["救援","搜救","救灾","募捐","救助","驰援","抢险","抢救","赈灾"],
        "slots": ["时间","地点","主体","对象"],
        "desc": "应急救援或慈善救助行动",
    },
    # ==== 二、政治/政策类 ====
    "政策发布": {
        "triggers": ["发布","印发","出台","实施","规定","办法","条例","颁布","施行"],
        "slots": ["时间","主体","政策","内容"],
        "desc": "政府/机构发布政策法规",
    },
    "政策调整": {
        "triggers": ["调整","放宽","收紧","取消","恢复","优化","下调","上调","改革"],
        "slots": ["时间","主体","政策","内容"],
        "desc": "现有政策的调整或优化",
    },
    "领导人活动": {
        "triggers": ["视察","出席","讲话","会见","访问","主持","考察","调研"],
        "slots": ["时间","地点","主体","活动"],
        "desc": "领导人公开活动或讲话",
    },
    "会议召开": {
        "triggers": ["召开","举行","举办","论坛","峰会","大会","会议","座谈会"],
        "slots": ["时间","地点","主体","会议"],
        "desc": "各类会议论坛召开",
    },
    "外交事件": {
        "triggers": ["会谈","声明","谴责","制裁","联合公报","抗议","交涉","对话"],
        "slots": ["时间","主体","对象","内容"],
        "desc": "国家间外交互动或争端",
    },
    "反腐倡廉": {
        "triggers": ["落马","双开","调查","违纪","受贿","被查","审查","留置","通报"],
        "slots": ["时间","主体","事由","结果"],
        "desc": "反腐或违纪查处事件",
    },
    # ==== 三、经济/商业类 ====
    "产品发布": {
        "triggers": ["发布","推出","上市","亮相","首发","开售","预售","登场"],
        "slots": ["时间","主体","产品","地点"],
        "desc": "新产品新版本发布上市",
    },
    "企业动态": {
        "triggers": ["成立","更名","搬迁","裁员","扩招","重组","架构调整","关停"],
        "slots": ["时间","主体","动作","内容"],
        "desc": "企业组织变动或经营动态",
    },
    "投融资": {
        "triggers": ["融资","投资","收购","并购","入股","上市","IPO","募资","注资"],
        "slots": ["时间","主体","对象","金额"],
        "desc": "企业投融资或上市事件",
    },
    "财报业绩": {
        "triggers": ["营收","净利润","增长","下滑","亏损","财报","业绩","盈利"],
        "slots": ["时间","主体","金额","结果"],
        "desc": "企业财报发布或业绩表现",
    },
    "价格变动": {
        "triggers": ["涨价","降价","上调","下调","打折","促销","补贴"],
        "slots": ["时间","主体","产品","金额"],
        "desc": "商品服务价格调整",
    },
    "产品问题": {
        "triggers": ["缺陷","投诉","故障","爆炸","翻车","失灵","质量门","维权"],
        "slots": ["时间","主体","产品","原因"],
        "desc": "产品质量问题引发投诉",
    },
    "产品召回": {
        "triggers": ["召回","下架","停售","退换","维修"],
        "slots": ["时间","主体","产品","原因"],
        "desc": "产品因质量问题被召回",
    },
    "合同签约": {
        "triggers": ["签约","合作","战略合作","协议","达成合作","签署","联手"],
        "slots": ["时间","主体","合作方","领域"],
        "desc": "企业间合作或合同签约",
    },
    "破产重组": {
        "triggers": ["破产","清算","重组","债务违约","退市","注销","倒闭"],
        "slots": ["时间","主体","事由","结果"],
        "desc": "企业破产清算或重组",
    },
    "市场竞争": {
        "triggers": ["对标","超越","碾压","反击","宣战","竞争","争夺"],
        "slots": ["时间","主体","对象","领域"],
        "desc": "企业间市场竞争行为",
    },
    # ==== 四、科技/文娱类 ====
    "技术突破": {
        "triggers": ["突破","首发","首创","自主研制","攻克","研发","创新"],
        "slots": ["时间","主体","领域","内容"],
        "desc": "科技或技术领域的突破",
    },
    "科研发布": {
        "triggers": ["论文","研究","发现","Nature","Science","期刊","成果"],
        "slots": ["时间","主体","领域","内容"],
        "desc": "科研成果或论文发布",
    },
    "影视上映": {
        "triggers": ["上映","播出","定档","首播","开播","上线","排片"],
        "slots": ["时间","作品","类型","平台"],
        "desc": "影视作品上映或播出",
    },
    "综艺动态": {
        "triggers": ["开播","收官","淘汰","晋级","总决赛","录制","路透"],
        "slots": ["时间","节目","内容","结果"],
        "desc": "综艺节目动态或赛况",
    },
    "明星动态": {
        "triggers": ["官宣","恋情","结婚","离婚","生子","生日","演唱会","粉丝见面会"],
        "slots": ["时间","主体","事件","内容"],
        "desc": "明星个人动态或公开活动",
    },
    "明星丑闻": {
        "triggers": ["出轨","家暴","吸毒","逃税","嫖娼","偷拍","塌房","封杀"],
        "slots": ["时间","主体","类型","结果"],
        "desc": "明星负面新闻或丑闻",
    },
    "粉丝事件": {
        "triggers": ["撕番","控评","刷榜","网暴","饭圈","打榜","粉丝大战"],
        "slots": ["时间","主体","事件","结果"],
        "desc": "粉丝群体争议或事件",
    },
    "游戏动态": {
        "triggers": ["上线","更新","停服","版号获批","内测","公测","DLC"],
        "slots": ["时间","产品","内容","平台"],
        "desc": "游戏行业内动态",
    },
    # ==== 五、体育/赛事类 ====
    "比赛结果": {
        "triggers": ["夺冠","晋级","淘汰","获胜","失利","冠军","金牌","捧杯","摘金"],
        "slots": ["时间","主体","赛事","结果"],
        "desc": "体育比赛成绩或结果",
    },
    "转会签约": {
        "triggers": ["转会","签约","加盟","续约","离队","自由身"],
        "slots": ["时间","主体","对象","金额"],
        "desc": "运动员或教练转会签约",
    },
    "伤病事件": {
        "triggers": ["受伤","骨折","报销","退赛","伤病","手术","康复"],
        "slots": ["时间","主体","原因","结果"],
        "desc": "运动员伤病或退赛",
    },
    "禁赛处罚": {
        "triggers": ["禁赛","罚款","停赛","处罚","兴奋剂"],
        "slots": ["时间","主体","事由","结果"],
        "desc": "运动员或球队受处罚",
    },
    "纪录突破": {
        "triggers": ["打破纪录","刷新","历史第一","破纪录","创纪录"],
        "slots": ["时间","主体","成绩","赛事"],
        "desc": "体育纪录被打破",
    },
}

# ═══════════════════════════════════════════════════════════
# 2. 事件抽取器
# ═══════════════════════════════════════════════════════════

class EventExtractor:
    """基于触发词 + 规则 + NER 的事件抽取器。

    通过检测事件触发词定位事件，利用 NER 模块提取的实体
    填充事件要素槽位。
    """

    def __init__(self):
        self._ner = NERExtractor()
        self._event_defs = EVENT_DEFINITIONS
        # 预构建触发词 → 事件类型映射
        self._trigger_map: Dict[str, str] = {}
        for ev_type, ev_def in self._event_defs.items():
            for trigger in ev_def["triggers"]:
                if trigger not in self._trigger_map:
                    self._trigger_map[trigger] = ev_type

    def extract(self, text: str) -> List[Dict]:
        """从文本中抽取所有事件。

        Returns
        -------
        List[Dict]: [
            {
                "type": "产品发布",
                "trigger": "发布",
                "trigger_pos": (start, end),
                "slots": {"主体": "某公司", "产品": "新款手机", ...},
            }
        ]
        """
        if not text or not text.strip():
            return []

        events = []

        # 1. NER 识别实体
        entities = self._ner.recognize(text)

        # 2. 按触发词检测事件
        # 按触发词长度降序匹配（长触发词优先，如"发布召回公告" vs "发布"）
        sorted_triggers = sorted(self._trigger_map.keys(), key=len, reverse=True)

        # 文本中所有触发词的位置
        trigger_matches = []
        for trigger in sorted_triggers:
            idx = 0
            while idx < len(text):
                pos = text.find(trigger, idx)
                if pos == -1:
                    break
                trigger_matches.append({
                    "trigger": trigger,
                    "type": self._trigger_map[trigger],
                    "start": pos,
                    "end": pos + len(trigger),
                })
                idx = pos + len(trigger)

        # 3. 按位置排序触发词
        trigger_matches.sort(key=lambda m: m["start"])

        # 4. 合并重叠触发词（保留最可能的事件类型）
        trigger_matches = self._merge_triggers(trigger_matches)

        # 5. 为每个触发词填充事件槽位
        for tm in trigger_matches:
            ev_type = tm["type"]
            ev_def = self._event_defs.get(ev_type, {})
            slots = self._fill_slots(
                text, entities, tm["start"], tm["end"],
                ev_def.get("slots", []), tm["trigger"],
            )
            events.append({
                "type": ev_type,
                "trigger": tm["trigger"],
                "trigger_pos": (tm["start"], tm["end"]),
                "slots": slots,
            })

        return events

    def extract_summary(self, text: str) -> List[Dict]:
        """抽取并返回简化的结构化事件摘要。"""
        events = self.extract(text)
        result = []
        for ev in events:
            slots = ev["slots"]
            entry = {
                "事件类型": ev["type"],
            }
            for slot_name, slot_val in slots.items():
                entry[slot_name] = slot_val
            result.append(entry)
        return result

    # ── 内部方法 ──

    def _merge_triggers(self, matches: List[Dict]) -> List[Dict]:
        """合并重叠的触发词匹配，保留最具体的。"""
        if not matches:
            return []
        merged = [matches[0]]
        for m in matches[1:]:
            last = merged[-1]
            # 重叠
            if m["start"] < last["end"]:
                # 保留长触发词
                if len(m["trigger"]) > len(last["trigger"]):
                    merged[-1] = m
            else:
                merged.append(m)
        return merged

    def _fill_slots(
        self,
        text: str,
        entities: List[Dict],
        trigger_start: int,
        trigger_end: int,
        expected_slots: List[str],
        trigger_word: str,
    ) -> Dict[str, str]:
        """为事件填充要素槽位。

        使用 NER 实体 + 上下文启发式规则填充。
        """
        slots: Dict[str, str] = {}

        # 触发词附近的上下文窗口
        context_start = max(0, trigger_start - 80)
        context_end = min(len(text), trigger_end + 80)
        context = text[context_start:trigger_start] + "【" + text[trigger_start:trigger_end] + "】" + text[trigger_end:context_end]

        # 找到附近的 NER 实体
        nearby_entities = [
            e for e in entities
            if abs(e["start"] - trigger_start) < 60
            or abs(e["end"] - trigger_end) < 60
        ]

        # 时间
        if "时间" in expected_slots:
            time_ents = [e for e in nearby_entities if e["type"] == "时间"]
            if time_ents:
                slots["时间"] = time_ents[0]["text"]

        # 地点
        if "地点" in expected_slots:
            place_ents = [e for e in nearby_entities if e["type"] == "地名"]
            if place_ents:
                slots["地点"] = place_ents[0]["text"]

        # 金额
        if "金额" in expected_slots:
            money_ents = [e for e in nearby_entities if e["type"] == "金额"]
            if money_ents:
                slots["金额"] = money_ents[0]["text"]

        # 主体（机构名优先，其次人名）
        if any(s in expected_slots for s in ["主体", "原告", "被告", "机构"]):
            org_ents = [e for e in nearby_entities if e["type"] == "机构名"]
            person_ents = [e for e in nearby_entities if e["type"] == "人名"]
            if org_ents:
                slots["主体"] = org_ents[0]["text"]
            elif person_ents:
                slots["主体"] = person_ents[0]["text"]

        # 产品
        if "产品" in expected_slots:
            prod_ents = [e for e in nearby_entities if e["type"] == "产品名"]
            if prod_ents:
                slots["产品"] = prod_ents[0]["text"]

        # 动作（触发词本身）
        if "动作" in expected_slots:
            slots["动作"] = trigger_word

        # 原因/事由（触发词前面的名词短语）
        if any(s in expected_slots for s in ["原因", "事由", "内容"]):
            # 取触发词前 20 字
            pre_text = text[max(0, trigger_start - 30):trigger_start]
            # 去除已知实体后，剩余的前半段作为原因
            reason = self._extract_reason(pre_text, nearby_entities)
            if reason:
                slots_key = "原因" if "原因" in expected_slots else "事由" if "事由" in expected_slots else "内容"
                slots[slots_key] = reason

        # 结果/影响（触发词后面 30 字）
        if "结果" in expected_slots:
            post_text = text[trigger_end:min(len(text), trigger_end + 30)]
            if post_text.strip():
                slots["结果"] = post_text.strip()[:40]

        return slots

    def _extract_reason(self, pre_text: str, entities: List[Dict]) -> Optional[str]:
        """从触发词前面的文本中提取原因/事由。"""
        if not pre_text.strip():
            return None
        # 去掉附近的实体（避免重复）
        cleaned = pre_text
        for e in entities:
            if e["start"] < len(pre_text):
                cleaned = cleaned.replace(e["text"], "", 1)
        # 取最后一个"因"或"为"后面的内容
        for sep in ["因", "为", "由于", "因为"]:
            if sep in cleaned:
                idx = cleaned.rfind(sep) + len(sep)
                reason = cleaned[idx:].strip()
                if reason and len(reason) > 1:
                    return reason[:20]
        # 如果没有"因为"结构，取 cleaned 文本的前 15 字
        cleaned = re.sub(r"[，,。！？、]+", "", cleaned).strip()
        if cleaned and len(cleaned) >= 2:
            return cleaned[:15]
        return None

    # ── 批量处理 ──

    def extract_batch(self, texts: List[str]) -> List[List[Dict]]:
        """批量抽取事件。"""
        return [self.extract(t) for t in texts]


# ═══════════════════════════════════════════════════════════
# 3. 格式化输出
# ═══════════════════════════════════════════════════════════

def format_events(events: List[Dict]) -> str:
    """格式化显示事件。"""
    if not events:
        return "  (未识别到事件)"

    lines = []
    for i, ev in enumerate(events):
        lines.append(f"  事件{i+1}: 【{ev['type']}】 (触发词: {ev['trigger']})")
        for slot_name, slot_val in ev["slots"].items():
            if slot_val:
                lines.append(f"    {slot_name}: {slot_val}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 4. Demo
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("事件抽取 Demo")
    print("=" * 60)

    ee = EventExtractor()

    test_cases = [
        "2026年6月，某公司在上海发布新款手机。",
        "某品牌因产品质量问题发布召回公告。",
        "北京市公安局刑侦总队政治处主任李小燕公布了典型案例",
        "#白鹿方否认更换编剧团队# 白鹿方六连辟谣",
        "今天天气真好，心情特别愉快！",
        "央视曝养生馆围猎老年人，涉案金额高达3000万余元",
        "上海一幼儿园教师离世，警方介入调查",
        "网信办发布公约整治涉企侵权信息",
    ]

    for text in test_cases:
        print(f"\n原文: {text[:60]}")
        events = ee.extract_summary(text)
        if events:
            for ev in events:
                print(f"  事件类型: {ev['事件类型']}")
                for k, v in ev.items():
                    if k != "事件类型":
                        print(f"    {k}: {v}")
        else:
            print("  (未识别到事件)")

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
                events = ee.extract_summary(text)
                print(f"\n[{topic}]")
                if events:
                    for ev in events[:2]:
                        print(f"  {ev['事件类型']}: ", end="")
                        info = "; ".join(f"{k}={v}" for k, v in ev.items() if k != "事件类型" and v)
                        print(info[:80])
                else:
                    print("  (未识别到事件)")
