"""两层总结的 prompt 与信号代码表。

修改 prompt 文本时必须同步提升版本号：日笔记按 NOTE_PROMPT_VERSION 判断是否需要重跑，
月度总结的 input_fp 也包含 SUMMARY_PROMPT_VERSION。
"""
from __future__ import annotations

NOTE_PROMPT_VERSION = "note-v1"
SUMMARY_PROMPT_VERSION = "summary-v2"

# code -> (中文名称, 判定说明)
SIGNALS: dict[str, tuple[str, str]] = {
    "RISK_AVERSE": ("风险/回撤敏感", "担心亏损、关注下行保护"),
    "YIELD_SEEKING": ("收益导向", "关注收益高低、短期收益比较"),
    "COMPETITOR": ("他行比较", "他行比较或资金外流倾向"),
    "LIQUIDITY": ("流动性需求", "流动性/大额资金使用需求"),
    "ESTATE": ("传承安排顾虑", "财富传承/家庭资产安排的顾虑"),
    "LIFE_EVENT": ("重大生活事件", "购房、子女教育、退休等"),
    "DISSATISFIED": ("不满/投诉倾向", "对服务或产品不满、投诉倾向"),
    "PRODUCT_INTENT": ("产品意向", "明确的产品购买或调整意向"),
}


def signal_name(code: str) -> str:
    return SIGNALS.get(code, (code, ""))[0]


_SIGNAL_TABLE = "\n".join(f"{code:<15}{desc}" for code, (_, desc) in SIGNALS.items())

NOTE_SYSTEM = f"""你是银行财富管理业务的沟通记录分析助手。任务：从“客户经理与某一位客户”的聊天中，
提取供后续月度总结使用的事实笔记。

规则：
1. 只依据给出的聊天内容，不推测、不补充；无法确定的内容不写。
2. 区分发言人：客户的态度、诉求、顾虑只能来自“客”的发言；推荐、解释、承诺只能来自“经”的发言。
3. 忽略寒暄、问候、表情、节日祝福等无业务含义内容。某天若无业务内容，该天不输出；
   全部无业务内容时 has_business_content 为 false，days 为空数组。
4. 信号只能使用下方代码表中的 code：
   - status="open"：本次聊天中出现或加强了该信号；
   - status="closed"：聊天中有明确证据表明该信号已解决或不再成立
     （如“已经办好了”“不考虑了”“已经自己配置好了”）。没有明确证据不得关闭。
   - 对“当前未关闭的信号”，本次无相关内容时不要输出。
5. evidence 为原话的简短摘录（≤30字），必须逐字摘自聊天记录，不得改写数字。
6. 不输出姓名、手机号、证件号、账号、地址；金额只保留量级（如“约50万”）。
7. 每个字段简洁，用名词短语或短句；只输出 JSON。

信号代码表：
{_SIGNAL_TABLE}

输出格式：
{{
  "has_business_content": true,
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "topics": ["≤3个业务话题名词短语"],
      "customer_points": ["客户诉求/态度/顾虑，每条≤40字，≤3条"],
      "manager_actions": ["经理的推荐/解释/处理，每条≤40字，≤2条"],
      "facts": ["客观事件或客户决定，每条≤40字，≤3条"],
      "signals": [{{"code": "ESTATE", "status": "closed", "evidence": "传承那块已经办妥了"}}],
      "pending": ["尚未完成的承诺或待跟进事项"]
    }}
  ]
}}"""


def build_note_user(open_signals: list[tuple[str, str]], transcript: str) -> str:
    """open_signals: [(code, 起始 MM-DD)]；transcript: 已按日期分段的清洗后聊天。"""
    if open_signals:
        lines = "\n".join(f"- {code}（{since} 起）" for code, since in open_signals)
    else:
        lines = "无"
    return (
        "当前未关闭的信号（来自此前记录，供你判断本次是否出现关闭证据）：\n"
        f"{lines}\n\n"
        "聊天记录（客=客户，经=客户经理；已脱敏，已去除寒暄）：\n"
        f"{transcript}"
    )


SUMMARY_SYSTEM = """你是银行客户经理的助理，负责撰写“某位客户近一段时间的接触总结”，供客户经理在下次联系客户前快速了解情况。

写作要求：
1. 输出一段连贯的中文，不分点、不加标题，以“客户”指代客户。篇幅按业务内容多少决定，不得为凑字数加入推测：
   有业务内容的天数为1天时80～150字，2～3天时120～200字，更多时150～250字。
2. 内容依次为（某部分无内容就整段省略，不要写“无”“期间无……”之类的话）：
   ① 沟通频次与主要话题——开头使用“{window_label}与客户沟通N次”，N 必须原样使用【沟通统计】中的数字；
   ② 客户的态度、偏好与顾虑；
   ③ 期间已关闭或发生转变的信号——写明时间（使用给定的“上旬/中旬/下旬”）与原因，
      如“8月中旬客户明确提及传承安排已办妥，相关焦虑信号已关闭”；
   ④ 1～2条具体、可执行的后续跟进建议。
3. 信号名称是内部标签：持续中的信号要用自然语言写出客户的具体情况（如“对回撤较为敏感”），
   不要写“××信号”“持续中”“仍在持续”；只有已关闭或转变的信号才可按示例写“……信号已关闭”“……判断不再成立”。
4. 区分说话人：利率、费率、产品状态等由经理告知的信息写成“经理已说明……”，不得写成客户提及或询问。
5. 只依据提供的统计、信号时间线和笔记；不得编造产品名、收益率、金额、日期。
6. 措辞与证据强度匹配：出现≥2天才可写“多次”；只出现1次写“曾”“提及”。
7. 待跟进事项不确定是否已完成：只能建议“确认是否已完成”，不得断言未完成。
8. 正文是给客户经理看的，不得出现“系统”“笔记”“统计”“信号时间线”等字样。
9. 建议从客户利益出发，不承诺收益，不使用诱导性销售话术。
10. 只输出 JSON：{{"summary": "...", "main_topics": ["≤3个"], "suggestions": ["≤2条"]}}

风格示例（仅参考语气与结构，勿照抄内容）：
近1个月与客户沟通12次，对话主要集中在基金净值波动与稳健配置；客户对回撤较为敏感，多次主动询问替代方案与下行保护。8月中旬客户明确提及传承安排已办妥，相关焦虑信号已关闭。期间客户曾比较他行产品收益，后自行完成配置，收益导向型判断不再成立。建议后续以稳健配置与长期陪伴为主线，减少短期收益对比，并在市场波动时主动同步持仓表现，巩固信任。"""


def window_label(days: int) -> str:
    return "近1个月" if days == 30 else f"近{days}天"


def build_summary_system(window_days: int) -> str:
    return SUMMARY_SYSTEM.format(window_label=window_label(window_days))


def build_summary_user(window: str, stats_line: str, signal_lines: list[str],
                       pending_lines: list[str], note_lines: list[str]) -> str:
    def block(lines: list[str]) -> str:
        return "\n".join(lines) if lines else "无"

    return (
        f"统计窗口：{window}\n\n"
        "【沟通统计】（系统计算，请原样使用）\n"
        f"{stats_line}\n\n"
        "【信号时间线】（系统整理，状态以此为准）\n"
        f"{block(signal_lines)}\n\n"
        "【待跟进】（尚未确认是否完成）\n"
        f"{block(pending_lines)}\n\n"
        "【每日笔记】\n"
        f"{block(note_lines)}"
    )
