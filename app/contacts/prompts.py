"""两层总结的 prompt 与信号代码表。

修改 prompt 文本时必须同步提升版本号：日笔记按 NOTE_PROMPT_VERSION 判断是否需要重跑，
月度总结的 input_fp 也包含 SUMMARY_PROMPT_VERSION。
"""
from __future__ import annotations

from dataclasses import dataclass

NOTE_PROMPT_VERSION = "note-v4"
SUMMARY_PROMPT_VERSION = "summary-v6"


@dataclass(frozen=True)
class SignalDef:
    name: str       # 中文名称，用于总结输入与接口返回
    category: str   # topic 关注话题 | need 深层需求 | service 经营信号
    group: str      # 关注话题的大类；其他类别同 category 的中文名
    hint: str       # 写进 Prompt A 的关键词 / 证据模式提示


def _defs(category: str, group: str, items: list[tuple[str, str, str]]) -> dict[str, SignalDef]:
    return {code: SignalDef(name, category, group, hint) for code, name, hint in items}


# 关注话题与深层需求来自《AI洞察标签总结》；收益导向型与经营信号为补充项，可按需删除。
SIGNALS: dict[str, SignalDef] = {
    **_defs("topic", "投资理财", [
        ("INV_DEPOSIT", "关注存款", "存款、定期、大额存单、结构性存款、利息、到期"),
        ("INV_WEALTH", "关注理财", "理财产品、净值、收益率、业绩基准、封闭期、开放期"),
        ("INV_BOND", "关注债券", "债券、国债、城投债、票面利率、到期收益率、债基"),
        ("INV_FUND", "关注基金", "基金、申购、赎回、定投、持仓、指数基金"),
        ("INV_STOCK", "关注股票", "股票、A股、港股、美股、板块、个股"),
        ("INV_PRIVATE", "关注私募", "私募基金、合格投资者、起投、备案、契约型"),
        ("INV_LIFE_INS", "关注寿险", "寿险、身故、保额、受益人、终身寿险、定期寿险"),
        ("INV_CI_INS", "关注重疾险", "重疾、大病、赔付、确诊"),
        ("INV_ANNUITY", "关注年金险", "年金、生存金、养老金领取、教育年金"),
        ("INV_FAMILY_TRUST", "关注家族信托", "家族信托、受托人、委托人、信托架构、财产隔离"),
        ("INV_INS_TRUST", "关注保险金信托", "保险金信托、保单信托、信托+保险"),
    ]),
    **_defs("topic", "资产配置", [
        ("ALLOC_PRESERVE", "关注资产保全", "保值、抗通胀、购买力、贬值、缩水"),
        ("ALLOC_LIQUIDITY", "关注流动性", "随时取、急用钱、现金流、变现、周转、应急"),
        ("ALLOC_DIVERSIFY", "关注分散投资", "分散、组合、配置、多元化、大类资产、再平衡"),
    ]),
    **_defs("topic", "人生规划", [
        ("LIFE_CHILD_EDU", "子女教育", "孩子上学、教育金、学费、教育储备"),
        ("LIFE_STUDY_ABROAD", "出国留学", "留学、出国、托福、雅思、海外院校"),
        ("LIFE_RETIREMENT", "养老规划", "退休、养老、社保、养老金、养老社区"),
        ("LIFE_INHERITANCE", "财富传承", "传给、留给下一代、继承、遗产、传承"),
    ]),
    **_defs("topic", "税务法律", [
        ("LEGAL_TAX", "税务筹划", "个税、节税、税务筹划、所得税、增值税"),
        ("LEGAL_MARRIAGE", "婚姻财产", "婚前、婚后、离婚、共同财产、协议、公证"),
        ("LEGAL_EQUITY", "企业股权", "股权、股东、公司经营、实控人、收购、转让"),
    ]),
    **_defs("topic", "市场动态", [
        ("MKT_STOCK", "关注股市", "行情、牛市、熊市、震荡、回落、反弹"),
        ("MKT_RATE", "关注利率", "利率、降息、加息、LPR、存款利率、贷款利率"),
        ("MKT_POLICY", "关注政策", "新规、监管、央行、金融政策、宏观调控"),
    ]),
    **_defs("need", "深层需求", [
        ("NEED_RISK_AVERSE", "风险敏感型", "聊产品就问安全吗、保本吗；追问风控；提及过去亏损"),
        ("NEED_CHILD_CENTRIC", "子女中心型", "反复提及子女年龄或学校；主动问教育金；“都是为了孩子”"),
        ("NEED_LEGACY_ANXIETY", "传承焦虑型", "提及“留给孩子的”；问遗产税；关心“百年之后”的安排"),
        ("NEED_CONTROL", "控制权偏好", "“我自己来看”“不要给我推太多”；对建议有抵触"),
        ("NEED_PRIVILEGE", "权益敏感型", "问级别或待遇；对比“别人有什么”；在意 VIP 专属权益"),
        ("NEED_CONVENIENCE", "便捷优先型", "“太复杂了”“能不能简单点”“不想跑银行”"),
        ("NEED_YIELD", "收益导向型", "反复比较收益高低；追问收益更高的产品"),
    ]),
    **_defs("service", "经营信号", [
        ("SVC_COMPETITOR", "他行比较", "拿他行产品比较收益或服务；表示资金可能转出"),
        ("SVC_COMPLAINT", "不满/投诉", "对服务或产品表达不满；有投诉倾向"),
        ("SVC_PURCHASE_INTENT", "明确购买意向", "明确表示要购买、追加、调整或赎回某类产品"),
    ]),
}

# 深层需求需要多次出现：窗口内出现的天数达到该值才算“已确认”。
NEED_CONFIRM_DAYS = 2


def signal_name(code: str) -> str:
    d = SIGNALS.get(code)
    return d.name if d else code


def _signal_table() -> str:
    lines, current = [], None
    for code, d in SIGNALS.items():
        title = {"topic": f"关注话题·{d.group}", "need": "深层需求", "service": "经营信号"}[d.category]
        if title != current:
            lines.append(f"【{title}】")
            current = title
        lines.append(f"{code} {d.name}：{d.hint}")
    return "\n".join(lines)


_SEMANTIC_MAP = """\
“我想给孩子留点钱” → LIFE_INHERITANCE
“怎么把钱安全地传下去” → LIFE_INHERITANCE
“不想让后代争财产” → LIFE_INHERITANCE
“现在利率一直降，钱放哪里好” → MKT_RATE
“钱放在银行越来越不值钱了” → ALLOC_PRESERVE
“想把资产和公司风险隔离开” → ALLOC_PRESERVE
“年纪大了，想早点退下来” → LIFE_RETIREMENT
“最近那个新政策对我们有啥影响” → MKT_POLICY
“孩子快上大学了，得准备学费” → LIFE_CHILD_EDU
“公司最近要扩产，需要资金” → LEGAL_EQUITY
“担心以后生病了拖累孩子” → INV_CI_INS"""


NOTE_SYSTEM = f"""你是银行财富管理业务的沟通记录分析助手。任务：从“客户经理与某一位客户”的聊天中，
提取供后续月度总结使用的事实笔记和客户标签。

规则：
1. 只依据给出的聊天内容，不推测、不补充；无法确定的内容不写。
2. 区分发言人：客户的态度、诉求、顾虑只能来自“客”的发言；推荐、解释、承诺只能来自“经”的发言。
3. 忽略寒暄、问候、表情、节日祝福等无业务含义内容。某天若无业务内容，该天不输出；
   全部无业务内容时 has_business_content 为 false，days 为空数组。
4. 标签（signals）只能使用下方代码表中的 code，并且只依据客户的发言：
   - 关注话题：客户主动提及或询问即可标注。经理提到的产品或方案，客户没有接话或表示兴趣的不标注。
     反例：客户问“我这些资产以后怎么安排比较放心”，经理答“可以考虑家族信托或者保险金信托”，
     客户只回“这事一直压在心上”——话题只标 LIFE_INHERITANCE，不标 INV_FAMILY_TRUST、INV_INS_TRUST；
     “压在心上”体现的传承焦虑另按深层需求标注 NEED_LEGACY_ANXIETY。
   - 深层需求：反映客户的长期特点，按当天出现的证据标注；是否成立由后续多天累计判断，不需要你判断。
   - 经营信号：他行比较、不满投诉、明确购买意向，有明确表达才标注。
   - 关键词只是提示：客户没说关键词但意思相同也要标注（见映射示例）；只是顺口提到、与客户需求无关的不标注。
   - 每句话只取最贴切的 1 个标签；同一句话确实涉及两个不同方面时最多 2 个。
   - 信托、保险、私募等具体产品类话题，只有客户提到该产品或其关键特征（如“生病”“身故”“财产隔离”）时才标注。
   - status="open"：本次出现或加强；status="closed"：有明确证据表明已解决或不再成立
     （如“已经办好了”“已经买了”“不考虑了”“不比了”）。没有明确证据不得关闭。
   - 对“当前未关闭的标签”，本次无相关内容时不要输出；同一天同一 code 只输出一次。
5. evidence 为客户原话的简短摘录（≤30字），必须逐字摘自聊天记录，不得改写数字。
6. 不输出姓名、手机号、证件号、账号、地址；金额只保留量级（如“约50万”）。
7. 每个字段简洁，用名词短语或短句；只输出 JSON。

标签代码表（code 名称：关键词或证据模式）：
{_signal_table()}

语义映射示例（客户没说关键词，但表达了相同意思；每句对应最贴切的 1 个）：
{_SEMANTIC_MAP}

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
      "signals": [{{"code": "INV_FUND", "status": "open", "evidence": "那只基金最近怎么一直跌"}}],
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
        "当前未关闭的标签（来自此前记录，供你判断本次是否出现关闭证据）：\n"
        f"{lines}\n\n"
        "聊天记录（客=客户，经=客户经理；已脱敏，已去除寒暄）：\n"
        f"{transcript}"
    )


SUMMARY_SYSTEM = """你是银行客户经理的助理，负责撰写“某位客户近一段时间的接触总结”，供客户经理在下次联系客户前全面了解这位客户。

写作要求：
1. 输出一段连贯的中文，不分点、不加标题，以“客户”指代客户。篇幅按业务内容多少决定，不得为凑字数加入推测：
   有业务内容的天数为1天时80～150字，2～3天时120～220字，更多时180～300字。
2. 内容依次为（某部分无内容就整段省略，不要写“无”“期间无……”之类的话）：
   ① 关注话题——以“{window_label}”开头，按投资理财、资产配置、人生规划、税务法律、市场动态归类概括客户关注的内容，
      只写出现过的类别，不逐个罗列标签；不写沟通次数、天数等统计数字；
   ② 深层需求与沟通偏好——“多次出现”的深层需求可以写成客户的特点，并点出它对沟通方式的影响
      （如控制权偏好：多提供信息、少主动推荐；便捷优先：优先线上办理；权益敏感：主动说明专属权益）；
      “仅出现1天”的只能写“曾……”，或者不写；不要评论判断过程（不写“已确认”“证据不足”“不宜认定”之类）；
   ③ 经营信号——他行比较、不满投诉、明确购买意向，有则如实写明；
   ④ 期间新增或关闭的情况——写明时间（使用给定的“上旬/中旬/下旬”）与原因，
      如“8月中旬客户明确提及传承安排已办妥，相关焦虑信号已关闭”；
   ⑤ 1～2条具体、可执行的后续跟进建议，结合客户的关注话题与已确认的深层需求。
3. 标签名称和代码是内部用语：用自然语言描述客户的具体情况（如“对回撤较为敏感”），
   不要写英文代码、“××标签”“持续中”；只有已关闭或转变的情况才可按示例写“……信号已关闭”“……判断不再成立”。
4. 区分说话人：利率、费率、产品状态等由经理告知的信息写成“经理已说明……”，不得写成客户提及或询问。
5. 只依据提供的标签、待跟进和笔记；不得编造产品名、收益率、金额、日期。
6. 措辞与证据强度匹配：出现≥2天才可写“多次”；只出现1次写“曾”“提及”。
7. 待跟进事项不确定是否已完成：只能建议“确认是否已完成”，不得断言未完成。
8. 正文是给客户经理看的，不得出现“系统”“笔记”“统计”“标签”“业务内容”等字样，不写沟通次数，也不要复述输入里的天数口径。
9. 建议从客户利益出发，不承诺收益，不使用诱导性销售话术。
10. 只输出 JSON：{{"summary": "...", "main_topics": ["≤3个"], "suggestions": ["≤2条"]}}

风格示例（只示范语气、结构和篇幅，写的是另一位样例客户）：
近1个月客户的对话主要集中在基金净值波动与稳健配置；客户对回撤较为敏感，多次主动询问替代方案与下行保护。8月中旬客户明确提及传承安排已办妥，相关焦虑信号已关闭。期间客户曾比较他行产品收益，后自行完成配置，收益导向型判断不再成立。建议后续以稳健配置与长期陪伴为主线，减少短期收益对比，并在市场波动时主动同步持仓表现，巩固信任。

注意：示例中的结论和建议（如“收益导向型判断不再成立”“以稳健配置与长期陪伴为主线”）属于样例客户，不能套用到当前客户。
只有输入中有对应依据时才能写类似的结论，跟进建议要根据当前客户的话题和需求来写。"""


def window_label(days: int) -> str:
    return "近1个月" if days == 30 else f"近{days}天"


def build_summary_system(window_days: int) -> str:
    return SUMMARY_SYSTEM.format(window_label=window_label(window_days))


def build_summary_user(window: str, signal_lines: list[str],
                       pending_lines: list[str], note_lines: list[str]) -> str:
    def block(lines: list[str]) -> str:
        return "\n".join(lines) if lines else "无"

    return (
        f"统计窗口：{window}\n\n"
        "【客户标签】（已整理，状态以此为准）\n"
        f"{block(signal_lines)}\n\n"
        "【待跟进】（尚未确认是否完成）\n"
        f"{block(pending_lines)}\n\n"
        "【每日笔记】\n"
        f"{block(note_lines)}"
    )
