"""生成模拟企业微信聊天数据写入 MySQL。

模拟客户经理与多个客户围绕**多种银行业务场景**的多轮对话，并混入寒暄、非文本
（图片/语音/系统）等噪声，用于验证预处理减量、聚类与热点总结效果。

为贴近真实数据、避免「完全相同的重复内容」（否则去重缓存会把大量会话压成极少
的唯一文本）：
  - 覆盖 14 类业务场景，每类有多条问法模板；
  - 模板含 {金额}/{期限}/{产品} 等占位符，运行时随机填充；
  - 每个会话随机抽取 2~5 条问法、随机顺序、随机开场白，使绝大多数会话文本各不相同。

用法：
    python scripts/seed_data.py --date 2026-06-27 --customers 300
"""
from __future__ import annotations

import argparse
import random
import re
from datetime import datetime, timedelta

from faker import Faker

from app.db import get_engine, get_session
from app.models import Base, Message

fake = Faker("zh_CN")

# ---- 占位符取值池：运行时随机填充，制造文本多样性 ----
SLOTS = {
    "金额": ["3万", "5万", "10万", "20万", "30万", "50万", "80万", "120万", "200万"],
    "产品": ["净值型理财", "固收类理财", "结构性存款", "大额存单", "债券基金",
            "货币基金", "养老理财", "黄金积存"],
    "期限": ["3个月", "6个月", "一年", "两年", "三年", "五年"],
    "到账": ["T+0", "T+1", "T+2", "当天", "次日", "两个工作日内"],
    "分期": ["3期", "6期", "12期", "18期", "24期"],
    "他行": ["招行", "建行", "工行", "他行", "外地银行"],
    "比例": ["20%", "30%", "35%", "40%"],
    "天数": ["3天", "一周", "10天", "15天", "一个月"],
}

# ---- 业务场景：每个场景一组「客户问法模板」+「客服回复模板」----
SCENARIOS: dict[str, dict[str, list[str]]] = {
    "提前还款": {
        "cust": [
            "我想咨询下房贷提前还款的流程是怎样的",
            "提前还款需要交违约金吗，大概多少",
            "我打算提前还{金额}，剩下的月供会变少吗",
            "提前还款是缩短期限划算还是减少月供划算",
            "提交申请后多久能放款核销",
            "经营贷可以提前结清吗，要提前几天预约",
            "部分提前还款有没有次数限制",
        ],
        "staff": ["提前还款需提前预约，可在手机银行操作。",
                  "您已还满一年，按合同约定不收违约金。",
                  "审批通过后一般{天数}内完成扣款核销。"],
    },
    "理财赎回申购": {
        "cust": [
            "我买的那款{产品}现在能赎回吗",
            "赎回后多久到账，是{到账}吗",
            "现在赎回会亏吗，最近收益怎么样",
            "帮我看看有没有收益更高的{产品}推荐",
            "我想申购{金额}的{产品}，今天能买吗",
            "这款{产品}是保本的吗，风险等级是几级",
            "封闭期还有多久，能提前赎回吗",
        ],
        "staff": ["该产品已过封闭期，可预约赎回。",
                  "{到账}到账，节假日顺延。",
                  "近一个月年化约3.2%，按当日净值结算。"],
    },
    "信用卡分期": {
        "cust": [
            "我这笔信用卡账单可以分期吗",
            "分{分期}的手续费率是多少，怎么算",
            "最低还款和账单分期哪个更划算",
            "帮我办{分期}分期吧",
            "分期之后可以提前结清吗，会退手续费吗",
            "这个月账单{金额}，分期每月还多少",
        ],
        "staff": ["您的账单支持分3/6/12期。",
                  "每期费率0.6%，随账单收取。",
                  "已为您提交{分期}分期申请。"],
    },
    "贷款利率": {
        "cust": [
            "现在房贷利率LPR是多少",
            "我的贷款利率什么时候重定价",
            "经营贷的利率比房贷低吗，年化多少",
            "帮我算下转固定利率划不划算",
            "利率下调了我的月供会跟着降吗",
            "首套和二套的执行利率差多少",
        ],
        "staff": ["当前5年期以上LPR为3.95%。",
                  "每年1月1日按最新LPR重定价。",
                  "目前LPR处于低位，建议维持浮动。"],
    },
    "公积金贷款": {
        "cust": [
            "公积金贷款额度怎么算的，最高能贷多少",
            "公积金和商贷能做组合贷吗",
            "首套房首付比例最低是{比例}吗",
            "社保断缴会影响公积金贷款吗",
            "公积金要连续缴存多久才能申请",
            "组合贷的放款顺序是怎样的",
        ],
        "staff": ["额度与缴存基数和年限挂钩，最高120万。",
                  "可做组合贷，公积金部分利率更低。",
                  "连续缴存满6个月即可申请。"],
    },
    "大额存单定期": {
        "cust": [
            "现在{期限}的大额存单利率是多少",
            "大额存单起存金额是{金额}吗",
            "大额存单能提前支取吗，利息怎么算",
            "定期和大额存单哪个利率高",
            "有没有{期限}的高息产品推荐",
        ],
        "staff": ["3年期大额存单年化约2.9%。",
                  "起存20万，可转让。",
                  "提前支取按活期计息。"],
    },
    "转账汇款": {
        "cust": [
            "我转账到{他行}怎么一直没到账",
            "单笔转账限额是多少，能调高吗",
            "跨行转账手续费怎么收",
            "我转错账户了能追回吗",
            "大额转账{金额}需要去柜台吗",
        ],
        "staff": ["跨行到账一般2小时内。",
                  "手机银行单日限额可在APP调整。",
                  "请尽快联系收款行协助。"],
    },
    "信用卡提额年费": {
        "cust": [
            "我的信用卡额度能提升吗，怎么申请",
            "这张卡年费多少，刷几次能免",
            "积分可以兑换什么，怎么用",
            "临时额度到期后会恢复吗",
            "信用卡逾期一天会上征信吗",
        ],
        "staff": ["可在APP申请提额，系统综合评估。",
                  "刷满6次免年费。",
                  "有1天宽限期，建议及时还款。"],
    },
    "征信逾期": {
        "cust": [
            "我想查下自己的征信报告在哪里查",
            "之前有一次逾期记录多久能消除",
            "频繁查征信会影响贷款审批吗",
            "我的逾期已经还清了为什么还显示",
            "征信花了还能办房贷吗",
        ],
        "staff": ["可在人民银行征信中心官网查询。",
                  "逾期记录自结清起保留5年。",
                  "查询次数过多会影响审批。"],
    },
    "账户服务": {
        "cust": [
            "我的银行卡丢了怎么挂失补办",
            "忘记密码了怎么重置",
            "人脸识别一直失败怎么办",
            "我想销户需要带什么材料",
            "怎么开通短信提醒服务",
        ],
        "staff": ["可在APP或网点办理挂失补办。",
                  "携带身份证到网点重置密码。",
                  "请在光线充足处重试人脸识别。"],
    },
    "基金定投": {
        "cust": [
            "我想做基金定投每月扣{金额}怎么设置",
            "定投的基金现在亏了要不要停",
            "定投可以随时赎回吗",
            "帮我推荐几只适合长期定投的基金",
            "定投扣款失败了会怎样",
        ],
        "staff": ["可在APP设置定投计划。",
                  "建议长期持有，不必频繁止盈。",
                  "扣款失败当期跳过，不影响后续。"],
    },
    "银行流水回单": {
        "cust": [
            "我要打印近半年的银行流水怎么弄",
            "电子回单在哪里下载",
            "对公账户的流水可以网银导出吗",
            "签证用的流水需要盖章吗",
        ],
        "staff": ["可在网银自助下载或网点打印。",
                  "电子回单在APP回单查询里下载。",
                  "盖章流水需到网点办理。"],
    },
    "代发工资": {
        "cust": [
            "公司工资发到这张卡了为什么没收到",
            "代发工资卡有什么优惠政策吗",
            "工资卡可以升级成贵宾卡吗",
            "代发客户办房贷有利率优惠吗",
        ],
        "staff": ["代发一般当天到账，请稍候。",
                  "代发客户可享专属理财额度。",
                  "达标可升级贵宾，享利率优惠。"],
    },
    "风险评级适当性": {
        "cust": [
            "我的风险评级是什么时候做的，过期了吗",
            "想买这款{产品}但提示风险不匹配怎么办",
            "风险评测能重新做一次吗",
            "稳健型客户能买{产品}吗",
        ],
        "staff": ["风险评级有效期一年，可重做。",
                  "需评级匹配后方可购买。",
                  "建议到APP重新测评。"],
    },
}

# 场景权重：制造不同热度（部分场景明显更高频）
SCENARIO_WEIGHTS = {
    "提前还款": 6, "理财赎回申购": 6, "信用卡分期": 5, "贷款利率": 5,
    "公积金贷款": 4, "大额存单定期": 4, "转账汇款": 3, "信用卡提额年费": 3,
    "征信逾期": 3, "账户服务": 3, "基金定投": 2, "银行流水回单": 2,
    "代发工资": 2, "风险评级适当性": 2,
}

OPENERS = ["", "", "", "你好，", "在吗？", "请问，", "麻烦问下，", "想咨询下，",
           "经理，", "老师，", "打扰一下，"]
CLOSERS = ["好的，谢谢", "明白了谢谢", "好的我知道了", "多谢", "辛苦了", "嗯嗯好的"]
CHITCHAT = ["在吗", "好的", "收到", "谢谢", "嗯嗯", "👍", "稍等", "您好", "哈哈"]
NON_TEXT = [("image", "[图片]"), ("voice", "[语音]"), ("file", "[文件]对账单.pdf"),
            ("emotion", "[微笑]"), ("system", "对方撤回了一条消息")]

_SLOT_RE = re.compile(r"\{(\w+)\}")


def _fill(template: str) -> str:
    """把模板里的 {占位符} 替换成随机取值，每个占位符独立采样。"""
    return _SLOT_RE.sub(lambda m: random.choice(SLOTS[m.group(1)]), template)


def init_tables() -> None:
    Base.metadata.create_all(get_engine())


def _mk(seq, sender, receiver, role, msg_type, content, t) -> Message:
    return Message(
        msg_id=f"M{t:%Y%m%d}{seq:08d}",
        sender=sender,
        receiver=receiver,
        role=role,
        msg_type=msg_type,
        content=content,
        msg_time=t,
    )


def gen_messages(date_str: str, num_customers: int) -> list[Message]:
    base_day = datetime.strptime(date_str, "%Y-%m-%d")
    scenarios = list(SCENARIO_WEIGHTS.keys())
    weights = list(SCENARIO_WEIGHTS.values())
    msgs: list[Message] = []
    seq = 0

    for c in range(num_customers):
        staff = f"RM{random.randint(1, 30):03d}"
        customer = f"EXT{c:05d}"
        scenario = random.choices(scenarios, weights=weights, k=1)[0]
        cust_pool = SCENARIOS[scenario]["cust"]
        staff_pool = SCENARIOS[scenario]["staff"]

        # 随机抽取 2~5 条问法、随机顺序 → 会话文本各不相同
        k = min(random.randint(2, 5), len(cust_pool))
        chosen = random.sample(cust_pool, k)

        t = base_day + timedelta(hours=random.randint(8, 20), minutes=random.randint(0, 59))

        # 偶尔以寒暄开场（噪声）
        if random.random() < 0.4:
            t += timedelta(seconds=random.randint(20, 90))
            seq += 1
            msgs.append(_mk(seq, customer, staff, "customer", "text",
                            random.choice(CHITCHAT), t))

        for tmpl in chosen:
            line = random.choice(OPENERS) + _fill(tmpl)
            t += timedelta(seconds=random.randint(30, 180))
            seq += 1
            msgs.append(_mk(seq, customer, staff, "customer", "text", line, t))

            # 偶尔插入非文本噪声
            if random.random() < 0.15:
                mt, body = random.choice(NON_TEXT)
                t += timedelta(seconds=random.randint(10, 60))
                seq += 1
                msgs.append(_mk(seq, customer, staff, "customer", mt, body, t))

            # 客服回复（不参与客户侧聚合，仅作真实语料）
            t += timedelta(seconds=random.randint(30, 180))
            seq += 1
            msgs.append(_mk(seq, staff, customer, "staff", "text", _fill(random.choice(staff_pool)), t))

        # 结尾寒暄（噪声）
        if random.random() < 0.6:
            t += timedelta(seconds=random.randint(20, 120))
            seq += 1
            msgs.append(_mk(seq, customer, staff, "customer", "text",
                            random.choice(CLOSERS), t))

    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--customers", type=int, default=300)
    ap.add_argument("--seed", type=int, default=None, help="随机种子（复现用，默认不固定）")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    init_tables()
    msgs = gen_messages(args.date, args.customers)

    session = get_session()
    try:
        session.query(Message).filter(
            Message.msg_time >= f"{args.date} 00:00:00",
            Message.msg_time <= f"{args.date} 23:59:59",
        ).delete(synchronize_session=False)
        session.bulk_save_objects(msgs)
        session.commit()
    finally:
        session.close()

    # 统计唯一客户侧文本占比，直观体现「减少重复」
    cust_texts = [m.content for m in msgs if m.role == "customer" and m.msg_type == "text"]
    uniq = len(set(cust_texts))
    print(f"[seed] 写入 {len(msgs)} 条消息 (date={args.date}, customers={args.customers}); "
          f"客户文本 {len(cust_texts)} 条, 其中唯一 {uniq} 条")


if __name__ == "__main__":
    main()
