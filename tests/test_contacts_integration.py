"""接触总结端到端测试：真实 MySQL + Redis，模型用确定性假客户端替代。

覆盖：冷启动补数 → 日笔记/信号/统计 → 月度总结 → 接口查询 → 再次提交全部复用。
测试数据使用 zzit_ 前缀的独立工号与客户 ID，结束后清理。MySQL/Redis 不可达时跳过。

本地运行（docker compose up -d mysql redis 之后）：
MYSQL_HOST=127.0.0.1 REDIS_HOST=127.0.0.1 python -m unittest tests.test_contacts_integration
"""
import json
import re
import threading
import unittest
import uuid
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text, update

from app.contacts import jobs
from app.contacts.limiter import SlidingWindowLimiter
from app.contacts.llm import ContactLlm
from app.contacts.prompts import NOTE_SYSTEM
from app.contacts.worker import execute_summary_job
from app.db import get_engine, get_session
from app.models import (Base, ContactDailyNote, ContactDailyStat, ContactNoteRun, ContactSignalEvent,
                        ContactSummary, ContactSummaryJob, LlmCall)
from app.sharding import all_shard_tables, shard_table

MANAGER = "zzit_mgr"
C1, C2 = "wmzzit_c1", "wmzzit_c2"
END = date(2026, 8, 19)

# (时间, 发送方是否客户, 客户, 内容)
CHATS = [
    ("2026-07-22 20:05", True, C1, "最近那只债基怎么一直在跌，这个月都亏了快两个点了"),
    ("2026-07-22 20:12", False, C1, "主要是债市调整导致净值回撤"),
    ("2026-07-22 20:15", True, C1, "我是想稳一点的，回撤多了心里不踏实"),
    ("2026-07-24 14:30", True, C1, "孩子在国外定居了，我这些资产以后怎么安排比较放心"),
    ("2026-07-24 14:40", True, C1, "嗯，这事一直压在心上"),
    ("2026-07-28 19:02", True, C1, "[他行]那款年化3.2%，你们这款才2.6%，差得有点多啊"),
    ("2026-07-30 08:10", False, C1, "台风今天登陆，您注意安全"),
    ("2026-07-30 08:32", True, C1, "好的谢谢"),
    ("2026-08-02 10:00", True, C1, "有没有带下行保护的产品，回撤太难受了"),
    ("2026-08-02 10:05", False, C1, "我下周把您整体持仓的收益和回撤整理一份发您"),
    ("2026-08-06 16:40", True, C1, "收益那个事我不纠结了，自己已经配好了"),
    ("2026-08-14 09:12", True, C1, "传承那块上周已经跟律师办妥了，不用再操心了"),
    ("2026-08-18 23:50", True, C1, "市场又跌了，我的持仓回撤多少"),
    ("2026-08-19 00:10", False, C1, "整体回撤约1.8%，在可承受范围"),   # 跨零点，属于前一天的会话
    ("2026-08-05 10:00", True, C2, "大额存单现在利率多少"),
    ("2026-08-05 10:02", False, C2, "三年期年化2.9%"),
    ("2026-08-08 09:00", True, C2, "好的"),                          # 纯寒暄，不调用模型
    ("2026-07-10 10:00", True, C1, "窗口之外的消息不应计入"),
]

RULES = [("回撤", "NEED_RISK_AVERSE", "open"), ("压在心上", "NEED_LEGACY_ANXIETY", "open"),
         ("差得有点多", "NEED_YIELD", "open"), ("不纠结", "NEED_YIELD", "closed"),
         ("办妥", "NEED_LEGACY_ANXIETY", "closed")]


def fake_note(user: str) -> dict:
    days = []
    for day, body in re.findall(r"\[(\d{4}-\d{2}-\d{2})\]\n(.*?)(?=\n\n\[|\Z)", user, re.S):
        customer = [l.split("客：", 1)[1] for l in body.splitlines() if " 客：" in l]
        manager = [l.split("经：", 1)[1] for l in body.splitlines() if " 经：" in l]
        pieces = [p for line in customer for p in line.split(" / ")]
        if not pieces:
            continue
        signals = [{"code": code, "status": status, "evidence": p[:30]}
                   for p in pieces for kw, code, status in RULES if kw in p]
        if any("大额存单" in p for p in pieces):
            signals.append({"code": "SVC_PURCHASE_INTENT", "status": "open", "evidence": "客户非常想买大额存单"})
        days.append({"date": day, "topics": ["业务咨询"], "customer_points": pieces[:3],
                     "manager_actions": manager[:2],
                     "facts": ["客户完成传承安排"] if any("办妥" in p for p in pieces) else [],
                     "signals": signals,
                     "pending": ["整理持仓收益与回撤"] if any("整理" in m for m in manager) else []})
    return {"has_business_content": bool(days), "days": days}


def fake_summary(user: str) -> dict:
    assert "沟通统计" not in user  # 统计不再送给模型
    return {"summary": "近1个月客户的对话主要集中在投资理财类的基金回撤与稳健配置；客户对回撤较为敏感，"
                       "多次主动询问下行保护及替代方案。期间客户曾比较他行理财收益，后表示已自行完成配置。"
                       "建议后续以稳健配置与长期陪伴为主线，市场波动时主动同步持仓表现。",
            "main_topics": ["基金回撤"], "suggestions": ["市场波动时主动同步持仓表现"]}


class FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, model, messages, **kwargs):
        system, user = messages[0]["content"], messages[1]["content"]
        kind = "note" if system == NOTE_SYSTEM else "summary"
        self.calls.append(kind)
        data = fake_note(user) if kind == "note" else fake_summary(user)
        usage = SimpleNamespace(prompt_tokens=len(system + user) // 2, completion_tokens=80,
                                prompt_cache_hit_tokens=0)
        message = SimpleNamespace(content=json.dumps(data, ensure_ascii=False))
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=message)])


def _reachable() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        from app.redis_client import get_redis
        get_redis().ping()
        return True
    except Exception:
        return False


def cleanup():
    session = get_session()
    try:
        ids = [MANAGER, C1, C2]
        for tbl in all_shard_tables():
            session.execute(text(f"DELETE FROM `{tbl}` WHERE from_user IN :ids OR to_user IN :ids")
                            .bindparams(ids=ids).execution_options(), {"ids": ids})
        for model in (ContactDailyStat, ContactDailyNote, ContactSignalEvent, ContactSummary,
                      ContactSummaryJob, ContactNoteRun, LlmCall):
            session.execute(delete(model).where(model.manager_id == MANAGER))
        session.commit()
    finally:
        session.close()


class ContactSummaryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _reachable():
            raise unittest.SkipTest("MySQL/Redis 不可达")
        Base.metadata.create_all(get_engine())
        cleanup()
        session = get_session()
        try:
            for i, (t, from_customer, customer, content) in enumerate(CHATS):
                sender, receiver = (customer, MANAGER) if from_customer else (MANAGER, customer)
                ts = datetime.fromisoformat(t)
                session.execute(text(
                    f"INSERT INTO `{shard_table(sender)}` (msg_id, content, create_date, msg_time, "
                    "from_user, to_user, msg_type) VALUES (:id, :c, :d, :t, :f, :to, 'text')"),
                    {"id": f"zzit-{i}", "c": content, "d": ts.date(), "t": ts, "f": sender, "to": receiver})
            session.commit()
        finally:
            session.close()

    @classmethod
    def tearDownClass(cls):
        cleanup()

    def run_job(self, client: FakeClient, **kwargs) -> ContactSummaryJob:
        job, _ = jobs.enqueue_summary_job(MANAGER, END, kwargs.get("customer_ids"), kwargs.get("force", False))
        # 只领取本测试创建的任务：库里可能有真实任务在排队或租约过期，不能用 claim_summary_job 抢。
        token = uuid.uuid4().hex
        session = get_session()
        try:
            session.execute(update(ContactSummaryJob).where(ContactSummaryJob.job_id == job.job_id).values(
                status="running", attempt_token=token, heartbeat_at=datetime.utcnow(), started_at=datetime.utcnow()))
            session.commit()
            row = session.get(ContactSummaryJob, job.job_id)
        finally:
            session.close()
        lease = jobs.Lease(ContactSummaryJob, "job_id", job.job_id, token)
        limiter = SlidingWindowLimiter(f"zzit_{uuid.uuid4().hex[:6]}", max_requests=100,
                                       request_window=1, max_tokens=100_000)
        self.addCleanup(lambda: limiter.redis.delete(*limiter.keys))
        with patch("app.contacts.worker.ContactLlm",
                   lambda stop=None: ContactLlm(limiter=limiter, client=client, stop=stop)):
            execute_summary_job(lease, row, threading.Event())
        session = get_session()
        try:
            return session.get(ContactSummaryJob, job.job_id)
        finally:
            session.close()

    def query(self, model, **filters):
        session = get_session()
        try:
            return session.execute(select(model).filter_by(manager_id=MANAGER, **filters)).scalars().all()
        finally:
            session.close()

    def test_end_to_end(self):
        client = FakeClient()
        job = self.run_job(client)
        self.assertEqual((job.status, job.stage), ("success", "done"))
        self.assertEqual((job.total_customers, job.done_customers, job.failed_count), (2, 2, 0))
        # 冷启动：两个客户各一次打包的日笔记调用 + 两次总结；纯寒暄的 08-08 不调用模型
        self.assertEqual(sorted(client.calls), ["note", "note", "summary", "summary"])

        runs = self.query(ContactNoteRun)
        self.assertEqual(len(runs), 30)
        self.assertTrue(all(r.status == "success" and r.trigger_type == "backfill" for r in runs))

        stats = {(s.customer_id, s.stat_date.isoformat()): s for s in self.query(ContactDailyStat)}
        self.assertNotIn((C1, "2026-07-10"), stats)
        self.assertEqual(stats[(C1, "2026-08-19")].session_count, 0)  # 跨零点延续
        self.assertEqual(stats[(C1, "2026-07-30")].customer_initiated, 0)

        notes = {(n.customer_id, n.note_date.isoformat()): n for n in self.query(ContactDailyNote)}
        self.assertEqual(notes[(C1, "2026-07-30")].status, "no_business")
        self.assertEqual(notes[(C2, "2026-08-08")].status, "no_business")
        self.assertIsNone(notes[(C2, "2026-08-08")].llm_call_id)
        self.assertEqual(notes[(C1, "2026-08-02")].pending, ["整理持仓收益与回撤"])

        events = self.query(ContactSignalEvent)
        estate = [(e.event_date.isoformat(), e.status) for e in events if e.signal_code == "NEED_LEGACY_ANXIETY"]
        self.assertEqual(sorted(estate), [("2026-07-24", "open"), ("2026-08-14", "closed")])
        fake = [e for e in events if e.signal_code == "SVC_PURCHASE_INTENT"]
        self.assertEqual([e.evidence_verified for e in fake], [False])

        summaries = {s.customer_id: s for s in self.query(ContactSummary)}
        c1 = summaries[C1]
        self.assertEqual(c1.status, "success")
        self.assertNotRegex(c1.summary, r"沟通\d+次")
        self.assertEqual(c1.stats_json["session_count"], 8)
        self.assertEqual(c1.stats_json["business_days"], 7)
        status = {s["code"]: s["status"] for s in c1.signals_json}
        self.assertEqual(status, {"NEED_RISK_AVERSE": "open", "NEED_LEGACY_ANXIETY": "closed", "NEED_YIELD": "closed"})
        self.assertEqual(summaries[C2].stats_json["session_count"], 2)
        self.assertEqual(summaries[C2].signals_json, [])  # 未核实的证据不进时间线

        calls = self.query(LlmCall)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(c.status == "ok" and c.prompt_tokens > 0 for c in calls))

        # 再次提交：笔记已覆盖、输入未变，全部复用，不再调用模型
        again = self.run_job(client)
        self.assertEqual((again.status, again.reused_count), ("success", 2))
        self.assertEqual(len(client.calls), 4)

        # prompt / 标签代码表升级后，旧版本笔记的日期需要重算
        session = get_session()
        try:
            session.execute(update(ContactDailyNote).where(
                ContactDailyNote.manager_id == MANAGER, ContactDailyNote.note_date == date(2026, 8, 6))
                .values(prompt_version="note-v0"))
            session.commit()
        finally:
            session.close()
        window = [date(2026, 7, 21) + timedelta(days=i) for i in range(30)]
        self.assertEqual(jobs.missing_note_dates(MANAGER, window), [date(2026, 8, 6)])

        self.check_api(job.job_id)

    def check_api(self, job_id):
        from app.api.main import app
        api = TestClient(app)

        r = api.post("/contacts/summary-job/query", json={"job_id": job_id}).json()
        self.assertEqual(r["progress"]["done_customers"], 2)

        r = api.post("/contacts/summary/list", json={"manager_id": MANAGER, "end_date": END.isoformat(),
                                                      "sort": "signal_change"}).json()
        self.assertEqual([i["customer_id"] for i in r["items"]], [C1, C2])
        self.assertEqual({s["code"] for s in r["items"][0]["closed_signals"]}, {"NEED_LEGACY_ANXIETY", "NEED_YIELD"})

        r = api.post("/contacts/summary/list", json={"manager_id": MANAGER, "end_date": END.isoformat(),
                                                      "signal": "NEED_RISK_AVERSE", "signal_status": "open"}).json()
        self.assertEqual(r["total"], 1)

        r = api.post("/contacts/summary/detail", json={"manager_id": MANAGER, "customer_id": C1,
                                                        "end_date": END.isoformat()}).json()
        closed = next(s for s in r["signals"] if s["code"] == "NEED_LEGACY_ANXIETY")
        self.assertEqual((closed["closed_date"], closed["evidence"]),
                         ("2026-08-14", "传承那块上周已经跟律师办妥了，不用再操心了"))

        r = api.post("/contacts/note/list", json={"manager_id": MANAGER, "customer_id": C2,
                                                   "start_date": "2026-08-01", "end_date": "2026-08-19"}).json()
        self.assertEqual([i["date"] for i in r["items"]], ["2026-08-05"])
        self.assertEqual(r["items"][0]["signals"][0]["verified"], False)
        self.assertEqual(r["missing_dates"], [])

        r = api.post("/contacts/signal/list", json={"manager_id": MANAGER, "code": "NEED_LEGACY_ANXIETY",
                                                     "end_date": END.isoformat()}).json()
        self.assertEqual([(i["customer_id"], i["status"]) for i in r["items"]], [(C1, "closed")])

        self.assertEqual(api.post("/contacts/summary/list", json={"manager_id": "zzit_nobody",
                                                                   "end_date": END.isoformat()}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
