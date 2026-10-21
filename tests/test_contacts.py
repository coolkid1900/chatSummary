"""接触总结纯逻辑单测：不访问 MySQL、Redis 或模型接口。"""
import unittest
from datetime import date, datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.contacts.notes import (DayTranscript, fit_transcript, open_signals_from_events, pack_batches,
                                parse_note_output)
from app.contacts.reader import ChatMsg, group_pair_days, to_chat_msg
from app.contacts.stats import aggregate_window, compute_day_stats
from app.contacts.summary import (build_signal_timeline, build_summary_input, fit_note_lines,
                                  signal_lines, template_summary, validate_summary)
from app.contacts.transcript import clean_day, estimate_tokens, period_label

PREFIXES = ("wm", "wo")


def msg(i, t, from_customer=True, content="内容", manager="mgr1", customer="wmC1"):
    return ChatMsg(id=i, manager_id=manager, customer_id=customer, from_customer=from_customer,
                   content=content, msg_time=datetime.fromisoformat(t))


def note(d, status="ok", topics=("基金回撤",), points=(), actions=(), facts=(), pending=()):
    return SimpleNamespace(note_date=date.fromisoformat(d), status=status, topics=list(topics),
                           customer_points=list(points), manager_actions=list(actions),
                           facts=list(facts), pending=list(pending))


def event(i, d, code, status, evidence="原话"):
    return SimpleNamespace(id=i, event_date=date.fromisoformat(d), signal_code=code,
                           status=status, evidence=evidence)


def stat(d, sessions=1, initiated=1, msgs=4, last="10:00:00"):
    return SimpleNamespace(stat_date=date.fromisoformat(d), session_count=sessions,
                           customer_initiated=initiated, msg_count=msgs,
                           last_msg_time=datetime.fromisoformat(f"{d} {last}"))


class ReaderTests(unittest.TestCase):
    def row(self, from_user, to_user):
        return SimpleNamespace(id=1, from_user=from_user, to_user=to_user, content="x",
                               msg_time=datetime(2026, 8, 14, 9, 0))

    def test_roles_from_external_id_prefix(self):
        m = to_chat_msg(self.row("wmC1", "mgr1"), PREFIXES)
        self.assertEqual((m.manager_id, m.customer_id, m.from_customer), ("mgr1", "wmC1", True))
        m = to_chat_msg(self.row("mgr1", "woC2"), PREFIXES)
        self.assertEqual((m.manager_id, m.customer_id, m.from_customer), ("mgr1", "woC2", False))

    def test_non_manager_customer_pairs_are_skipped(self):
        self.assertIsNone(to_chat_msg(self.row("mgr1", "mgr2"), PREFIXES))
        self.assertIsNone(to_chat_msg(self.row("wmC1", "wmC2"), PREFIXES))

    def test_group_sorts_each_pair_day_by_time_then_id(self):
        grouped = group_pair_days([msg(3, "2026-08-14 10:00"), msg(2, "2026-08-14 09:00"),
                                   msg(1, "2026-08-14 09:00"), msg(4, "2026-08-15 08:00")])
        self.assertEqual([m.id for m in grouped[("mgr1", "wmC1", date(2026, 8, 14))]], [1, 2, 3])
        self.assertEqual(len(grouped), 2)


class StatsTests(unittest.TestCase):
    def test_sessions_split_by_gap_and_initiator(self):
        msgs = [msg(1, "2026-08-14 09:00"), msg(2, "2026-08-14 09:10", False),
                msg(3, "2026-08-14 11:00", False), msg(4, "2026-08-14 11:05")]
        s = compute_day_stats(msgs, 30)
        self.assertEqual((s.session_count, s.customer_initiated), (2, 1))
        self.assertEqual((s.msg_count, s.customer_msg_count, s.manager_msg_count), (4, 2, 2))

    def test_session_crossing_midnight_counts_on_start_day(self):
        msgs = [msg(1, "2026-08-15 00:10"), msg(2, "2026-08-15 08:00", False)]
        s = compute_day_stats(msgs, 30, prev_last_time=datetime(2026, 8, 14, 23, 50))
        self.assertEqual((s.session_count, s.customer_initiated), (1, 0))

    def test_fingerprint_changes_when_content_changes(self):
        a = compute_day_stats([msg(1, "2026-08-14 09:00", content="a")], 30)
        b = compute_day_stats([msg(1, "2026-08-14 09:00", content="b")], 30)
        self.assertNotEqual(a.source_fp, b.source_fp)

    def test_aggregate_window(self):
        w = aggregate_window([stat("2026-08-02", 2, 1), stat("2026-08-18", 1, 0, last="20:00:00")])
        self.assertEqual((w.session_count, w.customer_initiated, w.active_days), (3, 1, 2))
        self.assertEqual(w.last_contact, date(2026, 8, 18))


class TranscriptTests(unittest.TestCase):
    def test_clean_masks_merges_and_drops_chitchat(self):
        day = date(2026, 8, 14)
        msgs = [msg(1, "2026-08-14 09:12", content="好的"),
                msg(2, "2026-08-14 09:13", content="我手机13812345678"),
                msg(3, "2026-08-14 09:14", content="想问下回撤"),
                msg(4, "2026-08-14 09:20", False, content="可以看看固收+")]
        tr = clean_day(day, msgs, frozenset({"好的"}))
        self.assertEqual(tr.lines, ["09:13 客：我手机[手机号] / 想问下回撤", "09:20 经：可以看看固收+"])
        self.assertTrue(tr.text.startswith("[2026-08-14]\n"))

    def test_only_chitchat_gives_empty_transcript(self):
        tr = clean_day(date(2026, 7, 30), [msg(1, "2026-07-30 08:10", content="谢谢")], frozenset({"谢谢"}))
        self.assertEqual(tr.lines, [])

    def test_long_message_truncated(self):
        tr = clean_day(date(2026, 8, 1), [msg(1, "2026-08-01 08:00", content="长" * 400)], frozenset())
        self.assertTrue(tr.lines[0].endswith("…"))
        self.assertLess(len(tr.lines[0]), 320)

    def test_period_label_boundaries(self):
        self.assertEqual([period_label(date(2026, 8, d)) for d in (10, 11, 20, 21)],
                         ["8月上旬", "8月中旬", "8月中旬", "8月下旬"])

    def test_estimate_weights_cjk_higher(self):
        self.assertGreater(estimate_tokens("中文中文中文中文"), estimate_tokens("abcdefgh"))


class NoteOutputTests(unittest.TestCase):
    def setUp(self):
        d = date(2026, 8, 14)
        self.trs = {d: DayTranscript(d, ["09:12 客：传承那块上周已经跟律师办妥了，不用再操心了",
                                         "09:20 客：有没有回撤小一点的替代方案"])}

    def test_evidence_ignores_punctuation_and_masking_brackets(self):
        d = date(2026, 8, 23)
        trs = {d: DayTranscript(d, ["19:02 客：[他行]那款一年期理财年化3.2%，你们这款才2.6%"])}
        data = {"days": [{"date": "2026-08-23", "signals": [
            {"code": "SVC_COMPETITOR", "status": "open", "evidence": "他行那款一年期理财年化3.2%"},
            {"code": "NEED_YIELD", "status": "open", "evidence": "那款年化3.2%"}]}]}
        day = parse_note_output(data, trs)[d]
        self.assertEqual([s.verified for s in day.signals], [True, False])

    def test_verifies_evidence_and_filters_invalid_items(self):
        data = {"has_business_content": True, "days": [
            {"date": "2026-08-14", "topics": ["传承安排", "基金回撤", "稳健", "多余"],
             "customer_points": ["传承已办妥"], "signals": [
                 {"code": "NEED_LEGACY_ANXIETY", "status": "closed", "evidence": "“传承那块上周已经跟律师办妥了”"},
                 {"code": "NEED_RISK_AVERSE", "status": "open", "evidence": "客户很害怕亏钱"},
                 {"code": "UNKNOWN", "status": "open", "evidence": "x"},
                 {"code": "NEED_LEGACY_ANXIETY", "status": "maybe", "evidence": "x"}]},
            {"date": "2026-08-20", "topics": ["不在本批次"]}]}
        parsed = parse_note_output(data, self.trs)
        self.assertEqual(list(parsed), [date(2026, 8, 14)])
        day = parsed[date(2026, 8, 14)]
        self.assertEqual(day.topics, ["传承安排", "基金回撤", "稳健"])
        self.assertEqual([(s.code, s.status, s.verified) for s in day.signals],
                         [("NEED_LEGACY_ANXIETY", "closed", True), ("NEED_RISK_AVERSE", "open", False)])
        self.assertEqual(day.signals[0].evidence, "传承那块上周已经跟律师办妥了")

    def test_missing_days_array_is_invalid(self):
        with self.assertRaises(ValueError):
            parse_note_output({"has_business_content": False}, self.trs)

    def test_no_business_output_gives_no_days(self):
        self.assertEqual(parse_note_output({"has_business_content": False, "days": []}, self.trs), {})

    def test_open_signals_track_latest_status(self):
        rows = [("NEED_RISK_AVERSE", "open", date(2026, 7, 22)), ("NEED_LEGACY_ANXIETY", "open", date(2026, 7, 24)),
                ("NEED_RISK_AVERSE", "open", date(2026, 8, 2)), ("NEED_LEGACY_ANXIETY", "closed", date(2026, 8, 14))]
        self.assertEqual(open_signals_from_events(rows), [("NEED_RISK_AVERSE", date(2026, 7, 22))])
        self.assertEqual(open_signals_from_events([("RISK_AVERSE", "open", date(2026, 7, 22))]), [])

    def test_same_code_kept_once_per_day(self):
        data = {"days": [{"date": "2026-08-14", "signals": [
            {"code": "NEED_LEGACY_ANXIETY", "status": "closed", "evidence": "传承那块上周已经跟律师办妥了"},
            {"code": "NEED_LEGACY_ANXIETY", "status": "open", "evidence": "不用再操心了"}]}]}
        day = parse_note_output(data, self.trs)[date(2026, 8, 14)]
        self.assertEqual([(s.code, s.status) for s in day.signals], [("NEED_LEGACY_ANXIETY", "closed")])

    def test_pack_batches_respects_budget(self):
        items = [(date(2026, 7, d), DayTranscript(date(2026, 7, d), ["10:00 客：" + "字" * 100]), "fp")
                 for d in (24, 22, 28)]
        batches = list(pack_batches(items, budget=150))
        self.assertEqual([[i[0].day for i in b] for b in batches], [[22], [24], [28]])
        self.assertEqual(len(list(pack_batches(items, budget=10_000))), 1)

    def test_fit_transcript_truncates_with_marker(self):
        tr = DayTranscript(date(2026, 8, 1), [f"10:{i:02d} 客：" + "字" * 50 for i in range(20)])
        fitted = fit_transcript(tr, 200)
        self.assertLess(len(fitted.lines), len(tr.lines))
        self.assertIn("已截断", fitted.lines[-1])


class SummaryInputTests(unittest.TestCase):
    def example(self):
        stats = [stat(d) for d in ("2026-07-22", "2026-07-24", "2026-07-28", "2026-07-30",
                                   "2026-08-02", "2026-08-06", "2026-08-11", "2026-08-14", "2026-08-18")]
        stats[0].session_count = 2
        notes = [note("2026-07-22", points=["担心继续下跌"]),
                 note("2026-07-24", topics=["财富传承"]),
                 note("2026-07-28", topics=["他行理财收益对比"]),
                 note("2026-07-30", status="no_business", topics=()),
                 note("2026-08-02", pending=["整理持仓收益与回撤情况"]),
                 note("2026-08-06", facts=["客户已自行完成部分资金配置"]),
                 note("2026-08-11", status="no_business", topics=()),
                 note("2026-08-14", facts=["已与律师完成传承安排"]),
                 note("2026-08-18", points=["询问持仓受影响多少"], actions=["告知整体回撤约1.8%"])]
        events = [event(1, "2026-07-22", "NEED_RISK_AVERSE", "open"), event(2, "2026-07-24", "NEED_LEGACY_ANXIETY", "open"),
                  event(3, "2026-07-28", "NEED_YIELD", "open"),
                  event(4, "2026-08-02", "NEED_RISK_AVERSE", "open"),
                  event(5, "2026-08-06", "NEED_YIELD", "closed", "收益那个事我不纠结了"),
                  event(6, "2026-08-14", "NEED_LEGACY_ANXIETY", "closed", "传承那块上周已经跟律师办妥了"),
                  event(7, "2026-08-14", "NEED_RISK_AVERSE", "open"), event(8, "2026-08-18", "NEED_RISK_AVERSE", "open")]
        return build_summary_input(date(2026, 7, 21), date(2026, 8, 19), stats, notes, events,
                                   30, 2500, "deepseek-chat")

    def test_stats_signals_and_pending_lines(self):
        inp = self.example()
        self.assertEqual(inp.stats["session_count"], 10)
        self.assertEqual((inp.business_days, inp.incomplete_days), (7, 0))
        # 统计仍计算并保存，但不再送给模型
        self.assertEqual((inp.stats["customer_initiated"], inp.stats["last_contact"]), (9, "2026-08-18"))
        self.assertNotIn("沟通统计", inp.user_prompt)
        self.assertNotIn("10次", inp.user_prompt)
        self.assertIn("深层需求：\n- 风险敏感型：多次出现（4天）；7月下旬起，最近8月中旬；依据：“原话”", inp.user_prompt)
        self.assertIn("- 传承焦虑型：已关闭；7月下旬出现 → 8月中旬关闭，依据：“传承那块上周已经跟律师办妥了”",
                      inp.user_prompt)
        self.assertIn("- 8月上旬：整理持仓收益与回撤情况", inp.user_prompt)
        self.assertIn("08-18｜话题：基金回撤｜客户：询问持仓受影响多少｜经理：告知整体回撤约1.8%", inp.user_prompt)
        self.assertNotIn("07-30", inp.user_prompt.split("【每日笔记】")[1])

    def test_input_fingerprint_is_stable_and_tracks_content(self):
        a, b = self.example(), self.example()
        self.assertEqual(a.input_fp, b.input_fp)
        stats = [stat("2026-08-01")]
        c = build_summary_input(date(2026, 7, 21), date(2026, 8, 19), stats, [note("2026-08-01")], [],
                                30, 2500, "deepseek-chat")
        d = build_summary_input(date(2026, 7, 21), date(2026, 8, 19), stats,
                                [note("2026-08-01", topics=["大额存单"])], [], 30, 2500, "deepseek-chat")
        self.assertNotEqual(c.input_fp, d.input_fp)

    def test_missing_or_failed_notes_count_as_incomplete(self):
        stats = [stat("2026-08-01"), stat("2026-08-02"), stat("2026-08-03")]
        inp = build_summary_input(date(2026, 7, 21), date(2026, 8, 19), stats,
                                  [note("2026-08-01"), note("2026-08-02", status="failed")], [],
                                  30, 2500, "m")
        self.assertEqual((inp.business_days, inp.incomplete_days), (1, 2))

    def test_timeline_closed_without_open_in_window(self):
        timeline = build_signal_timeline([event(1, "2026-08-06", "SVC_COMPETITOR", "closed", "不比了")])
        self.assertEqual(signal_lines(timeline), ["经营信号：", "- 他行比较：已关闭；8月上旬关闭，依据：“不比了”"])

    def test_timeline_groups_topics_and_confirms_needs_by_days(self):
        timeline = build_signal_timeline([
            event(1, "2026-08-02", "INV_FUND", "open", "基金怎么一直跌"),
            event(2, "2026-08-03", "NEED_CONTROL", "open", "我自己来看"),
            event(3, "2026-08-05", "RISK_AVERSE", "open", "旧代码"),   # 代码表调整前的旧代码
        ])
        self.assertEqual([t["code"] for t in timeline], ["INV_FUND", "NEED_CONTROL"])
        self.assertEqual(signal_lines(timeline), [
            "关注话题：", "- 关注基金（投资理财）：8月上旬起，共1天，最近8月上旬；依据：“基金怎么一直跌”",
            "深层需求：", "- 控制权偏好：仅出现1天；8月上旬起，最近8月上旬；依据：“我自己来看”"])

    def test_note_lines_compress_then_drop_low_priority_days(self):
        notes = [note(f"2026-08-{d:02d}", points=["客户诉求" * 10]) for d in range(1, 11)]
        lines = fit_note_lines(notes, {date(2026, 8, 10)}, budget=120)
        self.assertTrue(any(l.startswith("08-10") and "客户：" in l for l in lines))
        self.assertLess(len(lines), 10)

    def test_template_text(self):
        inp = build_summary_input(date(2026, 7, 21), date(2026, 8, 19), [stat("2026-08-01", 3)],
                                  [note("2026-08-01", status="no_business")], [], 30, 2500, "m")
        self.assertEqual(template_summary(inp, 30), "近1个月与客户的沟通以日常问候为主，暂未涉及业务诉求。")


class ValidateSummaryTests(unittest.TestCase):
    def setUp(self):
        self.inp = SummaryInputTests().example()
        self.text = ("近1个月客户的对话主要集中在基金回撤与稳健配置，涉及投资理财与人生规划；客户对回撤较为敏感，多次主动询问下行保护。"
                     "8月中旬客户明确提及传承安排已办妥，相关焦虑信号已关闭。建议市场波动时主动同步持仓表现，"
                     "并确认此前承诺的持仓报告是否已发送。")

    def test_valid_summary(self):
        summary, topics, suggestions = validate_summary(
            {"summary": self.text, "main_topics": ["基金回撤", "稳健配置", "财富传承", "多余"],
             "suggestions": ["同步持仓"]}, self.inp)
        self.assertEqual(summary, self.text)
        self.assertEqual(topics, ["基金回撤", "稳健配置", "财富传承"])

    def test_contact_count_rejected(self):
        for text in ("近1个月与客户沟通10次，", "客户主动发起9次，", "近1个月沟通十余次，"):
            with self.assertRaisesRegex(ValueError, "沟通次数"):
                validate_summary({"summary": text + self.text}, self.inp)
        validate_summary({"summary": self.text.replace("多次主动", "曾2次主动")}, self.inp)  # 非沟通次数的“次”不受影响

    def test_numbers_must_come_from_input(self):
        validate_summary({"summary": self.text + "整体回撤约1.8%。"}, self.inp)
        with self.assertRaisesRegex(ValueError, "数字"):
            validate_summary({"summary": self.text + "预计年化收益3.5%。"}, self.inp)

    def test_length_and_pii(self):
        with self.assertRaisesRegex(ValueError, "少于"):
            validate_summary({"summary": "近1个月客户关注基金回撤。"}, self.inp)
        with self.assertRaisesRegex(ValueError, "压缩到 300 字以内"):
            validate_summary({"summary": self.text * 4}, self.inp)
        longer = self.text + "建议后续结合客户对回撤的敏感，在市场波动时主动同步持仓表现。" * 9
        self.assertGreater(len(longer), 360)
        validate_summary({"summary": longer}, self.inp, strict_length=False)  # 重试时略长可接受
        with self.assertRaisesRegex(ValueError, "敏感"):
            validate_summary({"summary": self.text + "客户手机13812345678。"}, self.inp)


class ApiValidationTests(unittest.TestCase):
    def setUp(self):
        from app.api.main import app
        self.client = TestClient(app)

    def test_body_validation_returns_400(self):
        r = self.client.post("/contacts/summary/submit", json={"end_date": "2026-08-19"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("manager_id", r.json()["detail"])

    def test_future_date_rejected(self):
        r = self.client.post("/contacts/summary/submit", json={"manager_id": "m", "end_date": "2999-01-01"})
        self.assertEqual(r.status_code, 400)

    def test_unknown_signal_rejected(self):
        r = self.client.post("/contacts/signal/list", json={"manager_id": "m", "code": "NOPE"})
        self.assertEqual(r.status_code, 400)

    def test_note_list_range_limit(self):
        r = self.client.post("/contacts/note/list", json={
            "manager_id": "m", "customer_id": "c", "start_date": "2026-06-01", "end_date": "2026-08-01"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()


class FakeLimiter:
    def acquire(self, tokens, priority="low", call_id=None, stop=None):
        return call_id

    def reconcile(self, call_id, tokens):
        pass

    def cooldown(self, seconds):
        pass


class LlmFailFastTests(unittest.TestCase):
    def test_auth_error_stops_later_calls(self):
        import httpx
        from unittest.mock import patch
        from openai import AuthenticationError
        from app.contacts.llm import ContactLlm, LlmCallError

        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            response = httpx.Response(401, request=httpx.Request("POST", "http://llm.test"))
            raise AuthenticationError("invalid api key", response=response, body=None)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        llm = ContactLlm(limiter=FakeLimiter(), client=client)
        with patch.object(ContactLlm, "_save"):
            for _ in range(3):
                with self.assertRaises(LlmCallError):
                    llm.chat_json(purpose="note", system="s", user="u", prompt_version="v", max_tokens=10)
        self.assertEqual(len(calls), 1)
        self.assertIn("401", llm.fatal_error)


class SummarizePairTests(unittest.TestCase):
    def test_no_usable_notes_with_unanalyzed_days_is_failure(self):
        from unittest.mock import patch
        from app.contacts import summary

        records = ([stat("2026-08-01"), stat("2026-08-02")],
                   [note("2026-08-01", status="failed"), note("2026-08-02", status="no_business")], [])
        with patch.object(summary, "load_pair_records", return_value=records), \
             patch.object(summary, "_save_summary") as save:
            outcome = summary.summarize_pair("m", "c", date(2026, 8, 19))
        self.assertEqual(outcome.kind, "failed")
        save.assert_not_called()


class LlmTruncationTests(unittest.TestCase):
    def test_truncated_output_retries_with_larger_budget(self):
        from unittest.mock import patch
        from app.contacts.llm import ContactLlm

        budgets = []

        def create(**kwargs):
            budgets.append(kwargs["max_tokens"])
            truncated = len(budgets) == 1
            choice = SimpleNamespace(finish_reason="length" if truncated else "stop",
                                     message=SimpleNamespace(content='{"days": [' if truncated else '{"days": []}'))
            usage = SimpleNamespace(prompt_tokens=100, completion_tokens=kwargs["max_tokens"])
            return SimpleNamespace(usage=usage, choices=[choice])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        llm = ContactLlm(limiter=FakeLimiter(), client=client)
        with patch.object(ContactLlm, "_save") as save:
            res = llm.chat_json(purpose="note", system="s", user="u", prompt_version="v", max_tokens=900)
        self.assertEqual(res.data, {"days": []})
        self.assertEqual(budgets, [900, 1800])
        self.assertEqual(res.prompt_tokens, 200)
        record = save.call_args.args[0]
        self.assertEqual(record["completion_tokens"], 2700)


class SummaryLeakTests(unittest.TestCase):
    def test_internal_terms_rejected(self):
        inp = SummaryInputTests().example()
        text = ("近1个月客户的对话主要集中在基金回撤与稳健配置，涉及投资理财与人生规划；客户对回撤较为敏感，"
                "多次主动询问下行保护。8月中旬客户明确提及传承安排已办妥，相关焦虑信号已关闭。"
                "建议市场波动时主动同步持仓表现。")
        with self.assertRaisesRegex(ValueError, "内部用语"):
            validate_summary({"summary": text + "持仓报告是否发送系统未核实。"}, inp)
        with self.assertRaisesRegex(ValueError, "NEED_RISK_AVERSE"):
            validate_summary({"summary": text + "客户属于NEED_RISK_AVERSE。"}, inp)
        with self.assertRaisesRegex(ValueError, "已确认"):
            validate_summary({"summary": text + "风险敏感需求已确认。"}, inp)


class LlmStopTests(unittest.TestCase):
    def test_stopped_worker_makes_no_new_calls(self):
        import threading
        from app.contacts.llm import ContactLlm

        calls = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: calls.append(kw))))
        stop = threading.Event()
        stop.set()
        llm = ContactLlm(limiter=FakeLimiter(), client=client, stop=stop)
        with self.assertRaises(InterruptedError):
            llm.chat_json(purpose="note", system="s", user="u", prompt_version="v", max_tokens=10)
        self.assertEqual(calls, [])


class SummaryRetryFeedbackTests(unittest.TestCase):
    def test_retry_tells_model_why_output_failed(self):
        from unittest.mock import patch
        from app.contacts import summary
        from app.contacts.llm import LlmResult

        records = ([stat("2026-08-01", sessions=2)], [note("2026-08-01", points=["询问大额存单利率"])], [])
        good = "近1个月客户关注存款类产品，曾询问大额存单利率及起存条件。建议结合客户资金期限与流动性安排，介绍合适的存款产品。"
        prompts = []

        def chat_json(**kw):
            prompts.append(kw["user"])
            text = "近1个月与客户沟通2次，" + good[4:] if len(prompts) == 1 else good
            return LlmResult(data={"summary": text}, call_id=f"c{len(prompts)}", prompt_tokens=100)

        llm = SimpleNamespace(chat_json=chat_json)
        session = SimpleNamespace(execute=lambda *a, **k: SimpleNamespace(first=lambda: None), close=lambda: None)
        with patch.object(summary, "load_pair_records", return_value=records), \
             patch.object(summary, "get_session", return_value=session), \
             patch.object(summary, "_save_summary") as save:
            outcome = summary.summarize_pair("m", "c", date(2026, 8, 19), llm=llm)
        self.assertEqual(outcome.kind, "success")
        self.assertNotIn("修改要求", prompts[0])
        self.assertIn("【修改要求】上一次输出未通过检查：出现沟通次数：沟通2次，请删去次数", prompts[1])
        self.assertEqual(save.call_args.args[3]["summary"], good)


class PromptRuleTests(unittest.TestCase):
    def test_semantic_map_gives_one_tag_per_sentence(self):
        from app.contacts.prompts import SIGNALS, _SEMANTIC_MAP

        for line in _SEMANTIC_MAP.splitlines():
            codes = line.split("→")[1].strip().split(" / ")
            self.assertEqual(len(codes), 1, line)
            self.assertIn(codes[0], SIGNALS, line)

    def test_note_prompt_has_manager_mention_counter_example(self):
        from app.contacts.prompts import NOTE_SYSTEM

        self.assertIn("不标 INV_FAMILY_TRUST、INV_INS_TRUST", NOTE_SYSTEM)


class SchemaTests(unittest.TestCase):
    def test_signal_index_covers_cross_customer_query(self):
        from app.models import ContactSignalEvent

        indexes = {ix.name: [c.name for c in ix.columns] for ix in ContactSignalEvent.__table__.indexes}
        self.assertEqual(indexes["idx_cse_mgr_code_date"], ["manager_id", "signal_code", "event_date"])
        self.assertNotIn("idx_cse_mgr_code", indexes)
