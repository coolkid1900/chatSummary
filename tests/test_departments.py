import json
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import httpx
import numpy as np
from tenacity import wait_none

from app.config import Settings
from app.departments import hotspot, roster
from app.pipeline.cluster import ClusterResult, _accumulate, merge_similar_topics
from app.pipeline.vector_store import LocalVectorStore, Record
from tests.test_incremental import MemoryStore, batch, execute, settings as cluster_settings


def settings(**kwargs):
    defaults = dict(_env_file=None, top_n=20, repr_pool_size=6, nr_repr_docs=3, llm_concurrency=2,
                    department_roster_url="http://roster/managers", department_roster_page_size=2,
                    department_min_topic_sessions=2, department_min_topic_share=0,
                    department_drift_sim=0.9, department_split_min_sessions=10,
                    department_split_max_k=3, cluster_random_seed=42)
    defaults.update(kwargs)
    with patch.dict("os.environ", {}, clear=True):
        return Settings(**defaults)


class AssignmentTests(unittest.TestCase):
    def test_incremental_records_every_assigned_session(self):
        rng = np.random.default_rng(0)
        centers = np.eye(3, dtype=np.float32)
        source, start = [], 0
        for c in centers:
            vectors = (c + rng.normal(0, .02, (25, 3))).astype(np.float32)
            source.append(batch(vectors, start=start))
            start += 25
        result = execute(MemoryStore(source), cluster_settings())
        sessions, topics = result.final_assignments()
        self.assertEqual(len(sessions), 75)
        self.assertEqual(Counter(topics), Counter(result._seen))
        for topic, centroid in result.centroids().items():
            rows = [i for i, t in enumerate(topics) if t == topic]
            members = np.concatenate([b["vectors"] for b in source])[rows]
            np.testing.assert_allclose(centroid, members.mean(axis=0), rtol=1e-5)

    def test_merge_maps_sessions_to_canonical_topic(self):
        result = ClusterResult(None, {}, {})
        vectors = np.tile([1., 0.], (12, 1)).astype(np.float32)
        _accumulate(result, batch(vectors), [0] * 5 + [1] * 7, 6)
        with patch("app.pipeline.cluster.get_settings", return_value=cluster_settings()):
            merge_similar_topics(result, .85)
        sessions, topics = result.final_assignments()
        self.assertEqual(sessions, [str(i) for i in range(12)])
        self.assertEqual(set(topics), {1})  # 合并后保留热度更高的编号


class ArtifactTests(unittest.TestCase):
    def test_round_trip_staff_filter_and_shards_ignore_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("app.pipeline.vector_store.get_settings",
                       return_value=settings(vector_store_dir=tmp, cluster_batch_size=10)):
                store = LocalVectorStore()
                store.write_shard("2026-10-27", [Record(session_id="m1:c1:0", text="t", tokens="贷款",
                                                        msg_count=2, customer="c1", vector=[1., 0.])], 0)
                self.assertIsNone(store.read_topic_manifest("2026-10-27"))
                store.write_topic_artifacts("2026-10-27", "run1", ["m1:c1:0", "m2:c2:0"], [3, 5],
                                            {3: np.array([1., 0.]), 5: np.array([0., 1.])})
                self.assertEqual(store.read_topic_manifest("2026-10-27")["run_id"], "run1")
                self.assertEqual(store.read_topic_assignments("2026-10-27"), {"m1:c1:0": 3, "m2:c2:0": 5})
                self.assertEqual(store.read_topic_assignments("2026-10-27", {"m2"}), {"m2:c2:0": 5})
                np.testing.assert_allclose(store.read_topic_centroids("2026-10-27")[5], [0., 1.])
                self.assertEqual(store._parquet_count("2026-10-27"), 1)
                self.assertEqual(sum(len(b["session_id"]) for b in store.iter_batches("2026-10-27")), 1)


class RosterTests(unittest.TestCase):
    def fetch(self, handler, department="D1"):
        transport = httpx.MockTransport(handler)
        real = httpx.Client
        with patch("app.departments.roster.get_settings", return_value=settings()), \
                patch("app.departments.roster.httpx.Client",
                      side_effect=lambda **kw: real(transport=transport, **kw)), \
                patch.object(roster._fetch_page.retry, "wait", wait_none()):
            return roster.fetch_managers(department)

    def test_pages_until_total_and_dedupes(self):
        pages = {1: ["m1", "m2"], 2: ["m2", "m3"], 3: ["m4"]}
        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append(body)
            return httpx.Response(200, json={"total": 5, "manager_ids": pages[body["page"]]})

        self.assertEqual(self.fetch(handler), ["m1", "m2", "m3", "m4"])
        self.assertEqual([b["page"] for b in seen], [1, 2, 3])
        self.assertEqual(seen[0], {"department_id": "D1", "page": 1, "page_size": 2})

    def test_server_error_retried_then_fails_whole_roster(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(503)

        with self.assertRaises(roster.RosterError):
            self.fetch(handler)
        self.assertEqual(len(calls), 3)

    def test_client_error_and_bad_format_not_retried(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(404) if len(calls) == 1 else httpx.Response(200, json={"ids": []})

        with self.assertRaises(roster.RosterError):
            self.fetch(handler)
        with self.assertRaises(roster.RosterError):
            self.fetch(handler)
        self.assertEqual(len(calls), 2)


class FakeStore:
    def __init__(self, rows, assignments, centroids):
        self.rows, self.assignments, self.centroids = rows, assignments, centroids
        self.scans = 0

    def read_topic_assignments(self, date_str, staff=None):
        return {s: t for s, t in self.assignments.items() if staff is None or s.split(":")[0] in staff}

    def read_topic_centroids(self, date_str):
        return self.centroids

    def iter_batches(self, date_str):
        self.scans += 1
        for i in range(0, len(self.rows), 7):
            chunk = self.rows[i:i + 7]
            yield dict(session_id=[r[0] for r in chunk], text=[r[1] for r in chunk],
                       tokens=[r[2] for r in chunk], msg_count=[r[3] for r in chunk],
                       customer=[r[4] for r in chunk],
                       vectors=np.asarray([r[5] for r in chunk], dtype=np.float32))


class FakeSummarizer:
    def __init__(self, fail_topics=()):
        self.failures = 0
        self.fail_topics = set(fail_topics)
        self.settings = settings()

    def _pick_docs(self, topic):
        return [text for text, _ in topic.repr_pool]

    def describe(self, topic):
        docs = self._pick_docs(topic)[:3]
        if topic.topic_id in self.fail_topics:
            self.failures += 1
            return "（概括失败，回退）", [], docs, False
        return f"部门意图{topic.topic_id}", [w for w, _ in topic.hot_words[:2]], docs, True


class RunDepartmentsTests(unittest.TestCase):
    def dataset(self):
        rng = np.random.default_rng(1)
        rows, assignments = [], {}

        def add(manager, topic, direction, words, n):
            for _ in range(n):
                sid = f"{manager}:c{len(rows) % 9}:{len(rows)}"
                vector = np.asarray(direction, dtype=np.float32) + rng.normal(0, .02, 4)
                rows.append((sid, f"{words}{len(rows)}", words, 2, f"c{len(rows) % 9}", vector))
                assignments[sid] = topic

        # D1 在全公司主题 0 里聊的是两件别的事（被吞没），D2 与主题 0 主体一致。
        add("m1", 0, [.3, 1, 0, 0], "社保卡 激活", 15)
        add("m1", 0, [.3, 0, 1, 0], "养老金 领取", 12)
        add("m2", 0, [1, 0, 0, 0], "信用卡 额度", 20)
        add("m3", 1, [0, 0, 0, 1], "理财 赎回", 6)
        add("m1", 1, [0, 0, 0, 1], "理财 赎回", 3)
        add("m9", 1, [0, 0, 0, 1], "理财 赎回", 5)  # 不在任何名单内
        centroids = {0: np.array([1., .05, .05, 0]), 1: np.array([0., 0, 0, 1])}
        return FakeStore(rows, assignments, centroids)

    def run_departments(self, store, rosters, summarizer, config=None):
        config = config or settings()
        saved = {}

        def fetch(dept):
            if dept not in rosters:
                raise roster.RosterError(f"部门 {dept} 不存在")
            return rosters[dept]

        catalog = {0: {"run_id": "r", "customer_intent": "全公司意图0", "business_words": ["信用卡"]},
                   1: {"run_id": "r", "customer_intent": "全公司意图1", "business_words": ["理财"]}}
        with patch.object(hotspot, "get_settings", return_value=config), \
                patch("app.pipeline.hotwords.get_settings", return_value=config), \
                patch.object(hotspot, "check_ready", return_value=catalog), \
                patch.object(hotspot, "fetch_managers", side_effect=fetch), \
                patch.object(hotspot, "get_vector_store", return_value=store), \
                patch.object(hotspot, "IntentSummarizer", return_value=summarizer), \
                patch.object(hotspot, "enrich_with_trend", return_value=0), \
                patch.object(hotspot, "save_results",
                             side_effect=lambda d, results, scope, run_id: saved.__setitem__(scope[1], results)):
            report = hotspot.run_departments("2026-10-27", ["D1", "D2", "D404", "D1"], run_id="dh_test")
        return report, saved

    def test_slices_by_roster_and_splits_swallowed_topic(self):
        store = self.dataset()
        report, saved = self.run_departments(store, {"D1": ["m1"], "D2": ["m2", "m3"]}, FakeSummarizer())
        self.assertEqual(report["departments"]["D404"]["status"], "failed")
        self.assertNotIn("D404", saved)
        self.assertEqual(store.scans, 2)  # 一遍统计 + 一遍取被吞没主题的会话

        d1 = saved["D1"]
        split = [r for r in d1 if r["parent_topic_id"] == 0]
        self.assertEqual(len(split), 2)
        self.assertTrue(all(r["topic_id"] >= 2 for r in split))
        self.assertEqual(sorted(r["heat"] for r in split), [24, 30])
        words = {tuple(sorted(w for w, _ in r["hot_words"][:2])) for r in split}
        self.assertEqual(words, {tuple(sorted(("社保卡", "激活"))), tuple(sorted(("养老金", "领取")))})
        self.assertEqual(next(r for r in d1 if r["topic_id"] == 1)["heat"], 6)
        self.assertEqual([r["rank"] for r in d1], [1, 2, 3])

        d2 = {r["topic_id"]: r for r in saved["D2"]}
        self.assertEqual(set(d2), {0, 1})  # 与全公司主题一致，不细分
        self.assertIsNone(d2[0]["parent_topic_id"])
        self.assertEqual((d2[0]["heat"], d2[1]["heat"]), (40, 12))
        self.assertTrue(all(r["intent_source"] == "department" for r in d1 + saved["D2"]))
        self.assertEqual(report["departments"]["D1"]["split_topics"], 2)

    def test_intent_falls_back_to_catalog_and_threshold_filters(self):
        store = self.dataset()
        config = settings(department_min_topic_sessions=7, department_drift_sim=0)
        report, saved = self.run_departments(store, {"D1": ["m1"], "D2": ["m2", "m3"]},
                                             FakeSummarizer(fail_topics={0}), config)
        self.assertEqual(store.scans, 1)  # 关闭细分后只扫描一遍
        d1 = saved["D1"]
        self.assertEqual([r["topic_id"] for r in d1], [0])  # 主题 1 只有 3 个会话，低于门槛
        self.assertEqual((d1[0]["customer_intent"], d1[0]["intent_source"]), ("全公司意图0", "global"))
        self.assertEqual(d1[0]["business_words"], ["信用卡"])
        self.assertEqual(report["departments"]["D1"]["llm_failures"], 1)

    def test_empty_department_clears_rows(self):
        report, saved = self.run_departments(self.dataset(), {"D1": ["nobody"], "D2": ["m2"]},
                                             FakeSummarizer())
        self.assertEqual(saved["D1"], [])
        self.assertEqual(report["departments"]["D1"]["status"], "empty")


class IntentTests(unittest.TestCase):
    def test_vendor_params_passed_and_failure_reported(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from app.pipeline.hotwords import TopicHot
        from app.pipeline.intent import IntentSummarizer

        content = '{"intent": "客户咨询提前还款", "business_words": ["提前还款", "臆造"]}'
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        config = settings(contact_llm_extra_body={"thinking": {"type": "disabled"}})
        with patch("app.pipeline.intent.get_settings", return_value=config), \
                patch("app.pipeline.intent.TokenBucket"):
            summarizer = IntentSummarizer()
        summarizer._client = client
        topic = TopicHot(topic_id=1, heat=3, customer_count=1, hot_words=[("提前还款", .5), ("违约金", .2)],
                         repr_pool=[("我想提前还款", np.array([1., 0.]))])
        intent, words, docs, ok = summarizer.describe(topic)
        self.assertEqual((intent, words, docs, ok), ("客户咨询提前还款", ["提前还款"], ["我想提前还款"], True))
        self.assertEqual(client.chat.completions.create.call_args.kwargs["extra_body"],
                         {"thinking": {"type": "disabled"}})

        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
        with patch.object(IntentSummarizer._call_llm.retry, "wait", wait_none()):
            self.assertFalse(summarizer.describe(topic)[3])
        self.assertEqual(summarizer.failures, 1)


class ApiTests(unittest.TestCase):
    def test_department_ids_deduped_and_limited(self):
        from app.api.departments import HotspotRunReq

        with patch("app.api.departments.get_settings", return_value=settings(department_max_per_request=2)):
            self.assertEqual(HotspotRunReq(department_ids=[" D1", "D1", "D2"]).department_ids, ["D1", "D2"])
            with self.assertRaises(ValueError):
                HotspotRunReq(department_ids=["D1", "D2", "D3"])

    def test_mock_roster_pages(self):
        from fastapi.testclient import TestClient
        from app.api.main import app

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"D1": ["m1", "m2", "m3"]}, f)
        client = TestClient(app)
        with patch("app.api.departments.get_settings",
                   return_value=settings(department_mock_enabled=True, department_mock_file=f.name)):
            resp = client.post("/mock/departments/managers", json={"department_id": "D1", "page": 2, "page_size": 2})
            self.assertEqual(resp.json(), {"department_id": "D1", "page": 2, "page_size": 2,
                                           "total": 3, "manager_ids": ["m3"]})
        with patch("app.api.departments.get_settings", return_value=settings()):
            self.assertEqual(client.post("/mock/departments/managers",
                                         json={"department_id": "D1"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
