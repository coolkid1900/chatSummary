import io
import builtins
import importlib.util
import sys
import tempfile
import unittest
import weakref
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app.config import Settings
from app.pipeline.cluster import ClusterResult, _accumulate, _topic_words, group_topic_centers, merge_similar_topics, run_clustering
from app.pipeline.hotwords import rank_topics
from app.pipeline import incremental
from app.pipeline.incremental import _fit_pca, _predict, run_incremental
from app.pipeline.reduction import Float32IncrementalPCA
from app.pipeline.vector_store import LocalVectorStore, S3VectorStore, Record, _arrow_vectors, _to_table


def settings(**kwargs):
    defaults = dict(_env_file=None, cluster_backend="incremental", n_components=2,
                    n_clusters=3, cluster_batch_size=11, repr_pool_size=6,
                    incremental_epochs=3, incremental_init_sample_size=30,
                    topic_merge_sim=0)
    defaults.update(kwargs)
    # Compose 会注入本地调参配置，单测必须独立于部署环境。
    with patch.dict("os.environ", {}, clear=True):
        return Settings(**defaults)


def batch(vectors, start=0, tokens=None):
    n = len(vectors)
    return dict(vectors=vectors, session_id=[str(start+i) for i in range(n)],
                text=[f"会话{start+i}" for i in range(n)],
                tokens=tokens if tokens is not None else ["贷款 还款"] * n,
                msg_count=[i % 3 + 1 for i in range(n)],
                customer=[f"客户{(start+i) % 7}" for i in range(n)])


class MemoryStore:
    def __init__(self, batches):
        self.batches = batches
        self.vector_scans = 0
        self.full_scans = 0

    def iter_vector_batches(self, date_str):
        self.vector_scans += 1
        for value in self.batches:
            yield value["vectors"].copy()

    def iter_batches(self, date_str):
        self.full_scans += 1
        yield from self.batches


def execute(store, config):
    result = ClusterResult(None, {}, {})
    result._rng.seed(config.cluster_random_seed)
    with patch("app.pipeline.cluster.get_settings", return_value=config):
        return run_incremental(store, "2026-09-15", config, result)


class TelemetryTests(unittest.TestCase):
    def test_missing_resource_does_not_prevent_import_or_clustering(self):
        name = "_incremental_without_resource"
        spec = importlib.util.spec_from_file_location(name, incremental.__file__)
        module = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__

        def without_resource(module_name, *args, **kwargs):
            if module_name == "resource":
                raise ModuleNotFoundError("No module named 'resource'", name="resource")
            return original_import(module_name, *args, **kwargs)

        with patch.dict(sys.modules, {name: module}), \
             patch("builtins.__import__", side_effect=without_resource):
            spec.loader.exec_module(module)
        source = MemoryStore([batch(np.random.default_rng(8).normal(size=(12, 4)).astype(np.float32))])
        result = ClusterResult(None, {}, {})
        with patch("app.pipeline.cluster.get_settings", return_value=settings()):
            module.run_incremental(source, "date", settings(), result)
        self.assertEqual(result.metrics["assigned_sessions"], 12)
        for stage in result.metrics["stages"].values():
            self.assertIsNone(stage["process_peak_rss_mib"])
            self.assertGreaterEqual(stage["seconds"], 0)

    def test_peak_units_on_linux_and_macos(self):
        for platform, raw_peak in [("linux", 128 * 1024), ("darwin", 128 * 1024**2)]:
            reader = Mock()
            reader.getrusage.return_value.ru_maxrss = raw_peak
            with patch.object(incremental, "resource", reader), \
                 patch.object(incremental.sys, "platform", platform):
                self.assertEqual(incremental._process_peak_rss_mib(), 128.)

    def test_failed_metrics_read_does_not_mask_pipeline_exception(self):
        reader = Mock()
        reader.getrusage.side_effect = OSError("metrics unavailable")
        result = ClusterResult(None, {}, {})
        with patch.object(incremental, "resource", reader):
            with self.assertRaisesRegex(ValueError, "pipeline failed"):
                with incremental._stage(result, "failed"):
                    raise ValueError("pipeline failed")
        self.assertIsNone(result.metrics["stages"]["failed"]["process_peak_rss_mib"])


class IncrementalTests(unittest.TestCase):
    def dataset(self):
        rng = np.random.default_rng(123)
        centers = np.array([[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0]], dtype=np.float32)
        vectors = np.concatenate([center + rng.normal(0, .04, (25, 4)) for center in centers]).astype(np.float32)
        words = ["理财 收益"] * 25 + ["贷款 还款"] * 25 + ["信用卡 分期"] * 25
        return [batch(vectors[i:i+11], i, words[i:i+11]) for i in range(0, len(vectors), 11)]

    def test_final_labels_words_heat_and_customers_agree(self):
        source = self.dataset()
        store = MemoryStore(source)
        observed = []
        original = _accumulate

        def capture(result, value, topics, cap):
            observed.extend(zip(value["tokens"], value["msg_count"], value["customer"], topics))
            original(result, value, topics, cap)

        with patch("app.pipeline.cluster._accumulate", side_effect=capture):
            result = execute(store, settings())
        self.assertEqual((store.vector_scans, store.full_scans), (2, 1))
        self.assertEqual(result.metrics["n_sessions"], 75)
        self.assertEqual(sum(result._seen.values()), 75)
        self.assertEqual(len(result.metrics["epoch_sample_mean_squared_distance"]), 3)
        for topic in result.heat_by_topic:
            rows = [row for row in observed if row[3] == topic]
            expected = Counter(word for row in rows for word in row[0].split())
            self.assertEqual(result.term_counts[topic], expected)
            self.assertEqual(result.heat_by_topic[topic], sum(row[1] for row in rows))
            self.assertEqual(result.customers_by_topic[topic], {row[2] for row in rows})
            self.assertTrue(set(w for w, _ in result.hot_words_by_topic[topic]) <= set(expected))
        # 三个明显分离的真实簇应各自对应一个主题，不能只检验数组尺寸。
        self.assertEqual(result.n_topics, 3)
        for words in {row[0] for row in observed}:
            self.assertEqual(len({row[3] for row in observed if row[0] == words}), 1)
        with patch("app.pipeline.hotwords.get_settings", return_value=settings()):
            self.assertEqual(len(rank_topics(result)), 3)

    def test_pca_frozen_before_kmeans_and_assignment(self):
        events = []
        from sklearn.cluster import MiniBatchKMeans
        original_pca = Float32IncrementalPCA.partial_fit
        original_kmeans = MiniBatchKMeans.partial_fit

        def pca(model, *args, **kwargs):
            events.append("pca")
            return original_pca(model, *args, **kwargs)

        def kmeans(model, *args, **kwargs):
            events.append("kmeans")
            return original_kmeans(model, *args, **kwargs)

        with patch.object(Float32IncrementalPCA, "partial_fit", pca), patch.object(MiniBatchKMeans, "partial_fit", kmeans):
            execute(MemoryStore(self.dataset()), settings())
        self.assertNotIn("pca", events[events.index("kmeans"):])

    def test_pca_does_not_depend_on_physical_shard_boundaries(self):
        # 同一矩阵按 1/29 片输入，训练批大小相同，得到相同的投影。
        vectors = np.random.default_rng(97).normal(size=(299, 40)).astype(np.float32)
        config = settings(n_components=32, cluster_batch_size=128)
        one = MemoryStore([batch(vectors)])
        many = MemoryStore([batch(part) for part in np.array_split(vectors, 29)])
        left, n, digest = _fit_pca(one, "date", config)
        right, m, other = _fit_pca(many, "date", config)
        self.assertEqual((n, digest), (m, other))
        np.testing.assert_allclose(left.transform(vectors), right.transform(vectors), atol=1e-5)

    def test_small_dataset_pca_fits_once_without_artificial_tail(self):
        vectors = np.random.default_rng(3).normal(size=(100, 40)).astype(np.float32)
        calls = []
        original = Float32IncrementalPCA.partial_fit

        def capture(model, values, *args, **kwargs):
            calls.append(len(values))
            return original(model, values, *args, **kwargs)

        with patch.object(Float32IncrementalPCA, "partial_fit", capture):
            _fit_pca(MemoryStore([batch(p) for p in np.array_split(vectors, 10)]),
                     "date", settings(n_components=32, cluster_batch_size=2048))
        self.assertEqual(calls, [100])

    def test_repeated_runs_are_identical(self):
        first = execute(MemoryStore(self.dataset()), settings(topic_merge_sim=.85))
        second = execute(MemoryStore(self.dataset()), settings(topic_merge_sim=.85))
        self.assertEqual(first.heat_by_topic, second.heat_by_topic)
        self.assertEqual(first.members_by_topic, second.members_by_topic)
        self.assertEqual(first.hot_words_by_topic, second.hot_words_by_topic)
        self.assertEqual({t: [x[0] for x in pool] for t, pool in first.repr_pool.items()},
                         {t: [x[0] for x in pool] for t, pool in second.repr_pool.items()})

    def test_tiny_data_and_one_row_tail(self):
        for n in (1, 2, 7, 12):
            with self.subTest(n=n):
                vectors = np.random.default_rng(2).normal(size=(n, 4)).astype(np.float32)
                source = [batch(vectors[i:i+3], i, [""] * len(vectors[i:i+3])) for i in range(0, n, 3)]
                result = execute(MemoryStore(source), settings(n_components=10, n_clusters=50))
                self.assertEqual(result.metrics["effective_n_components"], min(4, n))
                self.assertEqual(result.metrics["effective_n_clusters"], n)
                self.assertEqual(sum(result._seen.values()), n)
                self.assertTrue(all(not words for words in result.hot_words_by_topic.values()))

    def test_empty_store(self):
        result = execute(MemoryStore([]), settings())
        self.assertEqual(result.n_topics, 0)
        self.assertEqual(result.metrics["n_sessions"], 0)

    def test_nonfinite_vectors_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            execute(MemoryStore([batch(np.array([[1, np.nan]], dtype=np.float32))]), settings())

    def test_cache_cleanup_on_success_and_changed_final_input(self):
        class ChangedStore(MemoryStore):
            def iter_batches(self, date_str):
                yield from reversed(self.batches)

        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "new-cache"
            config = settings(incremental_cache_dir=str(cache))
            execute(MemoryStore(self.dataset()), config)
            self.assertEqual(list(cache.iterdir()), [])
            with self.assertRaisesRegex(ValueError, "顺序或向量内容变化"):
                execute(ChangedStore(self.dataset()), config)
            self.assertEqual(list(cache.iterdir()), [])

    def test_change_between_pca_and_cache_detected(self):
        class ChangedStore(MemoryStore):
            def iter_vector_batches(self, date_str):
                for vectors in super().iter_vector_batches(date_str):
                    yield vectors if self.vector_scans == 1 else vectors + .1
        with self.assertRaisesRegex(ValueError, "分片发生变化"):
            execute(ChangedStore(self.dataset()), settings())

    def test_missing_final_batch_detected_and_cache_removed(self):
        class MissingStore(MemoryStore):
            def iter_batches(self, date_str):
                yield from self.batches[:-1]
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "分片缺失"):
                execute(MissingStore(self.dataset()), settings(incremental_cache_dir=root))
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_rejection_reports_coverage_and_excluded_messages(self):
        source = self.dataset()
        result = execute(MemoryStore(source), settings(incremental_max_distance=1e-12))
        self.assertEqual(result.metrics["assigned_sessions"], 0)
        self.assertEqual(result.metrics["rejected_sessions"], 75)
        self.assertEqual(result.metrics["rejected_messages"], sum(sum(b["msg_count"]) for b in source))
        self.assertEqual(result.metrics["coverage"], 0)
        self.assertEqual(result.n_topics, 0)

    def test_distance_and_margin_gates(self):
        class Centers:
            def transform(self, values):
                return np.array([[.1, 2], [4, 5], [1, 1.01]])
        labels, _ = _predict(Centers(), None, max_distance=3, min_margin=.1)
        self.assertEqual(labels.tolist(), [0, -1, -1])
        labels, _ = _predict(Centers(), None)
        self.assertEqual(labels.tolist(), [0, 0, 0])


class AggregationTests(unittest.TestCase):
    def test_umap_merge_override_is_independent_and_zero_disables(self):
        angles = np.deg2rad([0, 35])
        vectors = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        source = MemoryStore([batch(vectors)])
        topic_model = Mock()
        topic_model.fit_transform.return_value = ([0, 1], None)
        for threshold, expected in [(.85, 2), (.80, 1), (0., 2), (None, 1)]:
            config = settings(cluster_backend="umap_hdbscan", topic_merge_sim=.80,
                              topic_merge_linkage="average", umap_topic_merge_linkage="single",
                              umap_topic_merge_sim=threshold)
            with patch("app.pipeline.cluster.get_settings", return_value=config), \
                 patch("app.pipeline.cluster.get_vector_store", return_value=source), \
                 patch("app.pipeline.cluster.build_vectorizer"), \
                 patch("app.pipeline.cluster.build_topic_model", return_value=topic_model):
                result = run_clustering("date")
            self.assertEqual(result.n_topics, expected)
            self.assertEqual(result.metrics["topic_merge_linkage"], "single")

    def test_representative_does_not_retain_batch(self):
        vectors = np.ones((2048, 1024), dtype=np.float32)
        reference = weakref.ref(vectors)
        result = ClusterResult(None, {}, {})
        result._add_repr(0, "贷款", vectors[0], 1)
        self.assertFalse(np.shares_memory(result.repr_pool[0][0][1], vectors))
        del vectors
        self.assertIsNone(reference())

    def test_complete_linkage_prevents_chain_merge(self):
        angles = np.deg2rad([0, 25, 50])
        vectors = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        result = ClusterResult(None, {}, {})
        _accumulate(result, batch(vectors), [0, 1, 2], 6)
        with patch("app.pipeline.cluster.get_settings", return_value=settings(topic_merge_linkage="complete")):
            merge_similar_topics(result, .85)
        self.assertEqual(result.n_topics, 2)
        self.assertFalse(any(0 in members and 2 in members for members in result.members_by_topic.values()))
        self.assertEqual(sum(result._seen.values()), 3)
        self.assertEqual(sum(result.heat_by_topic.values()), 6)

    def test_linkage_tradeoff_and_disabled_merge(self):
        angles = np.deg2rad([0, 25, 50])
        centers = np.column_stack((np.cos(angles), np.sin(angles)))
        for method, threshold, expected in [("single", .85, 1), ("average", .85, 2),
                                            ("average", .75, 1), ("complete", .75, 2)]:
            with self.subTest(method=method, threshold=threshold):
                self.assertEqual(len(set(group_topic_centers(centers, threshold, method))), expected)
        self.assertEqual(len(set(group_topic_centers(centers, 0, "single"))), 3)
        with self.assertRaises(ValueError):
            settings(topic_merge_linkage="unknown")

    def test_merge_uses_all_vectors_not_reservoir_and_covers_children(self):
        result = ClusterResult(None, {}, {})
        vectors = np.tile([1., 0.], (12, 1)).astype(np.float32)
        _accumulate(result, batch(vectors, tokens=["贷款"] * 6 + ["理财"] * 6), [0]*6 + [1]*6, 6)
        # 故意污染代表池向量，合并决策仍应采用全部成员的向量和。
        result.repr_pool[1] = [(text, np.array([-1., 0.])) for text, _ in result.repr_pool[1]]
        with patch("app.pipeline.cluster.get_settings", return_value=settings()):
            merge_similar_topics(result, .85)
        self.assertEqual(result.n_topics, 1)
        topic = next(iter(result.heat_by_topic))
        texts = {text for text, _ in result.repr_pool[topic]}
        self.assertTrue(texts & {f"会话{i}" for i in range(6)})
        self.assertTrue(texts & {f"会话{i}" for i in range(6, 12)})
        self.assertEqual(len(texts), 6)
        self.assertEqual(result.term_counts[topic], Counter(贷款=6, 理财=6))
        self.assertEqual(set(w for w, _ in result.hot_words_by_topic[topic]), {"贷款", "理财"})
        self.assertEqual(len(result.customers_by_topic[topic]), 7)

    def test_ctfidf_matches_manual_final_counts(self):
        words = _topic_words({0: Counter(贷款=3, 还款=1), 1: Counter(理财=2)})
        self.assertAlmostEqual(dict(words[0])["贷款"], .75 * np.log1p(3/3))
        self.assertAlmostEqual(dict(words[0])["还款"], .25 * np.log1p(3/1))
        self.assertAlmostEqual(dict(words[1])["理财"], np.log1p(3/2))
        self.assertEqual(_topic_words({0: Counter()}), {0: []})

    def test_zero_centroids_stay_separate(self):
        result = ClusterResult(None, {}, {})
        _accumulate(result, batch(np.zeros((2, 2), dtype=np.float32)), [0, 1], 6)
        with patch("app.pipeline.cluster.get_settings", return_value=settings()):
            merge_similar_topics(result, .85)
        self.assertEqual(result.n_topics, 2)


class ParquetIntegrationTests(unittest.TestCase):
    def test_s3_order_projection_and_body_cleanup(self):
        import pyarrow.parquet as pq

        config = settings(cluster_batch_size=2)
        payloads = {}
        for name, value in [("shard-b.parquet", 2), ("shard-a.parquet", 1)]:
            buffer = io.BytesIO()
            records = [Record(session_id=f"{value}-{i}", text="测试", tokens="测试",
                              msg_count=1, customer="客户", vector=[value, i]) for i in range(3)]
            pq.write_table(_to_table(records), buffer)
            payloads[name] = buffer.getvalue()
        store = object.__new__(S3VectorStore)
        store.settings, store.bucket, store.client = config, "test", Mock()
        store.client.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": name} for name in payloads]}]
        streams = []

        def get_object(**kwargs):
            stream = io.BytesIO(payloads[kwargs["Key"]])
            streams.append(stream)
            return {"Body": stream}

        store.client.get_object.side_effect = get_object
        vectors = list(store.iter_vector_batches("date"))
        full = list(store.iter_batches("date"))
        np.testing.assert_array_equal(np.concatenate(vectors)[:, 0], [1, 1, 1, 2, 2, 2])
        for left, right in zip(vectors, full):
            np.testing.assert_array_equal(left, right["vectors"])
        self.assertTrue(all(stream.closed for stream in streams))

    def test_sliced_arrow_vectors_and_invalid_dimensions(self):
        import pyarrow as pa

        column = pa.array([[1., 2.], [3., 4.], [5., 6.]], type=pa.list_(pa.float32()))
        np.testing.assert_array_equal(_arrow_vectors(column.slice(1, 1)), [[3, 4]])
        with self.assertRaisesRegex(ValueError, "维度"):
            _arrow_vectors(pa.array([[1., 2.], [3.]], type=pa.list_(pa.float32())))
        with self.assertRaisesRegex(ValueError, "空值"):
            _arrow_vectors(pa.array([[1., None]], type=pa.list_(pa.float32())))

    def test_real_parquet_projected_scan_and_full_clustering(self):
        with tempfile.TemporaryDirectory() as root:
            config = settings(vector_store_dir=root, cluster_batch_size=7)
            with patch("app.pipeline.vector_store.get_settings", return_value=config):
                store = LocalVectorStore()
            rng = np.random.default_rng(14)
            for shard in range(3):
                records = [Record(session_id=f"{shard}-{i}", text=f"会话{shard}-{i}",
                                  tokens="贷款 还款" if shard < 2 else "理财 收益", msg_count=2,
                                  customer=f"客户{i}", vector=(rng.normal(size=4) + shard*4).tolist())
                           for i in range(13)]
                store.write_shard("2026-09-15", records, shard)
            expected = list(store.iter_batches("2026-09-15"))
            projected = list(store.iter_vector_batches("2026-09-15"))
            self.assertEqual([len(b["vectors"]) for b in expected], [7, 6]*3)
            for full, vectors in zip(expected, projected):
                np.testing.assert_array_equal(full["vectors"], vectors)
                self.assertEqual(vectors.dtype, np.float32)
            with patch("app.pipeline.cluster.get_settings", return_value=config), \
                 patch("app.pipeline.cluster.get_vector_store", return_value=store):
                result = run_clustering("2026-09-15")
            self.assertEqual(sum(result.heat_by_topic.values()), 78)
            self.assertEqual(result.metrics["n_sessions"], 39)


if __name__ == "__main__":
    unittest.main()
