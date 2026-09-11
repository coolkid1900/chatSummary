# 客户经理每日聊天热点总结

基于 **BERTopic** 的「某银行客户经理每日企业微信聊天 TOP 热点总结」批量主题挖掘流水线。
每天从 MySQL 取聊天记录 → 产出 TOP N 热点（**业务热词** + 大模型概括的**客户意图**），
落库 MySQL + Redis 缓存，FastAPI 只读查询。

> 实现严格对应需求文档 `客户经理聊天热点总结_需求文档.md`。本仓库是**本地可跑通的 MVP**，
> 用 OrbStack（docker-compose）起 MySQL8 + Redis5 + 应用，一条命令跑通整条流水线。

## 架构

```
[批处理 Pipeline] (app/pipeline/run_pipeline.py)
  MySQL 取数 → 预处理/会话聚合/去重 → bge-m3 嵌入(Redis 限流+缓存, 分片 parquet)
  → 降维+聚类(CLUSTER_BACKEND 切换) → c-TF-IDF 热词(jieba) → DeepSeek 意图概括
  → 写回 MySQL + Redis 缓存
[FastAPI] (app/api/main.py)  ← 仅读已算好的结果
```

模块对应：`app/pipeline/{run_pipeline,ingest,preprocess,ratelimit,embedding,vector_store,cluster,tokenizer,hotwords,intent,persist}.py`

## 快速开始（OrbStack）

```bash
cp .env.example .env          # 填入 SiliconFlow / DeepSeek 的 base_url 与 key
make build                    # 构建镜像（首次较慢，要装 CPU torch + bertopic）
make up                       # 起 MySQL + Redis + API

make seed                     # 造 ~300 个客户的模拟聊天（默认日期 2026-06-27）
make pipeline                 # 跑端到端流水线（真实 API，需 .env 配 key）

curl http://localhost:8000/topics/2026-06-27 | jq    # 查热点
curl http://localhost:8000/topics/today | jq
```

也可通过 API 覆盖写入某日的模拟聊天数据：
```bash
curl -X POST http://localhost:8000/seed-data \
  -H 'Content-Type: application/json' \
  -d '{"date":"2026-06-27","customers":300,"seed":42}'
```
该接口会先删除该日期已有的模拟聊天；随后重新跑该日期的热点统计时，请传入 `force=true`。

也可通过 HTTP 触发批处理（默认跑上一天，可指定日期）：
```bash
curl -X POST http://localhost:8000/pipeline/run -d '{"date":"2026-06-27"}'
curl http://localhost:8000/pipeline/runs/<run_id>   # 查运行状态
```

维护分词词库（业务词典 / 停用词，落库，改动**下次流水线**生效；首次自动从
`data/dict/*.txt` 播种）：
```bash
curl http://localhost:8000/lexicon/term                      # 列业务词典
curl http://localhost:8000/lexicon/stopword                  # 列停用词
curl -X POST http://localhost:8000/lexicon/term \
  -H 'Content-Type: application/json' -d '{"words":["数字人民币"]}'   # 新增
curl -X DELETE http://localhost:8000/lexicon/stopword/好的    # 删除
```
> 写操作（POST/DELETE）受 `TRIGGER_TOKEN` 保护（设置后需带 `X-Trigger-Token` 头）。

指定日期 / 客户数：`make seed RUN_DATE=2026-06-28 CUSTOMERS=500`，再 `make pipeline RUN_DATE=2026-06-28`。

## 关键设计（对应文档章节）

| 关注点 | 实现 | 章节 |
|---|---|---|
| **聚类后端可切换** | `CLUSTER_BACKEND=incremental`(默认, IncrementalPCA+MiniBatchKMeans 流式) / `umap_hdbscan`(UMAP+HDBSCAN，`USE_GPU=true` 走 cuML) —— 工厂 `cluster.build_models()` | §6.0 |
| **Embedding 减量** | 规则过滤 + 会话聚合 + `hash(text)` Redis 去重缓存 + 批量打满 | §8 |
| **全局限流** | Redis 令牌桶（Lua 原子），embedding/LLM 各一桶，多 worker 共享 | §12 |
| **向量存储** | 分片 parquet，`VECTOR_STORE_BACKEND=local`/`s3` 可切换 | §9 |
| **中文热词** | jieba + **数据库词库**(业务词典/停用词，API 维护) + c-TF-IDF | §10.1/10.2 |
| **意图概括** | DeepSeek(OpenAI 兼容)，按主题调用，中文 prompt，`nr_docs=8` | §10.4 |
| **FastAPI** | 仅读 Redis/MySQL，不现算 | §4 |

## 验证两套聚类后端（§6.0 强制要求）

```bash
make pipeline                               # incremental（默认）
make pipeline-umap                          # umap_hdbscan
```
两条路径都由 `CLUSTER_BACKEND` 环境变量决定，无需改码。

## 部署到 K8s 的注意点

- **向量存储务必用 `VECTOR_STORE_BACKEND=s3`**（对象存储 / MinIO）：pod 本地盘 / emptyDir
  不跨 pod 共享且重启即丢，N 个 embedding worker 写的分片聚类 pod 读不到；普通 PVC 是
  ReadWriteOnce 无法多挂。对象存储天然跨 pod，过期交给 **bucket lifecycle** 规则（短 TTL）。
- 分片文件名带 `WORKER_ID`（K8s 注入 pod 名），避免多 worker 并发写冲突。
- 聚类 `fit` 单点不可并行，按内存条件选后端：2G pod 用 `incremental`；大内存/GPU 用
  `umap_hdbscan`（对应文档 §6 方案 C，单独大内存 CronJob，16~32G）。
- API pod 仅查结果，可保持 2G。

## 环境变量

见 `.env.example`，关键项：`CLUSTER_BACKEND` / `N_CLUSTERS` / `MIN_CLUSTER_SIZE` /
`VECTOR_STORE_BACKEND` / `EMBEDDING_RATE_PER_SEC` / `LLM_RATE_PER_SEC` / `TOP_N` /
`SESSION_GAP_MINUTES` / `NR_DOCS`。

## GPU（可选）

CPU 版 umap/hdbscan 随 bertopic 安装。GPU 路径需 RAPIDS `cuml`（只能经 conda / nvidia
容器镜像安装，本仓库不内置），设 `CLUSTER_BACKEND=umap_hdbscan USE_GPU=true`。
