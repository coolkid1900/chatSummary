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
curl -X POST http://localhost:8000/pipeline/run -d '{"date":"2026-06-27","force":true}' # 强制启动并重算
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

## incremental 训练及质量验证

incremental 先扫描全部向量训练 PCA，固定投影后将降维向量写入临时缓存，再分批训练
MiniBatchKMeans，最后按最终标签统一计算热度、客户数、c-TF-IDF 和代表池。
该路径按需加载依赖，不导入 BERTopic；代表向量独立复制，避免代表池间接保留整批原始向量。
PCA 按训练批大小累积，文件分片边界不触发额外拟合；完整尾批不再被强行切成很小的一批。
本地 parquet 及 S3 均按列、按批次读取，缓存完成或异常退出后自动清理。
向量缓存约占 `会话数 × 降维维数 × 4` 字节磁盘；客户去重集合、词表仍随数据规模增长。

合并使用**原始向量空间的全体会话质心**，每条会话等权。`TOPIC_MERGE_LINKAGE` 支持：

| 配置 | 规则 | 特点 |
|---|---|---|
| `single`（通用默认） | 阈值图的连通分量 | 同类容易合并，但相似链条可能连接不同业务 |
| `average` | 两组初始簇之间的平均余弦距离 | 初始簇等权，折中考虑整体相似度 |
| `complete` | 两组初始簇之间的最大余弦距离 | 更严格，可能拆分同类业务 |

`TOPIC_MERGE_SIM=0` 关闭合并，其余值对应距离阈值 `1-SIM`。不同规则必须分别调阈值。
UMAP 可用 `UMAP_TOPIC_MERGE_LINKAGE` / `UMAP_TOPIC_MERGE_SIM` 独立覆盖；
未设置时兼容继承 `TOPIC_MERGE_*`。本地 UMAP 保持 `single / 0.85`，不使用 incremental 的调参阈值。
ARI 衡量与参考业务分组的一致性，同时惩罚误合并和同类拆分；它不是热点文案质量或准确率。

本地 `.env` 使用经过 300～10,000 会话模拟数据验证的配置：

```dotenv
CLUSTER_BACKEND=incremental
INCREMENTAL_N_COMPONENTS=32
N_CLUSTERS=30
INCREMENTAL_EPOCHS=1
CLUSTER_RANDOM_SEED=42
TOPIC_MERGE_LINKAGE=average
TOPIC_MERGE_SIM=0.80
```

这些值是小型银行业务模拟集的验证配置，通用默认值仍见 `.env.example`。
真实数据、更多业务类别、不同 embedding 或分片规模需要重新评估。

正式回归测试覆盖流式聚类、合并策略、分片边界、统计一致性和缓存清理：

```bash
python -m unittest discover -s tests -v
```

评估真实数据时，应使用独立标注集，并检查不同训练批大小和样本顺序下的分组稳定性。

## 部署到 K8s 的注意点

- **向量存储务必用 `VECTOR_STORE_BACKEND=s3`**（对象存储 / MinIO）：pod 本地盘 / emptyDir
  不跨 pod 共享且重启即丢，N 个 embedding worker 写的分片聚类 pod 读不到；普通 PVC 是
  ReadWriteOnce 无法多挂。对象存储天然跨 pod，过期交给 **bucket lifecycle** 规则（短 TTL）。
- 分片文件名带 `WORKER_ID`（K8s 注入 pod 名），避免多 worker 并发写冲突。
- 聚类 `fit` 单点执行，按实测内存选择后端：内存受限时优先评估 `incremental`；大内存/GPU 用
  `umap_hdbscan`（对应文档 §6 方案 C，单独大内存 CronJob，16~32G）。
- API pod 仅查结果。内存预算需包含运行时、客户集合和词表，不保证所有数据都能放入 2G。

## 环境变量

见 `.env.example`，关键项：`CLUSTER_BACKEND` / `N_CLUSTERS` / `MIN_CLUSTER_SIZE` /
`VECTOR_STORE_BACKEND` / `EMBEDDING_RATE_PER_SEC` / `LLM_RATE_PER_SEC` / `TOP_N` /
`SESSION_GAP_MINUTES` / `NR_DOCS`。

Embedding 单条输入由 `EMBEDDING_MAX_INPUT_CHARS=7500` 限制（正整数），
使用 Python `len(text)` 判断长度，超长时保留前 N 个 Unicode 码点，短文本保持原样；
空格、换行和标点也计入长度。无需 tokenizer 或下载分词文件。
字符数不等于 token 数，此配置不能严格保证接口的 8K token 限制，可根据实际文本调低上限。
原始脱敏会话仍完整保存；修改截断上限后，Redis 向量缓存及旧分片不再复用，
后续运行会重新嵌入。每批仍最多 64 条，此参数不限制单批总长度。
修改 `.env` 后重启进程；Compose 部署需重建镜像并重新创建容器以加载代码和配置。

## GPU（可选）

CPU 版 umap/hdbscan 随 bertopic 安装。GPU 路径需 RAPIDS `cuml`（只能经 conda / nvidia
容器镜像安装，本仓库不内置），设 `CLUSTER_BACKEND=umap_hdbscan USE_GPU=true`。

## 经理×客户接触总结（两层：日笔记 → 月度总结）

按“客户经理 + 客户”维度，生成近 1 个月的接触总结（沟通频次、主要话题、客户态度、
已关闭或转变的信号、跟进建议）。代码在 `app/contacts/`，接口在 `app/api/contacts.py`。

```
夜间批处理（前一天，只处理有消息的对话对）
  分表双向消息 ──代码──→ contact_daily_stats（沟通统计，含寒暄）
               └─清洗/脱敏→ Prompt A ─→ contact_daily_notes + contact_signal_events
按需查询（异步任务）
  补齐窗口内缺失的日笔记 → 代码拼装统计/信号时间线/笔记 → Prompt B → contact_summaries
```

- **原文只送进模型一次**：月度总结只读日笔记；输入不变（`input_fp`）时直接复用，不调用模型。
- **数字和状态由代码决定**：沟通次数、上中下旬、信号最终状态都由代码计算；信号证据必须是
  原文子串，否则不参与时间线。模型输出经校验（次数一致、字数、数字来源、PII），失败重试一次，
  仍失败标记为 `review` 待人工复核。
- **限流**：Redis 滑动窗口同时约束“5 次/10 秒”和“30,000 输入 token/分钟”，按估算预占、按
  实际 `prompt_tokens` 修正；429 时全局冷却；按需总结优先于夜间笔记。
- **只剩寒暄的天不调用模型**，直接记为 `no_business`。

### 运行

```bash
make build && make up        # 包含 contact-worker
# 生成某经理近 1 个月的总结（异步，返回 job_id）
curl -X POST localhost:8000/contacts/summary/submit -H 'Content-Type: application/json' \
  -d '{"manager_id":"manager001","end_date":"2026-08-19"}'
curl -X POST localhost:8000/contacts/summary-job/query -H 'Content-Type: application/json' \
  -d '{"job_id":"csj_..."}'
curl -X POST localhost:8000/contacts/summary/list -H 'Content-Type: application/json' \
  -d '{"manager_id":"manager001","end_date":"2026-08-19"}'
```

接口全部为 POST、参数放 body（工号和客户 ID 不出现在 URL 与访问日志），校验失败返回 400：

| 接口 | 用途 |
|---|---|
| `/contacts/summary/submit` | 提交总结任务（202）；同一经理同一窗口已有任务覆盖本次范围时返回已有 job_id |
| `/contacts/summary-job/query` | 任务状态、进度、估算剩余时间 |
| `/contacts/summary/list` | 经理名下客户总结列表，可按信号/状态过滤、排序、分页 |
| `/contacts/summary/detail` | 单个客户总结详情（信号含具体日期与证据） |
| `/contacts/note/list` | 单个客户的日笔记与信号证据（追溯用，区间≤31 天） |
| `/contacts/signal/list` | 跨客户查询某信号，例如风险敏感未关闭的客户 |
| `/contacts/note-run/submit` | 手动触发某天日笔记（全量或指定经理）；同范围有任务时 409 |
| `/contacts/note-run/query` | 日笔记批次状态 |

鉴权沿用 `X-Trigger-Token`（服务级）。**按操作人的数据权限尚未实现**：上线前需由调用方网关
保证经理只能查看自己的客户，或在此基础上增加操作人校验。

### 运维要点

- 夜间批次：设置 `CONTACT_NIGHTLY_ENABLED=true`，worker 每天 `CONTACT_NIGHTLY_HOUR` 点后自动
  创建前一天的全量批次；失败后通过 `/contacts/note-run/submit` 重跑。日期按本地时区，需设置 `TZ`。
- 聊天补录：已覆盖的日期不会自动重算。对该日期提交 `note-run`（不带 force）即可，只重算消息
  有变化的对话对；随后重新提交总结任务。
- 修改 prompt 后提升 `app/contacts/prompts.py` 中的版本号：日笔记版本变化会在下次批次重算，
  总结版本变化会让所有总结的 `input_fp` 失效。
- worker 可多副本部署：任务带租约，副本异常退出后其他副本在 `CONTACT_JOB_LEASE_SECONDS` 后接管；
  正常停止时任务放回队列。写库使用 READ COMMITTED 并对死锁自动重试。
- 用 `llm_calls` 表对比 `est_input_tokens` 与 `prompt_tokens`，校准 `CONTACT_TOKENS_PER_*`。

### 测试

```bash
# 纯逻辑单测
python -m unittest tests.test_contacts
# 限流 Lua 脚本与端到端流程（需要 docker compose up -d mysql redis；模型用确定性假客户端）
REDIS_HOST=127.0.0.1 python -m unittest tests.test_contacts_limiter
MYSQL_HOST=127.0.0.1 REDIS_HOST=127.0.0.1 python -m unittest tests.test_contacts_integration
```

集成测试使用 `zzit_` 前缀的工号和客户 ID 写入本地库，结束后清理。
