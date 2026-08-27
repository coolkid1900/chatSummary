"""集中式配置：全部来自环境变量（pydantic-settings）。

聚类后端（CLUSTER_BACKEND）与向量存储后端（VECTOR_STORE_BACKEND）均由环境变量
切换，不得硬编码（对应需求文档 §6.0 / §9）。
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 聊天记录分表（分表键 from_user，路由=Java hashCode，见 app/sharding.py）
    shard_count: int = 20
    shard_table_prefix: str = "user_chat_record_sharding_"
    # 角色判定（企业微信会话存档）：企业成员(客户经理)的 from/to 是 userid(工号)，
    # 外部联系人(微信客户)是 external_userid，以 wm/wo 开头（机器人 wb）。
    # 故 from_user 以下列任一前缀开头即视为客户(customer)，否则为客户经理(staff)。
    external_id_prefixes: str = "wm,wo"

    # MySQL
    mysql_host: str = "mysql"
    mysql_port: int = 3306
    mysql_user: str = "chat"
    mysql_password: str = "chatpass"
    mysql_database: str = "chat_summary"

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0

    # Embedding (SiliconFlow bge-m3)
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 64
    embedding_rate_per_sec: float = 4
    embedding_concurrency: int = 8  # 单 pod 内并发请求数，受令牌桶约束以打满 m/s

    # LLM (DeepSeek)
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_rate_per_sec: float = 8
    llm_concurrency: int = 8  # 意图概括并发数，受 LLM 令牌桶约束打满 n/s

    # 聚类后端（§6.0）
    cluster_backend: str = "incremental"  # incremental | umap_hdbscan
    n_components: int = 10
    n_clusters: int = 50
    min_cluster_size: int = 100
    use_gpu: bool = False

    # 向量存储（§9）
    vector_store_backend: str = "local"  # local | s3
    vector_store_dir: str = "/app/data/embeddings"
    vector_ttl_days: int = 7
    s3_endpoint: str = ""
    s3_bucket: str = "chat-embeddings"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region: str = "us-east-1"

    # 业务参数
    top_n: int = 20
    # 热度排序的客户广度权重：排序分 = heat × customer_count^w。
    # w=0 退化为纯按客户消息数排序；调大则更看重「多少客户在关心」，抑制话痨客户刷榜。
    rank_breadth_weight: float = 0.3
    session_gap_minutes: int = 30
    min_text_len: int = 5
    dedup_cache_ttl: int = 604800
    nr_docs: int = 8
    nr_repr_docs: int = 5

    # 流式/分片与性能
    shard_size: int = 2048          # 每个 parquet 分片的会话数（也是 embedding 累积窗口）
    cluster_batch_size: int = 2048  # 聚类分批读取大小
    jieba_workers: int = 4          # jieba 预分词多进程数（CPU 大户，§5 步骤7）
    embed_cache_max_len: int = 24   # 只把短文本（高频话术）放 Redis 缓存（§9 小热数据）
    repr_pool_size: int = 40        # 每主题代表文档蓄水池上限（送 LLM 前 MMR 选样）
    topic_merge_sim: float = 0.85   # 主题质心余弦≥此值则合并近重复簇；0 关闭

    # 趋势/突增（环比昨日）
    trend_match_sim: float = 0.3    # 与昨日主题的热词 Jaccard≥此值视为同一主题
    surge_pct: float = 0.5          # 环比涨幅≥此值（+50%）判为突增
    surge_min_heat: int = 20        # 突增的最小热度门槛，过滤长尾噪声

    worker_id: str = "local"

    # HTTP 触发批处理的鉴权 token（为空则不校验，仅建议本地/内网用）
    trigger_token: str = ""

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
