-- 客户经理聊天热点总结 —— 表结构
-- 由 docker-compose 在 MySQL 首次启动时自动执行（/docker-entrypoint-initdb.d）。
-- ORM 侧亦可 Base.metadata.create_all 兜底建表。

CREATE DATABASE IF NOT EXISTS chat_summary
  DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE chat_summary;

-- ===== 原始聊天记录 =====
-- 聊天记录按 from_user 分 20 张表 user_chat_record_sharding_1..20（分表键 from_user，
-- 路由=Java String.hashCode）。这些分表由应用启动时 app/sharding.ensure_shard_tables()
-- 按业务方 DDL 幂等创建（CREATE TABLE IF NOT EXISTS），不在此处建。

-- ===== 每日热点结果表（§13）=====
CREATE TABLE IF NOT EXISTS daily_hot_topics (
  id                  BIGINT       NOT NULL AUTO_INCREMENT,
  stat_date           VARCHAR(10)  NOT NULL,   -- YYYY-MM-DD
  `rank`              INT          NOT NULL,
  topic_id            INT          NOT NULL,
  heat                INT          NOT NULL,   -- 热度 = 该主题客户消息数
  customer_count      INT          NOT NULL DEFAULT 0,   -- 涉及去重客户数（广度）
  prev_heat           INT          NOT NULL DEFAULT 0,   -- 昨日同主题热度
  heat_change_pct     FLOAT        NULL,                 -- 环比涨幅，新主题为 NULL
  is_new              TINYINT(1)   NOT NULL DEFAULT 0,   -- 新出现主题
  is_surge            TINYINT(1)   NOT NULL DEFAULT 0,   -- 突增
  hot_words           TEXT         NOT NULL,   -- JSON: [["提前还款",0.41],...]
  business_words      TEXT         NOT NULL,   -- 大模型筛出的有业务意义热词 JSON: ["提前还款","违约金"]
  customer_intent     TEXT         NOT NULL,
  representative_docs TEXT         NOT NULL,   -- JSON: ["...","..."]
  cluster_backend     VARCHAR(32)  NOT NULL,
  created_at          DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_date_rank (stat_date, `rank`),
  KEY idx_stat_date (stat_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ===== 词库表 =====
-- 维护业务词(term)、停用词(stopword)和寒暄词(chitchat)。
CREATE TABLE IF NOT EXISTS lexicon (
  id          BIGINT       NOT NULL AUTO_INCREMENT,
  kind        VARCHAR(16)  NOT NULL,   -- term | stopword | chitchat
  word        VARCHAR(128) NOT NULL,
  enabled     TINYINT(1)   NOT NULL DEFAULT 1,  -- 软禁用标记
  created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_kind_word (kind, word),
  KEY idx_kind (kind)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ===== 流水线运行记录表 =====
-- 保存每次热点统计任务的状态、规模及失败信息，供监控和排障使用。
CREATE TABLE IF NOT EXISTS pipeline_runs (
  id                BIGINT       NOT NULL AUTO_INCREMENT,
  run_id            VARCHAR(64)  NOT NULL,
  stat_date         VARCHAR(10)  NOT NULL,   -- YYYY-MM-DD
  status            VARCHAR(16)  NOT NULL,   -- running | success | failed
  cluster_backend   VARCHAR(32)  NOT NULL DEFAULT '',
  n_messages        INT          NOT NULL DEFAULT 0,
  n_sessions        INT          NOT NULL DEFAULT 0,
  n_topics          INT          NOT NULL DEFAULT 0,
  embed_api_calls   INT          NOT NULL DEFAULT 0,
  embed_cache_hits  INT          NOT NULL DEFAULT 0,
  llm_failures      INT          NOT NULL DEFAULT 0,
  duration_sec      FLOAT        NOT NULL DEFAULT 0,
  error             TEXT         NOT NULL,
  created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_run_id (run_id),
  KEY idx_stat_date (stat_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ===== 经理×客户接触总结（两层：日笔记 → 月度总结）=====
-- 工号/客户 ID 使用 utf8mb4_bin，与聊天分表一致，保证大小写敏感的精确匹配。

-- 每对(经理,客户)每天的沟通统计：由代码从原始聊天计算，包含寒暄。
CREATE TABLE IF NOT EXISTS contact_daily_stats (
  manager_id          VARCHAR(100) COLLATE utf8mb4_bin NOT NULL,
  customer_id         VARCHAR(128) COLLATE utf8mb4_bin NOT NULL,
  stat_date           DATE         NOT NULL,
  msg_count           INT          NOT NULL DEFAULT 0,   -- 双向文本消息数
  customer_msg_count  INT          NOT NULL DEFAULT 0,
  manager_msg_count   INT          NOT NULL DEFAULT 0,
  session_count       INT          NOT NULL DEFAULT 0,   -- 跨零点会话归入开始那天
  customer_initiated  INT          NOT NULL DEFAULT 0,   -- 客户发出首条消息的会话数
  first_msg_time      DATETIME     NOT NULL,
  last_msg_time       DATETIME     NOT NULL,
  source_fp           VARCHAR(64)  NOT NULL,             -- 当天消息哈希，变化说明有补录
  updated_at          DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (manager_id, customer_id, stat_date),
  KEY idx_cds_mgr_date (manager_id, stat_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Prompt A 产出的日笔记；只有寒暄的天也写一行 status=no_business。
CREATE TABLE IF NOT EXISTS contact_daily_notes (
  id               BIGINT       NOT NULL AUTO_INCREMENT,
  manager_id       VARCHAR(100) COLLATE utf8mb4_bin NOT NULL,
  customer_id      VARCHAR(128) COLLATE utf8mb4_bin NOT NULL,
  note_date        DATE         NOT NULL,
  status           VARCHAR(16)  NOT NULL,   -- ok | no_business | failed
  topics           JSON         NOT NULL,
  customer_points  JSON         NOT NULL,
  manager_actions  JSON         NOT NULL,
  facts            JSON         NOT NULL,
  pending          JSON         NOT NULL,
  source_fp        VARCHAR(64)  NOT NULL DEFAULT '',
  prompt_version   VARCHAR(32)  NOT NULL DEFAULT '',
  model            VARCHAR(64)  NOT NULL DEFAULT '',
  run_id           VARCHAR(64)  NOT NULL DEFAULT '',
  llm_call_id      VARCHAR(64)  NULL,      -- 冷启动多天打包时多行共用
  error            VARCHAR(500) NULL,
  created_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_cdn_pair_date (manager_id, customer_id, note_date),
  KEY idx_cdn_mgr_date (manager_id, note_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 信号开启/关闭事件：信号状态的唯一来源，与日笔记同一事务写入。
CREATE TABLE IF NOT EXISTS contact_signal_events (
  id                 BIGINT       NOT NULL AUTO_INCREMENT,
  note_id            BIGINT       NOT NULL,
  manager_id         VARCHAR(100) COLLATE utf8mb4_bin NOT NULL,
  customer_id        VARCHAR(128) COLLATE utf8mb4_bin NOT NULL,
  event_date         DATE         NOT NULL,
  signal_code        VARCHAR(32)  NOT NULL,
  status             VARCHAR(8)   NOT NULL,   -- open | closed
  evidence           VARCHAR(200) NOT NULL DEFAULT '',
  evidence_verified  TINYINT(1)   NOT NULL DEFAULT 0,  -- 证据是否为脱敏原文子串
  created_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_cse_pair_date (manager_id, customer_id, event_date),
  KEY idx_cse_mgr_code (manager_id, signal_code, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 日笔记批次：manager_id='' 为夜间全量，有值为某位经理补数。
CREATE TABLE IF NOT EXISTS contact_note_runs (
  run_id          VARCHAR(64)  NOT NULL,
  stat_date       DATE         NOT NULL,
  manager_id      VARCHAR(100) COLLATE utf8mb4_bin NOT NULL DEFAULT '',
  trigger_type    VARCHAR(16)  NOT NULL,   -- nightly | backfill | manual
  `force`         TINYINT(1)   NOT NULL DEFAULT 0,
  status          VARCHAR(16)  NOT NULL DEFAULT 'queued',  -- queued | running | success | partial_success | failed
  pair_count      INT          NOT NULL DEFAULT 0,
  llm_pair_count  INT          NOT NULL DEFAULT 0,
  skipped_count   INT          NOT NULL DEFAULT 0,   -- 只剩寒暄、未调用模型
  reused_count    INT          NOT NULL DEFAULT 0,   -- 输入未变、沿用已有笔记
  failed_count    INT          NOT NULL DEFAULT 0,
  input_tokens    INT          NOT NULL DEFAULT 0,
  attempt_token   VARCHAR(64)  NOT NULL DEFAULT '',
  heartbeat_at    DATETIME     NULL,
  error           TEXT         NOT NULL,
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at      DATETIME     NULL,
  finished_at     DATETIME     NULL,
  PRIMARY KEY (run_id),
  KEY idx_cnr_date_mgr (stat_date, manager_id, status),
  KEY idx_cnr_status (status, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Prompt B 产出的月度总结。
CREATE TABLE IF NOT EXISTS contact_summaries (
  id                BIGINT       NOT NULL AUTO_INCREMENT,
  manager_id        VARCHAR(100) COLLATE utf8mb4_bin NOT NULL,
  customer_id       VARCHAR(128) COLLATE utf8mb4_bin NOT NULL,
  window_start      DATE         NOT NULL,
  window_end        DATE         NOT NULL,
  status            VARCHAR(16)  NOT NULL,   -- success | template | review
  stats_json        JSON         NOT NULL,
  signals_json      JSON         NOT NULL,
  pending_json      JSON         NOT NULL,
  summary           TEXT         NULL,       -- review 时为空
  main_topics       JSON         NOT NULL,
  suggestions       JSON         NOT NULL,
  incomplete_days   INT          NOT NULL DEFAULT 0,  -- 笔记失败或缺失的天数
  input_fp          VARCHAR(64)  NOT NULL,            -- 输入不变则复用，不调模型
  prompt_version    VARCHAR(32)  NOT NULL DEFAULT '',
  model             VARCHAR(64)  NOT NULL DEFAULT '',
  job_id            VARCHAR(64)  NOT NULL DEFAULT '',
  llm_call_id       VARCHAR(64)  NULL,
  validation_error  VARCHAR(500) NULL,
  created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_cs_pair_window (manager_id, customer_id, window_end),
  KEY idx_cs_mgr_window (manager_id, window_end)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 按需生成某经理月度总结的异步任务。
CREATE TABLE IF NOT EXISTS contact_summary_jobs (
  job_id           VARCHAR(64)  NOT NULL,
  manager_id       VARCHAR(100) COLLATE utf8mb4_bin NOT NULL,
  window_end       DATE         NOT NULL,
  customer_ids     JSON         NULL,      -- 为空表示窗口内有沟通的全部客户
  `force`          TINYINT(1)   NOT NULL DEFAULT 0,
  status           VARCHAR(16)  NOT NULL DEFAULT 'queued',
  stage            VARCHAR(32)  NOT NULL DEFAULT 'queued',  -- backfill_notes | summarizing | done
  total_customers  INT          NOT NULL DEFAULT 0,
  done_customers   INT          NOT NULL DEFAULT 0,
  reused_count     INT          NOT NULL DEFAULT 0,
  template_count   INT          NOT NULL DEFAULT 0,
  review_count     INT          NOT NULL DEFAULT 0,
  failed_count     INT          NOT NULL DEFAULT 0,
  input_tokens     INT          NOT NULL DEFAULT 0,
  attempt_token    VARCHAR(64)  NOT NULL DEFAULT '',
  heartbeat_at     DATETIME     NULL,
  error            TEXT         NOT NULL,
  created_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at       DATETIME     NULL,
  finished_at      DATETIME     NULL,
  PRIMARY KEY (job_id),
  KEY idx_csj_mgr_window (manager_id, window_end, status),
  KEY idx_csj_status (status, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 大模型调用日志：校准 token 估算、排查限流与统计成本；不保存 prompt 原文。
CREATE TABLE IF NOT EXISTS llm_calls (
  call_id            VARCHAR(64)  NOT NULL,
  purpose            VARCHAR(16)  NOT NULL,   -- note | summary
  manager_id         VARCHAR(100) COLLATE utf8mb4_bin NOT NULL DEFAULT '',
  customer_id        VARCHAR(128) COLLATE utf8mb4_bin NOT NULL DEFAULT '',
  est_input_tokens   INT          NOT NULL DEFAULT 0,
  prompt_tokens      INT          NOT NULL DEFAULT 0,
  cached_tokens      INT          NOT NULL DEFAULT 0,
  completion_tokens  INT          NOT NULL DEFAULT 0,
  latency_ms         INT          NOT NULL DEFAULT 0,
  http_status        INT          NOT NULL DEFAULT 0,
  status             VARCHAR(16)  NOT NULL,   -- ok | failed | rate_limited | invalid
  attempts           INT          NOT NULL DEFAULT 1,
  prompt_version     VARCHAR(32)  NOT NULL DEFAULT '',
  model              VARCHAR(64)  NOT NULL DEFAULT '',
  error              VARCHAR(500) NULL,
  created_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (call_id),
  KEY idx_llm_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
