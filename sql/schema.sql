-- 客户经理聊天热点总结 —— 表结构
-- 由 docker-compose 在 MySQL 首次启动时自动执行（/docker-entrypoint-initdb.d）。
-- ORM 侧亦可 Base.metadata.create_all 兜底建表。

CREATE DATABASE IF NOT EXISTS chat_summary
  DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE chat_summary;

-- ===== 原始消息表：企业微信聊天存档字段 =====
CREATE TABLE IF NOT EXISTS messages (
  id        BIGINT       NOT NULL AUTO_INCREMENT,
  msg_id    VARCHAR(64)  NOT NULL,
  sender    VARCHAR(64)  NOT NULL,             -- 发送方（客户经理工号 / 客户外部 ID）
  receiver  VARCHAR(64)  NOT NULL,             -- 接收方
  role      VARCHAR(16)  NOT NULL,             -- staff | customer
  msg_type  VARCHAR(16)  NOT NULL,             -- text|image|voice|file|emotion|system
  content   TEXT         NOT NULL,
  msg_time  DATETIME     NOT NULL,
  PRIMARY KEY (id),
  KEY idx_msg_id (msg_id),
  KEY idx_sender (sender),
  KEY idx_msg_time (msg_time),
  KEY idx_msg_time_role (msg_time, role),
  -- 取数查询专用：等值(role,msg_type) + ORDER BY sender,receiver,msg_time,id 走索引序，免 filesort
  KEY idx_ingest_order (role, msg_type, sender, receiver, msg_time, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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
