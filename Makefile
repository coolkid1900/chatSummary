.PHONY: build up down logs seed pipeline pipeline-umap api-logs psql redis-cli clean

RUN_DATE ?= 2026-06-27
CUSTOMERS ?= 300

build:            ## 构建镜像
	docker compose build

up:               ## 起 MySQL + Redis + API（OrbStack）
	docker compose up -d mysql redis api

down:             ## 停止并移除容器
	docker compose down

logs:             ## 跟随全部日志
	docker compose logs -f

seed:             ## 造模拟聊天数据写入 MySQL
	RUN_DATE=$(RUN_DATE) CUSTOMERS=$(CUSTOMERS) docker compose run --rm seed

pipeline:         ## 跑端到端流水线（真实 API，需 .env 配 key）
	RUN_DATE=$(RUN_DATE) docker compose run --rm pipeline

pipeline-umap:    ## 用 umap_hdbscan 后端跑（验证两套后端，§6.0）
	RUN_DATE=$(RUN_DATE) \
		docker compose run --rm -e CLUSTER_BACKEND=umap_hdbscan -e MIN_CLUSTER_SIZE=5 pipeline

api-logs:         ## 看 API 日志
	docker compose logs -f api

psql:             ## 进 MySQL 命令行
	docker compose exec mysql mysql -uchat -pchatpass chat_summary

redis-cli:        ## 进 Redis 命令行
	docker compose exec redis redis-cli

clean:            ## 停止并清空数据卷
	docker compose down -v
	rm -rf data/embeddings
