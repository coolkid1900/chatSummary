"""兼容旧的命令行入口；实际流水线逻辑位于 app.pipeline.run_pipeline。"""
from app.pipeline.run_pipeline import main


if __name__ == "__main__":
    main()
