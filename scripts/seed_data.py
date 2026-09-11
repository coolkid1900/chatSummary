"""兼容旧的命令行入口；实际造数逻辑位于 app.api.seed_data。"""
from app.api.seed_data import main


if __name__ == "__main__":
    main()
