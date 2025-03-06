import argparse
import os
import yaml

from vidur.config import load_config  # 确保导入正确
from vidur.logger import init_logger
from vidur.simulator import Simulator
from vidur.types import ReplicaSchedulerType  # 取消这行的注释

logger = init_logger(__name__)

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Vidur - A simulator for CLIP model serving")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (overrides config)"
    )
    return parser.parse_args()

def main():
    """Vidur 主函数。"""
    # 解析命令行参数
    args = parse_args()

    # 加载配置
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # 如果命令行参数中指定了输出目录，则覆盖配置中的设置
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    # 创建输出目录
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # 将使用的配置保存到输出目录
    config_out_path = os.path.join(config.output_dir, "config.yml")
    with open(config_out_path, "w") as f:
        yaml.dump(config.to_dict(), f)

    # 检查是否为CLIP模式
    is_clip_mode = (
            hasattr(config, "cluster_config") and
            hasattr(config.cluster_config, "replica_scheduler_config") and
            hasattr(config.cluster_config.replica_scheduler_config, "get_type") and
            config.cluster_config.replica_scheduler_config.get_type() == ReplicaSchedulerType.CLIP
    )

    if is_clip_mode:
        logger.info("Running in CLIP simulation mode")

    # 创建并运行模拟器
    simulator = Simulator(config)
    simulator.run()

if __name__ == "__main__":
    main()
