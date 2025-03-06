import argparse
import os
import yaml

from vidur.config.config import SimulationConfig  # 导入配置类
from vidur.logger import init_logger
from vidur.simulator import Simulator
from vidur.types import ReplicaSchedulerType

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

def load_config_from_path(config_path: str) -> SimulationConfig:
    """
    从YAML文件加载配置。

    Args:
        config_path: 配置文件路径

    Returns:
        加载的配置对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到：{config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return SimulationConfig(**config_dict)

def main():
    """Vidur 主函数。"""
    # 解析命令行参数
    args = parse_args()

    # 加载配置
    logger.info(f"Loading configuration from {args.config}")
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 确保必要的配置部分存在
        if 'metrics_config' not in config_dict:
            config_dict['metrics_config'] = {'output_dir': 'simulator_output'}
        elif 'output_dir' not in config_dict['metrics_config']:
            config_dict['metrics_config']['output_dir'] = 'simulator_output'

        # 手动创建配置对象
        from vidur.config.config import MetricsConfig, ClusterConfig

        # 先创建 metrics_config 对象
        metrics_config = MetricsConfig(**config_dict.get('metrics_config', {}))

        # 如果命令行参数中指定了输出目录，则覆盖配置中的设置
        if args.output_dir is not None:
            metrics_config.output_dir = args.output_dir

        # 其他配置项...

        # 确保输出目录存在
        os.makedirs(metrics_config.output_dir, exist_ok=True)

        # 创建配置对象
        config = SimulationConfig(
            seed=config_dict.get('seed', 42),
            log_level=config_dict.get('log_level', 'info'),
            time_limit=config_dict.get('time_limit', 0),
            cluster_config=ClusterConfig(**config_dict.get('cluster_config', {})),
            request_generator_config=config_dict.get('request_generator_config'),
            execution_time_predictor_config=config_dict.get('execution_time_predictor_config'),
            metrics_config=metrics_config
        )

    except Exception as e:
        logger.error(f"加载配置时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # 将使用的配置保存到输出目录
    config_out_path = os.path.join(metrics_config.output_dir, "config.yml")
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
