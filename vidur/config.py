"""
Configuration loading and handling for Vidur.
"""
import os
import yaml
from typing import Dict, Any, Optional

class BaseConfig:
    """基础配置类。"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """配置转字典。"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

class SimulationConfig(BaseConfig):
    """模拟配置类。"""
    def __init__(
            self,
            output_dir: str = "output",
            seed: Optional[int] = None,
            duration_sec: float = 100.0,
            request_generator: Optional[Dict[str, Any]] = None,
            cluster_config: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        """
        初始化模拟配置。

        Args:
            output_dir: 输出目录
            seed: 随机种子
            duration_sec: 模拟持续时间（秒）
            request_generator: 请求生成器配置
            cluster_config: 集群配置
        """
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.seed = seed
        self.duration_sec = duration_sec

        # 递归处理嵌套配置
        if request_generator:
            self.request_generator_config = RequestGeneratorConfig(**request_generator)
        else:
            self.request_generator_config = None

        if cluster_config:
            self.cluster_config = ClusterConfig(**cluster_config)
        else:
            self.cluster_config = None

class RequestGeneratorConfig(BaseConfig):
    """请求生成器配置类。"""
    def __init__(
            self,
            type: str,
            config: Dict[str, Any],
            **kwargs
    ):
        """
        初始化请求生成器配置。

        Args:
            type: 请求生成器类型
            config: 请求生成器特定配置
        """
        super().__init__(**kwargs)
        self.type = type
        self.config = config

    def get_type(self):
        """获取请求生成器类型。"""
        from vidur.types import RequestGeneratorType
        return RequestGeneratorType[self.type.upper()]

class ClusterConfig(BaseConfig):
    """集群配置类。"""
    def __init__(
            self,
            num_replicas: int = 1,
            replica_scheduler: Optional[Dict[str, Any]] = None,
            scheduler: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        """
        初始化集群配置。

        Args:
            num_replicas: 副本数量
            replica_scheduler: 副本调度器配置
            scheduler: 请求调度器配置
        """
        super().__init__(**kwargs)
        self.num_replicas = num_replicas

        if replica_scheduler:
            self.replica_scheduler_config = ReplicaSchedulerConfig(**replica_scheduler)
        else:
            self.replica_scheduler_config = None

        if scheduler:
            self.scheduler_config = SchedulerConfig(**scheduler)
        else:
            self.scheduler_config = None

class ReplicaSchedulerConfig(BaseConfig):
    """副本调度器配置类。"""
    def __init__(
            self,
            type: str,
            **kwargs
    ):
        """
        初始化副本调度器配置。

        Args:
            type: 副本调度器类型
        """
        super().__init__(**kwargs)
        self.type = type

    def get_type(self):
        """获取副本调度器类型。"""
        from vidur.types import ReplicaSchedulerType
        return ReplicaSchedulerType[self.type.upper()]

class SchedulerConfig(BaseConfig):
    """调度器配置类。"""
    def __init__(
            self,
            type: str,
            max_batch_size: int = 16,
            max_batch_tokens: int = 2048,
            **kwargs
    ):
        """
        初始化调度器配置。

        Args:
            type: 调度器类型
            max_batch_size: 最大批处理大小
            max_batch_tokens: 最大批处理token数
        """
        super().__init__(**kwargs)
        self.type = type
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens

    def get_type(self):
        """获取调度器类型。"""
        from vidur.types import SchedulerType
        return SchedulerType[self.type.upper()]

def load_config(config_path: str) -> SimulationConfig:
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
