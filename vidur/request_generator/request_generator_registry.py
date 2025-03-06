"""
Registry for request generators in Vidur.
"""
from typing import Dict, Type, Any
from vidur.request_generator.base_request_generator import BaseRequestGenerator

class RequestGeneratorRegistry:
    """请求生成器注册表。"""
    _registry: Dict[str, Type[BaseRequestGenerator]] = {}

    @classmethod
    def register(cls, generator_type: str):
        """
        注册请求生成器类型的装饰器。

        Args:
            generator_type: 生成器类型名称
        """
        def wrapper(generator_cls):
            cls._registry[generator_type] = generator_cls
            return generator_cls
        return wrapper

    @classmethod
    def get(cls, generator_type: str) -> Type[BaseRequestGenerator]:
        """
        获取指定类型的请求生成器类。

        Args:
            generator_type: 生成器类型名称

        Returns:
            请求生成器类
        """
        if generator_type not in cls._registry:
            raise ValueError(f"Unknown generator type: {generator_type}")
        return cls._registry[generator_type]

    @classmethod
    def create(cls, generator_type: str, config: Any) -> BaseRequestGenerator:
        """
        创建指定类型的请求生成器实例。

        Args:
            generator_type: 生成器类型名称
            config: 生成器配置

        Returns:
            请求生成器实例
        """
        generator_cls = cls.get(generator_type)
        return generator_cls(config)
from vidur.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from vidur.request_generator.trace_replay_request_generator import (
    TraceReplayRequestGenerator,
)
from vidur.types import RequestGeneratorType
from vidur.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):
    pass


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.TRACE_REPLAY, TraceReplayRequestGenerator
)
