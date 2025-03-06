"""
Vidur 分析工具
"""
from enum import Enum
from typing import Dict, List, Optional, Union

# 删除对 sarathi 的导入，并内联定义所需的类
class ProfileMethod(Enum):
    """分析方法枚举"""
    MLP = "mlp"
    ATTENTION = "attention"
    ALL_REDUCE = "all_reduce"
    SEND_RECV = "send_recv"
    CPU_OVERHEAD = "cpu_overhead"
    CLIP = "clip"  # 添加 CLIP 方法

class ParallelConfig:
    """并行配置类，替代原 sarathi.config.ParallelConfig"""
    def __init__(self, tp_size: int = 1, pp_size: int = 1):
        self.tensor_parallel_size = tp_size
        self.pipeline_parallel_size = pp_size

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer
from vidur.profiling.utils.singleton import Singleton

__all__ = ["ProfileMethod", "ParallelConfig", "RecordFunctionTracer", "Singleton"]
