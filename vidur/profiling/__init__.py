# 在 E:\sample\vidur\profiling\utils\__init__.py 中修改
# 原代码:
# from sarathi.config import ParallelConfig

# 修改为:
from vidur.profiling.utils.singleton import Singleton
from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer

__all__ = [
    "Singleton",
    "RecordFunctionTracer",
]
