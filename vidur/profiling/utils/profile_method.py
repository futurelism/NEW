# E:\sample\vidur\profiling\utils\profile_method.py

from enum import Enum, auto

class ProfileMethod(Enum):
    """分析方法枚举"""
    CUDA_EVENTS = auto()
    RECORD_FUNCTION = auto()
    NSYS_NV_TOOL = auto()
