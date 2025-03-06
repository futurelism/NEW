from enum import auto
from vidur.types.base_int_enum import BaseIntEnum

class ReplicaSchedulerType(BaseIntEnum):
    VLLM = auto()
    ORCA = auto()
    FASTER_TRANSFORMER = auto()
    SARATHI = auto()
    LIGHTLLM = auto()
    CLIP = auto()  # 添加 CLIP 类型
