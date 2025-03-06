from vidur.types.activation_type import ActivationType
from vidur.types.base_int_enum import BaseIntEnum
from vidur.types.device_sku_type import DeviceSKUType
from vidur.types.event_type import EventType
from vidur.types.execution_time_predictor_type import ExecutionTimePredictorType
from vidur.types.global_scheduler_type import GlobalSchedulerType
from vidur.types.node_sku_type import NodeSKUType
from vidur.types.norm_type import NormType
from vidur.types.replica_scheduler_type import ReplicaSchedulerType
from vidur.types.request_generator_type import RequestGeneratorType
from vidur.types.request_interval_generator_type import RequestIntervalGeneratorType
from vidur.types.request_length_generator_type import RequestLengthGeneratorType

__all__ = [
    EventType,
    ExecutionTimePredictorType,
    GlobalSchedulerType,
    RequestGeneratorType,
    RequestLengthGeneratorType,
    RequestIntervalGeneratorType,
    ReplicaSchedulerType,
    DeviceSKUType,
    NodeSKUType,
    NormType,
    ActivationType,
    BaseIntEnum,

]
class ReplicaSchedulerType(BaseIntEnum):
    VLLM = 1
    ORCA = 2
    LIGHTLLM = 3
    FASTER_TRANSFORMER = 4
    SARATHI = 5
    CLIP = 6  # 新增


class SchedulerType:
    pass