from vidur.scheduler.replica_scheduler.faster_transformer_replica_scheduler import (
    FasterTransformerReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.lightllm_replica_scheduler import (
    LightLLMReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.orca_replica_scheduler import (
    OrcaReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.vllm_replica_scheduler import (
    VLLMReplicaScheduler,
)
from vidur.types import ReplicaSchedulerType
from vidur.utils.base_registry import BaseRegistry
from vidur.types import ReplicaSchedulerType
from vidur.utils.base_registry import BaseRegistry
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import SarathiReplicaScheduler
from vidur.scheduler.replica_scheduler.vllm_replica_scheduler import VLLMReplicaScheduler
from vidur.scheduler.replica_scheduler.orca_replica_scheduler import OrcaReplicaScheduler
from vidur.scheduler.replica_scheduler.faster_transformer_replica_scheduler import FasterTransformerReplicaScheduler
from vidur.scheduler.replica_scheduler.lightllm_replica_scheduler import LightLLMReplicaScheduler
from vidur.scheduler.replica_scheduler.clip_replica_scheduler import ClipReplicaScheduler  # 添加导入

class ReplicaSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ReplicaSchedulerType:
        return ReplicaSchedulerType.from_str(key_str)


# 注册各种调度器
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.VLLM, VLLMReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.ORCA, OrcaReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.FASTER_TRANSFORMER, FasterTransformerReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.SARATHI, SarathiReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.LIGHTLLM, LightLLMReplicaScheduler)

ReplicaSchedulerRegistry.register(ReplicaSchedulerType.CLIP, ClipReplicaScheduler)
ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.FASTER_TRANSFORMER, FasterTransformerReplicaScheduler
)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.ORCA, OrcaReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.SARATHI, SarathiReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.VLLM, VLLMReplicaScheduler)
ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.LIGHTLLM, LightLLMReplicaScheduler
)
