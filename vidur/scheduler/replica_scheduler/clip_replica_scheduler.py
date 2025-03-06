import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from vidur.config import BaseReplicaSchedulerConfig, ReplicaConfig, MetricsConfig
from vidur.entities import Batch, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.types import ReplicaSchedulerType

class ClipReplicaScheduler(BaseReplicaScheduler):
    def __init__(
            self,
            replica_id: int,
            replica_config: ReplicaConfig,
            replica_scheduler_config: BaseReplicaSchedulerConfig,
            metrics_config: MetricsConfig,
            execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        super().__init__(
            replica_id,
            replica_config,
            replica_scheduler_config,
            metrics_config,
            execution_time_predictor,
        )
        # 分开处理图像和文本请求
        self._image_requests = []
        self._text_requests = []

        # CLIP特有的批处理参数
        self._max_batch_size = replica_scheduler_config.batch_size_cap

    def add_request(self, request: Request) -> None:
        """将请求添加到不同的队列中"""
        if request.num_image_tokens > 0 and request.num_text_tokens > 0:
            # 同时包含图像和文本的请求需要被拆分成两个子请求
            self._image_requests.append(request)
            self._text_requests.append(request)
        elif request.num_image_tokens > 0:
            self._image_requests.append(request)
        elif request.num_text_tokens > 0:
            self._text_requests.append(request)

    def on_schedule(self) -> List[Batch]:
        """调度决策逻辑，创建图像和文本批次"""
        batches = []

        # 处理图像批次
        if self._image_requests:
            image_batch_requests = self._image_requests[:self._max_batch_size]
            self._image_requests = self._image_requests[self._max_batch_size:]

            image_tokens = [req.num_image_tokens for req in image_batch_requests]
            image_batch = Batch(
                replica_id=self._replica_id,
                requests=image_batch_requests,
                num_tokens=image_tokens,
                is_image_batch=True
            )
            batches.append(image_batch)

        # 处理文本批次
        if self._text_requests:
            text_batch_requests = self._text_requests[:self._max_batch_size]
            self._text_requests = self._text_requests[self._max_batch_size:]

            text_tokens = [req.num_text_tokens for req in text_batch_requests]
            text_batch = Batch(
                replica_id=self._replica_id,
                requests=text_batch_requests,
                num_tokens=text_tokens,
                is_image_batch=False
            )
            batches.append(text_batch)

        return batches

    def on_batch_end(self, batch: Batch) -> None:
        """批次处理完成后的后续处理"""
        # 请求可能同时在图像和文本批次中，需要确保只有当两者都处理完才标记为完成
        for request in batch.requests:
            if request in self._image_requests and request in self._text_requests:
                if batch.is_image_batch:
                    self._image_requests.remove(request)
                else:
                    self._text_requests.remove(request)

    @property
    def is_empty(self) -> bool:
        """判断调度器是否为空"""
        return len(self._image_requests) == 0 and len(self._text_requests) == 0

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.CLIP
