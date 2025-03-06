from typing import List

from vidur.entities.base_entity import BaseEntity
from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)


# 装饰器：检查批次是否已经被调度
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


# 装饰器：检查批次是否已经完成
def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


class Batch(BaseEntity):
    """
    表示一个处理批次的类。
    在CLIP中，批次可以是图像批次或文本批次，分别处理图像和文本编码。
    """
    def __init__(
            self,
            replica_id: int,
            requests: List[Request],
            num_tokens: List[int],
            is_image_batch: bool = False,  # 表示这是否为图像批次
    ) -> None:
        """
        初始化批次对象。

        Args:
            replica_id: 处理该批次的复制品ID
            requests: 包含在批次中的请求列表
            num_tokens: 每个请求的token数量列表
            is_image_batch: 是否为图像批次（否则为文本批次）
        """
        self._id = Batch.generate_id()
        self._replica_id = replica_id

        self._requests = requests
        self._num_tokens = num_tokens
        self._is_image_batch = is_image_batch  # 新增：标识是否为图像批次
        self._total_num_tokens = sum(num_tokens)

        # 根据批次类型计算预填充token数量
        if is_image_batch:
            self._num_prefill_tokens = sum([r.num_image_tokens for r in self.requests])
        else:
            self._num_prefill_tokens = sum([
                (t if not r.is_prefill_complete else 0)
                for r, t in zip(self.requests, self._num_tokens)
            ])

        # 将总token数向上调整到8的倍数（为了硬件优化）
        self._total_num_tokens_rounded = (self._total_num_tokens + 7) // 8 * 8

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False
        self._completed = False

    @property
    def replica_id(self) -> int:
        """返回处理该批次的复制品ID。"""
        return self._replica_id

    @property
    def creation_time(self) -> float:
        """返回批次创建的时间。"""
        return getattr(self, "_creation_time", 0)

    @property
    def num_tokens(self) -> List[int]:
        """返回每个请求的token数量列表。"""
        return self._num_tokens

    @property
    def total_num_tokens(self) -> int:
        """返回批次中总token数量。"""
        return self._total_num_tokens

    @property
    def num_prefill_tokens(self) -> int:
        """
        返回预填充token数量。
        对于图像批次，这是图像token数量；
        对于文本批次，这是尚未完成预填充的文本token数量。
        """
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        """
        返回解码token数量。
        在CLIP中，这主要用于文本批次中的文本token。
        """
        return self.total_num_tokens - self.num_prefill_tokens

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        """返回批次被调度的时间。"""
        return self._scheduled_at

    @property
    @check_completed
    def completed_at(self) -> float:
        """返回批次完成的时间。"""
        return self._completed_at

    @property
    def completed(self) -> bool:
        """返回批次是否已完成。"""
        return self._completed

    @property
    def scheduled(self) -> bool:
        """返回批次是否已被调度。"""
        return self._scheduled

    @property
    def size(self) -> int:
        """返回批次中的请求数量。"""
        return len(self._requests)

    @property
    def requests(self) -> List[Request]:
        """返回批次中的请求列表。"""
        return self._requests

    @property
    def request_ids(self) -> List[int]:
        """返回批次中请求的ID列表。"""
        return [request.id for request in self._requests]

    @property
    def is_image_batch(self) -> bool:
        """
        返回这是否为图像批次。
        在CLIP中，批次要么是图像批次，要么是文本批次。
        """
        return self._is_image_batch

    @property
    def all_requests_completed(self) -> bool:
        """返回批次中的所有请求是否都已完成。"""
        return all([request.completed for request in self._requests])

    def on_schedule(
            self,
            time: float,
    ) -> None:
        """
        当批次被调度时调用。

        Args:
            time: 批次被调度的时间
        """
        self._scheduled_at = time
        self._scheduled = True

        # 通知每个请求它已被调度到批次
        for request in self._requests:
            request.on_batch_schedule(time)

    def on_batch_end(self, time: float):
        """
        当批次处理结束时调用。

        Args:
            time: 批次结束的时间
        """
        self._completed = True
        self._completed_at = time

        # 更新每个请求的处理进度
        for request, num_tokens in zip(self._requests, self._num_tokens):
            request.on_batch_end(time, num_tokens)

    @property
    def preempted_requests(self) -> List[Request]:
        """返回批次中被抢占的请求列表。"""
        return [request for request in self._requests if request.preempted]

    @property
    def completed_requests(self) -> List[Request]:
        """返回批次中已完成的请求列表。"""
        return [request for request in self._requests if request.completed]

    def to_dict(self) -> dict:
        """
        将批次对象转换为字典表示。

        Returns:
            包含批次信息的字典
        """
        return {
            "id": self._id,
            "size": self.size,
            "replica_id": self._replica_id,
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
            "num_prefill_tokens": self.num_prefill_tokens,
            "num_decode_tokens": self.num_decode_tokens,
            "is_image_batch": self._is_image_batch,
        }
