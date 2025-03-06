from typing import Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)


# 装饰器：检查请求是否已经被调度
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Request has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


# 装饰器：检查请求是否已经完成
def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Request has not been completed yet")
        return func(self, *args, **kwargs)

    return wrapper


class Request(BaseEntity):
    """
    表示一个处理请求的类。
    对于CLIP模型，请求包含图像和文本数据，用于计算它们之间的相似度。
    """
    def __init__(
            self,
            arrived_at: float,
            num_image_tokens: int,  # 图像token数量
            num_text_tokens: int,   # 文本token数量
            image_path: str = None, # 图像路径（可选）
            text: str = None,       # 文本内容（可选）
            num_processed_tokens: int = 0,
    ):
        """
        初始化请求对象。

        Args:
            arrived_at: 请求到达时间
            num_image_tokens: 图像token数量
            num_text_tokens: 文本token数量
            image_path: 图像文件路径（可选）
            text: 文本内容（可选）
            num_processed_tokens: 已处理的token数量
        """
        self._id = Request.generate_id()
        self._arrived_at = arrived_at
        self._num_image_tokens = num_image_tokens
        self._num_text_tokens = num_text_tokens
        self._image_path = image_path
        self._text = text
        self._num_processed_tokens = num_processed_tokens

        # 请求状态及时间信息
        self._scheduled_at = 0
        self._execution_time = 0
        self._model_execution_time = 0
        self._scheduling_delay = 0
        self._preempted_time = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0

        # 状态标志
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts = 0

    @property
    def size(self) -> Tuple[int, int]:
        """
        返回请求的大小，表示为(图像token数, 文本token数)元组。
        """
        return (self._num_image_tokens, self._num_text_tokens)

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        """返回请求被调度的时间。"""
        return self._scheduled_at

    @property
    @check_scheduled
    def latest_stage_scheduled_at(self) -> float:
        """返回最近一个阶段被调度的时间。"""
        return self._latest_stage_scheduled_at

    @property
    @check_scheduled
    def latest_stage_completed_at(self) -> float:
        """返回最近一个阶段完成的时间。"""
        return self._latest_stage_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduled_at(self) -> float:
        """返回最近一次迭代被调度的时间。"""
        return self._latest_iteration_scheduled_at

    @property
    @check_scheduled
    def latest_iteration_completed_at(self) -> float:
        """返回最近一次迭代完成的时间。"""
        return self._latest_iteration_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduling_delay(self) -> float:
        """返回最近一次迭代的调度延迟。"""
        return self._latest_iteration_scheduling_delay

    @property
    @check_scheduled
    def prefill_completed_at(self) -> float:
        """返回预填充阶段完成的时间。在CLIP中，这通常表示特征提取完成时间。"""
        return self._prefill_completed_at

    @property
    @check_scheduled
    def scheduling_delay(self) -> float:
        """返回请求的调度延迟（从到达到首次调度的时间）。"""
        return self._scheduling_delay

    @property
    @check_scheduled
    def preempted_time(self) -> float:
        """返回请求被抢占的总时间。"""
        return self._preempted_time

    @property
    @check_completed
    def completed_at(self) -> float:
        """返回请求完成的时间。"""
        return self._completed_at

    @property
    @check_scheduled
    def e2e_time(self) -> float:
        """返回请求的端到端处理时间（从到达到完成）。"""
        return self._completed_at - self._arrived_at

    @property
    @check_scheduled
    def e2e_time_normalized(self) -> float:
        """返回归一化的端到端处理时间（按解码token数量）。"""
        return self.e2e_time / self.num_decode_tokens if self.num_decode_tokens > 0 else 0

    @property
    @check_scheduled
    def execution_time(self) -> float:
        """返回请求的执行时间（不包括调度延迟和抢占）。"""
        return self._execution_time

    @property
    @check_scheduled
    def execution_time_normalized(self) -> float:
        """返回归一化的执行时间。"""
        return self._execution_time / self.num_decode_tokens if self.num_decode_tokens > 0 else 0

    @property
    @check_scheduled
    def model_execution_time(self) -> float:
        """返回模型执行时间（不包括系统开销）。"""
        return self._model_execution_time

    @property
    @check_scheduled
    def model_execution_time_normalized(self) -> float:
        """返回归一化的模型执行时间。"""
        return self._model_execution_time / self.num_decode_tokens if self.num_decode_tokens > 0 else 0

    @property
    def arrived_at(self) -> float:
        """返回请求到达的时间。"""
        return self._arrived_at

    @property
    def num_image_tokens(self) -> int:
        """返回图像的token数量。"""
        return self._num_image_tokens

    @property
    def num_text_tokens(self) -> int:
        """返回文本的token数量。"""
        return self._num_text_tokens

    @property
    def num_prefill_tokens(self) -> int:
        """返回预填充token的数量。在CLIP中，这通常是图像token。"""
        return self._num_image_tokens

    @property
    def num_decode_tokens(self) -> int:
        """返回解码token的数量。在CLIP中，这通常是文本token。"""
        return self._num_text_tokens

    @property
    def pd_ratio(self) -> float:
        """返回预填充与解码token的比率。"""
        return self._num_image_tokens / self._num_text_tokens if self._num_text_tokens > 0 else 0

    @property
    def image_path(self) -> str:
        """返回图像路径。"""
        return self._image_path

    @property
    def text(self) -> str:
        """返回文本内容。"""
        return self._text

    @property
    def num_processed_tokens(self) -> int:
        """返回已处理的token数量。"""
        return self._num_processed_tokens

    @property
    def total_tokens(self) -> int:
        """返回总token数量（图像和文本）。"""
        return self._num_image_tokens + self._num_text_tokens

    @property
    def num_processed_prefill_tokens(self) -> int:
        """返回已处理的预填充token数量。"""
        return min(self._num_processed_tokens, self._num_image_tokens)

    @property
    def num_processed_decode_tokens(self) -> int:
        """返回已处理的解码token数量。"""
        return max(self._num_processed_tokens - self._num_image_tokens, 0)

    @property
    def scheduled(self) -> bool:
        """返回请求是否已被调度。"""
        return self._scheduled

    @property
    def preempted(self) -> bool:
        """返回请求是否被抢占且未完成。"""
        return self._preempted and not self._completed

    @property
    def completed(self) -> bool:
        """返回请求是否已完成。"""
        return self._completed

    @property
    def num_restarts(self) -> int:
        """返回请求重启的次数。"""
        return self._num_restarts

    @property
    def is_prefill_complete(self) -> bool:
        """返回预填充阶段是否已完成。在CLIP中，表示特征提取是否完成。"""
        return self._is_prefill_complete

    @property
    def has_started_decode(self) -> bool:
        """返回是否已开始解码阶段。在CLIP中，表示是否已开始处理文本。"""
        return self._num_processed_tokens > self._num_image_tokens + 1

    def on_batch_schedule(
            self,
            time: float,
    ) -> None:
        """
        当请求被调度到一个批次时调用。

        Args:
            time: 批次调度的时间
        """
        self._latest_iteration_scheduled_at = time
        self._latest_iteration_scheduling_delay = (
                time - self._latest_iteration_completed_at
        )

        if self._scheduled:
            return

        if self._num_restarts > 0:
            self._scheduled = True
            return

        self._scheduled_at = time
        self._scheduling_delay = time - self._arrived_at
        self._scheduled = True

    def on_batch_end(
            self,
            time: float,
            num_tokens_processed: int,
    ) -> None:
        """
        当批次处理结束时调用。

        Args:
            time: 批次结束的时间
            num_tokens_processed: 本次批次处理的token数量
        """
        self._num_processed_tokens += num_tokens_processed
        self._latest_iteration_completed_at = time

        assert self._num_processed_tokens <= self.total_tokens

        # 对于CLIP，当图像处理完成时，标记预填充完成
        if self._num_processed_tokens == self._num_image_tokens:
            self._is_prefill_complete = True
            # 在预填充完成时增加一个token，表示开始处理文本
            self._num_processed_tokens += 1

            # 仅在第一次记录预填充完成时间
            if self._prefill_completed_at == 0:
                self._prefill_completed_at = time

        # 检查请求是否完成
        if self._num_processed_tokens == self.total_tokens:
            self._completed_at = time
            self._completed = True
            logger.debug(f"Request {self._id} completed at {self._completed_at}")

    def on_batch_stage_schedule(
            self,
            time: float,
    ) -> None:
        """
        当请求的一个处理阶段被调度时调用。

        Args:
            time: 阶段调度的时间
        """
        self._latest_stage_scheduled_at = time
        if self._latest_stage_completed_at == 0:
            self._preempted_time = 0
        else:
            self._preempted_time += time - self._latest_stage_completed_at
        self._preempted = False

    def on_batch_stage_end(
            self,
            time: float,
            execution_time: float,
            model_execution_time: float,
    ) -> None:
        """
        当请求的一个处理阶段结束时调用。

        Args:
            time: 阶段结束的时间
            execution_time: 阶段的执行时间
            model_execution_time: 模型在此阶段的执行时间
        """
        self._execution_time += execution_time
        self._model_execution_time += model_execution_time
        self._latest_stage_completed_at = time
        self._preempted = True

    def to_dict(self) -> dict:
        """
        将请求对象转换为字典表示。

        Returns:
            包含请求信息的字典
        """
        return {
            "id": self._id,
            "arrived_at": self._arrived_at,
            "execution_time": self._execution_time,
            "model_execution_time": self._model_execution_time,
            "scheduled_at": self._scheduled_at,
            "scheduling_delay": self._scheduling_delay,
            "preempted_time": self._preempted_time,
            "completed_at": self._completed_at,
            "num_image_tokens": self._num_image_tokens,
            "num_text_tokens": self._num_text_tokens,
            "image_path": self._image_path,
            "text": self._text,
            "num_processed_tokens": self._num_processed_tokens,
            "scheduled": self._scheduled,
            "preempted": self._preempted,
            "completed": self._completed,
            "latest_stage_scheduled_at": self._latest_stage_scheduled_at,
            "latest_stage_completed_at": self._latest_stage_completed_at,
            "latest_iteration_scheduled_at": self._latest_iteration_scheduled_at,
            "latest_iteration_completed_at": self._latest_iteration_completed_at,
            "num_restarts": self._num_restarts,
        }

    def restart(self):
        """
        重启请求处理。在CLIP中，通常用于处理内存不足等情况。
        """
        logger.debug(f"Restarting request {self._id}")

        # 当重启请求时，可以并行处理所有先前处理过的token
        total_tokens = self._num_image_tokens + self._num_text_tokens
        self._num_image_tokens = self._num_processed_tokens
        self._num_text_tokens = total_tokens - self._num_image_tokens

        self._num_processed_tokens = 0
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts += 1
