import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from vidur.config import BaseRequestGeneratorConfig, SyntheticRequestGeneratorConfig
from vidur.entities import Request
from vidur.logger import init_logger
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.request_generator.interval_generator import BaseIntervalGenerator
from vidur.request_generator.length_generator import BaseLengthGenerator
from vidur.types import RequestGeneratorType

logger = init_logger(__name__)


class SyntheticRequestGenerator(BaseRequestGenerator):
    """
    合成请求生成器，生成模拟的CLIP模型请求。
    为CLIP模型生成包含图像和文本的请求，用于测试和性能分析。
    """

    def __init__(
            self,
            config: SyntheticRequestGeneratorConfig,
            interval_generator: BaseIntervalGenerator,
            length_generator: BaseLengthGenerator,
    ):
        """
        初始化合成请求生成器。

        Args:
            config: 生成器配置
            interval_generator: 请求间隔生成器
            length_generator: 请求长度生成器
        """
        super().__init__(config)
        self._config = config
        self._interval_generator = interval_generator
        self._length_generator = length_generator

        # 如果配置了随机种子，则设置随机种子
        if self._config.seed is not None:
            random.seed(self._config.seed)
            np.random.seed(self._config.seed)

    def generate(self) -> List[Request]:
        """
        生成模拟的CLIP请求列表。
        为CLIP模型生成多模态请求，包含图像和文本部分。

        Returns:
            包含生成的请求的列表
        """
        # 确定请求数量
        num_requests = (
            self._config.num_requests
            if self._config.num_requests is not None
            else self._compute_requests_for_duration()
        )

        requests = []

        # 生成请求到达时间
        arrival_times = self._interval_generator.generate(num_requests)

        # 生成图像尺寸（token数量）
        image_sizes = []
        for _ in range(num_requests):
            # CLIP默认使用224×224的图像，每个patch为32×32, 16×16或14×14
            # 以ViT-B/32为例，会产生(224/32)^2+1=50个图像token（+1为cls token）
            if hasattr(self._config, "image_patch_size") and self._config.image_patch_size:
                patch_size = self._config.image_patch_size
            else:
                # 默认使用32×32的patch
                patch_size = 32

            if hasattr(self._config, "image_resolution") and self._config.image_resolution:
                image_size = self._config.image_resolution
            else:
                # 默认使用224×224的图像
                image_size = 224

            image_tokens = (image_size // patch_size) ** 2 + 1  # +1为cls token
            image_sizes.append(image_tokens)

        # 获取文本长度
        text_lengths = self._length_generator.generate(num_requests)

        # 创建请求对象
        for i in range(num_requests):
            req = Request(
                arrived_at=arrival_times[i],
                num_image_tokens=image_sizes[i],
                num_text_tokens=text_lengths[i],
                image_path=f"simulated_image_{i}.jpg",  # 模拟图像路径
                text=f"Simulated text for request {i}",  # 模拟文本内容
            )
            requests.append(req)

        return requests

    def _compute_requests_for_duration(self) -> int:
        """
        根据配置的持续时间和QPS计算请求数量。

        Returns:
            应生成的请求数量
        """
        # 获取均值到达间隔（秒）
        mean_arrival_interval = 1 / self._config.qps

        # 如果配置了持续时间，则根据持续时间和QPS计算请求数量
        if hasattr(self._config, "duration_sec") and self._config.duration_sec:
            return int(math.ceil(self._config.duration_sec / mean_arrival_interval))

        # 如果没有配置持续时间，则返回默认请求数（500）
        return 500

    def calc_avg_qps(self, batch_size: int, target_throughput: float) -> Tuple[float, int]:
        """
        计算平均QPS和每秒请求数。

        Args:
            batch_size: 批次大小
            target_throughput: 目标吞吐量（token/秒）

        Returns:
            (平均QPS, 每秒请求数)的元组
        """
        # 获取平均请求长度
        avg_length = self._length_generator.get_avg_length()

        # 计算并返回平均QPS和每秒请求数
        avg_qps = target_throughput / avg_length
        rps = avg_qps * batch_size
        return avg_qps, rps

    def _generate_arrival_times(self, num_requests: int) -> List[float]:
        """
        生成请求到达时间列表。

        Args:
            num_requests: 请求数量

        Returns:
            请求到达时间列表
        """
        # 使用间隔生成器生成到达时间
        return self._interval_generator.generate(num_requests)

    @staticmethod
    def get_type() -> RequestGeneratorType:
        """
        返回请求生成器类型。

        Returns:
            RequestGeneratorType.SYNTHETIC
        """
        return RequestGeneratorType.SYNTHETIC
