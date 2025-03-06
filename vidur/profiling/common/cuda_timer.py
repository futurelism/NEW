"""
CUDA 计时器
"""
import time
from contextlib import contextmanager

import torch

from vidur.profiling.common.timer_stats_store import TimerStatsStore

@contextmanager
def CudaTimer(name: str, timer_stats_store: TimerStatsStore):
    """
    使用 CUDA 事件计时的上下文管理器

    Args:
        name: 计时器名称
        timer_stats_store: 计时器统计存储对象
    """
    # 检查是否有 CUDA 设备可用
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        yield

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        timer_stats_store.update(name, elapsed_time_ms)
    else:
        # 如果没有 CUDA 设备，使用 time.time() 计时
        start = time.time()

        yield

        end = time.time()
        elapsed_time_ms = (end - start) * 1000
        timer_stats_store.update(name, elapsed_time_ms)
