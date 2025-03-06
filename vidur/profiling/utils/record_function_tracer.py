"""
PyTorch record_function 跟踪器
"""
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch

from vidur.profiling.common.timer_stats_store import TimerStatsStore

class RecordFunctionTracer:
    """
    PyTorch record_function 跟踪器
    允许跟踪和测量 PyTorch 操作的执行时间
    """

    def __init__(self):
        """初始化跟踪器"""
        self._active = False
        self._timer_stats_store = TimerStatsStore()

    @property
    def timer_stats_store(self) -> TimerStatsStore:
        """获取计时器统计存储"""
        return self._timer_stats_store

    @contextmanager
    def trace(self, enabled: bool = True):
        """
        启用或禁用跟踪的上下文管理器

        Args:
            enabled: 是否启用跟踪
        """
        if not enabled:
            yield
            return

        try:
            self._activate()
            yield
        finally:
            self._deactivate()

    def _activate(self):
        """激活跟踪器"""
        if self._active:
            return

        self._active = True
        # 这里可以添加额外的激活逻辑，如设置 PyTorch 钩子等

    def _deactivate(self):
        """停用跟踪器"""
        if not self._active:
            return

        self._active = False
        # 这里可以添加额外的停用逻辑
