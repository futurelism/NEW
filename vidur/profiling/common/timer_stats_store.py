"""
计时器统计存储
"""
from typing import Dict, List, Optional

import numpy as np

class TimerStats:
    """计时器统计类"""
    def __init__(self):
        self.counts = 0
        self.sum = 0
        self.min = float('inf')
        self.max = float('-inf')
        self._times: List[float] = []

    def update(self, time_ms: float) -> None:
        """更新统计"""
        self.counts += 1
        self.sum += time_ms
        self.min = min(self.min, time_ms)
        self.max = max(self.max, time_ms)
        self._times.append(time_ms)

    def clear(self) -> None:
        """清空统计"""
        self.counts = 0
        self.sum = 0
        self.min = float('inf')
        self.max = float('-inf')
        self._times = []

    @property
    def mean(self) -> float:
        """平均时间"""
        if self.counts == 0:
            return 0
        return self.sum / self.counts

    @property
    def median(self) -> float:
        """中位数时间"""
        if self.counts == 0:
            return 0
        return np.median(self._times)

    @property
    def p90(self) -> float:
        """90百分位时间"""
        if self.counts == 0:
            return 0
        return np.percentile(self._times, 90)

    @property
    def p95(self) -> float:
        """95百分位时间"""
        if self.counts == 0:
            return 0
        return np.percentile(self._times, 95)

    @property
    def p99(self) -> float:
        """99百分位时间"""
        if self.counts == 0:
            return 0
        return np.percentile(self._times, 99)

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "counts": self.counts,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }


class TimerStatsStore:
    """计时器统计存储类"""
    def __init__(self):
        self._stats: Dict[str, TimerStats] = {}

    def update(self, name: str, time_ms: float) -> None:
        """更新统计"""
        if name not in self._stats:
            self._stats[name] = TimerStats()
        self._stats[name].update(time_ms)

    def get_timer_stats(self, name: str) -> TimerStats:
        """获取指定名称的计时器统计"""
        if name not in self._stats:
            self._stats[name] = TimerStats()
        return self._stats[name]

    def clear(self) -> None:
        """清空所有统计"""
        for timer_stats in self._stats.values():
            timer_stats.clear()

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """转换为字典"""
        return {name: stats.to_dict() for name, stats in self._stats.items()}

    def merge(self, other: 'TimerStatsStore') -> None:
        """合并另一个计时器统计存储"""
        for name, stats in other._stats.items():
            if name not in self._stats:
                self._stats[name] = TimerStats()
            for time_ms in stats._times:
                self._stats[name].update(time_ms)
