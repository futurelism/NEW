"""
Interval generator implementations for request generation.
"""
from abc import ABC, abstractmethod
from typing import List

class BaseIntervalGenerator(ABC):
    """
    Base class for request interval generators.
    """

    def __init__(self, config):
        """
        Initialize the interval generator.

        Args:
            config: Configuration for the interval generator
        """
        self._config = config

    @abstractmethod
    def generate(self, num_intervals: int) -> List[float]:
        """
        Generate request intervals.

        Args:
            num_intervals: Number of intervals to generate

        Returns:
            List of arrival times
        """
        pass
