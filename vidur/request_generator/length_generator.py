"""
Length generator implementations for request generation.
"""
from abc import ABC, abstractmethod
from typing import List

class BaseLengthGenerator(ABC):
    """
    Base class for request length generators.
    """

    def __init__(self, config):
        """
        Initialize the length generator.

        Args:
            config: Configuration for the length generator
        """
        self._config = config

    @abstractmethod
    def generate(self, num_lengths: int) -> List[int]:
        """
        Generate request lengths.

        Args:
            num_lengths: Number of lengths to generate

        Returns:
            List of request lengths
        """
        pass

    @abstractmethod
    def get_avg_length(self) -> float:
        """
        Get the average request length.

        Returns:
            Average length
        """
        pass
