"""
Base request generator implementation.
"""
from abc import ABC, abstractmethod
from typing import List

from vidur.types import RequestGeneratorType

class BaseRequestGenerator(ABC):
    """
    Base class for request generators.
    """

    def __init__(self, config):
        """
        Initialize the request generator.

        Args:
            config: Configuration for the request generator
        """
        self._config = config

    @abstractmethod
    def generate(self) -> List:
        """
        Generate a list of requests.

        Returns:
            List of generated requests
        """
        pass

    @staticmethod
    @abstractmethod
    def get_type() -> RequestGeneratorType:
        """
        Get the type of the request generator.

        Returns:
            RequestGeneratorType enum value
        """
        pass
