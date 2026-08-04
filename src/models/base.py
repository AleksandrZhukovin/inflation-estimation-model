"""
Base model template.

Abstract class to specify required behavior template
for models implemented.
"""

import joblib

from abc import ABC, abstractmethod
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract class for model implementation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return model name string.
        """
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        """
        Return human-readable model label.
        """
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str):
        return joblib.load(Path(path))
