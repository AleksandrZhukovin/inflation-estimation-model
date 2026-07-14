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
    Abstract class for all models.
    """

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
