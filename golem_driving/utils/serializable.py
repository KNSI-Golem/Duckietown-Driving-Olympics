import pickle
from typing import Any


class Serializable:
    @staticmethod
    def load(filename: str) -> Any:
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(f)