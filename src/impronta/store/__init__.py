from .base import UNSET, VectorStore
from .memory import InMemoryStore

__all__ = ["UNSET", "VectorStore", "FaissLocalStore", "InMemoryStore"]


def __getattr__(name: str):
    # FaissLocalStore is lazy so that `import impronta` does not load faiss —
    # processes that only embed (or only use InMemoryStore) can then run
    # torch multithreaded without the macOS faiss/torch libomp clash.
    if name == "FaissLocalStore":
        from .faiss_local import FaissLocalStore

        return FaissLocalStore
    raise AttributeError(name)
