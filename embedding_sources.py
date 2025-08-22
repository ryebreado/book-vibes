from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd

class EmbeddingSource(ABC):
    # no __init__ needed
    class EmbeddingSource(ABC):
    @abstractmethod
    def get_embedding(self, book_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific book. Returns None if not found."""
        pass
    
    @abstractmethod
    def search_books(self, query: str, max_results: int = 10) -> List[Book]:
        """Fuzzy search for books by title/author."""
        pass
    
    @abstractmethod
    def get_all_books(self) -> List[Book]:
        """Get all available books (for finding recommendations)."""
        pass
    
    @abstractmethod
    def get_embeddings_batch(self, book_ids: List[str]) -> np.ndarray:
        """Get embeddings for multiple books efficiently."""
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of embeddings."""
        pass
