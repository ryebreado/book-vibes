from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from core import BookResult, Book

class VibeIdentifier(ABC):
    @abstractmethod
    def fit(self, book_embeddings: np.ndarray) -> None:
        """Learn the vibe from seed book embeddings"""
        pass
    
    @abstractmethod
    def find_similar(self, all_embeddings: np.ndarray, all_books: List[Book], n_results: int = 10) -> List[BookResult]:
        """Return ranked similar books"""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict:
        """Return vibe quality metrics"""
        pass

class PCAVibeIdentifier(VibeIdentifier):
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.pca = None
        self.seed_embeddings = None
        self.vibe_centroid = None
        
    def fit(self, book_embeddings: np.ndarray) -> None:
        """Learn vibe from seed books"""
        self.seed_embeddings = book_embeddings

        # centroid in the original embedding space — PCA.transform centers
        # on fit, so taking the mean of transformed seeds was always ~0
        self.vibe_centroid = np.mean(book_embeddings, axis=0)

        # PCA is still fit here so get_stats() can report explained variance
        # as a diagnostic on how tightly the seeds cluster
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(book_embeddings)

    def find_similar(self, all_embeddings: np.ndarray, all_books: List[Book], n_results: int = 10, exclude_ids: set = None) -> List[BookResult]:
        """Find books similar to the vibe"""
        if self.vibe_centroid is None:
            raise ValueError("must call fit() first")

        similarities = cosine_similarity([self.vibe_centroid], all_embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:n_results * 2]

        results = []
        for idx in top_indices:
            if exclude_ids and all_books[idx].id in exclude_ids:
                continue

            results.append(BookResult(
                book=all_books[idx],
                similarity=similarities[idx]
            ))

            if len(results) >= n_results:
                break

        return results
    
    def get_stats(self) -> dict:
        """Return vibe quality metrics"""
        if self.pca is None:
            return {}
            
        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'total_explained_variance': self.pca.explained_variance_ratio_.sum(),
            'n_components': self.n_components,
            'n_seed_books': len(self.seed_embeddings) if self.seed_embeddings is not None else 0
        }