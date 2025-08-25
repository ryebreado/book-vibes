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
        """Learn vibe subspace from seed books"""
        self.seed_embeddings = book_embeddings
        
        # fit pca on seed embeddings to find vibe subspace
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(book_embeddings)
        
        # compute centroid in the vibe subspace
        transformed_seeds = self.pca.transform(book_embeddings)
        self.vibe_centroid = np.mean(transformed_seeds, axis=0)
        
    def find_similar(self, all_embeddings: np.ndarray, all_books: List[Book], n_results: int = 10) -> List[BookResult]:
        """Find books similar to the vibe"""
        if self.pca is None:
            raise ValueError("must call fit() first")
            
        # project all books into vibe subspace
        projected_books = self.pca.transform(all_embeddings)
        
        # compute similarity to vibe centroid in subspace
        similarities = cosine_similarity([self.vibe_centroid], projected_books)[0]
        
        # get top n most similar
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            results.append(BookResult(
                book=all_books[idx],
                similarity=similarities[idx]
            ))
            
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