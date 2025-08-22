from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from core import Book

class EmbeddingSource(ABC):
    # no __init__ needed
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

class GoodbooksEmbeddings(EmbeddingSource):
    def __init__(self, data_path: str, n_components: int = 50):
        self.data_path = data_path
        self.n_components = n_components
        self._load_and_process()
    
    def _load_and_process(self):
        print("Loading data...")  # helpful to see progress
        ratings = pd.read_csv(f"{self.data_path}/ratings.csv")
        books = pd.read_csv(f"{self.data_path}/books.csv")
        
        unique_books = ratings['book_id'].unique()
        print(f"books with ratings: {len(unique_books)}")

        print("Creating ratings matrix...")
        ratings_matrix = ratings.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
        
        print("Running SVD...")
        svd = TruncatedSVD(n_components=self.n_components)
        self.book_embeddings = svd.fit_transform(ratings_matrix.T)
        
        # check if we have all books
        self.book_ids = ratings_matrix.columns.tolist()
        print(f"Generated embeddings for {len(self.book_ids)} books")
        
        self.books = books.set_index('book_id')
    
    def get_embedding(self, book_id: str) -> Optional[np.ndarray]:
        """Get embedding for ONE specific book"""
        # book_id comes in as string (interface consistency)
        # but our book_ids are integers 1-10000
        try:
            book_id_int = int(book_id)
            # our embeddings are stored as [book1_embedding, book2_embedding, ...]
            # but book_ids might not be 0-indexed! book_id=1 should map to index 0
            if book_id_int in self.book_ids:
                idx = self.book_ids.index(book_id_int)
                return self.book_embeddings[idx]  # returns 1D array of length n_components
            return None
        except ValueError:
            return None
    
    def get_embeddings_batch(self, book_ids: List[str]) -> np.ndarray:
        """Get embeddings for MULTIPLE books efficiently"""
        # instead of calling get_embedding() in a loop (slow),
        # do vectorized lookup
        embeddings = []
        for book_id in book_ids:
            emb = self.get_embedding(book_id)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return np.array([])  # empty array if no valid books
        
        return np.vstack(embeddings)  # stack into 2D array: n_books × n_components
    
    def search_books(self, query: str, max_results: int = 10) -> List[Book]:
        """Fuzzy search books by title/author"""
        # this searches the METADATA, not the embeddings
        # use the books.csv data for fuzzy matching
        query_lower = query.lower()
        matches = []
        
        for book_id, book_info in self.books.iterrows():
            title = str(book_info['title']).lower()
            author = str(book_info['authors']).lower()  # might be 'authors' not 'author'
            
            # simple fuzzy matching - could use fuzzywuzzy for better results
            if query_lower in title or query_lower in author:
                matches.append(Book(
                    id=str(book_id),
                    title=book_info['title'],
                    author=book_info['authors']
                ))
        
        return matches[:max_results]
    
    def get_all_books(self) -> List[Book]:
        """Return all available books"""
        # needed for finding recommendations from the full catalog
        return [
            Book(id=str(book_id), title=row['title'], author=row['authors'])
            for book_id, row in self.books.iterrows()
            if book_id in self.book_ids  # only books that have embeddings
        ]
    
    @property
    def embedding_dim(self) -> int:
        """Dimensionality of embeddings"""
        return self.n_components

# class OpenLibraryEmbeddings(EmbeddingSource):
#     def __init__(self, api_key: str, cache_dir: str = None):
#         self.api_key = api_key
#         self.cache = Cache(cache_dir) if cache_dir else None