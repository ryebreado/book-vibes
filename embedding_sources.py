from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from core import Book
import pickle
from pathlib import Path

class EmbeddingSource(ABC):
    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

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

    def _get_cache_path(self, cache_name: str) -> Path:
        """Generate cache file path"""
        return self.cache_dir / f"{cache_name}.pkl"
    
    def _load_from_cache(self, cache_name: str) -> Optional[dict]:
        """Load data from cache if it exists"""
        cache_file = self._get_cache_path(cache_name)
        if cache_file.exists():
            print(f"Loading cached {cache_name}...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_name: str, data: dict) -> None:
        """Save data to cache"""
        cache_file = self._get_cache_path(cache_name)
        print(f"Caching {cache_name}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

class GoodbooksEmbeddings(EmbeddingSource):
    def __init__(self, data_path: str, n_components: int = 50, cache_dir: str = "data/processed"):
        super().__init__(cache_dir)
        self.data_path = data_path
        self.n_components = n_components
        self._load_and_process()
    
    def _load_and_process(self):
        cache_name = f"goodbooks_embeddings_{self.n_components}"

        cached_data = self._load_from_cache(cache_name)

        # check if data cached already, o/w generate with SVD
        if cached_data:
            self.book_embeddings = cached_data['embeddings']
            self.book_ids = cached_data['book_ids']
            print(f"Loaded embeddings for {len(self.book_ids)} books")
        else:
            print("Loading data...")  # helpful to see progress
            ratings = pd.read_csv(f"{self.data_path}/ratings.csv")
            
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

            self._save_to_cache(cache_name, {
                'embeddings': self.book_embeddings,
                'book_ids': self.book_ids
            })
        
        books = pd.read_csv(f"{self.data_path}/books.csv")
        self.books = books.set_index('book_id')
    
    def get_embedding(self, book_id: str) -> Optional[np.ndarray]:
        book_id_int = int(book_id)
        if 1 <= book_id_int <= 10000: 
            idx = book_id_int - 1  # convert to 0-indexed
            return self.book_embeddings[idx]
        return None
    
    def get_embeddings_batch(self, book_ids: List[str]) -> np.ndarray:
        """Get embeddings for MULTIPLE books efficiently"""
        # vectorized lookup
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
        # for inputing books
        # use the books.csv data for fuzzy matching
        query_lower = query.lower()
        matches = []
        
        for book_id, book_info in self.books.iterrows():
            title = str(book_info['title']).lower()
            author = str(book_info['authors']).lower()  
            
            # simple fuzzy matching
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

class OpenLibraryEmbeddings(EmbeddingSource):
    def __init__(self, api_key: str, cache_dir: str = None, n_components=50):
        super().__init__(cache_dir)
        self.api_key = api_key
        self.cache = Cache(cache_dir) if cache_dir else None
    
    def _load_and_process(self):
        pass

    def get_embedding(self, book_id: str) -> Optional[np.ndarray]:
        pass

    def search_books(self, query: str, max_results: int = 10) -> List[Book]:
        pass 

    def get_all_books(self) -> List[Book]:
        """Get all available books (for finding recommendations)."""
        pass
    
    def get_embeddings_batch(self, book_ids: List[str]) -> np.ndarray:
        """Get embeddings for multiple books efficiently."""
        embeddings = []
        for book_id in book_ids:
            emb = self.get_embedding(book_id)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return np.array([])  # empty array if no valid books
        
        return np.vstack(embeddings)  # stack into 2D array: n_books × n_components